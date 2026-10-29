/*!
 * \file lower_dcu_ipc.cc
 * \brief M2 HIP-IPC lowering before SplitHostDevice.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>
#include <string>
#include <unordered_map>

namespace tvm {
namespace tir {
namespace {

using tvm::ffi::Array;
using tvm::ffi::String;

std::string OpName(const CallNode* call) {
  if (const auto* op = call->op.as<OpNode>()) return op->name;
  return "";
}

class DCUIPCLowerer final : public StmtExprMutator {
 public:
  explicit DCUIPCLowerer(const PrimFunc& func) : func_(func) {
    for (const Var& param : func->params) {
      if (param->name_hint == "__tl_ipc_rank") rank_ = param;
      if (param->name_hint == "__tl_ipc_world_size") world_ = param;
    }
    ICHECK(rank_.defined() && world_.defined()) << "M2 IPC ABI rank/world parameters missing";
    auto attr = func->attrs.GetAttr<String>("tl.dcu_ipc.bindings");
    ICHECK(attr.has_value()) << "M2 IPC binding attribute missing";
    std::stringstream entries(attr.value());
    std::string entry;
    while (std::getline(entries, entry, ';')) {
      const auto equal = entry.find('=');
      ICHECK_NE(equal, std::string::npos) << "malformed M2 IPC binding";
      bindings_.emplace(entry.substr(0, equal), entry.substr(equal + 1));
    }
    for (const auto& [data, buffer] : func->buffer_map) {
      buffers_by_data_.emplace(buffer->data.get(), buffer);
      for (const auto& [base, descriptor_name] : bindings_) {
        if (buffer->data->name_hint == descriptor_name) descriptors_.emplace(base, buffer->data);
      }
    }
    for (const auto& [base, _] : bindings_) {
      ICHECK(descriptors_.count(base)) << "M2 descriptor parameter absent for " << base;
    }
  }

  PrimFunc Run() {
    return PrimFunc(func_->params, VisitStmt(func_->body), func_->ret_type,
                    func_->buffer_map, func_->attrs, func_->span);
  }

 private:
  Var DescriptorForRegion(const PrimExpr& region) const {
    const auto* region_call = region.as<CallNode>();
    ICHECK(region_call && !region_call->args.empty()) << "IPC operation requires tl.region";
    const auto* load = region_call->args[0].as<BufferLoadNode>();
    ICHECK(load) << "IPC region must retain BufferLoad provenance";
    auto direct = descriptors_.find(load->buffer->name);
    if (direct != descriptors_.end()) return direct->second;
    auto remote = remote_descriptors_.find(load->buffer->data.get());
    ICHECK(remote != remote_descriptors_.end())
        << "IPC region has no explicit DeviceView binding for " << load->buffer->name;
    return remote->second;
  }

  PrimExpr Make(const CallNode* op, const char* name, Array<PrimExpr> args) {
    return Call(op->dtype, Op::Get(name), std::move(args), op->span);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    const std::string name = OpName(op);
    if (name == "tl.get_rank") return rank_;
    if (name == "tl.get_num_ranks") return world_;
    if (name == "tl.remote_address") {
      Var descriptor = DescriptorForRegion(op->args[0]);
      return Make(op, "tl.ipc.remote_address", {descriptor, VisitExpr(op->args[1]), rank_, world_});
    }

    Array<PrimExpr> args;
    for (const PrimExpr& arg : op->args) args.push_back(VisitExpr(arg));
    if (name == "tl.fence_sys") return Make(op, "tl.ipc.fence_sys", {rank_, world_});
    if (name == "tl.signal_set") {
      args.push_back(DescriptorForRegion(op->args[0])); args.push_back(rank_); args.push_back(world_);
      return Make(op, "tl.ipc.signal_set", std::move(args));
    }
    if (name == "tl.tileop.wait") {
      args.push_back(DescriptorForRegion(op->args[1])); args.push_back(rank_); args.push_back(world_);
      return Make(op, "tl.ipc.wait", std::move(args));
    }
    if (name == "tl.tileop.put_block") {
      args.push_back(DescriptorForRegion(op->args[1])); args.push_back(rank_); args.push_back(world_);
      return Make(op, "tl.ipc.put_block", std::move(args));
    }
    if (name == "tl.tileop.get_block") {
      args.push_back(DescriptorForRegion(op->args[0])); args.push_back(rank_); args.push_back(world_);
      return Make(op, "tl.ipc.get_block", std::move(args));
    }
    if (name == "tl.tileop.st") {
      args.push_back(DescriptorForRegion(op->args[0])); args.push_back(rank_); args.push_back(world_);
      return Make(op, "tl.ipc.st", std::move(args));
    }
    if (name == "tl.tileop.ld") {
      args.push_back(DescriptorForRegion(op->args[0])); args.push_back(rank_); args.push_back(world_);
      return Make(op, "tl.ipc.ld", std::move(args));
    }
    return Call(op->dtype, op->op, std::move(args), op->span);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    const auto* raw = op->value.as<CallNode>();
    const bool is_remote = raw && OpName(raw) == "tl.remote_address";
    Var descriptor;
    if (is_remote) {
      descriptor = DescriptorForRegion(raw->args[0]);
      remote_descriptors_.emplace(op->var.get(), descriptor);
    }
    PrimExpr value = VisitExpr(op->value);
    Stmt body = VisitStmt(op->body);
    if (is_remote) remote_descriptors_.erase(op->var.get());
    return LetStmt(op->var, value, body, op->span);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    const auto* call = op->value.as<CallNode>();
    // HIP's generic TileLang storage-sync printer deliberately suppresses
    // shared barriers for its existing GEMM schedule.  In an IPC function a
    // shared sync separates many producer threads from thread zero's system
    // release, so preserve it as an explicit, effectful IPC CTA barrier.
    if (call && call->op.same_as(builtin::tvm_storage_sync()) &&
        !call->args.empty()) {
      const auto* scope = call->args[0].as<StringImmNode>();
      if (scope && (scope->value == "shared" || scope->value == "shared.dyn")) {
        return Evaluate(Call(DataType::Int(32), Op::Get("tl.ipc.block_barrier"),
                             {rank_, world_}, op->span));
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimFunc func_;
  Var rank_, world_;
  std::unordered_map<std::string, std::string> bindings_;
  std::unordered_map<std::string, Var> descriptors_;
  std::unordered_map<const VarNode*, Buffer> buffers_by_data_;
  std::unordered_map<const VarNode*, Var> remote_descriptors_;
};

}  // namespace

PrimFunc LowerDCUIPC(PrimFunc func) { return DCUIPCLowerer(func).Run(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerDCUIPC", LowerDCUIPC);
}

}  // namespace tir
}  // namespace tvm
