/*!
 * \file inject_dcu_async_copy_shadow_addr.cc
 * \brief Remap shared.dyn base pointers for tl.dcu_async_copy with warp-dependent
 *        shadow offsets, and replace buf_dyn_shmem with per-buffer shadow addresses.
 */
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <functional>
#include <optional>
#include <string>
#include <unordered_set>

#include "../op/builtin.h"
#include "inject_utils.h"

using tvm::ffi::GetRef;

namespace tvm {
namespace tl {

using namespace tir;

namespace {

const Op& DCUAsyncCopyOp() { return Op::Get("tl.dcu_async_copy"); }
const Op& MakeDCUResourceOp() { return Op::Get("tl.make_dcu_resource"); }

bool IsABufferName(const std::string& name) { return name == "A" || name == "A_1"; }

bool IsBBufferName(const std::string& name) { return name == "B" || name == "B_1"; }

int GetElementBytes(DataType dtype) {
  if (dtype.lanes() > 1) {
    dtype = dtype.element_of();
  }
  return dtype.bytes();
}

std::optional<int> GetBElemBytesFromDType(DataType dtype) {
  const int elem_bytes = GetElementBytes(dtype);
  if (elem_bytes > 0) {
    return elem_bytes;
  }
  return std::nullopt;
}

bool IsWarpIdxInit(const PrimExpr& expr) {
  if (const auto* call = expr.as<CallNode>()) {
    return call->op.same_as(tl::get_warp_idx_sync());
  }
  return false;
}

bool IsThreadIdxXAttr(const AttrStmtNode* op, const IterVarNode** out_iv = nullptr) {
  if (op->attr_key != tir::attr::thread_extent) {
    return false;
  }
  const auto* iv = op->node.as<IterVarNode>();
  if (iv == nullptr || iv->thread_tag != "threadIdx.x") {
    return false;
  }
  if (out_iv != nullptr) {
    *out_iv = iv;
  }
  return true;
}

std::string GetAllocateStorageScope(const AllocateNode* op) {
  if (const auto* ptr_type = op->buffer_var->type_annotation.as<PointerTypeNode>()) {
    if (!ptr_type->storage_scope.empty()) {
      return ptr_type->storage_scope;
    }
  }
  if (op->annotations.count("storage_scope")) {
    return Downcast<String>(op->annotations.at("storage_scope"));
  }
  return "";
}

std::string GetGlobalBufferName(const PrimExpr& expr) {
  if (const auto* var = expr.as<VarNode>()) {
    return var->name_hint;
  }
  if (const auto* load = expr.as<BufferLoadNode>()) {
    return load->buffer->name;
  }
  return "";
}

bool IsSameVar(const PrimExpr& expr, const Var& var) {
  if (!var.defined()) {
    return false;
  }
  if (const auto* v = expr.as<VarNode>()) {
    return v == var.get();
  }
  return false;
}

class InfoCollector : public StmtExprVisitor {
 public:
  Var buf_dyn_shmem;
  PrimExpr thread_extent;
  std::unordered_set<const VarNode*> a_dcu_res_vars;
  std::unordered_set<const VarNode*> b_dcu_res_vars;
  std::optional<int> b_elem_bytes;

 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    const IterVarNode* iv = nullptr;
    if (IsThreadIdxXAttr(op, &iv)) {
      thread_extent = op->value;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        op->thread_binding.value()->thread_tag == "threadIdx.x") {
      thread_extent = op->extent;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode* op) final {
    const std::string storage_scope = GetAllocateStorageScope(op);
    if (storage_scope == "shared.dyn") {
      if (!buf_dyn_shmem.defined() ||
          op->buffer_var->name_hint == "buf_dyn_shmem") {
        buf_dyn_shmem = op->buffer_var;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode* op) final {
    if (op->buffer.scope() == "shared.dyn" && !buf_dyn_shmem.defined()) {
      buf_dyn_shmem = op->buffer->data;
    }
    if (IsBBufferName(op->buffer->name)) {
      b_elem_bytes = GetBElemBytesFromDType(op->buffer->dtype);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(MakeDCUResourceOp()) && call->args.size() >= 1) {
        const std::string global_name = GetGlobalBufferName(call->args[0]);
        if (global_name == "A" || global_name == "A_1") {
          a_dcu_res_vars.insert(op->var.get());
        } else if (IsBBufferName(global_name)) {
          b_dcu_res_vars.insert(op->var.get());
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(DCUAsyncCopyOp()) && op->args.size() >= 6) {
      if (const auto* var = op->args[2].as<VarNode>()) {
        if (b_dcu_res_vars.count(var) > 0) {
          if (const auto* dtype_str = op->args[5].as<StringImmNode>()) {
            DataType elem_dtype = DataType(ffi::StringToDLDataType(dtype_str->value));
            b_elem_bytes = GetBElemBytesFromDType(elem_dtype);
          }
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

PrimExpr ToUIntPtr(const PrimExpr& value) {
  const DataType u64 = DataType::UInt(64);
  if (value.dtype().is_handle()) {
    return Call(u64, builtin::reinterpret(), {value});
  }
  return Cast(u64, value);
}

PrimExpr FromUIntPtr(const DataType& target_dtype, const PrimExpr& value) {
  if (target_dtype.is_handle()) {
    return Call(target_dtype, builtin::reinterpret(), {value});
  }
  return Cast(target_dtype, value);
}

PrimExpr MakeShadowAddr(const PrimExpr& base, const PrimExpr& shadow_bits) {
  const DataType ptr_type = base.dtype();
  PrimExpr addr_uint = ToUIntPtr(base);
  PrimExpr shadow = Cast(DataType::UInt(64), shadow_bits);
  PrimExpr combined = bitwise_or(addr_uint, shadow);
  return FromUIntPtr(ptr_type, combined);
}

class ShadowAddrInjector : public StmtExprMutator {
 public:
  ShadowAddrInjector(Var buf_dyn_shmem, PrimExpr thread_extent,
                     std::unordered_set<const VarNode*> a_dcu_res_vars,
                     std::unordered_set<const VarNode*> b_dcu_res_vars, bool trans_a,
                     bool trans_b, int a_elem_bytes, int b_elem_bytes)
      : buf_dyn_shmem_(std::move(buf_dyn_shmem)),
        thread_extent_(std::move(thread_extent)),
        a_dcu_res_vars_(std::move(a_dcu_res_vars)),
        b_dcu_res_vars_(std::move(b_dcu_res_vars)),
        trans_a_(trans_a),
        trans_b_(trans_b),
        a_elem_bytes_(a_elem_bytes),
        b_elem_bytes_(b_elem_bytes) {}

  static PrimFunc Substitute(PrimFunc func, const InfoCollector& info, bool trans_a,
                             bool trans_b, int a_elem_bytes, int b_elem_bytes) {
    ShadowAddrInjector mutator(info.buf_dyn_shmem, info.thread_extent, info.a_dcu_res_vars,
                               info.b_dcu_res_vars, trans_a, trans_b, a_elem_bytes,
                               b_elem_bytes);
    auto* n = func.CopyOnWrite();
    n->body = mutator(std::move(n->body));
    return func;
  }

 private:
  bool IsBufDynShmem(const PrimExpr& expr) const {
    if (!buf_dyn_shmem_.defined()) {
      return false;
    }
    if (IsSameVar(expr, buf_dyn_shmem_)) {
      return true;
    }
    if (const auto* var = expr.as<VarNode>()) {
      return var->name_hint == buf_dyn_shmem_->name_hint;
    }
    return false;
  }

  bool IsADcuRes(const PrimExpr& expr) const {
    if (const auto* var = expr.as<VarNode>()) {
      return a_dcu_res_vars_.count(var) > 0;
    }
    return false;
  }

  bool IsBDcuRes(const PrimExpr& expr) const {
    if (const auto* var = expr.as<VarNode>()) {
      return b_dcu_res_vars_.count(var) > 0;
    }
    return false;
  }

  PrimExpr BuildShadowOffsetA() const {
    const DataType u32 = DataType::UInt(32);
    PrimExpr warp_num = truncdiv(thread_extent_, make_const(u32, 64));
    PrimExpr a_mod = truncdiv(warp_num, make_const(u32, 2));
    PrimExpr warp_mod = floormod(Cast(u32, warp_idx_var_), a_mod);
    return left_shift(warp_mod, make_const(u32, 16));
  }

  PrimExpr BuildShadowOffsetB(int elem_bytes) const {
    const DataType u32 = DataType::UInt(32);
    PrimExpr warp_mod;
    int shift_bits;
    if (elem_bytes == 1) {
      // int8 and other 1-byte types: (warp_idx & 3) << 17
      warp_mod = floormod(Cast(u32, warp_idx_var_), make_const(u32, 4));
      shift_bits = 17;
    } else {
      // bf16/half and other 2-byte types: (warp_idx & 1) << 18
      warp_mod = floormod(Cast(u32, warp_idx_var_), make_const(u32, 2));
      shift_bits = 18;
    }
    return left_shift(warp_mod, make_const(u32, shift_bits));
  }

  PrimExpr ShadowOffsetForA() const {
    // Default (A not transposed): A-style offset; A transposed: B-style offset.
    return trans_a_ ? BuildShadowOffsetB(a_elem_bytes_) : BuildShadowOffsetA();
  }

  PrimExpr ShadowOffsetForB() const {
    // Default (B not transposed): B-style offset; B transposed: A-style offset.
    return trans_b_ ? BuildShadowOffsetA() : BuildShadowOffsetB(b_elem_bytes_);
  }

  Stmt InjectShadowAddrLets(Stmt body) {
    ICHECK(buf_dyn_shmem_.defined())
        << "buf_dyn_shmem must be defined before shadow addr injection";
    ICHECK(warp_idx_var_.defined()) << "warp_idx must be defined before shadow addr injection";
    const DataType ptr_type = buf_dyn_shmem_->dtype;
    const DataType u32 = DataType::UInt(32);
    shmem_base_var_ = Var("shmem_base", ptr_type);
    shadow_offset_a_var_ = Var("shadow_offset_a", u32);
    shadow_offset_b_var_ = Var("shadow_offset_b", u32);
    a_shared_addr_var_ = Var("A_shared_addr", ptr_type);
    b_shared_addr_var_ = Var("B_shared_addr", ptr_type);
    // Outermost lets run first: bind shmem_base, then offsets, then shadow-tagged addrs.
    body = LetStmt(a_shared_addr_var_, MakeShadowAddr(shmem_base_var_, shadow_offset_a_var_), body);
    body = LetStmt(b_shared_addr_var_, MakeShadowAddr(shmem_base_var_, shadow_offset_b_var_), body);
    body = LetStmt(shadow_offset_a_var_, ShadowOffsetForA(), body);
    body = LetStmt(shadow_offset_b_var_, ShadowOffsetForB(), body);
    body = LetStmt(shmem_base_var_, buf_dyn_shmem_, body);
    shadow_injected_ = true;
    return body;
  }

  PrimExpr RewriteDCUAsyncCopy(const CallNode* op) {
    if (!op->op.same_as(DCUAsyncCopyOp()) || !shadow_injected_ || op->args.size() < 3 ||
        !IsBufDynShmem(op->args[0])) {
      return GetRef<PrimExpr>(op);
    }
    PrimExpr new_base = op->args[0];
    if (IsADcuRes(op->args[2])) {
      new_base = a_shared_addr_var_;
    } else if (IsBDcuRes(op->args[2])) {
      new_base = b_shared_addr_var_;
    } else {
      return GetRef<PrimExpr>(op);
    }
    Array<PrimExpr> new_args = op->args;
    new_args.Set(0, new_base);
    return Call(op->dtype, op->op, new_args);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    if (IsWarpIdxInit(op->value)) {
      warp_idx_var_ = op->var;
      Stmt body = InjectShadowAddrLets(op->body);
      body = VisitStmt(body);
      return LetStmt(op->var, op->value, body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      PrimExpr rewritten = RewriteDCUAsyncCopy(call);
      if (!rewritten.same_as(op->value)) {
        return Evaluate(rewritten);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    PrimExpr rewritten = RewriteDCUAsyncCopy(op);
    if (!rewritten.same_as(GetRef<PrimExpr>(op))) {
      return rewritten;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Var buf_dyn_shmem_;
  PrimExpr thread_extent_;
  std::unordered_set<const VarNode*> a_dcu_res_vars_;
  std::unordered_set<const VarNode*> b_dcu_res_vars_;
  bool trans_a_{false};
  bool trans_b_{false};
  int a_elem_bytes_;
  int b_elem_bytes_;
  Var warp_idx_var_;
  Var shmem_base_var_;
  Var shadow_offset_a_var_;
  Var shadow_offset_b_var_;
  Var a_shared_addr_var_;
  Var b_shared_addr_var_;
  bool shadow_injected_{false};
};

int ResolveElemBytes(const PrimFunc& func, const InfoCollector& collector,
                     const std::function<bool(const std::string&)>& is_buffer,
                     std::optional<int> collected_bytes, int default_bytes) {
  for (const auto& kv : func->buffer_map) {
    if (is_buffer(kv.second->name)) {
      if (auto elem_bytes = GetBElemBytesFromDType(kv.second->dtype)) {
        return *elem_bytes;
      }
    }
  }
  if (collected_bytes.has_value()) {
    return *collected_bytes;
  }
  return default_bytes;
}

int ResolveAElemBytes(const PrimFunc& func, const InfoCollector& collector) {
  return ResolveElemBytes(func, collector, IsABufferName, std::nullopt, 2);
}

int ResolveBElemBytes(const PrimFunc& func, const InfoCollector& collector) {
  return ResolveElemBytes(func, collector, IsBBufferName, collector.b_elem_bytes, 2);
}

}  // namespace

PrimFunc InjectDCUAsyncCopyShadowAddrToFunc(PrimFunc func) {
  InfoCollector collector;
  collector(func->body);

  if (!collector.buf_dyn_shmem.defined() || !collector.thread_extent.defined()) {
    return func;
  }

  class CopyChecker : public StmtExprVisitor {
   public:
    bool has_copy{false};

   private:
    void VisitExpr_(const CallNode* op) final {
      if (op->op.same_as(DCUAsyncCopyOp())) {
        has_copy = true;
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  } copy_checker;
  copy_checker(func->body);
  if (!copy_checker.has_copy) {
    return func;
  }

  if (collector.a_dcu_res_vars.empty() && collector.b_dcu_res_vars.empty()) {
    return func;
  }

  const int a_elem_bytes = ResolveAElemBytes(func, collector);
  const int b_elem_bytes = ResolveBElemBytes(func, collector);

  DCUGemmTransposeFlags flags = GetDCUGemmTransposeFlags(func);
  const bool trans_a = flags.defined && flags.trans_a;
  const bool trans_b = flags.defined && flags.trans_b;

  return ShadowAddrInjector::Substitute(func, collector, trans_a, trans_b, a_elem_bytes,
                                        b_elem_bytes);
}

namespace transform {

using namespace tir::transform;

tvm::transform::Pass InjectDCUAsyncCopyShadowAddr() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    if (!IsDCUTarget(m)) {
      return f;
    }
    return InjectDCUAsyncCopyShadowAddrToFunc(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectDCUAsyncCopyShadowAddr", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectDCUAsyncCopyShadowAddr", InjectDCUAsyncCopyShadowAddr);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
