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

#include <cctype>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

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

bool IsALocalBufferName(const std::string& name) {
  return name.find("A_local") != std::string::npos;
}

bool IsBLocalBufferName(const std::string& name) {
  return name.find("B_local") != std::string::npos;
}

const BufferLoadNode* PeelBufferLoad(const PrimExpr& value) {
  if (const auto* load = value.as<BufferLoadNode>()) {
    return load;
  }
  if (const auto* cast = value.as<CastNode>()) {
    return PeelBufferLoad(cast->value);
  }
  return nullptr;
}

bool IsSharedScopeName(const Buffer& buf) {
  const std::string scope = buf.scope();
  const std::string name(buf->name);
  return scope == "shared" || scope == "shared.dyn" ||
         name.find("shared") != std::string::npos || name.find("shmem") != std::string::npos;
}

// Q -> Q_shared, K -> K_shared, A / A_1 -> A_shared.
bool SharedNameMatchesGlobal(const std::string& shared_name, const std::string& global_name) {
  if (shared_name.empty() || global_name.empty()) {
    return false;
  }
  if (shared_name == global_name || shared_name == global_name + "_shared") {
    return true;
  }
  auto strip_numeric_suffix = [](std::string name) {
    while (!name.empty() && std::isdigit(static_cast<unsigned char>(name.back()))) {
      name.pop_back();
    }
    if (!name.empty() && name.back() == '_') {
      name.pop_back();
    }
    return name;
  };
  auto prefix_match = [&](const std::string& prefix) {
    if (prefix.empty() || shared_name.rfind(prefix, 0) != 0) {
      return false;
    }
    const std::string rest = shared_name.substr(prefix.size());
    return rest.empty() || rest[0] == '_' || rest.find("shared") != std::string::npos;
  };
  if (prefix_match(global_name)) {
    return true;
  }
  const std::string base = strip_numeric_suffix(global_name);
  return base != global_name && prefix_match(base);
}

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

std::optional<int64_t> GetConstInt(const PrimExpr& expr) {
  if (const auto* imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

constexpr int kDCUWarpSize = 64;

int GetRequiredShadowSlots(int copy_major) { return copy_major == 0 ? 4 : 2; }

int ComputeSplitFactor(int copy_major, int warp_num) {
  const int required_slots = GetRequiredShadowSlots(copy_major);
  if (warp_num >= required_slots) {
    return 1;
  }
  ICHECK_GT(warp_num, 0) << "warp_num must be positive, got " << warp_num;
  ICHECK_EQ(required_slots % warp_num, 0)
      << "required shadow slots " << required_slots << " must be divisible by warp_num "
      << warp_num;
  return required_slots / warp_num;
}

class VarReplacer : public ExprMutator {
 public:
  VarReplacer(Var target, PrimExpr replacement)
      : target_(std::move(target)), replacement_(std::move(replacement)) {}

 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == target_.get()) {
      return replacement_;
    }
    return GetRef<PrimExpr>(op);
  }

  Var target_;
  PrimExpr replacement_;
};

PrimExpr ReplaceVar(const PrimExpr& expr, const Var& target, const PrimExpr& replacement) {
  return VarReplacer(target, replacement)(expr);
}

enum class SplitBufferKind { kNone, kA, kB };

const CallNode* ExtractDCUAsyncCopyFromStmt(const Stmt& stmt) {
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      if (call->op.same_as(DCUAsyncCopyOp())) {
        return call;
      }
    }
  }
  return nullptr;
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
  std::unordered_set<std::string> a_shared_names;
  std::unordered_set<std::string> b_shared_names;
  std::vector<std::pair<const VarNode*, std::string>> pending_dcu_res;

  void ClassifyPendingResources() {
    for (const auto& item : pending_dcu_res) {
      const VarNode* var = item.first;
      const std::string& global_name = item.second;
      if (a_dcu_res_vars.count(var) > 0 || b_dcu_res_vars.count(var) > 0) {
        continue;
      }
      bool match_a = false;
      bool match_b = false;
      for (const std::string& shared_name : a_shared_names) {
        match_a = match_a || SharedNameMatchesGlobal(shared_name, global_name);
      }
      for (const std::string& shared_name : b_shared_names) {
        match_b = match_b || SharedNameMatchesGlobal(shared_name, global_name);
      }
      if (match_a && !match_b) {
        a_dcu_res_vars.insert(var);
      } else if (match_b && !match_a) {
        b_dcu_res_vars.insert(var);
      }
    }
  }

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
    if (IsBBufferName(op->buffer->name) || b_shared_names.count(op->buffer->name) > 0) {
      b_elem_bytes = GetBElemBytesFromDType(op->buffer->dtype);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    const bool is_a_local = IsALocalBufferName(op->buffer->name);
    const bool is_b_local = IsBLocalBufferName(op->buffer->name);
    if (is_a_local || is_b_local) {
      if (const BufferLoadNode* load = PeelBufferLoad(op->value)) {
        if (IsSharedScopeName(load->buffer)) {
          if (is_a_local) {
            a_shared_names.insert(load->buffer->name);
          } else {
            b_shared_names.insert(load->buffer->name);
            if (!b_elem_bytes.has_value()) {
              b_elem_bytes = GetBElemBytesFromDType(load->buffer->dtype);
            }
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(MakeDCUResourceOp()) && call->args.size() >= 1) {
        const std::string global_name = GetGlobalBufferName(call->args[0]);
        pending_dcu_res.emplace_back(op->var.get(), global_name);
        if (IsABufferName(global_name)) {
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
  ShadowAddrInjector(Var buf_dyn_shmem,
                     std::unordered_set<const VarNode*> a_dcu_res_vars,
                     std::unordered_set<const VarNode*> b_dcu_res_vars,
                     std::unordered_set<const VarNode*> noshadow_dcu_res_vars, int a_copy_major,
                     int b_copy_major, int a_elem_bytes, int b_elem_bytes, int warp_num,
                     bool a_need_layout, bool b_need_layout)
      : buf_dyn_shmem_(std::move(buf_dyn_shmem)),
        a_dcu_res_vars_(std::move(a_dcu_res_vars)),
        b_dcu_res_vars_(std::move(b_dcu_res_vars)),
        noshadow_dcu_res_vars_(std::move(noshadow_dcu_res_vars)),
        a_copy_major_(a_copy_major),
        b_copy_major_(b_copy_major),
        a_elem_bytes_(a_elem_bytes),
        b_elem_bytes_(b_elem_bytes),
        warp_num_(warp_num),
        a_need_layout_(a_need_layout),
        b_need_layout_(b_need_layout),
        a_split_factor_(a_need_layout ? ComputeSplitFactor(a_copy_major, warp_num) : 1),
        b_split_factor_(b_need_layout ? ComputeSplitFactor(b_copy_major, warp_num) : 1) {}

  static PrimFunc Substitute(PrimFunc func, const InfoCollector& info, int a_copy_major,
                             int b_copy_major, int a_elem_bytes, int b_elem_bytes,
                             bool a_need_layout, bool b_need_layout,
                             std::unordered_set<const VarNode*> noshadow_dcu_res_vars) {
    const int64_t thread_extent = GetConstInt(info.thread_extent).value_or(0);
    ICHECK_GT(thread_extent, 0) << "thread_extent must be positive for shadow addr injection";
    const int warp_num = std::max(1, static_cast<int>(thread_extent / kDCUWarpSize));
    ShadowAddrInjector mutator(info.buf_dyn_shmem, info.a_dcu_res_vars, info.b_dcu_res_vars,
                               std::move(noshadow_dcu_res_vars), a_copy_major, b_copy_major,
                               a_elem_bytes, b_elem_bytes, warp_num, a_need_layout, b_need_layout);
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

  bool IsNoShadowDcuRes(const PrimExpr& expr) const {
    if (const auto* var = expr.as<VarNode>()) {
      return noshadow_dcu_res_vars_.count(var) > 0;
    }
    return false;
  }

  // Shadow offset follows T.copy major order (0: row_major, 1: col_major).
  // half_t (2-byte): row -> (warp_idx & 3) << 16, col -> (warp_idx & 1) << 18.
  // When warp_num is smaller than the required shadow slots, the missing slots are
  // covered by splitting the surrounding loop and using per-sub shadow addresses:
  //   ((warp_idx + sub_idx * warp_num) & mask) << shift_bits
  // e.g. warp_num=2: sub0 -> (warp_idx & 3) << 16, sub1 -> ((warp_idx + 2) & 3) << 16.
  PrimExpr BuildShadowOffset(int copy_major, int elem_bytes, int sub_idx) const {
    const DataType u32 = DataType::UInt(32);
    int shift_bits;
    int slot_mask;
    if (copy_major == 0) {
      slot_mask = 4;
      shift_bits = 16;
    } else {
      slot_mask = 2;
      shift_bits = (elem_bytes == 1) ? 17 : 18;
    }
    PrimExpr slot_index = floormod(Cast(u32, warp_idx_var_) + make_const(u32, sub_idx * warp_num_),
                                   make_const(u32, slot_mask));
    return left_shift(slot_index, make_const(u32, shift_bits));
  }

  PrimExpr ShadowOffsetForA(int sub_idx = 0) const {
    return BuildShadowOffset(a_copy_major_, a_elem_bytes_, sub_idx);
  }

  PrimExpr ShadowOffsetForB(int sub_idx = 0) const {
    return BuildShadowOffset(b_copy_major_, b_elem_bytes_, sub_idx);
  }

  void InjectSplitBufferAddrs(Stmt& body, const DataType& ptr_type, const DataType& u32,
                              int split_factor, bool is_a_buffer, const std::string& offset_tag,
                              const std::string& addr_tag, std::vector<Var>* addr_vars) {
    addr_vars->clear();
    if (split_factor == 1) {
      Var shadow_offset_var = Var("shadow_offset_" + offset_tag, u32);
      Var shared_addr_var = Var(addr_tag + "_shared_addr", ptr_type);
      addr_vars->push_back(shared_addr_var);
      body = LetStmt(shared_addr_var, MakeShadowAddr(shmem_base_var_, shadow_offset_var), body);
      body = LetStmt(shadow_offset_var, is_a_buffer ? ShadowOffsetForA() : ShadowOffsetForB(), body);
      return;
    }

    for (int sub_idx = 0; sub_idx < split_factor; ++sub_idx) {
      const std::string suffix = std::to_string(sub_idx);
      Var shadow_offset_var = Var("shadow_offset_" + offset_tag + "_" + suffix, u32);
      Var shared_addr_var = Var(addr_tag + "_shared_addr_" + suffix, ptr_type);
      addr_vars->push_back(shared_addr_var);
      PrimExpr shadow_offset =
          is_a_buffer ? ShadowOffsetForA(sub_idx) : ShadowOffsetForB(sub_idx);
      body = LetStmt(shared_addr_var, MakeShadowAddr(shmem_base_var_, shadow_offset_var), body);
      body = LetStmt(shadow_offset_var, shadow_offset, body);
    }
  }

  Stmt InjectShadowAddrLets(Stmt body) {
    ICHECK(buf_dyn_shmem_.defined())
        << "buf_dyn_shmem must be defined before shadow addr injection";
    ICHECK(warp_idx_var_.defined()) << "warp_idx must be defined before shadow addr injection";
    const DataType ptr_type = buf_dyn_shmem_->dtype;
    const DataType u32 = DataType::UInt(32);
    shmem_base_var_ = Var("shmem_base", ptr_type);
    if (a_need_layout_) {
      InjectSplitBufferAddrs(body, ptr_type, u32, a_split_factor_, true, "a", "A",
                             &a_shared_addr_vars_);
    }
    if (b_need_layout_) {
      InjectSplitBufferAddrs(body, ptr_type, u32, b_split_factor_, false, "b", "B",
                             &b_shared_addr_vars_);
    }
    // Fast-copy tiles (need_layout=0) reuse the same LDS allocation but must
    // not inherit Q/K shadow bits. Declare a second pointer at the raw base.
    if (!noshadow_dcu_res_vars_.empty() && (a_need_layout_ || b_need_layout_)) {
      plain_shared_addr_var_ = Var("plain_shared_addr", ptr_type);
      body = LetStmt(plain_shared_addr_var_, shmem_base_var_, body);
    }
    if (a_need_layout_ || b_need_layout_) {
      body = LetStmt(shmem_base_var_, buf_dyn_shmem_, body);
    }
    shadow_injected_ = true;
    return body;
  }

  SplitBufferKind GetSplitBufferForCopy(const CallNode* call) const {
    if (call == nullptr || !call->op.same_as(DCUAsyncCopyOp())) {
      return SplitBufferKind::kNone;
    }
    if (IsNoShadowDcuRes(call->args[2])) {
      return SplitBufferKind::kNone;
    }
    if (IsADcuRes(call->args[2])) {
      return a_need_layout_ && a_split_factor_ > 1 ? SplitBufferKind::kA : SplitBufferKind::kNone;
    }
    if (IsBDcuRes(call->args[2])) {
      return b_need_layout_ && b_split_factor_ > 1 ? SplitBufferKind::kB : SplitBufferKind::kNone;
    }
    return SplitBufferKind::kNone;
  }

  Stmt SplitForWithAsyncCopy(const ForNode* op, const CallNode* copy_call, SplitBufferKind kind,
                             int split_factor) {
    const std::vector<Var>& addr_vars =
        kind == SplitBufferKind::kA ? a_shared_addr_vars_ : b_shared_addr_vars_;
    ICHECK_EQ(static_cast<int>(addr_vars.size()), split_factor);

    const auto extent_imm = GetConstInt(op->extent);
    ICHECK(extent_imm.has_value()) << "loop extent must be constant for shadow loop split";
    ICHECK_EQ(*extent_imm % split_factor, 0)
        << "loop extent " << *extent_imm << " must be divisible by split factor "
        << split_factor;

    Array<Stmt> seq;
    for (int sub_idx = 0; sub_idx < split_factor; ++sub_idx) {
      PrimExpr loop_replacement = op->loop_var * split_factor + sub_idx;
      Array<PrimExpr> new_args = copy_call->args;
      new_args.Set(0, addr_vars[sub_idx]);
      for (size_t arg_idx = 1; arg_idx < new_args.size(); ++arg_idx) {
        new_args.Set(arg_idx, ReplaceVar(new_args[arg_idx], op->loop_var, loop_replacement));
      }
      seq.push_back(Evaluate(Call(copy_call->dtype, copy_call->op, new_args)));
    }

    Stmt new_body = seq.size() == 1 ? seq[0] : SeqStmt(seq);
    const int64_t new_extent = *extent_imm / split_factor;
    return For(op->loop_var, op->min, make_const(op->extent.dtype(), new_extent), op->kind,
               new_body, op->thread_binding, op->annotations);
  }

  PrimExpr RewriteDCUAsyncCopy(const CallNode* op) {
    if (!op->op.same_as(DCUAsyncCopyOp()) || !shadow_injected_ || op->args.size() < 3 ||
        !IsBufDynShmem(op->args[0])) {
      return GetRef<PrimExpr>(op);
    }
    PrimExpr new_base = op->args[0];
    if (IsNoShadowDcuRes(op->args[2])) {
      if (!plain_shared_addr_var_.defined()) {
        return GetRef<PrimExpr>(op);
      }
      new_base = plain_shared_addr_var_;
    } else if (IsADcuRes(op->args[2])) {
      if (!a_need_layout_ || a_shared_addr_vars_.empty()) {
        return GetRef<PrimExpr>(op);
      }
      new_base = a_shared_addr_vars_[0];
    } else if (IsBDcuRes(op->args[2])) {
      if (!b_need_layout_ || b_shared_addr_vars_.empty()) {
        return GetRef<PrimExpr>(op);
      }
      new_base = b_shared_addr_vars_[0];
    } else {
      return GetRef<PrimExpr>(op);
    }
    Array<PrimExpr> new_args = op->args;
    new_args.Set(0, new_base);
    return Call(op->dtype, op->op, new_args);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if (shadow_injected_) {
      if (const CallNode* copy_call = ExtractDCUAsyncCopyFromStmt(op->body)) {
        const SplitBufferKind kind = GetSplitBufferForCopy(copy_call);
        if (kind != SplitBufferKind::kNone) {
          const int split_factor = kind == SplitBufferKind::kA ? a_split_factor_ : b_split_factor_;
          return SplitForWithAsyncCopy(op, copy_call, kind, split_factor);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
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
  std::unordered_set<const VarNode*> a_dcu_res_vars_;
  std::unordered_set<const VarNode*> b_dcu_res_vars_;
  std::unordered_set<const VarNode*> noshadow_dcu_res_vars_;
  int a_copy_major_{0};
  int b_copy_major_{0};
  int a_elem_bytes_;
  int b_elem_bytes_;
  int warp_num_{1};
  bool a_need_layout_{true};
  bool b_need_layout_{true};
  int a_split_factor_{1};
  int b_split_factor_{1};
  Var warp_idx_var_;
  Var shmem_base_var_;
  Var plain_shared_addr_var_;
  std::vector<Var> a_shared_addr_vars_;
  std::vector<Var> b_shared_addr_vars_;
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
  collector.ClassifyPendingResources();

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

  DCUCopyBufferMaps maps = GetDCUCopyBufferMaps(func);
  auto copy_major_for = [&](int operand, const std::string& fallback) {
    std::string first;
    for (const auto& kv : maps.gemm_operand) {
      if (kv.second != operand) {
        continue;
      }
      if (first.empty()) {
        first = kv.first;
      }
      if (LookupDCUNeedLayout(maps, kv.first, true)) {
        return LookupDCUCopyMajor(maps, kv.first, 0);
      }
    }
    if (!first.empty()) {
      return LookupDCUCopyMajor(maps, first, 0);
    }
    return LookupDCUCopyMajor(maps, fallback, 0);
  };
  auto any_operand_needs_layout = [&](int operand) {
    bool found = false;
    bool any = false;
    for (const auto& kv : maps.gemm_operand) {
      if (kv.second != operand) {
        continue;
      }
      found = true;
      any = any || LookupDCUNeedLayout(maps, kv.first, true);
    }
    return found ? any : true;
  };
  const int a_copy_major = copy_major_for(0, "A_shared");
  const int b_copy_major = copy_major_for(1, "B_shared");
  const bool a_need_layout = any_operand_needs_layout(0);
  const bool b_need_layout = any_operand_needs_layout(1);
  if (!a_need_layout && !b_need_layout) {
    return func;
  }

  std::unordered_set<const VarNode*> noshadow_dcu_res_vars;
  for (const auto& item : collector.pending_dcu_res) {
    if (!LookupDCUNeedLayout(maps, item.second, true)) {
      noshadow_dcu_res_vars.insert(item.first);
    }
  }

  return ShadowAddrInjector::Substitute(func, collector, a_copy_major, b_copy_major, a_elem_bytes,
                                        b_elem_bytes, a_need_layout, b_need_layout,
                                        std::move(noshadow_dcu_res_vars));
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
