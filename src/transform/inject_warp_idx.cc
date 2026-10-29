/*!
 * \file inject_warp_idx.cc
 * \brief Declare warp_idx via tl.get_warp_idx_sync and rewrite warp-level
 *        threadIdx.x expressions for DCU codegen stability.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>
#include <string>
#include <utility>

#include "../op/builtin.h"
#include "inject_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

constexpr const char* kDCUPersistentTileAnnotation = "tl.dcu_persistent_tile";
constexpr const char* kDCUPersistentResourceVerifiedAnnotation =
    "tl.dcu_persistent_resource_verified";

bool IsExactIntegerOne(const ffi::Any& value) {
  if (auto scalar = value.as<int64_t>()) {
    return scalar.value() == 1;
  }
  if (const auto* imm = value.as<IntImmNode>()) {
    return imm->value == 1;
  }
  return false;
}

bool HasExactIntegerOneAnnotation(const ForNode* op, const char* key) {
  auto value = op->annotations.Get(key);
  return value.has_value() && IsExactIntegerOne(value.value());
}

bool IsVerifiedPersistentTileLoop(const ForNode* op) {
  return HasExactIntegerOneAnnotation(op, kDCUPersistentTileAnnotation) &&
         HasExactIntegerOneAnnotation(op, kDCUPersistentResourceVerifiedAnnotation);
}

bool IsGfx936Target() {
  Target target = Target::Current(false);
  if (!target.defined()) {
    return false;
  }
  auto mcpu = target->GetAttr<String>("mcpu");
  return mcpu.has_value() && mcpu.value() == "gfx936";
}

std::optional<int64_t> GetConstIntValue(const PrimExpr& expr) {
  if (const auto* imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

bool IsThreadVar(const PrimExpr& expr, const Var& thread_var) {
  return expr.same_as(thread_var);
}

bool IsSharedScope(const Buffer& buf) {
  auto scope = buf.scope();
  return scope == "shared" || scope == "shared.dyn";
}

bool IsGlobalScope(const Buffer& buf) {
  auto scope = buf.scope();
  return scope == "global" || scope == "";
}

const BufferLoadNode* PeelGlobalLoad(const PrimExpr& value) {
  if (const auto* load = value.as<BufferLoadNode>()) {
    return load;
  }
  if (const auto* cast = value.as<CastNode>()) {
    return cast->value.as<BufferLoadNode>();
  }
  return nullptr;
}

bool IsGlobalToSharedCopy(const BufferStoreNode* op) {
  if (!IsSharedScope(op->buffer)) {
    return false;
  }
  const BufferLoadNode* load = PeelGlobalLoad(op->value);
  return load != nullptr && IsGlobalScope(load->buffer);
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

bool IsActiveNonXThreadIdxAttr(const AttrStmtNode* op) {
  if (op->attr_key != tir::attr::thread_extent) {
    return false;
  }
  const auto* iv = op->node.as<IterVarNode>();
  if (iv == nullptr || (iv->thread_tag != "threadIdx.y" && iv->thread_tag != "threadIdx.z")) {
    return false;
  }
  // A statically-one y/z binding is semantically absent for this narrow
  // profile.  Dynamic or wider bindings are deliberately outside it.
  return !GetConstIntValue(op->value).has_value() || *GetConstIntValue(op->value) != 1;
}

}  // namespace

class WarpIdxExprRewriter : public ExprMutator {
 public:
  WarpIdxExprRewriter(Var thread_var, Var warp_idx_var, bool use_direct_lane)
      : thread_var_(std::move(thread_var)),
        warp_idx_var_(std::move(warp_idx_var)),
        use_direct_lane_(use_direct_lane) {}

  using ExprMutator::VisitExpr;

  PrimExpr VisitExpr(const PrimExpr& expr) final {
    if (skip_rewrite_) {
      return expr;
    }
    if (auto rewritten = TryRewriteWarpLevel(expr)) {
      return rewritten.value();
    }
    return ExprMutator::VisitExpr(expr);
  }

  void SetSkipRewrite(bool skip) { skip_rewrite_ = skip; }

 private:
  std::optional<PrimExpr> TryRewriteWarpLevel(const PrimExpr& expr) {
    if (const auto* div = expr.as<FloorDivNode>()) {
      return TryRewriteFloorDiv(div);
    }
    if (const auto* mod = expr.as<FloorModNode>()) {
      return TryRewriteLaneFloorMod(mod);
    }
    return std::nullopt;
  }

  std::optional<PrimExpr> TryRewriteFloorDiv(const FloorDivNode* div) {
    if (IsThreadVar(div->a, thread_var_)) {
      if (auto denom = GetConstIntValue(div->b)) {
        if (*denom >= 64 && (*denom % 64) == 0) {
          return FloorDiv(warp_idx_expr_(), make_const(DataType::Int(32), *denom / 64));
        }
      }
    }
    if (const auto* mod = div->a.as<FloorModNode>()) {
      if (IsThreadVar(mod->a, thread_var_)) {
        if (auto denom = GetConstIntValue(div->b)) {
          if (*denom == 64) {
            if (auto mod_val = GetConstIntValue(mod->b)) {
              if (*mod_val >= 64 && (*mod_val % 64) == 0) {
                return FloorMod(warp_idx_expr_(), make_const(DataType::Int(32), *mod_val / 64));
              }
            }
          }
        }
      }
    }
    return std::nullopt;
  }

  std::optional<PrimExpr> TryRewriteLaneFloorMod(const FloorModNode* mod) {
    auto modulus = GetConstIntValue(mod->b);
    if (!modulus || *modulus <= 0 || (64 % *modulus) != 0) {
      return std::nullopt;
    }
    if (IsThreadVar(mod->a, thread_var_)) {
      if (*modulus == 64) {
        return lane_idx_expr_();
      }
      return FloorMod(lane_idx_expr_(), make_const(DataType::Int(32), *modulus));
    }
    // The GEMM lowering can spell a sublane as (threadIdx.x / d) % m
    // instead of (threadIdx.x % (d*m)) / d.  When d*m divides a wave, the
    // quotient's contribution from whole waves vanishes modulo m.
    if (const auto* div = mod->a.as<FloorDivNode>()) {
      if (IsThreadVar(div->a, thread_var_)) {
        if (auto divisor = GetConstIntValue(div->b)) {
          if (*divisor > 0 && *divisor <= 64 &&
              *divisor <= 64 / *modulus && (64 % (*divisor * *modulus)) == 0) {
            return FloorMod(FloorDiv(lane_idx_expr_(),
                                     make_const(DataType::Int(32), *divisor)),
                            make_const(DataType::Int(32), *modulus));
          }
        }
      }
    }
    return std::nullopt;
  }

  std::optional<PrimExpr> TryRewriteLaneBitwiseAnd(const CallNode* op) {
    if (!op->op.same_as(builtin::bitwise_and()) || op->args.size() != 2) {
      return std::nullopt;
    }
    PrimExpr thread_expr = op->args[0];
    PrimExpr mask_expr = op->args[1];
    if (!IsThreadVar(thread_expr, thread_var_)) {
      std::swap(thread_expr, mask_expr);
    }
    if (!IsThreadVar(thread_expr, thread_var_)) {
      return std::nullopt;
    }
    if (auto mask = GetConstIntValue(mask_expr)) {
      if (*mask >= 0 && *mask <= 63) {
        return Call(op->dtype, op->op, {lane_idx_expr_(), mask_expr});
      }
    }
    return std::nullopt;
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (auto rewritten = TryRewriteLaneBitwiseAnd(op)) {
      return rewritten.value();
    }
    if (op->op.same_as(builtin::shift_right())) {
      ICHECK_EQ(op->args.size(), 2U);
      if (IsThreadVar(op->args[0], thread_var_)) {
        if (auto shift = GetConstIntValue(op->args[1])) {
          if (*shift >= 6) {
            int64_t warp_div = 1LL << (*shift - 6);
            return FloorDiv(warp_idx_expr_(), make_const(DataType::Int(32), warp_div));
          }
        }
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  PrimExpr warp_idx_expr_() const { return warp_idx_var_;}
  PrimExpr lane_idx_expr_() const {
    if (use_direct_lane_) {
      // Persistent resource lowering has proved that this tile body uses
      // direct tile-local coordinates.  For the separately-gated gfx936 1-D
      // 512-thread profile, raw threadIdx.x % 64 is the desired sublane and
      // avoids emitting get_lane_idx()/mbcnt for this lexical scope only.
      return FloorMod(thread_var_, make_const(DataType::Int(32), 64));
    }
    return Call(DataType::Int(32), tl::get_lane_idx(),
                {IntImm(DataType::Int(32), 64)});
  }

  Var thread_var_;
  Var warp_idx_var_;
  bool use_direct_lane_{false};
  bool skip_rewrite_{false};
};

class InjectWarpIdxMutator : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc func, bool persistent_profile_enabled) {
    InjectWarpIdxMutator mutator(persistent_profile_enabled);
    auto* n = func.CopyOnWrite();
    n->body = mutator(std::move(n->body));
    return func;
  }

 private:
  explicit InjectWarpIdxMutator(bool persistent_profile_enabled)
      : persistent_profile_enabled_(persistent_profile_enabled) {}

  Stmt InjectWarpIdxIntoBody(Stmt body) {
    ICHECK(thread_var_.defined()) << "thread_binding must be defined before warp_idx init";
    Array<PrimExpr> args{IntImm(DataType::Int(32), 64), PrimExpr(thread_var_)};
    PrimExpr warp_init = Call(DataType::Int(32), tl::get_warp_idx_sync(), args);
    return LetStmt(warp_idx_var_, warp_init, body);
  }

  PrimExpr RewriteExpr(const PrimExpr& expr, bool skip = false) {
    if (!thread_var_.defined() || !warp_injected_) {
      return expr;
    }
    WarpIdxExprRewriter rewriter(thread_var_, warp_idx_var_,
                                 direct_lane_scope_ && IsDirectLaneProfileActive());
    rewriter.SetSkipRewrite(skip);
    return rewriter(expr);
  }

  PrimExpr VisitExpr(const PrimExpr& expr) final { return RewriteExpr(expr); }

  bool IsDirectLaneProfileActive() const {
    return persistent_profile_enabled_ && thread_extent_x_.has_value() &&
           *thread_extent_x_ == 512 && non_x_thread_dims_ == 0;
  }

  // T.launch_thread("threadIdx.x", extent) lowers to AttrStmt(thread_extent).
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    const IterVarNode* iv = nullptr;
    if (IsThreadIdxXAttr(op, &iv)) {
      const auto old_thread_extent = thread_extent_x_;
      thread_var_ = iv->var;
      thread_extent_x_ = GetConstIntValue(op->value);
      warp_idx_var_ = Var("warp_idx", DataType::Int(32));
      warp_injected_ = true;

      Stmt body = StmtExprMutator::VisitStmt(op->body);
      body = InjectAtDCUTarget(body, [&](const Stmt& target_body) {
        return InjectWarpIdxIntoBody(target_body);
      });
      thread_extent_x_ = old_thread_extent;
      return AttrStmt(op->node, op->attr_key, op->value, body);
    }
    if (IsActiveNonXThreadIdxAttr(op)) {
      ++non_x_thread_dims_;
      Stmt result = StmtExprMutator::VisitStmt_(op);
      --non_x_thread_dims_;
      return result;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // Fallback for IR that still uses ForKind::kThreadBinding.
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        op->thread_binding.value()->thread_tag == "threadIdx.x") {
      const auto old_thread_extent = thread_extent_x_;
      thread_var_ = op->loop_var;
      thread_extent_x_ = GetConstIntValue(op->extent);
      warp_idx_var_ = Var("warp_idx", DataType::Int(32));
      warp_injected_ = true;

      Stmt body = StmtExprMutator::VisitStmt(op->body);
      body = InjectAtDCUTarget(body, [&](const Stmt& target_body) {
        return InjectWarpIdxIntoBody(target_body);
      });
      thread_extent_x_ = old_thread_extent;
      return For(op->loop_var, op->min, op->extent, op->kind, body, op->thread_binding,
                 op->annotations);
    }
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        (op->thread_binding.value()->thread_tag == "threadIdx.y" ||
         op->thread_binding.value()->thread_tag == "threadIdx.z") &&
        (!GetConstIntValue(op->extent).has_value() || *GetConstIntValue(op->extent) != 1)) {
      ++non_x_thread_dims_;
      Stmt body = StmtExprMutator::VisitStmt_(op);
      --non_x_thread_dims_;
      return body;
    }
    if (IsVerifiedPersistentTileLoop(op) && IsDirectLaneProfileActive()) {
      const bool old_direct_lane_scope = direct_lane_scope_;
      direct_lane_scope_ = true;
      Stmt body = VisitStmt(op->body);
      direct_lane_scope_ = old_direct_lane_scope;
      return For(op->loop_var, VisitExpr(op->min), VisitExpr(op->extent), op->kind, body,
                 op->thread_binding, op->annotations);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (IsGlobalToSharedCopy(op)) {
      Array<PrimExpr> indices = op->indices;
      PrimExpr value = RewriteExpr(op->value);
      return BufferStore(op->buffer, value, indices, op->predicate);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Var thread_var_;
  Var warp_idx_var_;
  bool warp_injected_{false};
  bool persistent_profile_enabled_{false};
  bool direct_lane_scope_{false};
  std::optional<int64_t> thread_extent_x_;
  int non_x_thread_dims_{0};
};

PrimFunc InjectWarpIdxToFunc(PrimFunc func) {
  return InjectWarpIdxMutator::Substitute(std::move(func), IsGfx936Target());
}

namespace transform {

using namespace tir::transform;

tvm::transform::Pass InjectWarpIdx() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    if (!IsDCUTarget(m)) {
      return f;
    }
    return InjectWarpIdxToFunc(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectWarpIdx", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectWarpIdx", InjectWarpIdx);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
