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

#include "../op/builtin.h"
#include "inject_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

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

}  // namespace

class WarpIdxExprRewriter : public ExprMutator {
 public:
  WarpIdxExprRewriter(Var thread_var, Var warp_idx_var)
      : thread_var_(std::move(thread_var)),
    warp_idx_var_(std::move(warp_idx_var)) {}

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

  PrimExpr VisitExpr_(const CallNode* op) final {
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

  Var thread_var_;
  Var warp_idx_var_;
  bool skip_rewrite_{false};
};

class InjectWarpIdxMutator : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc func) {
    InjectWarpIdxMutator mutator;
    auto* n = func.CopyOnWrite();
    n->body = mutator(std::move(n->body));
    return func;
  }

 private:
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
    WarpIdxExprRewriter rewriter(thread_var_, warp_idx_var_);
    rewriter.SetSkipRewrite(skip);
    return rewriter(expr);
  }

  PrimExpr VisitExpr(const PrimExpr& expr) final { return RewriteExpr(expr); }

  // T.launch_thread("threadIdx.x", extent) lowers to AttrStmt(thread_extent).
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    const IterVarNode* iv = nullptr;
    if (IsThreadIdxXAttr(op, &iv)) {
      thread_var_ = iv->var;
      warp_idx_var_ = Var("warp_idx", DataType::Int(32));
      warp_injected_ = true;

      Stmt body = StmtExprMutator::VisitStmt(op->body);
      body = InjectAtDCUTarget(body, [&](const Stmt& target_body) {
        return InjectWarpIdxIntoBody(target_body);
      });
      return AttrStmt(op->node, op->attr_key, op->value, body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // Fallback for IR that still uses ForKind::kThreadBinding.
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        op->thread_binding.value()->thread_tag == "threadIdx.x") {
      thread_var_ = op->loop_var;
      warp_idx_var_ = Var("warp_idx", DataType::Int(32));
      warp_injected_ = true;

      Stmt body = StmtExprMutator::VisitStmt(op->body);
      body = InjectAtDCUTarget(body, [&](const Stmt& target_body) {
        return InjectWarpIdxIntoBody(target_body);
      });
      return For(op->loop_var, op->min, op->extent, op->kind, body, op->thread_binding,
                 op->annotations);
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
};

PrimFunc InjectWarpIdxToFunc(PrimFunc func) {
  return InjectWarpIdxMutator::Substitute(func);
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
