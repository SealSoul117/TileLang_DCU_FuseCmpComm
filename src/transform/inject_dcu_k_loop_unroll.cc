/*!
 * \file inject_dcu_k_loop_unroll.cc
 * \brief Unroll the unique DCU GEMM k-loop that contains global->shared copy and
 *        tvm_mmac by a configurable factor (default 4).
 */
#include <tvm/arith/pattern.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>
#include <string>
#include <vector>

#include "../op/builtin.h"
#include "inject_utils.h"
#include "common/pipeline_attrs.h"
#include "tir/transforms/ir_utils.h"

using tvm::ffi::GetRef;

namespace tvm {
namespace tl {

using namespace tir;

namespace {

struct CandidateLoop {
  const ForNode* node{nullptr};
};

class CandidateCollector : public StmtVisitor {
 public:
  std::vector<CandidateLoop> candidates;

 private:
  void VisitStmt_(const ForNode* op) final {
    if (IsDCUMainLoop(op)) {
      candidates.push_back({op});
    }
    StmtVisitor::VisitStmt_(op);
  }
};

std::optional<int64_t> GetConstIntExtent(const PrimExpr& extent) {
  if (const auto* imm = extent.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

std::optional<int64_t> GetConstIntValue(const PrimExpr& expr) {
  if (const auto* imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

/*! \brief Collect power-of-two divisors of the unroll factor (e.g. 2, 4 for factor 4). */
std::vector<int64_t> GetPipelineDivisors(int factor) {
  std::vector<int64_t> divs;
  for (int64_t d = 2; d <= factor; d *= 2) {
    if (factor % d == 0) {
      divs.push_back(d);
    }
  }
  return divs;
}

Stmt SubstituteLoopVar(const Stmt& stmt, const Var& loop_var, const PrimExpr& new_expr) {
  return Substitute(stmt, {{loop_var, new_expr}});
}

/*!
 * \brief Fold pipeline slot indices like (k * 4) % 4 or (k * 4 + 1) % 4 (and &3/&1)
 *        to constants after k is rewritten as outer_k * factor + unroll_offset.
 */
class PipelineIndexFolder : public StmtExprMutator {
 public:
  PipelineIndexFolder(const Var& loop_var, int factor)
      : loop_var_(loop_var), divisors_(GetPipelineDivisors(factor)) {}

 private:
  Var loop_var_;
  std::vector<int64_t> divisors_;

  std::optional<int64_t> MatchDivisor(const PrimExpr& rhs) const {
      if (auto val = GetConstIntValue(rhs)) {
          int64_t v = *val;
          // 必须是正数，且是 2 的幂
          if (v > 0 && (v & (v - 1)) == 0) {
              return v;  // 直接返回 2^n
          }
      }
      return std::nullopt;
  }

  std::optional<PrimExpr> TryFoldMod(const PrimExpr& lhs, int64_t divisor) const {
    ffi::Array<PrimExpr> linear = arith::DetectLinearEquation(lhs, {loop_var_});
    if (linear.size() != 2) {
      return std::nullopt;
    }
    auto coeff = GetConstIntValue(linear[0]);
    if (!coeff.has_value()) {
      return std::nullopt;
    }
    if (*coeff == 0) {
      // if (auto folded_val = GetConstIntValue(floormod(linear[1], make_const(loop_var_.dtype(), divisor)))) {
      //   return make_const(loop_var_.dtype(), *folded_val);
      // }
      return std::nullopt;
    }
    if ((*coeff % divisor) != 0) {
      return std::nullopt;
    }
    PrimExpr folded = floormod(linear[1], make_const(loop_var_.dtype(), divisor));
    if (auto folded_val = GetConstIntValue(folded)) {
      *folded_val;
      return make_const(loop_var_.dtype(), *folded_val);
    }
    return std::nullopt;
  }

  PrimExpr VisitExpr_(const FloorModNode* op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);

    if (auto divisor = MatchDivisor(rhs)) {
      if (auto folded = TryFoldMod(lhs, *divisor)) {
        return *folded;
      }
    }
    if (!lhs.same_as(op->a) || !rhs.same_as(op->b)) {
      return FloorMod(lhs, rhs);
    }
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::bitwise_and()) && op->args.size() == 2) {
      PrimExpr lhs = VisitExpr(op->args[0]);
      PrimExpr rhs = VisitExpr(op->args[1]);
      if (auto divisor = MatchDivisor(rhs)) {
        if (auto folded = TryFoldMod(lhs, *divisor)) {
          return *folded;
        }
      }
      if (!lhs.same_as(op->args[0]) || !rhs.same_as(op->args[1])) {
        return Call(op->dtype, op->op, {lhs, rhs});
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }
};

Stmt FoldPipelineIndices(const Stmt& stmt, const Var& loop_var, int factor) {
  return PipelineIndexFolder(loop_var, factor)(stmt);
}

Stmt BuildUnrolledCopy(const Stmt& body, const Var& loop_var, int factor, int64_t offset) {
  PrimExpr replaced = loop_var * factor + make_const(loop_var.dtype(), offset);
  Stmt substituted = SubstituteLoopVar(body, loop_var, replaced);
  return FoldPipelineIndices(substituted, loop_var, factor);
}

Stmt BuildUnrolledBody(const Stmt& body, const Var& loop_var, int factor) {
  if (factor == 1) {
    return body;
  }
  Array<Stmt> parts;
  parts.reserve(factor);
  for (int i = 0; i < factor; ++i) {
    parts.push_back(BuildUnrolledCopy(body, loop_var, factor, i));
  }
  return SeqStmt::Flatten(parts);
}

class DCUKLoopUnrollMutator : public StmtMutator {
 public:
  DCUKLoopUnrollMutator(int factor, bool* transformed)
      : factor_(factor), transformed_(transformed) {}

 private:
  int factor_;
  bool* transformed_;

  Stmt VisitStmt_(const ForNode* op) final {
    if (!IsDCUMainLoop(op) || *transformed_) {
      return StmtMutator::VisitStmt_(op);
    }
    // Opt-in via T.Pipelined(enable_register_pipeline=True). Unroll folds
    // register ping-pong indices such as (k * factor) % stages.
    if (!LoopHasRegisterPipeline(op)) {
      return StmtMutator::VisitStmt_(op);
    }

    auto extent_opt = GetConstIntExtent(op->extent);
    if (!extent_opt.has_value() || *extent_opt <= 0) {
      return StmtMutator::VisitStmt_(op);
    }

    const int64_t old_extent = *extent_opt;
    if (factor_ <= 1) {
      return StmtMutator::VisitStmt_(op);
    }

    const int64_t main_extent = old_extent / factor_;
    const int64_t remainder = old_extent % factor_;

    Stmt body = op->body;
    Stmt main_body = BuildUnrolledBody(body, op->loop_var, factor_);

    Array<Stmt> result;
    if (main_extent > 0) {
      result.push_back(
          For(op->loop_var, op->min, make_const(op->extent.dtype(), main_extent), op->kind,
              std::move(main_body), op->thread_binding, op->annotations));
    }

    if (remainder > 0) {
      Var k_epi = op->loop_var.copy_with_suffix("_epi");
      PrimExpr epilogue_base =
          make_const(op->loop_var.dtype(), main_extent * factor_);
      Stmt epilogue_body =
          SubstituteLoopVar(body, op->loop_var, epilogue_base + k_epi);
      if (remainder == 1) {
        epilogue_body = SubstituteLoopVar(epilogue_body, k_epi, make_zero(k_epi.dtype()));
      }
      epilogue_body = FoldPipelineIndices(epilogue_body, k_epi, factor_);
      result.push_back(For(k_epi, op->min, make_const(op->extent.dtype(), remainder),
                           op->kind, std::move(epilogue_body), std::nullopt, {}));
    }

    if (result.empty()) {
      return Evaluate(0);
    }

    *transformed_ = true;
    if (result.size() == 1) {
      return result[0];
    }
    return SeqStmt::Flatten(result);
  }
};

PrimFunc UnrollDCUKLoopOnFunc(PrimFunc func, int factor) {
  bool transformed = false;
  auto* n = func.CopyOnWrite();
  n->body = DCUKLoopUnrollMutator(factor, &transformed)(std::move(n->body));
  if (transformed) {
    n->body = ConvertSSA(std::move(n->body));
  }
  return func;
}

int GetUnrollFactor(const tvm::transform::PassContext& ctx) {
  constexpr const char* kKey = "tl.dcu_k_loop_unroll_factor";
  if (ctx.defined()) {
    auto val = ctx->GetConfig<Integer>(kKey);
    if (val.defined()) {
      return static_cast<int>((*val).IntValue());
    }
  }
  return 4;
}

}  // namespace

using namespace tir::transform;

tvm::transform::Pass UnrollDCUKLoop() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    if (!IsDCUTarget(m)) {
      return f;
    }

    CandidateCollector collector;
    collector(f->body);
    bool any_enabled = false;
    for (const auto &c : collector.candidates) {
      if (LoopHasRegisterPipeline(c.node)) {
        any_enabled = true;
        break;
      }
    }
    if (!any_enabled) {
      return f;
    }

    int factor = GetUnrollFactor(ctx);
    if (factor <= 1) {
      return f;
    }

    return UnrollDCUKLoopOnFunc(std::move(f), factor);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.UnrollDCUKLoop", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.UnrollDCUKLoop", UnrollDCUKLoop);
}

}  // namespace tl
}  // namespace tvm
