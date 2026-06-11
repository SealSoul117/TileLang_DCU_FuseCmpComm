/*!
 * \file inject_warp_k_loop_divergence.cc
 * \brief Split the DCU GEMM k-loop across warp halves with divergent mmac
 *        scheduling, and zero async_gld_sld_fence in the prologue.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>
#include <string>
#include <vector>

#include "../op/builtin.h"
#include "inject_utils.h"
#include "tir/transforms/ir_utils.h"

using tvm::ffi::GetRef;

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

bool IsWarpIdxInit(const PrimExpr& expr) {
  if (const auto* call = expr.as<CallNode>()) {
    return call->op.same_as(tl::get_warp_idx_sync());
  }
  return false;
}

bool IsAsyncGldSldFenceCall(const CallNode* call) {
  if (!call) return false;
  if (const auto* op = call->op.as<OpNode>()) {
    return op->name == "tl.async_gld_sld_fence";
  }
  return false;
}

bool IsAsyncGldFenceCall(const CallNode* call) {
  if (!call) return false;
  if (const auto* op = call->op.as<OpNode>()) {
    return op->name == "tl.async_gld_fence";
  }
  return false;
}

bool IsWaveBarrierCall(const CallNode* call) {
  if (!call) return false;
  if (const auto* op = call->op.as<OpNode>()) {
    return op->name == "tl.wave_barrier";
  }
  return false;
}

bool IsAsyncGldSldFenceStmt(const Stmt& stmt) {
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      return IsAsyncGldSldFenceCall(call);
    }
  }
  return false;
}

bool IsAsyncGldFenceStmt(const Stmt& stmt) {
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      return IsAsyncGldFenceCall(call);
    }
  }
  return false;
}

bool IsWaveBarrierStmt(const Stmt& stmt) {
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      return IsWaveBarrierCall(call);
    }
  }
  return false;
}

Stmt MakeAsyncGldSldFenceStmt(int wait_count) {
  auto fence_call =
      Call(DataType::Void(), Op::Get("tl.async_gld_sld_fence"), {Integer(wait_count)});
  return Evaluate(fence_call);
}

class ContainsMmacChecker : public StmtExprVisitor {
 public:
  bool has_mmac{false};

 private:
  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(tvm_mmac())) {
      has_mmac = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

bool ContainsMmac(const Stmt& stmt) {
  ContainsMmacChecker checker;
  checker(stmt);
  return checker.has_mmac;
}

Array<Stmt> FlattenStmts(const Stmt& stmt) {
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    Array<Stmt> result;
    for (const Stmt& s : seq->seq) {
      Array<Stmt> inner = FlattenStmts(s);
      for (const Stmt& inner_s : inner) {
        result.push_back(inner_s);
      }
    }
    return result;
  }
  return {stmt};
}

Stmt RebuildFlattened(const Array<Stmt>& stmts) {
  if (stmts.empty()) {
    return Evaluate(0);
  }
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return SeqStmt::Flatten(stmts);
}

Array<Stmt> ReorderMmacBeforeGldFence(const Array<Stmt>& stmts) {
  Array<Stmt> result;
  size_t i = 0;
  while (i < stmts.size()) {
    if (i + 2 < stmts.size() && IsAsyncGldSldFenceStmt(stmts[i]) &&
        IsAsyncGldFenceStmt(stmts[i + 1]) && IsWaveBarrierStmt(stmts[i + 2])) {
      result.push_back(stmts[i]);
      size_t j = i + 3;
      while (j < stmts.size() && ContainsMmac(stmts[j])) {
        result.push_back(stmts[j]);
        ++j;
      }
      result.push_back(stmts[i + 1]);
      result.push_back(stmts[i + 2]);
      i = j;
    } else {
      result.push_back(stmts[i]);
      ++i;
    }
  }
  return result;
}

Stmt TransformAltLoopBody(const Stmt& body) {
  return RebuildFlattened(ReorderMmacBeforeGldFence(FlattenStmts(body)));
}

class InfoCollector : public StmtExprVisitor {
 public:
  Var warp_idx_var;
  PrimExpr thread_extent;
  bool found_k_loop{false};

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
    if (IsDCUMainLoop(op)) {
      found_k_loop = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode* op) final {
    if (IsWarpIdxInit(op->value)) {
      warp_idx_var = op->var;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

enum class PipelinePhase { kBeforeMainLoop, kInMainLoop, kAfterMainLoop };

class WarpKLoopDivergenceMutator : public StmtMutator {
 public:
  WarpKLoopDivergenceMutator(Var warp_idx_var, int warp_half)
      : warp_idx_var_(std::move(warp_idx_var)), warp_half_(warp_half) {}

 private:
  Stmt VisitStmt_(const ForNode* op) override {
    if (IsDCUMainLoop(op)) {
      ICHECK(warp_idx_var_.defined()) << "warp_idx must be defined before main loop divergence";
      ICHECK(warp_half_ > 0) << "warp_half must be positive";

      phase_ = PipelinePhase::kInMainLoop;

      For orig_loop(op->loop_var, op->min, op->extent, op->kind, op->body, op->thread_binding,
                    op->annotations);
      Stmt alt_body = TransformAltLoopBody(op->body);
      For alt_loop(op->loop_var, op->min, op->extent, op->kind, alt_body, op->thread_binding,
                   op->annotations);

      PrimExpr cond = warp_idx_var_ < make_const(DataType::Int(32), warp_half_);
      phase_ = PipelinePhase::kAfterMainLoop;
      return IfThenElse(cond, orig_loop, alt_loop);
    }

    if (phase_ == PipelinePhase::kAfterMainLoop) {
      return StmtMutator::VisitStmt_(op);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode* op) override {
    if (phase_ == PipelinePhase::kBeforeMainLoop) {
      if (const auto* call = op->value.as<CallNode>()) {
        if (IsAsyncGldSldFenceCall(call)) {
          return MakeAsyncGldSldFenceStmt(0);
        }
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Var warp_idx_var_;
  int warp_half_;
  PipelinePhase phase_{PipelinePhase::kBeforeMainLoop};
};

std::optional<int> ComputeWarpHalf(const PrimExpr& thread_extent) {
  if (auto extent = GetConstIntValue(thread_extent)) {
    ICHECK(*extent > 0 && (*extent % 64) == 0)
        << "thread extent must be a positive multiple of 64, got " << *extent;
    int warp_num = static_cast<int>(*extent / 64);
    ICHECK(warp_num >= 2 && (warp_num % 2) == 0)
        << "warp count must be an even number >= 2, got " << warp_num;
    return warp_num / 2;
  }
  return std::nullopt;
}

PrimFunc InjectWarpKLoopDivergenceToFunc(PrimFunc func) {
  InfoCollector collector;
  collector(func->body);
  if (!collector.found_k_loop || !collector.warp_idx_var.defined() ||
      !collector.thread_extent.defined()) {
    return func;
  }

  auto warp_half = ComputeWarpHalf(collector.thread_extent);
  if (!warp_half.has_value()) {
    return func;
  }

  bool transformed = false;
  auto* n = func.CopyOnWrite();
  WarpKLoopDivergenceMutator mutator(collector.warp_idx_var, warp_half.value());
  n->body = mutator(std::move(n->body));
  transformed = true;

  if (transformed) {
    n->body = ConvertSSA(std::move(n->body));
  }
  return func;
}

}  // namespace

namespace transform {

using namespace tir::transform;

tvm::transform::Pass InjectWarpKLoopDivergence() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    if (!IsDCUTarget(m)) {
      return f;
    }
    return InjectWarpKLoopDivergenceToFunc(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectWarpKLoopDivergence", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectWarpKLoopDivergence", InjectWarpKLoopDivergence);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
