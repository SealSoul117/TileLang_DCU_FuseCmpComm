/*!
 * \file insert_async_gld_fence.cc
 * \brief Move async global wait from AttrStmt to tl.async_gld_fence between
 *        tl.async_gld_sld_fence and tl.wave_barrier.
 *
 * Only applies inside the main k-loop and in the epilogue after it.
 * Prologue (before the k-loop) is left unchanged: attrs are kept and no fence
 * is inserted.
 */
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>

#include <deque>
#include <string>

using tvm::ffi::Array;
namespace tvm {
namespace tl {
namespace {

using namespace tir;

enum class PipelinePhase { kBeforeMainLoop, kInMainLoop, kAfterMainLoop };

bool IsKLoopVar(const Var& loop_var) {
  const auto* var = loop_var.as<VarNode>();
  return var && var->name_hint == "k";
}

bool IsAsyncGldSldFenceCall(const CallNode* call) {
  if (!call) return false;
  if (const auto* op = call->op.as<OpNode>()) {
    return op->name == "tl.async_gld_sld_fence";
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

bool IsWaveBarrierStmt(const Stmt& stmt) {
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      return IsWaveBarrierCall(call);
    }
  }
  return false;
}

bool ShouldTransform(const PipelinePhase phase) {
  return phase == PipelinePhase::kInMainLoop || phase == PipelinePhase::kAfterMainLoop;
}

Stmt MakeAsyncGldFenceStmt(int wait_count) {
  auto fence_call =
      Call(DataType::Void(), Op::Get("tl.async_gld_fence"), {Integer(wait_count)});
  return Evaluate(fence_call);
}

class InsertAsyncGldFenceMutator : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) override {
    if (IsKLoopVar(op->loop_var)) {
      phase_ = PipelinePhase::kInMainLoop;
      Stmt new_body = VisitStmt(op->body);
      phase_ = PipelinePhase::kAfterMainLoop;
      return For(op->loop_var, op->min, op->extent, op->kind, new_body, op->thread_binding,
                 op->annotations);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (!ShouldTransform(phase_)) {
      return StmtMutator::VisitStmt_(op);
    }

    if (op->attr_key == tir::attr::async_wait_queue_scope ||
        op->attr_key == "async_wait_queue_scope") {
      const auto* inner = op->body.as<AttrStmtNode>();
      if (inner && (inner->attr_key == tir::attr::async_wait_inflight_count ||
                    inner->attr_key == "async_wait_inflight_count")) {
        if (const auto* imm = inner->value.as<IntImmNode>()) {
          pending_wait_counts_.push_back(static_cast<int>(imm->value));
        }
        return VisitStmt(inner->body);
      }
      return VisitStmt(op->body);
    }
    if (op->attr_key == tir::attr::async_wait_inflight_count ||
        op->attr_key == "async_wait_inflight_count") {
      if (const auto* imm = op->value.as<IntImmNode>()) {
        pending_wait_counts_.push_back(static_cast<int>(imm->value));
      }
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    std::deque<int> saved_pending = std::move(pending_wait_counts_);
    pending_wait_counts_.clear();

    Array<Stmt> result;
    result.reserve(op->seq.size());
    for (const Stmt& s : op->seq) {
      result.push_back(VisitStmt(s));
      TryInsertGldFenceAfterPair(result);
    }

    for (int count : pending_wait_counts_) {
      saved_pending.push_back(count);
    }
    pending_wait_counts_ = std::move(saved_pending);

    if (result.size() == 1) {
      return result[0];
    }
    return SeqStmt(result);
  }

 private:
  void TryInsertGldFenceAfterPair(Array<Stmt>& result) {
    if (!ShouldTransform(phase_) || pending_wait_counts_.empty()) {
      return;
    }
    size_t n = result.size();
    if (n >= 2 && IsAsyncGldSldFenceStmt(result[n - 2]) && IsWaveBarrierStmt(result[n - 1])) {
      int wait_count = pending_wait_counts_.front();
      pending_wait_counts_.pop_front();
      Stmt barrier = result[n - 1];
      result.pop_back();
      result.push_back(MakeAsyncGldFenceStmt(wait_count));
      result.push_back(barrier);
    }
  }

  PipelinePhase phase_{PipelinePhase::kBeforeMainLoop};
  std::deque<int> pending_wait_counts_;
};

}  // namespace

namespace transform {
using namespace tir::transform;

Pass InsertAsyncGldFence() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    InsertAsyncGldFenceMutator mutator;
    n->body = mutator(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InsertAsyncGldFence", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InsertAsyncGldFence", InsertAsyncGldFence);
}
}  // namespace transform
}  // namespace tl
}  // namespace tvm
