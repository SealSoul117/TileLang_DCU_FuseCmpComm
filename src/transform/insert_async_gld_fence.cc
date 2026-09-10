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
#include <vector>

#include "inject_utils.h"

using tvm::ffi::Array;
using tvm::ffi::GetRef;
namespace tvm {
namespace tl {
namespace {

using namespace tir;

enum class PipelinePhase { kBeforeMainLoop, kInMainLoop, kAfterMainLoop };

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

bool IsPtxCommitStmt(const Stmt& stmt) {
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      return call->op.same_as(builtin::ptx_commit_group());
    }
  }
  return false;
}

bool IsCommitAttr(const AttrStmtNode* op) {
  return op->attr_key == tir::attr::async_commit_queue_scope ||
         op->attr_key == "async_commit_queue_scope";
}

bool IsPrologueCommitStmt(const Stmt& stmt) {
  if (IsPtxCommitStmt(stmt)) {
    return true;
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    return IsCommitAttr(attr);
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

Stmt MakeWaveBarrierStmt() {
  auto barrier_call = Call(DataType::Void(), Op::Get("tl.wave_barrier"), {});
  return Evaluate(barrier_call);
}

int GetAsyncGldSldFenceValue(const Stmt& stmt) {
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      if (IsAsyncGldSldFenceCall(call) && !call->args.empty()) {
        if (const auto* imm = call->args[0].as<IntImmNode>()) {
          return static_cast<int>(imm->value);
        }
      }
    }
  }
  return -1;
}

std::vector<int> CollectEpilogueWaitCounts(const Stmt& root) {
  struct Collector : public StmtVisitor {
    std::vector<int> counts;
    bool past_main_loop = false;

    void VisitStmt_(const ForNode* op) override {
      StmtVisitor::VisitStmt_(op);
      if (IsDCUMainLoop(op)) {
        past_main_loop = true;
      }
    }

    void VisitStmt_(const AttrStmtNode* op) override {
      if (past_main_loop) {
        if (op->attr_key == tir::attr::async_wait_queue_scope ||
            op->attr_key == "async_wait_queue_scope") {
          if (const auto* inner = op->body.as<AttrStmtNode>()) {
            if (inner->attr_key == tir::attr::async_wait_inflight_count ||
                inner->attr_key == "async_wait_inflight_count") {
              if (const auto* imm = inner->value.as<IntImmNode>()) {
                counts.push_back(static_cast<int>(imm->value));
              }
              // 【修复点 1】直接访问 inner->body 并 return，避免内层节点被二次统计
              VisitStmt(inner->body);
              return;
            }
          }
        } else if (op->attr_key == tir::attr::async_wait_inflight_count ||
                   op->attr_key == "async_wait_inflight_count") {
          if (const auto* imm = op->value.as<IntImmNode>()) {
            counts.push_back(static_cast<int>(imm->value));
          }
          // 【修复点 2】同理，直接访问 body 并 return
          VisitStmt(op->body);
          return;
        }
      }
      StmtVisitor::VisitStmt_(op);
    }
  };

  Collector collector;
  collector(root);
  return collector.counts;
}
// Original epilogue: BX1(4) CX1 BX2(2) CX2 BX3(0) CX3 CX4
// Target epilogue:  CX1 BX2(2) CX2 BX3(0) CX3 CX4
// Drop the first async_wait count and use the remainder as gld_fence values.
std::vector<int> BuildEpilogueGldFenceValues(const std::vector<int>& wait_counts) {
  std::vector<int> shifted;
  if (wait_counts.size() <= 1) {
    return shifted;
  }
  shifted.reserve(wait_counts.size() - 1);
  for (size_t i = 1; i < wait_counts.size(); ++i) {
    shifted.push_back(wait_counts[i]);
  }
  return shifted;
}

class InsertAsyncGldFenceMutator : public StmtMutator {
 public:
  explicit InsertAsyncGldFenceMutator(const Stmt& root)
      : has_main_loop_(!CollectDCUMainLoops(root).empty()),
        epilogue_gld_fence_values_(BuildEpilogueGldFenceValues(CollectEpilogueWaitCounts(root))) {}

  Stmt VisitStmt_(const ForNode* op) override {
    if (IsDCUMainLoop(op)) {
      phase_ = PipelinePhase::kInMainLoop;
      Stmt new_body = VisitStmt(op->body);
      phase_ = PipelinePhase::kAfterMainLoop;
      return For(op->loop_var, op->min, op->extent, op->kind, new_body, op->thread_binding,
                 op->annotations);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    const bool is_wait_scope = op->attr_key == tir::attr::async_wait_queue_scope ||
                               op->attr_key == "async_wait_queue_scope" ||
                               op->attr_key == tir::attr::async_wait_inflight_count ||
                               op->attr_key == "async_wait_inflight_count";
    // Fully unrolled K has no main loop; the first wait is still prologue and
    // must be kept. Later waits are converted like the main-loop body.
    if (!has_main_loop_ && is_wait_scope && phase_ == PipelinePhase::kBeforeMainLoop) {
      Stmt kept = StmtMutator::VisitStmt_(op);
      phase_ = PipelinePhase::kInMainLoop;
      return kept;
    }

    if (!ShouldTransform(phase_)) {
      return StmtMutator::VisitStmt_(op);
    }

    if (op->attr_key == tir::attr::async_wait_queue_scope ||
        op->attr_key == "async_wait_queue_scope") {
      const auto* inner = op->body.as<AttrStmtNode>();
      if (inner && (inner->attr_key == tir::attr::async_wait_inflight_count ||
                    inner->attr_key == "async_wait_inflight_count")) {
        if (const auto* imm = inner->value.as<IntImmNode>()) {
          if (phase_ == PipelinePhase::kInMainLoop) {
            pending_wait_counts_.push_back(static_cast<int>(imm->value));
          }
        }
        return VisitStmt(inner->body);
      }
      return VisitStmt(op->body);
    }
    if (op->attr_key == tir::attr::async_wait_inflight_count ||
        op->attr_key == "async_wait_inflight_count") {
      if (const auto* imm = op->value.as<IntImmNode>()) {
        if (phase_ == PipelinePhase::kInMainLoop) {
          pending_wait_counts_.push_back(static_cast<int>(imm->value));
        }
      }
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    std::deque<int> saved_pending = std::move(pending_wait_counts_);
    pending_wait_counts_.clear();

    const bool emit_prologue_pair_barriers =
        seq_nesting_ == 0 && phase_ == PipelinePhase::kBeforeMainLoop;
    ++seq_nesting_;

    Stmt flattened = SeqStmt::Flatten(GetRef<Stmt>(op));
    const auto* seq_op = flattened.as<SeqStmtNode>();
    if (seq_op == nullptr) {
      Stmt inner = VisitStmt(flattened);
      --seq_nesting_;
      pending_wait_counts_ = std::move(saved_pending);
      return inner;
    }

    Array<Stmt> result;
    result.reserve(seq_op->seq.size() + 8);
    int prologue_stage_commits = 0;
    auto append_leaf = [&](const Stmt& leaf) {
      result.push_back(leaf);
      if (!emit_prologue_pair_barriers || !IsPrologueCommitStmt(leaf)) {
        return;
      }
      prologue_stage_commits += 1;
      // Reference gemm.log: wave_barrier after each A+B pair.
      if (prologue_stage_commits >= 2) {
        if (result.empty() || !IsWaveBarrierStmt(result.back())) {
          result.push_back(MakeWaveBarrierStmt());
        }
        prologue_stage_commits = 0;
      }
    };
    auto append_stmt = [&](const Stmt& stmt) {
      if (const auto* inner = stmt.as<SeqStmtNode>()) {
        for (const Stmt& child : inner->seq) {
          append_leaf(child);
        }
      } else {
        append_leaf(stmt);
      }
    };
    for (const Stmt& s : seq_op->seq) {
      append_stmt(VisitStmt(s));
      TryInsertEpilogueGldFenceAfterSldFence(result);
      TryInsertGldFenceAfterPair(result);
    }

    --seq_nesting_;

    for (int count : pending_wait_counts_) {
      saved_pending.push_back(count);
    }
    pending_wait_counts_ = std::move(saved_pending);

    if (result.empty()) {
      return Evaluate(0);
    }
    if (result.size() == 1) {
      return result[0];
    }
    return SeqStmt(result);
  }

 private:
  void TryInsertGldFenceAfterPair(Array<Stmt>& result) {
    if (phase_ != PipelinePhase::kInMainLoop || pending_wait_counts_.empty()) {
      return;
    }
    size_t n = result.size();
    if (n < 2 || !IsAsyncGldSldFenceStmt(result[n - 2]) || !IsWaveBarrierStmt(result[n - 1])) {
      return;
    }

    // CX4 uses async_gld_sld_fence(0) and has no async_gld_fence partner.
    if (GetAsyncGldSldFenceValue(result[n - 2]) == 0) {
      if (!pending_wait_counts_.empty()) {
        pending_wait_counts_.pop_front();
      }
      return;
    }

    int wait_count = pending_wait_counts_.front();
    pending_wait_counts_.pop_front();
    Stmt barrier = result[n - 1];
    result.pop_back();
    result.push_back(MakeAsyncGldFenceStmt(wait_count));
    result.push_back(barrier);
  }

  void TryInsertEpilogueGldFenceAfterSldFence(Array<Stmt>& result) {
    if (phase_ != PipelinePhase::kAfterMainLoop) {
      return;
    }
    size_t n = result.size();
    if (n < 1 || !IsAsyncGldSldFenceStmt(result[n - 1])) {
      return;
    }
    // CX4 uses async_gld_sld_fence(0) and has no async_gld_fence partner.
    if (GetAsyncGldSldFenceValue(result[n - 1]) == 0) {
      return;
    }
    if (epilogue_gld_insert_idx_ >= static_cast<int>(epilogue_gld_fence_values_.size())) {
      return;
    }

    int wait_count = epilogue_gld_fence_values_[epilogue_gld_insert_idx_++];
    result.push_back(MakeAsyncGldFenceStmt(wait_count));
    result.push_back(MakeWaveBarrierStmt());
  }

  PipelinePhase phase_{PipelinePhase::kBeforeMainLoop};
  bool has_main_loop_{true};
  int seq_nesting_{0};
  std::deque<int> pending_wait_counts_;
  const std::vector<int> epilogue_gld_fence_values_;
  int epilogue_gld_insert_idx_{0};
};

}  // namespace

namespace transform {
using namespace tir::transform;

Pass InsertAsyncGldFence() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    InsertAsyncGldFenceMutator mutator(n->body);
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
