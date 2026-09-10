#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>

#include "inject_utils.h"

using namespace tvm::tir;
using tvm::ffi::GetRef;

namespace tvm {
namespace tl {
using namespace tir;

namespace {

bool IsWaitAttr(const AttrStmtNode* op) {
  return op->attr_key == tir::attr::async_wait_queue_scope ||
         op->attr_key == "async_wait_queue_scope" ||
         op->attr_key == tir::attr::async_wait_inflight_count ||
         op->attr_key == "async_wait_inflight_count";
}

bool IsCommitAttr(const AttrStmtNode* op) {
  return op->attr_key == tir::attr::async_commit_queue_scope ||
         op->attr_key == "async_commit_queue_scope";
}

bool IsPtxCommitStmt(const Stmt& stmt) {
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      return call->op.same_as(builtin::ptx_commit_group());
    }
  }
  return false;
}

Stmt UnwrapWaitAttrs(Stmt stmt) {
  while (true) {
    const auto* attr = stmt.as<AttrStmtNode>();
    if (attr == nullptr || !IsWaitAttr(attr)) {
      break;
    }
    stmt = attr->body;
  }
  return stmt;
}

bool IsOutermostCommitStmt(const Stmt& stmt) {
  if (IsPtxCommitStmt(stmt)) {
    return true;
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    return IsCommitAttr(attr);
  }
  return false;
}

/*!
 * \brief Count outermost async commit groups in a statement (not individual
 *        cp_async / dcu_async_copy instructions).
 */
int CountOutermostCommits(const Stmt& stmt) {
  struct Scanner : public StmtVisitor {
    int commits{0};
    int commit_depth{0};

    void NoteCommit() {
      if (commit_depth == 0) {
        commits += 1;
      }
    }

    void VisitStmt_(const AttrStmtNode* op) override {
      if (IsWaitAttr(op)) {
        VisitStmt(UnwrapWaitAttrs(GetRef<AttrStmt>(op)));
        return;
      }
      if (IsCommitAttr(op)) {
        NoteCommit();
        ++commit_depth;
        VisitStmt(op->body);
        --commit_depth;
        return;
      }
      StmtVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const SeqStmtNode* op) override {
      Stmt flattened = SeqStmt::Flatten(GetRef<Stmt>(op));
      const auto* seq = flattened.as<SeqStmtNode>();
      if (seq == nullptr) {
        VisitStmt(flattened);
        return;
      }
      for (const Stmt& s : seq->seq) {
        Stmt cur = UnwrapWaitAttrs(s);
        if (IsOutermostCommitStmt(cur)) {
          NoteCommit();
          ++commit_depth;
          VisitStmt(cur);
          --commit_depth;
          continue;
        }
        VisitStmt(cur);
      }
    }

    void VisitStmt_(const EvaluateNode* op) override {
      if (IsPtxCommitStmt(GetRef<Evaluate>(op))) {
        NoteCommit();
      }
    }
  };
  Scanner scanner;
  scanner(stmt);
  return scanner.commits;
}

class GlobalMaxCommitFinder : public StmtVisitor {
 public:
  static int64_t FindMax(const Stmt& stmt) {
    GlobalMaxCommitFinder finder;
    finder.VisitStmt(stmt);
    return std::max(static_cast<int64_t>(1), finder.max_commits_);
  }

 private:
  void VisitStmt_(const ForNode* op) override {
    if (IsDCUMainLoop(op)) {
      int64_t cpt = CountOutermostCommits(op->body);
      if (cpt > max_commits_) {
        max_commits_ = cpt;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  int64_t max_commits_ = 0;
};

class ROCmWaitCountRewriter : public StmtMutator {
 public:
  static Stmt Substitute(const Stmt& stmt) {
    int64_t max_commits = GlobalMaxCommitFinder::FindMax(stmt);
    ROCmWaitCountRewriter rewriter(max_commits);
    return rewriter(stmt);
  }

 private:
  explicit ROCmWaitCountRewriter(int64_t mult) : commits_per_tile_(mult) {}

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == tir::attr::async_wait_inflight_count ||
        op->attr_key == "async_wait_inflight_count") {
      if (auto int_imm = op->value.as<IntImmNode>()) {
        int64_t new_val = int_imm->value * commits_per_tile_;
        return AttrStmt(op->node, op->attr_key, make_const(DataType::Int(32), new_val),
                        this->VisitStmt(op->body));
      }
    }
    return StmtMutator::VisitStmt_(op);
  }
  int64_t commits_per_tile_;
};

}  // namespace

namespace transform {
using namespace tir::transform;
tvm::transform::Pass FixDCUWaitCount() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ROCmWaitCountRewriter::Substitute(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "FixDCUWaitCount", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.FixDCUWaitCount", FixDCUWaitCount);
}
}  // namespace transform

}  // namespace tl
}  // namespace tvm
