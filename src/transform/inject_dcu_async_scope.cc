/*!
 * \file inject_dcu_async_scope.cc
 * \brief Synchronize global→shared copies for DCU/HIP.
 *
 *        1. Wrap *vectorized* 16B g2s BufferStores in tir::attr::async_scope so
 *           LowerSharedGlobalCopy can rewrite them to tl.dcu_async_copy
 *           (codegen: tl::cp_async_gs) + ptx_commit_group.
 *
 *        2. After every consecutive g2s cluster — whether lowered to cp_async
 *           or left as scalar predicated stores — emit
 *             ptx_commit_group + tl.cp_async_wait<0> + tl.wave_barrier
 *           so later shared→reg / mmac sees the data across waves.
 *
 *           Scalar g2s still needs wait/barrier: LDS writes are not
 *           cross-wave visible without s_barrier, and vmcnt(0) drains any
 *           outstanding global loads. Commit is a HIP no-op but keeps the
 *           same commit/wait pairing as the async path.
 *
 *        When a For already carries num_stages, software pipeline owns those
 *        in-loop copies; this pass skips them. Prologue copies outside such
 *        loops are still handled.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>
#include <vector>

#include "inject_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::Array;

namespace {

bool IsSharedScope(const ffi::String &scope) {
  std::string s = scope;
  return s == "shared" || s == "shared.dyn";
}

bool IsGlobalScope(const Buffer &buf) {
  auto s = buf.scope();
  return s == "global" || s == "";
}

const BufferLoadNode *PeelBufferLoad(const PrimExpr &value) {
  if (const auto *load = value.as<BufferLoadNode>()) {
    return load;
  }
  if (const auto *cast = value.as<CastNode>()) {
    return PeelBufferLoad(cast->value);
  }
  if (const auto *call = value.as<CallNode>()) {
    if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3) {
      return PeelBufferLoad(call->args[1]);
    }
  }
  return nullptr;
}

bool IsGlobalToSharedStore(const BufferStoreNode *op) {
  if (!IsSharedScope(op->buffer.scope())) {
    return false;
  }
  const BufferLoadNode *load = PeelBufferLoad(op->value);
  return load != nullptr && IsGlobalScope(load->buffer);
}

bool IsAsyncEligibleG2S(const BufferStoreNode *op) {
  const BufferLoadNode *load = PeelBufferLoad(op->value);
  return load != nullptr && IsDCUCpAsyncVectorCopy(op, load);
}

/*!
 * \brief Collect all global→shared BufferStores (scalar or vector).
 *        Also record which of them are eligible for cp_async lowering.
 */
class GlobalToSharedStoreCollector : public StmtExprVisitor {
 public:
  std::vector<const BufferStoreNode *> stores;
  std::vector<const BufferStoreNode *> async_stores;

 private:
  int async_scope_depth_{0};

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::async_scope || op->attr_key == "async_scope") {
      ++async_scope_depth_;
      StmtExprVisitor::VisitStmt_(op);
      --async_scope_depth_;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->annotations.count("num_stages")) {
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (async_scope_depth_ == 0 && IsGlobalToSharedStore(op)) {
      stores.push_back(op);
      if (IsAsyncEligibleG2S(op)) {
        async_stores.push_back(op);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

class NeedsInjectionScanner : public StmtVisitor {
 public:
  bool needs_injection{false};

 private:
  int async_scope_depth_{0};

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::async_scope || op->attr_key == "async_scope") {
      ++async_scope_depth_;
      StmtVisitor::VisitStmt_(op);
      --async_scope_depth_;
      return;
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->annotations.count("num_stages")) {
      return;
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (async_scope_depth_ == 0 && IsGlobalToSharedStore(op)) {
      needs_injection = true;
    }
    StmtVisitor::VisitStmt_(op);
  }
};

Stmt MakeCpAsyncCommitStmt() {
  return Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
}

Stmt MakeCpAsyncWaitStmt(int wait_count = 0) {
  auto wait_call =
      Call(DataType::Void(), Op::Get("tl.cp_async_wait"), {Integer(wait_count)});
  return Evaluate(wait_call);
}

Stmt MakeWaveBarrierStmt() {
  auto barrier_call = Call(DataType::Void(), Op::Get("tl.wave_barrier"), {});
  return Evaluate(barrier_call);
}

/*!
 * \brief Wrap async-eligible stores in async_scope. After every g2s cluster
 *        (async or scalar), emit commit (scalar only) + wait<0> + wave_barrier.
 */
class AsyncScopeInjector : public StmtExprMutator {
 public:
  explicit AsyncScopeInjector(
      std::unordered_set<const BufferStoreNode *> async_targets)
      : async_targets_(std::move(async_targets)) {}

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Stmt new_stmt = StmtExprMutator::VisitStmt_(op);
    if (async_targets_.count(op)) {
      return AttrStmt(ObjectRef(), tir::attr::async_scope, Integer(0), new_stmt);
    }
    return new_stmt;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> visited;
    visited.reserve(op->seq.size());
    for (const Stmt &child : op->seq) {
      visited.push_back(VisitStmt(child));
    }
    Stmt flattened = SeqStmt::Flatten(visited);
    const auto *seq = flattened.as<SeqStmtNode>();
    if (seq == nullptr) {
      return flattened;
    }

    bool all_g2s = !seq->seq.empty();
    for (const Stmt &s : seq->seq) {
      if (!IsG2SCluster(s)) {
        all_g2s = false;
        break;
      }
    }
    // Pure g2s cluster: parent SeqStmt emits the single commit/wait/barrier.
    if (all_g2s) {
      return flattened;
    }

    Array<Stmt> out;
    bool in_g2s = false;
    bool cluster_is_async = false;
    auto flush_sync = [&]() {
      if (!in_g2s) {
        return;
      }
      // Async g2s: LowerSharedGlobalCopy already emits ptx_commit_group per
      // cp_async. Scalar g2s has no Lower rewrite, so emit commit here.
      // Both paths need wait<0> + wave_barrier for LDS cross-wave visibility.
      if (!cluster_is_async) {
        out.push_back(MakeCpAsyncCommitStmt());
      }
      out.push_back(MakeCpAsyncWaitStmt(0));
      out.push_back(MakeWaveBarrierStmt());
      in_g2s = false;
      cluster_is_async = false;
    };
    for (const Stmt &s : seq->seq) {
      if (IsG2SCluster(s)) {
        out.push_back(s);
        in_g2s = true;
        if (ContainsAsyncScope(s)) {
          cluster_is_async = true;
        }
      } else {
        flush_sync();
        out.push_back(s);
      }
    }
    flush_sync();
    return SeqStmt::Flatten(out);
  }

 private:
  static bool ContainsAsyncScope(const Stmt &stmt) {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == tir::attr::async_scope ||
          attr->attr_key == "async_scope") {
        return true;
      }
      return ContainsAsyncScope(attr->body);
    }
    if (const auto *for_op = stmt.as<ForNode>()) {
      return ContainsAsyncScope(for_op->body);
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return ContainsAsyncScope(let->body);
    }
    if (const auto *iff = stmt.as<IfThenElseNode>()) {
      if (ContainsAsyncScope(iff->then_case)) {
        return true;
      }
      return iff->else_case.defined() &&
             ContainsAsyncScope(iff->else_case.value());
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (ContainsAsyncScope(s)) {
          return true;
        }
      }
    }
    return false;
  }

  // Identify g2s by shape, not by pre-collected pointers: StmtExprMutator may
  // rebuild BufferStore nodes so pointer identity is unreliable after Visit.
  static bool IsG2SCluster(const Stmt &stmt) {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == tir::attr::async_scope ||
          attr->attr_key == "async_scope") {
        return true;
      }
      return IsG2SCluster(attr->body);
    }
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      return IsGlobalToSharedStore(store);
    }
    if (const auto *for_op = stmt.as<ForNode>()) {
      return IsG2SCluster(for_op->body);
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return IsG2SCluster(let->body);
    }
    if (const auto *iff = stmt.as<IfThenElseNode>()) {
      if (!IsG2SCluster(iff->then_case)) {
        return false;
      }
      if (!iff->else_case.defined()) {
        return true;
      }
      return IsG2SCluster(iff->else_case.value());
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return false;
      }
      for (const Stmt &s : seq->seq) {
        if (!IsG2SCluster(s)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  std::unordered_set<const BufferStoreNode *> async_targets_;
};

PrimFunc InjectDCUAsyncScopeToFunc(PrimFunc func) {
  NeedsInjectionScanner scanner;
  scanner(func->body);
  if (!scanner.needs_injection) {
    return func;
  }

  GlobalToSharedStoreCollector collector;
  collector(func->body);
  std::unordered_set<const BufferStoreNode *> async_targets(
      collector.async_stores.begin(), collector.async_stores.end());

  AsyncScopeInjector mutator(std::move(async_targets));
  auto *n = func.CopyOnWrite();
  n->body = mutator(std::move(n->body));
  return func;
}

}  // namespace

namespace transform {

using namespace tir::transform;

tvm::transform::Pass InjectDCUAsyncScope() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    if (!IsDCUTarget(m)) {
      return f;
    }
    return InjectDCUAsyncScopeToFunc(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectDCUAsyncScope", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectDCUAsyncScope", InjectDCUAsyncScope);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
