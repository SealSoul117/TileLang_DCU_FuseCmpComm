/*!
 * \file inject_dcu_async_scope.cc
 * \brief Wrap non-pipelined global→shared BufferStores in tir::attr::async_scope
 *        AttrStmt so that LowerSharedGlobalCopy can rewrite them into
 *        tl.dcu_async_copy calls (which codegen lowers to tl::cp_async_gs<...>).
 *
 *        This pass enables async g→s copies when the user writes a plain
 *        ``for k in range(...)`` body without a software pipeline annotation
 *        (i.e. ``num_stages == 0`` in TileLang's Pipelined). When a For loop
 *        already carries a ``num_stages`` annotation, pipeline_planning /
 *        InjectSoftwarePipeline / LowerSharedGlobalCopy already produce the
 *        necessary async_scope wrapping, so this pass is a no-op for those
 *        loops. Functions that already contain an ``attr::async_scope``
 *        AttrStmt (e.g. coming from a previous pipeline injection) are also
 *        skipped to avoid double wrapping.
 *
 *        When num_stages == 0 the InjectSoftwarePipeline pass is bypassed,
 *        so the pipeline-style async_commit / async_wait AttrStmts that drive
 *        cp_async_wait<N> emission are never produced. To keep semantics
 *        correct, after every g2s copy we also emit an explicit
 *        tl.cp_async_wait<0>() + tl.wave_barrier() pair so the subsequent
 *        s2r reads from shared memory observe the loaded data. Mirrors the
 *        MakeWaveBarrierStmt / MakeAsyncGldFenceStmt helpers used in
 *        insert_async_gld_fence.cc.
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
    return cast->value.as<BufferLoadNode>();
  }
  return nullptr;
}

/*!
 * \brief Collect BufferStore nodes that copy from a global Buffer into a
 *        shared/shared.dyn Buffer. Only direct BufferLoad / Cast<BufferLoad>
 *        values are matched (mirrors the predicate inside
 *        LowerSharedGlobalCopy::CollectResources).
 */
class GlobalToSharedStoreCollector : public StmtExprVisitor {
 public:
  std::vector<const BufferStoreNode *> stores;

 private:
  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsSharedScope(op->buffer.scope())) {
      const BufferLoadNode *load = PeelBufferLoad(op->value);
      if (load != nullptr && IsGlobalScope(load->buffer)) {
        stores.push_back(op);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

/*!
 * \brief Cheap outer scan: returns true iff the function body contains at
 *        least one For loop without a num_stages annotation that has a
 *        global→shared BufferStore in its body. If the function already has
 *        an attr::async_scope AttrStmt, we skip it to avoid double wrapping
 *        (pipelines already take care of it).
 */
class NeedsInjectionScanner : public StmtVisitor {
 public:
  bool needs_injection{false};
  bool already_async{false};

 private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::async_scope) {
      already_async = true;
      return;
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    if (!op->annotations.count("num_stages")) {
      GlobalToSharedStoreCollector inner;
      inner(op->body);
      if (!inner.stores.empty()) {
        needs_injection = true;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
};

/*!
 * \brief Build a tl.cp_async_wait<N> Evaluate statement.
 *        Mirrors MakeAsyncGldFenceStmt in insert_async_gld_fence.cc.
 */
Stmt MakeCpAsyncWaitStmt(int wait_count = 0) {
  auto wait_call =
      Call(DataType::Void(), Op::Get("tl.cp_async_wait"), {Integer(wait_count)});
  return Evaluate(wait_call);
}

/*!
 * \brief Build a tl.wave_barrier Evaluate statement.
 *        Mirrors MakeWaveBarrierStmt in insert_async_gld_fence.cc.
 */
Stmt MakeWaveBarrierStmt() {
  auto barrier_call = Call(DataType::Void(), Op::Get("tl.wave_barrier"), {});
  return Evaluate(barrier_call);
}

/*!
 * \brief Wrap each collected BufferStore in its own AttrStmt with attr_key =
 *        tir::attr::async_scope. The downstream LowerSharedGlobalCopy pass
 *        recognizes this AttrStmt and consumes it (strips it and replaces the
 *        BufferStore with a tl.dcu_async_copy call). Wrapping each store
 *        individually is sufficient because LowerSharedGlobalCopy's
 *        CollectResources is gated by in_async and visits children of the
 *        AttrStmt before stripping it.
 *
 *        In addition, emit an explicit tl.cp_async_wait<0>() + tl.wave_barrier()
 *        pair immediately after every async copy. The HIP codegen already
 *        lowers tl.cp_async_wait and tl.wave_barrier to the matching
 *        tl::cp_async_wait<N>() / tl::wave_barrier() calls (see
 *        src/target/codegen_hip.cc).
 */
class AsyncScopeInjector : public StmtExprMutator {
 public:
  explicit AsyncScopeInjector(std::unordered_set<const BufferStoreNode *> targets)
      : targets_(std::move(targets)) {}

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Stmt new_stmt = StmtExprMutator::VisitStmt_(op);
    if (targets_.count(op)) {
      Stmt async_store =
          AttrStmt(ObjectRef(), tir::attr::async_scope, Integer(0), new_stmt);
      return SeqStmt({async_store, MakeCpAsyncWaitStmt(0), MakeWaveBarrierStmt()});
    }
    return new_stmt;
  }

 private:
  std::unordered_set<const BufferStoreNode *> targets_;
};

PrimFunc InjectDCUAsyncScopeToFunc(PrimFunc func) {
  NeedsInjectionScanner scanner;
  scanner(func->body);
  if (!scanner.needs_injection || scanner.already_async) {
    return func;
  }

  GlobalToSharedStoreCollector collector;
  collector(func->body);
  std::unordered_set<const BufferStoreNode *> targets(collector.stores.begin(),
                                                     collector.stores.end());

  AsyncScopeInjector mutator(std::move(targets));
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