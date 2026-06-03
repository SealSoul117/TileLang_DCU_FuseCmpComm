/*!
 * \file insert_async_global2shared_fence.cc
 * \brief Move async global-to-shared wait from AttrStmt to tl.cp_async_wait after
 *        tl.async_gld_sld_fence inside the main k-loop.
 */
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>

#include <optional>
#include <string>

using tvm::ffi::Array;
namespace tvm {
namespace tl {
namespace {

using namespace tir;

bool IsAsyncGldFenceCall(const CallNode* call) {
  if (!call) return false;
  if (const auto* op = call->op.as<OpNode>()) {
    return op->name == "tl.async_gld_sld_fence";
  }
  return false;
}

Stmt MakeCpAsyncWaitStmt(int wait_count) {
  auto wait_call =
      Call(DataType::Void(), Op::Get("tl.cp_async_wait"), {Integer(wait_count)});
  return Evaluate(wait_call);
}

// Rewrites the body of the main k-loop: strips async_wait AttrStmts and inserts
// tl.cp_async_wait after each tl.async_gld_sld_fence.
class KLoopBodyMutator : public StmtMutator {
 public:
  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == tir::attr::async_wait_queue_scope ||
        op->attr_key == "async_wait_queue_scope") {
      const auto* inner = op->body.as<AttrStmtNode>();
      if (inner && (inner->attr_key == tir::attr::async_wait_inflight_count ||
                    inner->attr_key == "async_wait_inflight_count")) {
        if (const auto* imm = inner->value.as<IntImmNode>()) {
          wait_count_ = static_cast<int>(imm->value);
        }
        return VisitStmt(inner->body);
      }
      return VisitStmt(op->body);
    }
    if (op->attr_key == tir::attr::async_wait_inflight_count ||
        op->attr_key == "async_wait_inflight_count") {
      if (const auto* imm = op->value.as<IntImmNode>()) {
        wait_count_ = static_cast<int>(imm->value);
      }
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode* op) override {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto* eval = stmt.as<EvaluateNode>();
    if (!eval) return stmt;
    const auto* call = eval->value.as<CallNode>();
    if (!IsAsyncGldFenceCall(call)) return stmt;
    if (!wait_count_.has_value()) return stmt;
    Array<Stmt> seq = {stmt, MakeCpAsyncWaitStmt(wait_count_.value())};
    return SeqStmt(seq);
  }

 private:
  std::optional<int> wait_count_;
};

class InsertAsyncGlobal2SharedFenceMutator : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    const auto* var = op->loop_var.as<VarNode>();
    if (var && var->name_hint == "k") {
      KLoopBodyMutator body_mutator;
      Stmt new_body = body_mutator(op->body);
      return For(op->loop_var, op->min, op->extent, op->kind, new_body, op->thread_binding,
                 op->annotations);
    }
    return StmtMutator::VisitStmt_(op);
  }
};

}  // namespace

namespace transform {
using namespace tir::transform;

Pass InsertAsyncGlobal2SharedFence() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    InsertAsyncGlobal2SharedFenceMutator mutator;
    n->body = mutator(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InsertAsyncGlobal2SharedFence", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InsertAsyncGlobal2SharedFence",
                        InsertAsyncGlobal2SharedFence);
}
}  // namespace transform
}  // namespace tl
}  // namespace tvm
