/*!
 * \file hoist_dcu_ipc_pointers.cc
 * \brief Conservative loop-local hoisting for M2 IPC remote pointers.
 *
 * `tl.ipc.remote_address` is deliberately registered opaque: its generated
 * helper performs runtime DeviceView validation and must not participate in a
 * general-purpose LICM pass.  This pass therefore recognizes only the narrow
 * normal form produced by LowerDCUIPC and moves direct, unconditional Let
 * bindings one lexical level, from a serial element loop to immediately
 * around that loop.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <sstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace {

using namespace tir;
using tvm::ffi::Array;
using tvm::ffi::GetRef;
using tvm::ffi::String;

constexpr const char* kIPCBindingsAttr = "tl.dcu_ipc.bindings";
constexpr const char* kIPCABIReadyAttr = "tl.dcu_ipc.abi_ready";
constexpr const char* kIPCRemoteAddress = "tl.ipc.remote_address";

std::string OpName(const CallNode* call) {
  if (const auto* op = call->op.as<OpNode>()) return op->name;
  return "";
}

bool IsIPCRemoteAddress(const CallNode* call) {
  return call != nullptr && OpName(call) == kIPCRemoteAddress && call->args.size() == 4;
}

int IPCDescriptorArg(const CallNode* call) {
  if (call == nullptr) return -1;
  const std::string name = OpName(call);
  if (name == kIPCRemoteAddress) return call->args.size() == 4 ? 0 : -1;
  if (name == "tl.ipc.signal_set") return call->args.size() == 6 ? 3 : -1;
  if (name == "tl.ipc.wait") return call->args.size() == 9 ? 6 : -1;
  if (name == "tl.ipc.put_block" || name == "tl.ipc.get_block") {
    return call->args.size() == 7 ? 4 : -1;
  }
  if (name == "tl.ipc.st" || name == "tl.ipc.ld") return call->args.size() == 8 ? 5 : -1;
  return -1;
}

bool HasIntrinsicOpaqueEffect(const CallNode* call) {
  if (const auto op = call->op.as<Op>()) {
    static auto call_effect = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    if (!call_effect.count(op.value())) return true;
    return static_cast<CallEffectKind>(call_effect[op.value()]->value) > CallEffectKind::kPure;
  }
  // call_extern and all non-Op callees are unknown by construction.
  return true;
}

bool IsScalarParameter(const Var& var) {
  return var.dtype().is_int() || var.dtype().is_uint();
}

struct IPCABI {
  std::unordered_set<const VarNode*> descriptors;
  std::unordered_set<const VarNode*> scalar_parameters;
  Var rank;
  Var world;
};

std::optional<IPCABI> GetVerifiedIPCABI(const PrimFunc& func) {
  // This pass is a no-op for ordinary functions.  Do not infer an IPC ABI
  // from names alone: the ready attribute is set only by bind_ipc's prepared
  // function, and every descriptor must be both a parameter and its exact
  // buffer data Var.
  if (!func->HasNonzeroAttr(kIPCABIReadyAttr)) return std::nullopt;
  auto bindings_attr = func->GetAttr<String>(kIPCBindingsAttr);
  if (!bindings_attr.has_value()) return std::nullopt;

  std::unordered_set<std::string> descriptor_names;
  std::stringstream records(static_cast<std::string>(bindings_attr.value()));
  std::string record;
  while (std::getline(records, record, ';')) {
    const size_t split = record.find('=');
    if (split == std::string::npos || split == 0 || split + 1 == record.size()) return std::nullopt;
    descriptor_names.insert(record.substr(split + 1));
  }
  if (descriptor_names.empty()) return std::nullopt;

  IPCABI result;
  std::unordered_map<std::string, Var> params_by_name;
  for (const Var& param : func->params) {
    params_by_name.emplace(param->name_hint, param);
    if (IsScalarParameter(param)) result.scalar_parameters.insert(param.get());
    if (param->name_hint == "__tl_ipc_rank") result.rank = param;
    if (param->name_hint == "__tl_ipc_world_size") result.world = param;
  }
  if (!result.rank.defined() || !result.world.defined()) return std::nullopt;

  for (const std::string& name : descriptor_names) {
    auto param_it = params_by_name.find(name);
    if (param_it == params_by_name.end()) return std::nullopt;
    bool exact_buffer_data = false;
    for (const auto& item : func->buffer_map) {
      if (item.first.same_as(param_it->second) && item.second->data.same_as(param_it->second)) {
        exact_buffer_data = true;
        break;
      }
    }
    if (!exact_buffer_data) return std::nullopt;
    result.descriptors.insert(param_it->second.get());
  }
  return result;
}

// A descriptor is immutable for one launch: allocation.cpp publishes its peer
// table only after construction, and runtime_bridge uploads/caches the six-word
// DeviceView until that allocation is freed.  That runtime contract is narrow;
// reject a function which writes, aliases, loads, or otherwise escapes a
// descriptor in TIR instead of treating an arbitrary writable pointer as
// immutable.
class DescriptorUseAudit final : public StmtExprVisitor {
 public:
  explicit DescriptorUseAudit(const std::unordered_set<const VarNode*>& descriptors)
      : descriptors_(descriptors) {}

  bool safe() const { return safe_; }

 private:
  void VisitExpr_(const VarNode* op) final {
    if (descriptors_.count(op)) safe_ = false;
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    if (descriptors_.count(op->buffer->data.get())) safe_ = false;
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    if (descriptors_.count(op->buffer->data.get())) safe_ = false;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    const int descriptor_arg = IPCDescriptorArg(op);
    if (descriptor_arg >= 0) {
      const auto* descriptor = op->args[descriptor_arg].as<VarNode>();
      if (!descriptor || !descriptors_.count(descriptor)) {
        safe_ = false;
        return;
      }
      // Every M2 ABI op has one fixed, read-only DeviceView operand.  Audit
      // every other operand normally, so descriptor use as payload, peer,
      // rank, world, or an arbitrary call argument is still an escape.
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (static_cast<int>(i) != descriptor_arg) VisitExpr(op->args[i]);
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  const std::unordered_set<const VarNode*>& descriptors_;
  bool safe_{true};
};

// Checks a peer expression without assigning any purity to remote_address.
// Scalar launch parameters and lexically enclosing pure Lets are allowed; a
// BufferLoad, call, pointer parameter, or current element-loop variable is not.
class PureInvariantExpr final : public ExprVisitor {
 public:
  PureInvariantExpr(const std::unordered_map<const VarNode*, PrimExpr>& let_defs,
                    const std::unordered_set<const VarNode*>& scalar_parameters,
                    const std::unordered_set<const VarNode*>& outer_invariant_vars,
                    const VarNode* loop_var)
      : let_defs_(let_defs), scalar_parameters_(scalar_parameters),
        outer_invariant_vars_(outer_invariant_vars), loop_var_(loop_var) {}

  bool Check(const PrimExpr& expr) {
    VisitExpr(expr);
    return safe_;
  }

 private:
  void VisitExpr_(const VarNode* op) final {
    if (op == loop_var_) {
      safe_ = false;
      return;
    }
    if (outer_invariant_vars_.count(op)) return;
    auto it = let_defs_.find(op);
    if (it != let_defs_.end()) {
      if (!visiting_.insert(op).second) {
        safe_ = false;
        return;
      }
      VisitExpr(it->second);
      visiting_.erase(op);
      return;
    }
    if (!scalar_parameters_.count(op)) safe_ = false;
  }

  void VisitExpr_(const BufferLoadNode* op) final { safe_ = false; }

  void VisitExpr_(const CallNode* op) final { safe_ = false; }

  const std::unordered_map<const VarNode*, PrimExpr>& let_defs_;
  const std::unordered_set<const VarNode*>& scalar_parameters_;
  const std::unordered_set<const VarNode*>& outer_invariant_vars_;
  const VarNode* loop_var_;
  std::unordered_set<const VarNode*> visiting_;
  bool safe_{true};
};

// Remote payload BufferLoad/Store and a payload-only If are intentionally
// allowed.  Extraction never enters an If, so it cannot speculate a pointer
// binding whose original execution was conditional.  The candidate address
// Lets are the only opaque calls allowed in the serial element loop.  A small
// statically-bounded unrolled/vectorized payload loop is also allowed so that
// an eight-lane reduction between AR loads and stores does not prevent the
// surrounding, unconditional address Lets from moving.  Candidates are never
// extracted from that nested loop: synchronization, unknown calls, serial or
// other nested loops, allocation, or conditional pointer producers still stop
// hoisting at this lexical scope.  BufferLoad is not globally rejected here:
// it is retained in place and only the verified address producer moves.
class ElementLoopEffectAudit final : public StmtExprVisitor {
 public:
  bool safe() const { return safe_; }

 private:
  bool IsSmallStaticPayloadLoop(const ForNode* op) const {
    const auto* min = op->min.as<IntImmNode>();
    const auto* extent = op->extent.as<IntImmNode>();
    constexpr int64_t kMaxStaticPayloadLanes = 16;
    return (op->kind == ForKind::kUnrolled || op->kind == ForKind::kVectorized) &&
           !op->thread_binding.defined() && op->annotations.empty() && min && extent &&
           min->value >= 0 && extent->value >= 0 && extent->value <= kMaxStaticPayloadLanes;
  }

  bool IsBooleanOrBitLiteral(const PrimExpr& value) const {
    const auto* literal = value.as<IntImmNode>();
    return literal != nullptr && (literal->value == 0 || literal->value == 1);
  }

  void VisitExpr_(const CallNode* op) final {
    if (IsIPCRemoteAddress(op)) {
      // A nested payload loop may contain arithmetic, but not a candidate
      // address producer.  The extractor only considers this serial loop's
      // direct, unconditional SeqStmt children.
      if (payload_loop_depth_ != 0 || conditional_depth_ != 0) {
        safe_ = false;
        return;
      }
      // The producer itself is the narrowly-authorized opaque operation, but
      // its operands still must not conceal a fence, call_extern, load, or
      // other effect.  In particular, a later non-candidate remote address
      // must not let an earlier candidate cross an unknown peer computation.
      for (const PrimExpr& arg : op->args) VisitExpr(arg);
      return;
    }
    // Do not use SideEffect(op) here: a pure if_then_else which contains a
    // payload BufferLoad is classified ReadState by its child.  The pointer
    // itself does not cross that load; recurse into children and reject only
    // an intrinsically effectful/unknown call such as fence, barrier, wait,
    // signal, or call_extern.
    if (HasIntrinsicOpaqueEffect(op)) {
      safe_ = false;
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const WhileNode* op) final { safe_ = false; }
  void VisitStmt_(const IfThenElseNode* op) final {
    ++conditional_depth_;
    // Conditions execute before either branch.  They must be audited in the
    // same conditional scope: an opaque call or remote-address helper hidden
    // here would otherwise let a later top-level pointer cross an effect.
    VisitExpr(op->condition);
    VisitStmt(op->then_case);
    if (op->else_case.defined()) VisitStmt(op->else_case.value());
    --conditional_depth_;
  }
  void VisitStmt_(const ForNode* op) final {
    // The AR reduction is an eight-lane payload loop.  Keep this allowance
    // deliberately small and literal-only: it is not a second serial scope,
    // and no loop-bound expression is re-evaluated by pointer hoisting.
    if (!IsSmallStaticPayloadLoop(op)) {
      safe_ = false;
      return;
    }
    ++payload_loop_depth_;
    VisitStmt(op->body);
    --payload_loop_depth_;
  }
  void VisitStmt_(const AllocateNode* op) final { safe_ = false; }
  void VisitStmt_(const AllocateConstNode* op) final { safe_ = false; }
  void VisitStmt_(const AttrStmtNode* op) final {
    // StorageRewrite emits this exact wrapper around the AR eight-lane
    // reduction.  It is a loop-local codegen hint, not a control or memory
    // effect.  Accept only a literal boolean/bit value targeting this very
    // small unrolled loop; do not traverse arbitrary AttrStmt bodies or use
    // the wrapper as an extraction source.
    const auto* loop = op->body.as<ForNode>();
    const auto* node = op->node.as<VarNode>();
    if (op->attr_key != tir::attr::pragma_unroll_explicit || !loop || !node ||
        node != loop->loop_var.get() || !IsBooleanOrBitLiteral(op->value) ||
        !IsSmallStaticPayloadLoop(loop)) {
      safe_ = false;
      return;
    }
    VisitStmt(op->body);
  }
  void VisitStmt_(const AssertStmtNode* op) final { safe_ = false; }
  void VisitStmt_(const BlockNode* op) final { safe_ = false; }
  void VisitStmt_(const BlockRealizeNode* op) final { safe_ = false; }

  bool safe_{true};
  int payload_loop_depth_{0};
  int conditional_depth_{0};
};

class DirectPureLoopBound final : public ExprVisitor {
 public:
  bool Check(const PrimExpr& expr) {
    VisitExpr(expr);
    return safe_;
  }

 private:
  void VisitExpr_(const BufferLoadNode* op) final { safe_ = false; }
  void VisitExpr_(const CallNode* op) final { safe_ = false; }
  bool safe_{true};
};

struct HoistedBinding {
  Var var;
  PrimExpr value;
  Span span;
};

class DCUIPCPointerHoister final : public StmtExprMutator {
 public:
  explicit DCUIPCPointerHoister(IPCABI abi) : abi_(std::move(abi)) {}

  Stmt Run(const Stmt& body) { return VisitStmt(body); }

 private:
  bool IsCandidate(const LetStmtNode* let, const Var& loop_var) const {
    const auto* call = let->value.as<CallNode>();
    if (!IsIPCRemoteAddress(call)) return false;
    const auto* descriptor = call->args[0].as<VarNode>();
    if (!descriptor || !abi_.descriptors.count(descriptor)) return false;
    const auto* rank = call->args[2].as<VarNode>();
    const auto* world = call->args[3].as<VarNode>();
    if (!rank || !world || rank != abi_.rank.get() || world != abi_.world.get()) return false;
    PureInvariantExpr pure_visitor(outer_let_defs_, abi_.scalar_parameters,
                                   outer_invariant_vars_, loop_var.get());
    return pure_visitor.Check(call->args[1]);
  }

  bool ExtractBindingsFromUnconditionalBody(const Stmt& body, const Var& loop_var,
                                            std::vector<HoistedBinding>* bindings,
                                            Stmt* remainder) const {
    if (const auto* let = body.as<LetStmtNode>()) {
      if (!IsCandidate(let, loop_var)) return false;
      bindings->push_back({let->var, let->value, let->span});
      *remainder = let->body;
      return true;
    }
    const auto* seq = body.as<SeqStmtNode>();
    if (!seq || seq->seq.empty()) return false;
    Array<Stmt> rest;
    for (const Stmt& stmt : seq->seq) {
      const auto* let = stmt.as<LetStmtNode>();
      if (let && IsCandidate(let, loop_var)) {
        // This Let itself is an unconditional top-level SeqStmt child.  Move
        // only its remote-address producer; its payload body stays at this
        // exact sequence position, so loads, arithmetic and stores neither
        // move nor reorder around one another.
        bindings->push_back({let->var, let->value, let->span});
        rest.push_back(let->body);
      } else {
        rest.push_back(stmt);
      }
    }
    if (bindings->empty()) return false;
    *remainder = SeqStmt(rest);
    return true;
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    PrimExpr value = VisitExpr(op->value);
    outer_let_defs_.emplace(op->var.get(), value);
    Stmt body = VisitStmt(op->body);
    outer_let_defs_.erase(op->var.get());
    if (value.same_as(op->value) && body.same_as(op->body)) return GetRef<Stmt>(op);
    return LetStmt(op->var, value, body, op->span);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Recurse first so a nested serial loop is considered only inside its own
    // lexical scope.  This loop's direct candidate bindings will never move
    // across an enclosing effect or control-flow boundary.
    PrimExpr min = VisitExpr(op->min);
    PrimExpr extent = VisitExpr(op->extent);
    // A remote peer may depend on a lexically enclosing *serial* loop or
    // launch-thread variable (AG's round-robin source peer does).  It remains
    // invariant for this loop's extraction, unlike this loop_var itself or
    // any nested var.  Do not admit parallel/vectorized/thread-bound loop
    // variables: their execution semantics are outside this narrow proof.
    const bool outer_serial = op->kind == ForKind::kSerial && op->HasTrivialStep();
    if (outer_serial) outer_invariant_vars_.insert(op->loop_var.get());
    Stmt body = VisitStmt(op->body);
    if (outer_serial) outer_invariant_vars_.erase(op->loop_var.get());
    if (op->kind != ForKind::kSerial) {
      if (min.same_as(op->min) && extent.same_as(op->extent) && body.same_as(op->body)) {
        return GetRef<Stmt>(op);
      }
      return For(op->loop_var, min, extent, op->kind, body, op->thread_binding, op->annotations);
    }

    ElementLoopEffectAudit effects;
    effects(body);
    // The zero-trip guard below reads the bounds once more.  It is valid only
    // when that extra read cannot reload memory or re-enter an opaque helper;
    // an already-evaluated enclosing Let is a Var here and remains safe.
    DirectPureLoopBound pure_min;
    DirectPureLoopBound pure_extent;
    std::vector<HoistedBinding> bindings;
    Stmt remainder;
    if (!effects.safe() || !pure_min.Check(min) || !pure_extent.Check(extent) ||
        !ExtractBindingsFromUnconditionalBody(body, op->loop_var, &bindings, &remainder)) {
      if (min.same_as(op->min) && extent.same_as(op->extent) && body.same_as(op->body)) {
        return GetRef<Stmt>(op);
      }
      return For(op->loop_var, min, extent, op->kind, body, op->thread_binding, op->annotations);
    }

    Stmt rewritten_loop = For(op->loop_var, min, extent, op->kind, remainder, op->thread_binding,
                              op->annotations);
    Stmt hoisted = rewritten_loop;
    for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
      hoisted = LetStmt(it->var, it->value, hoisted, it->span);
    }
    // A zero-trip loop did not evaluate its opaque pointer resolution.  Keep
    // that behavior by guarding the externalized Lets whenever positivity is
    // not a compile-time fact.
    const auto* constant_extent = extent.as<IntImmNode>();
    if (!constant_extent || constant_extent->value <= 0) {
      hoisted = IfThenElse(GT(extent, make_zero(extent.dtype())), hoisted);
    }
    return hoisted;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key != tir::attr::thread_extent) return StmtExprMutator::VisitStmt_(op);
    const auto* iv = op->node.as<IterVarNode>();
    if (!iv) return StmtExprMutator::VisitStmt_(op);
    outer_invariant_vars_.insert(iv->var.get());
    Stmt body = VisitStmt(op->body);
    outer_invariant_vars_.erase(iv->var.get());
    if (body.same_as(op->body)) return GetRef<Stmt>(op);
    return AttrStmt(op->node, op->attr_key, op->value, body);
  }

  IPCABI abi_;
  std::unordered_map<const VarNode*, PrimExpr> outer_let_defs_;
  std::unordered_set<const VarNode*> outer_invariant_vars_;
};

PrimFunc HoistDCUIPCPointers(PrimFunc func) {
  std::optional<IPCABI> abi = GetVerifiedIPCABI(func);
  if (!abi.has_value()) return func;
  DescriptorUseAudit descriptor_audit(abi->descriptors);
  descriptor_audit(func->body);
  if (!descriptor_audit.safe()) return func;
  Stmt body = DCUIPCPointerHoister(std::move(*abi)).Run(func->body);
  if (body.same_as(func->body)) return func;
  return PrimFunc(func->params, body, func->ret_type, func->buffer_map, func->attrs, func->span);
}

}  // namespace

namespace transform {
using namespace tir::transform;

tvm::transform::Pass HoistDCUIPCPointers() {
  auto pass_func = [](PrimFunc func, const IRModule&, PassContext) {
    return tl::HoistDCUIPCPointers(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.HoistDCUIPCPointers", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.HoistDCUIPCPointers", HoistDCUIPCPointers);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
