/*!
 * \file vectorize_dcu_ipc_communication.cc
 * \brief Narrow 16-bit-float x8 rewrite for verified scheduled RS, AR and AG communication loops.
 *
 * This is intentionally not a generic vectorizer or LICM.  It recognizes the
 * exact post-HoistDCUIPCPointers scalar forms: RS is P conditional or direct
 * pulls plus one store; AR is P direct pulls, a local rounded reduction, and P
 * rotated-peer stores; AG is one direct remote-to-local copy.  Its caller guarantees 16-byte alignment, valid and
 * representable allocation ranges, and that outputs do not overlap inputs;
 * this pass emits no runtime safety guard or scalar fallback.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace {

using namespace tir;
using namespace arith;
using tvm::ffi::Array;
using tvm::ffi::GetRef;
using tvm::ffi::Optional;
using tvm::ffi::String;

constexpr const char* kIPCABIReadyAttr = "tl.dcu_ipc.abi_ready";
constexpr const char* kIPCBindingsAttr = "tl.dcu_ipc.bindings";
constexpr const char* kIPCPeCountAttr = "tl.dcu_ipc.pe_count";
constexpr const char* kOpaqueThreadID = "tl::ipc::opaque_thread_id";
constexpr const char* kRemoteAddress = "tl.ipc.remote_address";

bool IsVerifiedGfx936IPCFunction(const PrimFunc& func) {
  if (!func->HasNonzeroAttr(kIPCABIReadyAttr)) return false;
  auto bindings = func->GetAttr<String>(kIPCBindingsAttr);
  if (!bindings.has_value() || bindings.value().empty()) return false;
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.has_value() || target.value()->kind->name != "hip") return false;
  auto mcpu = target.value()->GetAttr<String>("mcpu");
  return mcpu.has_value() && mcpu.value() == "gfx936";
}

// Legacy scheduled functions predate the explicit compile-time PE contract.
// Preserve their four-peer behavior when it is absent, but fail closed for
// any value other than the two IPC profiles supported by this narrow pass.
std::optional<int> GetIPCPeerCount(const PrimFunc& func) {
  auto attr = func->GetAttr<Integer>(kIPCPeCountAttr);
  const int64_t peer_count = attr.has_value() ? attr.value().IntValue() : 4;
  if (peer_count != 4 && peer_count != 8) return std::nullopt;
  return static_cast<int>(peer_count);
}

bool IsOpaqueThreadID(const PrimExpr& e, const Var& thread_var) {
  const auto* call = e.as<CallNode>();
  if (!call || !call->op.same_as(builtin::call_extern()) || call->dtype != DataType::Int(32) ||
      call->args.size() != 2) return false;
  const auto* name = call->args[0].as<StringImmNode>();
  const auto* arg = call->args[1].as<VarNode>();
  return name && name->value == kOpaqueThreadID && arg == thread_var.get();
}

bool IsRemoteAddress(const PrimExpr& e) {
  const auto* call = e.as<CallNode>();
  const auto* op = call ? call->op.as<OpNode>() : nullptr;
  return op && op->name == kRemoteAddress && call->args.size() == 4;
}

std::optional<int64_t> ConstInt(const PrimExpr& e) {
  if (const auto* imm = e.as<IntImmNode>()) return imm->value;
  return std::nullopt;
}

bool IsZero(const PrimExpr& e) {
  if (const auto* f = e.as<FloatImmNode>()) return f->value == 0.0;
  if (const auto* i = e.as<IntImmNode>()) return i->value == 0;
  return false;
}

bool HasOnlyArithmetic(const PrimExpr& e) {
  struct Checker final : ExprVisitor {
    bool safe{true};
    void VisitExpr_(const BufferLoadNode*) final { safe = false; }
    void VisitExpr_(const CallNode*) final { safe = false; }
  } checker;
  checker(e);
  return checker.safe && SideEffect(e) <= CallEffectKind::kPure;
}

class TwoVarSubstituter final : public ExprMutator {
 public:
  TwoVarSubstituter(const Var& a, PrimExpr av, const Var& b, PrimExpr bv)
      : a_(a.get()), av_(std::move(av)), b_(b.get()), bv_(std::move(bv)) {}
 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == a_) return av_;
    if (op == b_) return bv_;
    return GetRef<PrimExpr>(op);
  }
  const VarNode* a_;
  PrimExpr av_;
  const VarNode* b_;
  PrimExpr bv_;
};

PrimExpr SubstituteOldCoordinates(const PrimExpr& e, const Var& old_i, const Var& old_tx,
                                  const PrimExpr& new_i, const Var& tx, int lane) {
  const DataType t = tx.dtype();
  const PrimExpr old_i_value = new_i * make_const(new_i.dtype(), 8) +
                                floordiv(tx, make_const(t, 64));
  const PrimExpr old_tx_value = floormod(tx, make_const(t, 64)) * make_const(t, 8) +
                                 make_const(t, lane);
  return TwoVarSubstituter(old_i, old_i_value, old_tx, old_tx_value)(e);
}

bool Is16BitFloat(const DataType& dtype) {
  return !dtype.is_scalable_vector() && dtype.lanes() == 1 &&
         (dtype.is_float16() || dtype.is_bfloat16());
}

bool IsFlat16BitFloatBuffer(const Buffer& buffer) {
  if (!Is16BitFloat(buffer->dtype) || buffer.scope() != "global" || buffer->data_alignment < 16 ||
      buffer->data_alignment % 16 != 0 ||
      buffer->elem_offset.as<IntImmNode>() == nullptr ||
      buffer->elem_offset.as<IntImmNode>()->value != 0 || !buffer->strides.empty() ||
      buffer->shape.size() != 1) return false;
  auto n = ConstInt(buffer->shape[0]);
  if (!n.has_value() || *n <= 0 || static_cast<uint64_t>(*n) >
                                      std::numeric_limits<uint64_t>::max() / 2) return false;
  return static_cast<uint64_t>(*n) * 2 >= 16;
}

struct ScalarPull {
  Buffer temp;
  Buffer remote;
  PrimExpr source_index;
  PrimExpr predicate;
};

bool MatchScalarPull(const Stmt& stmt, ScalarPull* result) {
  const auto* store = stmt.as<BufferStoreNode>();
  if (!store || store->indices.size() != 1 || !IsZero(store->indices[0]) ||
      store->predicate.has_value() || !Is16BitFloat(store->buffer->dtype) ||
      store->buffer.scope() != "local") return false;
  const auto* call = store->value.as<CallNode>();
  if (!call || !call->op.same_as(builtin::if_then_else()) || call->args.size() != 3 ||
      !IsZero(call->args[2])) return false;
  const auto* load = call->args[1].as<BufferLoadNode>();
  if (!load || load->indices.size() != 1 || load->predicate.has_value() ||
      load->dtype != store->buffer->dtype) return false;
  result->temp = store->buffer;
  result->remote = load->buffer;
  result->source_index = load->indices[0];
  result->predicate = call->args[0];
  return HasOnlyArithmetic(result->source_index) && SideEffect(result->predicate) <= CallEffectKind::kPure;
}

bool MatchReduceStore(const Stmt& stmt, const std::vector<ScalarPull>& pulls,
                      Buffer* out, PrimExpr* out_index) {
  if (pulls.empty()) return false;
  const DataType payload = pulls[0].temp->dtype;
  const auto* store = stmt.as<BufferStoreNode>();
  if (!store || store->indices.size() != 1 || store->predicate.has_value() ||
      store->buffer->dtype != payload || store->buffer.scope() != "global") return false;
  const auto* cast = store->value.as<CastNode>();
  if (!cast || cast->dtype != payload) return false;
  PrimExpr value = cast->value;
  for (int i = static_cast<int>(pulls.size()) - 1; i >= 1; --i) {
    const auto* add = value.as<AddNode>();
    if (!add) return false;
    const auto* rhs_cast = add->b.as<CastNode>();
    const auto* rhs = rhs_cast ? rhs_cast->value.as<BufferLoadNode>() : nullptr;
    if (!rhs_cast || rhs_cast->dtype != DataType::Float(32) || !rhs || rhs->indices.size() != 1 ||
        rhs->predicate.has_value() || !rhs->buffer.same_as(pulls[i].temp) || !IsZero(rhs->indices[0])) return false;
    value = add->a;
  }
  const auto* lhs_cast = value.as<CastNode>();
  const auto* lhs = lhs_cast ? lhs_cast->value.as<BufferLoadNode>() : nullptr;
  if (!lhs_cast || lhs_cast->dtype != DataType::Float(32) || !lhs || lhs->indices.size() != 1 ||
      lhs->predicate.has_value() || !lhs->buffer.same_as(pulls[0].temp) || !IsZero(lhs->indices[0])) return false;
  *out = store->buffer;
  *out_index = store->indices[0];
  return HasOnlyArithmetic(*out_index);
}

// GEMM+AllReduce keeps all communication in one scalar element loop: P
// unconditional remote pulls, an FP32 accumulation stored in a local scalar,
// then P remote pushes of that same rounded 16-bit value.  Keep this matcher
// separate from the ReduceScatter conditional-pull/one-store normal form.
bool MatchARScalarPull(const Stmt& stmt, ScalarPull* result) {
  const auto* store = stmt.as<BufferStoreNode>();
  if (!store || store->indices.size() != 1 || !IsZero(store->indices[0]) ||
      store->predicate.has_value() || !Is16BitFloat(store->buffer->dtype) ||
      store->buffer.scope() != "local") return false;
  const auto* load = store->value.as<BufferLoadNode>();
  if (!load || load->indices.size() != 1 || load->predicate.has_value() ||
      load->dtype != store->buffer->dtype) return false;
  result->temp = store->buffer;
  result->remote = load->buffer;
  result->source_index = load->indices[0];
  result->predicate = make_const(DataType::Bool(), true);
  return HasOnlyArithmetic(result->source_index);
}

bool MatchARReduceTemp(const Stmt& stmt, const std::vector<ScalarPull>& pulls,
                       Buffer* reduced) {
  if (pulls.empty()) return false;
  const DataType payload = pulls[0].temp->dtype;
  const auto* store = stmt.as<BufferStoreNode>();
  if (!store || store->indices.size() != 1 || !IsZero(store->indices[0]) ||
      store->predicate.has_value() || store->buffer->dtype != payload ||
      store->buffer.scope() != "local") return false;
  const auto* cast = store->value.as<CastNode>();
  if (!cast || cast->dtype != payload) return false;
  PrimExpr value = cast->value;
  for (int i = static_cast<int>(pulls.size()) - 1; i >= 1; --i) {
    const auto* add = value.as<AddNode>();
    const auto* rhs_cast = add ? add->b.as<CastNode>() : nullptr;
    const auto* rhs = rhs_cast ? rhs_cast->value.as<BufferLoadNode>() : nullptr;
    if (!rhs_cast || rhs_cast->dtype != DataType::Float(32) || !rhs || rhs->indices.size() != 1 ||
        rhs->predicate.has_value() || !rhs->buffer.same_as(pulls[i].temp) || !IsZero(rhs->indices[0])) return false;
    value = add->a;
  }
  const auto* lhs_cast = value.as<CastNode>();
  const auto* lhs = lhs_cast ? lhs_cast->value.as<BufferLoadNode>() : nullptr;
  if (!lhs_cast || lhs_cast->dtype != DataType::Float(32) || !lhs || lhs->indices.size() != 1 ||
      lhs->predicate.has_value() || !lhs->buffer.same_as(pulls[0].temp) || !IsZero(lhs->indices[0])) return false;
  *reduced = store->buffer;
  return true;
}

struct ARScalarPush {
  Buffer remote;
  PrimExpr index;
};

// AllGather's scalar normal form is deliberately simpler than AR: one
// hoisted remote source and one rank-local gathered-A destination.  Keep the
// matcher distinct rather than weakening either reduction matcher.
struct AGScalarCopy {
  Buffer source;
  Buffer destination;
  PrimExpr source_index;
  PrimExpr destination_index;
};

bool MatchAGScalarCopy(const Stmt& stmt, AGScalarCopy* result) {
  const auto* store = stmt.as<BufferStoreNode>();
  if (!store || store->indices.size() != 1 || store->predicate.has_value() ||
      !Is16BitFloat(store->buffer->dtype) || store->buffer.scope() != "global") return false;
  const auto* load = store->value.as<BufferLoadNode>();
  if (!load || load->indices.size() != 1 || load->predicate.has_value() ||
      load->dtype != store->buffer->dtype) return false;
  result->source = load->buffer;
  result->destination = store->buffer;
  result->source_index = load->indices[0];
  result->destination_index = store->indices[0];
  return HasOnlyArithmetic(result->source_index) && HasOnlyArithmetic(result->destination_index);
}

bool MatchARScalarPush(const Stmt& stmt, const Buffer& reduced, ARScalarPush* result) {
  const auto* store = stmt.as<BufferStoreNode>();
  if (!store || store->indices.size() != 1 || store->predicate.has_value() ||
      store->buffer->dtype != reduced->dtype || store->buffer.scope() != "global") return false;
  const auto* value = store->value.as<BufferLoadNode>();
  if (!value || value->indices.size() != 1 || value->predicate.has_value() ||
      !value->buffer.same_as(reduced) || !IsZero(value->indices[0])) return false;
  result->remote = store->buffer;
  result->index = store->indices[0];
  return HasOnlyArithmetic(result->index);
}

std::vector<Stmt> FlattenSeq(const Stmt& stmt) {
  if (const auto* seq = stmt.as<SeqStmtNode>()) return std::vector<Stmt>(seq->seq.begin(), seq->seq.end());
  return {stmt};
}

class LocalTempUseCounter final : public StmtExprVisitor {
 public:
  void VisitExpr_(const VarNode* op) final { ++bare_vars[op]; }
  void VisitExpr_(const BufferLoadNode* op) final {
    ++loads[op->buffer->data.get()];
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode* op) final {
    ++stores[op->buffer->data.get()];
    StmtExprVisitor::VisitStmt_(op);
  }
  std::unordered_map<const VarNode*, int> loads, stores, bare_vars;
};

struct IPCABI {
  std::unordered_set<const VarNode*> descriptors;
  Var rank;
  Var world;
};

std::optional<IPCABI> GetIPCABI(const PrimFunc& func) {
  auto bindings = func->GetAttr<String>(kIPCBindingsAttr);
  if (!bindings.has_value()) return std::nullopt;
  std::unordered_set<std::string> names;
  std::stringstream ss(static_cast<std::string>(bindings.value()));
  std::string item;
  while (std::getline(ss, item, ';')) {
    const size_t eq = item.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == item.size()) return std::nullopt;
    names.insert(item.substr(eq + 1));
  }
  IPCABI abi;
  for (const Var& p : func->params) {
    if (p->name_hint == "__tl_ipc_rank") abi.rank = p;
    if (p->name_hint == "__tl_ipc_world_size") abi.world = p;
    if (!names.count(p->name_hint)) continue;
    bool exact_data = false;
    for (const auto& kv : func->buffer_map)
      exact_data |= kv.first.same_as(p) && kv.second->data.same_as(p);
    if (!exact_data) return std::nullopt;
    abi.descriptors.insert(p.get());
  }
  if (!abi.rank.defined() || !abi.world.defined() || abi.descriptors.empty()) return std::nullopt;
  return abi;
}

bool RemoteAddressMatches(const LetStmtNode* let, const Buffer& remote, const IPCABI& abi,
                          int peer_count, std::unordered_set<int64_t>* peers) {
  if (!remote->data.same_as(let->var)) return false;
  const auto* call = let->value.as<CallNode>();
  if (!call || !IsRemoteAddress(let->value) || !call->args[0].as<VarNode>() ||
      !abi.descriptors.count(call->args[0].as<VarNode>()) ||
      !call->args[2].same_as(abi.rank) || !call->args[3].same_as(abi.world)) return false;
  auto peer = ConstInt(call->args[1]);
  return peer.has_value() && *peer >= 0 && *peer < peer_count && peers->insert(*peer).second;
}

bool RemoteAddressMatchesAR(const LetStmtNode* let, const Buffer& remote, const IPCABI& abi,
                            const VarNode** descriptor, PrimExpr* peer) {
  if (!remote->data.same_as(let->var)) return false;
  const auto* call = let->value.as<CallNode>();
  const auto* descriptor_var = call && IsRemoteAddress(let->value) ? call->args[0].as<VarNode>() : nullptr;
  if (!descriptor_var || !abi.descriptors.count(descriptor_var) ||
      !call->args[2].same_as(abi.rank) || !call->args[3].same_as(abi.world)) return false;
  *descriptor = descriptor_var;
  *peer = call->args[1];
  return HasOnlyArithmetic(*peer);
}

// The matcher runs below several scheduled-control scopes.  Replaying those
// scopes into a fresh analyzer makes the allocation proof depend on the same
// lexical facts as the candidate, without sharing facts with a sibling branch
// or a later loop iteration.  Range bounds are intentionally materialized only
// after the preceding facts have been replayed: Bind(var, symbolic_range)
// alone does not necessarily feed a newly learned blockIdx bound back into the
// symbolic range's constant bound.
struct RangeProofFact {
  enum class Kind { kRange, kLet, kConstraint };
  Kind kind;
  Var var;
  PrimExpr first;
  PrimExpr second;
};

class RangeProofDomain {
 public:
  ~RangeProofDomain() {
    // ConstraintContext restores analyzer state on destruction.  Pop in strict
    // LIFO order while the analyzer is still alive.
    while (!constraints.empty()) constraints.pop_back();
  }

  bool BindRange(const Var& var, const PrimExpr& min, const PrimExpr& extent) {
    if (!var.dtype().is_int() || var.dtype().lanes() != 1 || !HasOnlyArithmetic(min) ||
        !HasOnlyArithmetic(extent) || (var.dtype().bits() != 32 && var.dtype().bits() != 64)) return false;
    // Entering a For/launch body implies a strictly positive extent, even if
    // a scheduled CTA computes a syntactically zero extent for another block.
    constraints.emplace_back(std::make_unique<With<ConstraintContext>>(&analyzer, extent > 0));
    // Derive a finite enclosing interval under the facts already in scope.
    // This is conservative (never narrower than the lexical range) and avoids
    // carrying an outer symbolic extent whose own bound was learned earlier.
    const ConstIntBound min_bound = analyzer.const_int_bound(min);
    const ConstIntBound extent_bound = analyzer.const_int_bound(extent);
    if (min_bound->min_value == ConstIntBound::kNegInf ||
        min_bound->max_value == ConstIntBound::kPosInf ||
        extent_bound->min_value == ConstIntBound::kNegInf ||
        extent_bound->max_value == ConstIntBound::kPosInf || extent_bound->max_value <= 0)
      return false;
    // Compute the enclosing range before narrowing back to the TIR index
    // dtype; a failed representability proof is a reject, never a wraparound.
    const __int128 lower = min_bound->min_value;
    const __int128 upper_exclusive = static_cast<__int128>(min_bound->max_value) +
                                     static_cast<__int128>(extent_bound->max_value);
    const __int128 width = upper_exclusive - lower;
    const int bits = var.dtype().bits();
    const __int128 dtype_min = bits == 64 ? static_cast<__int128>(std::numeric_limits<int64_t>::min())
                                           : -((static_cast<__int128>(1)) << (bits - 1));
    const __int128 dtype_max = bits == 64 ? static_cast<__int128>(std::numeric_limits<int64_t>::max())
                                           : ((static_cast<__int128>(1) << (bits - 1)) - 1);
    // The constraints below reconstruct `min + extent` in the index dtype;
    // require its exclusive endpoint to be representable as well.
    if (width <= 0 || lower < dtype_min || lower > dtype_max || width > dtype_max ||
        upper_exclusive < dtype_min || upper_exclusive > dtype_max) return false;
    analyzer.Bind(var, Range::FromMinExtent(make_const(var.dtype(), static_cast<int64_t>(lower)),
                                             make_const(var.dtype(), static_cast<int64_t>(width))));
    constraints.emplace_back(std::make_unique<With<ConstraintContext>>(&analyzer, var >= min));
    constraints.emplace_back(std::make_unique<With<ConstraintContext>>(&analyzer, var < min + extent));
    return true;
  }

  bool BindLet(const Var& var, const PrimExpr& value) {
    if (!HasOnlyArithmetic(value)) return false;
    analyzer.Bind(var, value);
    return true;
  }

  bool AddConstraint(const PrimExpr& condition) {
    if (!HasOnlyArithmetic(condition)) return false;
    constraints.emplace_back(std::make_unique<With<ConstraintContext>>(&analyzer, condition));
    return true;
  }

  Analyzer analyzer;

 private:
  std::vector<std::unique_ptr<With<ConstraintContext>>> constraints;
};

class IPCVectorizer final : public StmtExprMutator {
 public:
  IPCVectorizer(const PrimFunc& func, IPCABI abi, int peer_count)
      : func_(func), abi_(std::move(abi)), peer_count_(peer_count) {
    usages_(func->body);
    for (const Var& p : func->params) uniform_vars_.insert(p.get());
  }

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // InjectAssumes canonicalizes T.assume(condition) as an attribute whose
    // node is the condition.  Replay only arithmetic assumptions as lexical
    // proof facts; this is needed to prove the AR rank rotation is within the
    // fixed P-peer ABI, not to rewrite arbitrary attribute scopes.
    if (op->attr_key == tir::attr::tilelang_assume) {
      const auto* condition = op->node.as<PrimExprNode>();
      if (!condition || !HasOnlyArithmetic(GetRef<PrimExpr>(condition))) {
        return StmtExprMutator::VisitStmt_(op);
      }
      const size_t proof_scope = proof_facts_.size();
      proof_facts_.push_back(
          {RangeProofFact::Kind::kConstraint, Var(), GetRef<PrimExpr>(condition), PrimExpr()});
      Stmt body = StmtExprMutator::VisitStmt(op->body);
      proof_facts_.resize(proof_scope);
      if (body.same_as(op->body)) return GetRef<Stmt>(op);
      return AttrStmt(op->node, op->attr_key, op->value, body);
    }
    if (op->attr_key == tir::attr::volatile_scope) {
      const bool saved = uniform_control_;
      uniform_control_ = false;
      Stmt body = StmtExprMutator::VisitStmt(op->body);
      uniform_control_ = saved;
      return AttrStmt(op->node, op->attr_key, op->value, body);
    }
    if (op->attr_key == tir::attr::thread_extent) {
      const auto* iv = op->node.as<IterVarNode>();
      const auto extent = ConstInt(op->value);
      if (iv && (iv->thread_tag == "threadIdx.y" || iv->thread_tag == "threadIdx.z") &&
          (!extent.has_value() || *extent != 1)) {
        const bool saved = nonunit_nonx_thread_;
        nonunit_nonx_thread_ = true;
        Stmt body = StmtExprMutator::VisitStmt(op->body);
        nonunit_nonx_thread_ = saved;
        return AttrStmt(op->node, op->attr_key, op->value, body);
      }
      if (iv && std::string(iv->thread_tag).rfind("blockIdx.", 0) == 0) {
        uniform_vars_.insert(iv->var.get());
        const size_t proof_scope = proof_facts_.size();
        if (HasOnlyArithmetic(op->value)) {
          PushRangeProof(iv->var, make_zero(iv->var.dtype()), op->value);
        }
        Stmt body = StmtExprMutator::VisitStmt(op->body);
        proof_facts_.resize(proof_scope);
        uniform_vars_.erase(iv->var.get());
        return AttrStmt(op->node, op->attr_key, op->value, body);
      }
      if (iv && iv->thread_tag == "threadIdx.x") {
        const auto saved_var = thread_x_;
        const bool saved_extent = thread_x_512_;
        thread_x_.reset(); thread_x_512_ = false;
        if (extent.has_value() && *extent == 512 && iv->var.dtype() == DataType::Int(32)) {
          thread_x_ = iv->var; thread_x_512_ = true;
        }
        Stmt body = StmtExprMutator::VisitStmt(op->body);
        thread_x_ = saved_var; thread_x_512_ = saved_extent;
        if (body.same_as(op->body)) return GetRef<Stmt>(op);
        return AttrStmt(op->node, op->attr_key, op->value, body);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    if (uniform_control_ && !nonunit_nonx_thread_ && thread_x_512_ && thread_x_.has_value() &&
        IsOpaqueThreadID(op->value, *thread_x_)) {
      if (auto replacement = TryRewriteOpaqueScope(op)) return *replacement;
    }
    const bool saved = uniform_control_;
    uniform_control_ = uniform_control_ && IsUniformControlExpr(op->value);
    if (uniform_control_) uniform_vars_.insert(op->var.get());
    const size_t proof_scope = proof_facts_.size();
    if (HasOnlyArithmetic(op->value)) {
      proof_facts_.push_back({RangeProofFact::Kind::kLet, op->var, op->value, PrimExpr()});
    }
    Stmt result = StmtExprMutator::VisitStmt_(op);
    proof_facts_.resize(proof_scope);
    uniform_vars_.erase(op->var.get());
    uniform_control_ = saved;
    return result;
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    const bool saved = uniform_control_;
    uniform_control_ = uniform_control_ && IsUniformControlExpr(op->condition);
    const PrimExpr condition = VisitExpr(op->condition);
    const size_t proof_scope = proof_facts_.size();
    if (HasOnlyArithmetic(condition)) {
      proof_facts_.push_back({RangeProofFact::Kind::kConstraint, Var(), condition, PrimExpr()});
    }
    Stmt then_case = StmtExprMutator::VisitStmt(op->then_case);
    proof_facts_.resize(proof_scope);
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      if (HasOnlyArithmetic(condition)) {
        proof_facts_.push_back({RangeProofFact::Kind::kConstraint, Var(), Not(condition), PrimExpr()});
      }
      else_case = StmtExprMutator::VisitStmt(op->else_case.value());
      proof_facts_.resize(proof_scope);
    }
    uniform_control_ = saved;
    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) return GetRef<Stmt>(op);
    return IfThenElse(condition, then_case, else_case);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    const bool saved = uniform_control_;
    uniform_control_ = uniform_control_ && op->kind == ForKind::kSerial && op->HasTrivialStep() &&
                       IsUniformControlExpr(op->min) && IsUniformControlExpr(op->extent);
    if (uniform_control_) uniform_vars_.insert(op->loop_var.get());
    const size_t proof_scope = proof_facts_.size();
    if (op->kind == ForKind::kSerial && op->HasTrivialStep() && HasOnlyArithmetic(op->min) &&
        HasOnlyArithmetic(op->extent)) {
      PushRangeProof(op->loop_var, op->min, op->extent);
    }
    Stmt result = StmtExprMutator::VisitStmt_(op);
    proof_facts_.resize(proof_scope);
    uniform_vars_.erase(op->loop_var.get());
    uniform_control_ = saved;
    return result;
  }

  Stmt VisitStmt_(const WhileNode* op) final {
    const bool saved = uniform_control_;
    uniform_control_ = false;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    uniform_control_ = saved;
    return result;
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    const bool saved = uniform_control_;
    uniform_control_ = uniform_control_ && IsUniformControlExpr(op->condition);
    Stmt result = StmtExprMutator::VisitStmt_(op);
    uniform_control_ = saved;
    return result;
  }

  bool IsUniformControlExpr(const PrimExpr& e) const {
    return HasOnlyArithmetic(e) &&
           !UsesVar(e, [&](const VarNode* v) { return !uniform_vars_.count(v); });
  }

  bool IsParameterBuffer(const Buffer& buffer) const {
    for (const auto& kv : func_->buffer_map)
      if (kv.second->data.same_as(buffer->data)) {
        for (const Var& param : func_->params)
          if (param.same_as(kv.first)) return true;
      }
    return false;
  }

  void PushRangeProof(const Var& var, const PrimExpr& min, const PrimExpr& extent) {
    proof_facts_.push_back({RangeProofFact::Kind::kRange, var, min, extent});
  }

  std::unique_ptr<RangeProofDomain> MakeOuterProofDomain() const {
    auto domain = std::make_unique<RangeProofDomain>();
    for (const RangeProofFact& fact : proof_facts_) {
      bool accepted = false;
      switch (fact.kind) {
        case RangeProofFact::Kind::kRange:
          accepted = domain->BindRange(fact.var, fact.first, fact.second);
          break;
        case RangeProofFact::Kind::kLet:
          accepted = domain->BindLet(fact.var, fact.first);
          break;
        case RangeProofFact::Kind::kConstraint:
          accepted = domain->AddConstraint(fact.first);
          break;
      }
      if (!accepted) return nullptr;
    }
    return domain;
  }

  bool IsARRotatedOutputPeer(const PrimExpr& peer, int offset) const {
    if (offset == 0) return peer.same_as(abi_.rank);
    const DataType dtype = abi_.rank.dtype();
    const PrimExpr expected = floormod(abi_.rank + make_const(dtype, offset),
                                       make_const(dtype, peer_count_));
    return ExprDeepEqual()(peer, expected);
  }

  std::optional<Stmt> TryRewriteAGOpaqueScope(const LetStmtNode* opaque_let) {
    // AG has one remote source pointer.  HoistDCUIPCPointers may make the
    // pointer chain the direct opaque body (unlike AR's SeqStmt with trailing
    // stores), so accept only that direct chain or its first, unconditional
    // SeqStmt child.  Later siblings are preserved in lexical order.
    const auto* opaque_seq = opaque_let->body.as<SeqStmtNode>();
    if (opaque_seq && opaque_seq->seq.empty()) return std::nullopt;
    Stmt first = opaque_seq ? opaque_seq->seq[0] : opaque_let->body;
    const auto* ptr_let = first.as<LetStmtNode>();
    if (!ptr_let || !IsRemoteAddress(ptr_let->value)) return std::nullopt;
    const auto* old_for = ptr_let->body.as<ForNode>();
    if (!old_for || old_for->kind != ForKind::kSerial || old_for->thread_binding.defined() ||
        !old_for->HasTrivialStep() || !IsZero(old_for->min) ||
        ConstInt(old_for->extent) != std::optional<int64_t>(128) || !old_for->annotations.empty()) {
      return std::nullopt;
    }
    AGScalarCopy copy;
    if (!MatchAGScalarCopy(old_for->body, &copy) || !IsFlat16BitFloatBuffer(copy.source) ||
        !IsFlat16BitFloatBuffer(copy.destination) || !IsParameterBuffer(copy.destination) ||
        copy.source->data.same_as(copy.destination->data)) {
      return std::nullopt;
    }

    const VarNode* descriptor = nullptr;
    PrimExpr peer;
    if (!RemoteAddressMatchesAR(ptr_let, copy.source, abi_, &descriptor, &peer) || !descriptor ||
        UsesVar(peer, [&](const VarNode* v) {
          return v == old_for->loop_var.get() || v == opaque_let->var.get();
        }) || !IsUniformControlExpr(peer)) {
      return std::nullopt;
    }

    // Reconstruct the exact 128*512 scalar domain in the enclosing lexical
    // scope.  The canonical base+linear proof gives each old scalar writer a
    // unique vector lane; the explicit range proofs rule out descriptor or
    // ordinary-buffer overrun before emitting a no-fallback vector access.
    auto old_domain = MakeOuterProofDomain();
    if (!old_domain ||
        !old_domain->BindRange(old_for->loop_var, make_zero(old_for->loop_var.dtype()),
                                make_const(old_for->loop_var.dtype(), 128)) ||
        !old_domain->BindRange(opaque_let->var, make_zero(opaque_let->var.dtype()),
                                make_const(opaque_let->var.dtype(), 512))) {
      return std::nullopt;
    }
    if (!old_domain->analyzer.CanProve(peer >= make_zero(peer.dtype())) ||
        !old_domain->analyzer.CanProve(peer < make_const(peer.dtype(), peer_count_))) {
      return std::nullopt;
    }
    const PrimExpr zero_i = make_zero(old_for->loop_var.dtype());
    const PrimExpr zero_tx = make_zero(opaque_let->var.dtype());
    const PrimExpr linear = old_for->loop_var * make_const(old_for->loop_var.dtype(), 512) +
                            opaque_let->var;
    const PrimExpr source_base = TwoVarSubstituter(old_for->loop_var, zero_i, opaque_let->var,
                                                    zero_tx)(copy.source_index);
    const PrimExpr destination_base = TwoVarSubstituter(old_for->loop_var, zero_i, opaque_let->var,
                                                         zero_tx)(copy.destination_index);
    if (!IsUniformControlExpr(source_base) || !IsUniformControlExpr(destination_base) ||
        !old_domain->analyzer.CanProveEqual(copy.source_index, source_base + linear) ||
        !old_domain->analyzer.CanProveEqual(copy.destination_index, destination_base + linear) ||
        !old_domain->analyzer.CanProve(copy.source_index >= make_zero(copy.source_index.dtype())) ||
        !old_domain->analyzer.CanProve(copy.source_index < copy.source->shape[0]) ||
        !old_domain->analyzer.CanProve(copy.destination_index >= make_zero(copy.destination_index.dtype())) ||
        !old_domain->analyzer.CanProve(copy.destination_index < copy.destination->shape[0])) {
      return std::nullopt;
    }

    Var new_i("ipc_ag_half8_i", DataType::Int(32));
    const Var& tx = opaque_let->var;
    PrimExpr source0 = SubstituteOldCoordinates(copy.source_index, old_for->loop_var, tx, new_i, tx, 0);
    PrimExpr destination0 =
        SubstituteOldCoordinates(copy.destination_index, old_for->loop_var, tx, new_i, tx, 0);
    Analyzer analyzer;
    analyzer.Bind(new_i, Range::FromMinExtent(0, 16));
    analyzer.Bind(tx, Range::FromMinExtent(0, 512));
    auto contiguous_aligned_8 = [&](const PrimExpr& start, const PrimExpr& indexed) {
      if (!analyzer.CanProveEqual(floormod(start, 8), 0)) return false;
      for (int lane = 0; lane < 8; ++lane) {
        const PrimExpr expected = start + make_const(start.dtype(), lane);
        const PrimExpr actual =
            SubstituteOldCoordinates(indexed, old_for->loop_var, tx, new_i, tx, lane);
        if (!analyzer.CanProveEqual(actual, expected)) return false;
      }
      return true;
    };
    if (!contiguous_aligned_8(source0, copy.source_index) ||
        !contiguous_aligned_8(destination0, copy.destination_index)) {
      return std::nullopt;
    }
    source0 = analyzer.Simplify(source0);
    destination0 = analyzer.Simplify(destination0);
    Stmt rewritten = For(
        new_i, 0, 16, ForKind::kSerial,
        BufferStore(copy.destination, BufferLoad(copy.source, {Ramp(source0, 1, 8)}),
                    {Ramp(destination0, 1, 8)}));
    rewritten = LetStmt(ptr_let->var, ptr_let->value, rewritten);
    if (opaque_seq) {
      Array<Stmt> opaque_children{rewritten};
      for (size_t i = 1; i < opaque_seq->seq.size(); ++i) opaque_children.push_back(opaque_seq->seq[i]);
      rewritten = SeqStmt::Flatten(opaque_children);
    }
    return LetStmt(opaque_let->var, opaque_let->value, rewritten);
  }

  std::optional<Stmt> TryRewriteAROpaqueScope(const LetStmtNode* opaque_let) {
    // HoistDCUIPCPointers has already established immutable descriptor ABI
    // validity and places these input/output addresses at the head of one tile
    // scope.  Do not inspect or move candidates from a later child, branch, or
    // nested loop: the trailing fence/barrier remains in its original place.
    const auto* opaque_seq = opaque_let->body.as<SeqStmtNode>();
    if (!opaque_seq || opaque_seq->seq.empty()) return std::nullopt;
    std::vector<const LetStmtNode*> ptr_lets(2 * peer_count_);
    Stmt cursor = opaque_seq->seq[0];
    for (int i = 0; i < 2 * peer_count_; ++i) {
      ptr_lets[i] = cursor.as<LetStmtNode>();
      if (!ptr_lets[i] || !IsRemoteAddress(ptr_lets[i]->value)) return std::nullopt;
      cursor = ptr_lets[i]->body;
    }
    const auto* old_for = cursor.as<ForNode>();
    if (!old_for || old_for->kind != ForKind::kSerial || old_for->thread_binding.defined() ||
        !old_for->HasTrivialStep() || !IsZero(old_for->min) ||
        ConstInt(old_for->extent) != std::optional<int64_t>(128) || !old_for->annotations.empty()) {
      return std::nullopt;
    }
    const std::vector<Stmt> seq = FlattenSeq(old_for->body);
    if (seq.size() != static_cast<size_t>(2 * peer_count_ + 1)) return std::nullopt;
    std::vector<ScalarPull> pulls(peer_count_);
    for (int i = 0; i < peer_count_; ++i)
      if (!MatchARScalarPull(seq[i], &pulls[i])) return std::nullopt;
    Buffer reduced;
    if (!MatchARReduceTemp(seq[peer_count_], pulls, &reduced)) return std::nullopt;
    std::vector<ARScalarPush> pushes(peer_count_);
    for (int i = 0; i < peer_count_; ++i)
      if (!MatchARScalarPush(seq[peer_count_ + 1 + i], reduced, &pushes[i])) return std::nullopt;

    std::unordered_set<const VarNode*> data;
    const VarNode* partial_descriptor = nullptr;
    const VarNode* output_descriptor = nullptr;
    for (int i = 0; i < peer_count_; ++i) {
      const VarNode* descriptor = nullptr;
      PrimExpr peer;
      if (!IsFlat16BitFloatBuffer(pulls[i].remote) || !data.insert(pulls[i].remote->data.get()).second ||
          !RemoteAddressMatchesAR(ptr_lets[i], pulls[i].remote, abi_, &descriptor, &peer) ||
          ConstInt(peer) != std::optional<int64_t>(i) ||
          pulls[i].temp->dtype != pulls[0].temp->dtype ||
          pulls[i].remote->dtype != pulls[0].temp->dtype ||
          usages_.loads[pulls[i].temp->data.get()] != 1 || usages_.stores[pulls[i].temp->data.get()] != 1 ||
          usages_.bare_vars[pulls[i].temp->data.get()] != 0 ||
          (partial_descriptor && descriptor != partial_descriptor)) return std::nullopt;
      partial_descriptor = descriptor;
    }
    for (int i = 0; i < peer_count_; ++i) {
      const VarNode* descriptor = nullptr;
      PrimExpr peer;
      if (!IsFlat16BitFloatBuffer(pushes[i].remote) || !data.insert(pushes[i].remote->data.get()).second ||
          !RemoteAddressMatchesAR(ptr_lets[peer_count_ + i], pushes[i].remote, abi_, &descriptor, &peer) ||
          pushes[i].remote->dtype != pulls[0].temp->dtype ||
          !IsARRotatedOutputPeer(peer, i) ||
          (output_descriptor && descriptor != output_descriptor)) return std::nullopt;
      output_descriptor = descriptor;
    }
    if (!partial_descriptor || !output_descriptor || partial_descriptor == output_descriptor ||
        reduced->dtype != pulls[0].temp->dtype ||
        usages_.loads[reduced->data.get()] != peer_count_ || usages_.stores[reduced->data.get()] != 1 ||
        usages_.bare_vars[reduced->data.get()] != 0) return std::nullopt;
    for (int i = 0; i < peer_count_; ++i) {
      for (int j = i + 1; j < peer_count_; ++j) {
        if (pulls[i].temp->data.same_as(pulls[j].temp->data)) return std::nullopt;
      }
    }

    auto old_domain = MakeOuterProofDomain();
    if (!old_domain ||
        !old_domain->BindRange(old_for->loop_var, make_zero(old_for->loop_var.dtype()),
                                make_const(old_for->loop_var.dtype(), 128)) ||
        !old_domain->BindRange(opaque_let->var, make_zero(opaque_let->var.dtype()),
                                make_const(opaque_let->var.dtype(), 512)) ||
        !old_domain->analyzer.CanProve(abi_.rank >= make_zero(abi_.rank.dtype())) ||
        !old_domain->analyzer.CanProve(abi_.rank < make_const(abi_.rank.dtype(), peer_count_))) return std::nullopt;
    const PrimExpr zero_i = make_zero(old_for->loop_var.dtype());
    const PrimExpr zero_tx = make_zero(opaque_let->var.dtype());
    const PrimExpr linear = old_for->loop_var * make_const(old_for->loop_var.dtype(), 512) + opaque_let->var;
    const PrimExpr source_base =
        TwoVarSubstituter(old_for->loop_var, zero_i, opaque_let->var, zero_tx)(pulls[0].source_index);
    const PrimExpr output_base =
        TwoVarSubstituter(old_for->loop_var, zero_i, opaque_let->var, zero_tx)(pushes[0].index);
    if (!IsUniformControlExpr(source_base) || !IsUniformControlExpr(output_base)) return std::nullopt;
    // An old element iteration advances 512 16-bit values, i.e. exactly two
    // 256-wide tile rows.  AR pushes into the full row-major matrix, so its
    // flat output address is not `base + linear`: retain the scheduled
    // two-row mapping and prove its row stride before changing the loop map.
    const PrimExpr output_i_delta = old_domain->analyzer.Simplify(
        TwoVarSubstituter(old_for->loop_var, make_const(old_for->loop_var.dtype(), 1),
                          opaque_let->var, zero_tx)(pushes[0].index) - output_base);
    const auto two_row_delta = ConstInt(output_i_delta);
    if (!two_row_delta.has_value() || *two_row_delta < 512 || (*two_row_delta & 1)) return std::nullopt;
    const PrimExpr row_stride = make_const(linear.dtype(), *two_row_delta / 2);
    const PrimExpr canonical_output = output_base +
        floordiv(linear, make_const(linear.dtype(), 256)) * row_stride +
        floormod(linear, make_const(linear.dtype(), 256));
    for (int i = 0; i < peer_count_; ++i) {
      if (!old_domain->analyzer.CanProveEqual(pulls[i].source_index, pulls[0].source_index) ||
          !old_domain->analyzer.CanProveEqual(pushes[i].index, pushes[0].index) ||
          !old_domain->analyzer.CanProveEqual(pulls[i].source_index, source_base + linear) ||
          !old_domain->analyzer.CanProveEqual(pushes[i].index, canonical_output) ||
          !old_domain->analyzer.CanProve(pulls[i].source_index >= make_zero(pulls[i].source_index.dtype())) ||
          !old_domain->analyzer.CanProve(pulls[i].source_index < pulls[i].remote->shape[0]) ||
          !old_domain->analyzer.CanProve(pushes[i].index >= make_zero(pushes[i].index.dtype())) ||
          !old_domain->analyzer.CanProve(pushes[i].index < pushes[i].remote->shape[0])) return std::nullopt;
    }

    Var new_i("ipc_ar_half8_i", DataType::Int(32));
    const Var& tx = opaque_let->var;
    std::vector<PrimExpr> source0(peer_count_);
    std::vector<PrimExpr> output0(peer_count_);
    for (int i = 0; i < peer_count_; ++i) {
      source0[i] = SubstituteOldCoordinates(pulls[i].source_index, old_for->loop_var, tx, new_i, tx, 0);
      output0[i] = SubstituteOldCoordinates(pushes[i].index, old_for->loop_var, tx, new_i, tx, 0);
    }
    Analyzer analyzer;
    analyzer.Bind(new_i, Range::FromMinExtent(0, 16));
    analyzer.Bind(tx, Range::FromMinExtent(0, 512));
    auto contiguous_aligned_8 = [&](const PrimExpr& start, const PrimExpr& indexed) {
      if (!analyzer.CanProveEqual(floormod(start, 8), 0)) return false;
      for (int lane = 0; lane < 8; ++lane) {
        const PrimExpr expected = start + make_const(start.dtype(), lane);
        const PrimExpr actual = SubstituteOldCoordinates(indexed, old_for->loop_var, tx, new_i, tx, lane);
        if (!analyzer.CanProveEqual(actual, expected)) return false;
      }
      return true;
    };
    for (int i = 0; i < peer_count_; ++i) {
      if (!contiguous_aligned_8(source0[i], pulls[i].source_index) ||
          !contiguous_aligned_8(output0[i], pushes[i].index)) return std::nullopt;
      source0[i] = analyzer.Simplify(source0[i]);
      output0[i] = analyzer.Simplify(output0[i]);
    }

    const DataType payloadx8 = pulls[0].remote->dtype.with_lanes(8), f32x8 = DataType::Float(32, 8);
    std::vector<PrimExpr> values(peer_count_);
    for (int i = 0; i < peer_count_; ++i) values[i] = BufferLoad(pulls[i].remote, {Ramp(source0[i], 1, 8)});
    PrimExpr sum = Cast(f32x8, values[0]);
    for (int i = 1; i < peer_count_; ++i) sum = sum + Cast(f32x8, values[i]);
    // Bind the rounded vector once.  P BufferStores deliberately share the
    // same value, but TIR expression sharing alone is not a codegen contract;
    // this Let prevents each output push from rematerializing the P pulls
    // and FP32 accumulation.
    Var reduced_value("ipc_ar_reduced_f16x8", payloadx8);
    Array<Stmt> stores;
    for (int i = 0; i < peer_count_; ++i)
      stores.push_back(BufferStore(pushes[i].remote, reduced_value, {Ramp(output0[i], 1, 8)}));
    Stmt vector_body = LetStmt(reduced_value, Cast(payloadx8, sum), SeqStmt::Flatten(stores));
    Stmt rewritten = For(new_i, 0, 16, ForKind::kSerial, vector_body);
    for (int i = 2 * peer_count_ - 1; i >= 0; --i)
      rewritten = LetStmt(ptr_lets[i]->var, ptr_lets[i]->value, rewritten);
    Array<Stmt> opaque_children{rewritten};
    for (size_t i = 1; i < opaque_seq->seq.size(); ++i) opaque_children.push_back(opaque_seq->seq[i]);
    return LetStmt(opaque_let->var, opaque_let->value, SeqStmt::Flatten(opaque_children));
  }

  std::optional<Stmt> TryRewriteOpaqueScope(const LetStmtNode* opaque_let) {
    if (auto ag_rewrite = TryRewriteAGOpaqueScope(opaque_let)) return ag_rewrite;
    if (auto ar_rewrite = TryRewriteAROpaqueScope(opaque_let)) return ar_rewrite;
    auto reject = [](const char*) -> std::optional<Stmt> {
      return std::nullopt;
    };
    // The communication tile scope also owns the following fence/barrier, so
    // the pointer chain is its first SeqStmt child rather than the direct
    // body of opaque_thread_id.  Only rewrite that first child and retain all
    // later siblings byte-for-byte in their original lexical order.
    const auto* opaque_seq = opaque_let->body.as<SeqStmtNode>();
    if (!opaque_seq || opaque_seq->seq.empty()) return reject("opaque scope sequence");
    std::vector<const LetStmtNode*> ptr_lets(peer_count_);
    Stmt cursor = opaque_seq->seq[0];
    for (int i = 0; i < peer_count_; ++i) {
      ptr_lets[i] = cursor.as<LetStmtNode>();
      if (!ptr_lets[i] || !IsRemoteAddress(ptr_lets[i]->value)) return reject("remote lets");
      cursor = ptr_lets[i]->body;
    }
    const auto* old_for = cursor.as<ForNode>();
    if (!old_for || old_for->kind != ForKind::kSerial || old_for->thread_binding.defined() ||
        !old_for->HasTrivialStep() || !IsZero(old_for->min) ||
        ConstInt(old_for->extent) != std::optional<int64_t>(128) || !old_for->annotations.empty()) return reject("element for");
    const std::vector<Stmt> seq = FlattenSeq(old_for->body);
    if (seq.size() != static_cast<size_t>(peer_count_ + 1)) return reject("element sequence");
    std::vector<ScalarPull> pulls(peer_count_);
    for (int i = 0; i < peer_count_; ++i) {
      // Padding makes generalized RS source predicates simplify to true.  It
      // is safe to accept only the existing direct-pull normal form in that
      // case; mixed conditional/direct pulls still fail the predicate audit.
      if (!MatchScalarPull(seq[i], &pulls[i]) && !MatchARScalarPull(seq[i], &pulls[i]))
        return reject("scalar pull");
    }
    Buffer out; PrimExpr out_index;
    if (!MatchReduceStore(seq[peer_count_], pulls, &out, &out_index)) return reject("reduce store");
    for (int i = 1; i < peer_count_; ++i)
      if (!ExprDeepEqual()(pulls[0].predicate, pulls[i].predicate)) return reject("predicate mismatch");
    if (UsesVar(pulls[0].predicate, [&](const VarNode* v) {
          return v == old_for->loop_var.get() || v == opaque_let->var.get();
        })) return reject("predicate depends element");

    std::unordered_set<const VarNode*> data;
    std::unordered_set<int64_t> peers;
    for (int i = 0; i < peer_count_; ++i) {
      if (!IsFlat16BitFloatBuffer(pulls[i].remote) ||
          !data.insert(pulls[i].remote->data.get()).second ||
          !RemoteAddressMatches(ptr_lets[i], pulls[i].remote, abi_, peer_count_, &peers) ||
          usages_.loads[pulls[i].temp->data.get()] != 1 || usages_.stores[pulls[i].temp->data.get()] != 1 ||
          usages_.bare_vars[pulls[i].temp->data.get()] != 0)
        return reject("buffer or temp audit");
    }
    if (!IsFlat16BitFloatBuffer(out) || !IsParameterBuffer(out) ||
        !data.insert(out->data.get()).second) return reject("output audit");

    for (int i = 0; i < peer_count_; ++i) {
      if (pulls[i].temp->dtype != pulls[0].temp->dtype || pulls[i].remote->dtype != pulls[0].remote->dtype)
        return reject("payload dtype");
      for (int j = i + 1; j < peer_count_; ++j)
        if (pulls[i].temp->data.same_as(pulls[j].temp->data)) return reject("temp identity");
    }
    if (out->dtype != pulls[0].temp->dtype) return reject("output dtype");
    // Establish the only mapping this pass accepts.  The source is a flat
    // old-element linear address.  The output is the scheduled two-row
    // row-major scatter written as the same old linear coordinate.  This is
    // stronger than merely observing two adjacent samples: it proves every
    // reassigned store has exactly one old writer in the 128*512 domain.
    auto old_domain = MakeOuterProofDomain();
    if (!old_domain || !old_domain->BindRange(old_for->loop_var, make_zero(old_for->loop_var.dtype()),
                                               make_const(old_for->loop_var.dtype(), 128)) ||
        !old_domain->BindRange(opaque_let->var, make_zero(opaque_let->var.dtype()),
                               make_const(opaque_let->var.dtype(), 512))) {
      return reject("outer range replay");
    }
    const PrimExpr zero_i = make_zero(old_for->loop_var.dtype());
    const PrimExpr zero_tx = make_zero(opaque_let->var.dtype());
    const PrimExpr linear = old_for->loop_var * make_const(old_for->loop_var.dtype(), 512) +
                            opaque_let->var;
    const PrimExpr out_base = TwoVarSubstituter(old_for->loop_var, zero_i, opaque_let->var, zero_tx)(out_index);
    const PrimExpr out_i_delta = old_domain->analyzer.Simplify(
        TwoVarSubstituter(old_for->loop_var, make_const(old_for->loop_var.dtype(), 1),
                          opaque_let->var, zero_tx)(out_index) - out_base);
    const auto two_row_delta = ConstInt(out_i_delta);
    if (!two_row_delta.has_value() || *two_row_delta < 512 || (*two_row_delta & 1)) return reject("row stride");
    const PrimExpr row_stride = make_const(linear.dtype(), *two_row_delta / 2);
    const PrimExpr canonical_out = out_base + floordiv(linear, make_const(linear.dtype(), 256)) * row_stride +
                                  floormod(linear, make_const(linear.dtype(), 256));
    if (!IsUniformControlExpr(pulls[0].predicate) || !IsUniformControlExpr(out_base) ||
        !old_domain->analyzer.CanProveEqual(out_index, canonical_out) ||
        !old_domain->analyzer.CanProve(out_index >= make_zero(out_index.dtype())) ||
        !old_domain->analyzer.CanProve(out_index < out->shape[0])) return reject("output uniform/canonical/range");
    for (int i = 0; i < peer_count_; ++i) {
      const PrimExpr source_base = TwoVarSubstituter(old_for->loop_var, zero_i, opaque_let->var, zero_tx)(pulls[i].source_index);
      const PrimExpr canonical_source = source_base + linear;
      if (!IsUniformControlExpr(source_base) ||
          !old_domain->analyzer.CanProveEqual(pulls[i].source_index, canonical_source)) return reject("source uniform/canonical");
      tvm::With<ConstraintContext> predicate_scope(&old_domain->analyzer, pulls[i].predicate);
      if (!old_domain->analyzer.CanProve(pulls[i].source_index >= make_zero(pulls[i].source_index.dtype())) ||
          !old_domain->analyzer.CanProve(pulls[i].source_index < pulls[i].remote->shape[0])) return reject("source range");
    }
    Var new_i("ipc_half8_i", DataType::Int(32));
    const Var& tx = opaque_let->var;
    std::vector<PrimExpr> index0(peer_count_);
    PrimExpr out0 = SubstituteOldCoordinates(out_index, old_for->loop_var, tx, new_i, tx, 0);
    for (int i = 0; i < peer_count_; ++i) {
      index0[i] = SubstituteOldCoordinates(pulls[i].source_index, old_for->loop_var, tx, new_i, tx, 0);
    }
    Analyzer analyzer;
    analyzer.Bind(new_i, Range::FromMinExtent(0, 16));
    analyzer.Bind(tx, Range::FromMinExtent(0, 512));
    auto contiguous_aligned_8 = [&](const PrimExpr& start, const PrimExpr& indexed) {
      if (!analyzer.CanProveEqual(floormod(start, 8), 0)) return false;
      for (int lane = 0; lane < 8; ++lane) {
        PrimExpr expected = start + make_const(start.dtype(), lane);
        PrimExpr actual = SubstituteOldCoordinates(indexed, old_for->loop_var, tx, new_i, tx, lane);
        if (!analyzer.CanProveEqual(actual, expected)) return false;
      }
      return true;
    };
    if (!contiguous_aligned_8(out0, out_index)) return reject("output contiguous/aligned");
    for (int i = 0; i < peer_count_; ++i)
      if (!contiguous_aligned_8(index0[i], pulls[i].source_index)) return reject("source contiguous/aligned");
    // Only simplify the newly constructed affine coordinates under the local
    // proof facts.  The opaque helper call remains the same Let binding.
    out0 = analyzer.Simplify(out0);
    for (int i = 0; i < peer_count_; ++i) index0[i] = analyzer.Simplify(index0[i]);

    const DataType payloadx8 = pulls[0].remote->dtype.with_lanes(8), f32x8 = DataType::Float(32, 8);
    std::vector<PrimExpr> v(peer_count_);
    for (int i = 0; i < peer_count_; ++i) v[i] = BufferLoad(pulls[i].remote, {Ramp(index0[i], 1, 8)});
    PrimExpr sum = Cast(f32x8, v[0]);
    for (int i = 1; i < peer_count_; ++i) sum = sum + Cast(f32x8, v[i]);
    const Array<PrimExpr> ramp_out{Ramp(out0, 1, 8)};
    Stmt then_store = BufferStore(out, Cast(payloadx8, sum), ramp_out);
    Stmt else_store = BufferStore(out, Broadcast(make_zero(out->dtype), 8), ramp_out);
    Stmt vector_body = IfThenElse(pulls[0].predicate, then_store, else_store);
    Stmt vector_for = For(new_i, 0, 16, ForKind::kSerial, vector_body);
    Stmt rewritten = vector_for;
    for (int i = peer_count_ - 1; i >= 0; --i)
      rewritten = LetStmt(ptr_lets[i]->var, ptr_lets[i]->value, rewritten);
    Array<Stmt> opaque_children{rewritten};
    for (size_t i = 1; i < opaque_seq->seq.size(); ++i) opaque_children.push_back(opaque_seq->seq[i]);
    return LetStmt(opaque_let->var, opaque_let->value, SeqStmt::Flatten(opaque_children));
  }

  const PrimFunc& func_;
  LocalTempUseCounter usages_;
  IPCABI abi_;
  const int peer_count_;
  std::optional<Var> thread_x_;
  bool thread_x_512_{false};
  bool nonunit_nonx_thread_{false};
  bool uniform_control_{true};
  std::unordered_set<const VarNode*> uniform_vars_;
  std::vector<RangeProofFact> proof_facts_;
};

PrimFunc VectorizeDCUIPCCommunication(PrimFunc func) {
  if (!IsVerifiedGfx936IPCFunction(func)) return func;
  auto abi = GetIPCABI(func);
  if (!abi.has_value()) return func;
  auto peer_count = GetIPCPeerCount(func);
  if (!peer_count.has_value()) return func;
  Stmt body = IPCVectorizer(func, std::move(*abi), *peer_count)(func->body);
  if (body.same_as(func->body)) return func;
  return PrimFunc(func->params, body, func->ret_type, func->buffer_map, func->attrs, func->span);
}

}  // namespace

namespace transform {
using namespace tir::transform;
tvm::transform::Pass VectorizeDCUIPCCommunication() {
  return CreatePrimFuncPass([](PrimFunc func, const IRModule&, PassContext) {
    return tl::VectorizeDCUIPCCommunication(std::move(func));
  }, 0, "tl.VectorizeDCUIPCCommunication", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VectorizeDCUIPCCommunication", VectorizeDCUIPCCommunication);
}
}  // namespace transform
}  // namespace tl
}  // namespace tvm
