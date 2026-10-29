#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/memory.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/dtype.h>

#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "inject_utils.h"
#include "../op/builtin.h"

using tvm::ffi::GetRef;
using tvm::ffi::make_object;

namespace tvm {
namespace tl {
using namespace tir;
using ffi::Array;
using ffi::String;

PrimExpr MakeElementDTypeArg(DataType dtype) {
    DataType elem_dtype = dtype;
    if (dtype.lanes() > 1) {
        elem_dtype = dtype.element_of();
    }
    return StringImm(ffi::DLDataTypeToString(elem_dtype));
}

struct CopyInfo {
    Buffer dst_buffer;
    Buffer src_buffer;
    Array<PrimExpr> dst_indices;
    Array<PrimExpr> src_indices;
    Stmt store_stmt;
    bool predicated{false};
    PrimExpr predicate;
    PrimExpr else_value;
};

bool IsZeroValue(const PrimExpr& value) {
    if (const auto* broadcast = value.as<BroadcastNode>()) {
        return IsZeroValue(broadcast->value);
    }
    if (const auto* cast = value.as<CastNode>()) {
        return IsZeroValue(cast->value);
    }
    if (const auto* f = value.as<FloatImmNode>()) {
        return f->value == 0.0f;
    }
    if (const auto* i = value.as<IntImmNode>()) {
        return i->value == 0;
    }
    return false;
}

struct CollectResult {
    std::vector<CopyInfo> copies;
    std::unordered_map<String, Var> global_to_res_var;
    // One resource base is selected from its first copy and must be shared by
    // every later copy from that Buffer.  The async-copy offset is relative to
    // this exact expression, never an independently filtered index.
    std::unordered_map<String, PrimExpr> global_to_base_offset;
    std::unordered_map<String, std::pair<Var, PrimExpr>> shared_alloc_to_binding;
    std::unordered_set<const tvm::tir::VarNode*> thread_vars;
    // Ranges are recorded as the collector descends into the copy site.  They
    // let arith::Analyzer prove that a nonzero resource-relative offset cannot
    // become negative; a variable name alone is not such a proof.
    std::vector<std::pair<Var, Range>> loop_var_ranges;
};

class VariableEliminator : public tvm::tir::ExprMutator {
 public:
  explicit VariableEliminator(const std::unordered_set<const tvm::tir::VarNode*>& vars)
      : vars_to_remove_(vars) {}

  PrimExpr VisitExpr_(const tvm::tir::VarNode* op) override {
    if (vars_to_remove_.count(op)) {
      return tvm::tir::make_zero(op->dtype);
    }
    return GetRef<PrimExpr>(op);
  }

 private:
  const std::unordered_set<const tvm::tir::VarNode*>& vars_to_remove_;
};

int GetElementBytes(DataType dtype) {
    DataType elem_dtype = dtype;
    if (dtype.lanes() > 1) {
        elem_dtype = dtype.element_of();
    }
    return elem_dtype.bytes();
}

PrimExpr ScaleToByteOffset(const PrimExpr& offset, int elem_bytes) {
    if (elem_bytes <= 1) {
        return offset;
    }
    return offset * make_const(offset.dtype(), elem_bytes);
}

PrimExpr GetScalarOffset(const PrimExpr& offset) {
    if (const auto* ramp = offset.as<RampNode>()) {
        return ramp->base;
    }
    return offset;
}

// Vector async copies use the ramp base as the resource offset.  Copy.Lower
// only accepts one flat global source index, so preserving this convention in
// both resource construction and its residual is essential.
PrimExpr FlattenIndices(const Array<PrimExpr>& idx) {
    if (idx.empty()) return IntImm(DataType::Int(32), 0);
    PrimExpr r = idx[0];
    for (size_t i = 1; i < idx.size(); ++i) r = r + idx[i];
    return GetScalarOffset(r);
}

void FinalizeResourceBases(CollectResult* result);

class FreeVarCollector : public tvm::tir::ExprVisitor {
 public:
  std::unordered_set<const tvm::tir::VarNode*> vars;

  void VisitExpr_(const tvm::tir::VarNode* op) final {
    vars.insert(op);
    ExprVisitor::VisitExpr_(op);
  }
};

bool IsConstantPrimExpr(const PrimExpr& expr) {
    FreeVarCollector collector;
    collector(expr);
    return collector.vars.empty();
}

// Split dst offset into runtime offset and compile-time smem template offset.
// smem_offset must be a simplified constant (no threadIdx / loop vars).
std::pair<PrimExpr, PrimExpr> SplitSmemOffset(const PrimExpr& offset) {
    tvm::arith::Analyzer analyzer;
    PrimExpr scalar = GetScalarOffset(offset);

    FreeVarCollector collector;
    collector(scalar);

    VariableEliminator eliminator(collector.vars);
    PrimExpr smem_offset = analyzer.Simplify(eliminator(scalar));
    PrimExpr runtime_offset = analyzer.Simplify(scalar - smem_offset);

    ICHECK(IsConstantPrimExpr(smem_offset))
        << "tl.dcu_async_copy smem_offset must be constant, got: " << smem_offset;

    return {GetScalarOffset(runtime_offset), GetScalarOffset(smem_offset)};
}

CollectResult CollectResources(const Stmt& body) {
    class Collector : public StmtExprVisitor {
    public:
        CollectResult result;

    private:
        bool in_async{false};
        std::unordered_set<const tvm::tir::VarNode*> loop_vars_;
        std::unordered_set<const tvm::tir::VarNode*> thread_vars_;
        bool IsSharedScope(const Buffer& buf) {
            auto s = buf.scope();
            return s == "shared" || s == "shared.dyn";
        }
        bool IsGlobalScope(const Buffer& buf) {
            auto s = buf.scope();
            return s == "global" || s == "";
        }

        void VisitStmt_(const AttrStmtNode* attr) override {
            if (attr->attr_key == tir::attr::thread_extent) {
                auto iv = attr->node.as<tvm::tir::IterVarNode>();
                const std::string& tag = iv->thread_tag;

                if (tag.find("threadIdx") != std::string::npos) {
                    tvm::tir::Var thread_var = iv->var;
                    loop_vars_.insert(thread_var.get());
                    thread_vars_.insert(thread_var.get());
                    result.thread_vars.insert(thread_var.get());
                    result.loop_var_ranges.emplace_back(
                        thread_var, Range::FromMinExtent(make_zero(thread_var.dtype()), attr->value));

                    StmtExprVisitor::VisitStmt_(attr);

                    loop_vars_.erase(thread_var.get());
                } else {
                    StmtExprVisitor::VisitStmt_(attr);
                }

            } else if (attr->attr_key == kDCUNeedLayoutStmt ||
                       attr->attr_key == kDCUNeedShadowStmt) {
                StmtExprVisitor::VisitStmt_(attr);
            } else if (attr->attr_key == tir::attr::async_scope) {
                ICHECK(in_async == false) << "Nested async scopes not supported";
                in_async = true;
                
                StmtExprVisitor::VisitStmt_(attr);
                
                in_async = false;
            }
            else {
                StmtExprVisitor::VisitStmt_(attr);
            }
        }

        void VisitStmt_(const ForNode* op) override {
            loop_vars_.insert(op->loop_var.get());
            result.loop_var_ranges.emplace_back(
                op->loop_var, Range::FromMinExtent(op->min, op->extent));
            StmtExprVisitor::VisitStmt_(op);
            loop_vars_.erase(op->loop_var.get());
        }

        static const BufferLoadNode* PeelGlobalLoadValue(const PrimExpr& v) {
            if (const auto* load = v.as<BufferLoadNode>()) {
                return load;
            }
            if (const auto* cast = v.as<CastNode>()) {
                return cast->value.as<BufferLoadNode>();
            }
            return nullptr;
        }

        bool TryCollectCopy(const BufferStoreNode* op, const BufferLoadNode* load,
                            bool predicated, const PrimExpr& predicate,
                            const PrimExpr& else_value) {
            Buffer dst = op->buffer;
            Buffer src = load->buffer;
            if (!IsGlobalScope(src)) {
                return false;
            }
            // LDS dest must be a consecutive 16B vector (cp_async_gs M0 + lane*N).
            // Source may gather (XOR swizzle / idxen); do not reject XOR on src.
            if (!IsDCUCpAsyncVectorCopy(op, load)) {
                return false;
            }

            CopyInfo info{dst,
                          src,
                          op->indices,
                          load->indices,
                          GetRef<Stmt>(op),
                          predicated,
                          predicate,
                          else_value};
            result.copies.push_back(info);

            if (result.global_to_res_var.find(src->name) == result.global_to_res_var.end()) {
                Var var(src->name + "_dcu_res", DataType::Int(32, 4));

                VariableEliminator eliminator(loop_vars_);
                tvm::arith::Analyzer resource_analyzer;
                Array<PrimExpr> base_indices;
                for (const auto& idx : load->indices) {
                    PrimExpr no_loops = eliminator(idx);
                    base_indices.push_back(resource_analyzer.Simplify(no_loops));
                }

                Array<PrimExpr> args;
                args.push_back(src->data);

                for (const auto& idx : base_indices) {
                    args.push_back(idx);
                }
                PrimExpr val =
                    Call(DataType::Int(32, 4), Op::Get("tl.make_dcu_resource"), args);

                result.global_to_res_var[src->name] = var;
                result.global_to_base_offset[src->name] =
                    GetScalarOffset(FlattenIndices(base_indices));
                result.shared_alloc_to_binding[src->name] = {var, val};
            }
            return true;
        }

        void VisitStmt_(const BufferStoreNode* op) final {
            if (IsSharedScope(op->buffer) && op->value.defined() && in_async) {
                const BufferLoadNode* load = nullptr;
                bool predicated = false;
                PrimExpr predicate;
                PrimExpr else_value;

                if (const auto* direct_load = PeelGlobalLoadValue(op->value)) {
                    load = direct_load;
                } else if (const auto* call = op->value.as<CallNode>()) {
                    if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3 &&
                        IsZeroValue(call->args[2])) {
                        load = PeelGlobalLoadValue(call->args[1]);
                        if (load) {
                            predicated = true;
                            predicate = call->args[0];
                            else_value = call->args[2];
                        }
                    }
                }

                if (load) {
                    TryCollectCopy(op, load, predicated, predicate, else_value);
                }
            }
            StmtExprVisitor::VisitStmt_(op);
        }
    };

    Collector col;
    col(body);
    FinalizeResourceBases(&col.result);
    return col.result;
}

void FinalizeResourceBases(CollectResult* result) {
    tvm::arith::Analyzer analyzer;
    for (const auto& binding : result->loop_var_ranges) {
        analyzer.Bind(binding.first, binding.second);
    }
    for (const auto& item : result->global_to_base_offset) {
        const String& name = item.first;
        const PrimExpr& candidate_base = item.second;
        bool keep_candidate = true;
        for (const CopyInfo& copy : result->copies) {
            if (copy.src_buffer->name != name) continue;
            PrimExpr original = FlattenIndices(copy.src_indices);
            PrimExpr residual = analyzer.Simplify(original - candidate_base);
            if (!analyzer.CanProve(residual >= make_zero(residual.dtype()))) {
                keep_candidate = false;
                break;
            }
        }
        if (keep_candidate) continue;

        // Every copy of this resource must make the same decision.  A zero
        // base is always legal: the runtime offset is the original flattened
        // global index, including any floor/mod carry or differing constant.
        result->global_to_base_offset[name] = make_zero(candidate_base.dtype());
        auto binding = result->shared_alloc_to_binding.find(name);
        ICHECK(binding != result->shared_alloc_to_binding.end());
        const auto* old_resource = binding->second.second.as<CallNode>();
        ICHECK(old_resource && old_resource->op.same_as(Op::Get("tl.make_dcu_resource")))
            << "expected cached tl.make_dcu_resource binding";
        ICHECK_EQ(old_resource->args.size(), 2U)
            << "DCU vector async copies require one flat source index";
        Array<PrimExpr> args{old_resource->args[0], make_zero(candidate_base.dtype())};
        binding->second.second = Call(old_resource->dtype, old_resource->op, args);
    }
}

class StoreReplacer : public StmtExprMutator {
public:
    static Stmt Run(Stmt body, const std::vector<CopyInfo>& copies,
                    const std::unordered_map<String, Var>& global_to_var,
                    const std::unordered_map<String, PrimExpr>& global_to_base_offset) {
        StoreReplacer replacer(copies, global_to_var, global_to_base_offset);
        return replacer(std::move(body));
    }

private:
    StoreReplacer(const std::vector<CopyInfo>& copies,
                  const std::unordered_map<String, Var>& global_to_var,
                  const std::unordered_map<String, PrimExpr>& global_to_base_offset)
        : copies_(copies), global_to_var_(global_to_var),
          global_to_base_offset_(global_to_base_offset) {}

    Stmt VisitStmt_(const AttrStmtNode *attr) {
        if (attr->attr_key == tir::attr::async_scope) {
            auto body = this->VisitStmt(attr->body);
            return body;
        }
        return StmtMutator::VisitStmt_(attr); 
    }


    Stmt VisitStmt_(const BufferStoreNode* op) final {
        for (const auto& copy : copies_) {
            if (copy.store_stmt.same_as(GetRef<Stmt>(op))) {
                Var src_res = global_to_var_.at(copy.src_buffer->name);
                PrimExpr dst_res = copy.dst_buffer->data;

                PrimExpr elem_dtype = MakeElementDTypeArg(copy.dst_buffer->dtype);
                const int elem_bytes = GetElementBytes(copy.src_buffer->dtype);
                PrimExpr dst_offset = FlattenIndices(copy.dst_indices);
                auto [thread_dst_off, smem_offset] = SplitSmemOffset(dst_offset);
                thread_dst_off = ScaleToByteOffset(thread_dst_off, elem_bytes);
                smem_offset = ScaleToByteOffset(smem_offset, elem_bytes);
                // ``make_dcu_resource`` is constructed once per source
                // Buffer.  Preserve the exact original source index and take
                // its residual from that cached base.  The old VariableKeeper
                // reconstruction zeroed block/let/constant terms separately;
                // that retained loop-only/constant pieces independently of
                // the cached base, incorrectly duplicating constants and
                // splitting floor-div/mod carries across block and loop terms.
                PrimExpr original_src_offset = FlattenIndices(copy.src_indices);
                PrimExpr src_base_offset = global_to_base_offset_.at(copy.src_buffer->name);
                tvm::arith::Analyzer analyzer;
                PrimExpr src_off = analyzer.Simplify(original_src_offset - src_base_offset);
                Stmt async_copy = Evaluate(
                    Call(DataType::Int(32), Op::Get("tl.dcu_async_copy"),
                         {dst_res, thread_dst_off, src_res, src_off, smem_offset, elem_dtype}));
                Stmt commit = Evaluate(
                    Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
                Stmt async_with_commit = SeqStmt({async_copy, commit});

                if (copy.predicated) {
                    return IfThenElse(copy.predicate, async_with_commit);
                }
                return async_with_commit;
            }
        }
        return StmtExprMutator::VisitStmt_(op);
    }

    const std::vector<CopyInfo>& copies_;
    const std::unordered_map<String, Var>& global_to_var_;
    const std::unordered_map<String, PrimExpr>& global_to_base_offset_;
};

class DCUAsyncCopySmemOffsetFixer : public StmtExprMutator {
 public:
  static Stmt Run(Stmt body) { return DCUAsyncCopySmemOffsetFixer()(std::move(body)); }

 private:
  arith::Analyzer analyzer_;

  PrimExpr FixCall(const CallNode* call) {
    if (!call->op.same_as(Op::Get("tl.dcu_async_copy")) || call->args.size() < 6) {
      return GetRef<PrimExpr>(call);
    }
    PrimExpr smem_off = call->args[4];
    if (IsConstantPrimExpr(smem_off)) {
      return GetRef<PrimExpr>(call);
    }

    PrimExpr total = analyzer_.Simplify(call->args[1] + smem_off);
    auto [runtime_off, const_off] = SplitSmemOffset(total);
    Array<PrimExpr> new_args = call->args;
    new_args.Set(1, runtime_off);
    new_args.Set(4, const_off);
    return Call(call->dtype, call->op, new_args);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      PrimExpr fixed = FixCall(call);
      if (!fixed.same_as(op->value)) {
        return Evaluate(fixed);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    return FixCall(op);
  }
};

class UsedVarCollector : public ExprVisitor {
 public:
  std::unordered_set<const VarNode*> vars;
  void VisitExpr_(const VarNode* op) final { vars.insert(op); }
};

class LetVarCollector : public StmtVisitor {
 public:
  std::unordered_set<const VarNode*> vars;
  void VisitStmt_(const LetStmtNode* op) final {
    vars.insert(op->var.get());
    StmtVisitor::VisitStmt_(op);
  }
};

Stmt WrapWithDCUResourceLets(
    Stmt body, const std::unordered_map<String, std::pair<Var, PrimExpr>>& bindings) {
  Stmt new_stmt = std::move(body);
  for (const auto& item : bindings) {
    new_stmt = LetStmt(item.second.first, item.second.second, new_stmt);
  }
  return new_stmt;
}

// Insert *_dcu_res Lets after LetStmts they depend on (e.g. q_start_idx / k_start_idx).
Stmt InsertDCUResourceLetsAfterDeps(
    Stmt stmt, std::unordered_set<const VarNode*> pending,
    const std::unordered_map<String, std::pair<Var, PrimExpr>>& bindings) {
  if (const auto* let = stmt.as<LetStmtNode>()) {
    pending.erase(let->var.get());
    Stmt new_body = InsertDCUResourceLetsAfterDeps(let->body, std::move(pending), bindings);
    if (new_body.same_as(let->body)) {
      return stmt;
    }
    return LetStmt(let->var, let->value, new_body);
  }
  if (pending.empty()) {
    return WrapWithDCUResourceLets(std::move(stmt), bindings);
  }
  if (const auto* alloc = stmt.as<AllocateNode>()) {
    Stmt new_body = InsertDCUResourceLetsAfterDeps(alloc->body, std::move(pending), bindings);
    if (new_body.same_as(alloc->body)) {
      return stmt;
    }
    return Allocate(alloc->buffer_var, alloc->dtype, alloc->extents, alloc->condition, new_body,
                    alloc->annotations);
  }
  if (const auto* decl = stmt.as<DeclBufferNode>()) {
    Stmt new_body = InsertDCUResourceLetsAfterDeps(decl->body, std::move(pending), bindings);
    if (new_body.same_as(decl->body)) {
      return stmt;
    }
    return DeclBuffer(decl->buffer, new_body);
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    Stmt new_body = InsertDCUResourceLetsAfterDeps(attr->body, std::move(pending), bindings);
    if (new_body.same_as(attr->body)) {
      return stmt;
    }
    return AttrStmt(attr->node, attr->attr_key, attr->value, new_body);
  }
  if (const auto* asrt = stmt.as<AssertStmtNode>()) {
    Stmt new_body = InsertDCUResourceLetsAfterDeps(asrt->body, std::move(pending), bindings);
    if (new_body.same_as(asrt->body)) {
      return stmt;
    }
    return AssertStmt(asrt->condition, asrt->message, new_body);
  }
  return WrapWithDCUResourceLets(std::move(stmt), bindings);
}

Stmt InjectDCUResources(Stmt body,
                        const std::unordered_map<String, std::pair<Var, PrimExpr>>& bindings) {
    if (bindings.empty()) {
        return body;
    }

    UsedVarCollector used;
    for (const auto& item : bindings) {
      used(item.second.second);
    }
    LetVarCollector lets;
    lets(body);
    std::unordered_set<const VarNode*> pending;
    for (const VarNode* v : used.vars) {
      if (lets.vars.count(v)) {
        pending.insert(v);
      }
    }

    return InjectAtDCUTarget(body, [&](const Stmt& target_body) {
        return InsertDCUResourceLetsAfterDeps(target_body, pending, bindings);
    });
}

// ``tl.dcu_persistent_tile`` is deliberately an opt-in contract.  A normal
// GEMM continues through the historical path below byte-for-byte; this code
// only handles the body of a serial output-tile traversal explicitly marked by
// the frontend.  In particular, this must not become another heuristic for
// recognising arbitrary serial loops (the K loop is also serial).
constexpr const char* kDCUPersistentTileAnnotation = "tl.dcu_persistent_tile";
// This is an internal proof token, not a frontend annotation.  It is emitted
// only after the persistent collector has placed tile-local resources and
// proved every async-copy residual local/non-negative.
constexpr const char* kDCUPersistentResourceVerifiedAnnotation =
    "tl.dcu_persistent_resource_verified";

bool HasDCUPersistentTileAnnotation(const ForNode* op) {
  auto value = op->annotations.Get(kDCUPersistentTileAnnotation);
  if (!value.has_value()) {
    return false;
  }
  // Python's ordinary ``annotations={...: 1}`` is an ffi scalar, whereas
  // pipeline annotations commonly arrive as a TIR IntImm.  Accept exactly
  // those two integer representations, and no truthy/string substitute.
  if (auto scalar = value.value().as<int64_t>()) {
    ICHECK_EQ(scalar.value(), 1)
        << kDCUPersistentTileAnnotation << " must have the integer value 1";
    return true;
  }
  const auto* imm = value.value().as<IntImmNode>();
  ICHECK(imm != nullptr && imm->value == 1)
      << kDCUPersistentTileAnnotation << " must have the integer value 1";
  return true;
}

// The ordinary collector is intentionally keyed by source name for legacy
// compatibility.  Persistent resources are lexically scoped at each marked
// traversal, so keys must instead use the Buffer data Var identity: two
// different buffers (or two nested tile traversals) are not interchangeable
// merely because their printed names match.
using PersistentSourceKey = const VarNode*;

struct PersistentCollectResult {
  std::vector<CopyInfo> copies;
  std::unordered_map<PersistentSourceKey, Var> global_to_res_var;
  std::unordered_map<PersistentSourceKey, PrimExpr> global_to_base_offset;
  std::unordered_map<PersistentSourceKey, std::pair<Var, PrimExpr>> bindings;
  // Store identity -> the exact Analyzer-simplified residual that passed the
  // tile/block independence and non-negative proof.  The replacer must not
  // independently simplify it with an unbound Analyzer.
  std::unordered_map<const StmtNode*, PrimExpr> verified_residuals;
  std::vector<PersistentSourceKey> binding_order;
  std::vector<std::pair<Var, Range>> local_var_ranges;
  std::unordered_set<const VarNode*> local_vars;
  bool has_tile_dependent_base{false};
};

class PersistentLetDefinitions : public StmtVisitor {
 public:
  std::unordered_map<const VarNode*, PrimExpr> definitions;

 private:
  void VisitStmt_(const LetStmtNode* op) final {
    definitions.emplace(op->var.get(), op->value);
    StmtVisitor::VisitStmt_(op);
  }
};

// A resource Let is deliberately inserted near the beginning of the marked
// tile body.  A source pointer allocated/declared *inside* that body would not
// dominate that insertion point.  This is not a supported persistent GEMM
// shape; reject it explicitly instead of accidentally capturing a same-named
// pointer from another scope.
class PersistentInnerBufferScopeCollector : public StmtVisitor {
 public:
  std::unordered_set<const VarNode*> defined_data_vars;

 private:
  void VisitStmt_(const AllocateNode* op) final {
    defined_data_vars.insert(op->buffer_var.get());
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode* op) final {
    defined_data_vars.insert(op->buffer->data.get());
    StmtVisitor::VisitStmt_(op);
  }
};

// Check transitive dependencies through uniform Let bindings without replacing
// those Lets in the actual resource expression.  Keeping the original Let Var
// lets the resource be placed after that binding and before the K loop.
class PersistentDependencyChecker : public ExprVisitor {
 public:
  PersistentDependencyChecker(
      const std::unordered_map<const VarNode*, PrimExpr>& definitions,
      const std::unordered_set<const VarNode*>& targets)
      : definitions_(definitions), targets_(targets) {}

  bool depends{false};
  bool impure{false};

 private:
  void VisitExpr_(const VarNode* op) final {
    if (targets_.count(op)) {
      depends = true;
      return;
    }
    auto it = definitions_.find(op);
    if (it != definitions_.end() && expanding_.insert(op).second) {
      VisitExpr(it->second);
      expanding_.erase(op);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    impure = true;
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    // The resource address must be an arithmetic expression, not a load or
    // effectful/helper call evaluated once per outer tile.
    impure = true;
    ExprVisitor::VisitExpr_(op);
  }

  const std::unordered_map<const VarNode*, PrimExpr>& definitions_;
  const std::unordered_set<const VarNode*>& targets_;
  std::unordered_set<const VarNode*> expanding_;
};

bool DependsOnPersistentVars(
    const PrimExpr& expr,
    const std::unordered_map<const VarNode*, PrimExpr>& definitions,
    const std::unordered_set<const VarNode*>& vars) {
  PersistentDependencyChecker checker(definitions, vars);
  checker(expr);
  return checker.depends;
}

bool IsPurePersistentAddress(
    const PrimExpr& expr,
    const std::unordered_map<const VarNode*, PrimExpr>& definitions) {
  const std::unordered_set<const VarNode*> no_targets;
  PersistentDependencyChecker checker(definitions, no_targets);
  checker(expr);
  return !checker.impure;
}

bool IsUnsafePersistentControlExpr(
    const PrimExpr& expr,
    const std::unordered_map<const VarNode*, PrimExpr>& definitions,
    const std::unordered_set<const VarNode*>& thread_vars) {
  PersistentDependencyChecker checker(definitions, thread_vars);
  checker(expr);
  // An opaque call/load in a control expression may conceal divergent state.
  // This opt-in path deliberately declines such a scope rather than assuming
  // that its resource construction is CTA-uniform.
  return checker.depends || checker.impure;
}

class PersistentResourceCollector : public StmtExprVisitor {
 public:
  PersistentResourceCollector(
      Var tile_var, std::unordered_set<const VarNode*> outer_thread_vars,
      std::unordered_set<const VarNode*> outer_block_vars,
      std::unordered_map<const VarNode*, PrimExpr> let_definitions,
      std::vector<std::pair<Var, Range>> outer_thread_ranges)
      : tile_var_(std::move(tile_var)), outer_thread_vars_(std::move(outer_thread_vars)),
        outer_block_vars_(std::move(outer_block_vars)), let_definitions_(std::move(let_definitions)) {
    result.local_var_ranges = std::move(outer_thread_ranges);
    result.local_vars.insert(outer_thread_vars_.begin(), outer_thread_vars_.end());
  }

  PersistentCollectResult result;

 private:
  bool in_async_{false};
  Var tile_var_;
  std::unordered_set<const VarNode*> outer_thread_vars_;
  std::unordered_set<const VarNode*> outer_block_vars_;
  std::unordered_map<const VarNode*, PrimExpr> let_definitions_;

  static bool IsSharedScope(const Buffer& buf) {
    const auto scope = buf.scope();
    return scope == "shared" || scope == "shared.dyn";
  }

  static bool IsGlobalScope(const Buffer& buf) {
    const auto scope = buf.scope();
    return scope == "global" || scope == "";
  }

  static const BufferLoadNode* PeelGlobalLoadValue(const PrimExpr& value) {
    if (const auto* load = value.as<BufferLoadNode>()) {
      return load;
    }
    if (const auto* cast = value.as<CastNode>()) {
      return cast->value.as<BufferLoadNode>();
    }
    return nullptr;
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::async_scope) {
      ICHECK(!in_async_) << "nested async scopes are unsupported in "
                         << kDCUPersistentTileAnnotation;
      in_async_ = true;
      StmtExprVisitor::VisitStmt_(op);
      in_async_ = false;
      return;
    }
    if (op->attr_key == tir::attr::thread_extent) {
      if (const auto* iv = op->node.as<IterVarNode>()) {
        const std::string tag = iv->thread_tag;
        if (tag.find("threadIdx") != std::string::npos) {
          result.local_vars.insert(iv->var.get());
          result.local_var_ranges.emplace_back(
              iv->var, Range::FromMinExtent(make_zero(iv->var.dtype()), op->value));
          StmtExprVisitor::VisitStmt_(op);
          result.local_vars.erase(iv->var.get());
          return;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode* op) final {
    ICHECK(!HasDCUPersistentTileAnnotation(op))
        << "nested " << kDCUPersistentTileAnnotation << " loops are unsupported";
    result.local_vars.insert(op->loop_var.get());
    result.local_var_ranges.emplace_back(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    StmtExprVisitor::VisitStmt_(op);
    result.local_vars.erase(op->loop_var.get());
  }

  void AddCopy(const BufferStoreNode* store, const BufferLoadNode* load,
               bool predicated, const PrimExpr& predicate, const PrimExpr& else_value) {
    if (!IsGlobalScope(load->buffer) || !IsDCUCpAsyncVectorCopy(store, load)) {
      return;
    }
    CopyInfo copy{store->buffer, load->buffer, store->indices, load->indices,
                  GetRef<Stmt>(store), predicated, predicate, else_value};
    result.copies.push_back(copy);

    PersistentSourceKey key = load->buffer->data.get();
    if (result.global_to_res_var.count(key)) {
      return;
    }

    VariableEliminator eliminate_local(result.local_vars);
    arith::Analyzer analyzer;
    PrimExpr original = FlattenIndices(load->indices);
    PrimExpr candidate = analyzer.Simplify(eliminate_local(original));

    // At least one operand must genuinely vary with the marked output tile.
    // Do not impose that condition per source: for an M-only traversal B may
    // validly retain a constant resource base while A changes per tile.
    std::unordered_set<const VarNode*> tile_target{tile_var_.get()};
    result.has_tile_dependent_base |=
        DependsOnPersistentVars(candidate, let_definitions_, tile_target);
    ICHECK(IsPurePersistentAddress(candidate, let_definitions_))
        << kDCUPersistentTileAnnotation << " resource base for " << load->buffer->name
        << " contains a load or call";

    std::unordered_set<const VarNode*> forbidden = outer_thread_vars_;
    forbidden.insert(result.local_vars.begin(), result.local_vars.end());
    ICHECK(!DependsOnPersistentVars(candidate, let_definitions_, forbidden))
        << kDCUPersistentTileAnnotation << " resource base for " << load->buffer->name
        << " depends on thread-local or inner-loop state";

    Var resource(load->buffer->name + "_dcu_res", DataType::Int(32, 4));
    PrimExpr value = Call(DataType::Int(32, 4), Op::Get("tl.make_dcu_resource"),
                          {load->buffer->data, candidate});
    result.global_to_res_var.emplace(key, resource);
    result.global_to_base_offset.emplace(key, candidate);
    result.bindings.emplace(key, std::make_pair(resource, value));
    result.binding_order.push_back(key);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    if (in_async_ && IsSharedScope(op->buffer) && op->value.defined()) {
      const BufferLoadNode* load = nullptr;
      bool predicated = false;
      PrimExpr predicate;
      PrimExpr else_value;
      if (const auto* direct = PeelGlobalLoadValue(op->value)) {
        load = direct;
      } else if (const auto* call = op->value.as<CallNode>()) {
        if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3 &&
            IsZeroValue(call->args[2])) {
          load = PeelGlobalLoadValue(call->args[1]);
          if (load != nullptr) {
            predicated = true;
            predicate = call->args[0];
            else_value = call->args[2];
          }
        }
      }
      if (load != nullptr) {
        AddCopy(op, load, predicated, predicate, else_value);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

void FinalizePersistentResourceBases(
    PersistentCollectResult* result, Var tile_var,
    const std::unordered_set<const VarNode*>& outer_block_vars,
    const std::unordered_map<const VarNode*, PrimExpr>& let_definitions,
    Range tile_range) {
  arith::Analyzer analyzer;
  analyzer.Bind(tile_var, tile_range);
  for (const auto& range : result->local_var_ranges) {
    analyzer.Bind(range.first, range.second);
  }
  std::unordered_set<const VarNode*> outer_tile_or_block = outer_block_vars;
  outer_tile_or_block.insert(tile_var.get());

  for (const auto& entry : result->global_to_base_offset) {
    const PersistentSourceKey key = entry.first;
    const PrimExpr& base = entry.second;
    for (const CopyInfo& copy : result->copies) {
      if (copy.src_buffer->data.get() != key) {
        continue;
      }
      PrimExpr original = FlattenIndices(copy.src_indices);
      PrimExpr residual = analyzer.Simplify(original - base);
      ICHECK(!DependsOnPersistentVars(residual, let_definitions, outer_tile_or_block))
          << kDCUPersistentTileAnnotation << " cannot form a pure local offset for "
          << copy.src_buffer->name << ": " << residual;
      ICHECK(analyzer.CanProve(residual >= make_zero(residual.dtype())))
          << kDCUPersistentTileAnnotation << " cannot prove non-negative local offset for "
          << copy.src_buffer->name << ": " << residual;
      result->verified_residuals.emplace(copy.store_stmt.get(), residual);
    }
  }
}

class PersistentStoreReplacer : public StmtExprMutator {
 public:
  static Stmt Run(Stmt body, const std::vector<CopyInfo>& copies,
                  const std::unordered_map<PersistentSourceKey, Var>& resources,
                  const std::unordered_map<const StmtNode*, PrimExpr>& verified_residuals) {
    return PersistentStoreReplacer(copies, resources, verified_residuals)(std::move(body));
  }

 private:
  PersistentStoreReplacer(const std::vector<CopyInfo>& copies,
                          const std::unordered_map<PersistentSourceKey, Var>& resources,
                          const std::unordered_map<const StmtNode*, PrimExpr>& verified_residuals)
      : copies_(copies), resources_(resources), verified_residuals_(verified_residuals) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::async_scope) {
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    for (const CopyInfo& copy : copies_) {
      if (!copy.store_stmt.same_as(GetRef<Stmt>(op))) {
        continue;
      }
      const PersistentSourceKey key = copy.src_buffer->data.get();
      const Var src_resource = resources_.at(key);
      const int elem_bytes = GetElementBytes(copy.src_buffer->dtype);
      auto [thread_dst_off, smem_offset] = SplitSmemOffset(FlattenIndices(copy.dst_indices));
      thread_dst_off = ScaleToByteOffset(thread_dst_off, elem_bytes);
      smem_offset = ScaleToByteOffset(smem_offset, elem_bytes);
      PrimExpr src_off = verified_residuals_.at(copy.store_stmt.get());
      Stmt async_copy = Evaluate(Call(
          DataType::Int(32), Op::Get("tl.dcu_async_copy"),
          {copy.dst_buffer->data, thread_dst_off, src_resource, src_off, smem_offset,
           MakeElementDTypeArg(copy.dst_buffer->dtype)}));
      Stmt with_commit = SeqStmt({async_copy,
                                  Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}))});
      return copy.predicated ? IfThenElse(copy.predicate, with_commit) : with_commit;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  const std::vector<CopyInfo>& copies_;
  const std::unordered_map<PersistentSourceKey, Var>& resources_;
  const std::unordered_map<const StmtNode*, PrimExpr>& verified_residuals_;
};

std::vector<std::pair<Var, PrimExpr>> PersistentBindingsInCopyOrder(
    const PersistentCollectResult& result) {
  std::vector<std::pair<Var, PrimExpr>> bindings;
  bindings.reserve(result.binding_order.size());
  for (PersistentSourceKey key : result.binding_order) {
    bindings.push_back(result.bindings.at(key));
  }
  return bindings;
}

Stmt WrapWithPersistentResourceLets(Stmt body,
                                    const std::vector<std::pair<Var, PrimExpr>>& bindings) {
  Stmt wrapped = std::move(body);
  // Reverse nesting keeps the emitted Let order equal to first-copy order.
  for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
    wrapped = LetStmt(it->first, it->second, wrapped);
  }
  return wrapped;
}

Stmt InsertPersistentResourceLetsAfterDeps(
    Stmt stmt, std::unordered_set<const VarNode*> pending,
    const std::vector<std::pair<Var, PrimExpr>>& bindings) {
  if (const auto* let = stmt.as<LetStmtNode>()) {
    pending.erase(let->var.get());
    return LetStmt(let->var, let->value,
                   InsertPersistentResourceLetsAfterDeps(let->body, std::move(pending), bindings));
  }
  if (pending.empty()) {
    return WrapWithPersistentResourceLets(std::move(stmt), bindings);
  }
  if (const auto* alloc = stmt.as<AllocateNode>()) {
    return Allocate(alloc->buffer_var, alloc->dtype, alloc->extents, alloc->condition,
                    InsertPersistentResourceLetsAfterDeps(alloc->body, std::move(pending), bindings),
                    alloc->annotations);
  }
  if (const auto* decl = stmt.as<DeclBufferNode>()) {
    return DeclBuffer(decl->buffer,
                      InsertPersistentResourceLetsAfterDeps(decl->body, std::move(pending), bindings));
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    return AttrStmt(attr->node, attr->attr_key, attr->value,
                    InsertPersistentResourceLetsAfterDeps(attr->body, std::move(pending), bindings));
  }
  if (const auto* asrt = stmt.as<AssertStmtNode>()) {
    return AssertStmt(asrt->condition, asrt->message,
                      InsertPersistentResourceLetsAfterDeps(asrt->body, std::move(pending), bindings));
  }
  LOG(FATAL) << kDCUPersistentTileAnnotation
             << " cannot place resource Lets after their tile-coordinate dependencies";
  return stmt;
}

Stmt InjectPersistentResourceLets(Stmt body, const PersistentCollectResult& result) {
  auto bindings = PersistentBindingsInCopyOrder(result);
  UsedVarCollector used;
  for (const auto& item : bindings) {
    used(item.second);
  }
  LetVarCollector lets;
  lets(body);
  std::unordered_set<const VarNode*> pending;
  for (const VarNode* var : used.vars) {
    if (lets.vars.count(var)) {
      pending.insert(var);
    }
  }
  return InsertPersistentResourceLetsAfterDeps(std::move(body), std::move(pending), bindings);
}

class PersistentResourceRewriter : public StmtExprMutator {
 public:
  static Stmt Run(Stmt body) { return PersistentResourceRewriter()(std::move(body)); }

 private:
  std::unordered_set<const VarNode*> active_thread_vars_;
  std::unordered_set<const VarNode*> active_block_vars_;
  std::vector<std::pair<Var, Range>> active_thread_ranges_;
  std::unordered_map<const VarNode*, PrimExpr> enclosing_let_definitions_;
  bool in_thread_dependent_scope_{false};

  bool IsExplicitSingleIterationPersistentLoop(const ForNode* op) const {
    // The exception is intentionally syntactic: accepting a dynamic 0/1
    // traversal leaves a runtime outer loop that can lengthen the lifetime of
    // the GEMM epilogue addresses.  Do not use enclosing facts or a general
    // Analyzer here; only an explicit `for ... in T.serial(1)` is eligible.
    const auto* min = op->min.as<IntImmNode>();
    const auto* extent = op->extent.as<IntImmNode>();
    return min != nullptr && extent != nullptr && min->value == 0 && extent->value == 1;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      if (const auto* iv = op->node.as<IterVarNode>()) {
        const std::string tag = iv->thread_tag;
        const bool is_thread = tag.find("threadIdx") != std::string::npos;
        const bool is_block = tag.find("blockIdx") != std::string::npos;
        if (is_thread) {
          active_thread_vars_.insert(iv->var.get());
          active_thread_ranges_.emplace_back(
              iv->var, Range::FromMinExtent(make_zero(iv->var.dtype()), op->value));
        }
        if (is_block) active_block_vars_.insert(iv->var.get());
        Stmt body = VisitStmt(op->body);
        if (is_thread) {
          active_thread_vars_.erase(iv->var.get());
          active_thread_ranges_.pop_back();
        }
        if (is_block) active_block_vars_.erase(iv->var.get());
        return AttrStmt(op->node, op->attr_key, op->value, body);
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    const bool thread_dependent = DependsOnPersistentVars(
        op->value, enclosing_let_definitions_, active_thread_vars_);
    const bool old_scope = in_thread_dependent_scope_;
    in_thread_dependent_scope_ |= thread_dependent;
    enclosing_let_definitions_.emplace(op->var.get(), op->value);
    Stmt body = VisitStmt(op->body);
    enclosing_let_definitions_.erase(op->var.get());
    in_thread_dependent_scope_ = old_scope;
    PrimExpr value = VisitExpr(op->value);
    return LetStmt(op->var, value, body);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    const bool thread_dependent = IsUnsafePersistentControlExpr(
        op->condition, enclosing_let_definitions_, active_thread_vars_);
    const bool old_scope = in_thread_dependent_scope_;
    in_thread_dependent_scope_ |= thread_dependent;
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case = op->else_case.defined()
                                   ? Optional<Stmt>(VisitStmt(op->else_case.value()))
                                   : Optional<Stmt>();
    in_thread_dependent_scope_ = old_scope;
    return IfThenElse(op->condition, then_case, else_case);
  }

  Stmt VisitStmt_(const WhileNode* op) final {
    const bool unsafe = IsUnsafePersistentControlExpr(
        op->condition, enclosing_let_definitions_, active_thread_vars_);
    const bool old_scope = in_thread_dependent_scope_;
    in_thread_dependent_scope_ |= unsafe;
    Stmt body = VisitStmt(op->body);
    in_thread_dependent_scope_ = old_scope;
    return While(op->condition, body);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    const bool is_thread_binding =
        op->kind == ForKind::kThreadBinding && op->thread_binding.defined();
    const std::string thread_tag = is_thread_binding
                                       ? std::string(op->thread_binding.value()->thread_tag)
                                       : std::string();
    const bool is_thread = is_thread_binding && thread_tag.find("threadIdx") != std::string::npos;
    const bool is_block = is_thread_binding && thread_tag.find("blockIdx") != std::string::npos;
    if (is_thread) {
      active_thread_vars_.insert(op->loop_var.get());
      active_thread_ranges_.emplace_back(
          op->loop_var, Range::FromMinExtent(op->min, op->extent));
    }
    if (is_block) active_block_vars_.insert(op->loop_var.get());
    const bool unsafe_bounds =
        IsUnsafePersistentControlExpr(op->min, enclosing_let_definitions_, active_thread_vars_) ||
        IsUnsafePersistentControlExpr(op->extent, enclosing_let_definitions_, active_thread_vars_);
    const bool old_scope = in_thread_dependent_scope_;
    in_thread_dependent_scope_ |= unsafe_bounds;
    if (!HasDCUPersistentTileAnnotation(op)) {
      Stmt result = StmtMutator::VisitStmt_(op);
      in_thread_dependent_scope_ = old_scope;
      if (is_thread) {
        active_thread_vars_.erase(op->loop_var.get());
        active_thread_ranges_.pop_back();
      }
      if (is_block) active_block_vars_.erase(op->loop_var.get());
      return result;
    }
    ICHECK(!in_thread_dependent_scope_)
        << kDCUPersistentTileAnnotation
        << " must not be enclosed by thread-dependent or impure If/For/While/Let control";
    ICHECK_EQ(op->kind, ForKind::kSerial)
        << kDCUPersistentTileAnnotation << " is valid only on T.serial";
    FreeVarCollector loop_bounds;
    loop_bounds(op->min);
    loop_bounds(op->extent);
    for (const VarNode* var : active_thread_vars_) {
      ICHECK(!loop_bounds.vars.count(var))
          << kDCUPersistentTileAnnotation << " loop bounds must be block-uniform";
    }

    PersistentLetDefinitions lets;
    lets(op->body);
    // Resource expressions may refer to a uniform Let outside the marked
    // loop.  It dominates the loop for placement, but it must still be
    // expanded for purity/thread/tile dependency checks: an alias to a load
    // or call is not a valid cached resource base.
    auto resource_let_definitions = enclosing_let_definitions_;
    resource_let_definitions.insert(lets.definitions.begin(), lets.definitions.end());
    PersistentResourceCollector collector(op->loop_var, active_thread_vars_, active_block_vars_,
                                          resource_let_definitions, active_thread_ranges_);
    collector(op->body);
    ICHECK(!collector.result.copies.empty())
        << kDCUPersistentTileAnnotation << " loop contains no vector async global-to-shared copy";
    ICHECK(!CollectDCUMainLoops(op->body).empty())
        << kDCUPersistentTileAnnotation << " loop must enclose an innermost DCU GEMM K loop";
    const bool degenerate_single_iteration =
        !collector.result.has_tile_dependent_base && IsExplicitSingleIterationPersistentLoop(op);
    ICHECK(collector.result.has_tile_dependent_base || degenerate_single_iteration)
        << kDCUPersistentTileAnnotation
        << " must be an outer output-tile loop: no resource base depends on its loop variable "
        << "unless it is explicitly min=0 and extent=1";
    PersistentInnerBufferScopeCollector inner_buffer_defs;
    inner_buffer_defs(op->body);
    for (const CopyInfo& copy : collector.result.copies) {
      ICHECK(!inner_buffer_defs.defined_data_vars.count(copy.src_buffer->data.get()))
          << kDCUPersistentTileAnnotation << " does not support a global source buffer declared "
          << "inside the marked tile loop: " << copy.src_buffer->name;
    }
    FinalizePersistentResourceBases(&collector.result, op->loop_var, active_block_vars_,
                                    resource_let_definitions,
                                    Range::FromMinExtent(op->min, op->extent));

    // This deliberately does not use InjectAtDCUTarget: that helper targets
    // the function's thread body and would hoist a per-tile resource out of
    // this loop.  Dependencies force placement after tile-coordinate Lets,
    // while the result remains before the nested K pipeline.
    Stmt scoped = InjectPersistentResourceLets(op->body, collector.result);
    scoped = PersistentStoreReplacer::Run(scoped, collector.result.copies,
                                          collector.result.global_to_res_var,
                                          collector.result.verified_residuals);
    scoped = DCUAsyncCopySmemOffsetFixer::Run(std::move(scoped));
    auto annotations = op->annotations;
    annotations.Set(kDCUPersistentResourceVerifiedAnnotation, Integer(1));
    Stmt result = For(op->loop_var, op->min, op->extent, op->kind, scoped, op->thread_binding,
                      annotations);
    if (is_thread) {
      active_thread_vars_.erase(op->loop_var.get());
      active_thread_ranges_.pop_back();
    }
    if (is_block) active_block_vars_.erase(op->loop_var.get());
    in_thread_dependent_scope_ = old_scope;
    return result;
  }
};

class PersistentTileMarkerFinder : public StmtVisitor {
 public:
  bool found{false};

 private:
  void VisitStmt_(const ForNode* op) final {
    found |= HasDCUPersistentTileAnnotation(op);
    if (!found) {
      StmtVisitor::VisitStmt_(op);
    }
  }
};

bool HasPersistentTileMarker(const Stmt& body) {
  PersistentTileMarkerFinder finder;
  finder(body);
  return finder.found;
}

PrimFunc LowerSharedGlobalCopy(PrimFunc f) {
    auto* n = f.CopyOnWrite();

    // Preserve the historical ordinary-GEMM code path exactly when the
    // explicit opt-in is absent.  A marked function first rewrites only its
    // marked tile loop(s); any unrelated copies then continue through the
    // existing fallback below.
    if (HasPersistentTileMarker(n->body)) {
        n->body = PersistentResourceRewriter::Run(n->body);
    }
    
    auto res = CollectResources(n->body);
    if (res.copies.empty()){
        return f;
    }

    Stmt injected = InjectDCUResources(n->body, res.shared_alloc_to_binding);
    
    Stmt replaced = StoreReplacer::Run(injected, res.copies, res.global_to_res_var,
                                       res.global_to_base_offset);
    replaced = DCUAsyncCopySmemOffsetFixer::Run(std::move(replaced));

    n->body = std::move(replaced);
    
    return GetRef<PrimFunc>(n);
}

namespace transform {
using namespace tir::transform;

tvm::transform::Pass LowerSharedGlobalCopy() {
    auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
        return tl::LowerSharedGlobalCopy(std::move(f));
    };
    return CreatePrimFuncPass(pass_func, 0, "tl.LowerSharedGlobalCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::GlobalDef().def("tl.transform.LowerSharedGlobalCopy", LowerSharedGlobalCopy);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
