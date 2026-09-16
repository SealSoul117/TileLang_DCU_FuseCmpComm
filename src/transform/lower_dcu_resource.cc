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
    std::unordered_map<String, std::pair<Var, PrimExpr>> shared_alloc_to_binding;
    std::unordered_set<const tvm::tir::VarNode*> thread_vars;
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

class VariableKeeper : public tvm::tir::ExprMutator {
 public:
  explicit VariableKeeper(const std::unordered_set<const tvm::tir::VarNode*>& keep_vars)
      : keep_vars_(keep_vars) {}

  PrimExpr VisitExpr_(const tvm::tir::VarNode* op) override {
    if (keep_vars_.count(op)) {
      return GetRef<PrimExpr>(op);
    } else {
      return tvm::tir::make_zero(op->dtype);
    }
  }

  PrimExpr VisitExpr_(const tvm::tir::BufferLoadNode* op) override {
    return ExprMutator::VisitExpr_(op);
  }

 private:
  const std::unordered_set<const tvm::tir::VarNode*>& keep_vars_;
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

            VariableKeeper keeper(loop_vars_);
            tvm::arith::Analyzer analyzer;
            Array<PrimExpr> for_var_only_indices;

            for (const auto& idx : load->indices) {
                PrimExpr filtered = keeper(idx);
                for_var_only_indices.push_back(analyzer.Simplify(filtered));
            }
            CopyInfo info{dst,
                          src,
                          op->indices,
                          for_var_only_indices,
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
    return col.result;
}

class StoreReplacer : public StmtExprMutator {
public:
    static Stmt Run(Stmt body, const std::vector<CopyInfo>& copies,
                    const std::unordered_map<String, Var>& global_to_var) {
        StoreReplacer replacer(copies, global_to_var);
        return replacer(std::move(body));
    }

private:
    StoreReplacer(const std::vector<CopyInfo>& copies,
                  const std::unordered_map<String, Var>& global_to_var)
        : copies_(copies), global_to_var_(global_to_var) {}

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
                PrimExpr dst_offset = GetScalarOffset(Flatten(copy.dst_indices));
                auto [thread_dst_off, smem_offset] = SplitSmemOffset(dst_offset);
                thread_dst_off = ScaleToByteOffset(thread_dst_off, elem_bytes);
                smem_offset = ScaleToByteOffset(smem_offset, elem_bytes);
                PrimExpr src_off = GetScalarOffset(Flatten(copy.src_indices));
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

    PrimExpr Flatten(const Array<PrimExpr>& idx) {
        if (idx.empty()) return IntImm(DataType::Int(32), 0);
        if (idx.size() == 1) return idx[0];
        PrimExpr r = idx[0];
        for (size_t i = 1; i < idx.size(); ++i) r = r + idx[i];
        return r;
    }

    const std::vector<CopyInfo>& copies_;
    const std::unordered_map<String, Var>& global_to_var_;
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

PrimFunc LowerSharedGlobalCopy(PrimFunc f) {
    auto* n = f.CopyOnWrite();
    
    auto res = CollectResources(n->body);
    if (res.copies.empty()){
        return f;
    }

    Stmt injected = InjectDCUResources(n->body, res.shared_alloc_to_binding);
    
    Stmt replaced = StoreReplacer::Run(injected, res.copies, res.global_to_res_var);
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