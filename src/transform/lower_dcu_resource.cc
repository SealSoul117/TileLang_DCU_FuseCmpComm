#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/memory.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/arith/analyzer.h>

#include <vector>
#include <unordered_map>
#include <unordered_set>

using tvm::ffi::GetRef;
using tvm::ffi::make_object;

namespace tvm {
namespace tl {
using namespace tir;
using ffi::Array;
using ffi::String;

struct CopyInfo {
    Buffer dst_buffer;
    Buffer src_buffer;
    Array<PrimExpr> dst_indices;
    Array<PrimExpr> src_indices;
    Stmt store_stmt;
};

struct CollectResult {
    std::vector<CopyInfo> copies;
    std::unordered_map<String, Var> global_to_res_var;
    std::unordered_map<String, std::pair<Var, PrimExpr>> shared_alloc_to_binding;

    const StmtNode* inject_target = nullptr;
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

CollectResult CollectResources(const Stmt& body) {
    class Collector : public StmtExprVisitor {
    public:
        CollectResult result;

    private:
        bool in_async{false};
        std::unordered_set<const tvm::tir::VarNode*> loop_vars_;
        std::vector<const tvm::tir::StmtNode*> scope_stack_;
        bool IsSharedScope(const Buffer& buf) {
            auto s = buf.scope();
            return s == "shared" || s == "shared.dyn";
        }
        bool IsGlobalScope(const Buffer& buf) {
            auto s = buf.scope();
            return s == "global" || s == "";
        }

        void VisitStmt_(const AttrStmtNode* attr) override {
            scope_stack_.push_back(attr);
            if (attr->attr_key == tir::attr::thread_extent) {
                auto iv = attr->node.as<tvm::tir::IterVarNode>();
                const std::string& tag = iv->thread_tag;

                if (tag.find("threadIdx") != std::string::npos) {
                    tvm::tir::Var thread_var = iv->var;
                    loop_vars_.insert(thread_var.get());

                    StmtExprVisitor::VisitStmt_(attr);

                    loop_vars_.erase(thread_var.get());
                } else {
                    StmtExprVisitor::VisitStmt_(attr);
                }

            } else if (attr->attr_key == tir::attr::async_scope) {
                ICHECK(in_async == false) << "Nested async scopes not supported";
                in_async = true;
                
                StmtExprVisitor::VisitStmt_(attr);
                
                in_async = false;
            }
            else {
                StmtExprVisitor::VisitStmt_(attr);
            }
            scope_stack_.pop_back();
        }

        void VisitStmt_(const SeqStmtNode* op) override {
            scope_stack_.push_back(op);
            StmtExprVisitor::VisitStmt_(op);
            scope_stack_.pop_back();
        }

        void VisitStmt_(const ForNode* op) override {
            scope_stack_.push_back(op);
            loop_vars_.insert(op->loop_var.get());
            StmtExprVisitor::VisitStmt_(op);
            loop_vars_.erase(op->loop_var.get());
            scope_stack_.pop_back();
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

        void VisitStmt_(const BufferStoreNode* op) final {
            Buffer dst = op->buffer;
            if (IsSharedScope(dst) && op->value.defined() && in_async) {
                if (const auto* load = PeelGlobalLoadValue(op->value)) {
                    Buffer src = load->buffer;
                    if (IsGlobalScope(src)) {
                        const StmtNode* target = op;
                        if (result.inject_target == nullptr) {
                            for (int i = scope_stack_.size() - 1; i >= 0; --i) {
                                if (scope_stack_[i]->IsInstance<AttrStmtNode>()) {
                                    auto attr = static_cast<const AttrStmtNode*>(scope_stack_[i]);
                                    if (attr->attr_key == tvm::tir::attr::thread_extent) {
                                        if (i + 1 < scope_stack_.size()) {
                                            result.inject_target = scope_stack_[i + 1];
                                        }
                                        break;
                                    }
                                }
                            }
                            if (result.inject_target == nullptr && !scope_stack_.empty()) {
                                    for (const auto* node : scope_stack_) {
                                        if (node->IsInstance<ForNode>() || node->IsInstance<SeqStmtNode>()) {
                                            result.inject_target = node;
                                            break;
                                        }
                                    }
                                }
                            if (result.inject_target == nullptr) result.inject_target = op;
                        }


                        VariableKeeper keeper(loop_vars_);
                        tvm::arith::Analyzer analyzer;
                        Array<PrimExpr> for_var_only_indices;

                        for (const auto& idx : load->indices) {
                            PrimExpr filtered = keeper(idx); 
                            for_var_only_indices.push_back(analyzer.Simplify(filtered));
                        }
                        CopyInfo info{dst, src, op->indices, for_var_only_indices, GetRef<Stmt>(op)};
                        result.copies.push_back(info);

                        if (result.global_to_res_var.find(src->name) == result.global_to_res_var.end()) {
                            Var var(src->name + "_dcu_res", DataType::Int(32, 4));

                            VariableEliminator eliminator(loop_vars_);
                            tvm::arith::Analyzer analyzer;
                            Array<PrimExpr> base_indices;
                            for (const auto& idx : load->indices) {
                                PrimExpr no_loops = eliminator(idx);
                                base_indices.push_back(analyzer.Simplify(no_loops));
                            }

                            Array<PrimExpr> args;
                            args.push_back(src->data);

                            for (const auto& idx : base_indices) {
                                args.push_back(idx);
                            }
                            PrimExpr val = Call(DataType::Int(32, 4), 
                                                Op::Get("tl.make_dcu_resource"), args);
                            
                            result.global_to_res_var[src->name] = var;
                            result.shared_alloc_to_binding[src->name] = {var, val};
                        }
                    }
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
                
                PrimExpr copy_size = IntImm(DataType::Int(32), 1);
                PrimExpr predicate = Bool(true);
                
                return Evaluate(
                    Call(DataType::Int(32), Op::Get("tl.dcu_async_copy"),
                         {dst_res, Flatten(copy.dst_indices),
                          src_res, Flatten(copy.src_indices),
                          copy_size, predicate}));
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

class ResourceInjector : public tvm::tir::StmtExprMutator {
public:
    static Stmt Run(Stmt body, 
                    const std::unordered_map<String, std::pair<Var, PrimExpr>>& bindings,
                    const tvm::tir::StmtNode* target) {
        if (!target || bindings.empty()) return body;
        ResourceInjector mutator(bindings, target);
        return mutator(std::move(body));
    }

private:
    ResourceInjector(const std::unordered_map<String, std::pair<Var, PrimExpr>>& bindings,
                     const tvm::tir::StmtNode* target)
        : bindings_(bindings), target_(target) {}

    Stmt VisitStmt(const Stmt& stmt) override {
        if (stmt.get() == target_) {
            Stmt new_stmt = StmtExprMutator::VisitStmt(stmt);
            
            for (const auto& item : bindings_) {
                Var res_var = item.second.first;
                PrimExpr init_expr = item.second.second;
                new_stmt = tvm::tir::LetStmt(res_var, init_expr, new_stmt);
            }
            return new_stmt; 
        }
        return StmtExprMutator::VisitStmt(stmt);
    }

    std::unordered_map<String, std::pair<Var, PrimExpr>> bindings_;
    const tvm::tir::StmtNode* target_;
};

PrimFunc LowerSharedGlobalCopy(PrimFunc f) {
    auto* n = f.CopyOnWrite();
    
    auto res = CollectResources(n->body);
    if (res.copies.empty()){
        return f;
    }

    Stmt injected = ResourceInjector::Run(n->body, res.shared_alloc_to_binding, res.inject_target);
    
    Stmt replaced = StoreReplacer::Run(injected, res.copies, res.global_to_res_var);
    
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