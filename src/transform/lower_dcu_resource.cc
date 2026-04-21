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

// ============================================================================
// 数据结构
// ============================================================================
struct CopyInfo {
    Buffer dst_buffer;
    Buffer src_buffer;
    Array<PrimExpr> dst_indices;
    Array<PrimExpr> src_indices;
    Stmt store_stmt;
};

struct CollectResult {
    std::vector<CopyInfo> copies;
    // 映射: Global Buffer Name -> DCU Resource Var (用于替换Store)
    std::unordered_map<String, Var> global_to_res_var;
    // 映射: Shared Buffer Name -> 要注入的LetStmt绑定 (Var, PrimExpr)
    // 这样我们就可以根据 shared buffer 的位置来决定注入点
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
    // 关键调试：打印每一个遇到的变量及其地址
    if (keep_vars_.count(op)) {
      LOG(INFO) << "[KEEP] Found var in list: " << op->name_hint << " (" << op << ")";
      return GetRef<PrimExpr>(op);
    } else {
      LOG(INFO) << "[ERASE] Var not in list: " << op->name_hint << " (" << op << ")";
      return tvm::tir::make_zero(op->dtype);
    }
  }

  // 额外处理：防止 Load 节点中的变量丢失
  PrimExpr VisitExpr_(const tvm::tir::BufferLoadNode* op) override {
    // 如果你的索引里嵌套了 BufferLoad，Load 本身不是 Var，
    // 但它里面可能含有 Var。Mutator 默认会递归，但我们可以显式打印。
    return ExprMutator::VisitExpr_(op);
  }

 private:
  const std::unordered_set<const tvm::tir::VarNode*>& keep_vars_;
};

// ============================================================================
// Phase 1: 收集拷贝信息 & 生成资源绑定
// ============================================================================
CollectResult CollectResources(const Stmt& body) {
    class Collector : public StmtExprVisitor {
    public:
        CollectResult result;

    private:
        std::unordered_set<const tvm::tir::VarNode*> loop_vars_;
        std::vector<const tvm::tir::StmtNode*> scope_stack_; // 追踪当前遍历的 AST 路径
        bool IsSharedScope(const Buffer& buf) {
            auto s = buf.scope();
            return s == "shared" || s == "shared.dyn";
        }
        bool IsGlobalScope(const Buffer& buf) {
            auto s = buf.scope();
            return s == "global" || s == "";
        }

        void VisitStmt_(const AttrStmtNode* op) override {
            scope_stack_.push_back(op);
            if (op->attr_key == tvm::tir::attr::thread_extent) {
                // 1. 获取 IterVar
                auto iv = op->node.as<tvm::tir::IterVarNode>();
                const std::string& tag = iv->thread_tag;

                // 2. 只有当 tag 包含 "threadIdx" 时才加入 (过滤掉 blockIdx)
                // 比如: "threadIdx.x", "threadIdx.y", "threadIdx.z"
                if (tag.find("threadIdx") != std::string::npos) {
                    tvm::tir::Var thread_var = iv->var;
                    loop_vars_.insert(thread_var.get());

                    StmtExprVisitor::VisitStmt_(op);

                    loop_vars_.erase(thread_var.get());
                } else {
                    // 如果是 blockIdx 或其他，直接跳过当前层继续往下走
                    StmtExprVisitor::VisitStmt_(op);
                }

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

        void VisitStmt_(const BufferStoreNode* op) final {

            Buffer dst = op->buffer;
            if (IsSharedScope(dst) && op->value.defined()) {
                if (const auto* load = op->value.as<BufferLoadNode>()) {
                    Buffer src = load->buffer;
                    if (IsGlobalScope(src)) {
                        if (result.inject_target == nullptr) {
                            // 从下往上回溯栈，寻找最内层的 thread_extent
                            for (int i = scope_stack_.size() - 1; i >= 0; --i) {
                                if (scope_stack_[i]->IsInstance<AttrStmtNode>()) {
                                    auto attr = static_cast<const AttrStmtNode*>(scope_stack_[i]);
                                    if (attr->attr_key == tvm::tir::attr::thread_extent) {
                                        // 找到了最内层的线程绑定。它里面的下一个节点（i+1）就是我们应该包裹的节点
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
                            // 如果还是空，直接 fallback 到当前操作
                            if (result.inject_target == nullptr) result.inject_target = op;
                        }


                        // 1. 记录拷贝
                        VariableKeeper keeper(loop_vars_);
                        tvm::arith::Analyzer analyzer;
                        Array<PrimExpr> for_var_only_indices;

                        for (const auto& idx : load->indices) {
                            PrimExpr filtered = keeper(idx); 
                            for_var_only_indices.push_back(analyzer.Simplify(filtered));
                            LOG(INFO) << "ONLY Index: " << idx;
                        }
                        CopyInfo info{dst, src, op->indices, for_var_only_indices, GetRef<Stmt>(op)};
                        result.copies.push_back(info);

                        // 2. 只有当没处理过这个 Global Buffer 时才生成 Binding
                        if (result.global_to_res_var.find(src->name) == result.global_to_res_var.end()) {
                            Var var(src->name + "_dcu_res", DataType::Int(32, 4));

                            VariableEliminator eliminator(loop_vars_);
                            tvm::arith::Analyzer analyzer;
                            Array<PrimExpr> base_indices;
                        LOG(INFO) << loop_vars_.size() << " loop vars in context.";
                        for (const auto* var : loop_vars_) {
                            LOG(INFO) << "Loop Var: " << var->name_hint;
                        }
                            for (const auto& idx : load->indices) {
                                // 将所有外层循环变量 (k, i 等) 全部替换为 0
                                PrimExpr no_loops = eliminator(idx);
                                // 化简出最终的基地址表达式
                                base_indices.push_back(analyzer.Simplify(no_loops));
                            }

                            // ✅ 关键点：填充真实的地址信息 src->data (即 A.data)
                            Array<PrimExpr> args;
                            args.push_back(src->data);  // 先加 data

                            // 如果需要把 indices 的每个元素作为独立参数展开：
                            for (const auto& idx : base_indices) {
                                args.push_back(idx);
                                LOG(INFO) << "Clean Index: " << idx;
                            }
                            PrimExpr val = Call(DataType::Int(32, 4), 
                                                Op::Get("tl.make_dcu_resource"), args);
                            
                            result.global_to_res_var[src->name] = var;
                            // 将这个绑定关系和 destination 的 shared buffer 绑死
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

// ============================================================================
// Phase 2: 替换 BufferStore -> dcu_async_copy
// ============================================================================
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

    Stmt VisitStmt_(const BufferStoreNode* op) final {
        for (const auto& copy : copies_) {
            if (copy.store_stmt.same_as(GetRef<Stmt>(op))) {
                // Global 取 resource var (A_dcu_res)
                Var src_res = global_to_var_.at(copy.src_buffer->name);
                // Shared 取 data pointer (A_shared.data)
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

// ============================================================================
// Phase 3: 根据 Shared Alloc 位置进行精准注入
// ============================================================================
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
        // 当我们遍历到刚才标记的那个 AST 节点时
        if (stmt.get() == target_) {
            // 先向下遍历（保持 TVM Mutator 的习惯）
            Stmt new_stmt = StmtExprMutator::VisitStmt(stmt);
            
            // 在这个节点的外面套上所有的 LetStmt
            for (const auto& item : bindings_) {
                Var res_var = item.second.first;
                PrimExpr init_expr = item.second.second;
                new_stmt = tvm::tir::LetStmt(res_var, init_expr, new_stmt);
            }
            return new_stmt; // 返回包裹好的新节点
        }
        return StmtExprMutator::VisitStmt(stmt);
    }

    std::unordered_map<String, std::pair<Var, PrimExpr>> bindings_;
    const tvm::tir::StmtNode* target_;
};

// ============================================================================
// Pass 入口
// ============================================================================
PrimFunc LowerSharedGlobalCopy(PrimFunc f) {
    auto* n = f.CopyOnWrite();
    
    // 1. 收集信息并定位目标注入点
    auto res = CollectResources(n->body);
    if (res.copies.empty()) return f;

    // 【核心修改】：2. 先注入 LetStmt！
    // 此时使用的 n->body 是原始 AST，res.inject_target 指针百分之百匹配。
    Stmt injected = ResourceInjector::Run(n->body, res.shared_alloc_to_binding, res.inject_target);
    
    // 3. 替换拷贝语句 
    // injected 是套了 LetStmt 的新 AST，但底层的 BufferStore 还是原来的，可以被正常替换。
    Stmt replaced = StoreReplacer::Run(injected, res.copies, res.global_to_res_var);
    
    // 4. 写回 PrimFunc
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