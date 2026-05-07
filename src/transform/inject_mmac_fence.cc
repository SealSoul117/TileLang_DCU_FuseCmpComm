#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/arith/analyzer.h>

#include <string>
#include <vector>

namespace tvm {
namespace tl {
using ffi::Array;
using namespace tir;

// 1. 辅助类：统计 Shared -> Register 的加载量
class LoadCounter : public StmtExprVisitor {
public:
    int total_loads = 0;
    int current_multiplier = 1;

    void VisitStmt_(const ForNode* op) override {
        int64_t extent = 1;
        if (auto imm = op->extent.as<IntImmNode>()) {
            extent = imm->value;
        }
        int prev_multiplier = current_multiplier;
        current_multiplier *= static_cast<int>(extent);
        StmtVisitor::VisitStmt_(op);
        current_multiplier = prev_multiplier;
    }

    void VisitExpr_(const BufferLoadNode* op) override {
        std::string scope = op->buffer.scope();
        std::string name = op->buffer->name;
        if (scope == "shared" || name.find("shared") != std::string::npos || 
            name.find("shmem") != std::string::npos) {
            total_loads += current_multiplier;
        }
        ExprVisitor::VisitExpr_(op);
    }
};

// 2. 核心 Mutator
class MMABarrierMutator : public StmtExprMutator {
public:
    bool ContainsMMA(const Stmt& stmt) {
        bool found = false;
        PostOrderVisit(stmt, [&found](const ObjectRef& node) {
            if (const CallNode* call = node.as<CallNode>()) {
                std::string op_name = "";
                if (const OpNode* op = call->op.as<OpNode>()) {
                    op_name = op->name;
                } else if (const GlobalVarNode* gv = call->op.as<GlobalVarNode>()) {
                    op_name = gv->name_hint;
                }
                if (op_name.find("mmac") != std::string::npos || 
                    op_name.find("mma") != std::string::npos) {
                    found = true;
                }
            }
        });
        return found;
    }

    Stmt VisitStmt_(const SeqStmtNode* op) override {
        // --- 步骤 1: 预扫描，确定最后一个需要插入 Fence 的位置 ---
        int last_fence_idx = -1;
        int temp_pending_count = 0;
        for (size_t i = 0; i < op->seq.size(); ++i) {
            if (ContainsMMA(op->seq[i])) {
                if (temp_pending_count > 0) {
                    last_fence_idx = static_cast<int>(i);
                    temp_pending_count = 0; // 模拟重置
                }
            } else {
                LoadCounter counter;
                counter(op->seq[i]);
                temp_pending_count += counter.total_loads;
            }
        }

        // --- 步骤 2: 实际构造新的 Sequence ---
        Array<Stmt> new_seq;
        int pending_load_count = 0;

        for (size_t i = 0; i < op->seq.size(); ++i) {
            const auto& stmt = op->seq[i];
            
            if (ContainsMMA(stmt)) {
                if (pending_load_count > 0) {
                    // 判断是否是该序列中最后一个 Fence
                    int fence_val = (static_cast<int>(i) == last_fence_idx) ? 0 : pending_load_count;
                    
                    Array<PrimExpr> args = {Integer(fence_val)};
                    
                    // 构造 Fence
                    auto fence_call = Call(DataType::Void(), Op::Get("tl.async_gld_fence"), args);
                    new_seq.push_back(Evaluate(fence_call));

                    // 构造 Barrier
                    auto barrier_call = Call(DataType::Void(), Op::Get("tl.wave_barrier"), {});
                    new_seq.push_back(Evaluate(barrier_call));

                    pending_load_count = 0; 
                }
                new_seq.push_back(this->VisitStmt(stmt));
            } else {
                LoadCounter counter;
                counter(stmt);
                pending_load_count += counter.total_loads;
                new_seq.push_back(this->VisitStmt(stmt));
            }
        }
        return SeqStmt(new_seq);
    }
};

// 3. Pass 包装
namespace transform {
using namespace tir::transform;
Pass InsertAsyncMMAFence() {
    auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
        auto* n = f.CopyOnWrite();
        MMABarrierMutator mutator;
        n->body = mutator(n->body);
        return f;
    };
    return CreatePrimFuncPass(pass_func, 0, "tl.InsertAsyncMMAFence", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::GlobalDef().def("tl.transform.InsertAsyncMMAFence", InsertAsyncMMAFence);
}
}  // namespace transform
}  // namespace tl
}  // namespace tvm