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
    void VisitExpr_(const CallNode* op) override {
        std::string func_name = "";
        if (auto opt_op = op->op.as<OpNode>()) {
            func_name = opt_op->name;
        } else if (auto global_var = op->op.as<GlobalVarNode>()) {
            func_name = global_var->name_hint;
        }
        if (func_name.find("ds_read") != std::string::npos) {
            total_loads += current_multiplier;
        }
        ExprVisitor::VisitExpr_(op);
    }

private:
    bool IsSharedMem(const Buffer& buf) {
        std::string scope = buf.scope();
        std::string name = buf->name;
        return (scope == "shared" || name.find("shared") != std::string::npos || 
                name.find("shmem") != std::string::npos || name.find("LDS") != std::string::npos);
    }
};

namespace {

bool StmtContainsMMA(const Stmt& stmt) {
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

void ScanStmtDefault(const Stmt& s, std::vector<Stmt>* fence_targets);

void ScanSeqStmt(const SeqStmtNode* op, std::vector<Stmt>* fence_targets) {
    int pending = 0;
    for (size_t i = 0; i < op->seq.size(); ++i) {
        const Stmt& stmt = op->seq[i];
        if (StmtContainsMMA(stmt)) {
            if (pending > 0) {
                fence_targets->push_back(stmt);
                pending = 0;
            }
            ScanStmtDefault(stmt, fence_targets);
        } else {
            LoadCounter counter;
            counter(stmt);
            pending += counter.total_loads;
            ScanStmtDefault(stmt, fence_targets);
        }
    }
}

void ScanStmtDefault(const Stmt& s, std::vector<Stmt>* fence_targets) {
    if (const auto* seq = s.as<SeqStmtNode>()) {
        ScanSeqStmt(seq, fence_targets);
        return;
    }
    if (const auto* op = s.as<AttrStmtNode>()) {
        ScanStmtDefault(op->body, fence_targets);
        return;
    }
    if (const auto* op = s.as<LetStmtNode>()) {
        ScanStmtDefault(op->body, fence_targets);
        return;
    }
    if (const auto* op = s.as<IfThenElseNode>()) {
        ScanStmtDefault(op->then_case, fence_targets);
        if (op->else_case) {
            ScanStmtDefault(op->else_case.value(), fence_targets);
        }
        return;
    }
    if (const auto* op = s.as<ForNode>()) {
        ScanStmtDefault(op->body, fence_targets);
        return;
    }
    if (const auto* op = s.as<WhileNode>()) {
        ScanStmtDefault(op->body, fence_targets);
        return;
    }
    if (const auto* op = s.as<AllocateNode>()) {
        ScanStmtDefault(op->body, fence_targets);
        return;
    }
    if (const auto* op = s.as<AllocateConstNode>()) {
        ScanStmtDefault(op->body, fence_targets);
        return;
    }
    if (const auto* op = s.as<DeclBufferNode>()) {
        ScanStmtDefault(op->body, fence_targets);
        return;
    }
    if (const auto* op = s.as<BufferRealizeNode>()) {
        ScanStmtDefault(op->body, fence_targets);
        return;
    }
    if (const auto* op = s.as<AssertStmtNode>()) {
        ScanStmtDefault(op->body, fence_targets);
        return;
    }
    if (const auto* op = s.as<BlockNode>()) {
        if (op->init.defined()) {
            ScanStmtDefault(op->init.value(), fence_targets);
        }
        ScanStmtDefault(op->body, fence_targets);
        return;
    }
    if (const auto* op = s.as<BlockRealizeNode>()) {
        ScanStmtDefault(op->block, fence_targets);
        return;
    }
}

Stmt ComputeGlobalLastFenceMMAStmt(const Stmt& root) {
    std::vector<Stmt> fence_targets;
    ScanStmtDefault(root, &fence_targets);
    if (fence_targets.empty()) {
        return Stmt();
    }
    return fence_targets.back();
}

} 

class MMABarrierMutator : public StmtExprMutator {
public:
    explicit MMABarrierMutator(const Stmt& root_body)
        : global_last_fence_mma_(ComputeGlobalLastFenceMMAStmt(root_body)) {}

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
        Array<Stmt> new_seq;
        int pending_load_count = 0;

        for (size_t i = 0; i < op->seq.size(); ++i) {
            const auto& stmt = op->seq[i];
            
            if (ContainsMMA(stmt)) {
                if (pending_load_count > 0) {
                    int fence_val =
                        (global_last_fence_mma_.defined() && stmt.same_as(global_last_fence_mma_))
                            ? 0
                            : pending_load_count;
                    
                    Array<PrimExpr> args = {Integer(fence_val)};
                    
                    auto fence_call = Call(DataType::Void(), Op::Get("tl.async_gld_fence"), args);
                    new_seq.push_back(Evaluate(fence_call));

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

private:
    Stmt global_last_fence_mma_;
};

namespace transform {
using namespace tir::transform;
Pass InsertAsyncMMAFence() {
    auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
        auto* n = f.CopyOnWrite();
        MMABarrierMutator mutator(n->body);
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