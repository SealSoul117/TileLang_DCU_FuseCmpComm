#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/arith/analyzer.h>

#include <string>
#include <vector>

#include "inject_utils.h"

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

const Op& DCUAsyncCopyOp() { return Op::Get("tl.dcu_async_copy"); }
const Op& DCUAsyncCopyIdxenOp() { return Op::Get("tl.dcu_async_copy_idxen"); }

bool IsDCUAsyncCopyCall(const CallNode* call) {
    if (!call) return false;
    return call->op.same_as(DCUAsyncCopyOp()) || call->op.same_as(DCUAsyncCopyIdxenOp());
}

bool StmtIsDCUAsyncCopyRoot(const Stmt& stmt) {
    if (const auto* eval = stmt.as<EvaluateNode>()) {
        if (const auto* call = eval->value.as<CallNode>()) {
            return IsDCUAsyncCopyCall(call);
        }
    }
    if (const auto* attr = stmt.as<AttrStmtNode>()) {
        if (attr->attr_key == tir::attr::async_commit_queue_scope ||
            attr->attr_key == "async_commit_queue_scope") {
            return StmtIsDCUAsyncCopyRoot(attr->body);
        }
    }
    return false;
}

Stmt MakeWaveBarrierStmt() {
    auto barrier_call = Call(DataType::Void(), Op::Get("tl.wave_barrier"), {});
    return Evaluate(barrier_call);
}

size_t CountConsecutiveDCUAsyncCopyRun(const Array<Stmt>& seq, size_t start) {
    size_t len = 0;
    while (start + len < seq.size() && StmtIsDCUAsyncCopyRoot(seq[start + len])) {
        ++len;
    }
    return len;
}

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

Stmt ComputeEpilogueLastMMAStmt(const Stmt& root) {
    struct Collector : public StmtVisitor {
        Stmt last_mma;
        bool past_main_loop = false;

        void VisitStmt_(const ForNode* op) override {
            StmtVisitor::VisitStmt_(op);
            if (IsDCUMainLoop(op)) {
                past_main_loop = true;
            }
        }

        void VisitStmt_(const SeqStmtNode* op) override {
            for (const Stmt& s : op->seq) {
                if (past_main_loop && StmtContainsMMA(s)) {
                    last_mma = s;
                }
                VisitStmt(s);
            }
        }

        void VisitStmt_(const IfThenElseNode* op) override {
            if (past_main_loop) {
                if (StmtContainsMMA(op->then_case)) {
                    last_mma = op->then_case;
                }
                if (op->else_case.defined() && StmtContainsMMA(op->else_case.value())) {
                    last_mma = op->else_case.value();
                }
            }
            StmtVisitor::VisitStmt_(op);
        }
    };

    Collector collector;
    collector(root);
    return collector.last_mma;
}

enum class PipelinePhase { kBeforeMainLoop, kInMainLoop, kAfterMainLoop };

} 

class MMABarrierMutator : public StmtExprMutator {
public:
    explicit MMABarrierMutator(const Stmt& root_body)
        : global_last_fence_mma_(ComputeGlobalLastFenceMMAStmt(root_body)),
          epilogue_last_mma_(ComputeEpilogueLastMMAStmt(root_body)) {}

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

    Stmt VisitStmt_(const ForNode* op) override {
        bool is_main_loop = IsDCUMainLoop(op);
        if (is_main_loop) {
            phase_ = PipelinePhase::kInMainLoop;
            ++inside_k_loop_depth_;
        }
        Stmt body = VisitStmt(op->body);
        if (is_main_loop) {
            --inside_k_loop_depth_;
            phase_ = PipelinePhase::kAfterMainLoop;
        }
        return For(op->loop_var, op->min, op->extent, op->kind, body, op->thread_binding,
                   op->annotations);
    }

    Stmt VisitStmt_(const SeqStmtNode* op) override {
        Array<Stmt> new_seq;
        int pending_load_count = 0;
        bool prev_stmt_is_dcu_async_copy = false;

        for (size_t i = 0; i < op->seq.size(); ++i) {
            const auto& stmt = op->seq[i];
            
            if (ContainsMMA(stmt)) {
                if (pending_load_count > 0) {
                    int fence_val = pending_load_count;
                    if (phase_ == PipelinePhase::kAfterMainLoop) {
                        // Epilogue reuses the previous async_gld_sld_fence value.
                        if (last_sld_fence_val_ > 0) {
                            fence_val = last_sld_fence_val_;
                        }
                    } else if (global_last_fence_mma_.defined() &&
                               stmt.same_as(global_last_fence_mma_)) {
                        fence_val = 0;
                    }

                    auto fence_call = Call(DataType::Void(), Op::Get("tl.async_gld_sld_fence"),
                                           {Integer(fence_val)});
                    new_seq.push_back(Evaluate(fence_call));
                    // wave_barrier only pairs with async_gld_fence in the main loop.
                    if (phase_ != PipelinePhase::kAfterMainLoop) {
                        new_seq.push_back(MakeWaveBarrierStmt());
                    }

                    last_sld_fence_val_ = fence_val;
                    pending_load_count = 0;
                } else if (phase_ == PipelinePhase::kAfterMainLoop &&
                           epilogue_last_mma_.defined() && stmt.same_as(epilogue_last_mma_)) {
                    // Final epilogue drain: sld_fence(0) before the last mmac.
                    auto fence_call = Call(DataType::Void(), Op::Get("tl.async_gld_sld_fence"),
                                           {Integer(0)});
                    new_seq.push_back(Evaluate(fence_call));
                }
                new_seq.push_back(this->VisitStmt(stmt));
                prev_stmt_is_dcu_async_copy = false;
            } else if (StmtIsDCUAsyncCopyRoot(stmt)) {
                size_t run_len = CountConsecutiveDCUAsyncCopyRun(op->seq, i);
                // Skip leading wave_barrier inside the main k-loop; pipeline already
                // serializes consecutive cp_async there.
                if (!prev_stmt_is_dcu_async_copy && inside_k_loop_depth_ == 0) {
                    new_seq.push_back(MakeWaveBarrierStmt());
                }
                for (size_t j = 0; j < run_len; ++j) {
                    new_seq.push_back(this->VisitStmt(op->seq[i + j]));
                    // When more than 2 consecutive async copies, insert a wave_barrier
                    // after every pair.
                    if (run_len > 2 && (j + 1) % 2 == 0) {
                        new_seq.push_back(MakeWaveBarrierStmt());
                    }
                }
                i += run_len - 1;
                prev_stmt_is_dcu_async_copy = true;
            } else {
                LoadCounter counter;
                counter(stmt);
                pending_load_count += counter.total_loads;
                new_seq.push_back(this->VisitStmt(stmt));
                prev_stmt_is_dcu_async_copy = false;
            }
        }
        return SeqStmt(new_seq);
    }

private:
    Stmt global_last_fence_mma_;
    Stmt epilogue_last_mma_;
    PipelinePhase phase_{PipelinePhase::kBeforeMainLoop};
    int inside_k_loop_depth_ = 0;
    int last_sld_fence_val_ = 0;
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