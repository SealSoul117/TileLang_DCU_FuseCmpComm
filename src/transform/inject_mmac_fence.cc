#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/arith/analyzer.h>
#include <tvm/node/structural_equal.h>

#include <optional>
#include <string>
#include <unordered_set>
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

std::optional<PrimExpr> GetFirstDCUAsyncCopyResource(const Stmt& stmt) {
    struct Finder : public StmtVisitor {
        std::optional<PrimExpr> resource;

        void VisitStmt_(const EvaluateNode* op) final {
            if (resource.has_value()) {
                return;
            }
            if (const auto* call = op->value.as<CallNode>()) {
                if (IsDCUAsyncCopyCall(call) && call->args.size() >= 3) {
                    resource = call->args[2];
                }
            }
        }

        void VisitStmt_(const AttrStmtNode* op) final {
            if (!resource.has_value()) {
                VisitStmt(op->body);
            }
        }

        void VisitStmt_(const IfThenElseNode* op) final {
            if (resource.has_value()) {
                return;
            }
            VisitStmt(op->then_case);
            if (!resource.has_value() && op->else_case.defined()) {
                VisitStmt(op->else_case.value());
            }
        }
    };

    Finder finder;
    finder(stmt);
    return finder.resource;
}

struct PrefetchLoadGroup {
    const Object* stmt;
    PrimExpr first_global_resource;
    int segment_id;
};

void CollectPrefetchLoadGroupsImpl(const Stmt& stmt, std::vector<PrefetchLoadGroup>* groups,
                                   bool* past_main_loop, int* next_segment_id) {
    if (*past_main_loop) {
        return;
    }

    if (const auto* for_op = stmt.as<ForNode>()) {
        if (IsDCUMainLoop(for_op)) {
            *past_main_loop = true;
            return;
        }
        if (auto resource = GetFirstDCUAsyncCopyResource(for_op->body)) {
            groups->push_back({for_op, resource.value(), ++(*next_segment_id)});
        }
        return;
    }

    if (const auto* if_op = stmt.as<IfThenElseNode>()) {
        if (auto resource = GetFirstDCUAsyncCopyResource(stmt)) {
            groups->push_back({if_op, resource.value(), *next_segment_id});
            return;
        }
        CollectPrefetchLoadGroupsImpl(if_op->then_case, groups, past_main_loop, next_segment_id);
        if (if_op->else_case.defined()) {
            CollectPrefetchLoadGroupsImpl(if_op->else_case.value(), groups, past_main_loop,
                                            next_segment_id);
        }
        return;
    }

    if (StmtIsDCUAsyncCopyRoot(stmt)) {
        if (auto resource = GetFirstDCUAsyncCopyResource(stmt)) {
            groups->push_back({stmt.get(), resource.value(), *next_segment_id});
        }
        return;
    }

    if (const auto* seq = stmt.as<SeqStmtNode>()) {
        int flat_segment_id = *next_segment_id;
        for (size_t i = 0; i < seq->seq.size(); ++i) {
            if (*past_main_loop) {
                break;
            }
            const Stmt& child = seq->seq[i];
            if (child.as<ForNode>() || child.as<IfThenElseNode>()) {
                CollectPrefetchLoadGroupsImpl(child, groups, past_main_loop, next_segment_id);
                flat_segment_id = *next_segment_id;
            } else if (StmtIsDCUAsyncCopyRoot(child)) {
                size_t run_len = CountConsecutiveDCUAsyncCopyRun(seq->seq, i);
                for (size_t j = 0; j < run_len; ++j) {
                    if (auto resource = GetFirstDCUAsyncCopyResource(seq->seq[i + j])) {
                        groups->push_back(
                            {seq->seq[i + j].get(), resource.value(), flat_segment_id});
                    }
                }
                i += run_len - 1;
                flat_segment_id = ++(*next_segment_id);
            } else {
                CollectPrefetchLoadGroupsImpl(child, groups, past_main_loop, next_segment_id);
                flat_segment_id = *next_segment_id;
            }
        }
        return;
    }

    if (const auto* op = stmt.as<AttrStmtNode>()) {
        CollectPrefetchLoadGroupsImpl(op->body, groups, past_main_loop, next_segment_id);
        return;
    }
    if (const auto* op = stmt.as<LetStmtNode>()) {
        CollectPrefetchLoadGroupsImpl(op->body, groups, past_main_loop, next_segment_id);
        return;
    }
    if (const auto* op = stmt.as<AllocateNode>()) {
        CollectPrefetchLoadGroupsImpl(op->body, groups, past_main_loop, next_segment_id);
        return;
    }
    if (const auto* op = stmt.as<AllocateConstNode>()) {
        CollectPrefetchLoadGroupsImpl(op->body, groups, past_main_loop, next_segment_id);
        return;
    }
    if (const auto* op = stmt.as<DeclBufferNode>()) {
        CollectPrefetchLoadGroupsImpl(op->body, groups, past_main_loop, next_segment_id);
        return;
    }
    if (const auto* op = stmt.as<BufferRealizeNode>()) {
        CollectPrefetchLoadGroupsImpl(op->body, groups, past_main_loop, next_segment_id);
        return;
    }
    if (const auto* op = stmt.as<AssertStmtNode>()) {
        CollectPrefetchLoadGroupsImpl(op->body, groups, past_main_loop, next_segment_id);
        return;
    }
    if (const auto* op = stmt.as<BlockNode>()) {
        if (op->init.defined()) {
            CollectPrefetchLoadGroupsImpl(op->init.value(), groups, past_main_loop, next_segment_id);
        }
        CollectPrefetchLoadGroupsImpl(op->body, groups, past_main_loop, next_segment_id);
        return;
    }
    if (const auto* op = stmt.as<BlockRealizeNode>()) {
        CollectPrefetchLoadGroupsImpl(op->block, groups, past_main_loop, next_segment_id);
        return;
    }
}

std::vector<PrefetchLoadGroup> CollectPrefetchLoadGroups(const Stmt& root) {
    std::vector<PrefetchLoadGroup> groups;
    bool past_main_loop = false;
    int next_segment_id = 0;
    CollectPrefetchLoadGroupsImpl(root, &groups, &past_main_loop, &next_segment_id);
    return groups;
}

struct PrefetchBarrierPlan {
    std::unordered_set<const Object*> before;
    std::unordered_set<const Object*> after;
};

PrefetchBarrierPlan ComputePrefetchBarrierPlan(const Stmt& root) {
    PrefetchBarrierPlan plan;
    const auto groups = CollectPrefetchLoadGroups(root);
    if (groups.empty()) {
        return plan;
    }

    const PrimExpr& anchor = groups[0].first_global_resource;
    for (size_t i = 1; i < groups.size(); ++i) {
        const PrimExpr& cur = groups[i].first_global_resource;
        const PrimExpr& prev = groups[i - 1].first_global_resource;
        bool new_anchor_round = tvm::StructuralEqual()(cur, anchor) &&
                                (!tvm::StructuralEqual()(prev, anchor) ||
                                 groups[i].segment_id != groups[i - 1].segment_id);
        if (new_anchor_round) {
            plan.before.insert(groups[i].stmt);
        }
    }
    plan.after.insert(groups.back().stmt);
    return plan;
}

bool IsAsyncWaitScopeStmt(const Stmt& stmt) {
    const auto* op = stmt.as<AttrStmtNode>();
    if (!op) {
        return false;
    }
    return op->attr_key == tir::attr::async_wait_queue_scope ||
           op->attr_key == "async_wait_queue_scope" ||
           op->attr_key == tir::attr::async_wait_inflight_count ||
           op->attr_key == "async_wait_inflight_count";
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
          epilogue_last_mma_(ComputeEpilogueLastMMAStmt(root_body)),
          prefetch_barrier_plan_(ComputePrefetchBarrierPlan(root_body)) {}

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
        Stmt for_stmt = For(op->loop_var, op->min, op->extent, op->kind, body, op->thread_binding,
                            op->annotations);
        if (is_main_loop) {
            --inside_k_loop_depth_;
            phase_ = PipelinePhase::kAfterMainLoop;
            return for_stmt;
        }

        if (phase_ == PipelinePhase::kBeforeMainLoop && inside_k_loop_depth_ == 0) {
            Array<Stmt> seq;
            if (prefetch_barrier_plan_.before.count(op)) {
                seq.push_back(MakeWaveBarrierStmt());
            }
            seq.push_back(for_stmt);
            if (prefetch_barrier_plan_.after.count(op)) {
                seq.push_back(MakeWaveBarrierStmt());
            }
            if (seq.size() > 1) {
                return SeqStmt::Flatten(seq);
            }
        }
        return for_stmt;
    }

    Stmt VisitStmt_(const IfThenElseNode* op) override {
        Stmt then_case = VisitStmt(op->then_case);
        Stmt if_stmt;
        if (op->else_case.defined()) {
            Stmt else_case = VisitStmt(op->else_case.value());
            if_stmt = IfThenElse(op->condition, then_case, else_case);
        } else {
            if_stmt = IfThenElse(op->condition, then_case);
        }

        if (phase_ == PipelinePhase::kBeforeMainLoop && inside_k_loop_depth_ == 0) {
            Array<Stmt> seq;
            if (prefetch_barrier_plan_.before.count(op)) {
                seq.push_back(MakeWaveBarrierStmt());
            }
            seq.push_back(if_stmt);
            if (prefetch_barrier_plan_.after.count(op)) {
                seq.push_back(MakeWaveBarrierStmt());
            }
            if (seq.size() > 1) {
                return SeqStmt::Flatten(seq);
            }
        }
        return if_stmt;
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
                if (phase_ == PipelinePhase::kBeforeMainLoop && inside_k_loop_depth_ == 0) {
                    for (size_t j = 0; j < run_len; ++j) {
                        const Stmt& copy_stmt = op->seq[i + j];
                        Array<Stmt> wrapped;
                        if (prefetch_barrier_plan_.before.count(copy_stmt.get())) {
                            wrapped.push_back(MakeWaveBarrierStmt());
                        }
                        wrapped.push_back(this->VisitStmt(copy_stmt));
                        if (prefetch_barrier_plan_.after.count(copy_stmt.get())) {
                            wrapped.push_back(MakeWaveBarrierStmt());
                        }
                        if (wrapped.size() > 1) {
                            new_seq.push_back(SeqStmt::Flatten(wrapped));
                        } else {
                            new_seq.push_back(wrapped[0]);
                        }
                    }
                } else if (inside_k_loop_depth_ > 0) {
                    for (size_t j = 0; j < run_len; ++j) {
                        new_seq.push_back(this->VisitStmt(op->seq[i + j]));
                        // When more than 2 consecutive async copies, insert a wave_barrier
                        // after every pair.
                        if (run_len > 2 && (j + 1) % 2 == 0) {
                            new_seq.push_back(MakeWaveBarrierStmt());
                        }
                    }
                } else {
                    for (size_t j = 0; j < run_len; ++j) {
                        new_seq.push_back(this->VisitStmt(op->seq[i + j]));
                    }
                }
                i += run_len - 1;
                prev_stmt_is_dcu_async_copy = true;
            } else {
                LoadCounter counter;
                counter(stmt);
                // A wait-scope starts a new shared-load cluster (one A/B stage).
                // Without a main k-loop, multiple clusters sit in the same SeqStmt
                // and must not be summed into one fence (e.g. 4+4 + 4+4 = 16
                // instead of the latest A+B = 8).
                if (IsAsyncWaitScopeStmt(stmt) && counter.total_loads > 0) {
                    pending_load_count = counter.total_loads;
                } else {
                    pending_load_count += counter.total_loads;
                }
                new_seq.push_back(this->VisitStmt(stmt));
                prev_stmt_is_dcu_async_copy = false;
            }
        }
        return SeqStmt(new_seq);
    }

private:
    Stmt global_last_fence_mma_;
    Stmt epilogue_last_mma_;
    PrefetchBarrierPlan prefetch_barrier_plan_;
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