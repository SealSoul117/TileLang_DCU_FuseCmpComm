#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt.h>
#include <algorithm>

using namespace tvm::tir;
using tvm::ffi::GetRef;

namespace tvm {
namespace tl {
using namespace tir;

/**
 * @brief 分析器：计算 Stmt 内部的 async 指令贡献
 * 注意：这里计算的是“静态进入一次该 Stmt 后产生的指令总数”
 */
class AsyncCountAnalyzer : public StmtExprVisitor {
public:
    static int64_t Analyze(const Stmt& stmt) {
        AsyncCountAnalyzer analyzer;
        analyzer.VisitStmt(stmt);
        return analyzer.count_;
    }

private:
    void VisitStmt_(const ForNode* op) override {
        // 如果遇到了嵌套循环，需要计算：子循环内部单次产生的量 * 子循环次数
        int64_t sub_loop_body_count = Analyze(op->body);
        
        int64_t extent = 1;
        if (auto e = op->extent.as<IntImmNode>()) {
            extent = e->value;
        }
        count_ += sub_loop_body_count * extent;
        // 停止递归，因为 Analyze(op->body) 已经处理完了
    }

    void VisitExpr_(const CallNode* op) override {
        bool is_async = op->op.same_as(Op::Get("tl.dcu_async_copy")) ||
                        op->op.same_as(builtin::ptx_cp_async());
        if (is_async) {
            count_++;
        }
        StmtExprVisitor::VisitExpr_(op);
    }

    int64_t count_ = 0;
};

/**
 * @brief 寻找循环体内部倍率的最大值
 */
class GlobalMaxAsyncFinder : public StmtVisitor {
public:
    static int64_t FindMax(const Stmt& stmt) {
        GlobalMaxAsyncFinder finder;
        finder.VisitStmt(stmt);
        return std::max(static_cast<int64_t>(1), finder.max_multiplier_);
    }

private:
    void VisitStmt_(const ForNode* op) override {
        // 【关键修正】：我们只分析循环的 Body 产生的 async 数量
        // 这样对于最外层的 for k，得到的结果就是它 body 里的 2 个 async 
        int64_t inner_count = AsyncCountAnalyzer::Analyze(op->body);
        
        if (inner_count > max_multiplier_) {
            max_multiplier_ = inner_count;
        }
        
        // 继续向下递归，检查是否有更深层的循环内部产生了更多指令
        StmtVisitor::VisitStmt_(op);
    }

    int64_t max_multiplier_ = 0;
};

class ROCmWaitCountRewriter : public StmtMutator {
public:
    static Stmt Substitute(const Stmt& stmt) {
        int64_t max_mult = GlobalMaxAsyncFinder::FindMax(stmt);
        ROCmWaitCountRewriter rewriter(max_mult);
        return rewriter(stmt);
    }

private:
    explicit ROCmWaitCountRewriter(int64_t mult) : global_max_mult_(mult) {}

    Stmt VisitStmt_(const AttrStmtNode* op) override {
        if (op->attr_key == tir::attr::async_wait_inflight_count ||
            op->attr_key == "async_wait_inflight_count") {
            if (auto int_imm = op->value.as<IntImmNode>()) {
                int64_t new_val = int_imm->value * global_max_mult_;
                return AttrStmt(op->node, op->attr_key, make_const(DataType::Int(32), new_val),
                                this->VisitStmt(op->body));
            }
        }
        return StmtMutator::VisitStmt_(op);
    }
    int64_t global_max_mult_;
};

// Pass 包装省略 (同前)
namespace transform {
using namespace tir::transform;
tvm::transform::Pass FixDCUWaitCount() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ROCmWaitCountRewriter::Substitute(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "FixDCUWaitCount", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.FixDCUWaitCount", FixDCUWaitCount);
}
}

}  // namespace tl
}  // namespace tvm