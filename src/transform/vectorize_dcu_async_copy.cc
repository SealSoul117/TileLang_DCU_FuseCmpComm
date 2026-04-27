#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>

#include <vector>

using namespace tvm::tir;

using tvm::ffi::GetRef;
using tvm::ffi::make_object;

namespace tvm {
namespace tl {
using namespace tir;
using ffi::Array;
using ffi::String;

class AsyncCopySimplifier : public StmtExprMutator {
public:
    static Stmt Run(Stmt stmt) {
        AsyncCopySimplifier mutator;
        return mutator(std::move(stmt));
    }

private:
    arith::Analyzer analyzer_;
    Var k_var_;
    PrimExpr k_extent_;
    bool in_unrolled_i_ = false;

    // 通用的步长提取函数：从 expr 中提取指定 var 的步长，并返回剩余的 base
    std::pair<PrimExpr, PrimExpr> ExtractStride(PrimExpr expr, Var var) {
        if (!var.defined()) return {expr, make_zero(expr->dtype)};

        PrimExpr base = tvm::tir::Substitute(expr, {{var, make_zero(var.dtype())}});
        PrimExpr plus_one = tvm::tir::Substitute(expr, {{var, make_const(var.dtype(), 1)}});
        PrimExpr stride = analyzer_.Simplify(plus_one - base);
        return {analyzer_.Simplify(base), stride};
    }

    // 核心重写逻辑
    Stmt RewriteAsyncCopy(const CallNode* call, Var i_var, PrimExpr i_extent) {
        // 1. 预处理：剥离 RampNode 获得基础偏移
        PrimExpr raw_dst_off = call->args[1];
        PrimExpr raw_src_off = call->args[3];
        if (const RampNode* r = raw_dst_off.as<RampNode>()) raw_dst_off = r->base;
        if (const RampNode* r = raw_src_off.as<RampNode>()) raw_src_off = r->base;

        // 2. 提取 i 的步长
        auto [base_i_dst, i_stride_dst] = ExtractStride(raw_dst_off, i_var);
        auto [base_i_src, i_stride_src] = ExtractStride(raw_src_off, i_var);

        // 3. 提取 k 的步长 (始终尝试从 base_i_src 中提取，无论是否存在 i 循环)
        // 只要 k_var_ 在外层循环中定义了，这里就能提取出非 0 的步长
        auto [final_src_offset, k_stride_src] = ExtractStride(base_i_src, k_var_);

        // 构造最初要求的 8 个参数：
        // [dst, dst_off, src, src_off, i_extent, i_stride_dst, i_stride_src, k_stride_src]
        Array<PrimExpr> new_args = {
            call->args[0],      // dst_buf
            base_i_dst,         // 最终 dst 偏移
            call->args[2],      // src_buf
            final_src_offset,   // 最终 src 偏移
            i_extent,           // i 循环次数 (无循环时为 0)
            i_stride_dst,       // i 的 dst 步长 (无循环时为 0)
            i_stride_src,       // i 的 src 步长 (无循环时为 0)
            k_stride_src        // k 的 src 步长 (即便无 i 循环，这里也能拿到 k 的步长)
        };

        return Evaluate(Call(call->dtype, call->op, new_args));
    }

    // 处理无循环包裹的情况
    Stmt VisitStmt_(const EvaluateNode* op) final {
        if (!in_unrolled_i_) {
            if (const CallNode* call = op->value.as<CallNode>()) {
                static const Op& dcu_copy_op = Op::Get("tl.dcu_async_copy");
                // 只要参数个数不是 8 (我们重写后的目标个数)，就进行处理
                if (call->op.same_as(dcu_copy_op) && call->args.size() != 8) {
                    return RewriteAsyncCopy(call, Var(), make_zero(DataType::Int(32)));
                }
            }
        }
        return StmtExprMutator::VisitStmt_(op);
    }

    Stmt VisitStmt_(const ForNode* op) final {
        // 记录 k 信息 (假设 k 在外层)
        bool is_k = (op->loop_var->name_hint == "k");
        if (is_k) {
            k_var_ = op->loop_var;
            k_extent_ = op->extent;
        }

        bool is_unrolled = (op->kind == ForKind::kUnrolled);
        bool prev_in_unrolled = in_unrolled_i_;
        if (is_unrolled) in_unrolled_i_ = true;

        Stmt body = this->VisitStmt(op->body);
        
        in_unrolled_i_ = prev_in_unrolled;

        if (is_unrolled) {
            if (const EvaluateNode* eval = body.as<EvaluateNode>()) {
                if (const CallNode* call = eval->value.as<CallNode>()) {
                    static const Op& dcu_copy_op = Op::Get("tl.dcu_async_copy");
                    if (call->op.same_as(dcu_copy_op)) {
                        // 还原 k 并在返回前处理重写
                        Stmt result = RewriteAsyncCopy(call, op->loop_var, op->extent);
                        return result; 
                    }
                }
            }
        }
        
        // 退出循环时清理 k 信息
        if (is_k) {
            k_var_ = Var();
            k_extent_ = PrimExpr();
        }

        if (body.same_as(op->body)) return GetRef<Stmt>(op);
        auto n = CopyOnWrite(op);
        n->body = std::move(body);
        return Stmt(n);
    }
};
// ============================================================================
// Pass 入口
// ============================================================================
PrimFunc SimplifyDCUAsyncCopy(PrimFunc f) {
    auto* n = f.CopyOnWrite();
    n->body = AsyncCopySimplifier::Run(std::move(n->body));
    return GetRef<PrimFunc>(n);
}

namespace transform {
using namespace tir::transform;

tvm::transform::Pass SimplifyDCUAsyncCopy() {
    auto pass_func = [=](PrimFunc f, const IRModule &m, tvm::transform::PassContext ctx) {
        return tl::SimplifyDCUAsyncCopy(std::move(f));
    };
    return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0, "tl.SimplifyDCUAsyncCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
    tvm::ffi::reflection::GlobalDef().def("tl.transform.SimplifyDCUAsyncCopy", SimplifyDCUAsyncCopy);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm