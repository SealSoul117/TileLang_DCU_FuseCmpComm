#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/arith/analyzer.h>

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
    PrimExpr k_extent_; // 新增：记录 k 循环的次数

    std::pair<PrimExpr, PrimExpr> ExtractStride(PrimExpr expr, Var var) {
        if (!var.defined()) return {expr, make_zero(expr.dtype())};
        PrimExpr base = tvm::tir::Substitute(expr, {{var, make_zero(var.dtype())}});
        PrimExpr plus_one = tvm::tir::Substitute(expr, {{var, make_const(var.dtype(), 1)}});
        PrimExpr stride = analyzer_.Simplify(plus_one - base);
        return {analyzer_.Simplify(base), stride};
    }

    Stmt VisitStmt_(const ForNode* op) final {
        // 1. 记录 k 的信息
        bool is_k = (op->loop_var->name_hint == "k");
        if (is_k) {
            k_var_ = op->loop_var;
            k_extent_ = op->extent; // 获取 k 的循环次数 (如 64)
        }

        // 2. 递归访问子节点
        Stmt body = this->VisitStmt(op->body);

        // 3. 处理 Async Copy 简化
        if (op->kind == ForKind::kUnrolled) {
            if (const EvaluateNode* eval = body.as<EvaluateNode>()) {
                if (const CallNode* call = eval->value.as<CallNode>()) {
                    static const Op& dcu_copy_op = Op::Get("tl.dcu_async_copy");
                    if (call->op.same_as(dcu_copy_op)) {
                        
                        Var i_var = op->loop_var;
                        PrimExpr i_extent = op->extent; // 获取 i 的循环次数 (如 2)

                        auto get_i_info = [&](PrimExpr offset) {
                            if (const RampNode* ramp = offset.as<RampNode>()) {
                                auto [base, stride] = ExtractStride(ramp->base, i_var);
                                return std::make_pair(base, stride);
                            }
                            return ExtractStride(offset, i_var);
                        };

                        // 提取 i 的步长
                        auto [base_dst, i_stride_dst] = get_i_info(call->args[1]);
                        auto [base_src, i_stride_src] = get_i_info(call->args[3]);

                        // 提取 k 的步长 (从 base_src 继续解构)
                        auto [final_src_offset, k_stride_src] = ExtractStride(base_src, k_var_);

                        // 构造新的参数列表，包含循环次数
                        // 建议参数顺序：[dst, dst_off, src, src_off, size, i_extent, i_stride_dst, i_stride_src, k_stride_src]
                        // 这里的 size 保持原样 (如 8)，i_extent 传入 2
                        Array<PrimExpr> new_args = {
                            call->args[0],      // dst_buf
                            base_dst,           // 基础 dst 偏移
                            call->args[2],      // src_buf
                            final_src_offset,   // 基础 src 偏移
                            i_extent,           // i 循环次数
                            i_stride_dst,       // i 的 dst 步长
                            i_stride_src,       // i 的 src 步长
                            k_stride_src        // k 的 src 步长
                        };

                        return Evaluate(Call(call->dtype, call->op, new_args));
                    }
                }
            }
        }
        
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