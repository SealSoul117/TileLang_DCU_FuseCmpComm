#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>

#include <vector>

#include "inject_utils.h"

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
    Var main_loop_var_;
    PrimExpr main_loop_extent_;
    bool in_unrolled_i_ = false;

    std::pair<PrimExpr, PrimExpr> ExtractStride(PrimExpr expr, Var var) {
        if (!var.defined()) return {expr, make_zero(expr->dtype)};

        PrimExpr base = tvm::tir::Substitute(expr, {{var, make_zero(var.dtype())}});
        PrimExpr plus_one = tvm::tir::Substitute(expr, {{var, make_const(var.dtype(), 1)}});
        PrimExpr stride = analyzer_.Simplify(plus_one - base);
        return {analyzer_.Simplify(base), stride};
    }

    Stmt RewriteAsyncCopy(const CallNode* call, Var i_var, PrimExpr i_extent) {
        PrimExpr raw_dst_off = call->args[1];
        PrimExpr raw_src_off = call->args[3];
        if (const RampNode* r = raw_dst_off.as<RampNode>()) raw_dst_off = r->base;
        if (const RampNode* r = raw_src_off.as<RampNode>()) raw_src_off = r->base;

        auto [base_i_dst, i_stride_dst] = ExtractStride(raw_dst_off, i_var);
        auto [base_i_src, i_stride_src] = ExtractStride(raw_src_off, i_var);

        auto [final_src_offset, k_stride_src] = ExtractStride(base_i_src, main_loop_var_);

        Array<PrimExpr> new_args = {
            call->args[0],
            base_i_dst,    
            call->args[2],   
            final_src_offset,  
            i_extent,       
            i_stride_dst,    
            i_stride_src,  
            k_stride_src   
        };

        return Evaluate(Call(call->dtype, call->op, new_args));
    }

    // 处理无循环包裹的情况
    Stmt VisitStmt_(const EvaluateNode* op) final {
        if (!in_unrolled_i_) {
            if (const CallNode* call = op->value.as<CallNode>()) {
                static const Op& dcu_copy_op = Op::Get("tl.dcu_async_copy");
                if (call->op.same_as(dcu_copy_op) && call->args.size() != 8) {
                    return RewriteAsyncCopy(call, Var(), make_zero(DataType::Int(32)));
                }
            }
        }
        return StmtExprMutator::VisitStmt_(op);
    }

    Stmt VisitStmt_(const ForNode* op) final {
        bool is_main_loop = IsDCUMainLoop(op);
        if (is_main_loop) {
            main_loop_var_ = op->loop_var;
            main_loop_extent_ = op->extent;
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
                        Stmt result = RewriteAsyncCopy(call, op->loop_var, op->extent);
                        return result; 
                    }
                }
            }
        }
        
        if (is_main_loop) {
            main_loop_var_ = Var();
            main_loop_extent_ = PrimExpr();
        }

        if (body.same_as(op->body)) return GetRef<Stmt>(op);
        auto n = CopyOnWrite(op);
        n->body = std::move(body);
        return Stmt(n);
    }
};

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