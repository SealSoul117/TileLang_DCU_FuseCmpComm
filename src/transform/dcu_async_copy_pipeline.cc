#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/analysis.h>
using namespace tvm::tir;

using tvm::ffi::GetRef;
using tvm::ffi::make_object;

namespace tvm {
namespace tl {
using namespace tir;
using ffi::Array;
using ffi::String;

class ROCmWaitCountRewriter : public StmtMutator {
 public:
  static Stmt Substitute(Stmt stmt) {
    return ROCmWaitCountRewriter()(stmt);
  }

 private:
  // 辅助函数：统计一个代码块内 async 指令的总数
  int CountAsyncOps(const Stmt& stmt) {
    int total_count = 0;

    struct Visitor : public StmtExprVisitor {
      int count = 0;
      void VisitStmt_(const ForNode* op) override {
        // 如果内部还有循环（比如 T.unroll），需要乘上循环次数
        int current_count = count;
        count = 0;
        StmtExprVisitor::VisitStmt_(op);
        
        int loop_count = 0;
        if (const auto* extent = op->extent.as<IntImmNode>()) {
            loop_count = static_cast<int>(extent->value);
        } else {
            // 如果是非固定长度循环，这在流水线中很少见，默认按1处理或报警
            loop_count = 1;
        }
        
        int body_count = count;
        count = current_count + (body_count * loop_count);
      }

      void VisitExpr_(const CallNode* op) override {
        // 识别 ptx_cp_async 或对应的异步访存 Op
        if (op->op.same_as(builtin::ptx_cp_async()) || 
            op->op.same_as(Op::Get("tl.dcu_async_copy"))) {
                LOG(INFO) << "Found async copy: " << GetRef<Call>(op);
          count++;
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      
      // 兼容某些实现中把 cp_async 放在 Evaluate 里的情况
      void VisitStmt_(const EvaluateNode* op) override {
        StmtExprVisitor::VisitStmt_(op);
      }
    } visitor;

    visitor(stmt);
    return visitor.count;
  }

  Stmt VisitStmt_(const ForNode* op) override {
    // 1. 我们假设流水线的主循环是核心作用域
    // 先扫描该循环体内部每一轮会发出多少个 async 操作
    int ops_per_iter = CountAsyncOps(op->body);

    // 如果没有异步操作，直接跳过
    if (ops_per_iter == 0) return StmtMutator::VisitStmt_(op);

    // 2. 进入循环内部进行修改，记录当前的倍数
    int old_multiplier = multiplier_;
    multiplier_ = ops_per_iter;
    Stmt new_body = this->VisitStmt(op->body);
    multiplier_ = old_multiplier;

    if (new_body.same_as(op->body)) return GetRef<Stmt>(op);
    
    auto n = CopyOnWrite(op);
    n->body = std::move(new_body);
    return Stmt(n);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == "async_wait_inflight_count" && multiplier_ > 0) {
      // 获取原有的 wait 组数 (比如 1)
      if (auto int_imm = op->value.as<IntImmNode>()) {
        // 计算 ROCm 的指令数: N_groups * Ops_per_group
        int64_t new_cont = int_imm->value * multiplier_;

        LOG(INFO) << "Original wait count: " << new_cont << ", async ops per iter: " << multiplier_;
        
        // 返回修改后的节点
        return AttrStmt(op->node, op->attr_key, make_const(DataType::Int(32), new_cont), op->body);
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  int multiplier_ = 0; // 当前作用域下的指令倍率
};

// 包装成标准的 TVM Pass
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
}  // namespace transform

}  // namespace tl
}  // namespace tvm