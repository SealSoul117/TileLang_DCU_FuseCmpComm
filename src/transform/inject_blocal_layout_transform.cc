/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file inject_blocal_layout_transform.cc
 * \brief Transform B_local layout from shared memory thread-interleaved layout
 *        to local row-major layout using ds_read hardware instructions.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "tir/ir/buffer_common.h"
#include "tvm/tir/stmt.h"
#include "inject_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Transformer that handles B_local layout transformation with loop optimization
 *
 * This transformer handles two cases:
 * 1. B_local store with outer loop: halve the loop extent and double the offset
 * 2. B_local store without outer loop: just double the offset
 */
class BLocalLayoutTransformer : public StmtExprMutator {
 public:
  explicit BLocalLayoutTransformer(int expand)
      : expand_(expand) {}

 private:
  int expand_;

  Stmt VisitStmt_(const ForNode* op) final {
    // 1. 先递归处理子节点（重要：确保处理了嵌套的 For 或 Attr）
    Stmt new_body = this->VisitStmt(op->body);
    
    // 2. 检查当前循环是否是目标循环
    // 即使 body 变了，我们也尝试看看能不能在这个 loop 层级做变换
    auto store = new_body.as<BufferStoreNode>();
    if (op->kind != ForKind::kSerial) {
      return StmtExprMutator::VisitStmt_(op);
    }
    if (!store) {
      return StmtExprMutator::VisitStmt_(op);
    }

    if (!IsBLocal(store->buffer)) {
      return StmtExprMutator::VisitStmt_(op);
    }

    int64_t old_extent = op->extent.as<IntImmNode>()->value;

    ICHECK(old_extent % expand_ == 0)
        << "Loop extent must be divisible by expand factor.";

    int64_t new_extent = old_extent / expand_;

    For new_for =
        For(op->loop_var,
            op->min,
            Integer(new_extent),
            op->kind,
            MutateStore(store, op->loop_var));

    return new_for;
  }

  bool IsBLocal(const Buffer& buffer) {
    std::string name = buffer->name;
    return name.find("B_local") != std::string::npos;
  }
  PrimExpr UpdateIndexBase(PrimExpr base, const Var& loop_var, int expand) {
      if (const auto* add = base.as<AddNode>()) {
          return UpdateIndexBase(add->a, loop_var, expand) + 
                UpdateIndexBase(add->b, loop_var, expand);
      } else if (const auto* mul = base.as<MulNode>()) {
          if (mul->a.same_as(loop_var)) {
              return mul->a * (mul->b * expand);
          } else if (mul->b.same_as(loop_var)) {
              return (mul->a * expand) * mul->b;
          }
      }
      return base; 
  }
  Stmt MutateStore(const BufferStoreNode* store, const Var& loop_var) {
      auto n = tvm::ffi::make_object<BufferStoreNode>(*store);
      Array<PrimExpr> new_indices = store->indices;

      if (const auto* ramp = store->indices[0].as<RampNode>()) {
          PrimExpr new_base = UpdateIndexBase(ramp->base, loop_var, expand_);
          
          int new_lanes = ramp->lanes.as<IntImmNode>()->value * expand_;
          
          new_indices.Set(0, Ramp(new_base, ramp->stride, new_lanes));
      }

      PrimExpr new_value = store->value;
      if (const auto* load = store->value.as<BufferLoadNode>()) {
          if (const auto* l_ramp = load->indices[0].as<RampNode>()) {
              Array<PrimExpr> v_indices = load->indices;
              int v_new_lanes = l_ramp->lanes.as<IntImmNode>()->value * expand_;
              v_indices.Set(0, Ramp(l_ramp->base, l_ramp->stride, v_new_lanes));
              new_value = BufferLoad(load->buffer, v_indices);
          }
      }

      return BufferStore(store->buffer, new_value, new_indices);
  }
};


Stmt InjectBLocalLayoutTransformPass(Stmt stmt, int expand) {
  return BLocalLayoutTransformer(expand)(std::move(stmt));
}


using namespace tir::transform;

tvm::transform::Pass InjectBLocalLayoutTransform() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Only apply to DCU targets
    if (!IsDCUTarget(m)) {
      std::cout << "[DEBUG InjectBLocalLayoutTransform] Not a DCU target, skipping" << std::endl;
      return f;
    }

    auto* n = f.CopyOnWrite();
    n->body = InjectBLocalLayoutTransformPass(n->body, 2);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectBLocalLayoutTransform", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectBLocalLayoutTransform",
                        InjectBLocalLayoutTransform);
}

}  // namespace tl
}  // namespace tvm
