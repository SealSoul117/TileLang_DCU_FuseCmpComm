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
 *        to local row-major layout using ds_read_vector hardware instructions.
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
    // 只处理 serial 外层循环
    if (op->kind != ForKind::kSerial) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // 判断是否是 B_local 写循环
    auto store = op->body.as<BufferStoreNode>();
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

    // 修改循环范围
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

  Stmt MutateStore(const BufferStoreNode* store,
                   const Var& loop_var) {

    Array<PrimExpr> new_indices = store->indices;

    PrimExpr new_value = store->value;

    // 修改切片跨度:
    // 原来 j*vec : j*vec+vec
    // 改为 j*vec : j*vec*expand + vec

    PrimExpr idx = store->indices[0];
    //T.Ramp(j * 4, 1, 4) -> Ramp(j*8, 1, 4)
    std::cout << idx << std::endl;

    // 解析 j*vec 结构
    // 假设结构为 j * vec + const

    // 不改 RHS
    // PrimExpr value = store->value;

    // 修改写入向量宽度
    // 原 value 是 Ramp(base=j*4, stride=1, lanes=4)
    // 匹配 j * stride
    // Ramp(base=j*8, stride=1, lanes=8)
    if (const auto* ramp = idx.as<RampNode>()) {

      PrimExpr base = ramp->base;
      PrimExpr stride = ramp->stride;
      int old_lanes = ramp->lanes.as<IntImmNode>()->value;
      int new_lanes = old_lanes * expand_;

      // 匹配 base = j * stride_val
      if (const auto* mul = base.as<MulNode>()) {

        if (mul->a.same_as(loop_var)) {
          int64_t old_stride =
              mul->b.as<IntImmNode>()->value;

          int64_t new_stride =
              old_stride * expand_;

          PrimExpr new_base =
              loop_var *
              make_const(DataType::Int(32), new_stride);

          new_indices.Set(
              0,
              Ramp(new_base, stride, new_lanes));
        }
        else if (mul->b.same_as(loop_var)) {
          int64_t old_stride =
              mul->a.as<IntImmNode>()->value;

          int64_t new_stride =
              old_stride * expand_;

          PrimExpr new_base =
              make_const(DataType::Int(32), new_stride) *
              loop_var;

          new_indices.Set(
              0,
              Ramp(new_base, stride, new_lanes));
        }
      }
    }

    if (auto* load = new_value.as<BufferLoadNode>()) {
      // BufferLoad with region access: B_shared[start : end]
      // end - start = lanes，需要同步扩展
      Array<PrimExpr> value_indices = load->indices;
      if (auto* old_ramp = load->indices[0].as<RampNode>()) {

          PrimExpr scalar_base = old_ramp->base;     // 必须是 scalar
          PrimExpr stride      = old_ramp->stride;
          //RHS 4 lane
          int old_lanes = old_ramp->lanes.as<IntImmNode>()->value;
          //RHS 8 lane
          int new_lanes = old_lanes * expand_;

          value_indices.Set(
              0,
              Ramp(scalar_base, stride, new_lanes)
          );

          new_value = BufferLoad(load->buffer, value_indices);
      }
    }

    return BufferStore(store->buffer,
                       new_value,
                       new_indices);
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
