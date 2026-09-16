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
 * \brief Replace shared memory BufferLoad with ds_read hardware instructions
 * \file inject_ds_read.cc
 */
#include <iostream>
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

namespace {

// ds_read_m32x16_b16 / ds_read_m32x32_b8 write a float32x4 (16 bytes) per thread.
// Only consecutive vector BufferLoad/Store of that width can be rewritten.
bool TryGetConsecutiveVectorOffset(const PrimExpr& index, int expected_lanes,
                                   PrimExpr* offset) {
  const auto* ramp = index.as<RampNode>();
  if (!ramp) {
    return false;
  }
  if (!is_const_int(ramp->stride, 1) || !is_const_int(ramp->lanes, expected_lanes)) {
    return false;
  }
  *offset = ramp->base;
  return true;
}

}  // namespace

class DSReadInjector : public StmtExprMutator {
 public:
  explicit DSReadInjector(const DCUCopyBufferMaps& maps) : maps_(maps) {}

 private:
  DCUCopyBufferMaps maps_;

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Buffer buffer = op->buffer;

    const BufferLoadNode* load = op->value.as<BufferLoadNode>();
    if (!load) {
      return StmtExprMutator::VisitStmt_(op);
    }

    const auto dst_scope = std::string(buffer.scope());
    const auto src_scope = std::string(load->buffer.scope());
    if (dst_scope.find("local") == std::string::npos ||
        (src_scope != "shared" && src_scope != "shared.dyn")) {
      return StmtExprMutator::VisitStmt_(op);
    }

    DCUGemmTransposeFlags unused_flags;
    if (!NeedsDCULocalLayoutDSRead(unused_flags, maps_, std::string(buffer->name),
                                   std::string(load->buffer->name))) {
      return StmtExprMutator::VisitStmt_(op);
    }

    const int elem_bytes = buffer->dtype.bytes();
    if (elem_bytes != 1 && elem_bytes != 2) {
      return StmtExprMutator::VisitStmt_(op);
    }
    const int expected_lanes = 16 / elem_bytes;

    if (op->indices.size() != 1 || load->indices.size() != 1) {
      return StmtExprMutator::VisitStmt_(op);
    }

    PrimExpr local_offset;
    PrimExpr lds_offset;
    if (!TryGetConsecutiveVectorOffset(op->indices[0], expected_lanes, &local_offset) ||
        !TryGetConsecutiveVectorOffset(load->indices[0], expected_lanes, &lds_offset)) {
      return StmtExprMutator::VisitStmt_(op);
    }

    Array<PrimExpr> args = {buffer->data, local_offset, lds_offset};
    const Op& ds_op = elem_bytes == 1 ? ds_read_m32x32_b8() : ds_read_m32x16_b16();
    Call call = Call(DataType::Handle(), ds_op, args);
    return Evaluate(call);
  }
};

using namespace tir::transform;

tvm::transform::Pass InjectDSRead() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    if (!IsDCUTarget(m)) {
      return f;
    }

    DCUCopyBufferMaps maps = GetDCUCopyBufferMaps(f);
    if (DCUCopyBufferMapsEmpty(maps)) {
      return f;
    }

    auto *n = f.CopyOnWrite();
    n->body = DSReadInjector(maps)(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectDSRead", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectDSRead", InjectDSRead);
}

}  // namespace tl
}  // namespace tvm