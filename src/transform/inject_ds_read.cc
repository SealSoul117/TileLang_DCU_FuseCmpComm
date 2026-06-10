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

class DSReadInjector : public StmtExprMutator {
 public:
  bool IsBLocalBuffer(const Buffer& buffer) {
    std::string name = buffer->name;
    return name.find("B_local") != std::string::npos;
  }

 private:

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Buffer buffer = op->buffer;

  if (!IsBLocalBuffer(buffer)) {
    return StmtExprMutator::VisitStmt_(op);
  }

  const BufferLoadNode* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return StmtExprMutator::VisitStmt_(op);
  }

  // local offset
  ICHECK(op->indices.size() == 1);
  PrimExpr local_index = op->indices[0];
  PrimExpr local_offset;

  if (const RampNode* ramp = local_index.as<RampNode>()) {
      local_offset = ramp->base;
  } else {
      local_offset = local_index;
  }

  // lds offset
  ICHECK(load->indices.size() == 1);
  PrimExpr lds_index = load->indices[0];
  PrimExpr lds_offset ;

  if (const RampNode* ramp = lds_index.as<RampNode>()) {
    lds_offset = ramp->base;
  } else {
    lds_offset = lds_index;
  }

  // buffer pointer
  PrimExpr buffer_ptr = buffer->data;

  Array<PrimExpr> args = {
      buffer_ptr,
      local_offset,
      lds_offset
  };

  const Op& ds_op = buffer->dtype.bytes() == 1 ? ds_read_m32x32_b8() : ds_read_m32x16_b16();
  ICHECK(buffer->dtype.bytes() == 1 || buffer->dtype.bytes() == 2)
      << "InjectDSRead only supports 1-byte and 2-byte element types, got "
      << buffer->dtype;

  Call call = Call(
      DataType::Handle(),
      ds_op,
      args
  );

  return Evaluate(call);
  }
};

using namespace tir::transform;

tvm::transform::Pass InjectDSRead() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Only apply to DCU targets
    if (!IsDCUTarget(m)) {
      return f;
    }

    auto *n = f.CopyOnWrite();
    n->body = DSReadInjector()(n->body);
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