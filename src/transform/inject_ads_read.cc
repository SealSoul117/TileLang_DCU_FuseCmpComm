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
 * \brief Replace A_local BufferLoad from shared memory with ds_read_b64/b128
 * \file inject_ads_read.cc
 */
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "inject_utils.h"
#include "tir/ir/buffer_common.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {

using namespace tir;

class ADSReadInjector : public StmtExprMutator {
 public:
  bool IsALocalBuffer(const Buffer& buffer) {
    std::string name = buffer->name;
    return name.find("A_local") != std::string::npos;
  }

  bool IsSharedBuffer(const Buffer& buffer) {
    std::string scope = buffer.scope();
    std::string name = buffer->name;
    return scope == "shared" || scope == "shared.dyn" ||
           name.find("shared") != std::string::npos ||
           name.find("shmem") != std::string::npos;
  }

 private:
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Buffer buffer = op->buffer;

    if (!IsALocalBuffer(buffer)) {
      return StmtExprMutator::VisitStmt_(op);
    }

    const BufferLoadNode* load = op->value.as<BufferLoadNode>();
    if (!load || !IsSharedBuffer(load->buffer)) {
      return StmtExprMutator::VisitStmt_(op);
    }

    ICHECK(op->indices.size() == 1);
    PrimExpr local_index = op->indices[0];
    PrimExpr local_offset;
    int load_bytes = 0;

    if (const RampNode* ramp = local_index.as<RampNode>()) {
      local_offset = ramp->base;
      int lanes = Downcast<IntImm>(ramp->lanes)->value;
      load_bytes = lanes * buffer->dtype.bytes();
    } else {
      local_offset = local_index;
      load_bytes = buffer->dtype.bytes();
    }

    ICHECK(load->indices.size() == 1);
    PrimExpr lds_index = load->indices[0];
    PrimExpr lds_offset;

    if (const RampNode* ramp = lds_index.as<RampNode>()) {
      lds_offset = ramp->base;
      int src_lanes = Downcast<IntImm>(ramp->lanes)->value;
      ICHECK_EQ(src_lanes * buffer->dtype.bytes(), load_bytes)
          << "InjectADSRead: A_local and A_shared ramp sizes must match";
    } else {
      lds_offset = lds_index;
    }

    PrimExpr buffer_ptr = buffer->data;

    const Op& ds_op = load_bytes == 16 ? ds_read_b128() : ds_read_b64();
    ICHECK(load_bytes == 8 || load_bytes == 16)
        << "InjectADSRead only supports 8-byte (ds_read_b64) and 16-byte (ds_read_b128) "
        << "loads, got " << load_bytes << " bytes for dtype " << buffer->dtype;

    DataType elem_dtype = buffer->dtype;
    if (elem_dtype.lanes() > 1) {
      elem_dtype = elem_dtype.element_of();
    }
    Array<PrimExpr> args = {
        buffer_ptr,
        local_offset,
        lds_offset,
        StringImm(ffi::DLDataTypeToString(elem_dtype)),
    };

    Call call = Call(DataType::Handle(), ds_op, args);

    return Evaluate(call);
  }
};

using namespace tir::transform;

tvm::transform::Pass InjectADSRead() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    if (!IsDCUTarget(m)) {
      return f;
    }

    DCUGemmTransposeFlags flags = GetDCUGemmTransposeFlags(f);
    if (!NeedsDCUADSRead(flags)) {
      return f;
    }

    auto* n = f.CopyOnWrite();
    n->body = ADSReadInjector()(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectADSRead", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectADSRead", InjectADSRead);
}

}  // namespace tl
}  // namespace tvm
