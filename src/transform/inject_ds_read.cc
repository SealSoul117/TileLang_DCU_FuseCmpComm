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

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Check if the target is AMD DCU (gfx936, gfx942, etc.)
 */
bool IsDCUTarget(const IRModule& module) {
  for (auto& p : module->functions) {
    if (auto* prim_func = p.second.as<PrimFuncNode>()) {
      if (auto opt_target = prim_func->GetAttr<Target>("target")) {
        Target target = opt_target.value();
        if (target->attrs.count("mcpu")) {
          std::string mcpu = Downcast<tvm::ffi::String>(target->attrs.at("mcpu"));
          // if mcpu start with "gfx936", it is DCU
          return mcpu.find("gfx936") == 0;
        }
      }
    }
  }
  return false;
}

class DSReadInjector : public StmtMutator {
 public:
  Stmt VisitStmt_(const BufferStoreNode* store) final {
    // Check if the store is to a local register (not shared memory)
    bool is_local = store->buffer.scope() == "local" ||
                    store->buffer.scope() == "local.fragment";

    if (!is_local) {
      return StmtMutator::VisitStmt_(store);
    }

    // Check if the value is a BufferLoad from shared memory
    if (auto* load = store->value.as<BufferLoadNode>()) {
      bool is_shared_load = load->buffer.scope() == "shared" ||
                           load->buffer.scope() == "shared.dyn";

      if (!is_shared_load) {
        return StmtMutator::VisitStmt_(store);
      }

      // Skip if indices are vectorized (contain Ramp expressions)
      // ds_read is a scalar instruction, cannot handle vectorized indices
      if (HasVectorizedIndices(store->indices) || HasVectorizedIndices(load->indices)) {
        return StmtMutator::VisitStmt_(store);
      }

      // Check if the buffer is large enough for ds_read_vector
      // ds_read_vector<32, 16> with half_t reads 16 bytes (8 elements)
      // For small buffers (less than 16 bytes), skip this transformation
      if (store->buffer.defined()) {
        const auto& buffer_shape = store->buffer->shape;
        if (buffer_shape.size() == 1) {
          if (auto* int_shape = buffer_shape[0].as<IntImmNode>()) {
            int extent = int_shape->value;
            int dtype_bytes = load->dtype.bytes();
            // ds_read_vector<32,16> with half_t reads 16 bytes minimum
            // For buffers smaller than what ds_read_vector needs, skip
            if (extent * dtype_bytes < 16) {
              return StmtMutator::VisitStmt_(store);
            }
          }
        }
      }

      // Analyze the load pattern to determine which ds_read to use
      return InjectDSRead(store, load);
    }

    return StmtMutator::VisitStmt_(store);
  }

 private:

  // PrimExpr VisitExpr_(const CallNode *op)  {
  //   Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
  //   if (call->op.same_as(builtin::tvm_access_ptr())) {
  //     return RewriteBufferAccess(call, {1});
  //   }
  //   return call;
  // }
  /*!
   * \brief Check if any index expression contains a Ramp (vectorized) expression
   */
  bool HasVectorizedIndices(const Array<PrimExpr>& indices) {
    for (const auto& idx : indices) {
      if (idx.as<RampNode>()) {
        return true;
      }
    }
    return false;
  }
  Stmt InjectDSRead(const BufferStoreNode* store, const BufferLoadNode* load) {
    const Buffer& shared_buf = load->buffer;
    const Buffer& local_buf = store->buffer;

    // Analyze indices to determine the byte offset
    // PrimExpr offset = load->indices.size() > 0 ? load->indices[0] : make_zero(DataType::UInt(0));

    // Calculate buffer size in bytes
    int buffer_bytes = 0;
    if (local_buf.defined() && local_buf->shape.size() == 1) {
      if (auto* int_shape = local_buf->shape[0].as<IntImmNode>()) {
        int num_elements = int_shape->value;
        int dtype_bytes = local_buf->dtype.bytes();
        buffer_bytes = num_elements * dtype_bytes;
      }
    }

    // Determine which ds_read to use based on buffer size
    // ds_read_b64 loads 8 bytes (64 bits) = 1 element for half_t, 2 for float32
    // ds_read_m32x16_b16 loads 32 bytes (256 bits)
    int dtype_bits = local_buf->dtype.bits();
    int m = 16;

    // For buffer < 16 bytes, use single ds_read_b64 (M=32, N=1)
    // For buffer >= 16 bytes, use double ds_read_b64 (M=32, N=16)
    // ds_read_b64 reads 8 bytes per call
    int n = (buffer_bytes >= 32) ? 32 : 16;
    int offset = 0;

    return EmitDSRead(local_buf, shared_buf, m, n, offset);
  }

  Stmt EmitDSRead(const Buffer& local_buf,
                    const Buffer& shared_buf, int m, int n, int offset) {

    // ds_read_vector takes: (dst, shared_ptr, m, n, offset)
    Array<PrimExpr> args = {
        local_buf->data,                                    // dst: local buffer data pointer
        shared_buf.access_ptr(0, DataType::Handle(), 1, 0),    // src: shared buffer data pointer
        make_const(DataType::Int(32), m),
        make_const(DataType::Int(32), n),
        make_const(DataType::Int(32), offset)         // byte_offset: offset into shared memory
    };

    Stmt ds_read_stmt = Evaluate(
        Call(DataType::Handle(), ds_read_vector(), args));

    return ds_read_stmt;
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
