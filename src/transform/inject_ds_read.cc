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
  /*!
   * \brief Visit EvaluateNode to handle explicit ds_read_vector call
   * ds_read_vector Call is wrapped in Evaluate to become a statement
   * Parameters m, n, offset are passed explicitly via CallNode args
   */
  Stmt VisitStmt_(const EvaluateNode* op) override {
    std::cout << "[DEBUG VisitStmt_] Visiting EvaluateNode" << std::endl;
    const CallNode* call = op->value.as<CallNode>();
    std::cout << "[DEBUG VisitStmt_] CallNode ptr: " << call << std::endl;
    if (call != nullptr && call->op.same_as(ds_read_vector())) {
      ICHECK(call->args.size() == 5)
          << "ds_read_vector expects 5 arguments: (dst, src, m, n, offset)";

      // Print args for debugging - these are the actual CallNode args passed in
      std::cout << "[DEBUG ds_read_vector] args[0] (dst): " << call->args[0] << std::endl;
      std::cout << "[DEBUG ds_read_vector] args[1] (src): " << call->args[1] << std::endl;
      std::cout << "[DEBUG ds_read_vector] args[2] (m): " << call->args[2] << std::endl;
      std::cout << "[DEBUG ds_read_vector] args[3] (n): " << call->args[3] << std::endl;
      std::cout << "[DEBUG ds_read_vector] args[4] (offset): " << call->args[4] << std::endl;
    }
    // Continue with default traversal (don't replace the existing call)
    return StmtExprMutator::VisitStmt_(op);
  }

  /*!
   * \brief Visit BufferStoreNode to inject ds_read_vector call
   * Pattern: local_buffer[...] = shared_buffer[...] (BufferLoad)
   * Parameters m, n, offset are passed via a CallNode (tl.ds_read_config)
   */
  Stmt VisitStmt_(const BufferStoreNode* op) override {
    std::cout << "[DEBUG VisitStmt_] Visiting BufferStoreNode" << std::endl;

    // Check if the store is to a local register (not shared memory)
    bool is_local = op->buffer.scope() == "local" ||
                    op->buffer.scope() == "local.fragment";
    std::cout << "[DEBUG BufferStore] is_local: " << is_local
              << ", scope: " << op->buffer.scope() << std::endl;

    if (!is_local) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // Check if the value is a BufferLoad from shared memory
    const BufferLoadNode* load = op->value.as<BufferLoadNode>();
    if (load == nullptr) {
      std::cout << "[DEBUG BufferStore] value is not BufferLoad" << std::endl;
      return StmtExprMutator::VisitStmt_(op);
    }

    bool is_shared_load = load->buffer.scope() == "shared" ||
                         load->buffer.scope() == "shared.dyn";
    std::cout << "[DEBUG BufferStore] is_shared_load: " << is_shared_load
              << ", load scope: " << load->buffer.scope() << std::endl;

    if (!is_shared_load) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // For A_shared, use the actual shared memory base pointer
    PrimExpr m = make_const(DataType::Int(32), 32);
    PrimExpr n = make_const(DataType::Int(32), 16);
    PrimExpr offset = make_const(DataType::Int(32), 0);

    // Use buffer data vars directly
    Array<PrimExpr> new_args = {
        load->buffer->data,   // src
        op->buffer->data,      // dst
        m,
        n,
        offset
    };

    // Create the ds_read call
    Call ds_read_call = Call(DataType::Handle(), ds_read_vector(), new_args);
    return Evaluate(ds_read_call);
  }
};

using namespace tir::transform;

tvm::transform::Pass InjectDSRead() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    std::cout << "[DEBUG InjectDSRead] Pass is being executed" << std::endl;
    // Only apply to DCU targets
    if (!IsDCUTarget(m)) {
      std::cout << "[DEBUG InjectDSRead] Not a DCU target, skipping" << std::endl;
      return f;
    }

    std::cout << "[DEBUG InjectDSRead] Is DCU target, applying injector" << std::endl;
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