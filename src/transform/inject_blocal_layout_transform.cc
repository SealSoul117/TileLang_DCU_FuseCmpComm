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
 * \brief Check if a statement contains B_local stores
 */
bool ContainsBLocalStore(const Stmt& stmt) {
  bool found = false;
  tir::PreOrderVisit(stmt, [&](const ObjectRef& node) -> bool {
    if (found) {
      return false;
    }
    if (const auto* store = node.as<BufferStoreNode>()) {
      std::string name = store->buffer->name;
      if (name.find("B_local") != std::string::npos) {
        found = true;
        return false;
      }
    }
    return true;
  });
  return found;
}

/*!
 * \brief Check if this is a B_local store pattern
 *
 * Pattern to match:
 *   B_local[index] = B_shared[index_expr]
 *
 * Where B_shared[index_expr] is a complex expression involving:
 *   - thread_binding (threadIdx.x, threadIdx.y, etc.)
 *   - ki (iteration variable)
 *   - j and local_id (loop variables)
 */
bool IsBLocalStorePattern(const BufferStoreNode* op,
                          Var* local_var,
                          Var* shared_var,
                          PrimExpr* shared_offset) {
  // Check if store is to a local buffer named B_local
  std::string buffer_name = op->buffer->name;
  if (buffer_name.find("B_local") == std::string::npos) {
    return false;
  }

  // Must have exactly one index: B_local[index]
  if (op->indices.size() != 1) {
    return false;
  }

  // Check if value is a BufferLoad from shared memory
  const BufferLoadNode* load = op->value.as<BufferLoadNode>();
  if (load == nullptr) {
    return false;
  }

  // Check if load is from shared memory
  std::string load_buffer_name = load->buffer->name;
  std::cout<<"[DEBUG IsBLocalStorePattern] load buffer name: " << load_buffer_name << std::endl;
  if (load_buffer_name.find("B_shared") == std::string::npos) {
    return false;
  }

  // Get buffer variables
  *local_var = op->buffer->data;
  *shared_var = load->buffer->data;

  // Extract the shared memory offset from the load indices
  if (!load->indices.empty()) {
    *shared_offset = load->indices[0];
  } else {
    *shared_offset = make_const(DataType::Int(32), 0);
  }

  return true;
}

class BLocalLayoutTransformer : public StmtExprMutator {
 public:
  BLocalLayoutTransformer(const IRModule& module) : module_(module) {}

  Stmt VisitStmt_(const BufferStoreNode* op) override {
    // Check if this is a B_local store pattern BEFORE visiting
    // to get the original buffer->data vars (not mutated by VisitStmt_)
    Var local_var;
    Var shared_var;
    PrimExpr shared_offset;
    if (!IsBLocalStorePattern(op, &local_var, &shared_var, &shared_offset)) {
      // Only visit if not our target pattern
      return Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    }
    std::cout<<"[DEBUG BLocalLayoutTransformer VisitStmt_] BufferStoreNode buffer name: " << op->buffer->name << std::endl;
    // For ds_read_vector: ds_read_vector(dst, src, m, n, offset)
    // m, n describe the 2D layout of the shared memory tile
    // For B_local (16x32 tile): m=16, n=32
    PrimExpr m = make_const(DataType::Int(32), 16);
    PrimExpr n = make_const(DataType::Int(32), 32);
    PrimExpr offset = shared_offset;


    // Create the ds_read call
    // ds_read_vector(local_ptr, shared_ptr, m, n, offset)
    // Use the vars directly - don't call VisitExpr on them as that creates new Vars
    Array<PrimExpr> ds_read_args = {
        local_var,    // dst: local buffer pointer
        op->buffer->data,    // src: shared memory pointer
        m,             // m: rows in shared memory tile
        n,             // n: columns in shared memory tile
        offset         // offset: starting offset in shared memory
    };

    Call ds_read_call = Call(DataType::Handle(), ds_read_vector(), ds_read_args);

    // Replace the BufferStore with the ds_read call
    return Evaluate(ds_read_call);
  }

 private:
  const IRModule& module_;
};

/*!
 * \brief Inject prefetch for B_local using ds_read_vector
 */
class BLocalPrefetchInjector : public StmtMutator {
 public:
  BLocalPrefetchInjector(const IRModule& module) : module_(module) {}

  Stmt VisitStmt_(const ForNode* op) override {
    if (op->kind == ForKind::kParallel || op->kind == ForKind::kSerial ||
        op->kind == ForKind::kVectorized) {
      Stmt body = VisitStmt(op->body);

      // Check if body contains B_local stores
      if (ContainsBLocalStore(body)) {
        // Inject prefetch before the loop
        Stmt prefetch = GenerateBLocalPrefetch();
        return SeqStmt({prefetch, For(op->loop_var, op->min, op->extent,
                                       op->kind, body, op->thread_binding,
                                       op->annotations)});
      }

      return For(op->loop_var, op->min, op->extent, op->kind, body,
                 op->thread_binding, op->annotations);
    }

    return StmtMutator::VisitStmt_(op);
  }

 private:
  Stmt GenerateBLocalPrefetch() {
    // Placeholder: actual implementation depends on the specific
    // shared memory layout and thread block configuration
    return Evaluate(0);
  }

  const IRModule& module_;
};

using namespace tir::transform;

tvm::transform::Pass InjectBLocalLayoutTransform() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Only apply to DCU targets
    if (!IsDCUTarget(m)) {
      std::cout << "[DEBUG InjectBLocalLayoutTransform] Not a DCU target, skipping" << std::endl;
      return f;
    }

    auto* n = f.CopyOnWrite();
    n->body = BLocalLayoutTransformer(m)(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectBLocalLayoutTransform", {});
}

tvm::transform::Pass InjectBLocalLayoutTransformWithPrefetch() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Only apply to DCU targets
    if (!IsDCUTarget(m)) {
      return f;
    }

    auto* n = f.CopyOnWrite();
    n->body = BLocalPrefetchInjector(m)(n->body);
    n->body = BLocalLayoutTransformer(m)(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectBLocalLayoutTransformWithPrefetch", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectBLocalLayoutTransform",
                        InjectBLocalLayoutTransform);
  refl::GlobalDef().def("tl.transform.InjectBLocalLayoutTransformWithPrefetch",
                        InjectBLocalLayoutTransformWithPrefetch);
}

}  // namespace tl
}  // namespace tvm
