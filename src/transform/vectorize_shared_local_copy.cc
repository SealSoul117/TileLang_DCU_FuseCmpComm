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
 * \file vectorize_shared_local_copy.cc
 * \brief Merge inner loop dimensions into shared->local vector loads (max 16 bytes).
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "inject_utils.h"
#include "tir/ir/buffer_common.h"
#include "tvm/tir/stmt.h"

using tvm::ffi::Array;
using tvm::ffi::GetRef;
namespace tvm {
namespace tl {

using namespace tir;

namespace {

constexpr int kMaxVectorLoadBytes = 16;

bool IsLocalBuffer(const Buffer& buffer) {
  const auto& scope = buffer.scope();
  if (scope == "local") {
    return true;
  }
  std::string name = buffer->name;
  return name.find("_local") != std::string::npos;
}

bool IsSharedBuffer(const Buffer& buffer) {
  const auto& scope = buffer.scope();
  return scope == "shared" || scope == "shared.dyn";
}

std::pair<PrimExpr, PrimExpr> ExtractLinearCoeff(const PrimExpr& expr, const Var& var,
                                                   arith::Analyzer* analyzer) {
  PrimExpr at_zero = Substitute(expr, {{var, make_zero(var.dtype())}});
  PrimExpr at_one = Substitute(expr, {{var, make_const(var.dtype(), 1)}});
  PrimExpr stride = analyzer->Simplify(at_one - at_zero);
  return {analyzer->Simplify(at_zero), stride};
}

bool IsUnitStride(const PrimExpr& stride, arith::Analyzer* analyzer) {
  return analyzer->CanProveEqual(analyzer->Simplify(stride), make_const(stride.dtype(), 1));
}

bool IsConstantInt(const PrimExpr& expr, int64_t* out) {
  if (const auto* imm = expr.as<IntImmNode>()) {
    *out = imm->value;
    return true;
  }
  return false;
}

int MaxVectorLanes(DataType dtype) {
  return kMaxVectorLoadBytes * 8 / dtype.bits();
}

bool CanMergeInnerLoop(const BufferStoreNode* store, const Var& inner_var, int64_t inner_extent,
                       arith::Analyzer* analyzer, int64_t* merge_factor) {
  const auto* load = store->value.as<BufferLoadNode>();
  if (!load) {
    return false;
  }
  if (!IsLocalBuffer(store->buffer) || !IsSharedBuffer(load->buffer)) {
    return false;
  }
  if (store->indices.size() != 1 || load->indices.size() != 1) {
    return false;
  }

  const auto* dst_ramp = store->indices[0].as<RampNode>();
  const auto* src_ramp = load->indices[0].as<RampNode>();
  if (!dst_ramp || !src_ramp) {
    return false;
  }
  if (!IsUnitStride(dst_ramp->stride, analyzer) ||
      !IsUnitStride(src_ramp->stride, analyzer)) {
    return false;
  }

  int64_t current_lanes = 0;
  if (!IsConstantInt(dst_ramp->lanes, &current_lanes) ||
      !IsConstantInt(src_ramp->lanes, &current_lanes) || current_lanes <= 0) {
    return false;
  }

  auto [dst_base, dst_stride] = ExtractLinearCoeff(dst_ramp->base, inner_var, analyzer);
  auto [src_base, src_stride] = ExtractLinearCoeff(src_ramp->base, inner_var, analyzer);

  int64_t dst_stride_val = 0;
  int64_t src_stride_val = 0;
  if (!IsConstantInt(analyzer->Simplify(dst_stride), &dst_stride_val) ||
      !IsConstantInt(analyzer->Simplify(src_stride), &src_stride_val)) {
    return false;
  }
  if (dst_stride_val != src_stride_val || dst_stride_val != current_lanes) {
    return false;
  }

  const int max_lanes = MaxVectorLanes(store->buffer->dtype);
  int64_t new_lanes = current_lanes * inner_extent;
  if (new_lanes > max_lanes) {
    return false;
  }

  *merge_factor = inner_extent;
  return true;
}

BufferStore MergeInnerLoopStore(const BufferStoreNode* store, const Var& inner_var,
                                int64_t merge_factor, arith::Analyzer* analyzer) {
  const auto* load = store->value.as<BufferLoadNode>();
  ICHECK(load != nullptr);

  const auto* dst_ramp = store->indices[0].as<RampNode>();
  const auto* src_ramp = load->indices[0].as<RampNode>();
  ICHECK(dst_ramp != nullptr && src_ramp != nullptr);

  int64_t current_lanes = 0;
  IsConstantInt(dst_ramp->lanes, &current_lanes);
  const int64_t new_lanes = current_lanes * merge_factor;

  auto [dst_base, _dst_stride] = ExtractLinearCoeff(dst_ramp->base, inner_var, analyzer);
  auto [src_base, _src_stride] = ExtractLinearCoeff(src_ramp->base, inner_var, analyzer);

  Array<PrimExpr> dst_indices = {
      Ramp(dst_base, make_const(dst_base.dtype(), 1), Integer(new_lanes))};
  Array<PrimExpr> src_indices = {
      Ramp(src_base, make_const(src_base.dtype(), 1), Integer(new_lanes))};
  BufferLoad new_load(load->buffer, src_indices);
  return BufferStore(store->buffer, new_load, dst_indices);
}

}  // namespace

class SharedLocalCopyVectorizer : public StmtExprMutator {
 public:
  SharedLocalCopyVectorizer() = default;

 private:
  arith::Analyzer analyzer_;

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt body = StmtExprMutator::VisitStmt(op->body);

    if (const auto* inner_for = body.as<ForNode>()) {
      if (op->kind == ForKind::kSerial && inner_for->kind == ForKind::kSerial &&
          is_zero(inner_for->min)) {
        if (const auto* store = inner_for->body.as<BufferStoreNode>()) {
          int64_t inner_extent = 0;
          if (IsConstantInt(analyzer_.Simplify(inner_for->extent), &inner_extent) &&
              inner_extent > 1) {
            int64_t merge_factor = 0;
            if (CanMergeInnerLoop(store, inner_for->loop_var, inner_extent, &analyzer_,
                                  &merge_factor)) {
              BufferStore merged =
                  MergeInnerLoopStore(store, inner_for->loop_var, merge_factor, &analyzer_);
              auto outer = GetRef<For>(op);
              outer.CopyOnWrite()->body = std::move(merged);
              return outer;
            }
          }
        }
      }
    }

    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    auto n = CopyOnWrite(op);
    n->body = std::move(body);
    return Stmt(n);
  }
};

Stmt VectorizeSharedLocalCopyPass(Stmt stmt) {
  return SharedLocalCopyVectorizer()(std::move(stmt));
}

using namespace tir::transform;

tvm::transform::Pass VectorizeSharedLocalCopy() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    if (!IsDCUTarget(m)) {
      return f;
    }
    auto* n = f.CopyOnWrite();
    n->body = VectorizeSharedLocalCopyPass(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VectorizeSharedLocalCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VectorizeSharedLocalCopy", VectorizeSharedLocalCopy);
}

}  // namespace tl
}  // namespace tvm
