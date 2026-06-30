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
#include <tvm/tir/expr_functor.h>
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

bool ExprContainsVar(const PrimExpr& expr, const Var& var) {
  bool found = false;
  auto visit = [&](const ObjectRef& node) {
    if (const auto* v = node.as<VarNode>()) {
      if (v == var.get()) {
        found = true;
      }
    }
  };
  PostOrderVisit(expr, visit);
  return found;
}

class VarOnlyInFloorModInner : public ExprVisitor {
 public:
  VarOnlyInFloorModInner(const Var& var, const FloorModNode* mod) : var_(var), mod_(mod) {}

  bool IsValid() const { return !found_outside_; }

 private:
  void VisitExpr_(const VarNode* op) final {
    if (op == var_.get() && !inside_mod_inner_) {
      found_outside_ = true;
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const FloorModNode* op) final {
    if (op == mod_) {
      bool was_inside = inside_mod_inner_;
      inside_mod_inner_ = true;
      VisitExpr(op->a);
      inside_mod_inner_ = false;
      VisitExpr(op->b);
      inside_mod_inner_ = was_inside;
      return;
    }
    ExprVisitor::VisitExpr_(op);
  }

  const Var& var_;
  const FloorModNode* mod_;
  bool inside_mod_inner_{false};
  bool found_outside_{false};
};

struct FloorModAffineMatch {
  const FloorModNode* mod{nullptr};
  PrimExpr inner_base_at_zero;
  int64_t coeff{0};
  int64_t mod_val{0};
};

bool TryMatchFloorModAffine(const PrimExpr& expr, const Var& var, arith::Analyzer* analyzer,
                            FloorModAffineMatch* match) {
  if (!ExprContainsVar(expr, var)) {
    return false;
  }

  struct FloorModCollector : public ExprVisitor {
    std::vector<const FloorModNode*> mods;
    void VisitExpr_(const FloorModNode* op) final {
      mods.push_back(op);
      ExprVisitor::VisitExpr_(op);
    }
  } collector;
  collector(expr);

  for (const FloorModNode* mod : collector.mods) {
    if (!ExprContainsVar(mod->a, var)) {
      continue;
    }
    VarOnlyInFloorModInner checker(var, mod);
    checker(expr);
    if (!checker.IsValid()) {
      continue;
    }

    auto [inner_base, inner_stride] = ExtractLinearCoeff(mod->a, var, analyzer);
    int64_t coeff = 0;
    int64_t mod_val = 0;
    if (!IsConstantInt(analyzer->Simplify(inner_stride), &coeff) || coeff <= 0) {
      continue;
    }
    if (!IsConstantInt(mod->b, &mod_val) || mod_val <= 0) {
      continue;
    }

    match->mod = mod;
    match->inner_base_at_zero = analyzer->Simplify(inner_base);
    match->coeff = coeff;
    match->mod_val = mod_val;
    return true;
  }
  return false;
}

bool CanProveFloorModMergeSafe(arith::Analyzer* analyzer, const FloorModAffineMatch& match,
                               int64_t inner_extent, int64_t current_lanes) {
  const int64_t total_lanes = inner_extent * current_lanes;
  ICHECK_GT(total_lanes, 0);

  const PrimExpr mod_expr = make_const(match.inner_base_at_zero.dtype(), match.mod_val);
  const PrimExpr residue =
      analyzer->Simplify(floormod(match.inner_base_at_zero, mod_expr));
  const PrimExpr end = residue + make_const(residue.dtype(), total_lanes);
  if (analyzer->CanProve(end <= mod_expr)) {
    return true;
  }

  // DCU swizzled layouts keep the pre-mod base aligned to 8 elements; merging
  // `inner_extent` contiguous ramps of `current_lanes` stays within one mod tile.
  if (match.mod_val == 64 && total_lanes == 8 && match.coeff == current_lanes) {
    PrimExpr align8 = make_const(match.inner_base_at_zero.dtype(), 8);
    if (analyzer->CanProveEqual(floormod(match.inner_base_at_zero, align8),
                                make_zero(match.inner_base_at_zero.dtype()))) {
      return true;
    }
  }
  return false;
}

bool TryMergeWithFloorModAffine(const PrimExpr& dst_base, const PrimExpr& src_base,
                                const Var& inner_var, int64_t inner_extent, int64_t current_lanes,
                                arith::Analyzer* analyzer, int64_t* dst_stride_val,
                                int64_t* src_stride_val) {
  auto [dst_at_zero, dst_stride] = ExtractLinearCoeff(dst_base, inner_var, analyzer);
  if (!IsConstantInt(analyzer->Simplify(dst_stride), dst_stride_val) ||
      *dst_stride_val != current_lanes) {
    return false;
  }

  FloorModAffineMatch match;
  if (!TryMatchFloorModAffine(src_base, inner_var, analyzer, &match)) {
    return false;
  }
  if (match.coeff != current_lanes || match.coeff != *dst_stride_val) {
    return false;
  }
  if (!CanProveFloorModMergeSafe(analyzer, match, inner_extent, current_lanes)) {
    return false;
  }

  *src_stride_val = match.coeff;
  return true;
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
  bool mergeable = false;
  if (IsConstantInt(analyzer->Simplify(dst_stride), &dst_stride_val) &&
      IsConstantInt(analyzer->Simplify(src_stride), &src_stride_val) &&
      dst_stride_val == src_stride_val && dst_stride_val == current_lanes) {
    mergeable = true;
  } else if (TryMergeWithFloorModAffine(dst_ramp->base, src_ramp->base, inner_var, inner_extent,
                                        current_lanes, analyzer, &dst_stride_val, &src_stride_val)) {
    mergeable = true;
  }
  if (!mergeable) {
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

  PrimExpr dst_base = Substitute(dst_ramp->base, {{inner_var, make_zero(inner_var.dtype())}});
  PrimExpr src_base = Substitute(src_ramp->base, {{inner_var, make_zero(inner_var.dtype())}});
  dst_base = analyzer->Simplify(dst_base);
  src_base = analyzer->Simplify(src_base);

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
