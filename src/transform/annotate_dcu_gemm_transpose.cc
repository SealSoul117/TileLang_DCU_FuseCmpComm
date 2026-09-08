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
 * \file annotate_dcu_gemm_transpose.cc
 * \brief Collect T.gemm / T.copy annotations from lowered bodies into PrimFunc attrs.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "inject_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

std::string AttrStmtBufferName(const AttrStmtNode* op) {
  if (const auto* s = op->node.as<StringImmNode>()) {
    return s->value;
  }
  return "";
}

int AttrStmtIntValue(const PrimExpr& value) {
  if (const auto* imm = value.as<IntImmNode>()) {
    return static_cast<int>(imm->value);
  }
  return 0;
}

class DCUGemmTransposeCollector : public StmtExprVisitor {
 public:
  bool trans_a{false};
  bool trans_b{false};
  bool found{false};
  DCUCopyBufferMaps copy_maps;

 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == kDCUGemmTransAStmt) {
      trans_a |= Downcast<Bool>(op->value)->value;
      found = true;
    } else if (op->attr_key == kDCUGemmTransBStmt) {
      trans_b |= Downcast<Bool>(op->value)->value;
      found = true;
    } else if (op->attr_key == kDCUCopyMajorStmt) {
      const std::string name = AttrStmtBufferName(op);
      if (!name.empty()) {
        copy_maps.copy_major[name] = AttrStmtIntValue(op->value);
      }
    } else if (op->attr_key == kDCUNeedLayoutStmt) {
      const std::string name = AttrStmtBufferName(op);
      if (!name.empty()) {
        copy_maps.need_layout[name] = AttrStmtIntValue(op->value) != 0;
      }
    } else if (op->attr_key == kDCUNeedShadowStmt) {
      const std::string name = AttrStmtBufferName(op);
      if (!name.empty()) {
        copy_maps.need_shadow[name] = AttrStmtIntValue(op->value) != 0;
      }
    } else if (op->attr_key == kDCUGemmOperandStmt) {
      const std::string name = AttrStmtBufferName(op);
      if (!name.empty()) {
        copy_maps.gemm_operand[name] = AttrStmtIntValue(op->value);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

class DCUGemmTransposeStmtRemover : public StmtMutator {
 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == kDCUGemmTransAStmt || op->attr_key == kDCUGemmTransBStmt ||
        op->attr_key == kDCUCopyMajorStmt || op->attr_key == kDCUNeedLayoutStmt ||
        op->attr_key == kDCUNeedShadowStmt || op->attr_key == kDCUGemmOperandStmt) {
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }
};

Map<String, Integer> ToIntegerMap(const std::unordered_map<std::string, int>& src) {
  Map<String, Integer> out;
  for (const auto& kv : src) {
    out.Set(String(kv.first), Integer(kv.second));
  }
  return out;
}

Map<String, Integer> NeedLayoutToIntegerMap(const std::unordered_map<std::string, bool>& src) {
  Map<String, Integer> out;
  for (const auto& kv : src) {
    out.Set(String(kv.first), Integer(kv.second ? 1 : 0));
  }
  return out;
}

void MergeCopyMaps(DCUCopyBufferMaps* dst, const DCUCopyBufferMaps& src) {
  for (const auto& kv : src.copy_major) {
    dst->copy_major[kv.first] = kv.second;
  }
  for (const auto& kv : src.need_layout) {
    dst->need_layout[kv.first] = kv.second;
  }
  for (const auto& kv : src.need_shadow) {
    dst->need_shadow[kv.first] = kv.second;
  }
  for (const auto& kv : src.gemm_operand) {
    dst->gemm_operand[kv.first] = kv.second;
  }
}

}  // namespace

tvm::transform::Pass AnnotateDCUGemmTranspose() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    if (!IsDCUTarget(m)) {
      return f;
    }

    DCUGemmTransposeCollector collector;
    collector(f->body);

    DCUGemmTransposeFlags existing = GetDCUGemmTransposeFlags(f);
    bool trans_a = existing.defined ? existing.trans_a : false;
    bool trans_b = existing.defined ? existing.trans_b : false;
    if (collector.found) {
      trans_a |= collector.trans_a;
      trans_b |= collector.trans_b;
    }

    DCUCopyBufferMaps copy_maps = GetDCUCopyBufferMaps(f);
    MergeCopyMaps(&copy_maps, collector.copy_maps);

    if (!collector.found && !existing.defined && DCUCopyBufferMapsEmpty(copy_maps)) {
      return f;
    }

    auto* n = f.CopyOnWrite();
    n->body = DCUGemmTransposeStmtRemover()(n->body);
    if (collector.found || existing.defined) {
      f = WithAttr(std::move(f), kDCUGemmTransA, Bool(trans_a));
      f = WithAttr(std::move(f), kDCUGemmTransB, Bool(trans_b));
    }
    if (!copy_maps.copy_major.empty()) {
      f = WithAttr(std::move(f), kDCUCopyMajorMap, ToIntegerMap(copy_maps.copy_major));
    }
    if (!copy_maps.need_layout.empty()) {
      f = WithAttr(std::move(f), kDCUNeedLayoutMap, NeedLayoutToIntegerMap(copy_maps.need_layout));
    }
    if (!copy_maps.need_shadow.empty()) {
      f = WithAttr(std::move(f), kDCUNeedShadowMap, NeedLayoutToIntegerMap(copy_maps.need_shadow));
    }
    if (!copy_maps.gemm_operand.empty()) {
      f = WithAttr(std::move(f), kDCUGemmOperandMap, ToIntegerMap(copy_maps.gemm_operand));
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateDCUGemmTranspose", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateDCUGemmTranspose",
                        AnnotateDCUGemmTranspose);
}

}  // namespace tl
}  // namespace tvm
