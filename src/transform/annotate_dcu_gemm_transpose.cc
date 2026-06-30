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
 * \brief Collect T.gemm transpose flags from lowered GEMM bodies into PrimFunc attrs.
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

class DCUGemmTransposeCollector : public StmtExprVisitor {
 public:
  bool trans_a{false};
  bool trans_b{false};
  bool found{false};
  int copy_major_a{0};
  int copy_major_b{0};
  bool copy_major_a_found{false};
  bool copy_major_b_found{false};

 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == kDCUGemmTransAStmt) {
      trans_a |= Downcast<Bool>(op->value)->value;
      found = true;
    } else if (op->attr_key == kDCUGemmTransBStmt) {
      trans_b |= Downcast<Bool>(op->value)->value;
      found = true;
    } else if (op->attr_key == kDCUCopyMajorAStmt) {
      copy_major_a = Downcast<IntImm>(op->value)->value;
      copy_major_a_found = true;
    } else if (op->attr_key == kDCUCopyMajorBStmt) {
      copy_major_b = Downcast<IntImm>(op->value)->value;
      copy_major_b_found = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

class DCUGemmTransposeStmtRemover : public StmtMutator {
 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == kDCUGemmTransAStmt || op->attr_key == kDCUGemmTransBStmt ||
        op->attr_key == kDCUCopyMajorAStmt || op->attr_key == kDCUCopyMajorBStmt) {
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }
};

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

    DCUCopyMajorFlags existing_major = GetDCUCopyMajorFlags(f);
    int copy_major_a = existing_major.a_defined ? existing_major.copy_major_a : 0;
    int copy_major_b = existing_major.b_defined ? existing_major.copy_major_b : 0;
    bool copy_major_found = collector.copy_major_a_found || collector.copy_major_b_found ||
                            existing_major.a_defined || existing_major.b_defined;
    if (collector.copy_major_a_found) {
      copy_major_a = collector.copy_major_a;
    }
    if (collector.copy_major_b_found) {
      copy_major_b = collector.copy_major_b;
    }

    if (!collector.found && !existing.defined && !copy_major_found) {
      return f;
    }

    auto* n = f.CopyOnWrite();
    n->body = DCUGemmTransposeStmtRemover()(n->body);
    if (collector.found || existing.defined) {
      f = WithAttr(std::move(f), kDCUGemmTransA, Bool(trans_a));
      f = WithAttr(std::move(f), kDCUGemmTransB, Bool(trans_b));
    }
    if (copy_major_found) {
      f = WithAttr(std::move(f), kDCUCopyMajorA, Integer(copy_major_a));
      f = WithAttr(std::move(f), kDCUCopyMajorB, Integer(copy_major_b));
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
