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
 * \file inject_utils.cc
 * \brief Common utilities for injection transforms.
 */
#include "inject_utils.h"
#include "../op/builtin.h"
#include "../target/utils.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <string>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

bool IsThreadIdxTag(const ffi::String& thread_tag) {
  return thread_tag == "threadIdx.x" || thread_tag == "threadIdx.y" ||
         thread_tag == "threadIdx.z";
}

}  // namespace

bool IsDCUTarget(const IRModule& module) {
    return TargetIsDCU(Target::Current(false));
}

const StmtNode* FindDCUInjectTarget(const Stmt& body) {
  class Finder : public StmtVisitor {
   public:
    const StmtNode* swizzle_body = nullptr;
    const AttrStmtNode* innermost_thread_attr = nullptr;
    const ForNode* innermost_thread_for = nullptr;

   private:
    void VisitStmt_(const AttrStmtNode* op) final {
      if (op->attr_key == kThreadblockSwizzleAttr) {
        swizzle_body = op->body.get();
        return;
      }
      if (op->attr_key == tir::attr::thread_extent) {
        if (const auto* iv = op->node.as<IterVarNode>()) {
          if (IsThreadIdxTag(iv->thread_tag)) {
            innermost_thread_attr = op;
          }
        }
      }
      StmtVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const ForNode* op) final {
      if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
          IsThreadIdxTag(op->thread_binding.value()->thread_tag)) {
        innermost_thread_for = op;
      }
      StmtVisitor::VisitStmt_(op);
    }
  };

  Finder finder;
  finder(body);
  if (finder.swizzle_body != nullptr) {
    return finder.swizzle_body;
  }
  if (finder.innermost_thread_attr != nullptr) {
    return finder.innermost_thread_attr->body.get();
  }
  if (finder.innermost_thread_for != nullptr) {
    return finder.innermost_thread_for->body.get();
  }
  return body.get();
}

class InjectAtTargetMutator : public StmtMutator {
 public:
  InjectAtTargetMutator(const StmtNode* target, std::function<Stmt(const Stmt&)> wrap_fn)
      : target_(target), wrap_fn_(std::move(wrap_fn)) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    if (stmt.get() == target_) {
      return wrap_fn_(StmtMutator::VisitStmt(stmt));
    }
    return StmtMutator::VisitStmt(stmt);
  }

 private:
  const StmtNode* target_;
  std::function<Stmt(const Stmt&)> wrap_fn_;
};

Stmt InjectAtDCUTarget(const Stmt& body, const std::function<Stmt(const Stmt&)>& wrap_fn) {
  const StmtNode* target = FindDCUInjectTarget(body);
  InjectAtTargetMutator mutator(target, wrap_fn);
  return mutator(body);
}

namespace {

bool IsGlobalScope(const Buffer& buf) {
  auto scope = buf.scope();
  return scope == "global" || scope == "";
}

const BufferLoadNode* PeelBufferLoad(const PrimExpr& value) {
  if (const auto* load = value.as<BufferLoadNode>()) {
    return load;
  }
  if (const auto* cast = value.as<CastNode>()) {
    return cast->value.as<BufferLoadNode>();
  }
  return nullptr;
}

bool IsMmacCall(const CallNode* call) {
  if (!call) {
    return false;
  }
  if (call->op.same_as(tl::tvm_mmac())) {
    return true;
  }
  std::string op_name;
  if (const auto* op = call->op.as<OpNode>()) {
    op_name = op->name;
  } else if (const auto* gv = call->op.as<GlobalVarNode>()) {
    op_name = gv->name_hint;
  }
  return op_name.find("mmac") != std::string::npos || op_name.find("mma") != std::string::npos;
}

bool IsDSReadCall(const CallNode* call) {
  if (!call) {
    return false;
  }
  if (const auto* op = call->op.as<OpNode>()) {
    const std::string& name = op->name;
    return name.find("ds_read") != std::string::npos;
  }
  return false;
}

bool IsGlobalSrcAsyncCopyCall(const CallNode* call) {
  if (!call) {
    return false;
  }
  return call->op.same_as(Op::Get("tl.dcu_async_copy")) ||
         call->op.same_as(Op::Get("tl.dcu_async_copy_idxen")) ||
         call->op.same_as(builtin::ptx_cp_async());
}

class DCUMainLoopFeatureChecker : public StmtExprVisitor {
 public:
  bool has_mmac{false};
  bool has_memory_access{false};
  bool has_global_src_memory{false};

 private:
  void VisitExpr_(const BufferLoadNode* op) final {
    has_memory_access = true;
    if (IsGlobalScope(op->buffer)) {
      has_global_src_memory = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    has_memory_access = true;
    const BufferLoadNode* load = PeelBufferLoad(op->value);
    if (load != nullptr && IsGlobalScope(load->buffer)) {
      has_global_src_memory = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    if (IsMmacCall(op)) {
      has_mmac = true;
    }
    if (IsDSReadCall(op)) {
      has_memory_access = true;
    }
    if (IsGlobalSrcAsyncCopyCall(op)) {
      has_memory_access = true;
      has_global_src_memory = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

class DCUMainLoopCollector : public StmtVisitor {
 public:
  std::vector<const ForNode*> loops;

 private:
  void VisitStmt_(const ForNode* op) final {
    if (IsDCUMainLoop(op)) {
      loops.push_back(op);
    }
    StmtVisitor::VisitStmt_(op);
  }
};

}  // namespace

DCUMainLoopFeatures AnalyzeDCUMainLoopBody(const Stmt& body) {
  DCUMainLoopFeatureChecker checker;
  checker(body);
  return {checker.has_mmac, checker.has_memory_access, checker.has_global_src_memory};
}

bool IsDCUMainLoop(const ForNode* loop) {
  if (loop == nullptr) {
    return false;
  }
  DCUMainLoopFeatures features = AnalyzeDCUMainLoopBody(loop->body);
  return features.has_mmac && features.has_memory_access && features.has_global_src_memory;
}

std::vector<const ForNode*> CollectDCUMainLoops(const Stmt& stmt) {
  DCUMainLoopCollector collector;
  collector(stmt);
  return collector.loops;
}

DCUGemmTransposeFlags GetDCUGemmTransposeFlags(const PrimFunc& func) {
  DCUGemmTransposeFlags flags;
  if (!func->attrs.defined()) {
    return flags;
  }
  if (auto trans_a = func->attrs.GetAttr<Bool>(kDCUGemmTransA)) {
    flags.trans_a = trans_a.value()->value;
    flags.defined = true;
  }
  if (auto trans_b = func->attrs.GetAttr<Bool>(kDCUGemmTransB)) {
    flags.trans_b = trans_b.value()->value;
    flags.defined = true;
  }
  return flags;
}

DCUCopyMajorFlags GetDCUCopyMajorFlags(const PrimFunc& func) {
  DCUCopyMajorFlags flags;
  if (!func->attrs.defined()) {
    return flags;
  }
  if (auto copy_major_a = func->attrs.GetAttr<Integer>(kDCUCopyMajorA)) {
    flags.copy_major_a = copy_major_a.value()->value;
    flags.a_defined = true;
  }
  if (auto copy_major_b = func->attrs.GetAttr<Integer>(kDCUCopyMajorB)) {
    flags.copy_major_b = copy_major_b.value()->value;
    flags.b_defined = true;
  }
  return flags;
}

bool NeedsDCULocalLayoutDSRead(const DCUGemmTransposeFlags& flags,
                               const std::string& buffer_name) {
  if (!flags.defined) {
    return false;
  }
  if (flags.trans_a && buffer_name.find("A_local") != std::string::npos) {
    return true;
  }
  if (!flags.trans_b && buffer_name.find("B_local") != std::string::npos) {
    return true;
  }
  return false;
}

bool NeedsDCUADSRead(const DCUGemmTransposeFlags& flags) {
  return flags.defined && !flags.trans_a;
}

bool NeedsDCUAsyncCopyIdxenA(const DCUGemmTransposeFlags& flags) {
  return flags.defined && !flags.trans_a;
}

bool NeedsDCUAsyncCopyIdxenB(const DCUGemmTransposeFlags& flags) {
  return flags.defined && flags.trans_b;
}

}  // namespace tl
}  // namespace tvm
