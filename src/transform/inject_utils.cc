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
#include "../target/utils.h"

#include <tvm/tir/stmt_functor.h>

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

}  // namespace tl
}  // namespace tvm
