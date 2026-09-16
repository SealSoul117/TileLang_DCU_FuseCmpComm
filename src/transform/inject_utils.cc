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

#include <cctype>
#include <string>
#include <unordered_map>

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

bool DCUCopyBufferMapsEmpty(const DCUCopyBufferMaps& maps) {
  return maps.copy_major.empty() && maps.need_layout.empty() && maps.need_shadow.empty() &&
         maps.gemm_operand.empty() && maps.use_idxen.empty();
}

namespace {

void LoadIntegerMapAttr(const PrimFunc& func, const char* key,
                        std::unordered_map<std::string, int>* out) {
  if (!func->attrs.defined()) {
    return;
  }
  if (auto m = func->attrs.GetAttr<Map<String, Integer>>(key)) {
    for (const auto& kv : m.value()) {
      (*out)[std::string(kv.first)] = static_cast<int>(kv.second->value);
    }
  }
}

}  // namespace

DCUCopyBufferMaps GetDCUCopyBufferMaps(const PrimFunc& func) {
  DCUCopyBufferMaps maps;
  LoadIntegerMapAttr(func, kDCUCopyMajorMap, &maps.copy_major);
  LoadIntegerMapAttr(func, kDCUGemmOperandMap, &maps.gemm_operand);
  if (func->attrs.defined()) {
    if (auto m = func->attrs.GetAttr<Map<String, Integer>>(kDCUNeedLayoutMap)) {
      for (const auto& kv : m.value()) {
        maps.need_layout[std::string(kv.first)] = kv.second->value != 0;
      }
    }
    if (auto m = func->attrs.GetAttr<Map<String, Integer>>(kDCUNeedShadowMap)) {
      for (const auto& kv : m.value()) {
        maps.need_shadow[std::string(kv.first)] = kv.second->value != 0;
      }
    }
    if (auto m = func->attrs.GetAttr<Map<String, Integer>>(kDCUUseIdxenMap)) {
      for (const auto& kv : m.value()) {
        maps.use_idxen[std::string(kv.first)] = kv.second->value != 0;
      }
    }
  }
  return maps;
}

bool SharedNameMatchesGlobal(const std::string& shared_name, const std::string& global_name) {
  if (shared_name.empty() || global_name.empty()) {
    return false;
  }
  if (shared_name == global_name || shared_name == global_name + "_shared") {
    return true;
  }
  auto strip_numeric_suffix = [](std::string name) {
    while (!name.empty() && std::isdigit(static_cast<unsigned char>(name.back()))) {
      name.pop_back();
    }
    if (!name.empty() && name.back() == '_') {
      name.pop_back();
    }
    return name;
  };
  auto prefix_match = [&](const std::string& prefix) {
    if (prefix.empty() || shared_name.rfind(prefix, 0) != 0) {
      return false;
    }
    const std::string rest = shared_name.substr(prefix.size());
    return rest.empty() || rest[0] == '_' || rest.find("shared") != std::string::npos;
  };
  if (prefix_match(global_name)) {
    return true;
  }
  const std::string base = strip_numeric_suffix(global_name);
  return base != global_name && prefix_match(base);
}

static bool NamesRelated(const std::string& a, const std::string& b) {
  return a == b || SharedNameMatchesGlobal(a, b) || SharedNameMatchesGlobal(b, a);
}

std::optional<std::string> MatchDCUCopyBufferKey(const DCUCopyBufferMaps& maps,
                                                 const std::string& name) {
  if (name.empty()) {
    return std::nullopt;
  }
  auto find_in = [&](const auto& m) -> std::optional<std::string> {
    auto it = m.find(name);
    if (it != m.end()) {
      return it->first;
    }
    for (const auto& kv : m) {
      if (NamesRelated(kv.first, name)) {
        return kv.first;
      }
    }
    return std::nullopt;
  };
  if (auto key = find_in(maps.need_layout)) {
    return key;
  }
  if (auto key = find_in(maps.need_shadow)) {
    return key;
  }
  if (auto key = find_in(maps.copy_major)) {
    return key;
  }
  if (auto key = find_in(maps.use_idxen)) {
    return key;
  }
  return find_in(maps.gemm_operand);
}

bool LookupDCUNeedLayout(const DCUCopyBufferMaps& maps, const std::string& name,
                         bool default_value) {
  auto key = MatchDCUCopyBufferKey(maps, name);
  if (!key.has_value()) {
    return default_value;
  }
  auto it = maps.need_layout.find(*key);
  if (it == maps.need_layout.end()) {
    return default_value;
  }
  return it->second;
}

bool LookupDCUNeedShadow(const DCUCopyBufferMaps& maps, const std::string& name,
                         bool default_value) {
  auto key = MatchDCUCopyBufferKey(maps, name);
  if (!key.has_value()) {
    return default_value;
  }
  auto it = maps.need_shadow.find(*key);
  if (it == maps.need_shadow.end()) {
    return default_value;
  }
  return it->second;
}

bool LookupDCUNeedLayoutForOperand(const DCUCopyBufferMaps& maps, int gemm_operand,
                                   bool default_value) {
  for (const auto& kv : maps.gemm_operand) {
    if (kv.second == gemm_operand) {
      return LookupDCUNeedLayout(maps, kv.first, default_value);
    }
  }
  return default_value;
}

int LookupDCUCopyMajor(const DCUCopyBufferMaps& maps, const std::string& name,
                       int default_value) {
  auto key = MatchDCUCopyBufferKey(maps, name);
  if (!key.has_value()) {
    return default_value;
  }
  auto it = maps.copy_major.find(*key);
  if (it == maps.copy_major.end()) {
    return default_value;
  }
  return it->second;
}

std::optional<int> LookupDCUGemmOperand(const DCUCopyBufferMaps& maps,
                                        const std::string& name) {
  auto key = MatchDCUCopyBufferKey(maps, name);
  if (!key.has_value()) {
    return std::nullopt;
  }
  auto it = maps.gemm_operand.find(*key);
  if (it == maps.gemm_operand.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<bool> LookupDCUUseIdxen(const DCUCopyBufferMaps& maps,
                                      const std::string& name) {
  auto key = MatchDCUCopyBufferKey(maps, name);
  if (!key.has_value()) {
    return std::nullopt;
  }
  auto it = maps.use_idxen.find(*key);
  if (it == maps.use_idxen.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool NeedsDCULocalLayoutDSRead(const DCUGemmTransposeFlags& flags,
                               const DCUCopyBufferMaps& maps,
                               const std::string& local_name,
                               const std::string& shared_name) {
  if (!flags.defined) {
    return false;
  }
  const std::string lookup = !shared_name.empty() ? shared_name : local_name;
  const bool need_layout = LookupDCUNeedLayout(maps, lookup, /*default_value=*/true);
  if (auto operand = LookupDCUGemmOperand(maps, lookup)) {
    if (*operand == 0) {
      return flags.trans_a && need_layout;
    }
    if (*operand == 1) {
      return !flags.trans_b && need_layout;
    }
  }
  if (local_name.find("A_local") != std::string::npos) {
    return flags.trans_a && need_layout;
  }
  if (local_name.find("B_local") != std::string::npos) {
    return !flags.trans_b && need_layout;
  }
  return false;
}

bool NeedsDCUAsyncCopyIdxenForOperand(const DCUGemmTransposeFlags& flags, int gemm_operand,
                                      bool need_layout) {
  if (!flags.defined || !need_layout) {
    return false;
  }
  if (gemm_operand == 0) {
    return !flags.trans_a;
  }
  if (gemm_operand == 1) {
    return flags.trans_b;
  }
  return false;
}

bool NeedsDCUADSRead(const DCUGemmTransposeFlags& flags) {
  return flags.defined && !flags.trans_a;
}

namespace {

bool IsConsecutiveRamp(const PrimExpr& index, int expected_lanes) {
  const auto* ramp = index.as<RampNode>();
  if (!ramp) {
    return false;
  }
  return is_const_int(ramp->stride, 1) && is_const_int(ramp->lanes, expected_lanes);
}

}  // namespace

bool IsDCUCpAsyncVectorCopy(const BufferStoreNode* store, const BufferLoadNode* load) {
  if (store == nullptr || load == nullptr) {
    return false;
  }
  // Dest must be a flat 1-D index for cp_async_gs M0 + lane*N.
  // Source may gather (XOR swizzle / idxen); only require a single flat index.
  if (store->indices.size() != 1 || load->indices.size() != 1) {
    return false;
  }

  DataType dtype = store->buffer->dtype;
  if (dtype.lanes() > 1) {
    dtype = dtype.element_of();
  }
  const int elem_bytes = dtype.bytes();
  // codegen always emits tl::cp_async_gs<16> (16 bytes / thread).
  if (elem_bytes <= 0 || 16 % elem_bytes != 0) {
    return false;
  }
  const int expected_lanes = 16 / elem_bytes;
  return IsConsecutiveRamp(store->indices[0], expected_lanes);
}

}  // namespace tl
}  // namespace tvm
