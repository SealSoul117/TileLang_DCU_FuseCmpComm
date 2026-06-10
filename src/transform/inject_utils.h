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
 * \file inject_utils.h
 * \brief Common utilities for injection transforms.
 */
#ifndef TVM_TL_INJECT_UTILS_H_
#define TVM_TL_INJECT_UTILS_H_

#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <functional>

namespace tvm {
namespace tl {

/*!
 * \brief Check if the target is AMD DCU (gfx936, gfx942, etc.)
 */
bool IsDCUTarget(const IRModule& module);

/*!
 * \brief Attribute key for threadblock swizzle pattern.
 */
constexpr const char* kThreadblockSwizzleAttr = "threadblock_swizzle_pattern";

/*!
 * \brief Find the stmt where DCU-specific lets (warp_idx, shadow addr, dcu_res)
 *        should be injected.
 *
 * If a threadblock swizzle attr exists, injection happens at the beginning of
 * its body so codegen emits blockIdx before those lets. Otherwise injection
 * happens at the beginning of the innermost threadIdx binding body.
 */
const tir::StmtNode* FindDCUInjectTarget(const tir::Stmt& body);

/*!
 * \brief Wrap the stmt at the DCU injection target with \p wrap_fn.
 */
tir::Stmt InjectAtDCUTarget(const tir::Stmt& body,
                            const std::function<tir::Stmt(const tir::Stmt&)>& wrap_fn);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_INJECT_UTILS_H_
