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

#include <tvm/ir/function.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <functional>
#include <string>
#include <vector>

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

/*!
 * \brief Feature flags for identifying the DCU GEMM main loop body.
 */
struct DCUMainLoopFeatures {
  bool has_mmac{false};
  bool has_memory_access{false};
  bool has_global_src_memory{false};
};

/*!
 * \brief Analyze a loop body for DCU GEMM main-loop characteristics:
 *        mmac, memory traffic (BufferLoad/Store, ds_read, async copy), and
 *        at least one memory op with Global as source.
 */
DCUMainLoopFeatures AnalyzeDCUMainLoopBody(const tir::Stmt& body);

/*!
 * \brief Whether \p loop is the DCU GEMM main loop.
 */
bool IsDCUMainLoop(const tir::ForNode* loop);

/*!
 * \brief Collect all for-loops in \p stmt that match DCU main-loop features.
 */
std::vector<const tir::ForNode*> CollectDCUMainLoops(const tir::Stmt& stmt);

/*!
 * \brief Transpose flags collected from T.gemm lowering.
 */
struct DCUGemmTransposeFlags {
  bool trans_a{false};
  bool trans_b{false};
  bool defined{false};
};

/*!
 * \brief Read DCU GEMM transpose flags from PrimFunc attrs.
 */
DCUGemmTransposeFlags GetDCUGemmTransposeFlags(const tir::PrimFunc& func);

/*!
 * \brief Copy major order collected from T.copy(..., major=...) lowering.
 *
 * copy_major values: 0 = row_major, 1 = col_major.
 */
struct DCUCopyMajorFlags {
  int copy_major_a{0};
  int copy_major_b{0};
  bool a_defined{false};
  bool b_defined{false};
};

/*!
 * \brief Read DCU async-copy major flags from PrimFunc attrs.
 */
DCUCopyMajorFlags GetDCUCopyMajorFlags(const tir::PrimFunc& func);

/*!
 * \brief Whether InjectBLocalLayoutTransform / InjectDSRead should rewrite this local buffer.
 *
 * Layout-transform + ds_read_m32x16 applies when A is transposed (A_local) or B is not
 * transposed (B_local).
 */
bool NeedsDCULocalLayoutDSRead(const DCUGemmTransposeFlags& flags, const std::string& buffer_name);


/*!
 * \brief Whether InjectDCUAsyncCopyIdxen should rewrite A-matrix async copies.
 *
 * Applies when A is not transposed.
 */
bool NeedsDCUAsyncCopyIdxenA(const DCUGemmTransposeFlags& flags);

/*!
 * \brief Whether InjectDCUAsyncCopyIdxen should rewrite B-matrix async copies.
 *
 * Applies when B is transposed.
 */
bool NeedsDCUAsyncCopyIdxenB(const DCUGemmTransposeFlags& flags);

/*!
 * \brief Whether a global→shared BufferStore is a consecutive vector copy of
 *        exactly 16 element-bytes (the width of tl::cp_async_gs<16>).
 *
 * Scalar or non-unit-stride loads must stay as ordinary BufferStore; rewriting
 * them to dcu_async_copy would over-read (e.g. 1 half → dwordx4).
 */
bool IsDCUCpAsyncVectorCopy(const tir::BufferStoreNode* store,
                            const tir::BufferLoadNode* load);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_INJECT_UTILS_H_
