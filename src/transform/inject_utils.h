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

namespace tvm {
namespace tl {

/*!
 * \brief Check if the target is AMD DCU (gfx936, gfx942, etc.)
 */
bool IsDCUTarget(const IRModule& module);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_INJECT_UTILS_H_
