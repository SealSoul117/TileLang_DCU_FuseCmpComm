#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
_tl_source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
: "${ROCM_PATH:=${DTKROOT:-/opt/rocm}}"
export ROCM_PATH
_tl_python="$(command -v python)"
mkdir -p "$TILELANG_DEV_BUILD_ROOT/logs"
cmake -S "$_tl_source_root" -B "$TILELANG_DEV_BUILD_ROOT"     -DTILELANG_SKIP_SUBMODULE_UPDATE=OFF     -DUSE_CUDA=OFF -DUSE_ROCM="$ROCM_PATH" -DUSE_LLVM=OFF     -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON     -DPython_EXECUTABLE="$_tl_python" -DPython3_EXECUTABLE="$_tl_python"     2>&1 | tee "$TILELANG_DEV_BUILD_ROOT/logs/configure.log"
cmake --build "$TILELANG_DEV_BUILD_ROOT" --target tilelang_module tilelang_cython_wrapper tilelang_dcu_ipc_runtime     -j "${TL_OLD_BUILD_JOBS:-6}" 2>&1 | tee "$TILELANG_DEV_BUILD_ROOT/logs/build.log"
