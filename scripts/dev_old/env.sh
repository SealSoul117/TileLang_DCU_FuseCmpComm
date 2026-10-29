# Source in bash after activating your Python environment and HIP/MPI toolchain.
# Use the repository dependency configuration, not a server-specific cache.
_tl_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
unset TVM_ROOT TVM_HOME TVM_FFI_LIBRARY_PATH TVM_IMPORT_PYTHON_PATH TVM_LIBRARY_PATH PYTHONHOME
export TILELANG_DEV_BUILD_ROOT="$_tl_repo_root/build"
export PYTHONPATH="$_tl_repo_root"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$TILELANG_DEV_BUILD_ROOT/pycache"
export TL_TEMPLATE_PATH="$_tl_repo_root/src"
export TL_CUTLASS_PATH="$_tl_repo_root/3rdparty/cutlass/include"
export TL_COMPOSABLE_KERNEL_PATH="$_tl_repo_root/3rdparty/composable_kernel/include"
export TILELANG_CACHE_DIR="$TILELANG_DEV_BUILD_ROOT/kernel-cache"
export TILELANG_TMP_DIR="$TILELANG_CACHE_DIR/tmp"
export LD_LIBRARY_PATH="$TILELANG_DEV_BUILD_ROOT/lib:$TILELANG_DEV_BUILD_ROOT/tvm${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _tl_repo_root
