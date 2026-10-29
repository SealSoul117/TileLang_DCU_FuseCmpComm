#!/usr/bin/env bash
# Build the untouched reference using old's templates and embedded runtime.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$script_dir/../../.." && pwd -P)
reference_dir=${1:?Usage: build_reference.sh REFERENCE_V2_DIR OUTPUT_DIR}
output_dir=${2:?Usage: build_reference.sh REFERENCE_V2_DIR OUTPUT_DIR}
: "${ROCM_PATH:?Source scripts/dev_old/env.sh first}"
: "${MPI_HOME:?Source scripts/dev_old/env.sh first}"
mkdir -p "$output_dir"
for m in 2048 4096 8192 16384; do
  source_file="$reference_dir/fused_gemm_allreduce_m${m}_n${m}_k$((m / 4))_ipc.hip"
  test -f "$source_file"
  "$ROCM_PATH/bin/hipcc" -shared -fPIC -std=c++17 -O3 \
    --offload-arch=gfx936 --gpu-max-threads-per-block=256 \
    --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 \
    "-DGEMM_AR_REFERENCE_SOURCE=\"$source_file\"" \
    -I"$repo_dir/src" -I"$repo_dir/src/runtime/dcu_ipc/include" \
    -I"$MPI_HOME/include" "$script_dir/handwritten_adapter.cpp" \
    -L"$repo_dir/build-old/lib" -ltilelang_dcu_ipc_runtime \
    -Wl,-rpath,"$repo_dir/build-old/lib" \
    -L"$MPI_HOME/lib" -Wl,-rpath,"$MPI_HOME/lib" -lmpi \
    -o "$output_dir/reference-m${m}.so" > "$output_dir/reference-m${m}-build.log" 2>&1
  echo "Built reference-m${m}.so"
done
