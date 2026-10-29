#!/usr/bin/env bash
# Run from a four-GPU Slurm allocation after selecting scripts/dev_old/env.sh.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
artifact_dir=${1:?Usage: run_compare.sh ARTIFACT_DIR REFERENCE_LIBRARY_DIR [M ...]}
reference_lib_dir=${2:?Usage: run_compare.sh ARTIFACT_DIR REFERENCE_LIBRARY_DIR [M ...]}
shift 2
if [[ $# == 0 ]]; then set -- 2048 4096 8192 16384; fi
mkdir -p "$artifact_dir"
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
for m in "$@"; do
  timeout --signal=TERM --kill-after=15s 1200s \
    mpirun -np 4 --oversubscribe --bind-to none --stdin none \
    python "$script_dir/benchmark_gemm_allreduce.py" \
    --implementation compare --m "$m" --n "$m" --k "$((m / 4))" \
    --reference-library "$reference_lib_dir/reference-m${m}.so" \
    --warmup 30 --repeats 30 \
    --json-dir "$artifact_dir/m$m" --export-hip "$artifact_dir/m$m/hip" \
    > "$artifact_dir/m$m.log" 2>&1
  python "$script_dir/summarize.py" "$artifact_dir/m$m" > "$artifact_dir/m$m-summary.json"
  echo "PASS GEMM+AllReduce m=$m"
done
