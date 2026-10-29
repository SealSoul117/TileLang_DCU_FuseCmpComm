# TileLang DCU HIP GEMM

This reproduces a standalone FP16 GEMM for Hygon `gfx936`.  Input `B` is stored
as `(N, K)`, so the kernel computes `C = A @ B.T`.  The generated kernel uses
the DCU MMAC intrinsic and has FP32 accumulation and FP16 output.

On an allocated DCU node:

```bash
module load dtk/26.04 python/3.10 cmake/3.31.12
source .venv-dcu/bin/activate
export ROCM_PATH="$DTKROOT"
export LD_LIBRARY_PATH="$DTKROOT/lib:$DTKROOT/hip/lib:$DTKROOT/llvm/lib:${LD_LIBRARY_PATH:-}"

python examples/gemm/dcu_codegen/generate_gemm_hip.py
python examples/gemm/dcu_codegen/run_gemm_hip.py \
  examples/gemm/dcu_codegen/generated/gemm_m1024_n1024_k1024_gfx936.so
```

The specialized example requires `M` and `N` to be multiples of 128 and `K`
to be a multiple of 32.  Pass matching `--m`, `--n`, and `--k` to both scripts
to generate another fixed-shape kernel.

Configure the native build with explicit CMake values (environment variables
alone are not consumed by this checkout):

```bash
CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=$DTKROOT -DCMAKE_BUILD_TYPE=Release" \
  CMAKE_BUILD_PARALLEL_LEVEL=16 \
  python -m pip install -e . --no-build-isolation --no-deps
```
