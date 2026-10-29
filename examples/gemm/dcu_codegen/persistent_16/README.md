# Persistent GEMM + AllReduce prototype

This is a four-PE correctness prototype. Each PE computes a `4096 x 4096`
FP32 partial output from a `K_local=1024` FP16 GEMM. The grid has one block per
CU; on BW1000 that means 80 blocks. The number of persistent communication
blocks is selected with `--comm-blocks N` (default 16), and the remaining
blocks compute tiles.

The communication path is a same-node one-shot AllReduce. A compute block
writes a contiguous tile to symmetric memory and pushes a per-PE ready flag.
The corresponding communication block waits for all four flags, obtains every
peer's symmetric tile, and sums it locally. Both the original DUSHMEM variant
and a ROC_SHMEM 3.5 IPC variant are available.

Generate the TileLang tile helper and build on a DTK 26.04 DCU node:

```bash
python examples/gemm/dcu_codegen/persistent_16/generate_compute_tile.py

cmake -S examples/gemm/dcu_codegen/persistent_16 \
  -B examples/gemm/dcu_codegen/persistent_16/build-local \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="$DTKROOT/dcc/bin/dcc" \
  -DCMAKE_HIP_ARCHITECTURES=gfx936
cmake --build examples/gemm/dcu_codegen/persistent_16/build-local -j16
```

Run with one PE per DCU:

```bash
srun --mpi=pmi2 \
  --partition=bw1000 --account=ncic --gres=dcu:bw1000:4 \
  --nodes=1 --ntasks=4 --cpus-per-task=4 --mem=64G --time=00:05:00 \
  bash -lc '
    module load dtk/26.04
    export LD_LIBRARY_PATH="$DTKROOT/lib/dushmem:$DTKROOT/lib:$DTKROOT/hip/lib:$DTKROOT/llvm/lib:${LD_LIBRARY_PATH:-}"
    export DUSHMEM_BOOTSTRAP_PMI=PMI-2
    export DUSHMEM_SYMMETRIC_SIZE=512M
exec /public/home/baoboyuan/TileLang_DCU/examples/gemm/dcu_codegen/persistent_16/build-local/persistent_gemm_allreduce_16 --comm-blocks 16
  '
```

## ROC_SHMEM 3.5 / DTK 26.04 variant

The uploaded Debian package can be installed without root by extracting it to
a user-owned prefix:

```bash
mkdir -p /public/home/baoboyuan/rocshmem_3.5.0-dtk2604
dpkg-deb -x \
  /public/home/baoboyuan/rocshmem_3.5.0-dtk2604-gf0741e84-260904163403_amd64.deb \
  /public/home/baoboyuan/rocshmem_3.5.0-dtk2604
export ROCSHMEM_ROOT=/public/home/baoboyuan/rocshmem_3.5.0-dtk2604/opt/rocshmem
```

The package is built for `gfx936`/`gfx938`, DTK 26.04, and an external Open MPI.
Its exported CMake config requires a `FindNUMA.cmake` module that is not shipped
in the package, so this example imports `librocshmem.a` directly and links MPI,
NUMA, HSA, pthread, and `dl` explicitly.

Configure and build on a four-DCU compute node:

```bash
module purge
module load dtk/26.04 python/3.10 cmake/3.31.12 mpi/hpcx/2.20
export DTKROOT=/public/software/dtk/26.04
export ROCSHMEM_ROOT=/public/home/baoboyuan/rocshmem_3.5.0-dtk2604/opt/rocshmem
export LD_LIBRARY_PATH="$DTKROOT/lib:$DTKROOT/hip/lib:$DTKROOT/llvm/lib:${LD_LIBRARY_PATH:-}"

cmake -S examples/gemm/dcu_codegen/persistent_16 \
  -B examples/gemm/dcu_codegen/persistent_16/build-rocshmem-make \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="$DTKROOT/dcc/bin/dcc" \
  -DCMAKE_HIP_ARCHITECTURES=gfx936
cmake --build examples/gemm/dcu_codegen/persistent_16/build-rocshmem-make \
  --target persistent_gemm_allreduce_rocshmem -j16
```

From the interactive allocation that owns all four DCUs, launch one MPI rank
per device:

```bash
export ROCSHMEM_BACKEND=ipc
export ROCSHMEM_HEAP_SIZE=536870912
export ROCSHMEM_MAX_NUM_CONTEXTS=128

OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp \
mpirun -n 4 --oversubscribe --bind-to none \
  -x ROCSHMEM_BACKEND -x ROCSHMEM_HEAP_SIZE \
  -x ROCSHMEM_MAX_NUM_CONTEXTS -x LD_LIBRARY_PATH \
  examples/gemm/dcu_codegen/persistent_16/build-rocshmem-make/persistent_gemm_allreduce_rocshmem \
  --comm-blocks 16
```

HPC-X selects UCX by default on this machine, but that PML is not usable in the
nested interactive launch. Selecting `ob1` with the same-node `vader`/TCP BTLs
avoids that initialization failure. The XPMEM warning is harmless: Open MPI
falls back to another `vader` single-copy mechanism, while the actual GPU peer
traffic is performed through ROC_SHMEM's IPC mappings.

The ROC_SHMEM kernel intentionally does not create a work-group context. Such a
context adds 1 KiB of static shared memory, while the TileLang GEMM already
requires all 64 KiB of dynamic shared memory available on a BW1000 CU. Instead,
`rocshmem_ptr` obtains each peer's IPC mapping; the kernel reads partial tiles
directly and publishes ready flags with a peer-mapped `atomicExch`. Host-side
checks reject execution if every peer mapping is not available.

On 2026-09-06 both tested configurations passed exact four-PE validation:

| Comm blocks | Compute blocks | Slowest event (ms) | Slowest wall (ms) |
|---:|---:|---:|---:|
| 16 | 64 | 28.595 | 28.967 |
| 66 | 14 | 9.638 | 10.539 |

These are initial ROC_SHMEM measurements from one run, not a fresh parameter
sweep. They should not be compared directly with the earlier DUSHMEM table
without repeated interleaved measurements.

The inputs are exact constants. Each local partial tile should contain `2.0`;
the four-PE AllReduce result should contain `8.0` everywhere. With three warmup
iterations followed by ten measured iterations, every tested configuration
reported `PASS` and zero errors. A same-node sweep on 2026-08-31 produced the
following averages; the table uses the slowest of four PEs as the end-to-end
latency:

| Comm blocks | Compute blocks | Max event (ms) | Max wall (ms) | Speedup vs. 16 comm |
|---:|---:|---:|---:|---:|
| 4  | 76 | 70.916 | 70.959 | 0.255x |
| 8  | 72 | 36.879 | 36.916 | 0.490x |
| 12 | 68 | 24.326 | 24.369 | 0.742x |
| 16 | 64 | 18.053 | 18.090 | 1.000x |
| 20 | 60 | 14.829 | 14.908 | 1.214x |
| 24 | 56 | 12.342 | 12.378 | 1.462x |
| 32 | 48 |  9.211 |  9.247 | 1.956x |

An extended sweep from 34 through 72 communication blocks, in steps of two,
found the turnover point. All configurations passed exact validation:

| Comm blocks | Compute blocks | Max event (ms) | Max wall (ms) |
|---:|---:|---:|---:|
| 34 | 46 | 8.961 | 9.018 |
| 36 | 44 | 8.417 | 8.454 |
| 38 | 42 | 7.959 | 7.993 |
| 40 | 40 | 7.521 | 7.563 |
| 42 | 38 | 7.498 | 7.537 |
| 44 | 36 | 6.921 | 6.960 |
| 46 | 34 | 6.847 | 6.891 |
| 48 | 32 | 6.471 | 6.503 |
| 50 | 30 | 6.184 | 6.226 |
| 52 | 28 | 6.027 | 6.062 |
| 54 | 26 | 5.649 | 5.684 |
| 56 | 24 | 5.582 | 5.618 |
| 58 | 22 | 5.383 | 5.420 |
| 60 | 20 | 5.221 | 5.295 |
| 62 | 18 | 5.061 | 5.095 |
| 64 | 16 | 4.864 | 4.900 |
| **66** | **14** | **4.785** | **4.823** |
| 68 | 12 | 5.368 | 5.413 |
| 70 | 10 | 6.301 | 6.358 |
| 72 | 8 | 7.962 | 8.008 |

The neighboring 64/66/68 configurations were then run two additional times.
The mean of the three max-PE wall results was 4.990, 4.841, and 5.405 ms,
respectively. Therefore 66 communication blocks and 14 compute blocks is the
validated optimum at this step size for the current kernel and problem shape.

The `partial` and `reduced` buffers are tile-major (`1024` consecutive
`128 x 128` tiles), not conventional row-major matrices. A production consumer
should either consume this layout directly or fuse a tile-major-to-row-major
store. DTK 26.04 also prints a launch-bound warning for its internal proxy-quiet
kernel during finalization; it appears after the result has passed validation.

## PyTorch serial baseline

`torch_serial_baseline.py` performs the same local matrix multiplication and
then a blocking `torch.distributed.all_reduce`. Standard FP16 `torch.mm` returns
FP16, so the default path includes an FP16-to-FP32 cast before the FP32
AllReduce to match the fused prototype's result type and communication volume.
It reports GEMM+cast, AllReduce, and their serial sum using the slowest rank.

Run it from a DTK 26.04-compatible GPU PyTorch environment:

```bash
srun --mpi=pmi2 \
  --partition=bw1000 --account=ncic --gres=dcu:bw1000:4 \
  --nodes=1 --ntasks=4 --cpus-per-task=4 --mem=64G --time=00:05:00 \
  bash -lc '
    module load dtk/26.04
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=29517
    python /public/home/baoboyuan/TileLang_DCU/examples/gemm/dcu_codegen/persistent_16/torch_serial_baseline.py
  '
```

The current `.venv-dcu` contains CPU-only PyTorch 2.5.1 and cannot execute this
baseline. A DCU-enabled PyTorch build with `torch.cuda` and the NCCL-compatible
distributed backend must be activated first.

## Serial TileLang GEMM + DUSHMEM AllReduce

`serial_gemm_dushmem_allreduce` launches all TileLang GEMM tiles first and then
calls DUSHMEM's official FP32 on-stream collective:

```cpp
dushmemx_float_sum_reduce_on_stream(
    DUSHMEM_TEAM_WORLD, reduced, partial, 4096 * 4096, stream);
```

The four-BW1000 run (three warmups, ten measurements) passed exact validation
and measured, on the slowest PE:

```text
TileLang GEMM          0.746 ms
DUSHMEM AllReduce    110.936 ms
serial wall total    111.725 ms
fused wall total      18.881 ms
```

Thus this installed DUSHMEM collective baseline is about 5.92x slower than the
original 16-communication-block fused prototype, and about 23.08x slower than
the confirmed 66-communication-block result. The on-stream reduce also emits the
DTK 26.04 warning that its internal kernel requests 1024 threads despite a
256-thread launch bound, although every measured iteration writes a correct
result after the destination is cleared.
