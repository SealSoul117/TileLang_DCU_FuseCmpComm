"""IPC GEMM+ReduceScatter diagnostic with a conservative async-copy K loop.

This is a correctness probe, not a performance design.  Every PE launches one
512-thread CTA, which serially visits output tiles.  Each K step keeps the
normal asynchronous-copy layout metadata but explicitly waits before GEMM and
barriers before reusing LDS.  The IPC publication/reduction protocol is the
same five-tensor M2 prototype protocol used by ``gemm_reduce_scatter.py``.

The default action is source-only lowering.  ``--run`` requires four MPI ranks
on an allocated DCU node; it checks wrapper init/call return values that the
generic ctypes adapter currently ignores.
"""

import argparse
import ctypes
import os
from pathlib import Path

import numpy as np

import tilelang
import tilelang.language as T
from tilelang.distributed.ipc_binding import bind_ipc


PE_COUNT = 4
TILE_M = 256
TILE_N = 256
BLOCK_K = 16
THREADS = 512
TARGET = "hip -mcpu=gfx936"


def _check_shape(m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("M, N, and K must be positive")
    if m % (PE_COUNT * TILE_M):
        raise ValueError(f"M must be a multiple of {PE_COUNT * TILE_M}")
    if n % TILE_N:
        raise ValueError(f"N must be a multiple of {TILE_N}")
    if k % BLOCK_K:
        raise ValueError(f"K must be a multiple of {BLOCK_K}")


def _reject_old_schedule_knobs(sync_copy: bool, tile_grid: bool) -> None:
    if sync_copy or tile_grid:
        raise ValueError(
            "gemm_reduce_scatter_conservative has one fixed schedule; "
            "--sync-copy and --tile-grid are not applicable"
        )


def gemm_reduce_scatter(
    m: int = 1024, n: int = 256, k: int = 256, *,
    sync_copy: bool = False, tile_grid: bool = False,
):
    """Build the five-tensor, one-CTA-per-PE conservative IPC PrimFunc."""
    _reject_old_schedule_knobs(sync_copy, tile_grid)
    _check_shape(m, n, k)
    tile_rows = m // TILE_M
    tile_cols = n // TILE_N
    tiles = tile_rows * tile_cols
    tiles_per_rank = tile_rows // PE_COUNT
    shard_m = m // PE_COUNT

    @T.prim_func
    def fused_gemm_rs_conservative(
        a: T.Tensor((m, k), "float16"),
        b: T.Tensor((k, n), "float16"),
        partial: T.Tensor((tiles, TILE_M, TILE_N), "float16"),
        ready: T.Tensor((tiles, PE_COUNT), "uint32"),
        out: T.Tensor((shard_m, n), "float16"),
    ):
        with T.Kernel(1, threads=THREADS) as _:
            tx = T.get_thread_binding()
            rank = T.get_rank()
            remote_partial0 = T.remote_view(partial, 0)
            remote_partial1 = T.remote_view(partial, 1)
            remote_partial2 = T.remote_view(partial, 2)
            remote_partial3 = T.remote_view(partial, 3)
            remote_ready0 = T.remote_view(ready, 0)
            remote_ready1 = T.remote_view(ready, 1)
            remote_ready2 = T.remote_view(ready, 2)
            remote_ready3 = T.remote_view(ready, 3)
            # Remote FP16 loads must first land in local FP16 scalars.  The
            # existing alias normalizer intentionally rejects Cast(remote).
            p0 = T.alloc_local((1,), "float16")
            p1 = T.alloc_local((1,), "float16")
            p2 = T.alloc_local((1,), "float16")
            p3 = T.alloc_local((1,), "float16")

            # Hoisted storage is intentionally reused by the serial traversal.
            a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
            b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
            c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")

            for tile in T.serial(tiles):
                tile_row = tile // tile_cols
                tile_col = tile % tile_cols
                owner = tile_row // tiles_per_rank
                T.clear(c_local)
                # Conservative K schedule: async copies retain the DCU layout
                # metadata, then fence(0)/CTA-sync makes each step complete
                # before MMAC and before the next step can reuse LDS.
                for kk in T.serial(k // BLOCK_K):
                    T.copy(a[tile_row * TILE_M, kk * BLOCK_K], a_shared,
                           major="row_major", enable_async=True)
                    T.copy(b[kk * BLOCK_K, tile_col * TILE_N], b_shared,
                           major="col_major", enable_async=True)
                    T.evaluate(T.call_intrin(
                        "void", tilelang.tvm.ir.Op.get("tl.async_gld_fence"), 0
                    ))
                    T.sync_threads()
                    T.gemm(a_shared, b_shared, c_local,
                           k_pack=BLOCK_K // 16, transpose_C=True)
                    T.sync_threads()
                T.copy(c_local, partial[tile, 0, 0])
                T.sync_threads()

                # Publication ordering is unchanged: all writers fence and
                # rendezvous, then lane zero release-stores to the tile owner.
                T.fence_sys()
                T.sync_threads()
                if tx == 0:
                    if owner == 0:
                        T.signal_set(remote_ready0[tile, rank], 1, peer=0)
                    if owner == 1:
                        T.signal_set(remote_ready1[tile, rank], 1, peer=1)
                    if owner == 2:
                        T.signal_set(remote_ready2[tile, rank], 1, peer=2)
                    if owner == 3:
                        T.signal_set(remote_ready3[tile, rank], 1, peer=3)

                # Owner lane zero acquire-waits; the CTA barrier publishes that
                # acquisition to the 512 reduction lanes.
                if rank == owner:
                    if tx == 0:
                        T.wait_eq(ready[tile, 0], 1)
                        T.wait_eq(ready[tile, 1], 1)
                        T.wait_eq(ready[tile, 2], 1)
                        T.wait_eq(ready[tile, 3], 1)
                T.sync_threads()

                if rank == owner:
                    for linear in T.serial(tx, TILE_M * TILE_N, THREADS):
                        row = linear // TILE_N
                        col = linear % TILE_N
                        p0[0] = remote_partial0[tile, row, col]
                        p1[0] = remote_partial1[tile, row, col]
                        p2[0] = remote_partial2[tile, row, col]
                        p3[0] = remote_partial3[tile, row, col]
                        out[(tile_row % tiles_per_rank) * TILE_M + row,
                            tile_col * TILE_N + col] = T.cast(
                                T.cast(p0[0], "float32") + T.cast(p1[0], "float32")
                                + T.cast(p2[0], "float32") + T.cast(p3[0], "float32"),
                                "float16",
                            )
                T.sync_threads()

    return fused_gemm_rs_conservative


def build_bound(m: int, n: int, k: int, *, sync_copy: bool = False, tile_grid: bool = False):
    """Return the PrimFunc and the unchanged explicit partial/ready IPC binding."""
    func = gemm_reduce_scatter(m, n, k, sync_copy=sync_copy, tile_grid=tile_grid)
    return func, bind_ipc(func, {"partial": "ipc_view_partial", "ready": "ipc_view_ready"})


def _save(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def source_only(m: int, n: int, k: int, output: Path, *, sync_copy: bool = False,
                tile_grid: bool = False) -> None:
    """Lower without device compilation and save raw/bound/device/HIP artifacts."""
    import tvm

    raw, bound = build_bound(m, n, k, sync_copy=sync_copy, tile_grid=tile_grid)
    with tvm.target.Target(TARGET):
        artifact = tilelang.lower(
            bound, target=TARGET, enable_device_compile=False, enable_host_codegen=False
        )
    _save(output / "raw.tir", raw.script())
    _save(output / "bound.tir", bound.script())
    _save(output / "device.tir", artifact.device_mod.script())
    hip_path = output / "gemm_reduce_scatter_conservative.hip"
    _save(hip_path, artifact.kernel_source)
    _save(output / "artifact_params.txt", "\n".join(map(str, artifact.params)) + "\n")
    print(f"source-only HIP: {hip_path}", flush=True)


def _adapter_error(lib) -> str:
    lib.get_last_error.argtypes = []
    lib.get_last_error.restype = ctypes.c_char_p
    message = lib.get_last_error()
    return message.decode("utf-8", errors="replace") if message else "(no wrapper error text)"


def _check_adapter_init(kernel) -> None:
    """The generic ctypes adapter discards this status, so check it here."""
    lib = kernel.adapter.lib
    lib.init.argtypes = []
    lib.init.restype = ctypes.c_int
    status = lib.init()
    if status:
        raise RuntimeError(f"ctypes wrapper init failed ({status}): {_adapter_error(lib)}")


def compile_hip(m: int, n: int, k: int, output: Path, *, sync_copy: bool = False,
                tile_grid: bool = False):
    """Compile HIP and fail explicitly if wrapper initialization rejects it."""
    import tvm

    tilelang.disable_cache()
    _, bound = build_bound(m, n, k, sync_copy=sync_copy, tile_grid=tile_grid)
    with tvm.target.Target(TARGET):
        kernel = tilelang.compile(bound, target=TARGET, execution_backend="ctypes")
    _check_adapter_init(kernel)
    hip_path = output / "gemm_reduce_scatter_conservative.hip"
    _save(hip_path, kernel.get_kernel_source())
    print(f"compiled HIP: {hip_path}", flush=True)
    return kernel


def _checked_launch(kernel, hip, *args) -> None:
    """Directly invoke the known nine-argument IPC ctypes wrapper ABI."""
    if len(args) != 9:
        raise ValueError(f"expected nine IPC launch arguments, got {len(args)}")
    lib = kernel.adapter.lib
    lib.call.argtypes = ([ctypes.c_void_p] * 7 + [ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p])
    lib.call.restype = ctypes.c_int
    pointers = [ctypes.c_void_p(args[i].data_ptr()) for i in range(7)]
    status = lib.call(*pointers, ctypes.c_int32(args[7]), ctypes.c_int32(args[8]), ctypes.c_void_p(0))
    if status:
        raise RuntimeError(f"ctypes wrapper call failed ({status}): {_adapter_error(lib)}")
    hip_status = hip.hipGetLastError()
    if hip_status:
        message = hip.hipGetErrorString(hip_status)
        detail = message.decode("utf-8", errors="replace") if message else "unknown HIP error"
        raise RuntimeError(f"hipGetLastError after wrapper call failed ({hip_status}): {detail}")


def run(m: int, n: int, k: int, mode: str, output: Path) -> None:
    """Four-rank correctness runner using the existing inputs/oracle/runtime helpers."""
    # Reuse the established allocation, MPI safety, and oracle conventions,
    # but retain init/call error returns instead of the adapter's high-level
    # ``kernel(...)`` entry point.
    import run_gemm_reduce_scatter as runner
    from tilelang.distributed.ipc_runtime import IPCContext

    _check_shape(m, n, k)
    mpi = ctypes.CDLL("libmpi.so")
    provided = ctypes.c_int()
    runner._check(mpi.MPI_Init_thread(None, None, 1, ctypes.byref(provided)), "MPI_Init_thread")
    ctx = IPCContext()
    rank = ctx.rank
    if ctx.world_size != PE_COUNT:
        raise RuntimeError(f"this prototype requires {PE_COUNT} MPI ranks, got {ctx.world_size}")
    tile_count = (m // TILE_M) * (n // TILE_N)
    hip = runner._hip()
    hip.hipGetLastError.argtypes = []
    hip.hipGetLastError.restype = ctypes.c_int
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipGetErrorString.restype = ctypes.c_char_p
    allocations = [
        ctx.alloc("gemm-rs-conservative-a", m * k * 2),
        ctx.alloc("gemm-rs-conservative-b", k * n * 2),
        ctx.alloc("gemm-rs-conservative-partial", tile_count * TILE_M * TILE_N * 2),
        ctx.alloc("gemm-rs-conservative-ready", tile_count * PE_COUNT * 4),
        ctx.alloc("gemm-rs-conservative-out", (m // PE_COUNT) * n * 2),
    ]
    a_dev, b_dev, partial_dev, ready_dev, out_dev = allocations
    launched = False
    reader_safe = False
    try:
        a_host, b_host = runner._rank_inputs(rank, m, n, k, mode)
        runner._copy(hip, a_dev.data_ptr(), a_host.ctypes.data, a_host.nbytes, runner.HOST_TO_DEVICE)
        runner._copy(hip, b_dev.data_ptr(), b_host.ctypes.data, b_host.nbytes, runner.HOST_TO_DEVICE)
        runner._check(hip.hipMemset(ctypes.c_void_p(partial_dev.data_ptr()), 0, tile_count * TILE_M * TILE_N * 2), "partial reset")
        runner._check(hip.hipMemset(ctypes.c_void_p(ready_dev.data_ptr()), 0, tile_count * PE_COUNT * 4), "ready reset")
        runner._check(hip.hipMemset(ctypes.c_void_p(out_dev.data_ptr()), 0, (m // PE_COUNT) * n * 2), "out reset")
        runner._check(hip.hipDeviceSynchronize(), "input initialization")
        ctx.sync()

        # Each MPI rank compiles its own ctypes wrapper.  Keep its generated
        # artifact separate so ranks never concurrently overwrite one file.
        kernel = compile_hip(m, n, k, output / f"rank{rank}")
        # The host wrapper may submit a kernel before reporting a nonzero
        # status.  Treat a failing call as potentially reader-unsafe too.
        launched = True
        _checked_launch(
            kernel, hip, a_dev, b_dev, partial_dev, ready_dev, out_dev,
            partial_dev.device_view(), ready_dev.device_view(), rank, ctx.world_size,
        )
        runner._wait_kernel(hip)
        ctx.sync()
        reader_safe = True

        partial_host = np.empty((tile_count, TILE_M, TILE_N), dtype=np.float16)
        out_host = np.empty((m // PE_COUNT, n), dtype=np.float16)
        runner._copy(hip, partial_host.ctypes.data, partial_dev.data_ptr(), partial_host.nbytes, runner.DEVICE_TO_HOST)
        runner._copy(hip, out_host.ctypes.data, out_dev.data_ptr(), out_host.nbytes, runner.DEVICE_TO_HOST)
        runner._check(hip.hipDeviceSynchronize(), "result copy")
        local_partial = runner._partial_reference(rank, m, n, k, mode)
        runner._check_result(partial_host, runner._tile_view(local_partial), "partial", mode, rank)
        reduced = sum(runner._partial_reference(peer, m, n, k, mode).astype(np.float32) for peer in range(PE_COUNT))
        expected = reduced.astype(np.float16)[rank * (m // PE_COUNT):(rank + 1) * (m // PE_COUNT), :]
        runner._check_result(out_host, expected, "reduce-scatter", mode, rank)
        print(f"gemm-rs-conservative-ok rank={rank} M={m} N={n} K={k} mode={mode}", flush=True)
    except BaseException:
        if launched and not reader_safe:
            print(f"rank={rank}: kernel not reader-safe; bypassing collective free", flush=True)
            os._exit(1)
        raise
    finally:
        if reader_safe or not launched:
            for allocation in reversed(allocations):
                allocation.close()
            ctx.close()
            runner._check(mpi.MPI_Finalize(), "MPI_Finalize")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "generated" / "conservative")
    parser.add_argument("--compile", action="store_true", help="compile HIP and check wrapper init")
    parser.add_argument("--run", action="store_true", help="four-rank run with strict init/call checks")
    parser.add_argument("--input-mode", choices=("continuous", "dyadic"), default="continuous")
    # Retained only to make accidental reuse of old CLI invocations fail loudly.
    parser.add_argument("--sync-copy", action="store_true")
    parser.add_argument("--tile-grid", action="store_true")
    args = parser.parse_args()
    _reject_old_schedule_knobs(args.sync_copy, args.tile_grid)
    if args.run:
        run(args.m, args.n, args.k, args.input_mode, args.output)
    else:
        source_only(args.m, args.n, args.k, args.output)
        if args.compile:
            compile_hip(args.m, args.n, args.k, args.output)


if __name__ == "__main__":
    main()
