"""Four-PE numerical runner for the GEMM + ReduceScatter prototypes.

Run it under four MPI ranks on the same DCU node.  It compiles the public
TileLang PrimFunc, launches one kernel per rank, and verifies both the local
FP16 partial slots and the final continuous FP16 row shard against a CPU
oracle.  It is intentionally a correctness runner, never a benchmark.
"""
import argparse
import ctypes
import os
import time

import numpy as np
import tilelang

from gemm_reduce_scatter import PE_COUNT, TILE_M, TILE_N, _check_shape, build_bound
from gemm_reduce_scatter_tile_grid import build_bound as build_flat_tile_grid_bound


HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2


def _check(code: int, what: str) -> None:
    if code:
        raise RuntimeError(f"{what}: HIP status {code}")


def _hip():
    api = ctypes.CDLL("libamdhip64.so")
    api.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    api.hipMemcpy.restype = ctypes.c_int
    api.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    api.hipMemset.restype = ctypes.c_int
    api.hipDeviceSynchronize.restype = ctypes.c_int
    api.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    api.hipEventCreate.restype = ctypes.c_int
    api.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    api.hipEventRecord.restype = ctypes.c_int
    api.hipEventQuery.argtypes = [ctypes.c_void_p]
    api.hipEventQuery.restype = ctypes.c_int
    api.hipEventDestroy.argtypes = [ctypes.c_void_p]
    api.hipEventDestroy.restype = ctypes.c_int
    return api


def _copy(api, dst: int, src: int, nbytes: int, kind: int) -> None:
    _check(api.hipMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, kind), "hipMemcpy")


def _rank_inputs(rank: int, m: int, n: int, k: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return distinct deterministic continuous or exactly representable inputs."""
    rng = np.random.default_rng(1009 + rank * 313)
    if mode == "continuous":
        a = rng.uniform(-0.25, 0.25, size=(m, k)).astype(np.float16)
        b = rng.uniform(-0.25, 0.25, size=(k, n)).astype(np.float16)
    elif mode == "dyadic":
        # Products are multiples of 1/256; the FP32 K<=512 accumulation is
        # exact, making a bitwise partial/output comparison meaningful.
        a = (rng.integers(-8, 9, size=(m, k)) / 16).astype(np.float16)
        b = (rng.integers(-8, 9, size=(k, n)) / 16).astype(np.float16)
    else:
        raise ValueError(f"unknown input mode: {mode}")
    return a, b


def _partial_reference(rank: int, m: int, n: int, k: int, mode: str) -> np.ndarray:
    a, b = _rank_inputs(rank, m, n, k, mode)
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)


def _tile_view(matrix: np.ndarray) -> np.ndarray:
    m, n = matrix.shape
    return matrix.reshape(m // TILE_M, TILE_M, n // TILE_N, TILE_N).transpose(0, 2, 1, 3).reshape(
        (m // TILE_M) * (n // TILE_N), TILE_M, TILE_N
    )


def _wait_kernel(api, seconds: float = 60.0) -> None:
    event = ctypes.c_void_p()
    _check(api.hipEventCreate(ctypes.byref(event)), "hipEventCreate")
    try:
        _check(api.hipEventRecord(event, None), "hipEventRecord")
        deadline = time.monotonic() + seconds
        while api.hipEventQuery(event) == 600:  # hipErrorNotReady
            if time.monotonic() > deadline:
                raise RuntimeError("GEMM + ReduceScatter kernel timeout")
            time.sleep(0.01)
        _check(api.hipEventQuery(event), "hipEventQuery")
    finally:
        _check(api.hipEventDestroy(event), "hipEventDestroy")


def _check_result(actual: np.ndarray, expected: np.ndarray, label: str, mode: str, rank: int) -> None:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    max_abs = float(np.nanmax(diff, initial=0.0))
    nan_count = int(np.count_nonzero(np.isnan(actual)))
    bit_mismatches = int(np.count_nonzero(actual.view(np.uint16) != expected.view(np.uint16)))
    print(f"rank={rank} {label}: max_abs={max_abs:.8g} nan_count={nan_count} "
          f"bit_mismatches={bit_mismatches}", flush=True)
    if mode == "dyadic":
        if bit_mismatches:
            raise AssertionError(f"{label}: dyadic oracle requires bitwise equality")
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def run(m: int, n: int, k: int, mode: str, sync_copy: bool, tile_grid: bool,
        flat_tile_grid: bool) -> None:
    _check_shape(m, n, k)
    mpi = ctypes.CDLL("libmpi.so")
    provided = ctypes.c_int()
    _check(mpi.MPI_Init_thread(None, None, 1, ctypes.byref(provided)), "MPI_Init_thread")
    from tilelang.distributed.ipc_runtime import IPCContext

    ctx = IPCContext()
    rank = ctx.rank
    if ctx.world_size != PE_COUNT:
        raise RuntimeError(f"this prototype requires {PE_COUNT} MPI ranks, got {ctx.world_size}")
    tile_count = (m // TILE_M) * (n // TILE_N)
    api = _hip()
    # FineGrained symmetric allocations; only partial/ready are remote views.
    allocations = [
        ctx.alloc("gemm-rs-a", m * k * 2),
        ctx.alloc("gemm-rs-b", k * n * 2),
        ctx.alloc("gemm-rs-partial", tile_count * TILE_M * TILE_N * 2),
        ctx.alloc("gemm-rs-ready", tile_count * PE_COUNT * 4),
        ctx.alloc("gemm-rs-out", (m // PE_COUNT) * n * 2),
    ]
    a_dev, b_dev, partial_dev, ready_dev, out_dev = allocations
    launched = False
    reader_safe = False
    try:
        a_host, b_host = _rank_inputs(rank, m, n, k, mode)
        _copy(api, a_dev.data_ptr(), a_host.ctypes.data, a_host.nbytes, HOST_TO_DEVICE)
        _copy(api, b_dev.data_ptr(), b_host.ctypes.data, b_host.nbytes, HOST_TO_DEVICE)
        _check(api.hipMemset(ctypes.c_void_p(partial_dev.data_ptr()), 0, tile_count * TILE_M * TILE_N * 2),
               "partial reset")
        _check(api.hipMemset(ctypes.c_void_p(ready_dev.data_ptr()), 0, tile_count * PE_COUNT * 4), "ready reset")
        _check(api.hipMemset(ctypes.c_void_p(out_dev.data_ptr()), 0, (m // PE_COUNT) * n * 2), "out reset")
        _check(api.hipDeviceSynchronize(), "input initialization")
        ctx.sync()

        tilelang.disable_cache()
        if flat_tile_grid:
            if sync_copy or tile_grid:
                raise ValueError("--flat-tile-grid cannot be combined with --sync-copy or --tile-grid")
            _, bound = build_flat_tile_grid_bound(m, n, k)
        else:
            _, bound = build_bound(m, n, k, sync_copy=sync_copy, tile_grid=tile_grid)
        kernel = tilelang.compile(bound, target="hip -mcpu=gfx936", execution_backend="ctypes")
        kernel(a_dev, b_dev, partial_dev, ready_dev, out_dev,
               partial_dev.device_view(), ready_dev.device_view(), rank, ctx.world_size)
        launched = True
        _wait_kernel(api)
        # All PE kernels complete before copies/free; this is not an in-kernel
        # ready reset or ACK/reuse protocol.
        ctx.sync()
        reader_safe = True

        partial_host = np.empty((tile_count, TILE_M, TILE_N), dtype=np.float16)
        out_host = np.empty((m // PE_COUNT, n), dtype=np.float16)
        _copy(api, partial_host.ctypes.data, partial_dev.data_ptr(), partial_host.nbytes, DEVICE_TO_HOST)
        _copy(api, out_host.ctypes.data, out_dev.data_ptr(), out_host.nbytes, DEVICE_TO_HOST)
        _check(api.hipDeviceSynchronize(), "result copy")

        local_partial = _partial_reference(rank, m, n, k, mode)
        _check_result(partial_host, _tile_view(local_partial), "partial", mode, rank)
        reduced = sum(_partial_reference(peer, m, n, k, mode).astype(np.float32) for peer in range(PE_COUNT))
        expected = reduced.astype(np.float16)[rank * (m // PE_COUNT):(rank + 1) * (m // PE_COUNT), :]
        _check_result(out_host, expected, "reduce-scatter", mode, rank)
        print(f"gemm-rs-prototype-ok rank={rank} M={m} N={n} K={k} mode={mode} "
              f"sync_copy={sync_copy} tile_grid={tile_grid} flat_tile_grid={flat_tile_grid}",
              flush=True)
    except BaseException:
        if launched and not reader_safe:
            # A timed-out/incomplete kernel may still have remote readers.
            # Do not enter collective allocation close from one failed rank;
            # the external MPI timeout tears down the process group instead.
            print(f"rank={rank}: kernel not reader-safe; bypassing collective free", flush=True)
            os._exit(1)
        raise
    finally:
        if reader_safe or not launched:
            for allocation in reversed(allocations):
                allocation.close()
            ctx.close()
            _check(mpi.MPI_Finalize(), "MPI_Finalize")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--input-mode", choices=("continuous", "dyadic"), default="continuous")
    parser.add_argument("--sync-copy", action="store_true",
                        help="diagnostic serial-K/non-async-copy schedule")
    parser.add_argument("--tile-grid", action="store_true",
                        help="diagnostic <=8-CTA-per-PE grid")
    parser.add_argument("--flat-tile-grid", action="store_true",
                        help="direct no-outer-loop <=8-CTA grid from gemm_reduce_scatter_tile_grid.py")
    args = parser.parse_args()
    run(args.m, args.n, args.k, args.input_mode, args.sync_copy, args.tile_grid, args.flat_tile_grid)


if __name__ == "__main__":
    main()
