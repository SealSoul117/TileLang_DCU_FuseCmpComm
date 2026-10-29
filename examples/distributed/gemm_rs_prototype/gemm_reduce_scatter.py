"""Four-PE, one-CTA-per-PE GEMM + ReduceScatter IPC prototype.

This is deliberately a source-generating experiment, not an M3 implementation.
The default command only runs the regular TileLang lowering pipeline; it does
not probe a GPU or invoke a device compiler.
"""
import argparse
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.distributed.ipc_binding import bind_ipc


PE_COUNT = 4
TILE_M = 256
TILE_N = 256
BLOCK_K = 16
THREADS = 512


def _check_shape(m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("M, N, and K must be positive")
    if m % (PE_COUNT * TILE_M):
        raise ValueError(f"M must be a multiple of {PE_COUNT * TILE_M}")
    if n % TILE_N:
        raise ValueError(f"N must be a multiple of {TILE_N}")
    if k % BLOCK_K:
        raise ValueError(f"K must be a multiple of {BLOCK_K}")


def gemm_reduce_scatter(m: int = 1024, n: int = 256, k: int = 256, *,
                        sync_copy: bool = False, tile_grid: bool = False):
    """Build the conservative IPC PrimFunc for four local GEMM inputs.

    Each PE computes every ``256 x 256`` output tile into its own FP16
    ``partial[tile]`` slot.  The continuous row-shard owner waits for all four
    source publications, then pulls all four FP16 partials, sums them in FP32,
    and stores its FP16 row-major shard.  ``partial`` is intentionally never
    reused inside the kernel, so this protocol has no in-kernel ACK/reset.
    """
    _check_shape(m, n, k)
    tile_rows = m // TILE_M
    tile_cols = n // TILE_N
    tiles = tile_rows * tile_cols
    tiles_per_rank = tile_rows // PE_COUNT
    shard_m = m // PE_COUNT
    if tile_grid and tiles > 8:
        raise ValueError("tile-grid diagnostic mode is limited to at most 8 CTAs per PE")

    @T.prim_func
    def fused_gemm_rs(
        a: T.Tensor((m, k), "float16"),
        b: T.Tensor((k, n), "float16"),
        partial: T.Tensor((tiles, TILE_M, TILE_N), "float16"),
        ready: T.Tensor((tiles, PE_COUNT), "uint32"),
        out: T.Tensor((shard_m, n), "float16"),
    ):
        # One CTA on every PE traverses tiles in exactly the same order.  This
        # avoids the unscheduled-owner problem, but is intentionally not a
        # persistent/dynamic-role schedule.
        # tile_grid is a diagnostic alternative: one CTA owns one tile.  For
        # <=8 CTAs/PE and this 64KiB profile all tiles are expected to reside
        # together; it is not a general-grid/persistent scheduling claim.
        with T.Kernel(tiles if tile_grid else 1, threads=THREADS) as grid_tile:
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
            # The explicit local FP16 scalars keep each remote half BufferLoad
            # in a supported BufferStore position.  Cast only after that local
            # store; the current remote-alias normalizer rejects Cast(remote).
            p0 = T.alloc_local((1,), "float16")
            p1 = T.alloc_local((1,), "float16")
            p2 = T.alloc_local((1,), "float16")
            p3 = T.alloc_local((1,), "float16")

            # In tile-grid mode this is ``serial(blockIdx.x, blockIdx.x + 1)``;
            # Simplify removes that extent-one wrapper before DCU loop analysis.
            for tile in T.serial(grid_tile if tile_grid else 0,
                                 grid_tile + 1 if tile_grid else tiles):
                tile_row = tile // tile_cols
                tile_col = tile % tile_cols
                owner = tile_row // tiles_per_rank
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                T.clear(c_local)
                if sync_copy:
                    # Diagnostic-only conservative schedule.  It avoids the
                    # software-pipeline injection around the outer tile loop:
                    # copy, CTA barrier, GEMM, CTA barrier for every K block.
                    for kk in T.serial(k // BLOCK_K):
                        T.copy(a[tile_row * TILE_M, kk * BLOCK_K], a_shared,
                               major="row_major", enable_async=False)
                        T.copy(b[kk * BLOCK_K, tile_col * TILE_N], b_shared,
                               major="col_major", enable_async=False)
                        T.sync_threads()
                        T.gemm(a_shared, b_shared, c_local,
                               k_pack=BLOCK_K // 16, transpose_C=True)
                        T.sync_threads()
                else:
                    for kk in T.Pipelined(
                        k // BLOCK_K,
                        num_stages=4,
                        enable_register_pipeline=True,
                        enable_warp_divergence=True,
                    ):
                        T.copy(a[tile_row * TILE_M, kk * BLOCK_K], a_shared,
                               major="row_major", enable_async=True)
                        T.copy(b[kk * BLOCK_K, tile_col * TILE_N], b_shared,
                               major="col_major", enable_async=True)
                        T.gemm(a_shared, b_shared, c_local,
                               k_pack=BLOCK_K // 16, transpose_C=True)
                T.copy(c_local, partial[tile, 0, 0])

                # Every producer thread fences its part of the cooperative
                # tile store; only after CTA rendezvous does lane zero publish.
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

                # The owner does the system-acquire waits; the following CTA
                # barrier shares their publication with all reduction lanes.
                if rank == owner:
                    if tx == 0:
                        T.wait_eq(ready[tile, 0], 1)
                        T.wait_eq(ready[tile, 1], 1)
                        T.wait_eq(ready[tile, 2], 1)
                        T.wait_eq(ready[tile, 3], 1)
                T.sync_threads()

                if rank == owner:
                    # Preserve the historic communication coordinate as an
                    # opaque per-thread value, independently of GEMM's raw
                    # thread/wave/lane lowering.
                    comm_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
                    for linear in T.serial(comm_tx, TILE_M * TILE_N, THREADS):
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
                                "float16")
                T.sync_threads()

    return fused_gemm_rs


def build_bound(m: int, n: int, k: int, *, sync_copy: bool = False, tile_grid: bool = False):
    """Return the ordinary PrimFunc plus explicit IPC descriptors for partial/ready."""
    func = gemm_reduce_scatter(m, n, k, sync_copy=sync_copy, tile_grid=tile_grid)
    return func, bind_ipc(func, {"partial": "ipc_view_partial", "ready": "ipc_view_ready"})


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def source_only(m: int, n: int, k: int, output: Path, *, sync_copy: bool = False,
                tile_grid: bool = False) -> None:
    """Lower without device compilation and save the generated audit artifacts."""
    import tvm

    raw, bound = build_bound(m, n, k, sync_copy=sync_copy, tile_grid=tile_grid)
    artifact = tilelang.lower(
        bound,
        target="hip -mcpu=gfx936",
        enable_device_compile=False,
        enable_host_codegen=False,
    )
    _save_text(output / "raw.tir", raw.script())
    _save_text(output / "bound.tir", bound.script())
    _save_text(output / "device.tir", artifact.device_mod.script())
    _save_text(output / "gemm_reduce_scatter.hip", artifact.kernel_source)
    _save_text(output / "artifact_params.txt", "\n".join(map(str, artifact.params)) + "\n")
    print(f"source-only HIP: {output / 'gemm_reduce_scatter.hip'}", flush=True)
    # Keep tvm imported above as an early, clear dependency failure on a fresh
    # old-tree environment; it is otherwise needed only for TIR inspection.
    del tvm


def compile_hip(m: int, n: int, k: int, output: Path, *, sync_copy: bool = False,
                tile_grid: bool = False) -> None:
    """Request real HIP compilation (requires an allocated DCU node/runtime)."""
    # ctypes provides no supported persistent host-source cache for this IPC
    # path.  A fresh device compilation makes --compile an unambiguous check.
    tilelang.disable_cache()
    _, bound = build_bound(m, n, k, sync_copy=sync_copy, tile_grid=tile_grid)
    kernel = tilelang.compile(bound, target="hip -mcpu=gfx936", execution_backend="ctypes")
    _save_text(output / "gemm_reduce_scatter.hip", kernel.get_kernel_source())
    print(f"compiled HIP: {output / 'gemm_reduce_scatter.hip'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "generated")
    parser.add_argument("--compile", action="store_true",
                        help="also invoke the real ctypes/HIP compilation path")
    parser.add_argument("--sync-copy", action="store_true",
                        help="diagnostic serial-K/non-async-copy schedule, not the default pipeline")
    parser.add_argument("--tile-grid", action="store_true",
                        help="diagnostic <=8-CTA-per-PE grid; no outer tile traversal loop")
    args = parser.parse_args()
    source_only(args.m, args.n, args.k, args.output, sync_copy=args.sync_copy, tile_grid=args.tile_grid)
    if args.compile:
        compile_hip(args.m, args.n, args.k, args.output, sync_copy=args.sync_copy, tile_grid=args.tile_grid)


if __name__ == "__main__":
    main()
