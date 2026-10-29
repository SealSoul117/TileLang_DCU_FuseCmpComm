"""Diagnostic GEMM + ReduceScatter variant with one CTA per output tile.

Unlike the default prototype this file has no outer serial tile loop: the
ordinary 2-D TileLang kernel grid directly supplies (tile_col, tile_row).
It is capped at eight CTAs per PE for the local gfx936/64KiB experiment.
"""
import argparse
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.distributed.ipc_binding import bind_ipc

from gemm_reduce_scatter import BLOCK_K, PE_COUNT, THREADS, TILE_M, TILE_N, _check_shape


def tile_grid_gemm_reduce_scatter(m: int = 1024, n: int = 256, k: int = 256):
    _check_shape(m, n, k)
    tile_rows, tile_cols = m // TILE_M, n // TILE_N
    tiles = tile_rows * tile_cols
    if tiles > 8:
        raise ValueError("tile-grid prototype is capped at 8 CTAs per PE")
    tiles_per_rank, shard_m = tile_rows // PE_COUNT, m // PE_COUNT

    @T.prim_func
    def fused_gemm_rs_tile_grid(
        a: T.Tensor((m, k), "float16"),
        b: T.Tensor((k, n), "float16"),
        partial: T.Tensor((tiles, TILE_M, TILE_N), "float16"),
        ready: T.Tensor((tiles, PE_COUNT), "uint32"),
        out: T.Tensor((shard_m, n), "float16"),
    ):
        # No serial tile traversal exists here.  Each PE has the same bounded
        # tile grid, so its row-shard owner CTA is independently launchable.
        with T.Kernel(tile_cols, tile_rows, threads=THREADS) as (tile_col, tile_row):
            tx = T.get_thread_binding()
            rank = T.get_rank()
            tile = tile_row * tile_cols + tile_col
            owner = tile_row // tiles_per_rank
            remote_partial0 = T.remote_view(partial, 0)
            remote_partial1 = T.remote_view(partial, 1)
            remote_partial2 = T.remote_view(partial, 2)
            remote_partial3 = T.remote_view(partial, 3)
            remote_ready0 = T.remote_view(ready, 0)
            remote_ready1 = T.remote_view(ready, 1)
            remote_ready2 = T.remote_view(ready, 2)
            remote_ready3 = T.remote_view(ready, 3)
            p0 = T.alloc_local((1,), "float16")
            p1 = T.alloc_local((1,), "float16")
            p2 = T.alloc_local((1,), "float16")
            p3 = T.alloc_local((1,), "float16")
            a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
            b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
            c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
            T.clear(c_local)
            for kk in T.Pipelined(k // BLOCK_K, num_stages=4,
                                  enable_register_pipeline=True, enable_warp_divergence=True):
                T.copy(a[tile_row * TILE_M, kk * BLOCK_K], a_shared,
                       major="row_major", enable_async=True)
                T.copy(b[kk * BLOCK_K, tile_col * TILE_N], b_shared,
                       major="col_major", enable_async=True)
                T.gemm(a_shared, b_shared, c_local, k_pack=BLOCK_K // 16, transpose_C=True)
            T.copy(c_local, partial[tile, 0, 0])
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
            if rank == owner:
                if tx == 0:
                    T.wait_eq(ready[tile, 0], 1)
                    T.wait_eq(ready[tile, 1], 1)
                    T.wait_eq(ready[tile, 2], 1)
                    T.wait_eq(ready[tile, 3], 1)
            T.sync_threads()
            if rank == owner:
                # Preserve the historic opaque communication coordinate; do
                # not expose it to the GEMM thread/wave/lane algebra.
                comm_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
                for linear in T.serial(comm_tx, TILE_M * TILE_N, THREADS):
                    row, col = linear // TILE_N, linear % TILE_N
                    p0[0] = remote_partial0[tile, row, col]
                    p1[0] = remote_partial1[tile, row, col]
                    p2[0] = remote_partial2[tile, row, col]
                    p3[0] = remote_partial3[tile, row, col]
                    out[(tile_row % tiles_per_rank) * TILE_M + row, tile_col * TILE_N + col] = T.cast(
                        T.cast(p0[0], "float32") + T.cast(p1[0], "float32")
                        + T.cast(p2[0], "float32") + T.cast(p3[0], "float32"), "float16")
            T.sync_threads()

    return fused_gemm_rs_tile_grid


def build_bound(m: int, n: int, k: int):
    func = tile_grid_gemm_reduce_scatter(m, n, k)
    return func, bind_ipc(func, {"partial": "ipc_view_partial", "ready": "ipc_view_ready"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "generated" / "tile_grid")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    raw, bound = build_bound(args.m, args.n, args.k)
    artifact = tilelang.lower(bound, target="hip -mcpu=gfx936",
                             enable_device_compile=False, enable_host_codegen=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.tir").write_text(raw.script(), encoding="utf-8")
    (args.output / "bound.tir").write_text(bound.script(), encoding="utf-8")
    (args.output / "device.tir").write_text(artifact.device_mod.script(), encoding="utf-8")
    (args.output / "gemm_reduce_scatter_tile_grid.hip").write_text(artifact.kernel_source, encoding="utf-8")
    print(f"source-only HIP: {args.output / 'gemm_reduce_scatter_tile_grid.hip'}", flush=True)
    if args.compile:
        tilelang.disable_cache()
        kernel = tilelang.compile(bound, target="hip -mcpu=gfx936", execution_backend="ctypes")
        (args.output / "gemm_reduce_scatter_tile_grid.hip").write_text(kernel.get_kernel_source(), encoding="utf-8")
        print(f"compiled HIP: {args.output / 'gemm_reduce_scatter_tile_grid.hip'}", flush=True)


if __name__ == "__main__":
    main()
