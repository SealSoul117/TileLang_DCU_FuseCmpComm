"""Bounded 80-CTA TileLang GEMM + AllReduce prototype.

This intentionally mirrors the tile traversal and completion protocol of the
``fused_vector`` GEMM+AllReduce reference: 80 CTAs of 512 threads, 64 KiB LDS,
and owner ``tile % 4``.  It is a fixed four-PE experiment, not a general
collective implementation.
"""
import argparse
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.distributed.ipc_binding import bind_ipc


PE_COUNT = 4
THREADS = 512
TILE_M = 256
TILE_N = 256
BLOCK_K = 16
DEFAULT_TOTAL_BLOCKS = 80
# The reference uses these shape-specific compute/communication partitions.
VALIDATED_COMM_BLOCKS = {
    (2048, 2048, 512): 48,
    (4096, 4096, 1024): 24,
    (8192, 8192, 2048): 16,
    (16384, 16384, 4096): 8,
}


def default_comm_blocks(m: int, n: int, k: int, total_blocks: int = DEFAULT_TOTAL_BLOCKS) -> int:
    if total_blocks != DEFAULT_TOTAL_BLOCKS:
        raise ValueError("the fused-vector reference has a fixed 80-CTA grid")
    try:
        return VALIDATED_COMM_BLOCKS[(m, n, k)]
    except KeyError as error:
        raise ValueError("supported shapes are M=N=2048/4096/8192/16384 with K=M/4") from error


def _check_shape(m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0 or m % TILE_M or n % TILE_N or k % (4 * BLOCK_K):
        raise ValueError("M/N must be multiples of 256 and K a multiple of 64")


def schedule_parameters(m: int, n: int, total_blocks: int = DEFAULT_TOTAL_BLOCKS,
                        comm_blocks: int | None = None, *, k: int | None = None) -> dict[str, int]:
    """Return the fixed schedule facts shared by generation and benchmarking."""
    k = m // 4 if k is None else k
    _check_shape(m, n, k)
    if total_blocks != DEFAULT_TOTAL_BLOCKS:
        raise ValueError("this allreduce prototype requires exactly 80 total CTAs")
    if comm_blocks is None:
        comm_blocks = default_comm_blocks(m, n, k)
    if not 0 < comm_blocks < total_blocks:
        raise ValueError("comm_blocks must be in [1, 79]")
    tile_count = (m // TILE_M) * (n // TILE_N)
    if tile_count % PE_COUNT:
        raise ValueError("tile count must be divisible by four")
    # Four tile-ready sources per output tile, one peer-done flag per source
    # PE, then one local completion flag per communication CTA.
    return {
        "tile_count": tile_count,
        "tiles_per_rank": tile_count // PE_COUNT,
        "tile_cols": n // TILE_N,
        "total_blocks": total_blocks,
        "comm_blocks": comm_blocks,
        "compute_blocks": total_blocks - comm_blocks,
        "tile_ready_words": tile_count * PE_COUNT,
        "peer_done_base": tile_count * PE_COUNT,
        "block_done_base": tile_count * PE_COUNT + PE_COUNT,
        "flag_words": tile_count * PE_COUNT + PE_COUNT + comm_blocks,
    }


def gemm_allreduce_scheduled(m: int = 2048, n: int = 2048, k: int = 512, *,
                             total_blocks: int = DEFAULT_TOTAL_BLOCKS,
                             comm_blocks: int | None = None):
    """Build the fixed full-resident GEMM+AllReduce candidate.

    Compute CTAs use ``tile = block + iteration * compute_blocks`` exactly as
    the reference does; that is deliberately different from ReduceScatter's
    owner-interleaved traversal.  Communication owners consume tile
    ``rank + owned * 4`` and write the completed tile to every PE's full
    row-major output allocation.
    """
    _check_shape(m, n, k)
    schedule = schedule_parameters(m, n, total_blocks, comm_blocks, k=k)
    tile_count = schedule["tile_count"]
    tile_cols = schedule["tile_cols"]
    compute_blocks = schedule["compute_blocks"]
    comm_blocks = schedule["comm_blocks"]
    flags = schedule

    @T.prim_func
    def fused_gemm_allreduce_scheduled(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        partial: T.Tensor((tile_count, TILE_M, TILE_N), "float16"),
        ready: T.Tensor((flags["flag_words"],), "uint32"),
        out: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(total_blocks, threads=THREADS) as block_idx:
            tx = T.get_thread_binding()
            rank = T.get_rank()
            # The fixed runner binds exactly four IPC peers.  Keep the rank
            # range in front-end IR without changing the launch ABI.
            T.assume(0 <= rank)
            T.assume(rank < PE_COUNT)
            remote_partial0 = T.remote_view(partial, 0)
            remote_partial1 = T.remote_view(partial, 1)
            remote_partial2 = T.remote_view(partial, 2)
            remote_partial3 = T.remote_view(partial, 3)
            remote_ready0 = T.remote_view(ready, 0)
            remote_ready1 = T.remote_view(ready, 1)
            remote_ready2 = T.remote_view(ready, 2)
            remote_ready3 = T.remote_view(ready, 3)
            remote_out0 = T.remote_view(out, rank)
            remote_out1 = T.remote_view(out, (rank + 1) % PE_COUNT)
            remote_out2 = T.remote_view(out, (rank + 2) % PE_COUNT)
            remote_out3 = T.remote_view(out, (rank + 3) % PE_COUNT)
            # Keep this frontend communication body scalar.  The verified
            # IPC vectorization pass owns the optional half8 rewrite.
            p0 = T.alloc_local((1,), "float16")
            p1 = T.alloc_local((1,), "float16")
            p2 = T.alloc_local((1,), "float16")
            p3 = T.alloc_local((1,), "float16")
            reduced = T.alloc_local((1,), "float16")

            if block_idx < compute_blocks:
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                # Keeping the persistent annotation on this outer tile loop
                # is the resource-scope contract used by the established GEMM.
                for tile in T.serial(block_idx, tile_count, compute_blocks,
                                     annotations={"tl.dcu_persistent_tile": 1}):
                    tile_row = tile // tile_cols
                    tile_col = tile % tile_cols
                    owner = tile % PE_COUNT
                    T.clear(c_local)
                    for kk in T.Pipelined(k // BLOCK_K, num_stages=4,
                                          enable_register_pipeline=True,
                                          enable_warp_divergence=True):
                        T.copy(A[tile_row * TILE_M, kk * BLOCK_K], a_shared,
                               major="row_major", enable_async=True)
                        T.copy(B[kk * BLOCK_K, tile_col * TILE_N], b_shared,
                               major="col_major", enable_async=True)
                        T.gemm(a_shared, b_shared, c_local, k_pack=BLOCK_K // 16,
                               transpose_C=True)
                    T.copy(c_local, partial[tile, 0, 0])
                    # Every partial writer completes its data publication
                    # before lane zero release-publishes its source slot.
                    T.fence_sys()
                    T.sync_threads()
                    if tx == 0:
                        if owner == 0:
                            T.signal_set(remote_ready0[tile * PE_COUNT + rank], 1, peer=0)
                        if owner == 1:
                            T.signal_set(remote_ready1[tile * PE_COUNT + rank], 1, peer=1)
                        if owner == 2:
                            T.signal_set(remote_ready2[tile * PE_COUNT + rank], 1, peer=2)
                        if owner == 3:
                            T.signal_set(remote_ready3[tile * PE_COUNT + rank], 1, peer=3)
                    T.sync_threads()
            else:
                lane = block_idx - compute_blocks
                for owned in T.serial(lane, schedule["tiles_per_rank"], comm_blocks):
                    # These are loop-body facts only: inactive communication
                    # CTAs legitimately execute zero iterations.
                    T.assume(0 <= owned)
                    T.assume(owned < schedule["tiles_per_rank"])
                    tile = rank + owned * PE_COUNT
                    tile_row = tile // tile_cols
                    tile_col = tile % tile_cols
                    if tx == 0:
                        T.wait_eq(ready[tile * PE_COUNT + 0], 1)
                        T.wait_eq(ready[tile * PE_COUNT + 1], 1)
                        T.wait_eq(ready[tile * PE_COUNT + 2], 1)
                        T.wait_eq(ready[tile * PE_COUNT + 3], 1)
                        T.fence_sys()
                    T.sync_threads()
                    # Keep opaque indexing inside one owned-tile scope.  The
                    # pass recognizes this scalar loop after pointer hoisting.
                    comm_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
                    for linear in T.serial(comm_tx, TILE_M * TILE_N, THREADS):
                        p0[0] = remote_partial0[tile, linear // TILE_N, linear % TILE_N]
                        p1[0] = remote_partial1[tile, linear // TILE_N, linear % TILE_N]
                        p2[0] = remote_partial2[tile, linear // TILE_N, linear % TILE_N]
                        p3[0] = remote_partial3[tile, linear // TILE_N, linear % TILE_N]
                        reduced[0] = T.cast(T.cast(p0[0], "float32") + T.cast(p1[0], "float32")
                                              + T.cast(p2[0], "float32") + T.cast(p3[0], "float32"), "float16")
                        remote_out0[tile_row * TILE_M + linear // TILE_N, tile_col * TILE_N + linear % TILE_N] = reduced[0]
                        remote_out1[tile_row * TILE_M + linear // TILE_N, tile_col * TILE_N + linear % TILE_N] = reduced[0]
                        remote_out2[tile_row * TILE_M + linear // TILE_N, tile_col * TILE_N + linear % TILE_N] = reduced[0]
                        remote_out3[tile_row * TILE_M + linear // TILE_N, tile_col * TILE_N + linear % TILE_N] = reduced[0]
                    T.fence_sys()
                    T.sync_threads()

                # Do not return merely because this PE has written its local
                # subset: other PEs may still be pushing into our output.
                if tx == 0:
                    T.signal_set(ready[flags["block_done_base"] + lane], 1)
                    if lane == 0:
                        for done_lane in T.serial(comm_blocks):
                            T.wait_eq(ready[flags["block_done_base"] + done_lane], 1)
                        T.signal_set(remote_ready0[flags["peer_done_base"] + rank], 1, peer=0)
                        T.signal_set(remote_ready1[flags["peer_done_base"] + rank], 1, peer=1)
                        T.signal_set(remote_ready2[flags["peer_done_base"] + rank], 1, peer=2)
                        T.signal_set(remote_ready3[flags["peer_done_base"] + rank], 1, peer=3)
                        T.wait_eq(ready[flags["peer_done_base"] + 0], 1)
                        T.wait_eq(ready[flags["peer_done_base"] + 1], 1)
                        T.wait_eq(ready[flags["peer_done_base"] + 2], 1)
                        T.wait_eq(ready[flags["peer_done_base"] + 3], 1)
                        T.fence_sys()
                T.sync_threads()

    return fused_gemm_allreduce_scheduled


def build_bound(m: int, n: int, k: int, *, total_blocks: int = DEFAULT_TOTAL_BLOCKS,
                comm_blocks: int | None = None):
    func = gemm_allreduce_scheduled(m, n, k, total_blocks=total_blocks,
                                    comm_blocks=comm_blocks)
    return func, bind_ipc(func, {
        "partial": "ipc_view_partial", "ready": "ipc_view_ready", "out": "ipc_view_out",
    })


def source_only(m: int, n: int, k: int, output: Path, *, total_blocks: int,
                comm_blocks: int | None, vectorized: bool) -> None:
    raw, bound = build_bound(m, n, k, total_blocks=total_blocks,
                            comm_blocks=comm_blocks)
    # Both variants share the scalar frontend; --scalar disables only the
    # narrow verified IPC half8 pass.
    with tvm.transform.PassContext(config={"tl.enable_dcu_ipc_vectorize": vectorized}):
        artifact = tilelang.lower(bound, target="hip -mcpu=gfx936",
                                  enable_device_compile=False, enable_host_codegen=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw.tir").write_text(raw.script(), encoding="utf-8")
    (output / "bound.tir").write_text(bound.script(), encoding="utf-8")
    (output / "device.tir").write_text(artifact.device_mod.script(), encoding="utf-8")
    (output / "gemm_allreduce_scheduled.hip").write_text(artifact.kernel_source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--total-blocks", type=int, default=DEFAULT_TOTAL_BLOCKS)
    parser.add_argument("--comm-blocks", type=int)
    parser.add_argument("--scalar", action="store_true", help="disable the verified IPC half8 pass")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "generated" / "scheduled")
    args = parser.parse_args()
    source_only(args.m, args.n, args.k, args.output, total_blocks=args.total_blocks,
                comm_blocks=args.comm_blocks, vectorized=not args.scalar)


if __name__ == "__main__":
    main()
