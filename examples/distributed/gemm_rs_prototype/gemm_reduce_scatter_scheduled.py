"""Bounded 80-CTA GEMM + ReduceScatter schedule candidate.

This is a source-generating diagnostic that mirrors the hand-written
compute/communication split for one four-PE round.  It deliberately has a
fixed, fully-resident grid rather than a queue or general persistent-kernel
claim.  The old one-CTA prototype remains the diagnostic baseline.  The CLI
only generates source and enables the verified IPC communication-vectorization
optimization, selecting its validated communication-CTA default when possible.
"""
import argparse
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.distributed.ipc_binding import bind_ipc

from gemm_reduce_scatter import BLOCK_K, PE_COUNT, THREADS, TILE_M, TILE_N


DEFAULT_TOTAL_BLOCKS = 80
DEFAULT_COMM_BLOCKS = 32
VALIDATED_COMM_BLOCKS = {
    (2048, 2048, 512): 32,
    (4096, 4096, 1024): 32,
    (8192, 8192, 2048): 8,
    (16384, 16384, 4096): 4,
}


def default_comm_blocks(m: int, n: int, k: int, total_blocks: int) -> int:
    """Return the validated communication-CTA count for a default invocation."""
    if total_blocks == DEFAULT_TOTAL_BLOCKS:
        return VALIDATED_COMM_BLOCKS.get((m, n, k), DEFAULT_COMM_BLOCKS)
    return DEFAULT_COMM_BLOCKS


def _check_shape(m: int, n: int, k: int) -> None:
    """Validate the bounded reference's geometry and four-stage K pipeline."""
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("M, N, and K must be positive")
    if m % (PE_COUNT * TILE_M):
        raise ValueError(f"M must be a multiple of {PE_COUNT * TILE_M}")
    if n % TILE_N:
        raise ValueError(f"N must be a multiple of {TILE_N}")
    # Four pipeline stages are only an intended reference configuration once
    # there are at least four K tiles.  Keep this stricter than the legacy
    # single-CTA check so a source-only success cannot imply this schedule was
    # exercised with an underfilled pipeline.
    if k < 4 * BLOCK_K or k % (4 * BLOCK_K):
        raise ValueError(f"K must be at least {4 * BLOCK_K} and a multiple of {4 * BLOCK_K}")


def schedule_parameters(m: int, n: int, total_blocks: int = DEFAULT_TOTAL_BLOCKS,
                        comm_blocks: int = DEFAULT_COMM_BLOCKS) -> dict[str, int]:
    """Pure schedule facts used by the CPU checks and runner allocation."""
    if total_blocks <= 1:
        raise ValueError("total_blocks must be greater than one")
    if not 0 < comm_blocks < total_blocks:
        raise ValueError("comm_blocks must be in [1, total_blocks - 1]")
    tile_count = (m // TILE_M) * (n // TILE_N)
    tile_cols = n // TILE_N
    if tile_count % PE_COUNT:
        raise ValueError("tile count must divide evenly among the four row-shard owners")
    return {
        "tile_count": tile_count,
        "tiles_per_rank": tile_count // PE_COUNT,
        "tile_cols": tile_cols,
        "tile_rows_per_rank": (m // TILE_M) // PE_COUNT,
        "compute_blocks": total_blocks - comm_blocks,
        "total_blocks": total_blocks,
        "comm_blocks": comm_blocks,
        # tile ready: tile_count * four sources; then peer completion and one
        # local completion slot per communication CTA.
        "flag_words": tile_count * PE_COUNT + PE_COUNT + comm_blocks,
    }


def interleaved_tile(job: int, tile_count: int) -> int:
    """Map a compute job to an owner-interleaved, owner-monotonic tile."""
    tiles_per_rank = tile_count // PE_COUNT
    return (job % PE_COUNT) * tiles_per_rank + job // PE_COUNT


def owner_for_tile(tile: int, tile_count: int) -> int:
    return tile // (tile_count // PE_COUNT)


def flag_layout(tile_count: int, comm_blocks: int) -> dict[str, int]:
    tile_ready_words = tile_count * PE_COUNT
    return {
        "tile_ready_words": tile_ready_words,
        "peer_done_base": tile_ready_words,
        "block_done_base": tile_ready_words + PE_COUNT,
        "flag_words": tile_ready_words + PE_COUNT + comm_blocks,
    }


def gemm_reduce_scatter_scheduled(
    m: int = 2048, n: int = 2048, k: int = 512, *,
    total_blocks: int = DEFAULT_TOTAL_BLOCKS, comm_blocks: int = DEFAULT_COMM_BLOCKS,
):
    """Build the fixed-grid TileLang candidate with separate CTA roles.

    ``block_idx < compute_blocks`` performs owner-interleaved GEMM jobs.  The
    remaining CTAs are the owner-local RS workers and end in the explicit
    reader-safe completion handshake.  All shape/schedule values are compile
    parameters, so no user-controlled grid/role interface is added.
    """
    _check_shape(m, n, k)
    schedule = schedule_parameters(m, n, total_blocks, comm_blocks)
    tile_count = schedule["tile_count"]
    tiles_per_rank = schedule["tiles_per_rank"]
    tile_rows_per_rank = schedule["tile_rows_per_rank"]
    compute_blocks = schedule["compute_blocks"]
    flags = flag_layout(tile_count, comm_blocks)
    shard_m = m // PE_COUNT

    @T.prim_func
    def fused_gemm_rs_scheduled(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        partial: T.Tensor((tile_count, TILE_M, TILE_N), "float16"),
        ready: T.Tensor((flags["flag_words"],), "uint32"),
        out: T.Tensor((shard_m, n), "float16"),
    ):
        with T.Kernel(total_blocks, threads=THREADS) as block_idx:
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
            p0 = T.alloc_local((1,), "float16")
            p1 = T.alloc_local((1,), "float16")
            p2 = T.alloc_local((1,), "float16")
            p3 = T.alloc_local((1,), "float16")

            if block_idx < compute_blocks:
                # These allocations and the four-stage register/warp pipeline
                # intentionally retain the established 256x256/K16 GEMM path.
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                # This is the explicit resource-scope contract for the
                # owner-interleaved output-tile traversal.  It is not a
                # generic persistent-kernel hint and does not apply to the
                # communication serial loops below.
                for job in T.serial(
                    block_idx, tile_count, compute_blocks,
                    annotations={"tl.dcu_persistent_tile": 1},
                ):
                    # Give owners 0,1,2,3 the first, second, third, fourth
                    # jobs respectively while retaining monotonic local order.
                    tile = (job % PE_COUNT) * tiles_per_rank + job // PE_COUNT
                    tile_row = tile // (n // TILE_N)
                    tile_col = tile % (n // TILE_N)
                    owner = tile // tiles_per_rank
                    T.clear(c_local)
                    for kk in T.Pipelined(
                        k // BLOCK_K,
                        num_stages=4,
                        enable_register_pipeline=True,
                        enable_warp_divergence=True,
                    ):
                        T.copy(A[tile_row * TILE_M, kk * BLOCK_K], a_shared,
                               major="row_major", enable_async=True)
                        T.copy(B[kk * BLOCK_K, tile_col * TILE_N], b_shared,
                               major="col_major", enable_async=True)
                        T.gemm(a_shared, b_shared, c_local,
                               k_pack=BLOCK_K // 16, transpose_C=True)
                    T.copy(c_local, partial[tile, 0, 0])
                    # Block publication: every partial writer fences, then
                    # all waves rendezvous before lane zero release-publishes.
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
                # A rank owns a continuous row shard.  Its communication CTAs
                # take those tiles round-robin after the compute interleaving.
                for owned in T.serial(lane, tiles_per_rank, comm_blocks):
                    tile = rank * tiles_per_rank + owned
                    tile_row = tile // (n // TILE_N)
                    tile_col = tile % (n // TILE_N)
                    if tx == 0:
                        T.wait_eq(ready[tile * PE_COUNT + 0], 1)
                        T.wait_eq(ready[tile * PE_COUNT + 1], 1)
                        T.wait_eq(ready[tile * PE_COUNT + 2], 1)
                        T.wait_eq(ready[tile * PE_COUNT + 3], 1)
                        T.fence_sys()
                    T.sync_threads()
                    # Keep communication's historic opaque thread coordinate
                    # separate from the GEMM raw thread/wave/lane lowering.
                    comm_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
                    for linear in T.serial(comm_tx, TILE_M * TILE_N, THREADS):
                        row = linear // TILE_N
                        col = linear % TILE_N
                        p0[0] = remote_partial0[tile, row, col]
                        p1[0] = remote_partial1[tile, row, col]
                        p2[0] = remote_partial2[tile, row, col]
                        p3[0] = remote_partial3[tile, row, col]
                        out[(tile_row % tile_rows_per_rank) * TILE_M + row, tile_col * TILE_N + col] = T.cast(
                            T.cast(p0[0], "float32") + T.cast(p1[0], "float32")
                            + T.cast(p2[0], "float32") + T.cast(p3[0], "float32"), "float16")
                    T.fence_sys()
                    T.sync_threads()

                # The trailing flags are distinct from tile-ready slots.  A
                # local block completion is release-published by every comm
                # CTA; lane zero then publishes reader-safe to all PEs and
                # waits for all reciprocal peer completions before return.
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

    return fused_gemm_rs_scheduled


def build_bound(m: int, n: int, k: int, *, total_blocks: int = DEFAULT_TOTAL_BLOCKS,
                comm_blocks: int = DEFAULT_COMM_BLOCKS):
    func = gemm_reduce_scatter_scheduled(
        m, n, k, total_blocks=total_blocks, comm_blocks=comm_blocks,
    )
    return func, bind_ipc(func, {"partial": "ipc_view_partial", "ready": "ipc_view_ready"})


def source_only(m: int, n: int, k: int, output: Path, *, total_blocks: int,
                comm_blocks: int) -> None:
    """Emit source artifacts only, with verified scheduled-RS IPC vectorization."""
    raw, bound = build_bound(m, n, k, total_blocks=total_blocks, comm_blocks=comm_blocks)
    with tvm.transform.PassContext(config={"tl.enable_dcu_ipc_vectorize": True}):
        artifact = tilelang.lower(
            bound, target="hip -mcpu=gfx936", enable_device_compile=False, enable_host_codegen=False,
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw.tir").write_text(raw.script(), encoding="utf-8")
    (output / "bound.tir").write_text(bound.script(), encoding="utf-8")
    (output / "device.tir").write_text(artifact.device_mod.script(), encoding="utf-8")
    (output / "gemm_reduce_scatter_scheduled.hip").write_text(artifact.kernel_source, encoding="utf-8")
    print(f"source-only HIP: {output / 'gemm_reduce_scatter_scheduled.hip'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--total-blocks", type=int, default=DEFAULT_TOTAL_BLOCKS)
    parser.add_argument("--comm-blocks", type=int, default=None,
                        help="Communication CTAs; defaults to the validated shape-specific value")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "generated" / "scheduled")
    args = parser.parse_args()
    comm_blocks = args.comm_blocks
    if comm_blocks is None:
        comm_blocks = default_comm_blocks(args.m, args.n, args.k, args.total_blocks)
        comm_source = "auto"
    else:
        comm_source = "explicit"
    print(
        f"scheduled-RS configuration: total_blocks={args.total_blocks}, "
        f"comm_blocks={comm_blocks} ({comm_source}), "
        "dcu_ipc_vectorize=True",
        flush=True,
    )
    source_only(args.m, args.n, args.k, args.output,
                total_blocks=args.total_blocks, comm_blocks=comm_blocks)


if __name__ == "__main__":
    main()
