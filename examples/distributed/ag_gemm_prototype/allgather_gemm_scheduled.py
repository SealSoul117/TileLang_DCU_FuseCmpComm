"""Bounded four-PE AllGather + GEMM prototype with scalar IPC copy frontend."""

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.distributed.ipc_binding import bind_ipc


PE_COUNT = 4
THREADS = 512
TILE_M = TILE_N = 256
BLOCK_K = 16
TOTAL_BLOCKS = 80
CHUNK_ELEMENTS = 65536
COMM_BY_SHAPE = {2048: 32, 4096: 32, 8192: 24, 16384: 16}


def _check_shape(m: int, n: int, k: int) -> None:
    if m != k or n != m // PE_COUNT or m not in COMM_BY_SHAPE:
        raise ValueError("supported shapes are M=K=2048/4096/8192/16384 and N=M/4")


def schedule_parameters(m: int, n: int, k: int, *, total_blocks: int = TOTAL_BLOCKS,
                        comm_blocks: int | None = None) -> dict[str, int]:
    _check_shape(m, n, k)
    if total_blocks != TOTAL_BLOCKS:
        raise ValueError("the prototype fixes the full-resident grid at 80 CTAs")
    if comm_blocks is None:
        comm_blocks = COMM_BY_SHAPE[m]
    if not 0 < comm_blocks < total_blocks:
        raise ValueError("comm_blocks must be in [1, 79]")
    shard_rows = m // PE_COUNT
    if shard_rows % TILE_M or CHUNK_ELEMENTS % k or (TILE_M * k) % CHUNK_ELEMENTS:
        raise ValueError("the fixed 65536-half chunks must partition shard and 256-row tiles")
    rows_per_chunk = CHUNK_ELEMENTS // k
    chunks_per_row_tile = TILE_M // rows_per_chunk
    chunks_per_source = shard_rows // rows_per_chunk
    total_chunks = chunks_per_source * PE_COUNT
    return {
        "total_blocks": total_blocks,
        "comm_blocks": comm_blocks,
        "compute_blocks": total_blocks - comm_blocks,
        "shard_rows": shard_rows,
        "rows_per_chunk": rows_per_chunk,
        "chunks_per_row_tile": chunks_per_row_tile,
        "chunks_per_source": chunks_per_source,
        "total_chunks": total_chunks,
        "tile_rows": m // TILE_M,
        "tile_cols": n // TILE_N,
        "tile_count": (m // TILE_M) * (n // TILE_N),
        "peer_done_base": total_chunks,
        "block_done_base": total_chunks + PE_COUNT,
        "flag_words": total_chunks + PE_COUNT + comm_blocks,
    }


def allgather_gemm_scheduled(m: int = 2048, n: int = 512, k: int = 2048, *,
                             total_blocks: int = TOTAL_BLOCKS,
                             comm_blocks: int | None = None):
    """Generate fixed 80-CTA gather/copy then GEMM schedule.

    ``A_shard`` is rank-owned FineGrained input. ``A`` is the gathered,
    rank-local workspace deliberately named to follow the ordinary GEMM A/B
    resource path. Communication copies fixed 65536-half chunks; a 256-row
    GEMM input tile waits for every chunk which covers its rows.  Both the
    gather and the persistent GEMM traversal visit this rank's source rows
    first, then rotate through the other three sources.
    """
    schedule = schedule_parameters(m, n, k, total_blocks=total_blocks, comm_blocks=comm_blocks)
    flags = schedule
    shard_rows = schedule["shard_rows"]
    chunks_per_source = schedule["chunks_per_source"]
    rows_per_chunk = schedule["rows_per_chunk"]
    chunks_per_row_tile = schedule["chunks_per_row_tile"]
    total_chunks = schedule["total_chunks"]
    # Mark every output-tile traversal.  At S=2048 there are fewer output
    # tiles than compute CTAs.  Its active CTAs therefore use an explicit
    # one-trip loop after a tile-count guard, avoiding a dynamic 0/1 trip
    # loop around the GEMM epilogue.  Larger shapes keep the regular strided
    # traversal and its tile-varying resource-base proof.
    tile_annotations = {"tl.dcu_persistent_tile": 1}
    single_tile_per_compute = schedule["tile_count"] <= schedule["compute_blocks"]

    @T.prim_func
    def fused_allgather_gemm_scheduled(
        A_shard: T.Tensor((shard_rows, k), "float16"),
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        ready: T.Tensor((flags["flag_words"],), "uint32"),
        out: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(total_blocks, threads=THREADS) as block_idx:
            tx = T.get_thread_binding()
            rank = T.get_rank()
            T.assume(0 <= rank)
            T.assume(rank < PE_COUNT)
            remote_ready0 = T.remote_view(ready, 0)
            remote_ready1 = T.remote_view(ready, 1)
            remote_ready2 = T.remote_view(ready, 2)
            remote_ready3 = T.remote_view(ready, 3)

            if block_idx < schedule["compute_blocks"]:
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                tile_rows_per_source = shard_rows // TILE_M
                tiles_per_source = tile_rows_per_source * schedule["tile_cols"]
                if single_tile_per_compute:
                    # CTAs [tile_count, compute_blocks) remain compute-role
                    # idle; they do not acquire communication ownership.
                    if block_idx < schedule["tile_count"]:
                        for _once in T.serial(1, annotations=tile_annotations):
                            tile = block_idx
                            source_offset = tile // tiles_per_source
                            source = (rank + source_offset) % PE_COUNT
                            source_tile = tile % tiles_per_source
                            source_tile_row = source_tile // schedule["tile_cols"]
                            tile_row = source * tile_rows_per_source + source_tile_row
                            tile_col = source_tile % schedule["tile_cols"]
                            ready_chunk = source * chunks_per_source + source_tile_row * chunks_per_row_tile
                            if tx == 0:
                                for ready_part in T.serial(chunks_per_row_tile):
                                    T.wait_eq(ready[ready_chunk + ready_part], 1)
                                T.fence_sys()
                            T.sync_threads()
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
                            T.copy(c_local, out[tile_row * TILE_M, tile_col * TILE_N])
                else:
                    for tile in T.serial(block_idx, schedule["tile_count"], schedule["compute_blocks"],
                                         annotations=tile_annotations):
                        source_offset = tile // tiles_per_source
                        source = (rank + source_offset) % PE_COUNT
                        source_tile = tile % tiles_per_source
                        source_tile_row = source_tile // schedule["tile_cols"]
                        tile_row = source * tile_rows_per_source + source_tile_row
                        tile_col = source_tile % schedule["tile_cols"]
                        ready_chunk = source * chunks_per_source + source_tile_row * chunks_per_row_tile
                        if tx == 0:
                            for ready_part in T.serial(chunks_per_row_tile):
                                T.wait_eq(ready[ready_chunk + ready_part], 1)
                            T.fence_sys()
                        T.sync_threads()
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
                        T.copy(c_local, out[tile_row * TILE_M, tile_col * TILE_N])
            else:
                lane = block_idx - schedule["compute_blocks"]
                # Round-robin global chunks; source rank zero is this local
                # rank, followed by the other ranks in rotation order.
                for chunk in T.serial(lane, total_chunks, schedule["comm_blocks"]):
                    source_offset = chunk // chunks_per_source
                    source_peer = (rank + source_offset) % PE_COUNT
                    source_chunk = chunk % chunks_per_source
                    physical_chunk = source_peer * chunks_per_source + source_chunk
                    remote_source = T.remote_view(A_shard, source_peer)
                    opaque_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
                    for linear in T.serial(opaque_tx, CHUNK_ELEMENTS, THREADS):
                        A[source_peer * shard_rows + source_chunk * rows_per_chunk + linear // k,
                          linear % k] = remote_source[source_chunk * rows_per_chunk + linear // k, linear % k]
                    T.fence_sys()
                    T.sync_threads()
                    if tx == 0:
                        T.signal_set(ready[physical_chunk], 1)
                    T.sync_threads()

                # Every source PE waits until every reader has completed its
                # gather phase before the runner may recycle A_shard.
                if tx == 0:
                    T.signal_set(ready[flags["block_done_base"] + lane], 1) 
                    if lane == 0:
                        for done_lane in T.serial(schedule["comm_blocks"]):
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

    return fused_allgather_gemm_scheduled


def build_bound(m: int, n: int, k: int, *, total_blocks: int = TOTAL_BLOCKS,
                comm_blocks: int | None = None):
    func = allgather_gemm_scheduled(m, n, k, total_blocks=total_blocks, comm_blocks=comm_blocks)
    return func, bind_ipc(func, {"A_shard": "ipc_view_a_shard", "ready": "ipc_view_ready"})


def source_only(m: int, n: int, k: int, output: Path, *, total_blocks: int = TOTAL_BLOCKS,
                comm_blocks: int | None = None, vectorized: bool = True) -> None:
    raw, bound = build_bound(m, n, k, total_blocks=total_blocks, comm_blocks=comm_blocks)
    with tvm.transform.PassContext(config={"tl.enable_dcu_ipc_vectorize": vectorized}):
        artifact = tilelang.lower(bound, target="hip -mcpu=gfx936",
                                  enable_device_compile=False, enable_host_codegen=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw.tir").write_text(raw.script(), encoding="utf-8")
    (output / "bound.tir").write_text(bound.script(), encoding="utf-8")
    (output / "device.tir").write_text(artifact.device_mod.script(), encoding="utf-8")
    (output / "allgather_gemm_scheduled.hip").write_text(artifact.kernel_source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--total-blocks", type=int, default=TOTAL_BLOCKS)
    parser.add_argument("--comm-blocks", type=int)
    parser.add_argument("--scalar", action="store_true", help="disable the IPC half8 pass")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "generated")
    args = parser.parse_args()
    source_only(args.m, args.n, args.k, args.output, total_blocks=args.total_blocks,
                comm_blocks=args.comm_blocks, vectorized=not args.scalar)


if __name__ == "__main__":
    main()
