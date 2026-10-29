"""Generalized physical-layout AllGather + GEMM TileLang generator."""

import tilelang.language as T

from .contract import (BLOCK_K, CHUNK_ELEMENTS, THREADS, TILE_M, TILE_N,
                       FusionPlan, make_plan)


def allgather_gemm(plan: FusionPlan):
    """Generate a 4/8-PE AG+GEMM DSL function over padded physical tensors.

    ``A_shard`` has one equal physical row slab.  Every source chunk is
    exactly 65536 elements, but K_p need not divide 65536: source addressing
    converts its flat offset back with ``// K_p`` and ``% K_p``.
    """
    if plan.kind != "ag":
        raise ValueError("allgather_gemm requires an ag FusionPlan")
    p = plan.world_size
    row_tiles_per_source = plan.rows_per_rank_physical // TILE_M
    tiles_per_source = row_tiles_per_source * plan.tile_cols

    @T.prim_func
    def fused_allgather_gemm(
        A_shard: T.Tensor(plan.ag_shard_shape, plan.dtype),
        A: T.Tensor(plan.ag_gathered_a_shape, plan.dtype),
        B: T.Tensor(plan.packed_weight_shape, plan.dtype),
        ready: T.Tensor((plan.flag_words,), "uint32"),
        out: T.Tensor(plan.ag_output_shape, plan.dtype),
    ):
        T.func_attr({"tl.dcu_ipc.pe_count": p})
        with T.Kernel(plan.total_blocks, threads=THREADS) as block_idx:
            tx = T.get_thread_binding()
            rank = T.get_rank()
            T.assume(0 <= rank)
            T.assume(rank < p)
            remote_ready0 = T.remote_view(ready, 0)
            remote_ready1 = T.remote_view(ready, 1)
            remote_ready2 = T.remote_view(ready, 2)
            remote_ready3 = T.remote_view(ready, 3)
            if p == 8:
                remote_ready4 = T.remote_view(ready, 4)
                remote_ready5 = T.remote_view(ready, 5)
                remote_ready6 = T.remote_view(ready, 6)
                remote_ready7 = T.remote_view(ready, 7)

            if block_idx < plan.compute_blocks:
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), plan.dtype)
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), plan.dtype)
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                if plan.tile_count <= plan.compute_blocks:
                    if block_idx < plan.tile_count:
                        for _once in T.serial(1, annotations={"tl.dcu_persistent_tile": 1}):
                            tile = block_idx
                            source_offset = tile // tiles_per_source
                            source = (rank + source_offset) % p
                            source_tile = tile % tiles_per_source
                            source_tile_row = source_tile // plan.tile_cols
                            tile_row = source * row_tiles_per_source + source_tile_row
                            tile_col = source_tile % plan.tile_cols
                            ready_chunk = (source * plan.chunks_per_source
                                           + source_tile_row * plan.chunks_per_row_tile)
                            if tx == 0:
                                for ready_part in T.unroll(plan.chunks_per_row_tile):
                                    T.wait_eq(ready[ready_chunk + ready_part], 1)
                                T.fence_sys()
                            T.sync_threads()
                            T.clear(c_local)
                            for kk in T.Pipelined(plan.k_physical // BLOCK_K, num_stages=4,
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
                    for tile in T.serial(block_idx, plan.tile_count, plan.compute_blocks,
                                         annotations={"tl.dcu_persistent_tile": 1}):
                        source_offset = tile // tiles_per_source
                        source = (rank + source_offset) % p
                        source_tile = tile % tiles_per_source
                        source_tile_row = source_tile // plan.tile_cols
                        tile_row = source * row_tiles_per_source + source_tile_row
                        tile_col = source_tile % plan.tile_cols
                        ready_chunk = (source * plan.chunks_per_source
                                       + source_tile_row * plan.chunks_per_row_tile)
                        if tx == 0:
                            for ready_part in T.unroll(plan.chunks_per_row_tile):
                                T.wait_eq(ready[ready_chunk + ready_part], 1)
                            T.fence_sys()
                        T.sync_threads()
                        T.clear(c_local)
                        for kk in T.Pipelined(plan.k_physical // BLOCK_K, num_stages=4,
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
                lane = block_idx - plan.compute_blocks
                # ``source_chunk * 65536`` is a flat element address.  The
                # following div/mod are required even when K_p is not a
                # divisor of 65536, and thus keep each fixed 256x256 chunk.
                for chunk in T.serial(lane, plan.total_chunks, plan.comm_blocks):
                    source_offset = chunk // plan.chunks_per_source
                    source_peer = (rank + source_offset) % p
                    source_chunk = chunk % plan.chunks_per_source
                    physical_chunk = source_peer * plan.chunks_per_source + source_chunk
                    remote_source = T.remote_view(A_shard, source_peer)
                    comm_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
                    for linear in T.serial(comm_tx, CHUNK_ELEMENTS, THREADS):
                        offset = source_chunk * CHUNK_ELEMENTS + linear
                        A[source_peer * plan.rows_per_rank_physical + offset // plan.k_physical,
                          offset % plan.k_physical] = remote_source[offset // plan.k_physical,
                                                                      offset % plan.k_physical]
                    T.fence_sys()
                    T.sync_threads()
                    if tx == 0:
                        T.signal_set(ready[physical_chunk], 1)
                    T.sync_threads()
                if tx == 0:
                    T.signal_set(ready[plan.block_done_base + lane], 1)
                    if lane == 0:
                        for done_lane in T.unroll(plan.comm_blocks):
                            T.wait_eq(ready[plan.block_done_base + done_lane], 1)
                        T.signal_set(remote_ready0[plan.peer_done_base + rank], 1, peer=0)
                        T.signal_set(remote_ready1[plan.peer_done_base + rank], 1, peer=1)
                        T.signal_set(remote_ready2[plan.peer_done_base + rank], 1, peer=2)
                        T.signal_set(remote_ready3[plan.peer_done_base + rank], 1, peer=3)
                        if p == 8:
                            T.signal_set(remote_ready4[plan.peer_done_base + rank], 1, peer=4)
                            T.signal_set(remote_ready5[plan.peer_done_base + rank], 1, peer=5)
                            T.signal_set(remote_ready6[plan.peer_done_base + rank], 1, peer=6)
                            T.signal_set(remote_ready7[plan.peer_done_base + rank], 1, peer=7)
                        for peer in T.unroll(p):
                            T.wait_eq(ready[plan.peer_done_base + peer], 1)
                        T.fence_sys()
                T.sync_threads()

    return fused_allgather_gemm


def allgather_gemm_scheduled(m: int, n: int, k: int, **kwargs):
    """Convenience constructor for an ``ag`` plan and its DSL function."""
    return allgather_gemm(make_plan("ag", m, n, k, **kwargs))
