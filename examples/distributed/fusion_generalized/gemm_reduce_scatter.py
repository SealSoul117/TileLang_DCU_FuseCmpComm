"""Generalized physical-layout GEMM + ReduceScatter TileLang generator."""

import tilelang.language as T

from .contract import BLOCK_K, THREADS, TILE_M, TILE_N, FusionPlan, make_plan


def gemm_reduce_scatter(plan: FusionPlan):
    """Generate a 4/8-PE RS DSL function over the plan's physical tensors."""
    if plan.kind != "rs":
        raise ValueError("gemm_reduce_scatter requires an rs FusionPlan")
    p = plan.world_size
    rows_per_rank_tiles = plan.rows_per_rank_physical // TILE_M

    @T.prim_func
    def fused_gemm_reduce_scatter(
        A: T.Tensor(plan.ar_rs_packed_a_shape, plan.dtype),
        B: T.Tensor(plan.packed_weight_shape, plan.dtype),
        partial: T.Tensor(plan.partial_shape, plan.dtype),
        ready: T.Tensor((plan.flag_words,), "uint32"),
        out: T.Tensor(plan.rs_output_shape, plan.dtype),
    ):
        T.func_attr({"tl.dcu_ipc.pe_count": p})
        with T.Kernel(plan.total_blocks, threads=THREADS) as block_idx:
            tx = T.get_thread_binding()
            rank = T.get_rank()
            T.assume(0 <= rank)
            T.assume(rank < p)
            remote_partial0 = T.remote_view(partial, 0)
            remote_partial1 = T.remote_view(partial, 1)
            remote_partial2 = T.remote_view(partial, 2)
            remote_partial3 = T.remote_view(partial, 3)
            remote_ready0 = T.remote_view(ready, 0)
            remote_ready1 = T.remote_view(ready, 1)
            remote_ready2 = T.remote_view(ready, 2)
            remote_ready3 = T.remote_view(ready, 3)
            p0 = T.alloc_local((1,), plan.dtype)
            p1 = T.alloc_local((1,), plan.dtype)
            p2 = T.alloc_local((1,), plan.dtype)
            p3 = T.alloc_local((1,), plan.dtype)
            if p == 8:
                remote_partial4 = T.remote_view(partial, 4)
                remote_partial5 = T.remote_view(partial, 5)
                remote_partial6 = T.remote_view(partial, 6)
                remote_partial7 = T.remote_view(partial, 7)
                remote_ready4 = T.remote_view(ready, 4)
                remote_ready5 = T.remote_view(ready, 5)
                remote_ready6 = T.remote_view(ready, 6)
                remote_ready7 = T.remote_view(ready, 7)
                p4 = T.alloc_local((1,), plan.dtype)
                p5 = T.alloc_local((1,), plan.dtype)
                p6 = T.alloc_local((1,), plan.dtype)
                p7 = T.alloc_local((1,), plan.dtype)

            if block_idx < plan.compute_blocks:
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), plan.dtype)
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), plan.dtype)
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                if plan.tile_count <= plan.compute_blocks:
                    if block_idx < plan.tile_count:
                        for _once in T.serial(1, annotations={"tl.dcu_persistent_tile": 1}):
                            job = block_idx
                            tile = (job % p) * plan.tiles_per_rank + job // p
                            tile_row = tile // plan.tile_cols
                            tile_col = tile % plan.tile_cols
                            owner = tile // plan.tiles_per_rank
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
                            T.copy(c_local, partial[tile, 0, 0])
                            T.fence_sys()
                            T.sync_threads()
                            if tx == 0:
                                if owner == 0:
                                    T.signal_set(remote_ready0[tile * p + rank], 1, peer=0)
                                if owner == 1:
                                    T.signal_set(remote_ready1[tile * p + rank], 1, peer=1)
                                if owner == 2:
                                    T.signal_set(remote_ready2[tile * p + rank], 1, peer=2)
                                if owner == 3:
                                    T.signal_set(remote_ready3[tile * p + rank], 1, peer=3)
                                if p == 8:
                                    if owner == 4:
                                        T.signal_set(remote_ready4[tile * p + rank], 1, peer=4)
                                    if owner == 5:
                                        T.signal_set(remote_ready5[tile * p + rank], 1, peer=5)
                                    if owner == 6:
                                        T.signal_set(remote_ready6[tile * p + rank], 1, peer=6)
                                    if owner == 7:
                                        T.signal_set(remote_ready7[tile * p + rank], 1, peer=7)
                            T.sync_threads()
                else:
                    for job in T.serial(block_idx, plan.tile_count, plan.compute_blocks,
                                        annotations={"tl.dcu_persistent_tile": 1}):
                        tile = (job % p) * plan.tiles_per_rank + job // p
                        tile_row = tile // plan.tile_cols
                        tile_col = tile % plan.tile_cols
                        owner = tile // plan.tiles_per_rank
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
                        T.copy(c_local, partial[tile, 0, 0])
                        T.fence_sys()
                        T.sync_threads()
                        if tx == 0:
                            if owner == 0:
                                T.signal_set(remote_ready0[tile * p + rank], 1, peer=0)
                            if owner == 1:
                                T.signal_set(remote_ready1[tile * p + rank], 1, peer=1)
                            if owner == 2:
                                T.signal_set(remote_ready2[tile * p + rank], 1, peer=2)
                            if owner == 3:
                                T.signal_set(remote_ready3[tile * p + rank], 1, peer=3)
                            if p == 8:
                                if owner == 4:
                                    T.signal_set(remote_ready4[tile * p + rank], 1, peer=4)
                                if owner == 5:
                                    T.signal_set(remote_ready5[tile * p + rank], 1, peer=5)
                                if owner == 6:
                                    T.signal_set(remote_ready6[tile * p + rank], 1, peer=6)
                                if owner == 7:
                                    T.signal_set(remote_ready7[tile * p + rank], 1, peer=7)
                        T.sync_threads()
            else:
                lane = block_idx - plan.compute_blocks
                for owned in T.serial(lane, plan.tiles_per_rank, plan.comm_blocks):
                    tile = rank * plan.tiles_per_rank + owned
                    tile_row = tile // plan.tile_cols
                    tile_col = tile % plan.tile_cols
                    if tx == 0:
                        for peer in T.unroll(p):
                            T.wait_eq(ready[tile * p + peer], 1)
                        T.fence_sys()
                    T.sync_threads()
                    comm_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
                    for linear in T.serial(comm_tx, TILE_M * TILE_N, THREADS):
                        row = linear // TILE_N
                        col = linear % TILE_N
                        p0[0] = remote_partial0[tile, row, col]
                        p1[0] = remote_partial1[tile, row, col]
                        p2[0] = remote_partial2[tile, row, col]
                        p3[0] = remote_partial3[tile, row, col]
                        if p == 4:
                            out[(tile_row % rows_per_rank_tiles) * TILE_M + row,
                                tile_col * TILE_N + col] = T.cast(
                                    T.cast(p0[0], "float32") + T.cast(p1[0], "float32")
                                    + T.cast(p2[0], "float32") + T.cast(p3[0], "float32"), plan.dtype)
                        else:
                            p4[0] = remote_partial4[tile, row, col]
                            p5[0] = remote_partial5[tile, row, col]
                            p6[0] = remote_partial6[tile, row, col]
                            p7[0] = remote_partial7[tile, row, col]
                            out[(tile_row % rows_per_rank_tiles) * TILE_M + row,
                                tile_col * TILE_N + col] = T.cast(
                                    T.cast(p0[0], "float32") + T.cast(p1[0], "float32")
                                    + T.cast(p2[0], "float32") + T.cast(p3[0], "float32")
                                    + T.cast(p4[0], "float32") + T.cast(p5[0], "float32")
                                    + T.cast(p6[0], "float32") + T.cast(p7[0], "float32"), plan.dtype)
                    T.fence_sys()
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

    return fused_gemm_reduce_scatter


def gemm_reduce_scatter_scheduled(m: int, n: int, k: int, **kwargs):
    """Convenience constructor for an ``rs`` plan and its DSL function."""
    return gemm_reduce_scatter(make_plan("rs", m, n, k, **kwargs))
