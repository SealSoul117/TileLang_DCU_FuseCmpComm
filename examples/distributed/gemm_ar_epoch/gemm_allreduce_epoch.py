"""Epoch-tagged GEMM + AllReduce: no per-call flag reset or host barrier.

Derived from fusion_generalized/gemm_allreduce.py; the GEMM/data movement is
unchanged. Signals are single-writer monotone uint32 epochs, initialized once.
All ranks must invoke a workspace in the same order on an ordered stream.
No wraparound, concurrent workspace use, or fixed-epoch graph replay is valid.
The final peer_done handshake is a payload-consumption ACK, NOT optional.
"""

import tilelang.language as T

from ..fusion_generalized.contract import BLOCK_K, THREADS, TILE_M, TILE_N, FusionPlan, make_plan


def gemm_allreduce_epoch(plan: FusionPlan):
    """Generate one 4/8-PE epoch AR kernel for padded physical tensors.

    ``A`` and ``B`` are [M_p, K_p] and [K_p, N_p].  ``partial`` is local
    tile-major storage.  The caller owns logical packing/cropping and IPC
    binding; this function intentionally only emits the DSL frontend.
    """
    if plan.kind != "ar":
        raise ValueError("gemm_allreduce requires an ar FusionPlan")
    p = plan.world_size

    @T.prim_func
    def fused_gemm_allreduce_epoch(
        A: T.Tensor(plan.ar_rs_packed_a_shape, plan.dtype),
        B: T.Tensor(plan.packed_weight_shape, plan.dtype),
        partial: T.Tensor(plan.partial_shape, plan.dtype),
        ready: T.Tensor((plan.flag_words,), "uint32"),
        out: T.Tensor(plan.ar_output_shape, plan.dtype),
        epoch: T.uint32,
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
            remote_out0 = T.remote_view(out, rank)
            remote_out1 = T.remote_view(out, (rank + 1) % p)
            remote_out2 = T.remote_view(out, (rank + 2) % p)
            remote_out3 = T.remote_view(out, (rank + 3) % p)
            p0 = T.alloc_local((1,), plan.dtype)
            p1 = T.alloc_local((1,), plan.dtype)
            p2 = T.alloc_local((1,), plan.dtype)
            p3 = T.alloc_local((1,), plan.dtype)
            reduced = T.alloc_local((1,), plan.dtype)
            if p == 8:
                remote_partial4 = T.remote_view(partial, 4)
                remote_partial5 = T.remote_view(partial, 5)
                remote_partial6 = T.remote_view(partial, 6)
                remote_partial7 = T.remote_view(partial, 7)
                remote_ready4 = T.remote_view(ready, 4)
                remote_ready5 = T.remote_view(ready, 5)
                remote_ready6 = T.remote_view(ready, 6)
                remote_ready7 = T.remote_view(ready, 7)
                remote_out4 = T.remote_view(out, (rank + 4) % p)
                remote_out5 = T.remote_view(out, (rank + 5) % p)
                remote_out6 = T.remote_view(out, (rank + 6) % p)
                remote_out7 = T.remote_view(out, (rank + 7) % p)
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
                            tile = block_idx
                            tile_row = tile // plan.tile_cols
                            tile_col = tile % plan.tile_cols
                            owner = tile % p
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
                                    T.signal_set(remote_ready0[tile * p + rank], epoch, peer=0)
                                if owner == 1:
                                    T.signal_set(remote_ready1[tile * p + rank], epoch, peer=1)
                                if owner == 2:
                                    T.signal_set(remote_ready2[tile * p + rank], epoch, peer=2)
                                if owner == 3:
                                    T.signal_set(remote_ready3[tile * p + rank], epoch, peer=3)
                                if p == 8:
                                    if owner == 4:
                                        T.signal_set(remote_ready4[tile * p + rank], epoch, peer=4)
                                    if owner == 5:
                                        T.signal_set(remote_ready5[tile * p + rank], epoch, peer=5)
                                    if owner == 6:
                                        T.signal_set(remote_ready6[tile * p + rank], epoch, peer=6)
                                    if owner == 7:
                                        T.signal_set(remote_ready7[tile * p + rank], epoch, peer=7)
                            T.sync_threads()
                else:
                    for tile in T.serial(block_idx, plan.tile_count, plan.compute_blocks,
                                         annotations={"tl.dcu_persistent_tile": 1}):
                        tile_row = tile // plan.tile_cols
                        tile_col = tile % plan.tile_cols
                        owner = tile % p
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
                                T.signal_set(remote_ready0[tile * p + rank], epoch, peer=0)
                            if owner == 1:
                                T.signal_set(remote_ready1[tile * p + rank], epoch, peer=1)
                            if owner == 2:
                                T.signal_set(remote_ready2[tile * p + rank], epoch, peer=2)
                            if owner == 3:
                                T.signal_set(remote_ready3[tile * p + rank], epoch, peer=3)
                            if p == 8:
                                if owner == 4:
                                    T.signal_set(remote_ready4[tile * p + rank], epoch, peer=4)
                                if owner == 5:
                                    T.signal_set(remote_ready5[tile * p + rank], epoch, peer=5)
                                if owner == 6:
                                    T.signal_set(remote_ready6[tile * p + rank], epoch, peer=6)
                                if owner == 7:
                                    T.signal_set(remote_ready7[tile * p + rank], epoch, peer=7)
                        T.sync_threads()
            else:
                lane = block_idx - plan.compute_blocks
                for owned in T.serial(lane, plan.tiles_per_rank, plan.comm_blocks):
                    tile = rank + owned * p
                    tile_row = tile // plan.tile_cols
                    tile_col = tile % plan.tile_cols
                    if tx == 0:
                        for peer in T.unroll(p):
                            T.wait_ge(ready[tile * p + peer], epoch)
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
                            reduced[0] = T.cast(T.cast(p0[0], "float32") + T.cast(p1[0], "float32")
                                                + T.cast(p2[0], "float32") + T.cast(p3[0], "float32"), plan.dtype)
                        else:
                            p4[0] = remote_partial4[tile, row, col]
                            p5[0] = remote_partial5[tile, row, col]
                            p6[0] = remote_partial6[tile, row, col]
                            p7[0] = remote_partial7[tile, row, col]
                            reduced[0] = T.cast(T.cast(p0[0], "float32") + T.cast(p1[0], "float32")
                                                + T.cast(p2[0], "float32") + T.cast(p3[0], "float32")
                                                + T.cast(p4[0], "float32") + T.cast(p5[0], "float32")
                                                + T.cast(p6[0], "float32") + T.cast(p7[0], "float32"), plan.dtype)
                        remote_out0[tile_row * TILE_M + row, tile_col * TILE_N + col] = reduced[0]
                        remote_out1[tile_row * TILE_M + row, tile_col * TILE_N + col] = reduced[0]
                        remote_out2[tile_row * TILE_M + row, tile_col * TILE_N + col] = reduced[0]
                        remote_out3[tile_row * TILE_M + row, tile_col * TILE_N + col] = reduced[0]
                        if p == 8:
                            remote_out4[tile_row * TILE_M + row, tile_col * TILE_N + col] = reduced[0]
                            remote_out5[tile_row * TILE_M + row, tile_col * TILE_N + col] = reduced[0]
                            remote_out6[tile_row * TILE_M + row, tile_col * TILE_N + col] = reduced[0]
                            remote_out7[tile_row * TILE_M + row, tile_col * TILE_N + col] = reduced[0]
                    T.fence_sys()
                    T.sync_threads()
                if tx == 0:
                    T.signal_set(ready[plan.block_done_base + lane], epoch)
                    if lane == 0:
                        for done_lane in T.unroll(plan.comm_blocks):
                            T.wait_ge(ready[plan.block_done_base + done_lane], epoch)
                        T.signal_set(remote_ready0[plan.peer_done_base + rank], epoch, peer=0)
                        T.signal_set(remote_ready1[plan.peer_done_base + rank], epoch, peer=1)
                        T.signal_set(remote_ready2[plan.peer_done_base + rank], epoch, peer=2)
                        T.signal_set(remote_ready3[plan.peer_done_base + rank], epoch, peer=3)
                        if p == 8:
                            T.signal_set(remote_ready4[plan.peer_done_base + rank], epoch, peer=4)
                            T.signal_set(remote_ready5[plan.peer_done_base + rank], epoch, peer=5)
                            T.signal_set(remote_ready6[plan.peer_done_base + rank], epoch, peer=6)
                            T.signal_set(remote_ready7[plan.peer_done_base + rank], epoch, peer=7)
                        for peer in T.unroll(p):
                            T.wait_ge(ready[plan.peer_done_base + peer], epoch)
                        T.fence_sys()
                T.sync_threads()

    return fused_gemm_allreduce_epoch


def gemm_allreduce_epoch_scheduled(m: int, n: int, k: int, **kwargs):
    """Convenience constructor for an ``ar`` plan and its DSL function."""
    return gemm_allreduce_epoch(make_plan("ar", m, n, k, **kwargs))

