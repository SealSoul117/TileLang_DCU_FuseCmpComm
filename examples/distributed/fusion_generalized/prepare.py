"""GPU-only packing, weight preparation, and output crop DSL kernels."""

import tilelang.language as T

from .contract import FusionPlan, ceildiv


PREP_THREADS = 256


def pack_ar_rs_a(plan: FusionPlan):
    """Pack logical [M,K] A into P equal zero-padded physical row slabs."""
    if plan.kind not in ("ar", "rs"):
        raise ValueError("pack_ar_rs_a requires ar or rs")

    @T.prim_func
    def kernel(A_raw: T.Tensor((plan.m, plan.k), plan.dtype),
               A_packed: T.Tensor(plan.ar_rs_packed_a_shape, plan.dtype)):
        with T.Kernel(ceildiv(plan.m_physical * plan.k_physical, PREP_THREADS), threads=PREP_THREADS) as bx:
            tx = T.get_thread_binding()
            linear = bx * PREP_THREADS + tx
            if linear < plan.m_physical * plan.k_physical:
                row = linear // plan.k_physical
                col = linear % plan.k_physical
                owner = row // plan.rows_per_rank_physical
                local = row % plan.rows_per_rank_physical
                logical = plan.m * owner // plan.world_size + local
                if local < plan.m * (owner + 1) // plan.world_size - plan.m * owner // plan.world_size and col < plan.k:
                    A_packed[row, col] = A_raw[logical, col]
                else:
                    A_packed[row, col] = T.cast(0, plan.dtype)
    return kernel


def pack_ag_a_shard(plan: FusionPlan):
    """Pack this rank's floor-partitioned raw A rows into its physical slab."""
    if plan.kind != "ag":
        raise ValueError("pack_ag_a_shard requires ag")
    raw_rows = max(1, ceildiv(plan.m, plan.world_size))

    @T.prim_func
    def kernel(A_raw: T.Tensor((raw_rows, plan.k), plan.dtype), rank: T.int32,
               A_shard: T.Tensor(plan.ag_shard_shape, plan.dtype)):
        with T.Kernel(ceildiv(plan.rows_per_rank_physical * plan.k_physical, PREP_THREADS), threads=PREP_THREADS) as bx:
            tx = T.get_thread_binding()
            T.assume(0 <= rank)
            T.assume(rank < plan.world_size)
            linear = bx * PREP_THREADS + tx
            if linear < plan.rows_per_rank_physical * plan.k_physical:
                row = linear // plan.k_physical
                col = linear % plan.k_physical
                valid_rows = plan.m * (rank + 1) // plan.world_size - plan.m * rank // plan.world_size
                if row < valid_rows and col < plan.k:
                    A_shard[row, col] = A_raw[row, col]
                else:
                    A_shard[row, col] = T.cast(0, plan.dtype)
    return kernel


def pack_weight(plan: FusionPlan):
    """GPU transpose+pad [N,K] weights, or pad already-[K,N] weights."""
    raw_shape = (plan.n, plan.k) if plan.weight_layout == "nk" else (plan.k, plan.n)

    @T.prim_func
    def kernel(weight_raw: T.Tensor(raw_shape, plan.dtype),
               B_packed: T.Tensor(plan.packed_weight_shape, plan.dtype)):
        with T.Kernel(ceildiv(plan.k_physical * plan.n_physical, PREP_THREADS), threads=PREP_THREADS) as bx:
            tx = T.get_thread_binding()
            linear = bx * PREP_THREADS + tx
            if linear < plan.k_physical * plan.n_physical:
                row = linear // plan.n_physical
                col = linear % plan.n_physical
                if row < plan.k and col < plan.n:
                    if plan.weight_layout == "nk":
                        B_packed[row, col] = weight_raw[col, row]
                    else:
                        B_packed[row, col] = weight_raw[row, col]
                else:
                    B_packed[row, col] = T.cast(0, plan.dtype)
    return kernel


def crop_output(plan: FusionPlan):
    """Inverse physical packing into a compact logical AR/AG or RS result."""
    rows = max(1, ceildiv(plan.m, plan.world_size)) if plan.kind == "rs" else plan.m
    physical_shape = plan.rs_output_shape if plan.kind == "rs" else plan.ar_output_shape

    @T.prim_func
    def kernel(physical: T.Tensor(physical_shape, plan.dtype), rank: T.int32,
               output: T.Tensor((rows, plan.n), plan.dtype)):
        with T.Kernel(ceildiv(rows * plan.n, PREP_THREADS), threads=PREP_THREADS) as bx:
            tx = T.get_thread_binding()
            T.assume(0 <= rank)
            T.assume(rank < plan.world_size)
            linear = bx * PREP_THREADS + tx
            if linear < rows * plan.n:
                row = linear // plan.n
                col = linear % plan.n
                if plan.kind == "rs":
                    valid_rows = plan.m * (rank + 1) // plan.world_size - plan.m * rank // plan.world_size
                    if row < valid_rows:
                        output[row, col] = physical[row, col]
                    else:
                        output[row, col] = T.cast(0, plan.dtype)
                else:
                    owner = ((row + 1) * plan.world_size - 1) // plan.m
                    physical_row = owner * plan.rows_per_rank_physical + row - plan.m * owner // plan.world_size
                    output[row, col] = physical[physical_row, col]
    return kernel
