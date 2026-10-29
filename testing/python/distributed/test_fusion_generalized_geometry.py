"""Pure-Python geometry checks for generalized distributed fusion DSLs."""

import pytest

from examples.distributed.fusion_generalized.contract import (
    CHUNK_ELEMENTS,
    TILE_M,
    TILE_N,
    make_plan,
)


@pytest.mark.parametrize("kind", ["ag", "ar", "rs"])
@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_arbitrary_rectangle_uses_the_required_physical_geometry(kind, world_size, dtype):
    plan = make_plan(kind, 777, 513, 513, world_size=world_size, dtype=dtype)
    per_rank = ((777 + world_size - 1) // world_size + 255) // 256 * 256
    assert plan.rows_per_rank_physical == per_rank
    assert plan.m_physical == world_size * per_rank
    assert plan.n_physical == 768
    assert plan.k_physical == (768 if kind == "ag" else 576)
    assert plan.tile_count == plan.m_physical // TILE_M * (plan.n_physical // TILE_N)
    assert plan.tile_count % world_size == 0
    assert plan.tiles_per_rank == plan.tile_count // world_size
    assert plan.compute_blocks == 48


@pytest.mark.parametrize("world_size", [4, 8])
def test_floor_row_pack_and_inverse_are_lossless_for_ar_and_rs(world_size):
    plan = make_plan("rs", 1027, 259, 65, world_size=world_size)
    seen = []
    for rank in range(world_size):
        begin, end = plan.logical_rows_for_rank(rank)
        assert (begin, end) == (1027 * rank // world_size, 1027 * (rank + 1) // world_size)
        for row in range(begin, end):
            owner, packed = plan.logical_row_to_physical(row)
            assert owner == rank
            assert plan.physical_row_to_logical(packed) == row
            seen.append(packed)
        pad_begin = rank * plan.rows_per_rank_physical + end - begin
        assert all(plan.physical_row_to_logical(row) is None
                   for row in range(pad_begin, (rank + 1) * plan.rows_per_rank_physical))
    assert len(seen) == 1027
    assert len(set(seen)) == 1027


def test_weight_contract_exposes_nk_transpose_to_kn_gemm_shape():
    transposed = make_plan("ar", 33, 513, 65, weight_layout="nk")
    prepacked = make_plan("ar", 33, 513, 65, weight_layout="kn")
    assert transposed.logical_weight_shape == (513, 65)
    assert transposed.needs_weight_transpose
    assert prepacked.logical_weight_shape == (65, 513)
    assert not prepacked.needs_weight_transpose
    assert transposed.packed_weight_shape == prepacked.packed_weight_shape == (128, 768)


@pytest.mark.parametrize("kind", ["ag", "ar", "rs"])
@pytest.mark.parametrize("k", [1, 7, 64, 65])
def test_small_k_keeps_the_persistent_pipeline_physical_minimum(kind, k):
    plan = make_plan(kind, 3, 5, k, world_size=8)
    # AG remains 256-aligned for its fixed transfer chunks.  AR/RS pad every
    # tiny logical K through 65 to 128, leaving a legal persistent K mainloop
    # after the four BLOCK_K=16 pipeline stages are established.
    assert plan.k_physical == (256 if kind == "ag" else 128)


@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("k", [257, 513, 769])
def test_ag_fixed_chunks_cover_all_padded_shard_elements_without_k_divisibility(world_size, k):
    plan = make_plan("ag", 519, 257, k, world_size=world_size)
    assert plan.k_physical % TILE_M == 0
    assert CHUNK_ELEMENTS == TILE_M * TILE_N
    assert plan.chunks_per_row_tile == plan.k_physical // TILE_N
    assert plan.chunks_per_source * CHUNK_ELEMENTS == (
        plan.rows_per_rank_physical * plan.k_physical
    )
    # K_p=768 is deliberately not a divisor of 65536.  Flat source chunks
    # still concatenate to exactly one padded shard and the DSL div/mod maps
    # them back into its 2D row-major layout.
    assert (CHUNK_ELEMENTS % plan.k_physical != 0) if plan.k_physical == 768 else True
    offsets = [chunk * CHUNK_ELEMENTS + lane
               for chunk in range(plan.chunks_per_source)
               for lane in (0, CHUNK_ELEMENTS - 1)]
    assert offsets[0] == 0
    assert offsets[-1] == plan.rows_per_rank_physical * plan.k_physical - 1


def test_rs_owner_tiles_are_continuous_physical_row_slabs_and_cover_all_tiles():
    plan = make_plan("rs", 777, 513, 129, world_size=8)
    rows_per_rank_tiles = plan.rows_per_rank_physical // TILE_M
    all_tiles = []
    for rank in range(plan.world_size):
        owned = list(range(rank * plan.tiles_per_rank, (rank + 1) * plan.tiles_per_rank))
        all_tiles.extend(owned)
        assert [tile // plan.tile_cols for tile in owned] == [
            rank * rows_per_rank_tiles + row
            for row in range(rows_per_rank_tiles)
            for _ in range(plan.tile_cols)
        ]
    assert all_tiles == list(range(plan.tile_count))


@pytest.mark.parametrize("bad", [
    {"kind": "bad"}, {"world_size": 2}, {"dtype": "float32"},
    {"weight_layout": "bad"}, {"m": 0}, {"total_blocks": 81}, {"comm_blocks": 0},
])
def test_contract_rejects_unsupported_geometry(bad):
    kwargs = {"kind": "ag", "m": 1, "n": 1, "k": 1}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        make_plan(**kwargs)
