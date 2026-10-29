"""Construction-only checks for generalized GPU pack/crop DSLs."""

import pytest

from examples.distributed.fusion_generalized.contract import make_plan
from examples.distributed.fusion_generalized.prepare import (
    crop_output,
    pack_ag_a_shard,
    pack_ar_rs_a,
    pack_weight,
)


@pytest.mark.parametrize("kind", ["ag", "ar", "rs"])
@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_prepare_and_crop_dsl_construct_for_unaligned_rectangles(kind, world_size, dtype):
    plan = make_plan(kind, 1031, 521, 259, world_size=world_size, dtype=dtype)
    assert pack_weight(plan) is not None
    assert crop_output(plan) is not None
    if kind == "ag":
        assert pack_ag_a_shard(plan) is not None
    else:
        assert pack_ar_rs_a(plan) is not None


def test_ag_empty_rank_raw_input_is_still_allocatable_and_rs_crop_is_guarded():
    ag = make_plan("ag", 1, 3, 5, world_size=8)
    rs = make_plan("rs", 1, 3, 5, world_size=8)
    assert max(1, (ag.m + ag.world_size - 1) // ag.world_size) == 1
    assert crop_output(rs) is not None
