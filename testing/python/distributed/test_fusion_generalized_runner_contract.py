"""Pure runner input/oracle contracts; no GPU launch or IPC allocation."""

import numpy as np
import pytest

from examples.distributed.fusion_generalized.contract import make_plan
from examples.distributed.fusion_generalized.run import _dyadic_inputs, _dyadic_oracle, encode, prepare_inputs


def _round(values, dtype):
    import torch
    tdtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    return torch.from_numpy(values.astype(np.float32)).to(tdtype).float().numpy()


def test_ag_prepare_requires_real_floor_shard_rank_and_rejects_raw_bits():
    plan = make_plan("ag", 3, 5, 7, world_size=8)
    a = np.zeros((1, 7), dtype=np.float32)
    w = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises(ValueError):
        prepare_inputs(a, w, plan)
    with pytest.raises(TypeError):
        prepare_inputs(a.astype(np.uint16), w, plan, rank=7)
    bits_a, bits_w = prepare_inputs(a, w, plan, rank=7)
    assert bits_a.dtype == bits_w.dtype == np.uint16


def test_periodic_oracle_has_compact_logical_output_shapes_for_all_kinds():
    for kind in ("ag", "ar", "rs"):
        plan = make_plan(kind, 5, 7, 9, world_size=4, dtype="float16")
        got = _dyadic_oracle(1, 0, plan)
        rows = plan.logical_rows_for_rank(1)[1] - plan.logical_rows_for_rank(1)[0]
        assert got.shape == ((max(1, (plan.m + 3) // 4), plan.n) if kind == "rs" else (plan.m, plan.n))
        assert got.dtype == np.uint16
        if kind == "rs": assert rows <= got.shape[0]


@pytest.mark.parametrize("kind", ["ag", "ar", "rs"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_periodic_oracle_matches_small_real_dyadic_numpy_reference(kind, dtype):
    plan = make_plan(kind, 5, 7, 9, world_size=4, dtype=dtype)
    rank, epoch = 1, 1
    if kind == "ag":
        shards = [_dyadic_inputs(peer, epoch, plan)[0] for peer in range(plan.world_size)]
        a = np.concatenate(shards, axis=0)[:plan.m]
        _, weight = _dyadic_inputs(rank, epoch, plan)
        expected = encode(a @ weight.T, dtype)
    else:
        reduced = np.zeros((plan.m, plan.n), dtype=np.float32)
        for peer in range(plan.world_size):
            a, weight = _dyadic_inputs(peer, epoch, plan)
            reduced += _round(a @ weight.T, dtype)
        expected = encode(reduced, dtype)
        if kind == "rs":
            begin, end = plan.logical_rows_for_rank(rank)
            padded = np.zeros((max(1, (plan.m + plan.world_size - 1) // plan.world_size), plan.n), dtype=np.uint16)
            padded[:end - begin] = expected[begin:end]
            expected = padded
    assert np.array_equal(_dyadic_oracle(rank, epoch, plan), expected)
