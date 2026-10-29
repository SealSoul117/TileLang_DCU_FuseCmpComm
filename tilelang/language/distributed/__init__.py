"""M1 distributed TileLang DSL surface."""

from .common import get_num_ranks, get_rank, remote_address, remote_view, remote_view_token, sm_specialize_scope
from .comm import get_block, ld, put_block, st
from .sync import WaitScope, WaitSemantics, fence_sys, signal_set, wait, wait_eq, wait_ge
from .multimem import (
    MultimemReduceOp,
    multimem_ld_reduce,
    multimem_red,
    multimem_signal,
    multimem_signal_add,
    multimem_st,
    multimem_tma_store,
)
from .ir import (
    DistributedIRError,
    assert_remote_views_normalized,
    normalize_remote_views,
    verify_distributed_ir,
)

__all__ = [
    "DistributedIRError", "MultimemReduceOp", "WaitScope", "WaitSemantics", "assert_remote_views_normalized",
    "fence_sys", "get_block", "get_num_ranks", "get_rank", "ld", "normalize_remote_views",
    "multimem_ld_reduce", "multimem_red", "multimem_signal", "multimem_signal_add", "multimem_st", "multimem_tma_store",
    "put_block", "remote_address", "remote_view", "remote_view_token", "signal_set", "sm_specialize_scope", "st", "verify_distributed_ir",
    "wait", "wait_eq", "wait_ge",
]
