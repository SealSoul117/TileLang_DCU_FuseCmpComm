"""Explicit CUDA multimem capability rejection for the HIP M1 surface."""

from __future__ import annotations

from enum import Enum
from typing import Any


class MultimemReduceOp(Enum):
    ADD = 0
    MIN = 1
    MAX = 2
    AND = 3
    OR = 4
    XOR = 5


def _unsupported(*_: Any, **__: Any):
    raise NotImplementedError(
        "CUDA PTX multimem is not supported by the HIP M1 backend; "
        "there is no fallback that silently changes its collective semantics"
    )


multimem_ld_reduce = _unsupported
multimem_st = _unsupported
multimem_red = _unsupported
multimem_tma_store = _unsupported
multimem_signal = _unsupported
multimem_signal_add = _unsupported
