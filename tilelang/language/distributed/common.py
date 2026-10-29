"""DCU distributed-language primitives that do not perform device lowering.

The functions in this module build ordinary TIR nodes.  They deliberately do
not query a Python-process rank: rank and world size are runtime expressions
bound by a future distributed ABI.
"""

from __future__ import annotations

import re
from typing import Any

from tvm import tir
from tvm.script.ir_builder.tir import attr as _tir_attr

from tilelang.language.utils import region


_REMOTE_VIEW_TOKEN = "tl.remote_view_token"


def _is_rank_dtype(dtype: Any) -> bool:
    """Rank/peer is a scalar integer, never bool/uint1/vector."""
    # `tvm.DataType` has changed its Python convenience predicates across the
    # supported TVM revisions.  Its canonical spelling is stable, and is also
    # what TIR prints, so use it rather than `is_int()` / `lanes` methods that
    # are absent in the matching M1 TVM checkout.
    match = re.fullmatch(r"u?int([0-9]+)", str(dtype))
    return match is not None and int(match.group(1)) >= 8


def _buffer_type_for_decl(buffer: tir.Buffer) -> str:
    """Return a cross-version ``decl_buffer`` buffer type spelling.

    Older TIR exposes ``Buffer.buffer_type`` as an internal enum integer,
    whereas newer Python constructors accept a string.  M1 creates ordinary
    default buffers, so omitting that enum is the faithful portable form.
    """
    value = buffer.buffer_type
    return value if isinstance(value, str) else ""


def _static_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, tir.IntImm):
        return int(value.value)
    return None


def _validate_peer_type(peer: Any, operation: str) -> None:
    if _static_integer(peer) is not None:
        return
    if isinstance(peer, tir.PrimExpr) and _is_rank_dtype(peer.dtype):
        return
    raise TypeError(f"{operation} peer must be a scalar integer rank expression, not bool/uint1/vector")


def remote_view_token(provenance: tir.PrimExpr, peer: Any, base_offset: tir.PrimExpr, *metadata: Any):
    """Parser/printing hook for the serializable remote-view marker.

    It is deliberately an implementation detail of :func:`remote_view`; it is
    public only so a printed, pre-normalisation PrimFunc can be parsed back.
    """
    return tir.call_intrin(
        base_offset.dtype, tir.op.Op.get(_REMOTE_VIEW_TOKEN), provenance, peer, base_offset, *metadata
    )


def remote_address(provenance: tir.PrimExpr, peer: Any):
    """Parser/printing hook for M1's normalized remote-address binding."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.remote_address"), provenance, peer)


def get_rank() -> tir.PrimExpr:
    """Return the rank from the runtime communication context."""
    return tir.call_intrin("int32", tir.op.Op.get("tl.get_rank"))


def get_num_ranks() -> tir.PrimExpr:
    """Return the world size from the runtime communication context."""
    return tir.call_intrin("int32", tir.op.Op.get("tl.get_num_ranks"))


def sm_specialize_scope(*, auto_ws: bool = False):
    """Create a lexical persistent-kernel specialization scope.

    HIP has no implementation of CUDA automatic warp specialization in M1.
    Keeping ``auto_ws=False`` as an explicit attribute makes the scope visible
    to later verification/passes without claiming that it allocates an SM.
    """
    if auto_ws:
        raise NotImplementedError(
            "sm_specialize_scope(auto_ws=True) requires CUDA automatic warp "
            "specialization and is not supported by the HIP M1 backend"
        )
    return _tir_attr(0, "tilelang.sm_specialize", 0)


def _zero_indices(buffer: tir.Buffer) -> list[tir.PrimExpr]:
    return [tir.const(0, axis.dtype) for axis in buffer.shape]


def _full_region(buffer: tir.Buffer) -> tir.PrimExpr:
    """A serializable full-buffer region used as remote-view provenance."""
    return region(tir.BufferLoad(buffer, _zero_indices(buffer)), "rw", *buffer.shape)


def _offset_dtype(buffer: tir.Buffer) -> str:
    offset = buffer.elem_offset
    return offset.dtype if offset is not None else "int32"


def remote_view(buffer: tir.Buffer, peer: Any, *, remote: bool = True) -> tir.Buffer:
    """Return a typed global alias of ``buffer`` for a peer allocation.

    The returned object is a real :class:`tvm.tir.Buffer`, so ``T.copy`` and
    epilogues can consume it.  It retains dtype, shape, strides, element
    offset, alignment and offset factor.  For a remote peer its element offset
    carries a registered ``tl.remote_view_token``.  The token contains the
    base buffer's explicit ``tl.region`` provenance and peer expression; it is
    part of serializable TIR, not a Python side table.  ``normalize_remote_views``
    turns that token into a distinct bound pointer before lowering/execution.

    ``remote=False`` (and the conventional ``peer=-1``) deliberately creates a
    local alias with no token.  It is therefore safe to use in ordinary
    single-device TIR.
    """
    if not isinstance(buffer, tir.Buffer):
        raise TypeError("remote_view expects a tir.Buffer")
    if buffer.scope() != "global":
        raise ValueError("remote_view supports global buffers only; no synthetic storage scope is created")
    _validate_peer_type(peer, "remote_view")

    is_local = not remote or _static_integer(peer) == -1
    base_offset = buffer.elem_offset if buffer.elem_offset is not None else tir.const(0, "int32")
    if is_local:
        alias_offset = base_offset
        prefix = "local_view"
    else:
        # The token is algebraically added only as an unnormalized marker.  The
        # normalizer removes it and binds a unique data Var to remote_address;
        # no generated kernel may silently execute this representation locally.
        token = remote_view_token(_full_region(buffer), peer, base_offset, *buffer.shape, *buffer.strides)
        alias_offset = base_offset + token
        prefix = "remote_view"

    return tir.decl_buffer(
        buffer.shape,
        buffer.dtype,
        name=f"{prefix}_{buffer.name}",
        data=buffer.data,
        strides=buffer.strides,
        elem_offset=alias_offset,
        scope="global",
        data_alignment=buffer.data_alignment,
        offset_factor=buffer.offset_factor,
        buffer_type=_buffer_type_for_decl(buffer),
        axis_separators=buffer.axis_separators,
    )
