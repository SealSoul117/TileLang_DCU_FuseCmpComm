"""Typed M1 block-copy and scalar communication IR builders."""

from __future__ import annotations

from typing import Any

from tvm import ir, tir

from tilelang.language.utils import region
from .common import _is_rank_dtype, _validate_peer_type


_SCOPE = {"gpu": 1, "sys": 2}
_LD_SEM = {"weak": 0, "volatile": 1, "acquire": 2, "relaxed": 4}
_ST_SEM = {"weak": 0, "volatile": 1, "release": 3, "relaxed": 4}
_SCALAR_DTYPES = {"int32", "uint32", "float32"}


def _is_region(value: Any) -> bool:
    return isinstance(value, tir.Call) and isinstance(value.op, ir.Op) and value.op.name == "tl.tileop.region"


def _load(value: tir.BufferLoad | tir.Buffer, what: str) -> tir.BufferLoad:
    if isinstance(value, tir.Buffer):
        return tir.BufferLoad(value, [tir.const(0, axis.dtype) for axis in value.shape])
    if not isinstance(value, tir.BufferLoad):
        raise TypeError(f"{what} must be a tir.BufferLoad or tir.Buffer")
    return value


def _is_integer_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if not isinstance(value, tir.PrimExpr):
        return False
    return _is_rank_dtype(value.dtype)


def _validate_indices(load: tir.BufferLoad) -> None:
    for axis, (index, extent) in enumerate(zip(load.indices, load.buffer.shape)):
        if not _is_integer_scalar(index):
            raise TypeError("put_block/get_block endpoint indices must be scalar integers")
        static_index, static_extent = _as_static_int(index), _as_static_int(extent)
        if static_index is not None and static_index < 0:
            raise ValueError("put_block/get_block endpoint index must be non-negative")
        if static_index is not None and static_extent is not None and static_index >= static_extent:
            raise ValueError(f"put_block/get_block endpoint index is out of bounds on axis {axis}")


def _as_static_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, tir.IntImm):
        return int(value.value)
    return None


def _require_contiguous_run(load: tir.BufferLoad, elements: Any) -> None:
    """Reject a copy which M1 cannot prove is one typed contiguous interval.

    The block primitives are intentionally not a strided tensor-copy API.  A
    multi-dimensional endpoint may copy within its final row only; silently
    flattening across a row boundary would be wrong for a padded/strided view.
    Dynamic count has no proof-carrying range representation in M1 and is
    rejected.  A future dynamic form must establish ``0 <= index`` and
    ``index + size <= extent`` explicitly before it can be lowered.
    """
    buffer = load.buffer
    if not load.indices:
        raise ValueError("put_block/get_block require a buffer with at least one dimension")
    _validate_indices(load)
    strides = list(buffer.strides)
    if strides and not (isinstance(strides[-1], tir.IntImm) and int(strides[-1].value) == 1):
        raise ValueError("put_block/get_block require unit stride in the contiguous dimension")
    count = _as_static_int(elements)
    if count is None:
        raise ValueError(
            "dynamic put_block/get_block size is unsupported in M1 without an explicit "
            "range proof (0 <= start and start + size <= extent)"
        )
    last_extent = _as_static_int(buffer.shape[-1])
    last_index = _as_static_int(load.indices[-1])
    if count > 0 and (last_extent is None or last_index is None):
        raise ValueError(
            "put_block/get_block need a static final-dimension start and extent in M1; "
            "a dynamic interval requires an explicit range proof (0 <= start and start + size <= extent)"
        )
    if last_extent is not None and last_index is not None and count > last_extent - last_index:
        raise ValueError("put_block/get_block typed element interval exceeds the contiguous row")


def _contiguous_region(value: tir.BufferLoad | tir.Buffer, elements: Any, access: str) -> tir.PrimExpr:
    load = _load(value, "copy endpoint")
    _require_contiguous_run(load, elements)
    # A block copy is one typed contiguous run starting at the supplied load.
    # The extent is explicit rather than inferred from byte arithmetic.
    extents = [tir.const(1, index.dtype) for index in load.indices]
    extents[-1] = elements
    return region(load, access, *extents)


def _typed_size(size: Any) -> Any:
    if not _is_integer_scalar(size):
        raise TypeError("put_block/get_block size must be a scalar integer typed element count")
    static = _as_static_int(size)
    if static is not None and static < 0:
        raise ValueError("put_block/get_block size must be non-negative")
    return size


def _region_load(value: Any, what: str) -> tir.BufferLoad:
    if not _is_region(value) or not isinstance(value.args[0], tir.BufferLoad):
        raise TypeError(f"{what} must be a tl.region produced from a BufferLoad")
    return value.args[0]


def validate_block_copy_regions(src: Any, dst: Any, size: Any) -> None:
    """Apply the same typed-run checks to TIR printer/raw-region input."""
    size = _typed_size(size)
    src_load, dst_load = _region_load(src, "src"), _region_load(dst, "dst")
    if src_load.dtype != dst_load.dtype:
        raise TypeError("put_block/get_block require src and dst to have the same element dtype")
    _require_contiguous_run(src_load, size)
    _require_contiguous_run(dst_load, size)
    for endpoint, label in ((src, "src"), (dst, "dst")):
        extents = endpoint.args[2:]
        endpoint_load = _region_load(endpoint, label)
        if not endpoint_load.indices or len(extents) != len(endpoint_load.indices):
            raise ValueError("put_block/get_block region rank does not match endpoint rank")
        if any(_as_static_int(extent) != 1 for extent in extents[:-1]) or _as_static_int(extents[-1]) != _as_static_int(size):
            raise ValueError("put_block/get_block raw region must describe exactly one contiguous typed run")


def put_block(
    src: tir.BufferLoad | tir.Buffer,
    dst: tir.BufferLoad | tir.Buffer,
    size: Any,
    dst_pe: Any = -1,
    *,
    peer: Any | None = None,
):
    """Copy ``size`` elements cooperatively from local ``src`` to ``dst``.

    ``size`` is an element count in ``src.dtype``/``dst.dtype``; it is never a
    byte count and this primitive intentionally does not publish a ready flag.
    """
    if _is_region(src) and _is_region(dst):
        validate_block_copy_regions(src, dst, size)
        _validate_peer_type(dst_pe if peer is None else peer, "put_block")
        return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.put_block"), src, dst, size, dst_pe if peer is None else peer)
    src_load, dst_load = _load(src, "src"), _load(dst, "dst")
    if src_load.dtype != dst_load.dtype:
        raise TypeError("put_block requires src and dst to have the same element dtype")
    size = _typed_size(size)
    _validate_peer_type(dst_pe if peer is None else peer, "put_block")
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.tileop.put_block"),
        _contiguous_region(src_load, _typed_size(size), "r"),
        _contiguous_region(dst_load, _typed_size(size), "w"), size,
        dst_pe if peer is None else peer,
    )


def get_block(
    src: tir.BufferLoad | tir.Buffer,
    dst: tir.BufferLoad | tir.Buffer,
    size: Any,
    src_pe: Any = -1,
    *,
    peer: Any | None = None,
):
    """Copy ``size`` typed elements cooperatively from ``peer`` into ``dst``."""
    if _is_region(src) and _is_region(dst):
        validate_block_copy_regions(src, dst, size)
        _validate_peer_type(src_pe if peer is None else peer, "get_block")
        return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.get_block"), src, dst, size, src_pe if peer is None else peer)
    src_load, dst_load = _load(src, "src"), _load(dst, "dst")
    if src_load.dtype != dst_load.dtype:
        raise TypeError("get_block requires src and dst to have the same element dtype")
    size = _typed_size(size)
    _validate_peer_type(src_pe if peer is None else peer, "get_block")
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.tileop.get_block"),
        _contiguous_region(src_load, _typed_size(size), "r"),
        _contiguous_region(dst_load, _typed_size(size), "w"), size,
        src_pe if peer is None else peer,
    )


def _memory_mode(scope: str, sem: str, allowed: dict[str, int], operation: str, *, cache: Any, na: bool, nc: bool):
    if cache is not None or na or nc:
        raise NotImplementedError(
            f"{operation} PTX cache/na/nc options are not supported by the HIP M1 backend"
        )
    if scope not in _SCOPE:
        raise ValueError(f"{operation} scope must be one of {sorted(_SCOPE)}, got {scope!r}")
    if sem not in allowed:
        raise ValueError(f"{operation} semantics must be one of {sorted(allowed)}, got {sem!r}")
    return _SCOPE[scope], allowed[sem]


def _validate_scalar_buffer(load: tir.BufferLoad, operation: str) -> None:
    if load.dtype not in _SCALAR_DTYPES:
        raise TypeError(f"{operation} supports only HIP M1 scalar dtypes {sorted(_SCALAR_DTYPES)}, got {load.dtype}")
    _validate_indices(load)


def _value_matches_dtype(value: Any, dtype: str, operation: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{operation} value must have dtype {dtype}, not bool")
    if isinstance(value, int):
        if dtype not in {"int32", "uint32"}:
            raise TypeError(f"{operation} value must have dtype {dtype}")
        return
    if isinstance(value, float):
        if dtype != "float32":
            raise TypeError(f"{operation} value must have dtype {dtype}")
        return
    if not isinstance(value, tir.PrimExpr) or str(value.dtype) != dtype:
        raise TypeError(f"{operation} value must have dtype {dtype}")


def _raw_memory_mode(sem_id: Any, scope_id: Any, allowed: dict[str, int], operation: str) -> None:
    sem_value, scope_value = _as_static_int(sem_id), _as_static_int(scope_id)
    if scope_value not in set(_SCOPE.values()) or sem_value not in set(allowed.values()):
        raise ValueError(f"{operation} raw TIR has unsupported HIP M1 scope/semantics")


def st(
    dst: tir.BufferLoad | tir.Buffer,
    value: Any,
    scope: str = "gpu",
    sem: str = "weak",
    na: bool = False,
    dst_pe: Any = -1,
    *,
    peer: Any | None = None,
    cache: Any = None,
):
    """Emit an explicit scoped scalar store marker (HIP whitelist only)."""
    if _is_region(dst):
        # TIR printer supplies raw `(region, value, sem_id, scope_id, peer)`
        # positionally.  In this parser-only branch the public parameter names
        # therefore contain sem_id/scope_id/peer as scope/sem/na.
        target = _region_load(dst, "st destination")
        _validate_scalar_buffer(target, "st")
        _value_matches_dtype(value, target.dtype, "st")
        _raw_memory_mode(scope, sem, _ST_SEM, "st")
        _validate_peer_type(na, "st")
        return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.st"), dst, value, scope, sem, na)
    scope_id, sem_id = _memory_mode(scope, sem, _ST_SEM, "st", cache=cache, na=na, nc=False)
    target = _load(dst, "dst")
    _validate_scalar_buffer(target, "st")
    _value_matches_dtype(value, target.dtype, "st")
    _validate_peer_type(dst_pe if peer is None else peer, "st")
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.tileop.st"),
        region(target, "w", *([1] * len(target.indices))), value, sem_id, scope_id,
        dst_pe if peer is None else peer,
    )


def ld(
    src: tir.BufferLoad | tir.Buffer,
    value: tir.BufferLoad | tir.Buffer,
    scope: str = "gpu",
    sem: str = "weak",
    na: bool = False,
    nc: bool = False,
    src_pe: Any = -1,
    *,
    peer: Any | None = None,
    cache: Any = None,
):
    """Load scalar ``src`` into writable scalar ``value`` (HIP whitelist)."""
    if _is_region(src) and _is_region(value):
        # See `st`: this is the raw printer/parser form, not the public DSL
        # argument convention.
        load, result = _region_load(src, "ld source"), _region_load(value, "ld value")
        _validate_scalar_buffer(load, "ld")
        _validate_scalar_buffer(result, "ld")
        if load.dtype != result.dtype:
            raise TypeError("ld requires src and writable value to have the same element dtype")
        _raw_memory_mode(scope, sem, _LD_SEM, "ld")
        _validate_peer_type(na, "ld")
        return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.ld"), src, value, scope, sem, na)
    scope_id, sem_id = _memory_mode(scope, sem, _LD_SEM, "ld", cache=cache, na=na, nc=nc)
    load, result = _load(src, "src"), _load(value, "ld value")
    _validate_scalar_buffer(load, "ld")
    _validate_scalar_buffer(result, "ld")
    if load.dtype != result.dtype:
        raise TypeError("ld requires src and writable value to have the same element dtype")
    _validate_peer_type(src_pe if peer is None else peer, "ld")
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.tileop.ld"),
        region(load, "r", *([1] * len(load.indices))),
        region(result, "w", *([1] * len(result.indices))), sem_id, scope_id,
        src_pe if peer is None else peer,
    )
