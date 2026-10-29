"""Opaque M1 synchronization and signal TIR primitives."""

from __future__ import annotations

from enum import Enum
from typing import Any

from tvm import ir, tir

from tilelang.language.utils import region
from .common import _validate_peer_type


class WaitScope(Enum):
    SYS = 0


class WaitSemantics(Enum):
    ACQUIRE = 0


def _enum(value: Any, enum_cls: type[Enum], name: str) -> int:
    if isinstance(value, enum_cls):
        return value.value
    if isinstance(value, str) and value.upper() in enum_cls.__members__:
        return enum_cls[value.upper()].value
    if isinstance(value, tir.IntImm):
        value = int(value.value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value in {item.value for item in enum_cls}:
            return value
    raise ValueError(f"unsupported {name}: {value!r}")


def _uint32_load(value: tir.BufferLoad, what: str) -> tir.BufferLoad:
    if not isinstance(value, tir.BufferLoad):
        raise TypeError(f"{what} must be a uint32 tir.BufferLoad")
    if value.dtype != "uint32":
        raise TypeError(f"{what} must be uint32, got {value.dtype}")
    return value


def _scalar_region(value: tir.BufferLoad, access: str) -> tir.PrimExpr:
    return region(value, access, *([tir.const(1, index.dtype) for index in value.indices]))


def _uint32_value(value: Any, what: str) -> None:
    """Validate a uint32 payload/expected value without silently narrowing."""
    if isinstance(value, bool):
        raise TypeError(f"{what} must be uint32-representable, not bool")
    if isinstance(value, int):
        if 0 <= value <= 0xFFFFFFFF:
            return
        raise ValueError(f"{what} must fit uint32")
    if isinstance(value, tir.IntImm):
        # Do not accept a large unsigned immediate merely because its *source*
        # dtype is unsigned: M1 must not silently narrow uint64 to uint32.
        if 0 <= int(value.value) <= 0xFFFFFFFF:
            return
        raise ValueError(f"{what} must fit uint32")
    if isinstance(value, tir.PrimExpr) and value.dtype == "uint32":
        return
    raise TypeError(f"{what} must be a uint32 expression or a static uint32-representable integer")


def _region_flag(value: Any, what: str) -> None:
    if not (isinstance(value, tir.Call) and isinstance(value.op, ir.Op) and value.op.name == "tl.tileop.region"):
        raise TypeError(f"{what} must be a uint32 tl.region")
    load = value.args[0] if value.args else None
    if not isinstance(load, tir.BufferLoad) or load.dtype != "uint32":
        raise TypeError(f"{what} must be a uint32 tl.region")


def fence_sys():
    """Opaque system fence for writes issued by the calling thread.

    This is not a CTA barrier and M1 does not claim a validated HIP hardware
    acquire/release implementation; M2 owns that device-level proof.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.fence_sys"))


def signal_set(flag: tir.BufferLoad, value: Any, peer: Any = -1):
    """Opaque system-scope uint32 signal publication marker."""
    if isinstance(flag, tir.Call) and isinstance(flag.op, ir.Op) and flag.op.name == "tl.tileop.region":
        _region_flag(flag, "signal flag")
        _uint32_value(value, "signal value")
        _validate_peer_type(peer, "signal_set")
        return tir.call_intrin("handle", tir.op.Op.get("tl.signal_set"), flag, value, peer)
    flag = _uint32_load(flag, "signal flag")
    _uint32_value(value, "signal value")
    _validate_peer_type(peer, "signal_set")
    return tir.call_intrin("handle", tir.op.Op.get("tl.signal_set"), _scalar_region(flag, "w"), value, peer)


def _wait(relation: int, value: tir.BufferLoad, expected: Any, peer: Any = -1, *,
          scope: WaitScope | str = WaitScope.SYS,
          semantics: WaitSemantics | str = WaitSemantics.ACQUIRE):
    return wait(relation, value, expected, peer, scope=scope, semantics=semantics)


def wait(relation: int, value: tir.BufferLoad | tir.PrimExpr, expected: Any, peer: Any = -1,
         scope: WaitScope | str | int = WaitScope.SYS,
         semantics: WaitSemantics | str | int = WaitSemantics.ACQUIRE):
    """Raw wait form used by the TIR printer; prefer :func:`wait_eq`/`wait_ge`."""
    if relation not in (0, 2):
        raise ValueError("M1 wait relation must be EQ(0) or GE(2)")
    is_region = isinstance(value, tir.Call) and isinstance(value.op, ir.Op) and value.op.name == "tl.tileop.region"
    if not is_region:
        value = _uint32_load(value, "wait value")
        value = _scalar_region(value, "r")
    else:
        _region_flag(value, "wait value")
    _uint32_value(expected, "wait expected")
    _validate_peer_type(peer, "wait")
    scope_id = _enum(scope, WaitScope, "wait scope")
    sem_id = _enum(semantics, WaitSemantics, "wait semantics")
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.tileop.wait"), relation, value, expected, peer, scope_id, sem_id,
    )


def wait_eq(value: tir.BufferLoad, expected: Any, peer: Any = -1, *,
            scope: WaitScope | str = WaitScope.SYS,
            semantics: WaitSemantics | str = WaitSemantics.ACQUIRE):
    """Wait until a uint32 flag equals ``expected`` with SYS/ACQUIRE defaults."""
    return _wait(0, value, expected, peer, scope=scope, semantics=semantics)


def wait_ge(value: tir.BufferLoad, expected: Any, peer: Any = -1, *,
            scope: WaitScope | str = WaitScope.SYS,
            semantics: WaitSemantics | str = WaitSemantics.ACQUIRE):
    """Wait until a uint32 flag is at least ``expected`` with SYS/ACQUIRE defaults."""
    return _wait(2, value, expected, peer, scope=scope, semantics=semantics)
