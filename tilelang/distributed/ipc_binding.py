"""Explicit M2 HIP-IPC bindings for an ordinary TileLang ``PrimFunc``.

This module intentionally does not use the deferred ABI2/M0 proposal.  A
binding is local compilation metadata: it says which *existing* global buffer
is represented by which device-resident ``DeviceView`` descriptor argument.
The descriptor, rank, and world-size remain ordinary public kernel arguments.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tvm import ir, tir
from tilelang._ffi_compat import get_global_func

from tilelang.language.distributed.ir import normalize_remote_views, verify_distributed_ir

_ATTR = "tl.dcu_ipc.bindings"
_READY_ATTR = "tl.dcu_ipc.abi_ready"
_RANK = "__tl_ipc_rank"
_WORLD = "__tl_ipc_world_size"
_PREFIX = "__tl_ipc_view_"
# DeviceView is six 64-bit words on the supported LP64 HIP ABI.  Codegen also
# static_asserts this fact, so an ABI drift fails compilation rather than
# silently reinterpreting an undersized descriptor buffer.
DEVICE_VIEW_WORDS = 6


class IPCBindingError(ValueError):
    """The explicit HIP-IPC binding is incomplete or unsafe to lower."""


def _var_name(value: tir.Var) -> str:
    # The M1 TVM Python binding exposes ``name`` while newer bindings expose
    # ``name_hint``.  TIR serialization is independent of this convenience.
    return str(getattr(value, "name_hint", getattr(value, "name")))


def _buffer_index(func: tir.PrimFunc) -> dict[str, tir.Buffer]:
    result: dict[str, tir.Buffer] = {}
    for parameter, buffer in func.buffer_map.items():
        for name in (buffer.name, _var_name(parameter), _var_name(buffer.data)):
            previous = result.get(name)
            if previous is not None and previous.data != buffer.data:
                raise IPCBindingError(f"ambiguous PrimFunc buffer alias {name!r}")
            result[name] = buffer
    return result


def bind_ipc(func: tir.PrimFunc, views: Mapping[str, str] | Sequence[str]) -> tir.PrimFunc:
    """Attach explicit DeviceView bindings to ``func``.

    A sequence lists existing input buffer names.  A mapping additionally
    chooses each public descriptor argument name.  Thus no Python object is
    silently discarded: runtime launch code receives precisely the ordinary
    ``uint64[6]`` argument names declared by the prepared PrimFunc.
    """
    if not isinstance(func, tir.PrimFunc):
        raise TypeError("bind_ipc expects a tir.PrimFunc")
    if isinstance(views, Mapping):
        pairs = list(views.items())
    elif isinstance(views, Sequence) and not isinstance(views, (str, bytes)):
        pairs = [(item, "ipc_view_" + str(item)) for item in views]
    else:
        raise IPCBindingError("bind_ipc requires buffer names or a buffer-to-descriptor-name mapping")
    if not pairs:
        raise IPCBindingError("bind_ipc requires at least one existing buffer")
    known = _buffer_index(func)
    names: list[str] = []
    descriptor_names: list[str] = []
    for supplied, descriptor in pairs:
        if not isinstance(supplied, str) or supplied not in known:
            raise IPCBindingError(f"IPC binding {supplied!r} is not an existing PrimFunc buffer")
        if not isinstance(descriptor, str) or not descriptor.isidentifier() or descriptor.startswith("__tl_ipc_"):
            raise IPCBindingError("IPC descriptor names must be public C-like identifiers outside the reserved prefix")
        if descriptor in known:
            raise IPCBindingError("IPC descriptor argument must not shadow an existing buffer alias")
        # Buffer.name is stable across remote-view normalization; data Var
        # names are allowed to be rewritten by the TileLang parser.
        base = str(known[supplied].name)
        if base in names:
            # Aliases are allowed only when they explicitly select the same
            # descriptor; otherwise the base-to-allocation provenance is not
            # well-defined.
            if descriptor_names[names.index(base)] != descriptor:
                raise IPCBindingError(f"base buffer {base!r} is bound to two descriptor arguments")
        else:
            names.append(base)
            descriptor_names.append(descriptor)
    if len(set(descriptor_names)) != len(descriptor_names):
        raise IPCBindingError("each IPC allocation needs a distinct descriptor argument")
    return func.with_attr(_ATTR, ";".join(f"{base}={descriptor}" for base, descriptor in zip(names, descriptor_names)))


def has_ipc_binding(func: tir.PrimFunc) -> bool:
    return bool(isinstance(func, tir.PrimFunc) and func.attrs and _ATTR in func.attrs)


def prepare_ipc_primfunc(func: tir.PrimFunc, *, target_kind: str) -> tir.PrimFunc:
    """Normalize, verify, and append the explicit ordinary IPC ABI arguments.

    This is invoked by the normal ``engine.lower`` path.  It is idempotent so
    lowering an already prepared function is safe, while a non-HIP target
    fails loudly instead of emitting an unimplemented communication call.
    """
    if not has_ipc_binding(func):
        return func
    if target_kind != "hip":
        raise IPCBindingError("HIP IPC bindings require a HIP target")
    if func.attrs and _READY_ATTR in func.attrs:
        return func
    records = str(func.attrs[_ATTR]).split(";")
    pairs: list[tuple[str, str]] = []
    for record in records:
        base, separator, descriptor = record.partition("=")
        if not separator or not base or not descriptor:
            raise IPCBindingError("malformed IPC binding attribute")
        pairs.append((base, descriptor))
    known = _buffer_index(func)
    for name, _ in pairs:
        if name not in known:
            raise IPCBindingError(f"IPC bound base {name!r} is no longer a function buffer")
        if known[name].scope() != "global":
            raise IPCBindingError(f"IPC binding {name!r} must be global")
    normalized = normalize_remote_views(func)
    verify_distributed_ir(normalized)
    params = list(normalized.params)
    buffers = dict(normalized.buffer_map)
    occupied = {_var_name(var) for var in params}
    descriptors: list[tir.Var] = []
    for _, descriptor_name in pairs:
        if descriptor_name in occupied:
            raise IPCBindingError(f"reserved IPC descriptor parameter {descriptor_name!r} already exists")
        # Let decl_buffer allocate its data Var: this older TVM binding does
        # not expose a Python Var constructor with PointerType annotation.
        # Its generated data Var is therefore a normal typed buffer parameter
        # understood by extrac_params and the existing ctypes wrapper.
        descriptor_buffer = tir.decl_buffer(
            (DEVICE_VIEW_WORDS,), "uint64", name=descriptor_name,
            scope="global", data_alignment=8,
        )
        descriptor = descriptor_buffer.data
        buffers[descriptor] = descriptor_buffer
        params.append(descriptor)
        descriptors.append(descriptor)
        occupied.add(descriptor_name)
    for name in (_RANK, _WORLD):
        if name in occupied:
            raise IPCBindingError(f"reserved IPC scalar parameter {name!r} already exists")
        params.append(tir.Var(name, "int32"))
    attrs = dict(normalized.attrs) if normalized.attrs else {}
    attrs[_READY_ATTR] = 1
    # This TVM revision exposes no public DictAttrs constructor.  Build the
    # structural function first, then use the stable functional attr API.
    result = tir.PrimFunc(params, normalized.body, normalized.ret_type, buffers,
                          None, normalized.span)
    for key, value in attrs.items():
        result = result.with_attr(key, value)
    # The native lowering pass handles every expression position and opaque
    # call safely; the matched TVM Python mutator creates invalid TIR for a
    # remote-alias/serial-loop combination.
    return get_global_func("tl.transform.LowerDCUIPC")(result)


def ipc_descriptor_param_name(base_data_name: str) -> str:
    return "ipc_view_" + base_data_name
