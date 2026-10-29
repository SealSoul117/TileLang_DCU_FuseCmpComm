"""Use one TVM FFI registry for TileLang's Python bindings.

The bundled TVM exposes its FFI through :mod:`tvm.ffi`; another supported
layout provides it through the separately packaged :mod:`tvm_ffi`.  Loading
both registries in a process is unsafe: their native type-index registries are
independent.  Prefer the already-loaded TVM registry and only load the
standalone package when no usable in-tree TVM FFI is present.
"""

from __future__ import annotations

import sys
from typing import Any, Callable


def _loaded_tvm_ffi() -> Any | None:
    """Return the usable builtin FFI from an already-loaded ``tvm`` module."""

    tvm = sys.modules.get("tvm")
    ffi = getattr(tvm, "ffi", None)
    required = ("register_object", "register_func", "_init_api", "get_global_func", "Map")
    if ffi is not None and all(callable(getattr(ffi, name, None)) for name in required):
        return ffi
    return None


_ffi = _loaded_tvm_ffi()
if _ffi is None:
    # Do not import this package when TVM's builtin FFI is available.  Its
    # native registry must never be loaded alongside TVM's registry.
    import tvm_ffi as _ffi

    _register_func = _ffi.register_global_func
    _init_api = _ffi.init_ffi_api
else:
    _register_func = _ffi.register_func
    _init_api = _ffi._init_api


register_object = _ffi.register_object
get_global_func = _ffi.get_global_func
Array = _ffi.Array
Map = _ffi.Map


def register_global_func(
    func_name: str | Callable[..., Any],
    f: Callable[..., Any] | None = None,
    override: bool = False,
) -> Any:
    """Register a function with either FFI API's decorator-compatible form."""

    if f is None:
        return _register_func(func_name, override=override)
    return _register_func(func_name, f=f, override=override)


def init_ffi_api(namespace: str, target_module_name: str | None = None) -> None:
    """Populate a module with functions registered under ``namespace``."""

    _init_api(namespace, target_module_name)


__all__ = ["Array", "Map", "get_global_func", "init_ffi_api", "register_global_func", "register_object"]
