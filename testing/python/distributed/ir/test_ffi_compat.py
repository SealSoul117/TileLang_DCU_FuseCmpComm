"""Regression coverage for TileLang's single-registry FFI compatibility layer."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_ffi_compat_uses_one_real_registry():
    """Root import selects one real registry for registration and containers."""

    program = textwrap.dedent(
        """
        import sys

        import tvm

        import tilelang
        from tilelang import _ffi_compat

        builtin_ffi = getattr(tvm, "ffi", None)
        if _ffi_compat._ffi is builtin_ffi:
            assert not any(
                name == "tvm_ffi" or name.startswith("tvm_ffi.") for name in sys.modules
            )
            registry = builtin_ffi
        else:
            import tvm_ffi

            assert _ffi_compat._ffi is tvm_ffi
            registry = tvm_ffi

        assert _ffi_compat.Array is registry.Array

        @_ffi_compat.register_global_func(
            "testing.tilelang_ffi_compat.decorator", override=True
        )
        def add_one(value):
            return value + 1

        _ffi_compat.register_global_func(
            "testing.tilelang_ffi_compat.keyword", f=lambda value: value + 2, override=True
        )
        assert _ffi_compat.get_global_func("testing.tilelang_ffi_compat.decorator")(41) == 42
        assert _ffi_compat.get_global_func("testing.tilelang_ffi_compat.keyword")(40) == 42

        native_array = _ffi_compat.Array([42])
        assert native_array[0] == 42
        native_map = _ffi_compat.Map({"answer": 42})
        assert isinstance(native_map, _ffi_compat.Map)
        assert native_map["answer"] == 42
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
