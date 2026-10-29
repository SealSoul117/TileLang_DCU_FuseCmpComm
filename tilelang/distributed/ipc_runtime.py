"""Lazy ctypes access to the installed M2 HIP-IPC runtime.

The caller owns MPI initialization/finalization.  A :class:`DeviceAllocation`
or :class:`DeviceView` intentionally implements only ``data_ptr()``: this is
the small common protocol consumed by TileLang's ctypes adapter, so no torch
or DLPack wrapper is required for symmetric IPC allocations.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path


class IPCError(RuntimeError):
    """A status returned by the embedded HIP-IPC runtime."""


def _library_path() -> str:
    explicit = os.environ.get("TILELANG_DCU_IPC_LIBRARY")
    if explicit:
        return explicit
    package = Path(__file__).resolve().parents[1]
    candidates = [package / "lib" / "libtilelang_dcu_ipc_runtime.so"]
    # Development trees retain the normal CMake layout.  This is a TileLang
    # path, not the historical external dcu_ipc checkout.
    dev_root = os.environ.get("TILELANG_DEV_BUILD_ROOT")
    if dev_root:
        candidates.append(Path(dev_root) / "lib" / "libtilelang_dcu_ipc_runtime.so")
    candidates.append(package.parent / "build-old" / "lib" / "libtilelang_dcu_ipc_runtime.so")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise IPCError("embedded TileLang DCU-IPC runtime library was not found")


class _API:
    def __init__(self, path: str | None = None):
        self.lib = ctypes.CDLL(path or _library_path(), mode=ctypes.RTLD_GLOBAL)
        context = ctypes.c_void_p
        allocation = ctypes.c_void_p
        self.lib.tl_dcu_ipc_init_world.argtypes = [ctypes.c_int, ctypes.POINTER(context)]
        self.lib.tl_dcu_ipc_init_world.restype = ctypes.c_int
        self.lib.tl_dcu_ipc_rank.argtypes = [context]
        self.lib.tl_dcu_ipc_rank.restype = ctypes.c_int
        self.lib.tl_dcu_ipc_world_size.argtypes = [context]
        self.lib.tl_dcu_ipc_world_size.restype = ctypes.c_int
        self.lib.tl_dcu_ipc_alloc.argtypes = [context, ctypes.c_char_p, ctypes.c_size_t,
                                               ctypes.c_int, ctypes.POINTER(allocation)]
        self.lib.tl_dcu_ipc_alloc.restype = ctypes.c_int
        self.lib.tl_dcu_ipc_local_ptr.argtypes = [allocation]
        self.lib.tl_dcu_ipc_local_ptr.restype = ctypes.c_void_p
        self.lib.tl_dcu_ipc_device_view_words.argtypes = [context, allocation,
                                                           ctypes.POINTER(ctypes.c_void_p)]
        self.lib.tl_dcu_ipc_device_view_words.restype = ctypes.c_int
        self.lib.tl_dcu_ipc_sync.argtypes = [context]
        self.lib.tl_dcu_ipc_sync.restype = ctypes.c_int
        self.lib.tl_dcu_ipc_free.argtypes = [context, ctypes.POINTER(allocation)]
        self.lib.tl_dcu_ipc_free.restype = ctypes.c_int
        self.lib.tl_dcu_ipc_finalize.argtypes = [ctypes.POINTER(context)]
        self.lib.tl_dcu_ipc_finalize.restype = ctypes.c_int
        self.lib.tl_dcu_ipc_last_error.restype = ctypes.c_char_p

    def check(self, status: int, operation: str) -> None:
        if status:
            message = self.lib.tl_dcu_ipc_last_error()
            detail = message.decode(errors="replace") if message else "unknown runtime error"
            raise IPCError(f"{operation} failed ({status}): {detail}")


class DeviceView:
    """A device-resident ``uint64[6]`` descriptor accepted by ctypes JIT."""
    def __init__(self, allocation: "DeviceAllocation", pointer: int):
        self._allocation = allocation
        self._pointer = pointer

    def data_ptr(self) -> int:
        if self._allocation.closed:
            raise IPCError("DeviceView refers to a freed IPC allocation")
        return self._pointer


class DeviceAllocation:
    """A live symmetric allocation with a ctypes-adapter ``data_ptr``."""
    def __init__(self, context: "IPCContext", handle: ctypes.c_void_p):
        self._context, self._handle, self._view = context, handle, None
        self.closed = False

    def data_ptr(self) -> int:
        if self.closed:
            raise IPCError("IPC allocation is already freed")
        pointer = self._context._api.lib.tl_dcu_ipc_local_ptr(self._handle)
        if not pointer:
            raise IPCError("IPC allocation has no live local pointer")
        return int(pointer)

    def device_view(self) -> DeviceView:
        if self.closed:
            raise IPCError("IPC allocation is already freed")
        # Always ask the C ABI, even when Python has a wrapper cached: the C
        # side owns the active-context/owner-thread check and its descriptor
        # map makes this a no-allocation cache hit on the normal path.
        pointer = ctypes.c_void_p()
        self._context._api.check(
            self._context._api.lib.tl_dcu_ipc_device_view_words(
                self._context._handle, self._handle, ctypes.byref(pointer)),
            "device_view",
        )
        if self._view is None:
            self._view = DeviceView(self, int(pointer.value))
        elif self._view.data_ptr() != int(pointer.value):
            raise IPCError("runtime changed a live allocation's DeviceView address")
        return self._view

    def close(self) -> None:
        if not self.closed:
            status = self._context._api.lib.tl_dcu_ipc_free(self._context._handle, ctypes.byref(self._handle))
            # A descriptor hipFree failure is reported, but the core
            # allocation has already been collectively retired and its handle
            # is deliberately nulled to prevent an unsafe retry/launch.
            if not self._handle.value:
                self.closed = True
            self._context._api.check(status, "free")


class IPCContext:
    """One active embedded dcu_ipc context; MPI is initialized by the caller."""
    def __init__(self, device_ordinal: int = -1, *, library: str | None = None):
        self._api = _API(library)
        self._handle = ctypes.c_void_p()
        self._api.check(self._api.lib.tl_dcu_ipc_init_world(device_ordinal, ctypes.byref(self._handle)), "init_world")
        self.rank = int(self._api.lib.tl_dcu_ipc_rank(self._handle))
        self.world_size = int(self._api.lib.tl_dcu_ipc_world_size(self._handle))
        self.closed = False

    def alloc(self, tag: str, nbytes: int, *, fine_grained: bool = True) -> DeviceAllocation:
        if self.closed or nbytes <= 0:
            raise IPCError("allocation requires a live context and positive byte size")
        handle = ctypes.c_void_p()
        self._api.check(self._api.lib.tl_dcu_ipc_alloc(
            self._handle, tag.encode(), nbytes, int(fine_grained), ctypes.byref(handle)), "alloc")
        return DeviceAllocation(self, handle)

    def sync(self) -> None:
        self._api.check(self._api.lib.tl_dcu_ipc_sync(self._handle), "sync")

    def close(self) -> None:
        if not self.closed:
            self._api.check(self._api.lib.tl_dcu_ipc_finalize(ctypes.byref(self._handle)), "finalize")
            self.closed = True
