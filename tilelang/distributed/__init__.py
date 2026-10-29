"""Serializable contracts for the DCU distributed-fusion design.

This package keeps its scheduling/contracts importable without :mod:`tilelang`.
The deferred M0 IPC candidate remains a contract layer; the optional M2 HIP
binding is loaded only when explicitly requested.
"""

from .schedule import ConsumerAssignment, ProducerAssignment, ReadyKey, SchedulePlan
from .spec import (
    ABI_VERSION,
    BufferBinding,
    CompletionContract,
    DistributedConfigError,
    DistributedKernelABI,
    FusionSpec,
    LaunchConfig,
    ResetProtocol,
    RuntimeBinding,
    SymmetricBufferDesc,
    WorkspaceRequirement,
)
from .ipc_spec import IPC_ABI_VERSION, IPCDeviceBinding, IPCFusionSpec, IPCKernelABI, IPCMemoryRequirement
from .ipc_schedule import IPCAGQueueClaim, IPCAGTransfer, IPCSchedulePlan, IPCTileView

__all__ = [
    "ABI_VERSION",
    "BufferBinding",
    "CompletionContract",
    "ConsumerAssignment",
    "DistributedConfigError",
    "DistributedKernelABI",
    "FusionSpec",
    "LaunchConfig",
    "ProducerAssignment",
    "ReadyKey",
    "ResetProtocol",
    "RuntimeBinding",
    "SchedulePlan",
    "SymmetricBufferDesc",
    "WorkspaceRequirement",
    "IPC_ABI_VERSION",
    "IPCDeviceBinding",
    "IPCFusionSpec",
    "IPCKernelABI",
    "IPCMemoryRequirement",
    "IPCAGQueueClaim",
    "IPCAGTransfer",
    "IPCSchedulePlan",
    "IPCTileView",
    "DEVICE_VIEW_WORDS",
    "IPCBindingError",
    "bind_ipc",
    "IPCContext",
    "IPCError",
    "DeviceAllocation",
    "DeviceView",
]


def __getattr__(name):
    # Importing ``tilelang.distributed`` is intentionally safe for the
    # pre-existing CPU-only scheduling users.  The compiler-facing M2 binding
    # imports TIR and is therefore lazy rather than an eager package dependency.
    if name in {"DEVICE_VIEW_WORDS", "IPCBindingError", "bind_ipc"}:
        from . import ipc_binding
        return getattr(ipc_binding, name)
    if name in {"IPCContext", "IPCError", "DeviceAllocation", "DeviceView"}:
        from . import ipc_runtime
        return getattr(ipc_runtime, name)
    raise AttributeError(name)
