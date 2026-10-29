"""M0 immutable, JSON-serializable distributed-fusion contracts.

The objects in this file describe a compiled collective and its symmetric
workspaces.  They do not allocate memory, compile code, or hold native
pointers.  Native pointers, context handles, ranks, and allocation generations
are represented by :class:`RuntimeBinding` and intentionally excluded from a
shared compilation identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import gcd
from typing import Any, Iterable, Mapping, Tuple


ABI_VERSION = 1
_DTYPE_BYTES = {"fp16": 2, "float16": 2, "fp32": 4, "float32": 4, "uint32": 4}
_CANONICAL_DTYPE = {"float16": "fp16", "float32": "fp32"}
_BATCH_TILES = frozenset((1, 2, 4, 8))


class DistributedConfigError(ValueError):
    """Raised when an M0 contract would be ambiguous or unsafe to lower."""


def _fail(message: str) -> None:
    raise DistributedConfigError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_object(value: Any, name: str, fields: Iterable[str]) -> dict[str, Any]:
    """Accept exactly one declared JSON object schema, with no silent fields."""
    if not isinstance(value, Mapping):
        _fail(f"{name} JSON value must be an object")
    result = dict(value)
    expected = set(fields)
    missing = expected - set(result)
    unknown = set(result) - expected
    if missing:
        _fail(f"{name} JSON object is missing fields: {sorted(missing)}")
    if unknown:
        _fail(f"{name} JSON object has unknown fields: {sorted(unknown)}")
    return result


def _parse_json(value: Any, name: str) -> Any:
    if not isinstance(value, str):
        _fail(f"{name} JSON must be a string")
    try:
        return json.loads(value)
    except (TypeError, ValueError) as error:
        _fail(f"invalid {name} JSON: {error}")


def _canonical_dtype(dtype: str) -> str:
    if not isinstance(dtype, str):
        _fail("dtype must be a string")
    dtype = _CANONICAL_DTYPE.get(dtype, dtype)
    if dtype not in _DTYPE_BYTES:
        _fail(f"unsupported dtype {dtype!r}; supported dtypes are {sorted(_DTYPE_BYTES)}")
    return dtype


def _tuple_of_ints(value: Iterable[int], field: str, *, positive: bool = True) -> Tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError:
        _fail(f"{field} must be an iterable of integers")
    if not result:
        _fail(f"{field} must not be empty")
    if any(not _is_int(item) for item in result):
        _fail(f"{field} must contain only integers")
    if positive and any(item <= 0 for item in result):
        _fail(f"{field} must contain only positive integers")
    if not positive and any(item < 0 for item in result):
        _fail(f"{field} must contain only non-negative integers")
    return result


def _json_dump(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class FusionSpec:
    """Rank-independent semantic contract for one supported fused collective.

    M0 supports the direct-push owner-reduce algorithms for GEMM+AllReduce
    (``collective_kind='allreduce'``) and GEMM+ReduceScatter
    (``collective_kind='reduce_scatter'``).  AllGather is intentionally
    rejected: its input-readiness protocol is not the output-ready protocol
    modelled here.

    The pure schedule can describe matrix edge tiles via
    ``SchedulePlan.tile_extent``.  M0 does *not* claim that a GPU edge-tile
    epilogue exists: ``validate_v1_lowering`` rejects such a spec until a later
    module implements and validates padded/guarded lowering.  ``K_local`` is
    likewise kept explicit but v1 lowering requires its known 32-way alignment.
    """

    m: int = 4096
    n: int = 4096
    k_local: int = 1024
    nranks: int = 4
    tile_shape: Tuple[int, int] = (128, 128)
    compute_kind: str = "gemm"
    collective_kind: str = "allreduce"
    algorithm: str = "direct_push_owner_reduce_v1"
    input_dtype: str = "fp16"
    accumulator_dtype: str = "fp32"
    communication_dtype: str = "fp32"
    output_dtype: str = "fp32"
    output_layout: str = "tile_major"
    mathematical_output_distribution: str = "replicated"
    owner_policy: str = "cyclic"
    compute_tile_order: str = "row_major"
    comm_blocks: int = 19
    batch_tiles: int = 1
    completion_mode: str = "collective_host"

    def __post_init__(self) -> None:
        for field in ("m", "n", "k_local", "nranks", "comm_blocks", "batch_tiles"):
            value = getattr(self, field)
            if not _is_int(value) or value <= 0:
                _fail(f"{field} must be a positive integer")
        for field in (
            "compute_kind",
            "collective_kind",
            "algorithm",
            "output_layout",
            "mathematical_output_distribution",
            "owner_policy",
            "compute_tile_order",
            "completion_mode",
        ):
            if not isinstance(getattr(self, field), str):
                _fail(f"{field} must be a string")
        tile_shape = _tuple_of_ints(self.tile_shape, "tile_shape")
        if len(tile_shape) != 2:
            _fail("tile_shape must be exactly (tile_m, tile_n)")
        object.__setattr__(self, "tile_shape", tile_shape)
        for field in ("input_dtype", "accumulator_dtype", "communication_dtype", "output_dtype"):
            object.__setattr__(self, field, _canonical_dtype(getattr(self, field)))
        if self.compute_kind != "gemm":
            _fail("M0 supports compute_kind='gemm' only")
        if self.collective_kind == "allgather":
            _fail(
                "allgather is not supported by M0: it needs an input readiness protocol, "
                "not the v1 output-ready protocol"
            )
        if self.collective_kind not in {"allreduce", "reduce_scatter"}:
            _fail("collective_kind must be 'allreduce' or 'reduce_scatter'")
        if self.algorithm != "direct_push_owner_reduce_v1":
            _fail("M0 supports algorithm='direct_push_owner_reduce_v1' only")
        expected_policy = "cyclic" if self.collective_kind == "allreduce" else "contiguous_rows"
        if self.owner_policy != expected_policy:
            _fail(
                f"{self.collective_kind} requires owner_policy={expected_policy!r}; "
                "ownership is not an execution-order setting"
            )
        expected_compute_order = "row_major" if self.collective_kind == "allreduce" else "owner_interleaved"
        if self.compute_tile_order != expected_compute_order:
            _fail(
                f"{self.collective_kind} requires compute_tile_order={expected_compute_order!r}; "
                "compute execution order is separate from ownership and layout"
            )
        if self.output_layout != "tile_major":
            _fail("M0 v1 output is tile-major; row-major materialization is a later operation")
        expected_distribution = "replicated" if self.collective_kind == "allreduce" else "row_sharded"
        if self.mathematical_output_distribution != expected_distribution:
            _fail(
                f"{self.collective_kind} requires mathematical_output_distribution="
                f"{expected_distribution!r}; physical tile-major layout is separate"
            )
        if self.input_dtype != "fp16" or any(
            dtype != "fp32"
            for dtype in (self.accumulator_dtype, self.communication_dtype, self.output_dtype)
        ):
            _fail("M0 v1 requires fp16 inputs and fp32 accumulator/communication/output")
        if self.batch_tiles not in _BATCH_TILES:
            _fail("batch_tiles must be one of 1, 2, 4, or 8")
        if self.completion_mode != "collective_host":
            _fail("M0 requires completion_mode='collective_host'")

    @property
    def tile_m(self) -> int:
        return self.tile_shape[0]

    @property
    def tile_n(self) -> int:
        return self.tile_shape[1]

    @property
    def requires_edge_guards(self) -> bool:
        """Whether the pure mapping contains a short M/N edge tile."""
        return self.m % self.tile_m != 0 or self.n % self.tile_n != 0

    def validate_v1_lowering(self) -> None:
        """Reject shapes not implemented by the first HIP/GEMM lowering path.

        This is deliberately separate from construction so M0 can exercise the
        mapping and its guarded-tail metadata without pretending the later GPU
        implementation already supports partial tiles.
        """
        if self.requires_edge_guards:
            _fail(
                "v1 lowering does not yet implement padded/guarded M/N edge tiles; "
                "the M0 SchedulePlan may model them only"
            )
        if self.k_local % 32:
            _fail("v1 lowering requires K_local divisible by 32")
        if self.tile_shape != (128, 128):
            _fail("v1 lowering currently requires tile_shape=(128, 128)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "n": self.n,
            "k_local": self.k_local,
            "nranks": self.nranks,
            "tile_shape": list(self.tile_shape),
            "compute_kind": self.compute_kind,
            "collective_kind": self.collective_kind,
            "algorithm": self.algorithm,
            "input_dtype": self.input_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "communication_dtype": self.communication_dtype,
            "output_dtype": self.output_dtype,
            "output_layout": self.output_layout,
            "mathematical_output_distribution": self.mathematical_output_distribution,
            "owner_policy": self.owner_policy,
            "compute_tile_order": self.compute_tile_order,
            "comm_blocks": self.comm_blocks,
            "batch_tiles": self.batch_tiles,
            "completion_mode": self.completion_mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FusionSpec":
        return cls(
            **_strict_object(
                value,
                "FusionSpec",
                (
                    "m", "n", "k_local", "nranks", "tile_shape", "compute_kind", "collective_kind",
                    "algorithm", "input_dtype", "accumulator_dtype", "communication_dtype", "output_dtype",
                    "output_layout", "mathematical_output_distribution", "owner_policy", "compute_tile_order",
                    "comm_blocks", "batch_tiles", "completion_mode",
                ),
            )
        )

    def to_json(self) -> str:
        return _json_dump(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> "FusionSpec":
        return cls.from_dict(_parse_json(value, "FusionSpec"))


@dataclass(frozen=True)
class SymmetricBufferDesc:
    """A typed view into one symmetric allocation.

    ``alignment`` is the guaranteed **allocation-base** alignment in bytes;
    the view's effective alignment is derived from it and ``byte_offset``.
    ``strides`` are explicitly measured in **elements**, while ``byte_offset``
    and ``nbytes`` are measured in **bytes**.  ``nbytes`` is the backing
    allocation capacity, not just the view size; therefore a nonzero offset is
    validated with ``byte_offset + view_span_nbytes <= nbytes``.  Distinct
    allocation IDs are never treated as contiguous address ranges.
    """

    allocation_id: str
    dtype: str
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    alignment: int
    byte_offset: int
    nbytes: int
    symmetric: bool = True
    permissions: Tuple[str, ...] = ("read", "write")
    runtime_generation: int = 0
    stride_unit: str = "elements"

    def __post_init__(self) -> None:
        if not isinstance(self.allocation_id, str) or not self.allocation_id:
            _fail("allocation_id must be a non-empty string")
        object.__setattr__(self, "dtype", _canonical_dtype(self.dtype))
        shape = _tuple_of_ints(self.shape, "shape")
        strides = _tuple_of_ints(self.strides, "strides")
        if len(shape) != len(strides):
            _fail("shape and strides must have the same rank")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)
        if self.stride_unit != "elements":
            _fail("M0 buffer strides must declare stride_unit='elements'")
        if not _is_int(self.alignment) or self.alignment <= 0 or self.alignment & (self.alignment - 1):
            _fail("alignment must be a positive power of two in bytes")
        if self.alignment < self.itemsize:
            _fail("allocation-base alignment must be at least dtype element size")
        if not _is_int(self.byte_offset) or self.byte_offset < 0:
            _fail("byte_offset must be a non-negative byte count")
        if not _is_int(self.nbytes) or self.nbytes <= 0:
            _fail("nbytes must be a positive backing-allocation byte count")
        if not _is_int(self.runtime_generation) or self.runtime_generation < 0:
            _fail("runtime_generation must be a non-negative integer")
        try:
            permissions = tuple(self.permissions)
        except TypeError:
            _fail("permissions must be an iterable subset of ('read', 'write')")
        if not permissions or any(
            not isinstance(permission, str) or permission not in {"read", "write"}
            for permission in permissions
        ):
            _fail("permissions must be a non-empty subset of ('read', 'write')")
        if len(set(permissions)) != len(permissions):
            _fail("permissions must not contain duplicates")
        object.__setattr__(self, "permissions", permissions)
        if self.symmetric is not True:
            _fail("M0 distributed workspaces must be symmetric")
        if self.byte_offset % self.itemsize:
            _fail("byte_offset must be aligned to dtype element size")
        if self.byte_offset + self.view_span_nbytes > self.nbytes:
            _fail(
                "buffer view exceeds allocation: byte_offset + view_span_nbytes must be <= nbytes"
            )

    @property
    def itemsize(self) -> int:
        return _DTYPE_BYTES[self.dtype]

    @property
    def effective_alignment(self) -> int:
        """Guaranteed alignment of this view, accounting for its byte offset."""
        return self.alignment if self.byte_offset == 0 else gcd(self.alignment, self.byte_offset)

    @property
    def max_element_offset(self) -> int:
        return sum((extent - 1) * stride for extent, stride in zip(self.shape, self.strides))

    @property
    def view_span_nbytes(self) -> int:
        return (self.max_element_offset + 1) * self.itemsize

    def structural_dict(self, *, include_allocation_id: bool = True) -> dict[str, Any]:
        result = {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "strides": list(self.strides),
            "stride_unit": self.stride_unit,
            "alignment": self.alignment,
            "byte_offset": self.byte_offset,
            "nbytes": self.nbytes,
            "symmetric": self.symmetric,
            "permissions": list(self.permissions),
        }
        if include_allocation_id:
            result["allocation_id"] = self.allocation_id
        return result

    def to_dict(self) -> dict[str, Any]:
        result = self.structural_dict()
        result["runtime_generation"] = self.runtime_generation
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SymmetricBufferDesc":
        return cls(
            **_strict_object(
                value,
                "SymmetricBufferDesc",
                (
                    "allocation_id", "dtype", "shape", "strides", "alignment", "byte_offset", "nbytes",
                    "symmetric", "permissions", "runtime_generation", "stride_unit",
                ),
            )
        )

    def to_json(self) -> str:
        return _json_dump(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> "SymmetricBufferDesc":
        return cls.from_dict(_parse_json(value, "SymmetricBufferDesc"))


@dataclass(frozen=True)
class LaunchConfig:
    """Launch requirements for one persistent-kernel specialization."""

    grid_blocks: int
    threads_per_block: int = 256
    dynamic_smem_bytes: int = 65536

    def __post_init__(self) -> None:
        for field in ("grid_blocks", "threads_per_block"):
            value = getattr(self, field)
            if not _is_int(value) or value <= 0:
                _fail(f"{field} must be a positive integer")
        if not _is_int(self.dynamic_smem_bytes) or self.dynamic_smem_bytes < 0:
            _fail("dynamic_smem_bytes must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_blocks": self.grid_blocks,
            "threads_per_block": self.threads_per_block,
            "dynamic_smem_bytes": self.dynamic_smem_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LaunchConfig":
        return cls(**_strict_object(value, "LaunchConfig", ("grid_blocks", "threads_per_block", "dynamic_smem_bytes")))


@dataclass(frozen=True)
class WorkspaceRequirement:
    """Rank-independent symmetric workspace layout requirement."""

    role: str
    dtype: str
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    byte_offset: int = 0
    stride_unit: str = "elements"

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            _fail("workspace role must be a non-empty string")
        object.__setattr__(self, "dtype", _canonical_dtype(self.dtype))
        shape = _tuple_of_ints(self.shape, "workspace shape")
        strides = _tuple_of_ints(self.strides, "workspace strides")
        if len(shape) != len(strides):
            _fail("workspace shape and strides must have the same rank")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)
        if self.stride_unit != "elements":
            _fail("workspace strides must use elements")
        if not _is_int(self.byte_offset) or self.byte_offset < 0:
            _fail("workspace byte_offset must be non-negative")
        if self.byte_offset % _DTYPE_BYTES[self.dtype]:
            _fail("workspace byte_offset must be dtype aligned")

    @property
    def min_nbytes(self) -> int:
        max_offset = sum((extent - 1) * stride for extent, stride in zip(self.shape, self.strides))
        return self.byte_offset + (max_offset + 1) * _DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "strides": list(self.strides),
            "byte_offset": self.byte_offset,
            "stride_unit": self.stride_unit,
            "min_nbytes": self.min_nbytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkspaceRequirement":
        value = _strict_object(
            value,
            "WorkspaceRequirement",
            ("role", "dtype", "shape", "strides", "byte_offset", "stride_unit", "min_nbytes"),
        )
        advertised_min_nbytes = value.pop("min_nbytes")
        result = cls(**value)
        if not _is_int(advertised_min_nbytes) or advertised_min_nbytes != result.min_nbytes:
            _fail("WorkspaceRequirement min_nbytes must equal the derived layout span")
        return result


_REQUIRED_CAPABILITIES = frozenset(("rocshmem_ipc", "system_scope_atomics"))


def _expected_workspace_layouts(spec: FusionSpec, schedule: Any) -> Tuple[WorkspaceRequirement, ...]:
    """Derive v1 workspace contracts from the schedule, never from bindings."""
    tile_elements = spec.tile_m * spec.tile_n
    max_owned = max(schedule.owned_tile_count(rank) for rank in range(spec.nranks))
    output_tiles = schedule.tile_count if spec.collective_kind == "allreduce" else max_owned
    return (
        WorkspaceRequirement(
            "staging",
            "fp32",
            (max_owned, spec.nranks, spec.tile_m, spec.tile_n),
            (spec.nranks * tile_elements, tile_elements, spec.tile_n, 1),
        ),
        WorkspaceRequirement("output", "fp32", (output_tiles, spec.tile_m, spec.tile_n), (tile_elements, spec.tile_n, 1)),
        WorkspaceRequirement("ready", "uint32", (schedule.tile_count, spec.nranks), (spec.nranks, 1)),
    )


@dataclass(frozen=True)
class BufferBinding:
    """Binds a named kernel argument/workspace role to a symmetric view."""

    argument: str
    role: str
    buffer: SymmetricBufferDesc

    def __post_init__(self) -> None:
        if not isinstance(self.argument, str) or not self.argument:
            _fail("binding argument must be a non-empty string")
        if not isinstance(self.role, str) or not self.role:
            _fail("binding role must be a non-empty string")
        if not isinstance(self.buffer, SymmetricBufferDesc):
            _fail("binding buffer must be a SymmetricBufferDesc")

    def to_dict(self, *, for_compile_identity: bool = False, for_agreement: bool = False) -> dict[str, Any]:
        if for_compile_identity:
            buffer = self.buffer.structural_dict(include_allocation_id=False)
        elif for_agreement:
            buffer = self.buffer.structural_dict(include_allocation_id=True)
        else:
            buffer = self.buffer.to_dict()
        return {"argument": self.argument, "role": self.role, "buffer": buffer}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BufferBinding":
        value = _strict_object(value, "BufferBinding", ("argument", "role", "buffer"))
        value["buffer"] = SymmetricBufferDesc.from_dict(value["buffer"])
        return cls(**value)


@dataclass(frozen=True)
class ResetProtocol:
    """Host/device reset lifecycle required before a collective launch."""

    ready_reset: str = "zero_before_each_collective"
    local_stream_sync_before_barrier: bool = True
    pre_launch_barrier: bool = True
    workspace_reuse_after: str = "collective_host_completion"

    def __post_init__(self) -> None:
        if self.ready_reset != "zero_before_each_collective":
            _fail("M0 binary ready flags require ready_reset='zero_before_each_collective'")
        if self.local_stream_sync_before_barrier is not True or self.pre_launch_barrier is not True:
            _fail("M0 reset requires local stream synchronization and a pre-launch barrier")
        if self.workspace_reuse_after != "collective_host_completion":
            _fail("workspace reuse must wait for collective_host_completion")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_reset": self.ready_reset,
            "local_stream_sync_before_barrier": self.local_stream_sync_before_barrier,
            "pre_launch_barrier": self.pre_launch_barrier,
            "workspace_reuse_after": self.workspace_reuse_after,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResetProtocol":
        return cls(
            **_strict_object(
                value,
                "ResetProtocol",
                ("ready_reset", "local_stream_sync_before_barrier", "pre_launch_barrier", "workspace_reuse_after"),
            )
        )


@dataclass(frozen=True)
class CompletionContract:
    """The local launch event is insufficient; output needs a host barrier."""

    mode: str = "collective_host"
    local_kernel_completion_required: bool = True
    post_launch_barrier: bool = True

    def __post_init__(self) -> None:
        if self.mode != "collective_host":
            _fail("M0 completion mode must be collective_host")
        if self.local_kernel_completion_required is not True or self.post_launch_barrier is not True:
            _fail("collective_host completion requires local completion and a post-launch barrier")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "local_kernel_completion_required": self.local_kernel_completion_required,
            "post_launch_barrier": self.post_launch_barrier,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompletionContract":
        return cls(
            **_strict_object(
                value,
                "CompletionContract",
                ("mode", "local_kernel_completion_required", "post_launch_barrier"),
            )
        )


@dataclass(frozen=True)
class RuntimeBinding:
    """Per-process, non-cacheable binding data for a compiled ABI.

    Pointer values and the context handle are intentionally opaque strings in
    this CPU contract.  A later runtime module owns their native interpretation.
    """

    rank: int
    nranks: int
    context_handle: str
    allocation_pointers: Tuple[Tuple[str, str], ...]
    allocation_generations: Tuple[Tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not _is_int(self.rank) or not _is_int(self.nranks) or not 0 <= self.rank < self.nranks:
            _fail("runtime rank must satisfy 0 <= rank < nranks")
        if not isinstance(self.context_handle, str) or not self.context_handle:
            _fail("context_handle must be a non-empty opaque string")
        try:
            pointers = tuple(tuple(item) for item in self.allocation_pointers)
            generations = tuple(tuple(item) for item in self.allocation_generations)
        except TypeError:
            _fail("runtime allocation bindings must be iterable pairs")
        if any(len(item) != 2 or not isinstance(item[0], str) or not isinstance(item[1], str) for item in pointers):
            _fail("allocation_pointers must contain (allocation_id, opaque_pointer) pairs")
        if any(len(item) != 2 or not isinstance(item[0], str) or not _is_int(item[1]) or item[1] < 0 for item in generations):
            _fail("allocation_generations must contain (allocation_id, non-negative_generation) pairs")
        if len({item[0] for item in pointers}) != len(pointers) or len({item[0] for item in generations}) != len(generations):
            _fail("runtime allocation IDs must be unique")
        if {item[0] for item in pointers} != {item[0] for item in generations}:
            _fail("runtime pointers and generations must describe the same allocation IDs")
        object.__setattr__(self, "allocation_pointers", tuple(sorted(pointers)))
        object.__setattr__(self, "allocation_generations", tuple(sorted(generations)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "nranks": self.nranks,
            "context_handle": self.context_handle,
            "allocation_pointers": [list(item) for item in self.allocation_pointers],
            "allocation_generations": [list(item) for item in self.allocation_generations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeBinding":
        return cls(
            **_strict_object(
                value,
                "RuntimeBinding",
                ("rank", "nranks", "context_handle", "allocation_pointers", "allocation_generations"),
            )
        )


@dataclass(frozen=True)
class DistributedKernelABI:
    """Versioned immutable ABI manifest for a distributed kernel artifact.

    ``compile_identity`` deliberately omits runtime allocation IDs, allocation
    generations, pointers, rank, and context.  ``agreement_identity`` is the
    cross-PE configuration comparison and retains allocation IDs plus all
    meaningful layout fields, while still excluding transient pointers and
    generations.
    """

    spec: FusionSpec
    schedule: Any
    launch: LaunchConfig
    required_capabilities: Tuple[str, ...]
    workspace: Tuple[WorkspaceRequirement, ...]
    bindings: Tuple[BufferBinding, ...]
    reset: ResetProtocol = ResetProtocol()
    completion: CompletionContract = CompletionContract()
    abi_version: int = ABI_VERSION

    def __post_init__(self) -> None:
        if not _is_int(self.abi_version) or self.abi_version != ABI_VERSION:
            _fail(f"unsupported ABI version {self.abi_version}; expected {ABI_VERSION}")
        if not isinstance(self.spec, FusionSpec):
            _fail("ABI spec must be a FusionSpec")
        self.spec.validate_v1_lowering()
        # Import lazily to keep spec.py directly importable in minimal tooling.
        from .schedule import SchedulePlan

        if not isinstance(self.schedule, SchedulePlan) or self.schedule.spec != self.spec:
            _fail("ABI schedule must be a SchedulePlan built from the same FusionSpec")
        if not isinstance(self.launch, LaunchConfig):
            _fail("ABI launch must be a LaunchConfig")
        if self.launch.grid_blocks - self.spec.comm_blocks != self.schedule.compute_blocks:
            _fail("launch grid_blocks - spec.comm_blocks must equal schedule.compute_blocks")
        if self.launch.threads_per_block != 256:
            _fail("M0 v1 requires 256 threads_per_block")
        try:
            raw_capabilities = tuple(self.required_capabilities)
        except TypeError:
            _fail("required_capabilities must be an iterable of non-empty strings")
        if not raw_capabilities or any(not isinstance(capability, str) or not capability for capability in raw_capabilities):
            _fail("required_capabilities must be a non-empty set of non-empty strings")
        capabilities = tuple(sorted(set(raw_capabilities)))
        missing_capabilities = _REQUIRED_CAPABILITIES - set(capabilities)
        if missing_capabilities:
            _fail(f"direct-push ABI requires capabilities: {sorted(missing_capabilities)}")
        object.__setattr__(self, "required_capabilities", capabilities)
        try:
            workspace = tuple(self.workspace)
            bindings = tuple(self.bindings)
        except TypeError:
            _fail("workspace and bindings must be iterables")
        if not workspace or not bindings:
            _fail("ABI requires workspace layouts and symmetric argument bindings")
        if any(not isinstance(item, WorkspaceRequirement) for item in workspace):
            _fail("workspace entries must be WorkspaceRequirement instances")
        if any(not isinstance(item, BufferBinding) for item in bindings):
            _fail("bindings entries must be BufferBinding instances")
        if len({item.role for item in workspace}) != len(workspace):
            _fail("workspace roles must be unique")
        if len({item.argument for item in bindings}) != len(bindings) or len({item.role for item in bindings}) != len(bindings):
            _fail("binding arguments and roles must be unique")
        expected_workspace = _expected_workspace_layouts(self.spec, self.schedule)
        if workspace != expected_workspace:
            _fail("workspace layouts do not match the shapes/strides derived from spec and schedule")
        requirements = {item.role: item for item in expected_workspace}
        if set(requirements) != {item.role for item in bindings}:
            _fail("each schedule-derived workspace role must have exactly one symmetric argument binding")
        for binding in bindings:
            requirement = requirements[binding.role]
            buffer = binding.buffer
            if buffer.dtype != requirement.dtype or buffer.shape != requirement.shape or buffer.strides != requirement.strides:
                _fail(f"binding {binding.argument!r} does not match workspace layout for role {binding.role!r}")
            if buffer.byte_offset != requirement.byte_offset or buffer.stride_unit != requirement.stride_unit:
                _fail(f"binding {binding.argument!r} has incompatible offset or stride unit")
            if buffer.nbytes < requirement.min_nbytes:
                _fail(f"binding {binding.argument!r} is smaller than workspace requirement")
            if set(buffer.permissions) != {"read", "write"}:
                _fail(f"binding {binding.argument!r} must permit both reads and writes")
            required_alignment = 16 if binding.role in {"staging", "output"} else 4
            if buffer.effective_alignment < required_alignment:
                _fail(
                    f"binding {binding.argument!r} has effective view alignment {buffer.effective_alignment}; "
                    f"v1 requires at least {required_alignment} bytes"
                )
        allocation_ids = tuple(binding.buffer.allocation_id for binding in bindings)
        if len(set(allocation_ids)) != len(allocation_ids):
            _fail("M0 ABI v1 forbids workspace allocation aliasing; staging, output, and ready need distinct allocation_ids")
        if not isinstance(self.reset, ResetProtocol) or not isinstance(self.completion, CompletionContract):
            _fail("ABI reset and completion must use M0 protocol objects")
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "bindings", bindings)

    @classmethod
    def v1(
        cls,
        spec: FusionSpec,
        schedule: Any,
        *,
        grid_blocks: int = 80,
        threads_per_block: int = 256,
        dynamic_smem_bytes: int = 65536,
        allocation_ids: Mapping[str, str] | None = None,
        runtime_generation: int = 0,
        required_capabilities: Iterable[str] = ("rocshmem_ipc", "system_scope_atomics"),
    ) -> "DistributedKernelABI":
        """Build the v1 workspace manifest with safely rank-independent sizes."""
        from .schedule import SchedulePlan

        if not isinstance(schedule, SchedulePlan) or schedule.spec != spec:
            _fail("v1 ABI requires the SchedulePlan for spec")
        spec.validate_v1_lowering()
        ids = {"staging": "staging", "output": "output", "ready": "ready"}
        if allocation_ids is not None:
            unknown = set(allocation_ids) - set(ids)
            if unknown:
                _fail(f"unknown allocation ID roles: {sorted(unknown)}")
            ids.update(allocation_ids)
        if any(not isinstance(value, str) or not value for value in ids.values()) or len(set(ids.values())) != len(ids):
            _fail("M0 ABI v1 requires distinct non-empty staging/output/ready allocation_ids")
        workspace = _expected_workspace_layouts(spec, schedule)
        bindings = tuple(
            BufferBinding(
                argument=requirement.role,
                role=requirement.role,
                buffer=SymmetricBufferDesc(
                    allocation_id=ids[requirement.role],
                    dtype=requirement.dtype,
                    shape=requirement.shape,
                    strides=requirement.strides,
                    alignment=256,
                    byte_offset=requirement.byte_offset,
                    nbytes=requirement.min_nbytes,
                    runtime_generation=runtime_generation,
                ),
            )
            for requirement in workspace
        )
        return cls(
            spec=spec,
            schedule=schedule,
            launch=LaunchConfig(grid_blocks, threads_per_block, dynamic_smem_bytes),
            required_capabilities=tuple(required_capabilities),
            workspace=workspace,
            bindings=bindings,
        )

    def _to_dict(self, *, for_compile_identity: bool = False, for_agreement: bool = False) -> dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "spec": self.spec.to_dict(),
            "schedule": self.schedule.to_dict(),
            "launch": self.launch.to_dict(),
            "required_capabilities": list(self.required_capabilities),
            "workspace": [item.to_dict() for item in self.workspace],
            "bindings": [
                item.to_dict(for_compile_identity=for_compile_identity, for_agreement=for_agreement)
                for item in self.bindings
            ],
            "reset": self.reset.to_dict(),
            "completion": self.completion.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return self._to_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DistributedKernelABI":
        from .schedule import SchedulePlan

        value = _strict_object(
            value,
            "DistributedKernelABI",
            (
                "abi_version", "spec", "schedule", "launch", "required_capabilities", "workspace", "bindings",
                "reset", "completion",
            ),
        )
        if not isinstance(value["required_capabilities"], list):
            _fail("DistributedKernelABI required_capabilities must be a JSON array")
        if not isinstance(value["workspace"], list) or not isinstance(value["bindings"], list):
            _fail("DistributedKernelABI workspace and bindings must be JSON arrays")
        return cls(
            abi_version=value["abi_version"],
            spec=FusionSpec.from_dict(value["spec"]),
            schedule=SchedulePlan.from_dict(value["schedule"]),
            launch=LaunchConfig.from_dict(value["launch"]),
            required_capabilities=tuple(value["required_capabilities"]),
            workspace=tuple(WorkspaceRequirement.from_dict(item) for item in value["workspace"]),
            bindings=tuple(BufferBinding.from_dict(item) for item in value["bindings"]),
            reset=ResetProtocol.from_dict(value["reset"]),
            completion=CompletionContract.from_dict(value["completion"]),
        )

    def to_json(self) -> str:
        return _json_dump(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> "DistributedKernelABI":
        return cls.from_dict(_parse_json(value, "DistributedKernelABI"))

    def compile_identity(self) -> str:
        """Stable cross-process cache key; no pointers, rank, context, or generation."""
        return sha256(_json_dump(self._to_dict(for_compile_identity=True)).encode("ascii")).hexdigest()

    def agreement_identity(self) -> str:
        """Cross-PE structural agreement key, retaining allocation IDs and layouts."""
        return sha256(_json_dump(self._to_dict(for_agreement=True)).encode("ascii")).hexdigest()

    def bind_runtime(self, binding: RuntimeBinding) -> None:
        """Validate a per-process binding without mutating this immutable manifest."""
        if not isinstance(binding, RuntimeBinding):
            _fail("bind_runtime expects a RuntimeBinding")
        if binding.nranks != self.spec.nranks:
            _fail("runtime nranks disagrees with ABI FusionSpec")
        expected = {item.buffer.allocation_id: item.buffer.runtime_generation for item in self.bindings}
        supplied_generations = dict(binding.allocation_generations)
        if set(supplied_generations) != set(expected):
            _fail("runtime allocation IDs do not match ABI symmetric bindings")
        for allocation_id, generation in expected.items():
            if supplied_generations[allocation_id] != generation:
                _fail(f"runtime generation disagrees for allocation_id {allocation_id!r}")
