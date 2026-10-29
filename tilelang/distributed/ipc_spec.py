"""CPU-only ABI2 IPC metadata.  It allocates nothing and has no GPU/MPI path."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import sys
from typing import Any, Iterable, Mapping, TYPE_CHECKING
from .spec import DistributedConfigError
if TYPE_CHECKING:
    from .ipc_schedule import IPCSchedulePlan

IPC_ABI_VERSION = 2
_MAX = sys.maxsize
_I32 = (1 << 31) - 1
_U32 = (1 << 32) - 1

def _fail(message: str) -> None: raise DistributedConfigError(message)
def _is_int(value: Any) -> bool: return isinstance(value, int) and not isinstance(value, bool)
def _pos(value: Any, name: str) -> int:
    if not _is_int(value) or value <= 0: _fail(f"{name} must be a positive non-bool integer")
    return value
def _mul(*values: int, name: str) -> int:
    result = 1
    for value in values:
        if not _is_int(value) or value < 0 or (value and result > _MAX // value): _fail(f"{name} exceeds size_t")
        result *= value
    return result
def _i32(value: int, name: str) -> int:
    if not _is_int(value) or value < 0 or value > _I32:
        _fail(f"{name} exceeds signed 32-bit native indexing")
    return value
def _strict(value: Any, name: str, fields: Iterable[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping): _fail(f"{name} JSON must be an object")
    got, expected = dict(value), set(fields)
    if set(got) != expected: _fail(f"{name} JSON fields differ; missing={sorted(expected-set(got))}, unknown={sorted(set(got)-expected)}")
    return got
def _dump(value: Mapping[str, Any]) -> str: return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

@dataclass(frozen=True)
class IPCMemoryRequirement:
    role: str; dtype: str; shape: tuple[int, ...]; strides: tuple[int, ...]
    layout: str; memory_kind: str; ipc_allocation: bool; peer_access: bool
    alignment: int; offset_bytes: int = 0
    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or not isinstance(self.strides, tuple):
            _fail("memory shape and strides must be immutable tuples")
        if not isinstance(self.role, str) or not self.role: _fail("memory role is invalid")
        if self.dtype not in {"fp16", "uint32"}: _fail("memory dtype must be fp16 or uint32")
        if not self.shape or len(self.shape) != len(self.strides) or any(not _is_int(v) or v <= 0 for v in self.shape + self.strides): _fail("shape/strides must be positive integer tuples")
        if self.layout not in {"row_major", "tile_major", "linear"} or self.memory_kind not in {"FineGrained", "DeviceDefault"}: _fail("invalid layout or memory kind")
        if not isinstance(self.ipc_allocation, bool) or not isinstance(self.peer_access, bool) or self.peer_access and not self.ipc_allocation: _fail("invalid IPC/peer-access declaration")
        minimum = 4 if self.dtype == "uint32" else 16
        if not _is_int(self.alignment) or self.alignment < minimum or self.alignment & (self.alignment-1) or not _is_int(self.offset_bytes) or self.offset_bytes < 0 or self.offset_bytes % self.alignment: _fail("invalid alignment or offset_bytes")
    @property
    def nbytes(self) -> int:
        # A view occupies its full strided span, not merely product(shape).
        elements = 1 + sum((extent - 1) * stride for extent, stride in zip(self.shape, self.strides))
        return _mul(elements, 4 if self.dtype == "uint32" else 2, name=f"{self.role}.nbytes")
    @property
    def required_span_bytes(self) -> int:
        if self.offset_bytes > _MAX-self.nbytes: _fail(f"{self.role} span exceeds size_t")
        return self.offset_bytes+self.nbytes

@dataclass(frozen=True)
class IPCFusionSpec:
    """Per-PE GEMM dimensions for a fixed 4-PE reference profile, not GPU support."""
    profile: str = "ar_v2"; m: int = 4096; n: int = 4096; k: int = 1024
    nranks: int = 4; batch_tiles: int = 1; comm_blocks: int = 24; cu_count: int = 80
    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or self.profile not in {"ar_v2", "rs_v2", "ag_v2"}: _fail("profile must be ar_v2, rs_v2, or ag_v2")
        for field in ("m","n","k","nranks","batch_tiles","comm_blocks","cu_count"): _pos(getattr(self,field),field)
        for field in ("m", "n", "k"):
            _i32(getattr(self, field), field)
        if self.nranks != 4 or self.batch_tiles != 1: _fail("ABI2 first profile requires exactly 4 PE and batch_tiles=1")
        if self.cu_count > 1024 or self.comm_blocks >= self.cu_count or self.comm_blocks > 1023: _fail("require 0 < comm_blocks < cu_count <= 1024")
        if self.m % 256 or self.n % 256 or self.k % 64: _fail("reference requires M/N whole tiles and K divisible by 64")
        if self.profile == "ar_v2" and self.tile_count % 4: _fail("AR requires tile_count divisible by 4")
        if self.profile in {"rs_v2","ag_v2"} and self.m % 1024: _fail("RS/AG require M divisible by 1024")
        if self.profile == "ag_v2" and (self.k < 256 or self.k % 256): _fail("AG requires K >=256 and divisible by 256")
        if any(value > _I32 for value in self.global_shape): _fail("global dimension exceeds signed 32-bit native indexing")
        if self.tile_count > _I32 or self.flags_count > _U32 or self.tile_count + self.cu_count > _I32: _fail("reference index/count overflow")
        if self.profile == "ag_v2" and self.tile_rows * (256 * self.k // 65536) > _I32: _fail("AG communication job count exceeds int32")
        for requirement in self.memory_requirements():
            if requirement.nbytes > _I32: _fail(f"{requirement.role} element span exceeds signed 32-bit native indexing")
    @property
    def tile_rows(self) -> int: return self.m//256
    @property
    def tiles_x(self) -> int: return self.n//256
    @property
    def tile_count(self) -> int: return self.tile_rows*self.tiles_x
    @property
    def compute_blocks(self) -> int: return self.cu_count-self.comm_blocks
    @property
    def flags_payload_count(self) -> int: return self.tile_rows if self.profile == "ag_v2" else self.tile_count*4
    @property
    def flags_count(self) -> int: return self.flags_payload_count+4+1024
    @property
    def global_shape(self) -> tuple[int,int,int]: return (self.m, self.n*4, self.k) if self.profile == "ag_v2" else (self.m,self.n,self.k*4)
    @property
    def local_input_shapes(self) -> Mapping[str,tuple[int,int]]:
        return {"A":(self.m//4,self.k),"B":(self.k,self.n)} if self.profile == "ag_v2" else {"A":(self.m,self.k),"B":(self.k,self.n)}
    @property
    def local_output_shape(self) -> tuple[int,int]: return (self.m//4,self.n) if self.profile == "rs_v2" else (self.m,self.n)
    @property
    def precision_contract(self) -> Mapping[str,str]:
        if self.profile == "ag_v2":
            return {"inputs":"fp16","partial":"none","output":"fp16","gemm_accumulator":"fp32","reduce_accumulator":"none","math":"A @ B (no transpose)","rounding":"FP32 GEMM rounds only at row-major output store"}
        return {"inputs":"fp16","partial":"fp16","output":"fp16","gemm_accumulator":"fp32","reduce_accumulator":"fp32","math":"A @ B (no transpose); source PE 0..3 sum in FP32","rounding":"FP32 GEMM rounds to tile-major partial; owner sums four FP16 partials in FP32 then rounds row-major output"}
    @property
    def completion_contract(self) -> Mapping[str,str]:
        return {"output_ready":"local kernel/stream completion after device handshake; local output usable","no_more_remote_readers":"all-PE communication completion handshake proves no peer will further read/write this PE's round payload; AG may still have local compute","flags_reusable":"all PE local completion then host collective, reset, second host collective, then launch; flags may still be polled before that boundary","reuse":"kernel return alone never permits reset; no epoch/ring reuse","ag_counter":"future M1/M2 SYSTEM ACQ_REL RMW; relaxed local queue only allocates jobs","launch":"full-resident exclusive launch; occupancy runtime check pending; gfx936 maxthreads256/launch512 is special only"}
    @classmethod
    def reference_shape(cls, profile: str, s: int, *, comm_blocks: int) -> "IPCFusionSpec":
        return cls(profile,s,s//4,s,comm_blocks=comm_blocks) if profile == "ag_v2" else cls(profile,s,s,s//4,comm_blocks=comm_blocks)
    def memory_requirements(self) -> tuple[IPCMemoryRequirement,...]:
        row=lambda role,shape,kind,ipc,peer: IPCMemoryRequirement(role,"fp16",shape,(shape[1],1),"row_major",kind,ipc,peer,16)
        partial=IPCMemoryRequirement("localpartial","fp16",(self.tile_count,256,256),(65536,256,1),"tile_major","FineGrained",True,True,16)
        flags=IPCMemoryRequirement("flags","uint32",(self.flags_count,),(1,),"linear","FineGrained",True,True,4)
        if self.profile == "ar_v2": return (row("A",self.local_input_shapes["A"],"DeviceDefault",False,False),row("B",self.local_input_shapes["B"],"DeviceDefault",False,False),partial,row("output",(self.m,self.n),"FineGrained",True,True),flags)
        if self.profile == "rs_v2": return (row("A",self.local_input_shapes["A"],"DeviceDefault",False,False),row("B",self.local_input_shapes["B"],"DeviceDefault",False,False),partial,row("output",self.local_output_shape,"DeviceDefault",False,False),flags)
        return (row("sharedinput",self.local_input_shapes["A"],"FineGrained",True,True),row("B",self.local_input_shapes["B"],"DeviceDefault",False,False),row("gatheredA",(self.m,self.k),"DeviceDefault",True,False),row("output",self.local_output_shape,"DeviceDefault",False,False),flags)
    def to_dict(self) -> dict[str,Any]: return {"profile":self.profile,"m":self.m,"n":self.n,"k":self.k,"nranks":self.nranks,"batch_tiles":self.batch_tiles,"comm_blocks":self.comm_blocks,"cu_count":self.cu_count}
    @classmethod
    def from_dict(cls,value:Mapping[str,Any])->"IPCFusionSpec": return cls(**_strict(value,"IPCFusionSpec",("profile","m","n","k","nranks","batch_tiles","comm_blocks","cu_count")))
    def to_json(self)->str:return _dump(self.to_dict())
    @classmethod
    def from_json(cls,value:str)->"IPCFusionSpec":
        if not isinstance(value,str): _fail("IPCFusionSpec JSON must be string")
        try:return cls.from_dict(json.loads(value))
        except (TypeError,ValueError) as error:_fail(f"invalid IPCFusionSpec JSON: {error}")

@dataclass(frozen=True)
class IPCDeviceBinding:
    """CPU metadata only; local_base and peer base pointers are intentionally absent."""
    role:str; dtype:str; shape:tuple[int,...]; strides:tuple[int,...]; layout:str; memory_kind:str
    capacity_bytes:int; offset_bytes:int; alignment:int; ipc_allocation:bool; allocation_id:int; generation:int; context_id:str; rank:int
    def __post_init__(self)->None:
        IPCMemoryRequirement(self.role,self.dtype,self.shape,self.strides,self.layout,self.memory_kind,self.ipc_allocation,False,self.alignment,self.offset_bytes)
        for field in ("capacity_bytes","allocation_id","generation"):_pos(getattr(self,field),field)
        if not isinstance(self.context_id,str) or not self.context_id:_fail("context_id must be non-empty string")
        if not _is_int(self.rank) or not 0 <= self.rank < 4:_fail("rank must be in [0,4)")

@dataclass(frozen=True)
class IPCKernelABI:
    spec:IPCFusionSpec; schedule:"IPCSchedulePlan"; abi_version:int=IPC_ABI_VERSION
    def __post_init__(self)->None:
        if not _is_int(self.abi_version) or self.abi_version != IPC_ABI_VERSION:_fail("unsupported IPC ABI version")
        from .ipc_schedule import IPCSchedulePlan
        if not isinstance(self.spec,IPCFusionSpec) or not isinstance(self.schedule,IPCSchedulePlan) or self.schedule.spec != self.spec:_fail("ABI spec/schedule mismatch")
    @property
    def requirements(self)->tuple[IPCMemoryRequirement,...]:return self.spec.memory_requirements()
    @property
    def required_roles(self)->frozenset[str]:return frozenset(r.role for r in self.requirements)
    @property
    def device_view_fields(self)->tuple[str,...]:return ("local_base","peer_bases_device","nbytes","my_pe","npes","allocation_id","generation")
    def _identity(self,tag:str)->str:return sha256(_dump({"tag":tag,"abi":self.abi_version,"spec":self.spec.to_dict()}).encode()).hexdigest()
    @property
    def compile_identity(self)->str:return self._identity("compile")
    @property
    def agreement_identity(self)->str:return self._identity("agreement")
    def validate_bindings(self,bindings:Iterable[IPCDeviceBinding], *, expected_generation:int|None=None, expected_context:str|None=None, expected_rank:int|None=None)->None:
        items=tuple(bindings)
        if any(not isinstance(x,IPCDeviceBinding) for x in items):_fail("bindings must contain IPCDeviceBinding")
        if {x.role for x in items} != self.required_roles or len({x.role for x in items}) != len(items):_fail("bindings require exact unique role set")
        if len({x.allocation_id for x in items}) != len(items):_fail("bindings require unique allocation_id")
        if len({x.generation for x in items}) != 1 or len({x.context_id for x in items}) != 1 or len({x.rank for x in items}) != 1:_fail("bindings require one generation, context and rank")
        if expected_generation is not None and items[0].generation != expected_generation:_fail("binding generation mismatches expectation")
        if expected_context is not None and items[0].context_id != expected_context:_fail("binding context mismatches expectation")
        if expected_rank is not None and items[0].rank != expected_rank:_fail("binding rank mismatches expectation")
        expected={x.role:x for x in self.requirements}
        for item in items:
            req=expected[item.role]
            if (item.dtype,item.shape,item.strides,item.layout,item.memory_kind,item.ipc_allocation)!=(req.dtype,req.shape,req.strides,req.layout,req.memory_kind,req.ipc_allocation):_fail(f"binding {item.role} disagrees with requirement")
            if item.alignment < req.alignment or item.offset_bytes%req.alignment or item.offset_bytes+req.nbytes>item.capacity_bytes:_fail(f"binding {item.role} has invalid offset/span")
    def manifest(self)->dict[str,Any]:
        return {"launch":{"threads":512,"block_k":16,"stages":4,"dynamic_lds_bytes":65536,"full_resident":True,"gfx936_max_threads_256_launch_512":True},"precision":dict(self.spec.precision_contract),"completion":dict(self.spec.completion_contract),"device_view":{"fields":["local_base","peer_bases_device","nbytes","my_pe","npes","allocation_id","generation"],"offset_bytes":"allocation-relative bytes; no fixed struct sizeof"},"memory":[{"role":r.role,"dtype":r.dtype,"shape":list(r.shape),"strides":list(r.strides),"layout":r.layout,"memory_kind":r.memory_kind,"ipc_allocation":r.ipc_allocation,"peer_access":r.peer_access,"alignment":r.alignment,"offset_bytes":r.offset_bytes,"nbytes":r.nbytes} for r in self.requirements]}
    def to_dict(self)->dict[str,Any]:return {"abi_version":self.abi_version,"spec":self.spec.to_dict(),"manifest":self.manifest()}
    @classmethod
    def from_dict(cls,value:Mapping[str,Any])->"IPCKernelABI":
        data=_strict(value,"IPCKernelABI",("abi_version","spec","manifest")); spec=IPCFusionSpec.from_dict(data["spec"])
        from .ipc_schedule import IPCSchedulePlan
        result=cls(spec,IPCSchedulePlan(spec),data["abi_version"])
        if data["manifest"] != result.manifest():_fail("IPCKernelABI manifest contains altered derived metadata")
        return result
    def to_json(self)->str:return _dump(self.to_dict())
    @classmethod
    def from_json(cls,value:str)->"IPCKernelABI":
        if not isinstance(value,str):_fail("IPCKernelABI JSON must be string")
        try:return cls.from_dict(json.loads(value))
        except (TypeError,ValueError) as error:_fail(f"invalid IPCKernelABI JSON: {error}")
