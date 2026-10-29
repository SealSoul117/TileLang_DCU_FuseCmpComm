"""CPU-only AR/RS/AG ABI2 formulas; no atomic or CTA ownership is implemented."""
from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Any, Mapping
from .ipc_spec import IPCFusionSpec, _is_int, _strict, _fail

@dataclass(frozen=True)
class IPCTileView:
    tile_id:int; row:int; col:int; offset_bytes:int; row_stride_bytes:int
@dataclass(frozen=True)
class IPCAGTransfer:
    rank:int; job:int; row:int; chunk:int; source:int; src_offset_bytes:int; dst_offset_bytes:int; counter_index:int; threshold:int
@dataclass(frozen=True)
class IPCAGQueueClaim:
    """Claim mapping only: dynamic CTAs are not assigned by this CPU metadata."""
    rank:int; claim_index:int; row:int; col:int

@dataclass(frozen=True)
class IPCSchedulePlan:
    spec: IPCFusionSpec
    def __post_init__(self)->None:
        if not isinstance(self.spec,IPCFusionSpec):_fail("IPCSchedulePlan requires IPCFusionSpec")
    @property
    def compute_blocks(self)->int:return self.spec.compute_blocks
    @property
    def comm_blocks(self)->int:return self.spec.comm_blocks
    @property
    def tiles_x(self)->int:return self.spec.tiles_x
    @property
    def tile_rows(self)->int:return self.spec.tile_rows
    @property
    def tile_count(self)->int:return self.spec.tile_count
    def _rank(self,rank:int)->None:
        if not _is_int(rank) or not 0<=rank<4:_fail("rank must be in [0,4)")
    def _tile(self,tile:int)->None:
        if not _is_int(tile) or not 0<=tile<self.tile_count:_fail("tile out of range")
    def _ag(self)->None:
        if self.spec.profile!="ag_v2":_fail("operation is AG-only")
    def tile_coords(self,tile:int)->tuple[int,int]:
        self._tile(tile); return tile%self.tiles_x,tile//self.tiles_x
    def coords_tile(self,col:int,row:int)->int:
        if not _is_int(col) or not 0<=col<self.tiles_x or not _is_int(row) or not 0<=row<self.tile_rows:_fail("invalid tile coordinate")
        return row*self.tiles_x+col
    def owner(self,tile:int)->int:
        if self.spec.profile=="ag_v2":_fail("AG has no output owner mapping")
        self._tile(tile)
        return tile%4 if self.spec.profile=="ar_v2" else tile//(self.tile_count//4)
    def owned_index(self,tile:int)->int:
        if self.spec.profile=="ag_v2":_fail("AG has no owned partial mapping")
        self._tile(tile)
        return tile//4 if self.spec.profile=="ar_v2" else tile%(self.tile_count//4)
    def tile_from_owned(self,owner:int,owned:int)->int:
        if self.spec.profile=="ag_v2":_fail("AG has no owned partial mapping")
        self._rank(owner)
        if not _is_int(owned) or not 0<=owned<self.tile_count//4:_fail("invalid owned index")
        return owned*4+owner if self.spec.profile=="ar_v2" else owner*(self.tile_count//4)+owned
    def producer_tile(self,job:int)->int:
        if self.spec.profile=="ag_v2":_fail("AG has no static producer tile mapping")
        if not _is_int(job) or not 0<=job<self.tile_count:_fail("invalid producer job")
        return (job%4)*(self.tile_count//4)+job//4 if self.spec.profile=="rs_v2" else job
    def producer_job(self,tile:int)->int:
        if self.spec.profile=="ag_v2":_fail("AG has no static producer tile mapping")
        self._tile(tile)
        return (tile%(self.tile_count//4))*4+self.owner(tile) if self.spec.profile=="rs_v2" else tile
    def ready_index(self,tile:int,source:int)->int:
        if self.spec.profile=="ag_v2":_fail("AG ready is a row counter, not tile/source")
        self._tile(tile);self._rank(source);return tile*4+source
    def partial_offset_bytes(self,tile:int)->int:
        if self.spec.profile=="ag_v2":_fail("AG has no localpartial allocation")
        self._tile(tile);return tile*65536*2
    def output_tile_view(self,tile:int)->IPCTileView:
        self._tile(tile); col,row=self.tile_coords(tile)
        if self.spec.profile=="rs_v2":row-=self.owner(tile)*(self.spec.m//4//256)
        return IPCTileView(tile,row,col,(row*256*self.spec.n+col*256)*2,self.spec.n*2)
    def compute_job(self,block:int,iteration:int)->int|None:
        if self.spec.profile=="ag_v2":_fail("AG uses dynamic queue claims, not static compute jobs")
        if not _is_int(block) or not 0<=block<self.compute_blocks or not _is_int(iteration) or iteration<0:_fail("invalid compute lane/iteration")
        job=block+iteration*self.compute_blocks;return job if job<self.tile_count else None
    def comm_owned(self,lane:int,iteration:int)->int|None:
        if self.spec.profile=="ag_v2":_fail("AG communication uses row/chunk jobs, not owned tiles")
        if not _is_int(lane) or not 0<=lane<self.comm_blocks or not _is_int(iteration) or iteration<0:_fail("invalid comm lane/iteration")
        owned=lane+iteration*self.comm_blocks;return owned if owned<self.tile_count//4 else None
    @property
    def flags_payload_count(self)->int:return self.spec.flags_payload_count
    def pe_done_index(self,pe:int)->int:self._rank(pe);return self.flags_payload_count+pe
    def comm_done_index(self,lane:int)->int:
        if not _is_int(lane) or not 0<=lane<self.comm_blocks:_fail("invalid comm lane")
        return self.flags_payload_count+4+lane
    @property
    def flags_count(self)->int:return self.spec.flags_count
    @property
    def ag_queue_index(self)->int:self._ag();return self.flags_payload_count+4+1023
    @property
    def ag_rows_per_rank(self)->int:self._ag();return self.tile_rows//4
    @property
    def ag_chunks_per_row(self)->int:self._ag();return 256*self.spec.k//65536
    def ag_transfer(self,rank:int,job:int)->IPCAGTransfer:
        self._ag();self._rank(rank);limit=self.tile_rows*self.ag_chunks_per_row
        if not _is_int(job) or not 0<=job<limit:_fail("invalid AG communication job")
        row=(job//self.ag_chunks_per_row+rank*self.ag_rows_per_rank)%self.tile_rows;chunk=job%self.ag_chunks_per_row
        source=row//self.ag_rows_per_rank
        return IPCAGTransfer(rank,job,row,chunk,source,((row%self.ag_rows_per_rank)*256*self.spec.k+chunk*65536)*2,(row*256*self.spec.k+chunk*65536)*2,row,self.ag_chunks_per_row)
    def ag_transfer_job(self,rank:int,row:int,chunk:int)->int:
        self._ag();self._rank(rank)
        if not _is_int(row) or not 0<=row<self.tile_rows or not _is_int(chunk) or not 0<=chunk<self.ag_chunks_per_row:_fail("invalid AG row/chunk")
        return ((row-rank*self.ag_rows_per_rank)%self.tile_rows)*self.ag_chunks_per_row+chunk
    def ag_queue_claim(self,rank:int,claim_index:int)->IPCAGQueueClaim:
        self._ag();self._rank(rank);limit=self.tile_rows*self.tiles_x
        if not _is_int(claim_index) or not 0<=claim_index<limit:_fail("invalid AG claim")
        return IPCAGQueueClaim(rank,claim_index,(claim_index//self.tiles_x+rank*self.ag_rows_per_rank)%self.tile_rows,claim_index%self.tiles_x)
    def ag_queue_claim_index(self,rank:int,row:int,col:int)->int:
        self._ag();self._rank(rank)
        if not _is_int(row) or not 0<=row<self.tile_rows or not _is_int(col) or not 0<=col<self.tiles_x:_fail("invalid AG queue coordinate")
        return ((row-rank*self.ag_rows_per_rank)%self.tile_rows)*self.tiles_x+col
    @property
    def ag_role_join_semantics(self)->str:
        self._ag();return "Claim sequence maps jobs only; it assigns no fixed CTA. Communication roles join compute after gather and reader completion."
    def to_dict(self)->dict[str,Any]:return {"spec":self.spec.to_dict()}
    @classmethod
    def from_dict(cls,value:Mapping[str,Any])->"IPCSchedulePlan":return cls(IPCFusionSpec.from_dict(_strict(value,"IPCSchedulePlan",("spec",))["spec"]))
    def to_json(self)->str:return json.dumps(self.to_dict(),sort_keys=True,separators=(",",":"),ensure_ascii=True)
    @classmethod
    def from_json(cls,value:str)->"IPCSchedulePlan":
        if not isinstance(value,str):_fail("IPCSchedulePlan JSON must be string")
        try:return cls.from_dict(json.loads(value))
        except (TypeError,ValueError) as error:_fail(f"invalid IPCSchedulePlan JSON: {error}")
