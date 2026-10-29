"""Pure-Python geometry contract for generalized DCU fusion examples.

Logical tensors may be arbitrary positive rectangles.  The accompanying DSL
generators operate only on the physical tensors described here: callers pack
logical inputs with zero padding beforehand and crop/unpack outputs afterward.
This keeps GPU setup and result ownership outside this source-only module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FusionKind = Literal["ag", "rs", "ar"]
WeightLayout = Literal["nk", "kn"]

TILE_M = 256
TILE_N = 256
BLOCK_K = 16
THREADS = 512
TOTAL_BLOCKS = 80
DEFAULT_COMM_BLOCKS = 32
CHUNK_ELEMENTS = 65536


def ceildiv(value: int, divisor: int) -> int:
    """Return ``ceil(value / divisor)`` for positive integer geometry."""
    if value <= 0 or divisor <= 0:
        raise ValueError("ceildiv expects positive integers")
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class FusionPlan:
    """Compile-time physical shape and fixed-grid schedule contract.

    ``weight_layout='nk'`` denotes the user-facing [N, K] weight.  It must be
    transposed and zero-padded into ``packed_weight_shape == [K_p, N_p]``
    before it is supplied to a DSL generator.  ``'kn'`` denotes that this
    packed orientation is already available (it may still need zero padding).
    """

    kind: FusionKind
    m: int
    n: int
    k: int
    world_size: int
    dtype: str
    weight_layout: WeightLayout
    total_blocks: int
    comm_blocks: int
    m_physical: int
    n_physical: int
    k_physical: int
    rows_per_rank_physical: int
    tile_rows: int
    tile_cols: int
    tile_count: int
    tiles_per_rank: int
    chunks_per_row_tile: int
    chunks_per_source: int

    @property
    def compute_blocks(self) -> int:
        return self.total_blocks - self.comm_blocks

    @property
    def packed_weight_shape(self) -> tuple[int, int]:
        """Physical TileLang GEMM B shape, always [K_p, N_p]."""
        return (self.k_physical, self.n_physical)

    @property
    def needs_weight_transpose(self) -> bool:
        return self.weight_layout == "nk"

    @property
    def logical_weight_shape(self) -> tuple[int, int]:
        return (self.n, self.k) if self.weight_layout == "nk" else (self.k, self.n)

    @property
    def ar_rs_packed_a_shape(self) -> tuple[int, int]:
        return (self.m_physical, self.k_physical)

    @property
    def ag_shard_shape(self) -> tuple[int, int]:
        return (self.rows_per_rank_physical, self.k_physical)

    @property
    def ag_gathered_a_shape(self) -> tuple[int, int]:
        return (self.m_physical, self.k_physical)

    @property
    def ar_output_shape(self) -> tuple[int, int]:
        return (self.m_physical, self.n_physical)

    @property
    def rs_output_shape(self) -> tuple[int, int]:
        return (self.rows_per_rank_physical, self.n_physical)

    @property
    def ag_output_shape(self) -> tuple[int, int]:
        return (self.m_physical, self.n_physical)

    @property
    def partial_shape(self) -> tuple[int, int, int]:
        return (self.tile_count, TILE_M, TILE_N)

    @property
    def flag_words(self) -> int:
        if self.kind == "ag":
            return self.total_chunks + self.world_size + self.comm_blocks
        return self.tile_count * self.world_size + self.world_size + self.comm_blocks

    @property
    def tile_ready_words(self) -> int:
        return self.tile_count * self.world_size

    @property
    def peer_done_base(self) -> int:
        return self.total_chunks if self.kind == "ag" else self.tile_ready_words

    @property
    def block_done_base(self) -> int:
        return self.peer_done_base + self.world_size

    @property
    def total_chunks(self) -> int:
        return self.chunks_per_source * self.world_size

    def logical_rows_for_rank(self, rank: int) -> tuple[int, int]:
        """The required floor-partitioned logical row interval for ``rank``."""
        self._check_rank(rank)
        return (self.m * rank // self.world_size,
                self.m * (rank + 1) // self.world_size)

    def logical_row_to_physical(self, row: int) -> tuple[int, int]:
        """Map a logical row to ``(owner, packed-row)`` in equal physical slabs."""
        if not 0 <= row < self.m:
            raise ValueError("logical row is outside [0, M)")
        # Binary search is unnecessary for P=4/8 and this formula respects
        # the specified floor boundaries even when M is not divisible by P.
        owner = next(r for r in range(self.world_size)
                     if row < self.m * (r + 1) // self.world_size)
        begin, _ = self.logical_rows_for_rank(owner)
        return owner, owner * self.rows_per_rank_physical + row - begin

    def physical_row_to_logical(self, physical_row: int) -> int | None:
        """Inverse row mapping; return ``None`` for a zero-padding row."""
        if not 0 <= physical_row < self.m_physical:
            raise ValueError("physical row is outside [0, M_p)")
        owner, local = divmod(physical_row, self.rows_per_rank_physical)
        begin, end = self.logical_rows_for_rank(owner)
        return begin + local if local < end - begin else None

    def _check_rank(self, rank: int) -> None:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size})")


def make_plan(kind: FusionKind, m: int, n: int, k: int, *,
              world_size: int = 4, dtype: str = "float16",
              weight_layout: WeightLayout = "nk", total_blocks: int = TOTAL_BLOCKS,
              comm_blocks: int = DEFAULT_COMM_BLOCKS) -> FusionPlan:
    """Validate a generalized fusion request and derive its physical geometry."""
    if kind not in ("ag", "rs", "ar"):
        raise ValueError("kind must be one of 'ag', 'rs', or 'ar'")
    if world_size not in (4, 8):
        raise ValueError("world_size must be 4 or 8")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("dtype must be 'float16' or 'bfloat16'")
    if weight_layout not in ("nk", "kn"):
        raise ValueError("weight_layout must be 'nk' or 'kn'")
    if min(m, n, k) <= 0:
        raise ValueError("logical M, N, and K must all be positive")
    if total_blocks != TOTAL_BLOCKS:
        raise ValueError("this fixed-resident prototype requires exactly 80 CTAs")
    if not 0 < comm_blocks < total_blocks:
        raise ValueError("comm_blocks must be in [1, 79]")

    rows_per_rank_physical = ceildiv(ceildiv(m, world_size), TILE_M) * TILE_M
    m_physical = world_size * rows_per_rank_physical
    n_physical = ceildiv(n, TILE_N) * TILE_N
    k_multiple = TILE_M if kind == "ag" else 64
    # AR/RS use a four-stage BLOCK_K=16 pipeline.  A 128-wide physical K
    # leaves an innermost main K loop after pipeline setup, which is required
    # by the persistent DCU tile lowering.  AG already has a 256-wide tile
    # granularity for its fixed 65536-element communication chunks.
    minimum_k = TILE_M if kind == "ag" else 128
    k_physical = max(minimum_k, ceildiv(k, k_multiple) * k_multiple)
    tile_rows = m_physical // TILE_M
    tile_cols = n_physical // TILE_N
    tile_count = tile_rows * tile_cols
    chunks_per_row_tile = k_physical // TILE_M if kind == "ag" else 0
    chunks_per_source = (rows_per_rank_physical // TILE_M) * chunks_per_row_tile
    return FusionPlan(
        kind=kind, m=m, n=n, k=k, world_size=world_size, dtype=dtype,
        weight_layout=weight_layout, total_blocks=total_blocks, comm_blocks=comm_blocks,
        m_physical=m_physical, n_physical=n_physical, k_physical=k_physical,
        rows_per_rank_physical=rows_per_rank_physical, tile_rows=tile_rows,
        tile_cols=tile_cols, tile_count=tile_count,
        tiles_per_rank=tile_count // world_size,
        chunks_per_row_tile=chunks_per_row_tile, chunks_per_source=chunks_per_source,
    )
