"""Formal DCU LDS rewind storage (hygon AT/BN + AN/BT).

Triggered by the same two access patterns as hygon async copy + ds_read:

* **AT/BN** — A not-transposed / B transposed. Physical LDS is ``[MN, K]``.
  Matches ``MakeStorageLayout`` in ``gemm_at_bn_lds_strategy.cc``.
* **AN/BT** — A transposed / B not-transposed. Physical LDS is ``[K, MN]``.
  Matches ``MakeStorageLayout`` in ``gemm_an_bt_lds_strategy.cc``.

This is one storage map, not a pipeline of base/slab/warp/iter/rewind with
per-lane predicates. Logical GEMM ``(row, col)`` maps to physical LDS through
row-permute + wave wrap (``floormod(elem + wrap_class * offset, span)``).

LDS→VGPR (``ds_read`` / ldmatrix) applies the same map to the 16×16 matrix-core
coordinates. G→LDS async copy writes ``src[logical] → dst[storage(logical)]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .mmac_layout import shared_to_local_layout_A_swizzle, shared_to_local_layout_B_swizzle

_WARP_SIZE = 64
_BANK_COUNT = 32
_BANK_WIDTH_BYTES = 4
_DS_READ_TILE_K = 16
_DS_READ_PANEL_N = 32
_DS_READ_BYTES_PER_LANE = 16
_DS_READ_WRAP_STEP_ANBT = 64
_DS_READ_WRAP_COUNT_ANBT = 2
_DS_READ_B128_WRAP_STEP = 32
_LDS_ISSUE_BYTES = 128
_WRAP_GRANULARITY_DWORDS = 8
_MAX_WRAP_OFFSET_DWORDS = 24


def _is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _copy_txn_bytes(bytes_per_row: int, copy_bytes_per_lane: int, element_bytes: int) -> int:
    if bytes_per_row <= 0 or copy_bytes_per_lane <= 0 or element_bytes <= 0:
        return 0
    for txn in (16, 8, 4):
        if (txn % element_bytes == 0 and bytes_per_row % txn == 0 and
                copy_bytes_per_lane % txn == 0):
            return txn
    return 0


def _legal_wrap(wrap_step_bytes: int, wrap_count: int, num_waves: int) -> bool:
    if not _is_pow2(wrap_count) or wrap_count > num_waves or num_waves % wrap_count != 0:
        return False
    if wrap_count == 1:
        return wrap_step_bytes == 0
    if wrap_step_bytes <= 0 or wrap_step_bytes % 4 != 0:
        return False
    off = wrap_step_bytes // 4
    if off % _WRAP_GRANULARITY_DWORDS != 0:
        return False
    return off * (wrap_count - 1) <= _MAX_WRAP_OFFSET_DWORDS


@dataclass(frozen=True)
class AtBnRewindGeom:
    """Spatial-leading rewind: LDS ``[block_mn, block_k]``."""

    block_mn: int
    block_k: int
    block_threads: int
    warp_size: int
    element_bytes: int
    copy_elements_per_lane: int
    copy_segments_per_row: int
    row_period: int
    rows_per_copy_warp_transaction: int
    rows_per_copy_transaction: int
    wrap_offset: int
    wrap_idx_mask: int

    @staticmethod
    def try_create(
        block_mn: int,
        block_k: int,
        block_threads: int,
        element_bytes: int,
        warp_mn_count: int,
        operand_mn_atom: int,
        k_pack: int = 1,
        mmac_k: int = 16,
    ):
        if (block_mn <= 0 or block_k <= 0 or block_threads <= 0 or element_bytes <= 0 or
                warp_mn_count <= 0 or operand_mn_atom <= 0 or k_pack <= 0 or mmac_k <= 0):
            return None
        if (block_threads % _WARP_SIZE != 0 or not _is_pow2(warp_mn_count) or
                block_k % mmac_k != 0 or block_mn % warp_mn_count != 0):
            return None
        packed = operand_mn_atom * mmac_k * k_pack
        if packed % _WARP_SIZE != 0:
            return None
        read_bytes = packed // _WARP_SIZE * element_bytes
        row_bytes = block_k * element_bytes
        if read_bytes <= 0 or row_bytes % read_bytes != 0:
            return None
        if read_bytes == _DS_READ_BYTES_PER_LANE:
            issue_lanes = 4
        elif read_bytes > 0 and _LDS_ISSUE_BYTES % read_bytes == 0:
            issue_lanes = _LDS_ISSUE_BYTES // read_bytes
        else:
            return None
        if issue_lanes <= 0 or _WARP_SIZE % issue_lanes != 0 or row_bytes % _BANK_WIDTH_BYTES != 0:
            return None
        row_bank_stride = row_bytes // _BANK_WIDTH_BYTES
        row_period = _BANK_COUNT // math.gcd(_BANK_COUNT, row_bank_stride)
        if row_period <= 0:
            return None
        if row_period >= issue_lanes:
            required_wrap = 1
        elif issue_lanes % row_period == 0:
            required_wrap = issue_lanes // row_period
        else:
            return None
        tile_bytes = block_mn * block_k * element_bytes
        if tile_bytes % block_threads != 0:
            return None
        copy_bytes = tile_bytes // block_threads
        txn = _copy_txn_bytes(row_bytes, copy_bytes, element_bytes)
        if txn < 4:
            return None
        segs = row_bytes // txn
        if segs <= 0 or _WARP_SIZE % segs != 0:
            return None
        copy_warps = block_threads // _WARP_SIZE
        b128 = read_bytes == _DS_READ_BYTES_PER_LANE
        wrap_count = required_wrap if b128 else max(warp_mn_count, required_wrap)
        if not _is_pow2(wrap_count) or wrap_count > copy_warps:
            return None
        if wrap_count == 1:
            wrap_step, wrap_offset = 0, 0
        elif b128:
            wrap_step, wrap_offset = _DS_READ_B128_WRAP_STEP, _DS_READ_B128_WRAP_STEP // 4
        else:
            return None
        if not _legal_wrap(wrap_step, wrap_count, copy_warps):
            return None
        rows_per_copy_warp_txn = _WARP_SIZE // segs
        if (block_mn % row_period != 0 or rows_per_copy_warp_txn % row_period != 0 or
                operand_mn_atom % row_period != 0):
            return None
        return AtBnRewindGeom(
            block_mn=block_mn,
            block_k=block_k,
            block_threads=block_threads,
            warp_size=_WARP_SIZE,
            element_bytes=element_bytes,
            copy_elements_per_lane=txn // element_bytes,
            copy_segments_per_row=segs,
            row_period=row_period,
            rows_per_copy_warp_transaction=rows_per_copy_warp_txn,
            rows_per_copy_transaction=block_threads // segs,
            wrap_offset=wrap_offset,
            wrap_idx_mask=wrap_count - 1,
        )


def at_bn_row_permute(row, geom: AtBnRewindGeom):
    if geom.wrap_idx_mask == 0:
        return row
    copy_warps = geom.block_threads // geom.warp_size
    groups_per_copy_warp = geom.rows_per_copy_warp_transaction // geom.row_period
    row_group = row // geom.row_period
    inner = row % geom.row_period
    copy_warp = row_group % copy_warps
    group_round = row_group // copy_warps
    return ((group_round // groups_per_copy_warp) * geom.rows_per_copy_transaction +
            copy_warp * geom.rows_per_copy_warp_transaction +
            (group_round % groups_per_copy_warp) * geom.row_period + inner)


def at_bn_storage_map(row, col, geom: AtBnRewindGeom):
    """Logical ``[MN, K]`` → physical LDS ``[MN, K]``."""
    mapped_row = at_bn_row_permute(row, geom)
    segment = mapped_row * geom.copy_segments_per_row + col // geom.copy_elements_per_lane
    transaction = segment // geom.block_threads
    thread = segment % geom.block_threads
    copy_warp = thread // geom.warp_size
    lane = thread % geom.warp_size
    e_per_dword = 4 // geom.element_bytes
    wrap_cur = (copy_warp % (geom.wrap_idx_mask + 1)) * geom.wrap_offset
    wave = lane * geom.copy_elements_per_lane + (col % geom.copy_elements_per_lane)
    span = geom.warp_size * geom.copy_elements_per_lane
    phys_wave = (wave + wrap_cur * e_per_dword) % span
    linear = ((transaction * geom.block_threads + copy_warp * geom.warp_size) *
              geom.copy_elements_per_lane + phys_wave)
    return linear // geom.block_k, linear % geom.block_k


@dataclass(frozen=True)
class AnBtRewindGeom:
    """K-leading rewind: LDS ``[block_k, block_mn]`` (B NT / A T)."""

    block_k: int
    block_mn: int
    block_threads: int
    warp_size: int
    element_bytes: int
    copy_elements_per_lane: int
    copy_segments_per_row: int
    read_bytes_per_lane: int
    wrap_offset: int
    wrap_idx_mask: int
    permute_rows: bool
    rows_per_copy_wave: int

    @staticmethod
    def try_create(block_k: int, block_mn: int, block_threads: int, element_bytes: int):
        if (block_k <= 0 or block_mn <= 0 or block_threads <= 0 or element_bytes != 2 or
                block_k % _DS_READ_TILE_K != 0 or block_mn % _DS_READ_PANEL_N != 0 or
                not _is_pow2(block_k) or not _is_pow2(block_mn) or
                block_threads % _WARP_SIZE != 0):
            return None
        tile_bytes = block_k * block_mn * element_bytes
        if tile_bytes % block_threads != 0:
            return None
        copy_bytes = tile_bytes // block_threads
        bytes_per_row = block_mn * element_bytes
        txn = _copy_txn_bytes(bytes_per_row, copy_bytes, element_bytes)
        if txn < 4 or _DS_READ_BYTES_PER_LANE % txn != 0:
            return None
        copy_elements = txn // element_bytes
        segs = block_mn // copy_elements
        segments_per_row = bytes_per_row // txn
        rows_per_copy_wave = _WARP_SIZE // math.gcd(segments_per_row, _WARP_SIZE)
        permute_rows = False
        wrap_offset = 0
        wrap_idx_mask = 0
        if bytes_per_row >= _BANK_COUNT * _BANK_WIDTH_BYTES:
            wrap_count = _DS_READ_WRAP_COUNT_ANBT
            if (rows_per_copy_wave * wrap_count > _DS_READ_TILE_K or
                    _DS_READ_TILE_K % (rows_per_copy_wave * wrap_count) != 0):
                return None
            if not _legal_wrap(_DS_READ_WRAP_STEP_ANBT, wrap_count, block_threads // _WARP_SIZE):
                return None
            permute_rows = True
            wrap_offset = _DS_READ_WRAP_STEP_ANBT // 4
            wrap_idx_mask = wrap_count - 1
        return AnBtRewindGeom(
            block_k=block_k,
            block_mn=block_mn,
            block_threads=block_threads,
            warp_size=_WARP_SIZE,
            element_bytes=element_bytes,
            copy_elements_per_lane=copy_elements,
            copy_segments_per_row=segs,
            read_bytes_per_lane=_DS_READ_BYTES_PER_LANE,
            wrap_offset=wrap_offset,
            wrap_idx_mask=wrap_idx_mask,
            permute_rows=permute_rows,
            rows_per_copy_wave=rows_per_copy_wave,
        )


def an_bt_copy_loop_map(k, n, geom: AnBtRewindGeom):
    k_tile = k // _DS_READ_TILE_K
    k_inner = k % _DS_READ_TILE_K
    mapped_k_inner = k_inner
    if geom.permute_rows:
        wrap_count = geom.wrap_idx_mask + 1
        row_pair = k_inner // wrap_count
        wrap_class = k_inner % wrap_count
        rp = geom.rows_per_copy_wave
        mapped_k_inner = ((row_pair // rp) * (wrap_count * rp) + wrap_class * rp +
                          (row_pair % rp))
    mapped_k = k_tile * _DS_READ_TILE_K + mapped_k_inner
    n_segment = n // geom.copy_elements_per_lane
    canonical = mapped_k * geom.copy_segments_per_row + n_segment
    transaction = canonical // geom.block_threads
    thread = canonical % geom.block_threads
    intra = n % geom.copy_elements_per_lane
    return thread, transaction, intra


def an_bt_storage_map(k, n, geom: AnBtRewindGeom):
    """Logical ``[K, MN]`` → physical LDS ``[K, MN]``."""
    read_elems = geom.read_bytes_per_lane // geom.element_bytes
    pack_start = (n // read_elems) * read_elems
    thread, transaction, intra = an_bt_copy_loop_map(k, pack_start, geom)
    copy_warp = thread // geom.warp_size
    copy_lane = thread % geom.warp_size
    e_per_dword = 4 // geom.element_bytes
    wrap_dwords = (copy_warp % (geom.wrap_idx_mask + 1)) * geom.wrap_offset
    wave = copy_lane * geom.copy_elements_per_lane + intra
    span = geom.warp_size * geom.copy_elements_per_lane
    phys_wave = (wave + wrap_dwords * e_per_dword) % span
    pack_base = ((transaction * geom.block_threads + copy_warp * geom.warp_size) *
                 geom.copy_elements_per_lane + phys_wave)
    return pack_base // geom.block_mn, pack_base % geom.block_mn + (n % read_elems)


def shared_to_local_rewind_at_bn(tx, local_id, warp_mn, idx, k_pack, mn_warps, k_offset, geom):
    """Matrix-core A / B-T coords composed with AT/BN storage (vector_load path)."""
    row, col = shared_to_local_layout_A_swizzle(tx, local_id, warp_mn, idx, k_pack, mn_warps)
    return at_bn_storage_map(row, col + k_offset, geom)


def shared_to_local_rewind_an_bt(tx, local_id, warp_n, idx, warp_num, k_offset, geom,
                                 element_per_row=32):
    """Matrix-core B-NT coords composed with AN/BT storage (async + ds_read path)."""
    row, col = shared_to_local_layout_B_swizzle(tx, local_id, warp_n, idx, warp_num, element_per_row)
    return an_bt_storage_map(row + k_offset, col, geom)
