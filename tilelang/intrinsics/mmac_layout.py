# from tvm import DataType
# from tvm.runtime import convert
import tilelang.language as T


from dataclasses import dataclass

@dataclass(frozen=True)
class SharedToLocalKDim32LayoutA:
    """
    LDS memory access data layout for A matrix with k_dim=16.
    """

    @staticmethod
    def base(tx, local_id):
        row = tx % 16
        col = (tx // 16) * 8 + local_id
        return row, col

    @staticmethod
    def slab(row, col):
        slab_num = 1024
        row_num = 32
        new_row = (row % 4) + (row // 4) * (slab_num // (row_num))
        return new_row, col

    @staticmethod
    def warp(row, col, warp_m):
        new_row = row + warp_m * 4
        return new_row, col

    @staticmethod
    def itera(row, col, idx, block_row_warps):
        slab_size = 1024
        per_warp_itera_num = 4 * 32
        per_warp_itera_row = 4
        slab_4_row = slab_size * 4 // 32
        intra_iter = slab_size // (per_warp_itera_num * block_row_warps)

        new_row = (
            row
            + (idx % intra_iter) * per_warp_itera_row * block_row_warps
            + (idx // intra_iter) * slab_4_row
        )
        return new_row, col
    
    def rewind(row, col, tx, warp_m, idx, block_row_warps):
        slab_size = 1024
        k_dim = 32
        per_slab_row = slab_size // k_dim
        slab_4_row = per_slab_row * 4
        per_warp_itera_num = 4 * k_dim
        intra_iter = slab_size // (per_warp_itera_num * block_row_warps)
        new_col = (((row % slab_4_row) // per_slab_row) * 16 + col) % k_dim
        new_row = (((row % slab_4_row) // per_slab_row) * 16 + col) // k_dim + row
        select_thread = (((tx // 16)) + ((tx & 3) * 4) + (((tx & 15) // 4) * 2)) // 16
        cond = select_thread & (warp_m == (block_row_warps - 1)) & ((idx % intra_iter) == (intra_iter - 1))
        new_row = new_row - per_slab_row * cond
        return new_row, new_col

@dataclass(frozen=True)
class SharedToLocalKDim32LayoutB:
    """
    LDS memory access data layout for B matrix with k_dim=16.
    """

    @staticmethod
    def base(tx, local_id):
        row = tx // 2
        col = (tx & 1) * 16 + local_id
        return row, col
    
    @staticmethod
    def slab(row, col):
        new_row = (row & 15) // 4 + ((row & 3)) * 4 + (row // 16) * 16
        return new_row, col

    @staticmethod
    def warp(row, col, warp_n):
        per_warp_itera_num = 2 * 16
        new_col = col + warp_n * per_warp_itera_num
        return row, new_col

    @staticmethod
    def itera(row, col, idx, block_col_warps):
        per_warp_itera_num = 2 * 16
        new_col = col + per_warp_itera_num * block_col_warps * idx
        return row, new_col
    
    def rewind(row, col, tx, warp_n, idx, warp_cols, block_row_warps):
        per_warp_itera_num = 2 * 16
        col_size = 256# warp_cols * block_row_warps * per_warp_itera_num // 2
        col = (((tx & 7) // 2) * 32 + col)
        new_col = col % col_size
        new_row = row + (col // col_size)
        # row_limit = row // 4 * 4 + 4
        cond1 = (((tx & 31) == 26) | ((tx & 31) == 27)) & (warp_n == (block_row_warps - 1)) & (idx == ((warp_cols // 2) - 1))
        cond2 = (((tx & 31) == 28) | ((tx & 31) == 29)) & (idx == ((warp_cols // 2) - 1))
        cond3 = (((tx & 31) == 30) | ((tx & 31) == 31)) & (((warp_n == (block_row_warps - 1)) & (idx == ((warp_cols // 2) - 2))) | (idx == ((warp_cols // 2) - 1)))
        
        # (warp_n == (block_row_warps - 1)) & (idx == ((warp_cols // 2) - 1)) & (((tx & 31) // 8) == 3) & ((tx & 3) // 2)
        new_row = new_row - 4 * (cond1 | cond2 | cond3)
        
        return new_row, new_col

class LayoutConfig():
    def __init__(self, k_pack, warp_num, iter_num, dtype, is_B=False):
        self.micro_m = 16
        self.micro_n = 16
        self.k_pack = k_pack
        self.warp_num = warp_num
        self.iter_num = iter_num
        self.slab_size = 1024
        self.bank_size = 128
        self.infer_type_size(dtype)
        self.slab_num = self.slab_size // self.type_size
        self.bank_num = self.bank_size // self.type_size
        self.intra_iter_num = self.slab_num * 4 // (self.warp_num * self.micro_m * self.micro_n * self.k_pack)
        self.row_per_4slab = self.slab_num * 4 // (self.micro_n * self.k_pack)
        self.row_per_slab = self.slab_num // (self.micro_n * self.k_pack)
        self.slabs = self.micro_m * self.micro_n * self.k_pack * self.iter_num * self.warp_num // self.slab_num
        if is_B:
            self.padding_bytes = 64
        else:
            self.padding_bytes = 16
        self.padding_num = self.padding_bytes // self.type_size
        
        
        self.row_per_warp = self.slab_num // (self.micro_n * self.k_pack * self.warp_num * self.intra_iter_num)
        self.bank_row = self.bank_num // (self.micro_n * self.k_pack)
        self.permute_offset = 2 * self.bank_row * self.bank_row // self.row_per_warp
        if is_B:
            self.element_per_row = 32
            self.row_num = self.element_per_row * self.iter_num * self.warp_num
            self.row_per_slab = self.slab_num // self.row_num
            self.row_per_2slab = self.row_per_slab * 2
            
            
        
    
    def infer_type_size(self, dtype):
        if dtype == "float16" or dtype == "bfloat16":
            self.type_size = 2
        elif dtype == "float32":
            self.type_size = 4
        elif dtype == "int8":
            self.type_size = 1
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")


def shared_to_local_layout_A(tx, local_id, warp_m, idx, layout_config):
    # base
    row = tx % 16
    col = (tx // 16) * 4 * layout_config.k_pack + local_id
    
    # permute
    row = (row // 8) * layout_config.permute_offset + ((row % 8) % layout_config.permute_offset) + ((row % 8) // layout_config.permute_offset) * layout_config.row_per_warp
    row = (row % layout_config.row_per_warp)+ (row // layout_config.row_per_warp) * layout_config.row_per_slab
    # warp
    row = row + warp_m * layout_config.row_per_warp
    
    # iter
    row = (
            row
            + (idx % layout_config.intra_iter_num) * layout_config.row_per_warp * layout_config.warp_num
            + (idx // layout_config.intra_iter_num) * layout_config.row_per_4slab
        )
    
    if layout_config.slabs < 4:
        return row, col
    
    # rewind
    new_col = (((row % layout_config.row_per_4slab) // layout_config.row_per_slab) * layout_config.padding_num + col) % (layout_config.micro_n * layout_config.k_pack)
    row = (((row % layout_config.row_per_4slab) // layout_config.row_per_slab) * layout_config.padding_num + col) // (layout_config.micro_n * layout_config.k_pack) + row
    if layout_config.k_pack == 1:
        select_thread = (((tx // 16)) + ((tx & 3) * 4) + (((tx & 15) // 4) * 2)) // 16
    elif layout_config.k_pack == 2:
        select_thread = (tx == 31) | (tx == 45) | (tx == 47) | (tx == 59) | (tx == 61) | (tx == 63)
    else:
        assert False
    
    cond = select_thread & (warp_m == (layout_config.warp_num - 1)) & ((idx % layout_config.intra_iter_num) == (layout_config.intra_iter_num - 1))
    row = row - layout_config.row_per_slab * cond
    return row, new_col

def ki_shared_cvt(row, col, stride, new_stride):
    index = row * stride + col
    new_row = index // new_stride
    new_col = index - new_row * new_stride
    return new_row, new_col


# DCU LDS layout modes (keep in sync with tilelang.language.gemm_op.DCULdsLayout).
LDS_AUTO = 0
LDS_SIMPLE = 1
LDS_SWIZZLE = 2
LDS_SLAB = 3


def dtype_elem_bytes(dtype):
    name = str(dtype)
    if "bfloat16" in name or name in ("float16", "fp16"):
        return 2
    if "float32" in name or name == "fp32":
        return 4
    if "int8" in name or "uint8" in name:
        return 1
    raise ValueError(f"Unsupported dtype: {dtype}")


def compute_dcu_lds_slabs(mn_dim, k_pack, type_size, k_axis):
    """Match copy.cc slab count: k_axis 0 uses 32-wide B NT tiles, else 16-wide."""
    if type_size <= 0 or mn_dim <= 0:
        return 0
    slab_num = 1024 // type_size
    if slab_num <= 0:
        return 0
    mn_per_iter_warp = 32 if k_axis == 0 else 16
    if mn_dim < mn_per_iter_warp:
        return 0
    iter_times_warp = mn_dim // mn_per_iter_warp
    return 16 * 16 * k_pack * iter_times_warp // slab_num


def xor_swizzle_col(row, col, inner_dim, type_size, vec_bytes=16):
    """AMD/NVIDIA-style XOR bank swizzle on the innermost (contiguous) dimension.

    Matches ``make_mfma_swizzle_layout`` / ``makeMatrixCoreSwizzleLayout``:
    phase = (row // per_phase) % max_phase
    col' = ((col // vec) XOR phase) * vec + col % vec
    """
    if type_size <= 0 or inner_dim <= 0:
        return row, col
    vec_size = vec_bytes // type_size
    if vec_size <= 0 or inner_dim < vec_size:
        return row, col
    elems_per_bank_row = (32 * 32) // (type_size * 8)
    per_phase = max(1, elems_per_bank_row // inner_dim)
    max_phase = min(16 // per_phase, inner_dim // vec_size)
    if max_phase <= 1:
        return row, col
    phase = (row // per_phase) % max_phase
    col_off = ((col // vec_size) ^ phase) * vec_size + (col % vec_size)
    return row, col_off


def pick_dcu_lds_layout(user_mode, slabs, mn_dim, k_axis, k_pack=1):
    """Select SIMPLE / SWIZZLE / SLAB. ``user_mode`` 0 means AUTO.

    SIMPLE when the operand cannot use ds_read_m32x16:
      B NT (k_axis=0): N < 32;  A / B-T (k_axis=1): MN <= 16.
    Forced SWIZZLE/SLAB on a too-small tile degrades (SLAB also degrades
    when slabs < 4).
    """
    too_small = (mn_dim < 32) if k_axis == 0 else (mn_dim * k_pack <= 16)
    if user_mode == LDS_SIMPLE:
        return LDS_SIMPLE
    if too_small:
        return LDS_SIMPLE
    if user_mode == LDS_SWIZZLE:
        return LDS_SWIZZLE
    if user_mode == LDS_SLAB:
        return LDS_SWIZZLE if slabs < 4 else LDS_SLAB
    if slabs < 4:
        return LDS_SWIZZLE
    return LDS_SLAB


def shared_to_local_layout_A_swizzle(tx, local_id, warp_m, idx, k_pack, block_row_warps=1):
    """16x16 matrix-core A coords without slab permute / rewind.

    Warps are interleaved every 16 rows to match ``make_mmac_store_layout``:
    row = lane + warp_m * 16 + idx * 16 * m_warp.
    """
    row = tx % 16 + warp_m * 16 + idx * 16 * block_row_warps
    col = (tx // 16) * 4 * k_pack + local_id
    return row, col


def shared_to_local_layout_B_swizzle(tx, local_id, warp_n, idx, warp_num, element_per_row=32):
    """16x16 matrix-core B coords (K-row / N-col) without slab permute / rewind."""
    row = tx // 4
    col = (tx % 4) * 8 + local_id
    col = col + warp_n * element_per_row + idx * element_per_row * warp_num
    return row, col


@dataclass(frozen=True)
class SharedToLocalLayoutA:
    """
    LDS memory access data layout for A matrix with k_dim=32.
    """

    @staticmethod
    def base(tx, local_id, k_pack):
        row = tx % 16
        col = (tx // 16) * 4 * k_pack + local_id
        return row, col
    
    @staticmethod
    def permute(row, col, k_pack, warp_rows, block_row_warps):
        k_dim = 16
        col_size = k_pack * k_dim
        bank_size = 128 // 2
        bank_row = bank_size // col_size
        slab_num = 512
        row_per_iter_per_warp = slab_num // col_size // warp_rows // block_row_warps
        
        return row, col

    @staticmethod
    def slab(row, col):
        slab_num = 512
        row_num = 16
        new_row = (row % 4) + (row // 4) * (slab_num // (row_num))
        return new_row, col

    @staticmethod
    def warp(row, col, warp_m):
        new_row = row + warp_m * 4
        return new_row, col

    @staticmethod
    def itera(row, col, idx, block_row_warps):
        slab_size = 512
        per_warp_itera_num = 4 * 16
        per_warp_itera_row = 4
        slab_4_row = slab_size * 4 // 16
        intra_iter = slab_size // (per_warp_itera_num * block_row_warps)

        new_row = (
            row
            + (idx % intra_iter) * per_warp_itera_row * block_row_warps
            + (idx // intra_iter) * slab_4_row
        )
        return new_row, col
    
    def rewind(row, col, tx, warp_m, idx, block_row_warps):        
        slab_size = 512
        k_dim = 16
        per_slab_row = slab_size // k_dim
        slab_4_row = slab_size * 4 // k_dim
        per_warp_itera_num = 4 * k_dim
        intra_iter = slab_size // (per_warp_itera_num * block_row_warps)
        new_col = (((row % slab_4_row) // per_slab_row) * 8 + col) % k_dim
        new_row = (((row % slab_4_row) // per_slab_row) * 8 + col) // k_dim + row
        select_thread = (((tx // 16)) + ((tx & 3) * 4) + (((tx & 15) // 4) * 2)) // 16
        cond = select_thread & (warp_m == (block_row_warps - 1)) & ((idx % intra_iter) == (intra_iter - 1))
        new_row = new_row - per_slab_row * cond
        return new_row, new_col

    def permute(row, col, warp_cols, block_row_warps):
        per_warp_itera_num = 4 * 8
        col_size = warp_cols * block_row_warps * per_warp_itera_num // 2
        slab_size = 512
        row_per_slab = slab_size // col_size
        assert row_per_slab % 2 == 0 or row_per_slab == 1
        row_per_2_slab = 2 * row_per_slab
        new_row = (row % row_per_2_slab) // 2 + (row % 2) * row_per_slab + (row // row_per_2_slab) * row_per_2_slab
        return new_row ,col
def shared_to_local_layout_B(tx, local_id, warp_n, idx, layout_config):
    # base
    row = tx // 4
    col = (tx % 4) * 8 + local_id
    
    # permute
    row = (row % layout_config.row_per_2slab) // 2 + (row % 2) * layout_config.row_per_slab + (row // layout_config.row_per_2slab) * layout_config.row_per_2slab
    
    # warp
    col = col + warp_n * layout_config.element_per_row
    
    # iter
    col = col + idx * layout_config.element_per_row * layout_config.warp_num
    
    if layout_config.slabs < 4:
        return row, col
    
    # rewind
    col = (((row % layout_config.row_per_2slab) // layout_config.row_per_slab) * layout_config.padding_num + col)
    new_col = col % layout_config.row_num
    row = row + col // layout_config.row_num
    if layout_config.row_num == 256:
        select_thread = (((tx >> 2) & 3) == 3)
    elif layout_config.row_num == 128:
        select_thread = (((tx % 32) >> 2) == 7)
    elif layout_config.row_num == 64:
        select_thread = ((tx >> 2) == 15)
    cond = select_thread & (warp_n == (layout_config.warp_num - 1)) & (idx == (layout_config.iter_num - 1))
    row = row - layout_config.row_per_slab * cond
    return row, new_col

def shared_to_local_layout_B_simple(tx, local_id, warp_n=0, idx=0, micro_n=16):
    row = local_id + (tx // 16) * 4
    col = tx % 16 + warp_n * micro_n + idx * micro_n
    return row, col


def shared_to_local_layout_A_simple(tx, local_id, warp_m, idx, k_pack, block_row_warps=1):
    """Sequential 16x16 A coords (same as swizzle, without XOR)."""
    return shared_to_local_layout_A_swizzle(
        tx, local_id, warp_m, idx, k_pack, block_row_warps)

    
@dataclass(frozen=True)
class SharedToLocalLayoutB:
    """
    LDS memory access data layout for B matrix with k_dim=16.
    """

    @staticmethod
    def base(tx, local_id):
        row = tx // 4
        col = (tx % 4) * 8 + local_id
        return row, col
    
    @staticmethod
    def slab(row, col):
        return row, col
    
    @staticmethod
    def warp(row, col, warp_n):
        per_warp_itera_num = 4 * 8
        new_col = col + warp_n * per_warp_itera_num
        return row, new_col

    @staticmethod
    def itera(row, col, idx, block_col_warps):
        per_warp_itera_num = 4 * 8
        new_col = col + per_warp_itera_num * block_col_warps * idx
        return row, new_col
    
    def permute(row, col, warp_cols, block_row_warps):
        per_warp_itera_num = 4 * 8
        col_size = warp_cols * block_row_warps * per_warp_itera_num // 2
        slab_size = 512
        row_per_slab = slab_size // col_size
        assert row_per_slab % 2 == 0 or row_per_slab == 1
        row_per_2_slab = 2 * row_per_slab
        new_row = (row % row_per_2_slab) // 2 + (row % 2) * row_per_slab + (row // row_per_2_slab) * row_per_2_slab
        return new_row ,col
    
    def rewind(row, col, tx, warp_n, idx, warp_cols, block_row_warps):
        per_warp_itera_num = 4 * 8
        col_size = warp_cols * block_row_warps * per_warp_itera_num // 2
        col = (((tx & 7) // 4) * 32 + col)
        new_col = col % col_size
        new_row = row + (col // col_size)
        cond = (warp_n == (block_row_warps - 1)) & (idx == ((warp_cols // 2) - 1)) & (((tx >> 2) & 3) == 3)
        new_row = new_row - 2 * cond
        
        return new_row, new_col

'''
256x16 A share mem layout
  0   16   32   48
  1   17   33   49
  2   18   34   50
  3   19   35   51
---------------------
  4   20   36   52
  5   21   37   53


16x16 A Reg  layout
    0  16  32  48
    1  17  33  49
    2  18  34  50
    3  19  35  51
'''

def thread_id_shared_access_64x4_to_16x16_layout_A(tx, local_id):
    row = tx % 4
    col = (tx // 16) * 4 + local_id
    return row, col

def thread_id_shared_access_layout_slab_A(row, col):
    slab_num = 512
    row_num = 16
    new_row = (row % 4) + (row // 4) * (slab_num // (row_num * 4))
    return new_row, col

def thread_id_shared_access_layout_warp_A(row, col, warp_m):
    new_row = row + warp_m * 4
    return new_row, col

def thread_id_shared_access_layout_iter_A(row, col, idx, block_row_warps):
    slab_size = 512
    per_warp_iter_num = 4 * 16
    per_warp_iter_row = 4
    slab_4_row = slab_size * 4 // 16
    intra_iter = slab_size // (per_warp_iter_num * block_row_warps)
    new_row = row + (idx % intra_iter) * per_warp_iter_row * block_row_warps + (idx // intra_iter) * slab_4_row
    return new_row, col

def shared_to_local_layout_A_padding(row, col, idx, warp_rows, block_row_warps, tx, local_id):
    col = (tx // 16) * 4 + local_id
    row = tx % 4
    new_col = col
    if warp_rows > 1:
        inter_idx_padding = 2
    else:
        inter_idx_padding = 1
    
    paddings = inter_idx_padding * block_row_warps * 4
        
    new_row = row + paddings * ((tx & 15) // 4)
    new_row += (idx & 1) * (paddings // 2) + (idx // 2) * 16 * 2 * block_row_warps
    return new_row, new_col

def shared_to_local_layout_A_wrap(row, col, idx,  warp_rows, warp_m, block_row_warps, tx):
    if warp_rows > 1:
        inter_idx_padding = 2
    else:
        inter_idx_padding = 1
    paddings = inter_idx_padding * block_row_warps * 4
    # 跨 warp 的 padding
    row += warp_m * 4
    
    new_row = row + (col + ((tx & 15) // 4) * 8) // 16
    new_col = (col + ((tx & 15) // 4) * 8) % 16
    condition_col = (((tx // 16)) + ((tx & 3) * 4) + (((tx & 15) // 4) * 2)) // 16
    cond = condition_col & (warp_m == block_row_warps-1) & (idx & 1)
    new_row = new_row - 32 * cond
    
    # limit_row = (((tx & 15) // 4) + 1) * paddings
    # new_row = new_row - paddings * (row >= limit_row)
    # (warp_rows == 1) || (idx + 1) % 2 == 0) && warp_m == block_row_warps - 1 &&  ((((tx & 15) // 4)+1) * paddings) & (new_row) == 1
    # max_nooffset_row = (((tx & 15) // 4) + 1) * paddings + (idx // 2) * 16 * 2 * block_row_warps
    # new_row -= paddings * (
    #     ((warp_rows == 1) or ((idx + 1) % 2 == 0))
    #     & (warp_m == block_row_warps - 1)
    #     & ((max_nooffset_row & new_row) // max_nooffset_row == 1)
    # )
    return new_row, new_col

def shared_to_local_layout_B_padding(row, col, idx, warp_cols, block_col_warps, tx, local_id):
    # ???
    # T0  T1  T2   T3   T4   T5   T6  T7  T8   T9   T10  T11  T12  T13  T14  T15  T0  T1 ...
    # T0                                  T1                                      T2
    new_col = (tx % 4) * 8 + local_id # ((tx % 4) * 8) + (col // 16) * 4 + (row % 4)
    # new_col = ((col % 16 * 8 ) // 32 * 8) + (col // 16) * 4 + (row // 4)
    # swap per 4 row idx1 <-> idx2
    row_tmp = tx // 4
    bit0 = row_tmp & 1
    bit1 = (row_tmp >> 1) & 1
    new_row = ((row_tmp // 4) << 2)  + (bit0 * 2) + bit1
    
    # return new_row, new_col
    
    inter_idx_padding = block_col_warps
    
    paddings = 32
    new_col += idx * inter_idx_padding * paddings
    return new_row, new_col

def shared_to_local_layout_B_wrap(row, col, idx, warp_cols, warp_n, block_col_warps, tx, local_id):
    paddings = 32
    # 跨 idx 的 padding
    col = col + warp_n * paddings
    # 回转 每隔2行 列 向右偏移 32
    col += (((tx & 7) // 4) * 32)
    # col += ((row & 2) >> 1) * paddings
    #  new_col 对 256 宽度 取余
    new_col = col % (warp_cols * block_col_warps * 16)
    new_row = col // (warp_cols * block_col_warps * 16) + row
    
    cond = (warp_n == block_col_warps - 1) & (idx == 3) & (((tx >> 2) & 3) == 3)
    new_row = new_row - 2 * cond
    
    # limit_row = (tx // 16 + 1) * 4
    # new_row = new_row - 2 * (new_row >= limit_row)
    
    # row_tmp = row
    # new_row = row
    # # idx == warp_cols - 1 && warp_n == block_col_warps - 1 时，每隔两行，上下两行对调
    # new_row += (
    # (idx == warp_cols - 1)
    # & (warp_n == block_col_warps - 1)
    # & ((((row_tmp % 4) + 1) % 3) == 0)
    # )
    
    # new_row -= (
    # (idx == warp_cols - 1)
    # & (warp_n == block_col_warps - 1)
    # & ((((row_tmp % 4) + 1) % 4) == 0)
    # )
    
    return new_row, new_col

def shared_16x16_to_local_64x4_layout_A(i, j):
    thread_id = i + 16 * (j // 4)
    local_id = j % 4
    return thread_id, local_id

def thread_id_shared_access_64x4_to_16x16_layout_A(tx, local_id):
    # tx: thread id within warp (0-63)
    # local_id: element index within thread's 4-element vector (0-3)
    #
    # 根据布局:
    #   tx=0-15  处理行 0-15 的同一位置，列块 0 (j=0-3)
    #   tx=16-31 处理行 0-15 的同一位置，列块 1 (j=4-7)
    #   tx=32-47 处理行 0-15 的同一位置，列块 2 (j=8-11)
    #   tx=48-63 处理行 0-15 的同一位置，列块 3 (j=12-15)
    #
    # 例如 tx=0, local_id=0 → (i=0, j=0)
    #     tx=0, local_id=1 → (i=0, j=1)
    #     tx=16, local_id=0 → (i=0, j=4)
    i = tx % 16                  # 0-15: row position within the 16-row block
    j = (tx // 16) * 4 + local_id  # 0-15: column position
    return i, j

def shared_32x16_to_local_64x8_layout_B(i, j):
    # i: row (0-15)  j : col (0-31)
    # Layout:
    #             0   1    2    3    4   5   6   7   8    9   10   11   12   13   14   15
    #  row=0-3   T0  T1  T2   T3   T4  T5  T6  T7  T8   T9   T10  T11  T12  T13  T14  T15
    #  row=4-7   T16 T17 T18  T19  T20 T21 T22 T23 T24  T25  T26  T27  T28  T29  T30  T31
    #  row=8-11  T32 T33 T34  T35  T36 T37 T38 T39 T40  T41  T42  T43  T44  T45  T46  T47
    #  row=12-15 T48 T49 T50  T51  T52 T53 T54 T55 T56  T57  T58  T59  T60  T61  T62  T63
    thread_id = 16 * (i // 4) + j % 16
    local_id = (j // 16) * 4 + (i % 4)
    # print("T{:^2} V{:^2} ".format(thread_id,local_id),end=" ")
    return thread_id, local_id
    

def thread_id_shared_access_64x8_to_32x16_layout_B(tx, local_id):
    # tx: thread id within warp (0-63)
    # local_id: element index within thread's 4-element vector (0-7)
    # i : row (0-15)  j : col (0-31)
    i = (tx // 15) * 4 + local_id % 4
    j = (tx % 15) + (local_id // 4) * 16
    return i,j


def thread_id_shared_access_64x4_to_16x16_layout_C_m_n(thread_id, local_id):
    i = local_id * 4 + (thread_id // 16)
    j = thread_id % 16
    return i, j


def thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id):
    i = thread_id % 16
    j = local_id * 4 + (thread_id // 16)
    return i, j

shared_16x16_to_local_64x4_layout_m_n = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_k = shared_16x16_to_local_64x4_layout_A
shared_32x16_to_local_64x8_layout_n_m = shared_32x16_to_local_64x8_layout_B
shared_32x16_to_local_64x8_layout_k_n = shared_32x16_to_local_64x8_layout_B
