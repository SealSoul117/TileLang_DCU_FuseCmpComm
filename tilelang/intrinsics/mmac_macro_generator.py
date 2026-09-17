from __future__ import annotations
from tilelang import tvm as tvm
import tilelang.language as T
from typing import Literal, Callable
from tvm import DataType
from tvm.tir import PrimExpr, IndexMap, Buffer, Var, BufferRegion, BufferLoad
from tvm.runtime import convert
from tilelang.utils import is_fragment
from .mfma_layout import (
    shared_16x4_to_local_64x1_layout_A,
    shared_4x16_to_local_64x1_layout_B,
    # shared_16x16_to_local_64x4_layout_A,
    shared_16x16_to_local_64x4_layout_B,
    shared_16x32_to_local_64x8_layout_A,
    shared_16x32_to_local_64x8_layout_B,
    shared_16x64_to_local_64x16_layout_A,
    shared_16x64_to_local_64x16_layout_B,
    thread_id_shared_access_64x1_to_16x4_layout_A,
    thread_id_shared_access_64x1_to_4x16_layout_B,
    # thread_id_shared_access_64x4_to_16x16_layout_A,
    thread_id_shared_access_64x4_to_16x16_layout_B,
    thread_id_shared_access_64x8_to_16x32_layout_A,
    thread_id_shared_access_64x8_to_16x32_layout_B,
    thread_id_shared_access_64x16_to_16x64_layout_A,
    thread_id_shared_access_64x16_to_16x64_layout_B,
)

from .mmac_layout import (
    shared_16x16_to_local_64x4_layout_A,
    thread_id_shared_access_64x4_to_16x16_layout_A,
    thread_id_shared_access_64x4_to_16x16_layout_C_m_n,
    thread_id_shared_access_64x4_to_16x16_layout_C_n_m,
    shared_to_local_layout_A_padding,
    shared_to_local_layout_A_wrap,
    shared_32x16_to_local_64x8_layout_B,
    thread_id_shared_access_64x8_to_32x16_layout_B,
    shared_to_local_layout_B_padding,
    shared_to_local_layout_B_wrap,
    SharedToLocalLayoutA,
    SharedToLocalLayoutB,
    SharedToLocalKDim32LayoutA,
    SharedToLocalKDim32LayoutB,
    shared_to_local_layout_A,
    shared_to_local_layout_B,
    shared_to_local_layout_B_simple,
    shared_to_local_layout_A_simple,
    shared_to_local_layout_A_swizzle,
    shared_to_local_layout_B_swizzle,
    ki_shared_cvt,
    xor_swizzle_col,
    LayoutConfig,
    dtype_elem_bytes,
    LDS_SIMPLE,
    LDS_SWIZZLE,
    LDS_SLAB,
    LDS_REWIND,
)
from .dcu_lds_rewind import (
    AnBtRewindGeom,
    AtBnRewindGeom,
    shared_to_local_rewind_an_bt,
    shared_to_local_rewind_at_bn,
)

lift = convert


class MatrixCoreIntrinEmitter:
    """
    To eliminate Python syntax within TIR Macro.
    """

    M_DIM = 16
    N_DIM = 16
    WARP_SIZE = 64
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e5m2": "e5m2",
        "float8_e4m3fnuz": "e4m3fnuz",
    }

    # k_pack represents the number of elements in a vectorized instruction
    # Detail information can be found in the triton documentation
    # https://github.com/triton-lang/triton/blob/433037206d8870f0b82a3cd669097001084a29ed/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp#L419
    k_pack = 1
    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first = False

    def __init__(
        self,
        a_dtype: str = T.float16,
        b_dtype: str = T.float16,
        accum_dtype: str = T.float16,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        k_pack: int | None = None,
        m_pack: int | None = None,
        is_m_first: bool | None = False,
        b_preshuffle: bool | None = False,
        thread_var: Var | None = None,
        enable_fast_lds: bool | None = False,
        transpose_C: bool = True,
        a_lds_layout: int | None = None,
        b_lds_layout: int | None = None,
    ):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        # True: tvm_mmac(B, A) => BT * AT = CT (default, cheaper store).
        # False: tvm_mmac(A, B) => A * B = C. Load layouts are unchanged.
        self.transpose_C = transpose_C
        # Hint Information
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.chunk = chunk
        if a_lds_layout is None:
            a_lds_layout = LDS_SLAB if enable_fast_lds else LDS_SIMPLE
        if b_lds_layout is None:
            b_lds_layout = LDS_SLAB if enable_fast_lds else LDS_SIMPLE
        self.a_lds_layout = a_lds_layout
        self.b_lds_layout = b_lds_layout
        self.enable_fast_lds = b_lds_layout in (LDS_SLAB, LDS_REWIND)
        
        self._initialize_k_dim(a_dtype)
        self._initialize_abbrev(a_dtype, b_dtype, accum_dtype)
        self._initialize_local_size(self.M_DIM, self.N_DIM, self.k_dim, self.WARP_SIZE)
        self._initialize_mmac_prefix(self.k_dim)
        self._initialize_micro_size(self.M_DIM, self.N_DIM, self.k_dim)
        self._initialize_k_pack(k_pack)
        # M-side 4-slab grouping for fragment A load / C store. Independent of
        # this GEMM's K packing so gemm_rs can inherit a producer k_pack=2 C ABI.
        self.m_pack = int(m_pack) if m_pack is not None else self.k_pack
        self._initialize_is_m_first(is_m_first)
        self._initialize_b_preshuffle(b_preshuffle)

        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y
        self.reduce_k = reduce_k
        self.threads = self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k
        self.num_elems_per_byte = num_elems_per_byte
        self.thread_var = thread_var

    def _initialize_k_dim(self, a_dtype=T.float16):
        if isinstance(a_dtype, str):
            if a_dtype in ["float8_e4m3fnuz", T.int8]:
                self.k_dim = 32
                return
            a_dtype = DataType(a_dtype)

        if a_dtype.bits == 32:
            self.k_dim = 4
        elif a_dtype.bits in {16, 8}:
            self.k_dim = 16
        else:
            raise ValueError(f"Unsupported a_dtype = {a_dtype}")

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_mmac_prefix(self, k_dim=16):
        in_dtype, out_dtype = self.a_dtype, self.accum_dtype
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        out_dtype_abbrv = {"float16": "f16", "float32": "f32", "int8": "i8", "int32": "i32"}[out_dtype]

        in_dtype_abbrv = {"float16": "f16", "float32": "f32", "int8": "i8", "bfloat16": "bf16"}[in_dtype]

        self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}{in_dtype_abbrv}"

    def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    def _initialize_k_pack(self, k_pack: int | None = None):
        if k_pack is not None:
            self.k_pack = k_pack

    def _initialize_is_m_first(self, is_m_first: bool | None = False):
        if is_m_first is not None:
            self.is_m_first = is_m_first

    def _initialize_b_preshuffle(self, b_preshuffle: bool | None = False):
        if b_preshuffle is not None:
            self.b_preshuffle = b_preshuffle

    def get_ldmatrix_index_map(self, is_b=False):
        k_dim = self.k_dim * self.k_pack
        transposed = self.a_transposed if not is_b else self.b_transposed
        if k_dim == 4:
            index_map = shared_16x4_to_local_64x1_layout_A
            reverse_index_map = thread_id_shared_access_64x1_to_16x4_layout_A
            if is_b:
                index_map = shared_16x4_to_local_64x1_layout_A if transposed else shared_4x16_to_local_64x1_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x1_to_16x4_layout_A if transposed else thread_id_shared_access_64x1_to_4x16_layout_B
                )
        elif k_dim == 16:
            index_map = shared_16x16_to_local_64x4_layout_B if transposed else shared_16x16_to_local_64x4_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x4_to_16x16_layout_B if transposed else thread_id_shared_access_64x4_to_16x16_layout_A
            )

            if is_b:
                index_map = shared_16x16_to_local_64x4_layout_A if transposed else shared_16x16_to_local_64x4_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x4_to_16x16_layout_A if transposed else thread_id_shared_access_64x4_to_16x16_layout_B
                )
        elif k_dim == 32:
            index_map = shared_16x32_to_local_64x8_layout_B if transposed else shared_16x32_to_local_64x8_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x8_to_16x32_layout_B if transposed else thread_id_shared_access_64x8_to_16x32_layout_A
            )

            if is_b:
                index_map = shared_16x32_to_local_64x8_layout_A if transposed else shared_16x32_to_local_64x8_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x8_to_16x32_layout_A if transposed else thread_id_shared_access_64x8_to_16x32_layout_B
                )
        elif k_dim == 64:
            index_map = shared_16x64_to_local_64x16_layout_B if transposed else shared_16x64_to_local_64x16_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x16_to_16x64_layout_B if transposed else thread_id_shared_access_64x16_to_16x64_layout_A
            )

            if is_b:
                index_map = shared_16x64_to_local_64x16_layout_A if transposed else shared_16x64_to_local_64x16_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x16_to_16x64_layout_A if transposed else thread_id_shared_access_64x16_to_16x64_layout_B
                )
        else:
            raise ValueError("k_dim must be 4 or 16 or 32 or 64 currently")

        return index_map, reverse_index_map

    def _mmac_store_index_map_fn(self):
        # 16x16 C fragment: CT (default) uses m-major; C uses n-major.
        # Warp tiling offsets in make_mmac_store_layout stay unchanged.
        if self.transpose_C:
            return thread_id_shared_access_64x4_to_16x16_layout_C_m_n
        return thread_id_shared_access_64x4_to_16x16_layout_C_n_m

    def get_store_index_map(self, inverse: bool = False) -> IndexMap:
        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        index_map = IndexMap.from_func(self._mmac_store_index_map_fn(), index_dtype=T.int32)
        if not inverse:
            return index_map
        inverse_index_map = index_map.inverse([warp_size, local_size_c])
        return inverse_index_map
    
    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        else:
            return self.thread_var
        
    def extract_thread_binding(self, thread_id, is_m_first=None) -> tuple[PrimExpr, PrimExpr, PrimExpr]:
        """
        is_m_first: True if the thread binding is in the form of (tx, warp_n, warp_m)
        which represents [warp_size, block_row_warps (split n), block_col_warps (split m)]
        Otherwise, it is in the form of [warp_size, block_col_warps (split m), block_row_warps (split n)]
        """
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps

        # if is_m_first is None, then use the default value
        if is_m_first is None:
            is_m_first = self.is_m_first

        if is_m_first:
            lane_id, warp_n, warp_m = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_col_warps,
                (thread_id // (WARP_SIZE * block_col_warps)) % block_row_warps,
            )
            return lane_id, warp_n, warp_m
        else:
            lane_id, warp_m, warp_n = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_row_warps,
                (thread_id // (WARP_SIZE * block_row_warps)) % block_col_warps,
            )
            return lane_id, warp_n, warp_m

    def _rewind_block_threads(self) -> int:
        return self.block_row_warps * self.block_col_warps * self.WARP_SIZE

    def _at_bn_rewind_geom(self, block_mn, warp_mn_count, operand_mn_atom, dtype):
        return AtBnRewindGeom.try_create(
            int(block_mn),
            int(self.chunk),
            self._rewind_block_threads(),
            dtype_elem_bytes(dtype),
            int(warp_mn_count),
            int(operand_mn_atom),
            int(self.k_pack),
            int(self.micro_size_k),
        )

    def _an_bt_rewind_geom(self, block_mn, dtype):
        return AnBtRewindGeom.try_create(
            int(self.chunk),
            int(block_mn),
            self._rewind_block_threads(),
            dtype_elem_bytes(dtype),
        )

    def ldmatrix_a(self, A_local_buf, A_shared_buf: Buffer | BufferRegion, ki, rk=0):
        # share mem a needs warp number
        warp_num = self.block_row_warps
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        m_dim = self.block_row_warps * self.warp_row_tiles      
        k_pack = self.k_pack
        is_transposed = self.a_transposed
        thread_binding = self.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=False)
        layoutInfer = SharedToLocalLayoutA if micro_size_k == 16 else SharedToLocalKDim32LayoutA
        layout_config = LayoutConfig(k_pack, self.block_row_warps, max(1, warp_rows), self.a_dtype)
        # legalize shared buffer to region
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        if is_transposed:
            new_stride = A_region.buffer.shape[0]
        else:
            new_stride = A_region.buffer.shape[1]
        stride = micro_size_k * k_pack

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            # warp_n[0-256] -> {0,1,2,3}
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            # {0..3,16..19,32..35,48..51} -> 0
            # {4..7,20..23,36..39,52..55} -> 1
            # {8..11,24..27,40..43,56..59} -> 2
            # {12..15,28..31,44..47,60..63} -> 3

            # warp_rows 轮次 warp需要拆分成多轮来访问完整的行块
            if is_transposed:
                raise NotImplementedError("Transposed A is not implemented yet")
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        # 每轮初始位置行偏移
                        # row += i * warp_row_init
                        # # warp 组行间隔
                        # row += warp_group_idx * 4
                        # # warp 内行间隔
                        # row += warp_interval_idx * warp_row_interval
                        row, col = shared_to_local_layout_A_padding(row, col, i, warp_rows, self.block_row_warps, tx)
                        l, r = (rk * chunk + ki * (k_pack * micro_size_k), warp_m * warp_row_tiles + i * micro_size_x)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[A_base0 + l + row, A_base1 + r + col]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        if self.a_lds_layout == LDS_SWIZZLE:
                            row, col = shared_to_local_layout_A_swizzle(
                                tx, local_id, warp_m, i, k_pack, self.block_row_warps)
                            inner_dim = int(A_buf.shape[-1])
                            row, col = xor_swizzle_col(
                                row, col + ki * k_pack * micro_size_k,
                                inner_dim, layout_config.type_size)
                        elif self.a_lds_layout == LDS_SIMPLE:
                            row, col = shared_to_local_layout_A_simple(
                                tx, local_id, warp_m, i, k_pack, self.block_row_warps)
                            row, col = row, col + ki * k_pack * micro_size_k
                        elif self.a_lds_layout == LDS_REWIND:
                            geom = self._at_bn_rewind_geom(
                                m_dim, self.block_row_warps, warp_row_tiles, self.a_dtype)
                            if geom is not None:
                                row, col = shared_to_local_rewind_at_bn(
                                    tx, local_id, warp_m, i, k_pack, self.block_row_warps,
                                    ki * k_pack * micro_size_k, geom)
                            else:
                                row, col = shared_to_local_layout_A(
                                    tx, local_id, warp_m, i, layout_config)
                                row, col = ki_shared_cvt(
                                    row, col + ki * m_dim * self.k_pack * micro_size_k,
                                    stride, new_stride)
                        else:
                            row, col = shared_to_local_layout_A(tx, local_id, warp_m, i, layout_config)
                            row, col = ki_shared_cvt(
                                row, col + ki * m_dim * self.k_pack * micro_size_k, stride, new_stride)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[A_base0 + row, A_base1 + col]

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)      

    def ldmatrix_b(self, B_local_buf, B_shared_buf: Buffer | BufferRegion, ki, kp=0, rk=0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        n_dim = self.block_col_warps * self.warp_col_tiles
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        k_pack = self.k_pack
        is_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=True)
        
        if not is_transposed:
            if self.b_lds_layout != LDS_SIMPLE:
                layout_config = LayoutConfig(
                    k_pack, self.block_col_warps, max(1, warp_cols // 2),
                    self.b_dtype, True)
            else:
                layout_config = LayoutConfig(
                    k_pack, self.block_col_warps, max(1, warp_cols), self.b_dtype, True)
        else:
            layout_config = LayoutConfig(k_pack, self.block_col_warps, warp_cols, self.b_dtype)
        # legalize shared buffer to region
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        if is_transposed:
            new_stride = B_region.buffer.shape[1]
        else:
            new_stride = B_region.buffer.shape[0]
        stride = micro_size_k * k_pack
        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            kp,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                inner_dim = int(B_buf.shape[-1])
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        if self.b_lds_layout == LDS_SWIZZLE:
                            row, col = shared_to_local_layout_A_swizzle(
                                tx, local_id, warp_n, j, k_pack, self.block_col_warps)
                            row, col = xor_swizzle_col(
                                row, col + ki * k_pack * micro_size_k,
                                inner_dim, layout_config.type_size)
                        elif self.b_lds_layout == LDS_SIMPLE:
                            row, col = shared_to_local_layout_A_simple(
                                tx, local_id, warp_n, j, k_pack, self.block_col_warps)
                            row, col = row, col + ki * k_pack * micro_size_k
                        elif self.b_lds_layout == LDS_REWIND:
                            geom = self._at_bn_rewind_geom(
                                n_dim, self.block_col_warps, warp_col_tiles, self.b_dtype)
                            if geom is not None:
                                row, col = shared_to_local_rewind_at_bn(
                                    tx, local_id, warp_n, j, k_pack, self.block_col_warps,
                                    ki * k_pack * micro_size_k, geom)
                            else:
                                row, col = shared_to_local_layout_A(
                                    tx, local_id, warp_n, j, layout_config)
                                row, col = ki_shared_cvt(
                                    row, col + ki * n_dim * self.k_pack * micro_size_k,
                                    stride, new_stride)
                        else:
                            row, col = shared_to_local_layout_A(
                                tx, local_id, warp_n, j, layout_config)
                            row, col = ki_shared_cvt(
                                row, col + ki * n_dim * self.k_pack * micro_size_k,
                                stride, new_stride)
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[B_base0 + row, B_base1 + col]

            else:
                if self.b_lds_layout == LDS_SWIZZLE:
                    inner_dim = int(B_buf.shape[-1])
                    for j in T.serial(warp_cols):
                        for local_id in T.vectorized(k_pack * local_size_b):
                            row, col = shared_to_local_layout_B_simple(
                                tx, local_id, warp_n, j, micro_size_y, self.block_col_warps)
                            row, col = xor_swizzle_col(
                                row + ki * micro_size_k,
                                col, inner_dim, layout_config.type_size)
                            B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[B_base0 + row, B_base1 + col]
                elif self.b_lds_layout == LDS_REWIND:
                    geom = self._an_bt_rewind_geom(n_dim, self.b_dtype)
                    if geom is not None:
                        for j in T.serial(warp_cols // 2):
                            for local_id in T.vectorized(2 * local_size_b):
                                row, col = shared_to_local_rewind_an_bt(
                                    tx, local_id, warp_n, j, self.block_col_warps,
                                    ki * self.k_pack * micro_size_k + kp * micro_size_k, geom)
                                B_local_buf[kp * local_size_b * warp_cols + j * local_size_b * 2 + local_id] = B_buf[B_base0 + row, B_base1 + col]
                    else:
                        for j in T.serial(warp_cols // 2):
                            for local_id in T.vectorized(2 * local_size_b):
                                row, col = shared_to_local_layout_B(tx, local_id, warp_n, j, layout_config)
                                B_local_buf[kp * local_size_b * warp_cols + j * local_size_b * 2 + local_id] = B_buf[B_base0 + row + ki * self.k_pack *micro_size_k + kp * micro_size_k, B_base1 + col]
                elif self.b_lds_layout != LDS_SIMPLE:
                    for j in T.serial(warp_cols // 2):
                        for local_id in T.vectorized(2 * local_size_b):
                            row, col = shared_to_local_layout_B(tx, local_id, warp_n, j, layout_config)
                            B_local_buf[kp * local_size_b * warp_cols + j * local_size_b * 2 + local_id] = B_buf[B_base0 + row + ki * self.k_pack *micro_size_k + kp * micro_size_k, B_base1 + col]
                else:
                    for j in T.serial(warp_cols):
                        for local_id in T.vectorized(k_pack * local_size_b):
                            row, col = shared_to_local_layout_B_simple(
                                tx, local_id, warp_n, j, micro_size_y, self.block_col_warps)
                            B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[B_base0 + row + ki * micro_size_k, B_base1 + col]
        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, kp, thread_binding, rk)

    def mmac(
        self,
        A_local_buf: Buffer,
        B_local_buf: Buffer,
        C_local_buf: Buffer,
        k_inner: PrimExpr | None = 0,
        transpose_C: bool | None = None,
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        k_pack = self.k_pack
        mmac_suffix = self.mmac_suffix
        a_dtype, b_dtype, out_dtype = self.a_dtype, self.b_dtype, self.accum_dtype
        compute_a_dtype = a_dtype if local_size_a == 1 else f"{a_dtype}x{local_size_a}"
        compute_b_dtype = b_dtype if local_size_b == 1 else f"{b_dtype}x{local_size_b}"
        compute_out_dtype = out_dtype if local_size_out == 1 else f"{out_dtype}x{local_size_out}"
        transpose_c = self.transpose_C if transpose_C is None else transpose_C

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        # gemm_ss keeps the original packed-K-then-M indexing (A is ss local).
        # gemm_rs A-fragment matches store/load K-major: 4-reg groups walk K
        # tiles first, then M, so float16x4 index is ki + i * n_k_tiles.
        a_is_rs_fragment = a_is_fragment and (not b_is_fragment)
        n_k_tiles = self.chunk // (self.micro_size_k * k_pack)
        a_local_stride: PrimExpr = k_inner * warp_rows * k_pack * local_size_a if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * k_pack * local_size_b if b_is_fragment else 0

        # Hardware slot 0 is matrixcore-A, slot 1 is matrixcore-B.
        # A/B keep their own load layouts; only the mmac argument order changes.
        # True  (default): mmac(B, A) => BT * AT = CT (store-friendly).
        # False:           mmac(A, B) => A  * B  = C.

        @T.macro
        def _warp_mmac_ct(A_local_buf, B_local_buf, C_local_buf):
            for kp, i, j in T.grid(k_pack, warp_rows, warp_cols):
                if a_is_rs_fragment:
                    a_idx = (i * n_k_tiles + k_inner) * k_pack + kp
                else:
                    a_idx = (a_local_stride + (i * k_pack + kp) * local_size_a) // local_size_a
                c_idx = (i * warp_cols * local_size_out + j * local_size_out) // local_size_out
                if self.b_transposed:
                    b_idx = (b_local_stride + (j * k_pack + kp) * local_size_b) // local_size_b
                else:
                    b_idx = (b_local_stride + j * local_size_b + kp * warp_cols * local_size_b) // local_size_b
                T.tvm_mmac(
                    mmac_suffix,
                    "row",
                    "row",
                    compute_b_dtype,
                    compute_a_dtype,
                    compute_out_dtype,
                    B_local_buf.data,
                    b_idx,
                    A_local_buf.data,
                    a_idx,
                    C_local_buf.data,
                    c_idx,
                    dtype=compute_out_dtype,
                )

        @T.macro
        def _warp_mmac_c(A_local_buf, B_local_buf, C_local_buf):
            for kp, i, j in T.grid(k_pack, warp_rows, warp_cols):
                if a_is_rs_fragment:
                    a_idx = (i * n_k_tiles + k_inner) * k_pack + kp
                else:
                    a_idx = (a_local_stride + (i * k_pack + kp) * local_size_a) // local_size_a
                c_idx = (i * warp_cols * local_size_out + j * local_size_out) // local_size_out
                if self.b_transposed:
                    b_idx = (b_local_stride + (j * k_pack + kp) * local_size_b) // local_size_b
                else:
                    b_idx = (b_local_stride + j * local_size_b + kp * warp_cols * local_size_b) // local_size_b
                T.tvm_mmac(
                    mmac_suffix,
                    "row",
                    "row",
                    compute_a_dtype,
                    compute_b_dtype,
                    compute_out_dtype,
                    A_local_buf.data,
                    a_idx,
                    B_local_buf.data,
                    b_idx,
                    C_local_buf.data,
                    c_idx,
                    dtype=compute_out_dtype,
                )

        if transpose_c:
            return _warp_mmac_ct(A_local_buf, B_local_buf, C_local_buf)
        return _warp_mmac_c(A_local_buf, B_local_buf, C_local_buf)

    def stmatrix(self, C_local_buf, C_buf, pid_m=None, pid_n=None):
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_out = self.local_size_out
        thread_binding = self.get_thread_binding()
        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"

        # STS
        # MMAC Store must be in simulated instead of TVM Intrins
        # As TVM Intrins is like a hack that the threadIdx.x should be always
        # equal to the warp_size
        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    row, col = T.meta_var(self._mmac_store_index_map_fn()(tx, local_id))
                    if C_buf_dims == 2:
                        C_buf[(warp_m * warp_rows + i) * M_DIM + row, (warp_n * warp_cols + j) * N_DIM + col] = C_local_buf[
                            j * (warp_rows * local_size_out) + i * local_size_out + local_id
                        ]
                    else:
                        C_buf[warp_n * warp_cols + j, warp_m * warp_rows + i, row, col] = C_local_buf[
                            j * warp_rows * local_size_out + i * local_size_out + local_id
                        ]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    row, col = T.meta_var(self._mmac_store_index_map_fn()(tx, local_id))
                    C_buf[
                        (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row, (pid_n * BLOCK_N + warp_n * warp_cols + j) * N_DIM + col
                    ] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out + local_id]

        return (
            _warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
            if is_global
            else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding)
        )
    
    def make_mmac_load_layout(self, local_buf: Buffer, matrix: Literal["A", "B"] = "A") -> T.Fragment:
        """
        Create a layout function for storing MFMA results into a fragment buffer.

        Parameters
        ----------
        local_buf : tir.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment

        assert matrix in ["A", "B"], "matrix should be either A or B"
        matrix_is_a: bool = matrix == "A"
        matrix_is_b: bool = matrix == "B"
        transposed = self.a_transposed if matrix_is_a else self.b_transposed

        # s represents spatial axis
        # r represents reduction axis
        # sr represents the two dims are spatial + reduction
        # rs represents the two dims are reduction + spatial
        # sr also can represent a non-transposed basic layout
        # then rs also can represent a transposed basic layout
        transform_func_sr_a: Callable = None
        transform_func_sr_b: Callable = None

        k_dim = self.k_dim * self.k_pack

        if k_dim == 4:
            transform_func_sr_a = shared_16x4_to_local_64x1_layout_A
            transform_func_sr_b = shared_16x4_to_local_64x1_layout_A
        elif k_dim == 16:
            transform_func_sr_a = shared_16x16_to_local_64x4_layout_A
            transform_func_sr_b = shared_16x16_to_local_64x4_layout_A
        elif k_dim == 32:
            transform_func_sr_a = shared_16x32_to_local_64x8_layout_A
            transform_func_sr_b = shared_16x32_to_local_64x8_layout_A
        elif k_dim == 64:
            transform_func_sr_a = shared_16x64_to_local_64x16_layout_A
            transform_func_sr_b = shared_16x64_to_local_64x16_layout_A
        else:
            raise ValueError("k_dim must be 4 or 16 or 32 or 64 currently")

        is_sr_conditions = [False]
        is_sr_conditions.append(matrix_is_a and not transposed)
        is_sr_conditions.append(matrix_is_b and transposed)
        is_sr_axis_order = any(is_sr_conditions)

        transform_func: Callable = None
        if matrix_is_a:
            transform_func = transform_func_sr_a if is_sr_axis_order else lambda i, j: transform_func_sr_a(j, i)
        elif matrix_is_b:
            transform_func = transform_func_sr_b if is_sr_axis_order else lambda i, j: transform_func_sr_b(j, i)
        else:
            raise ValueError(f"Unsupported matrix {matrix}")

        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"

        inverse_mmac_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        # Register A: 16x16 inner uses A-load layout (after mmac_ct_to_a_16x16).
        # Warp / iter tiling outside 16x16 copies make_mmac_store_layout, with
        # the second axis taken from K (chunk) so every warp holds full K.
        if matrix_is_a:
            shape = local_buf.shape
            micro_size_x, micro_size_k = self.micro_size_x, self.micro_size_k
            k_pack = self.k_pack
            inner_k = micro_size_k * k_pack
            local_vec = self.local_size_a * k_pack
            block_row_warps = self.block_row_warps
            warp_size = self.WARP_SIZE
            # M tiling matches make_mmac_store_layout (k_pack=2 → 32 rows/warp).
            # Inner K is still 16*k_pack (16x32 primitive).
            data_byte = max(1, DataType(self.a_dtype).bits // 8)
            slab4_elems = (1024 // data_byte) * 4
            raw_slab4 = (self.block_row_warps * self.warp_row_tiles *
                         self.micro_size_k * self.m_pack) // slab4_elems
            if raw_slab4 < 1:
                micro_size_dcu_x = micro_size_x
            else:
                micro_size_dcu_x = raw_slab4 * 16
            micro_size_dcu_k = inner_k
            n_k_tiles = max(1, self.chunk // inner_k)
            x_group = max(1, micro_size_dcu_x // micro_size_x)
            warp_rows = max(1, self.warp_rows)

            def forward_thread_a(i: int, j: int) -> int:
                block_i = (i // micro_size_dcu_x) % block_row_warps
                mmac_i, mmac_j = i % micro_size_x, j % inner_k
                lane_id, _ = inverse_mmac_load_layout.map_indices([mmac_i, mmac_j])
                return block_i * warp_size + lane_id

            def forward_index_a(i: int, j: int) -> int:
                warp_i = (i // micro_size_dcu_x) // block_row_warps
                tile_i = (i // micro_size_x) % x_group
                warp_j = j // micro_size_dcu_k
                mmac_i, mmac_j = i % micro_size_x, j % inner_k
                _, local_id = inverse_mmac_load_layout.map_indices([mmac_i, mmac_j])
                i_reg = warp_i + tile_i * (warp_rows // x_group)
                return (i_reg * n_k_tiles + warp_j) * local_vec + local_id

            return T.Fragment(
                shape,
                forward_thread_fn=forward_thread_a,
                forward_index_fn=forward_index_a,
            )

        micro_size_r, micro_size_s = self.micro_size_k, self.micro_size_y
        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )

        def forward_thread(i: int, j: int) -> int:
            lane_id, _ = inverse_mmac_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            _, local_id = inverse_mmac_load_layout.map_indices([i, j])
            return local_id

        base_fragment = T.Fragment(
            [micro_size_s, micro_size_r * self.k_pack] if is_sr_axis_order else [micro_size_r * self.k_pack, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        chunk = self.chunk
        warp_s = warp_cols
        warp_r = chunk // (micro_size_r * self.k_pack)
        block_s = block_col_warps
        replicate = block_row_warps

        if is_sr_axis_order:
            warp_fragment = base_fragment.repeat([warp_s, warp_r], repeat_on_thread=False, lower_dim_first=False)
            return warp_fragment.replicate(replicate).repeat(
                [block_s, 1], repeat_on_thread=True, lower_dim_first=True)
        warp_fragment = base_fragment.repeat([warp_r, warp_s], repeat_on_thread=False, lower_dim_first=True)
        return warp_fragment.replicate(replicate).repeat(
            [1, block_s], repeat_on_thread=True, lower_dim_first=True)
    
    
    def make_mmac_store_layout(self, local_buf: Buffer) -> T.Fragment:
        """
        Create a layout function for storing MMAC results into a fragment buffer.
        
        Parameters
        ----------
        local_buf : tir.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object describing the thread and index layout for MMAC.
        """
        from tilelang.utils import is_fragment
        shape = local_buf.shape
        inverse_mmac_store_layout = self.get_store_index_map(inverse=True)
        assert is_fragment(local_buf), "local_buf must be a fragment"
        
        micro_size_x, micro_size_y = self.micro_size_x, self.micro_size_y
        local_size_out = self.local_size_out
        block_row_warps, block_col_warps = self.block_row_warps, self.block_col_warps
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        warp_size = self.WARP_SIZE
        is_m_first = self.is_m_first
        data_byte = max(1, DataType(self.a_dtype).bits // 8)
        slab4_elems = (1024 // data_byte) * 4
        raw_slab4 = (self.block_row_warps * self.warp_row_tiles * self.micro_size_k *
                     self.m_pack) // slab4_elems
        # Small tiles (e.g. 16x16) yield slab4_num=0 and i//0 in IndexMap inverse.
        if raw_slab4 < 1:
            slab4_num = 1
            micro_size_dcu_x = micro_size_x
        else:
            slab4_num = raw_slab4
            micro_size_dcu_x = raw_slab4 * 16
        n_extent = self.block_col_warps * self.warp_col_tiles
        raw_slab4_b = (n_extent * self.micro_size_k * self.k_pack) // slab4_elems
        # 16-wide B (SWIZZLE / SIMPLE) cannot use ds_read_m32x16, so C must keep
        # 16-col tiles. NT store packing (forward_index_nt) is not that layout:
        # at N=64 it is identical to NN, and at N>=128 it permutes 16-col groups.
        # SLAB/REWIND may group N in 32 for ds_read once a 4-slab exists.
        b_16wide = self.b_lds_layout in (LDS_SWIZZLE, LDS_SIMPLE)
        if b_16wide or n_extent < 32 or raw_slab4_b < 1:
            slab4_num_b = 1
            micro_size_dcu_y = micro_size_y
        else:
            slab4_num_b = max(1, raw_slab4_b)
            micro_size_dcu_y = 32
        if (not b_16wide) and self.b_transposed and raw_slab4_b >= 1:
            micro_size_dcu_y = slab4_num_b * 16
        def forward_thread(i: int, j: int) -> int:
            """
            Map fragment row `i` and column `j` to a thread index.
            """
            y_group = max(1, micro_size_dcu_y // micro_size_y)
            warp_cols_dcu = warp_cols // y_group
            block_i, block_j = (i // micro_size_dcu_x) % block_row_warps, (j // micro_size_dcu_y) % block_col_warps
            # (i // micro_size_x) // warp_rows, (j // micro_size_y) // warp_cols
            mmac_i, mmac_j = i % micro_size_x, j % micro_size_y
            
            lane_id, _ = inverse_mmac_store_layout.map_indices([mmac_i, mmac_j])
            
            if is_m_first:
                thread_id = block_i * (block_col_warps * warp_cols_dcu) + block_j * warp_size + lane_id
            else:
                thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            return thread_id

        def forward_index(i: int, j: int) -> int:
            """
            Map fragment row `i` and column `j` to a local index within a thread's registers.
            """
            warp_i, warp_j = (i // micro_size_dcu_x) // block_row_warps, j // micro_size_dcu_y // block_col_warps
            x_group = max(1, micro_size_dcu_x // micro_size_x)
            y_group = max(1, micro_size_dcu_y // micro_size_y)
            warp_i_in_32_16 = (i // 16) % x_group
            warp_j_in_32_16 = (j // 16) % y_group
            # (i // micro_size_x) % warp_rows, (j // micro_size_y) % warp_cols
            mmac_i, mmac_j = i % micro_size_x, j % micro_size_y
            
            # 使用 MMAC 的底层硬件逆映射获取局部偏移
            _, local_id = inverse_mmac_store_layout.map_indices([mmac_i, mmac_j])
            return warp_j * (y_group * local_size_out) + warp_j_in_32_16 * local_size_out + local_id + (warp_i + warp_i_in_32_16 * warp_rows // x_group) * (warp_cols * local_size_out)
            # return warp_j * (warp_rows * local_size_out) * 2 + warp_i * local_size_out + local_id + warp_j_in_32_16 * local_size_out

        def forward_thread_nt(i: int, j: int) -> int:
            """
            Map fragment row `i` and column `j` to a thread index.
            """
            # micro_size_dcu_x = 32
            # micro_size_dcu_y = 32
            # warp_cols_dcu = warp_cols // (micro_size_dcu_y // micro_size_y)
            # block_i, block_j = (i // micro_size_dcu_x) % block_row_warps, (j // micro_size_dcu_y) % block_col_warps
            # mmac_i, mmac_j = i % micro_size_x, j % micro_size_y
            
            # lane_id, _ = inverse_mmac_store_layout.map_indices([mmac_i, mmac_j])
            
            # if is_m_first:
            #     thread_id = block_i * (block_col_warps * warp_cols_dcu) + block_j * warp_size + lane_id
            # else:
            #     thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            # return thread_id
            y_group = max(1, micro_size_dcu_y // micro_size_y)
            warp_cols_dcu = warp_cols // y_group
            block_i, block_j = (i // micro_size_dcu_x) % block_row_warps, (j // micro_size_dcu_y) % block_col_warps
            # (i // micro_size_x) // warp_rows, (j // micro_size_y) // warp_cols
            mmac_i, mmac_j = i % micro_size_x, j % micro_size_y
            
            lane_id, _ = inverse_mmac_store_layout.map_indices([mmac_i, mmac_j])
            
            if is_m_first:
                thread_id = block_i * (block_col_warps * warp_cols_dcu) + block_j * warp_size + lane_id
            else:
                thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            return thread_id
        
        def forward_index_nt(i: int, j: int) -> int:
            """
            Map fragment row `i` and column `j` to a local index within a thread's registers.
            """
            # micro_size_dcu_x = 32
            # micro_size_dcu_y = 32
            # warp_i, warp_j = (i // micro_size_dcu_x) // block_row_warps, j // micro_size_dcu_y // block_col_warps
            # warp_j_in_64_16 = (j // 64)
            # warp_i_in_32_16 = (i // 16) % 2
            # warp_j_in_32_16 = (j // 16) % 2
            # mmac_i, mmac_j = i % micro_size_x, j % micro_size_y
            
            # # 使用 MMAC 的底层硬件逆映射获取局部偏移
            # _, local_id = inverse_mmac_store_layout.map_indices([mmac_i, mmac_j])
            # return warp_j_in_32_16 * (4 * local_size_out) + warp_j_in_64_16 * local_size_out + local_id + (warp_i + warp_i_in_32_16 * warp_rows // (micro_size_dcu_x // micro_size_x)) * (warp_cols * local_size_out)
            # # return warp_j * (warp_rows * local_size_out) * 2 + warp_i * local_size_out + local_id + warp_j_in_32_16 * local_size_out
            warp_i, warp_j = (i // micro_size_dcu_x) // block_row_warps, j // micro_size_dcu_y // block_col_warps
            x_group = max(1, micro_size_dcu_x // micro_size_x)
            y_group = max(1, micro_size_dcu_y // micro_size_y)
            warp_i_in_32_16 = (i // 16) % x_group
            warp_j_in_32_16 = (j // 16) % y_group
            # (i // micro_size_x) % warp_rows, (j // micro_size_y) % warp_cols
            mmac_i, mmac_j = i % micro_size_x, j % micro_size_y
            
            # 使用 MMAC 的底层硬件逆映射获取局部偏移
            _, local_id = inverse_mmac_store_layout.map_indices([mmac_i, mmac_j])
            return (warp_j + warp_j_in_32_16 * warp_cols // y_group) * local_size_out + local_id + (warp_i + warp_i_in_32_16 * warp_rows // x_group) * (warp_cols * local_size_out)
        # warp_j * ((16 // micro_size_y) * local_size_out) + warp_j_in_32_16 * local_size_out + local_id + (warp_i + warp_i_in_32_16 * warp_rows // (micro_size_dcu_x // micro_size_x)) * (warp_cols * local_size_out)
            # return warp_j * (warp_rows * local_size_out) * 2 + warp_i * local_size_out + local_id + warp_j_in_32_16 * local_size_out

        if not self.b_transposed and not self.a_transposed:
            return T.Fragment(
                shape,
                forward_thread_fn=forward_thread,
                forward_index_fn=forward_index,
            )
        return T.Fragment(
            shape,
            forward_thread_fn=forward_thread_nt,
            forward_index_fn=forward_index_nt,
        )

    @staticmethod
    def _legalize_to_buffer_region(obj: Buffer | BufferLoad | BufferRegion) -> BufferRegion:
        """
        Convert Buffer/BufferRegion/BufferLoad to a BufferRegion.

        - Buffer -> full-region BufferRegion covering entire shape
        - BufferRegion -> returned as-is
        - BufferLoad -> best-effort convert via get_buffer_region_from_load;
        if scalar, fall back to 1-sized ranges at given indices
        """
        if isinstance(obj, BufferRegion):
            return obj
        if isinstance(obj, Buffer):
            mins = [tir.IntImm("int32", 0) for _ in obj.shape]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, obj.shape)]
            return BufferRegion(obj, ranges)
        if isinstance(obj, BufferLoad):
            region = get_buffer_region_from_load(obj)
            if region is not None:
                return region
            # Fallback: scalar load -> 1-sized ranges at indices
            mins = [idx for idx in obj.indices]
            ones = [tir.IntImm("int32", 1) for _ in obj.indices]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, ones)]
            return BufferRegion(obj.buffer, ranges)
        raise ValueError(f"Unsupported argument type for BufferRegion: {type(obj)}")
    
class MatrixCorePreshuffleIntrinEmitter(MatrixCoreIntrinEmitter):
    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float16",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        k_pack: int | None = None,
        is_m_first: bool | None = False,
        a_preshuffle: bool | None = False,
        b_preshuffle: bool | None = False,
        thread_var: Var | None = None,
    ):
        super().__init__(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
            k_pack=k_pack,
            is_m_first=is_m_first,
            thread_var=thread_var,
        )
        self._initialize_preshuffle(a_preshuffle, b_preshuffle)

    def _initialize_preshuffle(self, a_preshuffle: bool, b_preshuffle: bool):
        if a_preshuffle is not None:
            self.a_preshuffle = a_preshuffle
        if b_preshuffle is not None:
            self.b_preshuffle = b_preshuffle

    def ldmatrix_a(self, A_local_buf, A_buf, ki, rk=0, pid_m=None, pid_n=None):
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        k_pack = self.k_pack
        is_transposed = self.a_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=False)
        is_global = pid_m is not None and pid_n is not None

        # no preshuffle, use the default implementation
        if self.a_preshuffle is False:
            return super().ldmatrix_a(A_local_buf, A_buf, ki, rk)

        def _warp_ldmatrix_a_global(
            A_local_buf,
            A_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            (pid_m * self.block_row_warps + warp_m) * warp_rows + i,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[l, r, row, col]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            (pid_m * self.block_row_warps + warp_m) * warp_rows + i,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[l, r, row, col]

        @T.macro
        def _warp_ldmatrix_a_shared(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            warp_m * warp_rows + i,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_shared_buf[l, r, row, col]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (warp_m * warp_rows + i, rk * (chunk // micro_size_k) + ki)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_shared_buf[l, r, row, col]

        return (
            _warp_ldmatrix_a_global(A_local_buf, A_buf, ki, thread_binding, rk)
            if is_global
            else _warp_ldmatrix_a_shared(A_local_buf, A_buf, ki, thread_binding, rk)
        )

    def ldmatrix_b(self, B_local_buf, B_buf, ki, rk=0, pid_m=None, pid_n=None):
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        k_pack = self.k_pack
        is_transposed = self.b_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=True)
        is_global = pid_m is not None and pid_n is not None

        if self.b_preshuffle is False:
            return super().ldmatrix_b(B_local_buf, B_buf, ki, rk, pid_m, pid_n)

        @T.macro
        def _warp_ldmatrix_b_global(
            B_local_buf,
            B_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            (pid_n * self.block_col_warps + warp_n) * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[l, r, row, col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            (pid_n * self.block_col_warps + warp_n) * warp_cols + j,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[l, r, row, col]

        @T.macro
        def _warp_ldmatrix_b_shared(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(((tx & 15) >> 2) + ((tx & 3) << 2) + ((tx >> 4) << 4), local_id))
                        l, r = (
                            warp_n * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_shared_buf[l, r, row, col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(((tx & 15) >> 2) + ((tx & 3) << 2) + ((tx >> 4) << 4), local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            warp_n * warp_cols + j,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_shared_buf[l, r, row, col]

        return (
            _warp_ldmatrix_b_global(B_local_buf, B_buf, ki, thread_binding, rk)
            if is_global
            else _warp_ldmatrix_b_shared(B_local_buf, B_buf, ki, thread_binding, rk)
        )
