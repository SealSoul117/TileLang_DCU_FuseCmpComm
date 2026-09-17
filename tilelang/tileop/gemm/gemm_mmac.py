from .gemm_base import GemmBase
from tilelang.layout import make_swizzled_layout, make_linear_layout
from tilelang.layout.fragment import Fragment
from tilelang.intrinsics.mmac_macro_generator import (
    MatrixCoreIntrinEmitter,
)
from tilelang.intrinsics.mmac_layout import (
    pick_dcu_lds_layout,
    compute_dcu_lds_slabs,
    dtype_elem_bytes,
)
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


# Producer GEMM C fragment -> M-side pack. LayoutMap probing is unreliable
# (C++ may not pass it; BindThreadRange makes ForwardThread non-const).
# InferLayout visits gemms in program order, so GEMM1 registers before GEMM2.
_FRAGMENT_M_PACK = {}


def _buf_data(buf):
    return getattr(buf, "data", buf)


def _register_fragment_m_pack(buf, m_extent: int, m_pack: int):
    if buf is None or not is_fragment(buf):
        return
    _FRAGMENT_M_PACK[_buf_data(buf)] = (int(m_extent), int(m_pack))


def _lookup_fragment_m_pack(buf, m_extent: int):
    if buf is None:
        return None
    key = _buf_data(buf)
    if key in _FRAGMENT_M_PACK:
        stored_m, stored_pack = _FRAGMENT_M_PACK[key]
        if stored_m == int(m_extent):
            return stored_pack
    packs = [pack for (stored_m, pack) in _FRAGMENT_M_PACK.values() if stored_m == int(m_extent)]
    return max(packs) if packs else None


def _expr_const_int(expr):
    if expr is None:
        return None
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    try:
        simplified = tvm.arith.Analyzer().simplify(expr)
        if isinstance(simplified, tir.IntImm):
            return int(simplified.value)
    except Exception:
        return None
    return None


def _recover_m_pack_from_fragment(layout, m_extent: int):
    if not isinstance(layout, Fragment):
        return None
    shape = [int(x) for x in layout.get_input_shape()]
    if len(shape) < 2 or shape[0] != m_extent:
        return None
    t0 = _expr_const_int(layout.map_forward_thread([0, 0]))
    if t0 is None:
        return None
    warp0 = t0 // 64
    group = 16
    while group * 2 <= shape[0]:
        t = _expr_const_int(layout.map_forward_thread([group, 0]))
        if t is None or (t // 64) != warp0:
            break
        group *= 2
    return max(1, group // 16)


def _inherit_m_pack_from_layout_map(layout_map, m_extent: int, a_buf=None):
    if layout_map is None:
        return None
    packs = []
    candidates = []
    if a_buf is not None:
        try:
            if a_buf in layout_map:
                candidates.append(layout_map[a_buf])
        except Exception:
            pass
    try:
        candidates.extend(layout for _buf, layout in layout_map.items())
    except Exception:
        pass
    for layout in candidates:
        recovered = _recover_m_pack_from_fragment(layout, m_extent)
        if recovered is not None:
            packs.append(recovered)
    return max(packs) if packs else None


def _resolve_dcu_lds_layouts(gemm, thread_nums: int):
    """Per-operand SIMPLE / SWIZZLE / SLAB / REWIND, matching copy.cc heuristics."""
    user_mode = gemm.lds_layout
    k_pack = int(gemm.k_pack or 1)
    elem = dtype_elem_bytes(gemm.in_dtype)
    a_k_axis = 0 if gemm.trans_A else 1
    b_k_axis = 1 if gemm.trans_B else 0
    warp_num = max(1, int(thread_nums) // 64)
    a_mode = pick_dcu_lds_layout(
        user_mode,
        compute_dcu_lds_slabs(int(gemm.M), k_pack, elem),
        int(gemm.M),
        a_k_axis,
        k_pack,
        warp_num,
    )
    b_mode = pick_dcu_lds_layout(
        user_mode,
        compute_dcu_lds_slabs(int(gemm.N), k_pack, elem),
        int(gemm.N),
        b_k_axis,
        k_pack,
        warp_num,
    )
    return a_mode, b_mode


def _resolve_m_pack(gemm, layout_map):
    k_pack = int(gemm.k_pack or 1)
    m_extent = int(gemm.M)
    # Shared-A GEMMs originate a new C fragment ABI. Drop leftover packs from
    # the previous kernel so a k_pack=1 compile cannot inherit k_pack=2.
    if gemm.is_gemm_ss() or gemm.is_gemm_sr():
        _FRAGMENT_M_PACK.clear()
        _register_fragment_m_pack(gemm.C, m_extent, k_pack)
        return k_pack
    inherited = _lookup_fragment_m_pack(gemm.A, m_extent)
    probed = _inherit_m_pack_from_layout_map(layout_map, m_extent, gemm.A)
    candidates = [p for p in (inherited, probed) if p is not None]
    m_pack = max(candidates) if candidates else k_pack
    _register_fragment_m_pack(gemm.C, m_extent, m_pack)
    _register_fragment_m_pack(gemm.A, m_extent, m_pack)
    return m_pack


class GemmMMAC(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int, layout_map=None):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, False)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        a_lds_layout, b_lds_layout = _resolve_dcu_lds_layouts(self, thread_nums)
        m_pack = _resolve_m_pack(self, layout_map)
        
        mmac_emitter = MatrixCoreIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            k_pack=self.k_pack,
            m_pack=m_pack,
            a_lds_layout=a_lds_layout,
            b_lds_layout=b_lds_layout,
            transpose_C=self.trans_C,
        )

        if self.is_gemm_ss():
            return {
                self.A: make_linear_layout(self.A),
                self.B: make_linear_layout(self.B),
                self.C: mmac_emitter.make_mmac_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.A: make_linear_layout(self.A),
                self.B: mmac_emitter.make_mmac_load_layout(self.B, matrix="B"),
                self.C: mmac_emitter.make_mmac_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            return {
                self.A: mmac_emitter.make_mmac_load_layout(self.A, matrix="A"),
                self.B: make_linear_layout(self.B),
                self.C: mmac_emitter.make_mmac_store_layout(self.C),
            }
        elif self.is_gemm_rr():
            return {
                self.A: mmac_emitter.make_mmac_load_layout(self.A, matrix="A"),
                self.B: mmac_emitter.make_mmac_load_layout(self.B, matrix="B"),
                self.C: mmac_emitter.make_mmac_store_layout(self.C),
            }
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(self, layout_map: dict, target: Target, thread_nums: int, thread_var: tir.Var):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, False)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        a_lds_layout, b_lds_layout = _resolve_dcu_lds_layouts(self, thread_nums)
        m_pack = _resolve_m_pack(self, layout_map)
        mmac_emitter = MatrixCoreIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_var,
            k_pack=self.k_pack,
            m_pack=m_pack,
            a_lds_layout=a_lds_layout,
            b_lds_layout=b_lds_layout,
            transpose_C=self.trans_C,
        )

        in_dtype = self.in_dtype
        warp_rows = mmac_emitter.warp_rows
        warp_cols = mmac_emitter.warp_cols
        local_size_a = mmac_emitter.local_size_a
        local_size_b = mmac_emitter.local_size_b
        block_K = mmac_emitter.chunk
        micro_size_k = mmac_emitter.micro_size_k
        # Use region for shared-memory operands if available
        # We use region for memory input to support strided gemm
        # T.gemm(A_shared[0:128, :], B_shared, C_local)
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        clear_accum = self.clear_accum

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"

        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Matrix Core mmac ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a * self.k_pack), in_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b * self.k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    # Load A into fragment
                    mmac_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Load B into fragment
                    if self.trans_B:
                        mmac_emitter.ldmatrix_b(
                            B_local,
                            B_region,
                            ki,
                        )
                    else:
                        for kp in T.serial(0, self.k_pack):
                            mmac_emitter.ldmatrix_b(
                                B_local,
                                B_region,
                                ki,
                                kp,
                            )

                    # Perform Matrix Multiplication
                    mmac_emitter.mmac(A_local, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_sr():
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_srr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Matrix Core mmac ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a * self.k_pack), in_dtype)

                if clear_accum:
                    T.clear(C_buf)

                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    # Load A into fragment
                    mmac_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mmac_emitter.mmac(A_local, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            # alloc_buffers body
            # insert into parent block
            return _Simplify(_gemm_srr, inline_let=True)
        elif self.is_gemm_rs():
            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Matrix Core mmac ops,
                accumulating into C_local.
                """
                B_local = T.alloc_local((warp_cols * local_size_b * self.k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    # Load B into fragment
                    mmac_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mmac_emitter.mmac(A_buf, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        elif self.is_gemm_rr():
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Matrix Core mmac ops,
                accumulating into C_local.
                """

                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    # Perform Matrix Multiplication
                    mmac_emitter.mmac(A_buf, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
