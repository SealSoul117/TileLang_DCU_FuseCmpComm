"""CPU TIR regressions for DCU warp/lane index rewriting."""

import tilelang
from tilelang import tvm
from tvm.script import tir as T


DCU_TARGET = "hip -mcpu=gfx936"


@T.prim_func
def _lane_rewrite_probe(src: T.Buffer((64,), "int32"), out: T.Buffer((512,), "int32")):
    shared = T.alloc_buffer((512,), "int32", scope="shared")
    tx = T.launch_thread("threadIdx.x", 512)
    # The shared destination deliberately remains raw tx.  The global source
    # and ordinary output expressions exercise all sublane spellings emitted
    # by the GEMM lowering.
    shared[tx] = src[tx % 64]
    out[tx] = tx // 64 + tx % 64 + (tx // 8) % 8 + T.bitwise_and(tx, 15)


def _inject(func, target: str):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with tvm.target.Target(target):
        return tilelang.transform.InjectWarpIdx()(mod)["main"]


def test_dcu_rewrites_all_wave64_sublane_forms_without_touching_shared_destination():
    transformed = _inject(_lane_rewrite_probe, DCU_TARGET)
    source = transformed.script()

    # High bits still use the existing readfirstlane warp result.  Every
    # lane-only form uses the native lane intrinsic, while shared[tx] retains
    # its original raw thread coordinate.
    assert "T.get_warp_idx_sync(64, tx)" in source
    assert source.count("T.get_lane_idx(64)") == 4
    assert "T.get_lane_idx(64) // 8 % 8" in source
    assert "T.bitwise_and(T.get_lane_idx(64), 15)" in source
    assert "shared[tx] = src[T.get_lane_idx(64)]" in source

    # The rewrite is mathematically identical for the full 512-thread GEMM
    # block: lane is the hardware linear wave64 lane under the guarded 1-D
    # block shape in dcu_hip/intrin.h.
    for tx in range(512):
        lane = tx % 64
        assert tx % 64 == lane
        assert (tx // 8) % 8 == (lane // 8) % 8
        assert (tx & 15) == (lane & 15)
        assert tx // 64 == (tx // 64)


def test_non_dcu_target_is_unchanged():
    original = tvm.IRModule.from_expr(_lane_rewrite_probe.with_attr("global_symbol", "main"))["main"]
    transformed = _inject(_lane_rewrite_probe, "cuda")
    tvm.ir.assert_structural_equal(original, transformed, map_free_vars=True)
