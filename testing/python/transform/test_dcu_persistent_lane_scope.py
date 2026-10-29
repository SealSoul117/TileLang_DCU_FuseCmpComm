"""Scope tests for the verified persistent-GEMM direct-lane exception.

These probes construct the internal verified annotation directly to isolate
InjectWarpIdx's gate.  Acceptance of that annotation remains owned by
LowerSharedGlobalCopy and is exercised below with the real examples.
"""
import importlib.util
from pathlib import Path

import tilelang
from tilelang import tvm
from tvm.script import tir as T


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = ROOT / "examples" / "distributed" / "gemm_rs_prototype"
GFX936 = "hip -mcpu=gfx936"
GFX942 = "hip -mcpu=gfx942"


def _inject(func, target=GFX936):
    return _inject_func(func, target).script()


def _inject_func(func, target=GFX936):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with tvm.target.Target(target):
        return tilelang.transform.InjectWarpIdx()(mod)["main"]


def _find_warp_let_and_div_by_one(func):
    warp_let = None
    div_by_one = None

    def visit(node):
        nonlocal warp_let, div_by_one
        if isinstance(node, tvm.tir.LetStmt) and isinstance(node.value, tvm.tir.Call):
            if node.value.op.name == "tl.get_warp_idx_sync":
                warp_let = node
        if isinstance(node, tvm.tir.FloorDiv) and isinstance(node.b, tvm.tir.IntImm):
            if int(node.b.value) == 1:
                div_by_one = node

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    assert warp_let is not None and div_by_one is not None
    return warp_let, div_by_one


@T.prim_func
def _verified_marker_with_sibling(
    src: T.Buffer((64,), "int32"), out: T.Buffer((1024,), "int32"),
):
    tx = T.launch_thread("threadIdx.x", 512)
    for tile in T.serial(
        1,
        annotations={
            "tl.dcu_persistent_tile": 1,
            "tl.dcu_persistent_resource_verified": 1,
        },
    ):
        out[tx] = src[tx % 64] + tx // 64
    # This sibling is deliberately unmarked and must retain the native lane
    # helper even in the same thread binding.
    out[512 + tx] = src[tx % 64]


@T.prim_func
def _marker_only(
    src: T.Buffer((64,), "int32"), out: T.Buffer((512,), "int32"),
):
    tx = T.launch_thread("threadIdx.x", 512)
    for tile in T.serial(1, annotations={"tl.dcu_persistent_tile": 1}):
        out[tx] = src[tx % 64]


@T.prim_func
def _verified_only(
    src: T.Buffer((64,), "int32"), out: T.Buffer((512,), "int32"),
):
    tx = T.launch_thread("threadIdx.x", 512)
    for tile in T.serial(1, annotations={"tl.dcu_persistent_resource_verified": 1}):
        out[tx] = src[tx % 64]


@T.prim_func
def _verified_marker_256_threads(
    src: T.Buffer((64,), "int32"), out: T.Buffer((256,), "int32"),
):
    tx = T.launch_thread("threadIdx.x", 256)
    for tile in T.serial(
        1,
        annotations={
            "tl.dcu_persistent_tile": 1,
            "tl.dcu_persistent_resource_verified": 1,
        },
    ):
        out[tx] = src[tx % 64]


@T.prim_func
def _verified_marker_with_inner_y(
    src: T.Buffer((64,), "int32"), out: T.Buffer((1024,), "int32"),
):
    tx = T.launch_thread("threadIdx.x", 512)
    for tile in T.serial(
        1,
        annotations={
            "tl.dcu_persistent_tile": 1,
            "tl.dcu_persistent_resource_verified": 1,
        },
    ):
        # This binding is deliberately nested in the otherwise-eligible
        # marker body.  RewriteExpr must re-check current dimensions, rather
        # than trusting only the marker entry decision.
        ty = T.launch_thread("threadIdx.y", 2)
        out[ty * 512 + tx] = src[tx % 64]


@T.prim_func
def _verified_marker_with_outer_y(
    src: T.Buffer((64,), "int32"), out: T.Buffer((1024,), "int32"),
):
    tx = T.launch_thread("threadIdx.x", 512)
    ty = T.launch_thread("threadIdx.y", 2)
    for tile in T.serial(
        1,
        annotations={
            "tl.dcu_persistent_tile": 1,
            "tl.dcu_persistent_resource_verified": 1,
        },
    ):
        out[ty * 512 + tx] = src[tx % 64]


def test_verified_marker_uses_direct_lane_but_sibling_keeps_native_helper():
    transformed = _inject_func(_verified_marker_with_sibling)
    source = transformed.script()
    assert source.count("T.get_lane_idx(64)") == 1
    assert "out[tx] = src[tx % 64]" in source
    assert "out[512 + tx] = src[T.get_lane_idx(64)]" in source
    # The historical rewriter leaves warp as FloorDiv(warp_idx, 1).  Check
    # its structure and Analyzer equivalence rather than requiring a separate
    # Simplify pass to print the cosmetic identity away.
    warp_let, div_by_one = _find_warp_let_and_div_by_one(transformed)
    assert isinstance(warp_let.value.args[0], tvm.tir.IntImm)
    assert int(warp_let.value.args[0].value) == 64
    thread_var = warp_let.value.args[1]
    expected_warp = tvm.tir.FloorDiv(thread_var, tvm.tir.IntImm(thread_var.dtype, 64))
    substituted = tvm.tir.stmt_functor.substitute(div_by_one, {warp_let.var: expected_warp})
    assert tvm.ir.structural_equal(
        tvm.arith.Analyzer().simplify(substituted), expected_warp, map_free_vars=True,
    )


def test_marker_without_resource_proof_and_outside_profile_keep_helper():
    assert _inject(_marker_only).count("T.get_lane_idx(64)") == 1
    assert _inject(_verified_only).count("T.get_lane_idx(64)") == 1
    original = tvm.IRModule.from_expr(
        _verified_marker_with_sibling.with_attr("global_symbol", "main")
    )["main"]
    tvm.ir.assert_structural_equal(
        original, _inject_func(_verified_marker_with_sibling, GFX942), map_free_vars=True,
    )
    assert _inject(_verified_marker_256_threads).count("T.get_lane_idx(64)") == 1
    # Active y dimensions both outside and inside a marker revoke the
    # exception for the affected lexical body.
    assert _inject(_verified_marker_with_inner_y).count("T.get_lane_idx(64)") == 1
    assert _inject(_verified_marker_with_outer_y).count("T.get_lane_idx(64)") == 1


def _load_module(name):
    path = EXAMPLE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_verified_persistent_examples_avoid_mbcnt_lane_helper(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    pure = _load_module("gemm_persistent_resource")
    scheduled = _load_module("gemm_reduce_scatter_scheduled")
    with tvm.target.Target(GFX936):
        pure_artifact = tilelang.lower(
            pure.persistent_gemm(2048, 2048, 512), target=GFX936,
            enable_device_compile=False, enable_host_codegen=False,
        )
        _, scheduled_func = scheduled.build_bound(2048, 2048, 512, total_blocks=80, comm_blocks=32)
        scheduled_artifact = tilelang.lower(
            scheduled_func, target=GFX936,
            enable_device_compile=False, enable_host_codegen=False,
        )
    # Neither example has a non-persistent GEMM sibling.  Their communication
    # path remains opaque_thread_id and does not use the lane helper.
    for source in (pure_artifact.kernel_source, scheduled_artifact.kernel_source):
        assert "tl::get_lane_idx(64)" not in source
        assert "tl::get_warp_idx_sync(64, " in source
