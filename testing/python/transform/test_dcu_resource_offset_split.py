"""CPU TIR regressions for DCU resource-base / async-offset splitting."""
import importlib.util
from pathlib import Path
import re
import sys

import tilelang
from tilelang import tvm
from tvm.script import tir as T


ROOT = Path(__file__).resolve().parents[3]
PROTOTYPE_DIR = ROOT / "examples" / "distributed" / "gemm_rs_prototype"
TARGET = "hip -mcpu=gfx936"


@T.prim_func
def _resource_split_probe(
    a: T.Buffer((131072,), "float16"), b: T.Buffer((131072,), "float16"),
):
    """Real 16-byte vector stores consumed by LowerSharedGlobalCopy."""
    shared = T.alloc_buffer((256,), "float16", scope="shared")
    # Use launch_thread AttrStmts, matching the stage where this pass sees a
    # lowered TileLang kernel.  In particular blockIdx is intentionally *not*
    # a serial loop variable eligible for removal from a cached resource base.
    block = T.launch_thread("blockIdx.x", 80)
    thread = T.launch_thread("threadIdx.x", 8)
    # Six iterations cover every valid job=block+48*iteration below the
    # 4096 profile's 256 tiles for physical compute blocks 0..47.
    for iteration in T.serial(6):
        with T.attr(0, "async_scope", 1):
            # 4096 scheduled profile: job=block+48*iteration,
            # B tile column=(job//4)%16.
            shared[T.Ramp(thread * 8, 1, 8)] = b[T.Ramp(
                ((block + 48 * iteration) // 4) % 16 * 256 + thread * 8, 1, 8
            )]
            # Same B resource, a separate nonzero copy constant.
            shared[T.Ramp(64 + thread * 8, 1, 8)] = b[T.Ramp(
                ((block + 48 * iteration) // 4) % 16 * 256 + 64 + thread * 8, 1, 8
            )]
            # Ordinary affine source retains a nonzero common base.
            shared[T.Ramp(128 + thread * 8, 1, 8)] = a[T.Ramp(
                block * 1024 + iteration * 128 + 32 + thread * 8, 1, 8
            )]


def _var_name(var) -> str:
    return var.name if hasattr(var, "name") else var.name_hint


def _collect_resource_calls(func):
    """Return {source-name: (cached-base, [runtime-offsets])} from real TIR."""
    resource_source, resource_base, copies = {}, {}, {}

    def visit(node):
        if isinstance(node, tvm.tir.LetStmt) and isinstance(node.value, tvm.tir.Call):
            if node.value.op.name == "tl.make_dcu_resource":
                resource_source[_var_name(node.var)] = _var_name(node.value.args[0])
                base = node.value.args[1]
                resource_base[_var_name(node.var)] = base.base if isinstance(base, tvm.tir.Ramp) else base
        if isinstance(node, tvm.tir.Call) and node.op.name == "tl.dcu_async_copy":
            copies.setdefault(_var_name(node.args[2]), []).append(node.args[3])

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return {source: (resource_base[resource], copies[resource])
            for resource, source in resource_source.items()}


def _evaluate(expr, **values) -> int:
    variables = {}

    def visit(node):
        if isinstance(node, tvm.tir.Var) and _var_name(node) in values:
            variables[node] = tvm.tir.IntImm(node.dtype, values[_var_name(node)])

    tvm.tir.stmt_functor.post_order_visit(expr, visit)
    result = tvm.arith.Analyzer().simplify(tvm.tir.stmt_functor.substitute(expr, variables))
    assert isinstance(result, tvm.tir.IntImm), result
    return int(result.value)


def test_resource_offsets_keep_exact_original_address_and_never_go_negative():
    lowered = tilelang.transform.LowerSharedGlobalCopy()(
        tvm.IRModule({"main": _resource_split_probe})
    )["main"]
    resources = _collect_resource_calls(lowered)

    # B's floor/mod cross term falls back consistently for all copies: base=0
    # and each runtime value is the original flattened global index.
    b_base, b_offsets = resources["b"]
    assert len(b_offsets) == 2
    assert _evaluate(b_base, block=16, iteration=1, thread=0) == 0
    expected_b = (64 // 4) % 16 * 256
    assert [_evaluate(offset, block=16, iteration=1, thread=0) for offset in b_offsets] == [
        expected_b, expected_b + 64,
    ]
    # Exhaust the real compute schedule's valid jobs, both the first and last
    # vector thread, and both same-resource copy constants.  This is stronger
    # than a text check: base + runtime must remain the original flattened B
    # address across every floor/mod carry and no runtime offset may be below
    # zero.
    for block in range(48):
        for iteration in range(6):
            job = block + 48 * iteration
            if job >= 256:
                continue
            for thread in (0, 7):
                expected = (job // 4) % 16 * 256 + thread * 8
                actual_offsets = [_evaluate(offset, block=block, iteration=iteration, thread=thread)
                                  for offset in b_offsets]
                assert actual_offsets == [expected, expected + 64]
                assert all(offset >= 0 for offset in actual_offsets)
                assert [_evaluate(b_base + offset, block=block, iteration=iteration, thread=thread)
                        for offset in b_offsets] == [expected, expected + 64]

    # A's affine source keeps its first-copy nonzero base.  The residual and
    # reconstructed address are both exact for a representative loop/thread.
    a_base, a_offsets = resources["a"]
    assert len(a_offsets) == 1
    assert _evaluate(a_base, block=3, iteration=1, thread=2) == 3 * 1024 + 32
    assert _evaluate(a_offsets[0], block=3, iteration=1, thread=2) == 128 + 2 * 8
    assert _evaluate(a_base + a_offsets[0], block=3, iteration=1, thread=2) == (
        3 * 1024 + 128 + 32 + 2 * 8
    )


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, PROTOTYPE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_4096_marked_scheduled_profile_uses_tile_scoped_resources_and_flat_gemm_still_lowers():
    saved = {name: sys.modules.get(name) for name in (
        "gemm_reduce_scatter", "gemm_reduce_scatter_scheduled",
    )}
    try:
        base = _load_module("gemm_reduce_scatter")
        scheduled = _load_module("gemm_reduce_scatter_scheduled")
        _, matched = scheduled.build_bound(4096, 4096, 1024, total_blocks=80, comm_blocks=32)
        _, flat = base.build_bound(1024, 256, 256)
        with tvm.target.Target(TARGET):
            matched_tir = tilelang.lower(
                matched, target=TARGET, enable_device_compile=False, enable_host_codegen=False,
            ).device_mod.script()
            flat_tir = tilelang.lower(
                flat, target=TARGET, enable_device_compile=False, enable_host_codegen=False,
            ).device_mod.script()
        # The scheduled frontend now explicitly marks its outer output-tile
        # loop and spells operands A/B.  It therefore exercises the new
        # tile-scoped path rather than this file's first test's unmarked
        # zero-base fallback.  This expectation changes because the actual
        # 4096 owner-interleaved address path needs a pure local A/B offset;
        # it does not weaken that fallback's carry/non-negativity checks.
        assert "tl.dcu_persistent_tile" in matched_tir
        assert re.search(r"A_dcu_res.*T\.make_dcu_resource\(A,", matched_tir)
        assert re.search(r"B_dcu_res.*T\.make_dcu_resource\(B,", matched_tir)
        assert not re.search(r"B_dcu_res.*T\.make_dcu_resource\(B, 0\)", matched_tir)
        assert "T.dcu_async_copy" in matched_tir
        assert "T.make_dcu_resource(a," in flat_tir
        assert "T.make_dcu_resource(b," in flat_tir
        assert "T.dcu_async_copy" in flat_tir
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
