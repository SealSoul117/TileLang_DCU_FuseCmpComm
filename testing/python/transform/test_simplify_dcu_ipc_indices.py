"""Scope tests for the verified IPC opaque-thread index simplification.

The range fact is intentionally internal to ``tl.transform.Simplify``.  These
tests verify that it is available only below the exact opaque helper Let and
never becomes an IR assumption or a generic thread-index rewrite.
"""

import importlib.util
from pathlib import Path

import tilelang
from tilelang import tvm
from tvm.script import tir as T


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = ROOT / "examples" / "distributed" / "gemm_rs_prototype"
GFX936 = tvm.target.Target({"kind": "hip", "mcpu": "gfx936", "max_num_threads": 256})
GFX942 = tvm.target.Target({"kind": "hip", "mcpu": "gfx942", "max_num_threads": 256})
CUDA = tvm.target.Target("cuda -arch=sm_80")
OPAQUE = "tl::ipc::opaque_thread_id"
CALL_EXTERN = tvm.ir.Op.get("tir.call_extern")


def _bound(func):
    return func.with_attr("tl.dcu_ipc.abi_ready", 1).with_attr(
        "tl.dcu_ipc.bindings", "partial=ipc_view_partial",
    )


def _apply(func, target=GFX936):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    return tilelang.transform.Simplify()(mod)["main"]


def _control(func, target=GFX936):
    """The ordinary Simplify result, without the ABI range gate."""
    return _apply(func, target)


def _all_nodes(stmt, node_type):
    found = []

    def visit(node):
        if isinstance(node, node_type):
            found.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return found


def _contains_var(expr, var):
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tvm.tir.Var) and node.same_as(var):
            found = True

    tvm.tir.stmt_functor.post_order_visit(expr, visit)
    return found


def _extern_names(stmt):
    names = []

    def visit(node):
        if (
            isinstance(node, tvm.tir.Call)
            and node.op.same_as(CALL_EXTERN)
            and node.args
            and isinstance(node.args[0], tvm.tir.StringImm)
        ):
            names.append(node.args[0].value)

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return names


def _opaque_let(func):
    matches = []
    for let in _all_nodes(func.body, tvm.tir.LetStmt):
        if (
            isinstance(let.value, tvm.tir.Call)
            and let.value.op.same_as(CALL_EXTERN)
            and isinstance(let.value.args[0], tvm.tir.StringImm)
            and let.value.args[0].value == OPAQUE
        ):
            matches.append(let)
    assert len(matches) == 1
    return matches[0]


@T.prim_func
def _eligible(
    out: T.Buffer((128,), "int32"), rank: T.int32,
):
    tx = T.launch_thread("threadIdx.x", 512)
    comm_tx: T.int32 = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
    if comm_tx < 65536:
        for i in T.serial((66047 - comm_tx) // 512):
            # The first conjunct is redundant only after the lexical opaque
            # range proof.  The rank predicate is intentionally not proven.
            if 0 <= i * 2 + comm_tx // 256 and 3 <= rank * 4 and rank * 4 < 19:
                out[i] = 1
            # This matches the output-side non-negative guard shape.
            if 0 <= i % 4 * 256 + i * 2 + comm_tx // 256:
                out[i] = 2
            # max_num_threads=256 on the target must not make high lanes dead:
            # the actual launch extent is 512.
            if comm_tx >= 256:
                out[i] = 3


@T.prim_func
def _unknown_helper(out: T.Buffer((128,), "int32")):
    tx = T.launch_thread("threadIdx.x", 512)
    comm_tx: T.int32 = T.call_extern("int32", "other_opaque_thread_id", tx)
    if comm_tx < 65536:
        for i in T.serial((66047 - comm_tx) // 512):
            if 0 <= i * 2 + comm_tx // 256:
                out[i] = 1


@T.prim_func
def _non_var_argument(out: T.Buffer((128,), "int32")):
    tx = T.launch_thread("threadIdx.x", 512)
    comm_tx: T.int32 = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx + 1)
    if comm_tx < 65536:
        for i in T.serial((66047 - comm_tx) // 512):
            if 0 <= i * 2 + comm_tx // 256:
                out[i] = 1


@T.prim_func
def _no_thread_scope(out: T.Buffer((128,), "int32"), seed: T.int32):
    comm_tx: T.int32 = T.call_extern("int32", "tl::ipc::opaque_thread_id", seed)
    if comm_tx < 65536:
        for i in T.serial((66047 - comm_tx) // 512):
            if 0 <= i * 2 + comm_tx // 256:
                out[i] = 1


def _pointer(name, dtype):
    return tvm.tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), "global"))


def _manual_thread_kernel(thread_extent, *, helper=OPAQUE, argument_is_var=True,
                          tail=66047, dtype="int32", side_effect_extent=False,
                          side_effect_body=False):
    """Build unusual AttrStmt shapes that script syntax intentionally avoids."""
    out = _pointer("out", "int32")
    out_buffer = tvm.tir.decl_buffer((256,), "int32", data=out, name="out", scope="global")
    tx = tvm.tir.Var("tx", dtype)
    comm_tx = tvm.tir.Var("comm_tx", dtype)
    i = tvm.tir.Var("i", dtype)
    argument = tx if argument_is_var else tx + tvm.tir.const(1, dtype)
    call = tvm.tir.call_extern(dtype, helper, argument)
    extent = (tvm.tir.const(tail, dtype) - comm_tx) // tvm.tir.const(512, dtype)
    if side_effect_extent:
        extent = tvm.tir.call_extern(dtype, "side_effect_extent", comm_tx)
    body = tvm.tir.BufferStore(out_buffer, tvm.tir.const(1, "int32"), [i])
    if side_effect_body:
        body = tvm.tir.Evaluate(tvm.tir.call_extern("int32", "loop_effect", comm_tx))
    loop = tvm.tir.For(
        i, 0, extent, tvm.tir.ForKind.SERIAL, body,
    )
    body = tvm.tir.LetStmt(comm_tx, call, loop)
    thread = tvm.tir.IterVar(None, tx, tvm.tir.IterVar.ThreadIndex, "threadIdx.x")
    params = [out]
    if isinstance(thread_extent, tvm.tir.Var):
        params.append(thread_extent)
    return tvm.tir.PrimFunc(
        params, tvm.tir.AttrStmt(thread, "thread_extent", thread_extent, body),
        buffer_map={out: out_buffer},
    )


def _assert_body_unchanged_by_gate(func, target=GFX936):
    candidate = _apply(_bound(func), target)
    baseline = _control(func, target)
    tvm.ir.assert_structural_equal(candidate.body, baseline.body, map_free_vars=True)


@T.prim_func
def _dynamic_thread_extent(out: T.Buffer((128,), "int32"), extent: T.int32):
    tx = T.launch_thread("threadIdx.x", extent)
    comm_tx: T.int32 = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
    if comm_tx < 65536:
        for i in T.serial((66047 - comm_tx) // 512):
            if 0 <= i * 2 + comm_tx // 256:
                out[i] = 1


def test_verified_opaque_range_uses_launch_512_not_target_256_and_keeps_rank_guard():
    transformed = _apply(_bound(_eligible))
    opaque = _opaque_let(transformed)
    assert isinstance(opaque.value, tvm.tir.Call)
    # The helper Call itself remains opaque and is not replaced by threadIdx.x.
    assert isinstance(opaque.value.args[1], tvm.tir.Var)
    assert not opaque.value.args[1].same_as(opaque.var)

    loops = _all_nodes(transformed.body, tvm.tir.For)
    assert len(loops) == 1
    assert isinstance(loops[0].extent, tvm.tir.IntImm)
    assert int(loops[0].extent.value) == 128

    conditions = [stmt.condition for stmt in _all_nodes(transformed.body, tvm.tir.IfThenElse)]
    # The outer zero-trip guard and both non-negative index guards are proven
    # true.  In particular, the high-half predicate is not proven false from
    # target.max_num_threads=256.
    assert all(not _contains_var(condition, opaque.var) or "65536" not in str(condition)
               for condition in conditions)
    assert not any("0 <=" in str(condition) and _contains_var(condition, opaque.var)
                   for condition in conditions)
    assert any(_contains_var(condition, opaque.var) and "256" in str(condition)
               for condition in conditions)
    # rank is a runtime ABI parameter, so its tile predicate remains.
    rank = transformed.params[1]
    rank_conditions = [condition for condition in conditions if _contains_var(condition, rank)]
    assert len(rank_conditions) == 1
    original_rank_guard = tvm.tir.And(3 <= rank * 4, rank * 4 < 19)
    analyzer = tvm.arith.Analyzer()
    tvm.ir.assert_structural_equal(
        analyzer.simplify(rank_conditions[0]), analyzer.simplify(original_rank_guard),
        map_free_vars=True,
    )


def test_only_the_exact_verified_hip_attr_scope_enables_the_range_fact():
    controls = (
        (_bound(_unknown_helper), _unknown_helper, GFX936),
        (_bound(_non_var_argument), _non_var_argument, GFX936),
        (_bound(_no_thread_scope), _no_thread_scope, GFX936),
        (_bound(_dynamic_thread_extent), _dynamic_thread_extent, GFX936),
        (_bound(_eligible), _eligible, GFX942),
        (_bound(_eligible), _eligible, CUDA),
        (_eligible.with_attr("tl.dcu_ipc.abi_ready", 1), _eligible, GFX936),
    )
    for func, unbound, target in controls:
        candidate = _apply(func, target)
        # Both sides run ordinary Simplify under the same target; only the
        # ABI gate differs.
        baseline = _control(unbound, target)
        tvm.ir.assert_structural_equal(candidate.body, baseline.body, map_free_vars=True)


def test_malformed_extent_dtype_tail_effect_and_scope_identity_fail_closed():
    dynamic_extent = tvm.tir.Var("dynamic_extent", "int32")
    # Zero, negative, dynamic and wrong-width launch extents never establish a
    # helper range.  A different helper or non-Var helper argument is likewise
    # not a contract match.
    for func in (
        _manual_thread_kernel(tvm.tir.const(0, "int32")),
        _manual_thread_kernel(tvm.tir.const(-1, "int32")),
        _manual_thread_kernel(dynamic_extent),
        _manual_thread_kernel(tvm.tir.const(512, "int64"), dtype="int64"),
        _manual_thread_kernel(tvm.tir.const(512, "int32"), helper="other_helper"),
        _manual_thread_kernel(tvm.tir.const(512, "int32"), argument_is_var=False),
        # 66048 makes the trip count 128 or 129 across legal lanes; it must
        # stay symbolic rather than being forced to the scheduled 128 value.
        _manual_thread_kernel(tvm.tir.const(512, "int32"), tail=66048),
        # An opaque extent cannot be replaced merely because this is inside an
        # otherwise valid helper Let.
        _manual_thread_kernel(tvm.tir.const(512, "int32"), side_effect_extent=True),
    ):
        _assert_body_unchanged_by_gate(func)

    nondivisible = _apply(_bound(_manual_thread_kernel(tvm.tir.const(512, "int32"), tail=66048)))
    loop = _all_nodes(nondivisible.body, tvm.tir.For)[0]
    assert not isinstance(loop.extent, tvm.tir.IntImm)
    assert "tl.assume" not in nondivisible.script()


def test_valid_512_zero_trip_and_single_lane_tail_keep_loop_effects_in_place():
    # These are valid opaque-helper scopes.  The first trip count is exactly
    # zero for every legal lane; the second is one only for lane 0.  Neither
    # permits a loop-body call to escape the serial loop.
    all_zero = _apply(_bound(_manual_thread_kernel(
        tvm.tir.const(512, "int32"), tail=511, side_effect_body=True,
    )))
    zero_loop = _all_nodes(all_zero.body, tvm.tir.For)[0]
    assert isinstance(zero_loop.extent, tvm.tir.IntImm)
    assert int(zero_loop.extent.value) == 0
    assert "loop_effect" in _extern_names(zero_loop.body)

    one_lane = _apply(_bound(_manual_thread_kernel(
        tvm.tir.const(512, "int32"), tail=512, side_effect_body=True,
    )))
    one_lane_loop = _all_nodes(one_lane.body, tvm.tir.For)[0]
    assert not isinstance(one_lane_loop.extent, tvm.tir.IntImm)
    assert "loop_effect" in _extern_names(one_lane_loop.body)
    trip_counts = [(512 - lane) // 512 for lane in range(512)]
    assert trip_counts.count(1) == 1 and trip_counts.count(0) == 511


def test_nested_dynamic_thread_scope_and_same_name_different_var_do_not_leak_range():
    out = _pointer("out", "int32")
    out_buffer = tvm.tir.decl_buffer((256,), "int32", data=out, name="out", scope="global")
    outer_tx = tvm.tir.Var("tx", "int32")
    inner_tx = tvm.tir.Var("tx_inner", "int32")
    dynamic_extent = tvm.tir.Var("dynamic_extent", "int32")
    comm_tx = tvm.tir.Var("comm_tx", "int32")
    i = tvm.tir.Var("i", "int32")
    loop = tvm.tir.For(
        i, 0, (66047 - comm_tx) // 512, tvm.tir.ForKind.SERIAL,
        tvm.tir.BufferStore(out_buffer, tvm.tir.const(1, "int32"), [i]),
    )
    inner = tvm.tir.AttrStmt(
        tvm.tir.IterVar(None, inner_tx, tvm.tir.IterVar.ThreadIndex, "threadIdx.x"),
        "thread_extent", dynamic_extent,
        tvm.tir.LetStmt(comm_tx, tvm.tir.call_extern("int32", OPAQUE, outer_tx), loop),
    )
    outer = tvm.tir.AttrStmt(
        tvm.tir.IterVar(None, outer_tx, tvm.tir.IterVar.ThreadIndex, "threadIdx.x"),
        "thread_extent", tvm.tir.const(512, "int32"), inner,
    )
    nested = tvm.tir.PrimFunc(
        [out, dynamic_extent], outer, buffer_map={out: out_buffer},
    )
    _assert_body_unchanged_by_gate(nested)

    # The parameter and the Let deliberately share a spelling but not a Var
    # identity.  The lexical fact for the Let may not simplify the sibling.
    sibling_tx = tvm.tir.Var("comm_tx", "int32")
    first_tx = tvm.tir.Var("comm_tx", "int32")
    first_i = tvm.tir.Var("i", "int32")
    first = tvm.tir.LetStmt(
        first_tx, tvm.tir.call_extern("int32", OPAQUE, outer_tx),
        tvm.tir.For(
            first_i, 0, (66047 - first_tx) // 512, tvm.tir.ForKind.SERIAL,
            tvm.tir.BufferStore(out_buffer, tvm.tir.const(1, "int32"), [first_i]),
        ),
    )
    sibling = tvm.tir.IfThenElse(
        sibling_tx < 65536,
        tvm.tir.BufferStore(out_buffer, tvm.tir.const(7, "int32"), [0]),
        None,
    )
    outer_sibling = tvm.tir.AttrStmt(
        tvm.tir.IterVar(None, outer_tx, tvm.tir.IterVar.ThreadIndex, "threadIdx.x"),
        "thread_extent", tvm.tir.const(512, "int32"), tvm.tir.SeqStmt([first, sibling]),
    )
    same_name = tvm.tir.PrimFunc(
        [out, sibling_tx], outer_sibling, buffer_map={out: out_buffer},
    )
    transformed = _apply(_bound(same_name))
    assert any(
        _contains_var(stmt.condition, sibling_tx) and "65536" in str(stmt.condition)
        for stmt in _all_nodes(transformed.body, tvm.tir.IfThenElse)
    )


def _load_module(name):
    path = EXAMPLE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_scheduled_source_keeps_opaque_helper_and_runtime_rank_guard(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    scheduled = _load_module("gemm_reduce_scatter_scheduled")
    _, func = scheduled.build_bound(4096, 4096, 1024, total_blocks=80, comm_blocks=32)
    artifact = tilelang.lower(
        func, target="hip -mcpu=gfx936", enable_device_compile=False,
        enable_host_codegen=False,
    )
    source = artifact.kernel_source
    assert "tl::ipc::opaque_thread_id(((int)threadIdx.x))" in source
    assert "for (int _tmp_2 = 0; _tmp_2 < 128; ++_tmp_2)" in source
    assert "if (comm_tx < 65536)" not in source
    # The rank-dependent tile validity test is not a thread-range fact.
    assert "3 <= (((__tl_ipc_rank * 4)" in source
    assert "< 19" in source
