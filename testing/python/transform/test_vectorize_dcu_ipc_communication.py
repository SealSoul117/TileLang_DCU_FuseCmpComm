"""Scoped checks for the scheduled-RS IPC half8 communication rewrite.

The arithmetic enumeration below validates the mapper used by the narrow
backend matcher; it is not a substitute for the real scheduled lowering test,
which checks that the production normal form actually reaches the pass.
"""

import importlib.util
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = ROOT / "examples" / "distributed" / "gemm_rs_prototype"
AR_EXAMPLE_DIR = ROOT / "examples" / "distributed" / "gemm_ar_prototype"
AG_EXAMPLE_DIR = ROOT / "examples" / "distributed" / "ag_gemm_prototype"
TARGET = "hip -mcpu=gfx936"
REMOTE_ADDRESS = tvm.ir.Op.get("tl.ipc.remote_address")
FENCE = tvm.ir.Op.get("tl.ipc.fence_sys")


def _load_module(name, example_dir=EXAMPLE_DIR):
    spec = importlib.util.spec_from_file_location(name, example_dir / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(bound, *, enable_ipc_vectorize=False):
    config = {"tl.enable_dcu_ipc_vectorize": True} if enable_ipc_vectorize else {}
    with tvm.transform.PassContext(config=config):
        return tilelang.lower(
            bound, target=TARGET, enable_device_compile=False, enable_host_codegen=False,
        ).kernel_source


def test_half8_mapper_is_a_bijection_of_the_old_128x512_domain():
    covered = set()
    for new_i in range(16):
        for tx in range(512):
            for lane in range(8):
                old_i = 8 * new_i + tx // 64
                old_tx = 8 * (tx % 64) + lane
                assert 0 <= old_i < 128
                assert 0 <= old_tx < 512
                old_linear = old_i * 512 + old_tx
                assert old_linear == new_i * 4096 + tx * 8 + lane
                assert old_linear not in covered
                covered.add(old_linear)
    assert covered == set(range(128 * 512))


def test_non_ipc_primfunc_is_structurally_unchanged():
    data = tvm.tir.Var("data", tvm.ir.PointerType(tvm.ir.PrimType("float16"), "global"))
    buf = tvm.tir.decl_buffer((8,), "float16", data=data, scope="global")
    i = tvm.tir.Var("i", "int32")
    func = tvm.tir.PrimFunc(
        [data], tvm.tir.For(i, 0, 8, tvm.tir.ForKind.SERIAL,
                             tvm.tir.BufferStore(buf, tvm.tir.const(0, "float16"), [i])),
        buffer_map={data: buf},
    ).with_attr("target", tvm.target.Target(TARGET))
    result = tilelang.transform.VectorizeDCUIPCCommunication()(tvm.IRModule({"main": func}))["main"]
    tvm.ir.assert_structural_equal(func, result, map_free_vars=True)


def _ptr(name, dtype, scope="global"):
    return tvm.tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))


def _exact_candidate(*, source_bias=0, row_stride=4096, nonunit_y=False,
                     volatile=False, alias_temp=False, bad_descriptor=False,
                     data_alignment=64, elem_offset=0, remote_shape=65536, outer_scope=None):
    """A minimal exact normal form; negatives mutate one proof obligation."""
    out_handle, out_data = _ptr("out_handle", "float16"), _ptr("out_data", "float16")
    desc = _ptr("ipc_view_partial", "uint64")
    other_desc = _ptr("other_desc", "uint64")
    rank, world, tx, comm = (tvm.tir.Var("__tl_ipc_rank", "int32"),
                             tvm.tir.Var("__tl_ipc_world_size", "int32"),
                             tvm.tir.Var("tx", "int32"), tvm.tir.Var("comm_tx", "int32"))
    bx = tvm.tir.Var("bx", "int32") if outer_scope else None
    owned = tvm.tir.Var("owned", "int32") if outer_scope else None
    out = tvm.tir.decl_buffer(
        (1048576,), "float16", data=out_data, scope="global", data_alignment=data_alignment,
        elem_offset=elem_offset,
    )
    desc_buf = tvm.tir.decl_buffer((6,), "uint64", data=desc, scope="global", data_alignment=8)
    i = tvm.tir.Var("i", "int32")
    temp_vars = [_ptr(f"p{k}", "float16", "local") for k in range(4)]
    if alias_temp:
        temp_vars[1] = temp_vars[0]
    temps = [tvm.tir.decl_buffer((1,), "float16", data=p, scope="local") for p in temp_vars]
    remote_vars = [_ptr(f"remote{k}", "float16") for k in range(4)]
    remotes = [tvm.tir.decl_buffer(
        (remote_shape,), "float16", data=p, scope="global", data_alignment=data_alignment,
        elem_offset=elem_offset,
    )
               for p in remote_vars]
    # Preserve a uniform but unknown predicate; a constant True could be
    # folded before the matcher sees the production predicated-pull form.
    pred = rank >= tvm.tir.const(0, "int32")
    source_prefix = owned * 65536 if outer_scope in ("then", "sibling", "else_let", "unknown") else tvm.tir.const(0, "int32")
    pulls = [tvm.tir.BufferStore(
        temps[k], tvm.tir.if_then_else(pred, tvm.tir.BufferLoad(
            remotes[k], [source_prefix + i * 512 + comm + source_bias]),
                                        tvm.tir.const(0, "float16")), [0]) for k in range(4)]
    values = [tvm.tir.Cast("float32", tvm.tir.BufferLoad(temp, [0])) for temp in temps]
    value = values[0] + values[1] + values[2] + values[3]
    out_index = i * (2 * row_stride) + comm // 256 * row_stride + comm % 256
    body = tvm.tir.SeqStmt([*pulls, tvm.tir.BufferStore(out, tvm.tir.Cast("float16", value), [out_index])])
    loop = tvm.tir.For(i, 0, 128, tvm.tir.ForKind.SERIAL, body)
    chain = loop
    for k in reversed(range(4)):
        descriptor = other_desc if bad_descriptor and k == 0 else desc
        call = tvm.tir.Call(remote_vars[k].dtype, REMOTE_ADDRESS,
                            [descriptor, tvm.tir.const(k, "int32"), rank, world])
        chain = tvm.tir.LetStmt(remote_vars[k], call, chain)
    fence = tvm.tir.Evaluate(tvm.tir.Call("int32", FENCE, [rank, world]))
    opaque = tvm.tir.LetStmt(comm, tvm.tir.call_extern("int32", "tl::ipc::opaque_thread_id", tx),
                             tvm.tir.SeqStmt([chain, fence]))
    x_iv = tvm.tir.IterVar(None, tx, tvm.tir.IterVar.ThreadIndex, "threadIdx.x")
    stmt = tvm.tir.AttrStmt(x_iv, "thread_extent", tvm.tir.const(512, "int32"), opaque)
    if nonunit_y:
        y = tvm.tir.Var("ty", "int32")
        y_iv = tvm.tir.IterVar(None, y, tvm.tir.IterVar.ThreadIndex, "threadIdx.y")
        stmt = tvm.tir.AttrStmt(y_iv, "thread_extent", tvm.tir.const(2, "int32"), stmt)
    if volatile:
        stmt = tvm.tir.AttrStmt(tvm.tir.const(0, "int32"), "volatile_scope", tvm.tir.const(1, "int32"), stmt)
    for p in reversed(list(dict.fromkeys(temp_vars))):
        stmt = tvm.tir.Allocate(p, "float16", [1], tvm.tir.const(True, "bool"), stmt)
    if outer_scope:
        # The serial min depends on blockIdx.  The then case is safe only when
        # its branch fact is active; the sibling form verifies no branch fact
        # leaks into a later candidate.
        if outer_scope == "then":
            stmt = tvm.tir.For(owned, bx, 1, tvm.tir.ForKind.SERIAL, stmt)
            stmt = tvm.tir.IfThenElse(bx < 1, stmt, None)
        elif outer_scope == "sibling":
            stmt = tvm.tir.For(owned, bx, 1, tvm.tir.ForKind.SERIAL, stmt)
            stmt = tvm.tir.SeqStmt([tvm.tir.IfThenElse(bx < 1, tvm.tir.Evaluate(0), None), stmt])
        elif outer_scope == "else_let":
            base = tvm.tir.Var("base", "int32")
            stmt = tvm.tir.For(owned, base, 1, tvm.tir.ForKind.SERIAL, stmt)
            stmt = tvm.tir.LetStmt(base, bx - 1, stmt)
            stmt = tvm.tir.IfThenElse(bx < 1, tvm.tir.Evaluate(0), stmt)
        elif outer_scope == "unknown":
            stmt = tvm.tir.For(owned, 0, rank, tvm.tir.ForKind.SERIAL, stmt)
        elif outer_scope == "overflow32":
            stmt = tvm.tir.For(owned, tvm.tir.const(2147483646, "int32"), 2,
                               tvm.tir.ForKind.SERIAL, stmt)
        elif outer_scope == "overflow64":
            owned64 = tvm.tir.Var("owned64", "int64")
            stmt = tvm.tir.For(owned64, tvm.tir.const(9223372036854775806, "int64"),
                               tvm.tir.const(2, "int64"),
                               tvm.tir.ForKind.SERIAL, stmt)
        else:
            raise ValueError(f"unknown outer scope mode: {outer_scope}")
        if bx is not None:
            bx_iv = tvm.tir.IterVar(None, bx, tvm.tir.IterVar.ThreadIndex, "blockIdx.x")
            stmt = tvm.tir.AttrStmt(bx_iv, "thread_extent", tvm.tir.const(2, "int32"), stmt)
    func = tvm.tir.PrimFunc([out_handle, desc, rank, world], stmt, buffer_map={out_handle: out, desc: desc_buf})
    return func.with_attr("target", tvm.target.Target(TARGET)).with_attr("tl.dcu_ipc.abi_ready", 1).with_attr(
        "tl.dcu_ipc.bindings", "partial=ipc_view_partial")


def _ar_exact_candidate(*, source_bias=0, output_bias=0, element_extent=128,
                        remote_shape=65536, data_alignment=64, duplicate_input_peer=False,
                        duplicate_output_peer=False, wrong_output_value=False, interleave=None,
                        assume_scope="enclosing"):
    """Post-hoist AR normal form, including its fixed four-peer rotation."""
    partial_desc, output_desc = _ptr("ipc_view_partial", "uint64"), _ptr("ipc_view_out", "uint64")
    rank, world = tvm.tir.Var("__tl_ipc_rank", "int32"), tvm.tir.Var("__tl_ipc_world_size", "int32")
    tx, comm, i = (tvm.tir.Var("tx", "int32"), tvm.tir.Var("comm_tx", "int32"),
                   tvm.tir.Var("i", "int32"))
    partial_desc_buf = tvm.tir.decl_buffer((6,), "uint64", data=partial_desc, scope="global", data_alignment=8)
    output_desc_buf = tvm.tir.decl_buffer((6,), "uint64", data=output_desc, scope="global", data_alignment=8)
    temp_vars = [_ptr(f"ar_p{k}", "float16", "local") for k in range(4)]
    reduced_var = _ptr("ar_reduced", "float16", "local")
    temps = [tvm.tir.decl_buffer((1,), "float16", data=p, scope="local") for p in temp_vars]
    reduced = tvm.tir.decl_buffer((1,), "float16", data=reduced_var, scope="local")
    input_vars = [_ptr(f"ar_input{k}", "float16") for k in range(4)]
    output_vars = [_ptr(f"ar_output{k}", "float16") for k in range(4)]
    inputs = [tvm.tir.decl_buffer((remote_shape,), "float16", data=p, scope="global",
                                  data_alignment=data_alignment) for p in input_vars]
    outputs = [tvm.tir.decl_buffer((remote_shape,), "float16", data=p, scope="global",
                                   data_alignment=data_alignment) for p in output_vars]
    index = i * 512 + comm + source_bias
    pulls = [tvm.tir.BufferStore(temps[k], tvm.tir.BufferLoad(inputs[k], [index]), [0]) for k in range(4)]
    values = [tvm.tir.Cast("float32", tvm.tir.BufferLoad(temp, [0])) for temp in temps]
    reduced_store = tvm.tir.BufferStore(reduced, tvm.tir.Cast("float16", values[0] + values[1] + values[2] + values[3]), [0])
    pushed_value = tvm.tir.BufferLoad(temps[0] if wrong_output_value else reduced, [0])
    pushes = [tvm.tir.BufferStore(outputs[k], pushed_value, [index + (output_bias if k == 3 else 0)]) for k in range(4)]
    body = [*pulls, reduced_store]
    if interleave == "opaque":
        body.append(tvm.tir.Evaluate(tvm.tir.call_extern("int32", "ar_side_effect")))
    if interleave == "fence":
        body.append(tvm.tir.Evaluate(tvm.tir.Call("int32", FENCE, [rank, world])))
    body.extend(pushes)
    loop = tvm.tir.For(i, 0, element_extent, tvm.tir.ForKind.SERIAL, tvm.tir.SeqStmt(body))
    chain = loop
    remote_vars = [*input_vars, *output_vars]
    for k in reversed(range(8)):
        descriptor = partial_desc if k < 4 else output_desc
        if k < 4:
            peer = 0 if duplicate_input_peer and k == 1 else k
        else:
            offset = k - 4
            peer = rank if offset == 0 else tvm.tir.floormod(rank + offset, 4)
            if duplicate_output_peer and offset == 3:
                peer = rank
        chain = tvm.tir.LetStmt(
            remote_vars[k], tvm.tir.Call(remote_vars[k].dtype, REMOTE_ADDRESS, [descriptor, peer, rank, world]), chain,
        )
    opaque = tvm.tir.LetStmt(
        comm, tvm.tir.call_extern("int32", "tl::ipc::opaque_thread_id", tx),
        tvm.tir.SeqStmt([chain, tvm.tir.Evaluate(tvm.tir.Call("int32", FENCE, [rank, world]))]),
    )
    x_iv = tvm.tir.IterVar(None, tx, tvm.tir.IterVar.ThreadIndex, "threadIdx.x")
    stmt = tvm.tir.AttrStmt(x_iv, "thread_extent", tvm.tir.const(512, "int32"), opaque)
    # These are the frontend's lexical range facts, not an inferred property
    # of differently named output peers.  A sibling assumption is deliberately
    # not in scope for the candidate and must not be leaked by the pass.
    if assume_scope == "enclosing":
        stmt = tvm.tir.AttrStmt(rank >= 0, "tl.assume", tvm.tir.StringImm("rank lower"), stmt)
        stmt = tvm.tir.AttrStmt(rank < 4, "tl.assume", tvm.tir.StringImm("rank upper"), stmt)
    elif assume_scope == "sibling":
        lower = tvm.tir.AttrStmt(rank >= 0, "tl.assume", tvm.tir.StringImm("rank lower"), tvm.tir.Evaluate(0))
        upper = tvm.tir.AttrStmt(rank < 4, "tl.assume", tvm.tir.StringImm("rank upper"), tvm.tir.Evaluate(0))
        stmt = tvm.tir.SeqStmt([lower, upper, stmt])
    elif assume_scope != "none":
        raise ValueError(f"unknown assume scope {assume_scope}")
    for p in [*temp_vars, reduced_var]:
        stmt = tvm.tir.Allocate(p, "float16", [1], tvm.tir.const(True, "bool"), stmt)
    func = tvm.tir.PrimFunc(
        [partial_desc, output_desc, rank, world], stmt,
        buffer_map={partial_desc: partial_desc_buf, output_desc: output_desc_buf},
    )
    return func.with_attr("target", tvm.target.Target(TARGET)).with_attr(
        "tl.dcu_ipc.abi_ready", 1,
    ).with_attr("tl.dcu_ipc.bindings", "partial=ipc_view_partial;out=ipc_view_out")


def _ag_exact_candidate(*, source_bias=0, destination_bias=0, element_extent=128,
                        peer_from_opaque=False, remote_shape=131072, data_alignment=64):
    """Post-hoist AG one-pull/one-push normal form with a dynamic outer peer."""
    gathered_handle, gathered_data = _ptr("A_handle", "float16"), _ptr("A_data", "float16")
    desc = _ptr("ipc_view_a_shard", "uint64")
    rank, world = tvm.tir.Var("__tl_ipc_rank", "int32"), tvm.tir.Var("__tl_ipc_world_size", "int32")
    tx, comm, i, chunk = (tvm.tir.Var("tx", "int32"), tvm.tir.Var("comm_tx", "int32"),
                          tvm.tir.Var("i", "int32"), tvm.tir.Var("chunk", "int32"))
    gathered = tvm.tir.decl_buffer((131072,), "float16", data=gathered_data, scope="global",
                                   data_alignment=data_alignment)
    desc_buf = tvm.tir.decl_buffer((6,), "uint64", data=desc, scope="global", data_alignment=8)
    remote_var = _ptr("ag_remote", "float16")
    remote = tvm.tir.decl_buffer((remote_shape,), "float16", data=remote_var, scope="global",
                                 data_alignment=data_alignment)
    linear = i * 512 + comm
    source_index = chunk * 65536 + linear + source_bias
    destination_index = chunk * 65536 + linear + destination_bias
    loop = tvm.tir.For(
        i, 0, element_extent, tvm.tir.ForKind.SERIAL,
        tvm.tir.BufferStore(gathered, tvm.tir.BufferLoad(remote, [source_index]), [destination_index]),
    )
    peer = tvm.tir.floormod(rank + (comm if peer_from_opaque else chunk), 4)
    chain = tvm.tir.LetStmt(
        remote_var, tvm.tir.Call(remote_var.dtype, REMOTE_ADDRESS, [desc, peer, rank, world]), loop,
    )
    opaque = tvm.tir.LetStmt(
        comm, tvm.tir.call_extern("int32", "tl::ipc::opaque_thread_id", tx), chain,
    )
    x_iv = tvm.tir.IterVar(None, tx, tvm.tir.IterVar.ThreadIndex, "threadIdx.x")
    stmt = tvm.tir.AttrStmt(x_iv, "thread_extent", tvm.tir.const(512, "int32"), opaque)
    stmt = tvm.tir.For(chunk, 0, 1, tvm.tir.ForKind.SERIAL, stmt)
    stmt = tvm.tir.AttrStmt(rank >= 0, "tl.assume", tvm.tir.StringImm("rank lower"), stmt)
    stmt = tvm.tir.AttrStmt(rank < 4, "tl.assume", tvm.tir.StringImm("rank upper"), stmt)
    func = tvm.tir.PrimFunc(
        [gathered_handle, desc, rank, world], stmt,
        buffer_map={gathered_handle: gathered, desc: desc_buf},
    )
    return func.with_attr("target", tvm.target.Target(TARGET)).with_attr(
        "tl.dcu_ipc.abi_ready", 1,
    ).with_attr("tl.dcu_ipc.bindings", "a_shard=ipc_view_a_shard")


def _apply(func):
    return tilelang.transform.VectorizeDCUIPCCommunication()(tvm.IRModule({"main": func}))["main"]


def _has_ramp(stmt):
    found = []
    tvm.tir.stmt_functor.post_order_visit(stmt, lambda n: found.append(n) if isinstance(n, tvm.tir.Ramp) else None)
    return bool(found)


def _has_fence(stmt):
    found = []
    tvm.tir.stmt_functor.post_order_visit(
        stmt, lambda n: found.append(n) if isinstance(n, tvm.tir.Call) and n.op.same_as(FENCE) else None,
    )
    return bool(found)


def _for_extents(stmt):
    extents = []

    def visit(node):
        if isinstance(node, tvm.tir.For) and isinstance(node.extent, tvm.tir.IntImm):
            extents.append(node.extent.value)

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return extents


def _has_ipc_vectorize_guard(stmt):
    found = []

    def visit(node):
        if isinstance(node, tvm.tir.Call) and "tl::ipc::can_vectorize_" in str(node):
            found.append(node)

    tvm.tir.stmt_functor.post_order_visit(
        stmt,
        visit,
    )
    return bool(found)


def test_exact_normal_form_and_safety_mutations_are_scoped():
    positive = _exact_candidate()
    transformed = _apply(positive)
    assert _has_ramp(transformed.body)
    assert _has_fence(transformed.body)
    assert not _has_ipc_vectorize_guard(transformed.body)
    assert 16 in _for_extents(transformed.body)
    assert 128 not in _for_extents(transformed.body)
    for kwargs in (
        {"source_bias": 1}, {"row_stride": 128}, {"nonunit_y": True},
        {"volatile": True}, {"alias_temp": True}, {"bad_descriptor": True},
        {"data_alignment": 8}, {"data_alignment": 24}, {"elem_offset": 8},
    ):
        original = _exact_candidate(**kwargs)
        tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)

    # These remain in allocation range, isolating the half8 start-alignment
    # proof from the ordinary source-range rejection.  +2 was half2-aligned
    # but is not a valid 16-byte vector start; +8 is a valid nonzero base.
    half2_only = _exact_candidate(source_bias=2, remote_shape=65538)
    tvm.ir.assert_structural_equal(half2_only, _apply(half2_only), map_free_vars=True)
    aligned_offset = _exact_candidate(source_bias=8, remote_shape=65544)
    assert _has_ramp(_apply(aligned_offset).body)


def test_allreduce_four_pull_four_push_normal_form_is_exact_and_conservative():
    positive = _ar_exact_candidate()
    transformed = _apply(positive)
    assert _has_ramp(transformed.body)
    assert _has_fence(transformed.body)
    assert 16 in _for_extents(transformed.body)
    assert 128 not in _for_extents(transformed.body)

    # Every rejection below changes a property used by the production AR
    # matcher.  In particular, no one-output reduction, repeated peer, or
    # cross-element side effect may be silently treated as the fixed 4x4 form.
    for kwargs in (
        {"output_bias": 1}, {"wrong_output_value": True},
        {"duplicate_input_peer": True}, {"duplicate_output_peer": True},
        {"interleave": "opaque"}, {"interleave": "fence"},
        {"source_bias": 2}, {"element_extent": 127},
        {"remote_shape": 65535}, {"data_alignment": 8},
        {"assume_scope": "none"}, {"assume_scope": "sibling"},
    ):
        original = _ar_exact_candidate(**kwargs)
        tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)


def test_allgather_one_remote_copy_requires_outer_peer_and_exact_half8_mapping():
    # This is the post-hoist AG form: `chunk` is lexical outer state, so the
    # dynamic local-rank-first peer is fixed for all 128 scalar iterations.
    positive = _ag_exact_candidate()
    transformed = _apply(positive)
    assert _has_ramp(transformed.body)
    assert 16 in _for_extents(transformed.body)
    assert 128 not in _for_extents(transformed.body)

    # These each invalidate an obligation of the AG-only matcher: a half2
    # start, insufficient source range, changing peer across element lanes,
    # or a non-128x512 scalar cover.  They must leave the fake post-hoist IR
    # untouched rather than receive a speculative vector copy.
    for kwargs in (
        {"source_bias": 1}, {"destination_bias": 1}, {"remote_shape": 65535},
        {"peer_from_opaque": True}, {"element_extent": 127}, {"data_alignment": 8},
    ):
        original = _ag_exact_candidate(**kwargs)
        tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)


def test_outer_block_serial_and_branch_ranges_are_replayed_without_leaking():
    # The source allocation is exactly one old 128x512 tile.  `owned * 65536`
    # is in range only under bx < 1; it exercises replay ordering (block,
    # branch, serial loop) rather than granting an unscoped rank/world bound.
    in_branch = _exact_candidate(outer_scope="then")
    transformed = _apply(in_branch)
    assert _has_ramp(transformed.body)
    assert 16 in _for_extents(transformed.body)

    # The first child carries the same branch fact, but the independent
    # candidate is its sibling.  A scope-restoration error would vectorize the
    # sibling using bx < 1 even though bx may be one and its remote index is
    # out of range.
    sibling = _exact_candidate(outer_scope="sibling")
    tvm.ir.assert_structural_equal(sibling, _apply(sibling), map_free_vars=True)

    # The else arm contributes Not(bx < 1), and the arithmetic Let binds the
    # serial minimum to bx - 1.  Together with blockIdx.x < 2, owned is zero.
    # This reaches the pass only if both replay paths are retained in order.
    else_let = _exact_candidate(outer_scope="else_let")
    assert _has_ramp(_apply(else_let).body)

    # Unknown extent and endpoints whose exclusive upper bound overflows the
    # TIR index dtype are conservative no-ops, not analyzer exceptions or
    # wrapped range assumptions.
    for mode in ("unknown", "overflow32", "overflow64"):
        original = _exact_candidate(outer_scope=mode)
        assert tvm.tir.analysis.verify_well_formed(tvm.IRModule({"main": original}))
        tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)


def test_real_scheduled_lower_has_unchecked_half8_without_scalar_fallback(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    scheduled = _load_module("gemm_reduce_scatter_scheduled")
    # These are the scheduled configurations that previously only proved the
    # inner 128x512 domain.  The outer block/serial/branch replay must make
    # all four reach the real vectorized lowering.
    for m, n, k, comm_blocks in (
        (2048, 2048, 512, 32),
        (4096, 4096, 1024, 32),
        (8192, 8192, 2048, 8),
        (16384, 16384, 4096, 4),
    ):
        _, bound = scheduled.build_bound(m, n, k, total_blocks=80, comm_blocks=comm_blocks)
        default_source = _source(bound)
        assert "tl::ipc::can_vectorize_" not in default_source
        assert "for (int _tmp_2 = 0; _tmp_2 < 128; ++_tmp_2)" in default_source
        assert "ipc_half8_i" not in default_source
        source = _source(bound, enable_ipc_vectorize=True)
        # The opt-in pass relies on its caller's allocation contract: no
        # runtime guard or scalar fallback is emitted.
        assert "tl::ipc::can_vectorize_" not in source
        assert "for (int ipc_half8_i = 0; ipc_half8_i < 16; ++ipc_half8_i)" in source
        assert "for (int _tmp_2 = 0; _tmp_2 < 128; ++_tmp_2)" not in source
        assert "tl::ipc::opaque_thread_id(((int)threadIdx.x))" in source
        assert "float16x8" in source or "uint4" in source
        assert "fence_sys" in source and "__syncthreads" in source


@pytest.mark.parametrize(
    "m,n,k,comm_blocks",
    [
        (2048, 2048, 512, 48),
        (4096, 4096, 1024, 24),
        (8192, 8192, 2048, 16),
        (16384, 16384, 4096, 8),
    ],
)
def test_real_allreduce_lower_vectorizes_four_loads_and_four_stores(monkeypatch, m, n, k, comm_blocks):
    monkeypatch.syspath_prepend(str(AR_EXAMPLE_DIR))
    scheduled = _load_module("gemm_allreduce_scheduled", AR_EXAMPLE_DIR)
    _, bound = scheduled.build_bound(m, n, k, total_blocks=80, comm_blocks=comm_blocks)
    scalar_source = _source(bound)
    assert "ipc_ar_half8_i" not in scalar_source
    source = _source(bound, enable_ipc_vectorize=True)
    assert "for (int ipc_ar_half8_i = 0; ipc_ar_half8_i < 16; ++ipc_ar_half8_i)" in source
    assert "tl::ipc::opaque_thread_id(((int)threadIdx.x))" in source
    # The narrowed AR rewrite emits exactly the four partial vector reads and
    # four full-output vector writes; no frontend T.vectorized request remains.
    partial_uint4 = [line for line in source.splitlines() if "remote_partial" in line and "uint4" in line]
    output_uint4 = [line for line in source.splitlines() if "remote_out" in line and "uint4" in line]
    assert len(partial_uint4) == 4
    assert len(output_uint4) == 4
    pointer_lines = [
        index for index, line in enumerate(source.splitlines())
        if "tl::ipc::peer<std::byte>(ipc_view_partial" in line
        or "tl::ipc::peer<std::byte>(ipc_view_out" in line
    ]
    assert len(pointer_lines) == 8
    vector_loop = next(index for index, line in enumerate(source.splitlines()) if "ipc_ar_half8_i" in line)
    assert max(pointer_lines) < vector_loop
