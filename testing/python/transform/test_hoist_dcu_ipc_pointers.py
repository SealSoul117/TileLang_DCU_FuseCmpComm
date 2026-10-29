"""CPU scope tests for the conservative M2 IPC pointer hoister.

These synthetic TIR probes exercise only the new helper layer.  The final test
uses the real scheduled reduce-scatter frontend to cover normal pipeline
integration; it does not replace multi-PE device validation.
"""

import importlib.util
from pathlib import Path

import pytest

import tilelang
from tilelang import tvm


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = ROOT / "examples" / "distributed" / "gemm_rs_prototype"
AR_EXAMPLE_DIR = ROOT / "examples" / "distributed" / "gemm_ar_prototype"
REMOTE_ADDRESS = tvm.ir.Op.get("tl.ipc.remote_address")
FENCE = tvm.ir.Op.get("tl.ipc.fence_sys")


def _pointer(name, dtype):
    return tvm.tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), "global"))


def _remote_call(descriptor, peer, rank, world):
    return tvm.tir.Call(descriptor.dtype, REMOTE_ADDRESS, [descriptor, peer, rank, world])


def _func(body, *, bound=True, extent=None):
    out = _pointer("out", "float16")
    descriptor = _pointer("ipc_view_payload", "uint64")
    rank = tvm.tir.Var("__tl_ipc_rank", "int32")
    world = tvm.tir.Var("__tl_ipc_world_size", "int32")
    extent = tvm.tir.Var("extent", "int32") if extent is None else extent
    out_buffer = tvm.tir.decl_buffer((64,), "float16", data=out, name="out", scope="global")
    descriptor_buffer = tvm.tir.decl_buffer(
        (6,), "uint64", data=descriptor, name="ipc_view_payload", scope="global",
    )
    params = [out, descriptor, rank, world]
    if isinstance(extent, tvm.tir.Var):
        params.append(extent)
    func = tvm.tir.PrimFunc(
        params, body(out_buffer, descriptor, rank, world, extent),
        buffer_map={out: out_buffer, descriptor: descriptor_buffer},
    )
    if bound:
        func = func.with_attr("tl.dcu_ipc.abi_ready", 1)
        func = func.with_attr("tl.dcu_ipc.bindings", "payload=ipc_view_payload")
    return func


def _payload_let(out, descriptor, rank, world, loop_var, peer, suffix):
    ptr = _pointer(f"remote_ptr_{suffix}", "float16")
    remote = tvm.tir.decl_buffer((64,), "float16", data=ptr, name=f"remote_{suffix}", scope="global")
    return tvm.tir.LetStmt(
        ptr, _remote_call(descriptor, peer, rank, world),
        tvm.tir.BufferStore(out, tvm.tir.BufferLoad(remote, [loop_var]), [loop_var]),
    )


def _four_pointer_loop(out, descriptor, rank, world, extent):
    i = tvm.tir.Var("i", "int32")
    copies = [_payload_let(out, descriptor, rank, world, i, tvm.tir.const(peer, "int32"), peer)
              for peer in range(4)]
    # This If represents the predicated output store in scheduled codegen. It
    # is not a source for pointer extraction, but it must not prevent the
    # leading pointer bindings from being moved out of the element loop.
    tail = tvm.tir.IfThenElse(
        i < extent, tvm.tir.BufferStore(out, tvm.tir.const(0, "float16"), [i]), None,
    )
    return tvm.tir.For(i, 0, extent, tvm.tir.ForKind.SERIAL, tvm.tir.SeqStmt([*copies, tail]))


def _apply(func):
    mod = tvm.IRModule({"main": func})
    return tilelang.transform.HoistDCUIPCPointers()(mod)["main"]


def _remote_calls(node):
    calls = []

    def visit(item):
        if isinstance(item, tvm.tir.Call) and item.op.same_as(REMOTE_ADDRESS):
            calls.append(item)

    tvm.tir.stmt_functor.post_order_visit(node, visit)
    return calls


def _remote_calls_under_serial_loops(stmt):
    found = []

    def walk(node, serial_depth=0):
        if isinstance(node, tvm.tir.For):
            walk(node.body, serial_depth + (node.kind == tvm.tir.ForKind.SERIAL))
            return
        if isinstance(node, tvm.tir.LetStmt):
            if isinstance(node.value, tvm.tir.Call) and node.value.op.same_as(REMOTE_ADDRESS):
                found.append(serial_depth)
            walk(node.body, serial_depth)
            return
        if isinstance(node, tvm.tir.SeqStmt):
            for child in node.seq:
                walk(child, serial_depth)
            return
        if isinstance(node, tvm.tir.IfThenElse):
            walk(node.then_case, serial_depth)
            if node.else_case:
                walk(node.else_case, serial_depth)
            return
        if isinstance(node, tvm.tir.AttrStmt):
            # thread_extent is a lexical wrapper, not a serial execution
            # loop.  Descend so the helper reports the hoisted thread-peer
            # candidate rather than accidentally treating the wrapper as a
            # leaf.
            walk(node.body, serial_depth)

    walk(stmt)
    return found


def test_hoists_four_verified_payload_pointers_outside_element_loop_with_zero_guard():
    transformed = _apply(_func(_four_pointer_loop))
    assert len(_remote_calls(transformed.body)) == 4
    assert _remote_calls_under_serial_loops(transformed.body) == [0, 0, 0, 0]
    # Dynamic extent receives the required non-speculation guard, while the
    # original serial loop and its predicated payload store remain in its body.
    assert isinstance(transformed.body, tvm.tir.IfThenElse)
    assert isinstance(transformed.body.then_case, tvm.tir.LetStmt)


def test_constant_zero_and_positive_extents_preserve_execution_boundary():
    zero = _apply(_func(_four_pointer_loop, extent=tvm.tir.const(0, "int32")))
    # Even a known zero extent keeps the opaque address checks unreachable.
    assert isinstance(zero.body, tvm.tir.IfThenElse)
    assert len(_remote_calls(zero.body.then_case)) == 4

    positive = _apply(_func(_four_pointer_loop, extent=tvm.tir.const(3, "int32")))
    assert not isinstance(positive.body, tvm.tir.IfThenElse)
    assert _remote_calls_under_serial_loops(positive.body) == [0, 0, 0, 0]


def test_outer_loop_and_thread_binding_peers_are_invariant_but_element_peer_is_not():
    def outer_loop_peer(out, descriptor, rank, world, extent):
        chunk = tvm.tir.Var("chunk", "int32")
        element = tvm.tir.Var("element", "int32")
        payload = tvm.tir.For(
            element, 0, extent, tvm.tir.ForKind.SERIAL,
            _payload_let(out, descriptor, rank, world, element, chunk, "outer_chunk"),
        )
        return tvm.tir.For(chunk, 0, 4, tvm.tir.ForKind.SERIAL, payload)

    transformed = _apply(_func(outer_loop_peer))
    # The pointer remains inside the chunk traversal but leaves its element
    # loop, matching AG's dynamic source-peer lifetime.
    assert _remote_calls_under_serial_loops(transformed.body) == [1]

    def thread_peer(out, descriptor, rank, world, extent):
        tx = tvm.tir.Var("tx", "int32")
        element = tvm.tir.Var("element", "int32")
        body = tvm.tir.For(
            element, 0, extent, tvm.tir.ForKind.SERIAL,
            _payload_let(out, descriptor, rank, world, element, tx, "thread_peer"),
        )
        iv = tvm.tir.IterVar(None, tx, tvm.tir.IterVar.ThreadIndex, "threadIdx.x")
        return tvm.tir.AttrStmt(iv, "thread_extent", tvm.tir.const(512, "int32"), body)

    assert _remote_calls_under_serial_loops(_apply(_func(thread_peer)).body) == [0]

    def element_peer(out, descriptor, rank, world, extent):
        element = tvm.tir.Var("element", "int32")
        return tvm.tir.For(
            element, 0, extent, tvm.tir.ForKind.SERIAL,
            _payload_let(out, descriptor, rank, world, element, element, "element_peer"),
        )

    original = _func(element_peer)
    tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)


def test_hoists_distributed_lets_around_static_unrolled_payload_reduction():
    def dispersed(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        lane = tvm.tir.Var("lane", "int32")
        # The unrolled loop stands for the eight FP32 lane reductions between
        # AR remote loads and remote stores.  It has no address producer of
        # its own, so top-level pointer Lets on either side remain eligible.
        reduction_loop = tvm.tir.For(
            lane, 0, 8, tvm.tir.ForKind.UNROLLED,
            tvm.tir.BufferStore(out, tvm.tir.const(0, "float16"), [lane]),
        )
        # Match StorageRewrite's real AR form: the explicit-unroll hint wraps
        # the reduction loop but is not an address-producer extraction scope.
        reduction = tvm.tir.AttrStmt(
            lane, "pragma_unroll_explicit", tvm.tir.const(False, "bool"), reduction_loop,
        )
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "load0"),
                _payload_let(out, descriptor, rank, world, i, 1, "load1"),
                reduction,
                _payload_let(out, descriptor, rank, world, i, 2, "store0"),
                _payload_let(out, descriptor, rank, world, i, 3, "store1"),
            ]),
        )

    transformed = _apply(_func(dispersed))
    assert len(_remote_calls(transformed.body)) == 4
    assert _remote_calls_under_serial_loops(transformed.body) == [0, 0, 0, 0]


def test_only_literal_explicit_unroll_attr_for_its_own_small_loop_is_allowed():
    def attr_payload(out, descriptor, rank, world, extent, *, key, value, same_node=True, for_body=True):
        i = tvm.tir.Var("i", "int32")
        lane = tvm.tir.Var("lane", "int32")
        small_loop = tvm.tir.For(
            lane, 0, 8, tvm.tir.ForKind.UNROLLED,
            tvm.tir.BufferStore(out, tvm.tir.const(0, "float16"), [lane]),
        )
        attr_body = small_loop if for_body else tvm.tir.Evaluate(0)
        attr = tvm.tir.AttrStmt(lane if same_node else i, key, value, attr_body)
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "before_attr"),
                attr,
                _payload_let(out, descriptor, rank, world, i, 1, "after_attr"),
            ]),
        )

    # Non-matching attributes must retain the original loop verbatim.
    for kwargs in (
        {"key": "unknown_payload_attr", "value": tvm.tir.const(0, "int32")},
        {"key": "pragma_unroll_explicit", "value": tvm.tir.Var("hint", "int32")},
        {"key": "pragma_unroll_explicit", "value": tvm.tir.const(1, "int32"), "same_node": False},
        {"key": "pragma_unroll_explicit", "value": tvm.tir.const(1, "int32"), "for_body": False},
    ):
        def factory(out, descriptor, rank, world, extent, kwargs=kwargs):
            return attr_payload(out, descriptor, rank, world, extent, **kwargs)

        original = _func(factory)
        tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)


def test_static_vectorized_payload_loop_is_allowed_but_nested_address_is_not():
    def vectorized_payload(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        lane = tvm.tir.Var("lane", "int32")
        payload = tvm.tir.For(
            lane, 0, 8, tvm.tir.ForKind.VECTORIZED,
            tvm.tir.BufferStore(out, tvm.tir.const(0, "float16"), [lane]),
        )
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([_payload_let(out, descriptor, rank, world, i, 0, "before"), payload]),
        )

    transformed = _apply(_func(vectorized_payload))
    assert _remote_calls_under_serial_loops(transformed.body) == [0]

    def nested_address(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        lane = tvm.tir.Var("lane", "int32")
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.For(
                lane, 0, 8, tvm.tir.ForKind.UNROLLED,
                _payload_let(out, descriptor, rank, world, i, 0, "nested_address"),
            ),
        )

    original = _func(nested_address)
    tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)


def test_distributed_hoist_rejects_conditional_effects_and_nonliteral_payload_loops():
    def unknown_condition(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        condition = tvm.tir.call_extern("int32", "unknown_condition", rank) != 0
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "before_unknown_condition"),
                tvm.tir.IfThenElse(condition, tvm.tir.Evaluate(0), None),
                _payload_let(out, descriptor, rank, world, i, 1, "after_unknown_condition"),
            ]),
        )

    def address_condition(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        # Preserve remote_address's real handle dtype; isnullptr is the legal
        # boolean consumer for a handle-valued expression in an If condition.
        hidden_address = _remote_call(descriptor, tvm.tir.const(2, "int32"), rank, world)
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "before_address_condition"),
                tvm.tir.IfThenElse(tvm.tir.isnullptr(hidden_address), tvm.tir.Evaluate(0), None),
                _payload_let(out, descriptor, rank, world, i, 1, "after_address_condition"),
            ]),
        )

    def dynamic_unroll_bound(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        lane = tvm.tir.Var("lane", "int32")
        dynamic_payload = tvm.tir.For(
            lane, 0, extent, tvm.tir.ForKind.UNROLLED,
            tvm.tir.BufferStore(out, tvm.tir.const(0, "float16"), [lane]),
        )
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "before_dynamic_unroll"),
                dynamic_payload,
                _payload_let(out, descriptor, rank, world, i, 1, "after_dynamic_unroll"),
            ]),
        )

    def opaque_unroll_bound(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        lane = tvm.tir.Var("lane", "int32")
        opaque_extent = tvm.tir.call_extern("int32", "unknown_payload_extent", extent)
        payload = tvm.tir.For(
            lane, 0, opaque_extent, tvm.tir.ForKind.UNROLLED,
            tvm.tir.BufferStore(out, tvm.tir.const(0, "float16"), [lane]),
        )
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([_payload_let(out, descriptor, rank, world, i, 0, "before_opaque_unroll"), payload]),
        )

    def nested_serial(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        nested = tvm.tir.Var("nested", "int32")
        nested_loop = tvm.tir.For(
            nested, 0, 2, tvm.tir.ForKind.SERIAL,
            tvm.tir.BufferStore(out, tvm.tir.const(0, "float16"), [nested]),
        )
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "before_nested_serial"),
                nested_loop,
                _payload_let(out, descriptor, rank, world, i, 1, "after_nested_serial"),
            ]),
        )

    def annotated_unroll(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        lane = tvm.tir.Var("lane", "int32")
        payload = tvm.tir.For(
            lane, 0, 8, tvm.tir.ForKind.UNROLLED,
            tvm.tir.BufferStore(out, tvm.tir.const(0, "float16"), [lane]),
            annotations={"tl.test_payload_annotation": 1},
        )
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([_payload_let(out, descriptor, rank, world, i, 0, "before_annotated"), payload]),
        )

    for factory in (
        unknown_condition, address_condition, dynamic_unroll_bound, opaque_unroll_bound, nested_serial,
        annotated_unroll,
    ):
        original = _func(factory)
        tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)


def test_sync_in_payload_loop_and_later_dynamic_peer_remain_scoped():
    def sync_in_unroll(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        lane = tvm.tir.Var("lane", "int32")
        reduction = tvm.tir.For(
            lane, 0, 8, tvm.tir.ForKind.UNROLLED,
            tvm.tir.Evaluate(tvm.tir.Call("int32", FENCE, [rank, world])),
        )
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "before_sync_unroll"),
                reduction,
                _payload_let(out, descriptor, rank, world, i, 1, "after_sync_unroll"),
            ]),
        )

    def dynamic_later_peer(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "safe_first"),
                _payload_let(out, descriptor, rank, world, i, i, "dynamic_later"),
            ]),
        )

    original = _func(sync_in_unroll)
    tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)

    transformed = _apply(_func(dynamic_later_peer))
    # The independent first pointer may move, but the loop-dependent second
    # peer must remain at serial depth one; both original calls are retained.
    assert len(_remote_calls(transformed.body)) == 2
    assert _remote_calls_under_serial_loops(transformed.body) == [0, 1]


def test_unbound_loop_and_loop_dependent_peer_are_identity():
    unbound = _func(_four_pointer_loop, bound=False)
    tvm.ir.assert_structural_equal(unbound, _apply(unbound), map_free_vars=True)

    def dynamic_peer(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            _payload_let(out, descriptor, rank, world, i, i, "dynamic"),
        )

    dependent = _func(dynamic_peer)
    tvm.ir.assert_structural_equal(dependent, _apply(dependent), map_free_vars=True)

    def loop_local_let_peer(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        peer = tvm.tir.Var("peer", "int32")
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.LetStmt(peer, i + 1, _payload_let(out, descriptor, rank, world, i, peer, "local")),
        )

    local_peer = _func(loop_local_let_peer)
    tvm.ir.assert_structural_equal(local_peer, _apply(local_peer), map_free_vars=True)


def test_pure_outer_let_dynamic_peer_is_hoisted():
    def outer_peer(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        peer = tvm.tir.Var("peer", "int32")
        loop = tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            _payload_let(out, descriptor, rank, world, i, peer, "outer"),
        )
        return tvm.tir.LetStmt(peer, (rank + 1) % world, loop)

    transformed = _apply(_func(outer_peer))
    assert len(_remote_calls(transformed.body)) == 1
    assert _remote_calls_under_serial_loops(transformed.body) == [0]


def test_condition_sync_descriptor_write_alias_escape_and_opaque_peer_do_not_hoist():
    def conditional(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.IfThenElse(i < extent, _payload_let(out, descriptor, rank, world, i, 0, "if"), None),
        )

    def conditional_later(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "before_if"),
                tvm.tir.IfThenElse(
                    i < extent, _payload_let(out, descriptor, rank, world, i, 1, "inside_if"), None,
                ),
            ]),
        )

    def sync_tail(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "sync"),
                tvm.tir.Evaluate(tvm.tir.Call("int32", FENCE, [rank, world])),
            ]),
        )

    def descriptor_write(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        descriptor_buffer = tvm.tir.decl_buffer(
            (6,), "uint64", data=descriptor, name="ipc_view_payload", scope="global",
        )
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "write"),
                tvm.tir.BufferStore(descriptor_buffer, tvm.tir.const(0, "uint64"), [0]),
            ]),
        )

    def opaque_later_peer(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        opaque_peer = tvm.tir.call_extern("int32", "unknown_peer", rank)
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                _payload_let(out, descriptor, rank, world, i, 0, "first"),
                _payload_let(out, descriptor, rank, world, i, opaque_peer, "later"),
            ]),
        )

    def descriptor_alias(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        alias = _pointer("descriptor_alias", "uint64")
        loop = tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            _payload_let(out, alias, rank, world, i, 0, "alias"),
        )
        return tvm.tir.LetStmt(alias, descriptor, loop)

    def descriptor_escape(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        return tvm.tir.SeqStmt([
            tvm.tir.Evaluate(tvm.tir.call_extern("int32", "descriptor_escape", descriptor)),
            tvm.tir.For(
                i, 0, extent, tvm.tir.ForKind.SERIAL,
                _payload_let(out, descriptor, rank, world, i, 0, "escape"),
            ),
        ])

    for factory in (
        conditional, conditional_later, sync_tail, descriptor_write, descriptor_alias, descriptor_escape,
        opaque_later_peer,
    ):
        original = _func(factory)
        tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)


def test_impure_bounds_and_outer_barrier_never_cross_scope():
    def buffer_load_bound(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        # This is deliberately an arbitrary input load, not a descriptor use:
        # the bound guard must reject any extra BufferLoad evaluation itself.
        min_bound = tvm.tir.Cast("int32", tvm.tir.BufferLoad(out, [0]))
        return tvm.tir.For(
            i, min_bound, extent, tvm.tir.ForKind.SERIAL,
            _payload_let(out, descriptor, rank, world, i, 0, "buffer_bound"),
        )

    def opaque_bound(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        opaque_extent = tvm.tir.call_extern("int32", "unknown_extent", extent)
        return tvm.tir.For(
            i, 0, opaque_extent, tvm.tir.ForKind.SERIAL,
            _payload_let(out, descriptor, rank, world, i, 0, "opaque_bound"),
        )

    for factory in (buffer_load_bound, opaque_bound):
        original = _func(factory)
        tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)

    def outer_tile_with_barrier(out, descriptor, rank, world, extent):
        tile = tvm.tir.Var("tile", "int32")
        i = tvm.tir.Var("i", "int32")
        element = tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            _payload_let(out, descriptor, rank, world, i, 0, "nested"),
        )
        return tvm.tir.For(
            tile, 0, 2, tvm.tir.ForKind.SERIAL,
            tvm.tir.SeqStmt([
                tvm.tir.Evaluate(tvm.tir.Call("int32", FENCE, [rank, world])),
                element,
            ]),
        )

    nested = _apply(_func(outer_tile_with_barrier))
    # The inner loop is optimized locally, but its pointer is still inside the
    # outer tile loop and therefore has not crossed the fence.
    assert _remote_calls_under_serial_loops(nested.body) == [1]


def test_thread_and_allocate_scopes_are_not_hoist_sources():
    def thread_bound(out, descriptor, rank, world, extent):
        tx = tvm.tir.Var("tx", "int32")
        binding = tvm.tir.IterVar(
            tvm.ir.Range(0, 64), tx, iter_type=tvm.tir.IterVar.ThreadIndex, thread_tag="threadIdx.x",
        )
        return tvm.tir.For(
            tx, 0, 64, tvm.tir.ForKind.THREAD_BINDING,
            _payload_let(out, descriptor, rank, world, tx, 0, "thread"), thread_binding=binding,
        )

    def inside_allocate(out, descriptor, rank, world, extent):
        i = tvm.tir.Var("i", "int32")
        local = _pointer("local", "float16")
        return tvm.tir.For(
            i, 0, extent, tvm.tir.ForKind.SERIAL,
            tvm.tir.Allocate(
                local, "float16", [1], tvm.tir.const(True, "bool"),
                _payload_let(out, descriptor, rank, world, i, 0, "allocated"),
            ),
        )

    for factory in (thread_bound, inside_allocate):
        original = _func(factory)
        tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)


def _load_module(name, example_dir=EXAMPLE_DIR):
    path = example_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_scheduled_lower_keeps_four_payload_pointers_before_element_loop(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    scheduled = _load_module("gemm_reduce_scatter_scheduled")
    _, func = scheduled.build_bound(2048, 2048, 512, total_blocks=80, comm_blocks=32)
    artifact = tilelang.lower(
        func, target="hip -mcpu=gfx936", enable_device_compile=False, enable_host_codegen=False,
    )
    source = artifact.kernel_source
    lines = source.splitlines()
    pointer_lines = [
        index for index, line in enumerate(lines)
        if "tl::ipc::peer<std::byte>(ipc_view_partial" in line
    ]
    assert len(pointer_lines) == 4
    # All four declarations must dominate their payload use.  The exact loop
    # variable name is intentionally not part of this integration assertion.
    first_payload_line = next(
        index for index, line in enumerate(lines)
        if "(half_t*)remote_partial0_remote_ptr_" in line
    )
    element_loop_line = next(
        index for index, line in enumerate(lines)
        if index > max(pointer_lines) and "for (int " in line
    )
    fence_line = max(index for index, line in enumerate(lines[:min(pointer_lines)]) if "fence_sys" in line)
    barrier_line = max(index for index, line in enumerate(lines[:min(pointer_lines)]) if "__syncthreads" in line)
    # Pointer hoisting itself preserves the dynamic zero-trip guard (covered
    # above).  The later, ABI-gated opaque-thread range proof may lawfully
    # remove that guard for this concrete 512-thread HIP launch, but it must
    # still leave the helper and synchronization prefix intact.
    assert "tl::ipc::opaque_thread_id(((int)threadIdx.x))" in source
    assert "comm_tx < 65536" not in source
    assert "for (int _tmp_2 = 0; _tmp_2 < 128; ++_tmp_2)" in source
    assert fence_line < barrier_line < min(pointer_lines)
    assert max(pointer_lines) < element_loop_line < first_payload_line


@pytest.mark.parametrize(
    "m,n,k,comm_blocks",
    [
        (2048, 2048, 512, 48),
        (4096, 4096, 1024, 24),
        (8192, 8192, 2048, 16),
        (16384, 16384, 4096, 8),
    ],
)
def test_real_allreduce_hoists_eight_payload_pointers_before_scalar_element_loop(
    monkeypatch, m, n, k, comm_blocks,
):
    """AR keeps compute unchanged and hoists four loads plus four stores per tile."""
    monkeypatch.syspath_prepend(str(AR_EXAMPLE_DIR))
    scheduled = _load_module("gemm_allreduce_scheduled", AR_EXAMPLE_DIR)
    _, func = scheduled.build_bound(m, n, k, total_blocks=80, comm_blocks=comm_blocks)
    # Inspect the Hoist result before the optional AR half8 pass.  This keeps
    # the scalar loop visible while still checking that the eight bases remain
    # inside the tile's synchronization boundary.
    with tvm.transform.PassContext(config={"tl.enable_dcu_ipc_vectorize": False}):
        artifact = tilelang.lower(
            func, target="hip -mcpu=gfx936", enable_device_compile=False, enable_host_codegen=False,
        )
    source = artifact.kernel_source
    lines = source.splitlines()
    payload_pointers = [
        index for index, line in enumerate(lines)
        if "tl::ipc::peer<std::byte>(ipc_view_partial" in line
        or "tl::ipc::peer<std::byte>(ipc_view_out" in line
    ]
    assert len(payload_pointers) == 8
    element_loop = next(
        index for index, line in enumerate(lines)
        if index > max(payload_pointers) and "for (int " in line and "< 128" in line
    )
    tile_loop = max(
        index for index, line in enumerate(lines[:min(payload_pointers)])
        if "for (int " in line
    )
    wait_line = max(
        index for index, line in enumerate(lines[tile_loop:min(payload_pointers)], start=tile_loop)
        if "wait_eq_acquire" in line
    )
    fence_line = max(
        index for index, line in enumerate(lines[wait_line:min(payload_pointers)], start=wait_line)
        if "fence_sys" in line
    )
    barrier_line = max(
        index for index, line in enumerate(lines[fence_line:min(payload_pointers)], start=fence_line)
        if "__syncthreads" in line
    )
    # The remote bases are tile-local: no declaration crosses the tile loop,
    # and every one remains after its tile's four waits, fence and barrier.
    assert tile_loop < wait_line < fence_line < barrier_line < min(payload_pointers)
    assert max(payload_pointers) < element_loop
    # The persistent compute branch remains separately guarded and retains
    # the original four-stage GEMM lowering; hoisting is communication-local.
    assert "if (((int)blockIdx.x) <" in source
    assert "A_dcu_res" in source and "B_dcu_res" in source
