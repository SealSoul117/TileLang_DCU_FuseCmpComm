"""4/8-PE and FP16/BF16 contract checks for the IPC communication vectorizer.

These are deliberately synthetic post-hoist TIR candidates.  Existing tests
continue to cover the legacy four-PE schedules; this file verifies the new
compile-time PE contract and the generalized scalar reduction normal form.
"""

import pytest
import tilelang
from tilelang import tvm


TARGET = "hip -mcpu=gfx936"
REMOTE_ADDRESS = tvm.ir.Op.get("tl.ipc.remote_address")
FENCE = tvm.ir.Op.get("tl.ipc.fence_sys")


def _ptr(name, dtype, scope="global"):
    return tvm.tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))


def _generic_rs(*, peer_count, dtype, direct, attr_peer_count=None, mixed_dtype=False):
    """Create exactly P pulls, a left-associated FP32 sum, and one store."""
    out_handle, out_data = _ptr("out_handle", dtype), _ptr("out_data", dtype)
    descriptor = _ptr("ipc_view_partial", "uint64")
    rank = tvm.tir.Var("__tl_ipc_rank", "int32")
    world = tvm.tir.Var("__tl_ipc_world_size", "int32")
    tx, comm, i = (tvm.tir.Var("tx", "int32"), tvm.tir.Var("comm_tx", "int32"),
                   tvm.tir.Var("i", "int32"))
    out = tvm.tir.decl_buffer((65536,), dtype, data=out_data, scope="global", data_alignment=64)
    descriptor_buf = tvm.tir.decl_buffer((6,), "uint64", data=descriptor, scope="global", data_alignment=8)
    payload_dtypes = [dtype] * peer_count
    if mixed_dtype:
        payload_dtypes[-1] = "bfloat16" if dtype == "float16" else "float16"
    temp_vars = [_ptr(f"p{peer}", payload_dtypes[peer], "local") for peer in range(peer_count)]
    temps = [tvm.tir.decl_buffer((1,), payload_dtypes[peer], data=temp, scope="local")
             for peer, temp in enumerate(temp_vars)]
    remote_vars = [_ptr(f"remote{peer}", payload_dtypes[peer]) for peer in range(peer_count)]
    remotes = [tvm.tir.decl_buffer((65536,), payload_dtypes[peer], data=remote, scope="global",
                                   data_alignment=64)
               for peer, remote in enumerate(remote_vars)]
    index = i * 512 + comm
    pulls = []
    for peer in range(peer_count):
        load = tvm.tir.BufferLoad(remotes[peer], [index])
        value = load if direct else tvm.tir.if_then_else(
            rank >= 0, load, tvm.tir.const(0, payload_dtypes[peer]),
        )
        pulls.append(tvm.tir.BufferStore(temps[peer], value, [0]))
    values = [tvm.tir.Cast("float32", tvm.tir.BufferLoad(temp, [0])) for temp in temps]
    total = values[0]
    for value in values[1:]:
        total = total + value
    body = tvm.tir.SeqStmt([*pulls, tvm.tir.BufferStore(out, tvm.tir.Cast(dtype, total), [index])])
    loop = tvm.tir.For(i, 0, 128, tvm.tir.ForKind.SERIAL, body)
    chain = loop
    for peer in reversed(range(peer_count)):
        chain = tvm.tir.LetStmt(
            remote_vars[peer],
            tvm.tir.Call(remote_vars[peer].dtype, REMOTE_ADDRESS,
                         [descriptor, tvm.tir.const(peer, "int32"), rank, world]),
            chain,
        )
    opaque = tvm.tir.LetStmt(
        comm, tvm.tir.call_extern("int32", "tl::ipc::opaque_thread_id", tx),
        tvm.tir.SeqStmt([chain, tvm.tir.Evaluate(tvm.tir.Call("int32", FENCE, [rank, world]))]),
    )
    x_iv = tvm.tir.IterVar(None, tx, tvm.tir.IterVar.ThreadIndex, "threadIdx.x")
    stmt = tvm.tir.AttrStmt(x_iv, "thread_extent", tvm.tir.const(512, "int32"), opaque)
    stmt = tvm.tir.AttrStmt(rank >= 0, "tl.assume", tvm.tir.StringImm("rank lower"), stmt)
    stmt = tvm.tir.AttrStmt(rank < peer_count, "tl.assume", tvm.tir.StringImm("rank upper"), stmt)
    for peer, temp in enumerate(temp_vars):
        stmt = tvm.tir.Allocate(temp, payload_dtypes[peer], [1], tvm.tir.const(True, "bool"), stmt)
    func = tvm.tir.PrimFunc(
        [out_handle, descriptor, rank, world], stmt,
        buffer_map={out_handle: out, descriptor: descriptor_buf},
    ).with_attr("target", tvm.target.Target(TARGET)).with_attr(
        "tl.dcu_ipc.abi_ready", 1,
    ).with_attr("tl.dcu_ipc.bindings", "partial=ipc_view_partial")
    if attr_peer_count is not None:
        func = func.with_attr("tl.dcu_ipc.pe_count", attr_peer_count)
    return func


def _apply(func):
    return tilelang.transform.VectorizeDCUIPCCommunication()(tvm.IRModule({"main": func}))["main"]


def _has_ramp(stmt):
    seen = []
    tvm.tir.stmt_functor.post_order_visit(
        stmt, lambda node: seen.append(node) if isinstance(node, tvm.tir.Ramp) else None,
    )
    return bool(seen)


@pytest.mark.parametrize("peer_count", [4, 8])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("direct", [False, True], ids=["conditional", "direct"])
def test_generalized_rs_vectorizes_exact_peer_count_and_16bit_payload(peer_count, dtype, direct):
    original = _generic_rs(peer_count=peer_count, dtype=dtype, direct=direct, attr_peer_count=peer_count)
    transformed = _apply(original)
    assert _has_ramp(transformed.body)


def test_missing_pe_attribute_preserves_legacy_four_peer_default():
    assert _has_ramp(_apply(_generic_rs(peer_count=4, dtype="float16", direct=False, attr_peer_count=None)).body)


@pytest.mark.parametrize("bad_peer_count", [0, 3, 5, 9])
def test_unsupported_pe_contract_fails_closed(bad_peer_count):
    original = _generic_rs(peer_count=4, dtype="float16", direct=False, attr_peer_count=bad_peer_count)
    tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)


def test_mixed_16bit_payload_dtypes_are_rejected():
    original = _generic_rs(peer_count=8, dtype="bfloat16", direct=True,
                           attr_peer_count=8, mixed_dtype=True)
    tvm.ir.assert_structural_equal(original, _apply(original), map_free_vars=True)
