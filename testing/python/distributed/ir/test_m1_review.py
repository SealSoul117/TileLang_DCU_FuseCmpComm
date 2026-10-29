"""Independent M1 acceptance probes maintained by the reviewing agent."""

import pytest
import tvm
from tvm import tir

from tilelang import language as T


def _function(buffer, body, *params):
    return tir.PrimFunc([buffer.data, *params], body, buffer_map={buffer.data: buffer})


def test_repeated_remote_view_use_preserves_ssa():
    base = tir.decl_buffer((16,), "float32", name="base")
    remote = T.remote_view(base, 1)
    body = tir.SeqStmt([
        tir.BufferStore(remote, tir.const(1, "float32"), [0]),
        tir.BufferStore(remote, tir.const(2, "float32"), [1]),
    ])
    result = T.normalize_remote_views(_function(base, body))
    assert tir.analysis.verify_ssa(result), result.script()
    T.verify_distributed_ir(result)


def test_wait_peer_is_checked_not_semantics_argument():
    flag = tir.decl_buffer((1,), "uint32", name="flag")
    tx = tir.Var("arbitrary_name", "int32")
    iv = tir.IterVar(tvm.ir.Range(0, 64), tx, tir.IterVar.ThreadIndex, "threadIdx.x")
    wait = T.wait_eq(tir.BufferLoad(flag, [0]), 1, peer=tx)
    body = tir.AttrStmt(iv, "thread_extent", 64, tir.Evaluate(wait))
    with pytest.raises(T.DistributedIRError, match="uniform"):
        T.verify_distributed_ir(_function(flag, body))


@pytest.mark.parametrize("bad_size", [tir.IntImm("int32", -1), 1.5, True])
def test_block_copy_rejects_invalid_element_count(bad_size):
    src = tir.decl_buffer((16,), "float32", name="src")
    dst = tir.decl_buffer((16,), "float32", name="dst")
    with pytest.raises((TypeError, ValueError)):
        T.put_block(src, dst, bad_size)


@pytest.mark.parametrize("index", [-1, 16])
def test_block_copy_rejects_invalid_start(index):
    src = tir.decl_buffer((16,), "float32", name="src")
    dst = tir.decl_buffer((16,), "float32", name="dst")
    with pytest.raises((TypeError, ValueError)):
        T.get_block(tir.BufferLoad(src, [index]), dst, 1)


def test_scalar_load_rejects_mismatched_destination_type():
    src = tir.decl_buffer((1,), "float32", name="src")
    dst = tir.decl_buffer((1,), "uint32", name="dst")
    with pytest.raises((TypeError, ValueError)):
        T.ld(src, dst)


def test_floating_point_peer_is_not_a_valid_rank():
    base = tir.decl_buffer((16,), "float32", name="base")
    peer = tir.Var("peer", "float32")
    with pytest.raises((TypeError, ValueError)):
        remote = T.remote_view(base, peer)
        func = _function(base, tir.BufferStore(remote, tir.const(1, "float32"), [0]), peer)
        T.verify_distributed_ir(func, require_normalized=False)


def test_raw_wait_region_cannot_bypass_flag_dtype_validation():
    base = tir.decl_buffer((1,), "float32", name="not_a_flag")
    region = T.region(tir.BufferLoad(base, [0]), "r", 1, 1)
    with pytest.raises((TypeError, ValueError)):
        raw_wait = tir.call_intrin("handle", tvm.ir.Op.get("tl.tileop.wait"), 0, region, 1, -1, 0, 0)
        T.verify_distributed_ir(_function(base, tir.Evaluate(raw_wait)))


@pytest.mark.parametrize("scope", [1, True, tir.IntImm("int32", 7)])
def test_wait_scope_rejects_unsupported_numeric_modes(scope):
    flag = tir.decl_buffer((1,), "uint32", name="flag")
    with pytest.raises((TypeError, ValueError)):
        T.wait_eq(tir.BufferLoad(flag, [0]), 1, scope=scope)


@pytest.mark.parametrize("value", [-1, 2**32, 1.5, tir.const(2**40, "uint64")])
def test_signal_rejects_values_outside_uint32(value):
    flag = tir.decl_buffer((1,), "uint32", name="flag")
    with pytest.raises((TypeError, ValueError)):
        T.signal_set(tir.BufferLoad(flag, [0]), value)


def test_tir_minus_one_peer_is_local():
    base = tir.decl_buffer((16,), "float32", name="base")
    local = T.remote_view(base, tir.const(-1, "int32"))
    result = T.normalize_remote_views(_function(base, tir.BufferStore(local, tir.const(1, "float32"), [0])))
    calls = []
    tir.stmt_functor.post_order_visit(result.body, lambda node: calls.append(node) if isinstance(node, tir.Call) else None)
    assert all(getattr(call.op, "name", "") != "tl.remote_address" for call in calls)


def test_thread_varying_serial_step_does_not_prove_uniform_peer():
    base = tir.decl_buffer((16,), "float32", name="base")
    tx, i = tir.Var("tx", "int32"), tir.Var("i", "int32")
    iv = tir.IterVar(tvm.ir.Range(0, 64), tx, tir.IterVar.ThreadIndex, "threadIdx.x")
    remote = T.remote_view(base, i)
    loop = tir.For(i, 0, 4, tir.ForKind.SERIAL,
                   tir.BufferStore(remote, tir.const(1, "float32"), [0]), step=tx + 1)
    func = _function(base, tir.AttrStmt(iv, "thread_extent", 64, loop))
    with pytest.raises(T.DistributedIRError, match="uniform"):
        T.normalize_remote_views(func)


def test_unknown_row_position_cannot_prove_padded_copy_contiguous():
    src = tir.decl_buffer((2, 4), "float32", strides=(8, 1), name="padded")
    dst = tir.decl_buffer((4,), "float32", name="dst")
    position = tir.Var("position", "int32")
    with pytest.raises((TypeError, ValueError)):
        T.put_block(tir.BufferLoad(src, [0, position]), dst, 2)


def test_raw_signal_uint64_overflow_is_rejected():
    flag = tir.decl_buffer((1,), "uint32", name="flag")
    region = T.region(tir.BufferLoad(flag, [0]), "w", 1, 1)
    raw_signal = tir.call_intrin("handle", tvm.ir.Op.get("tl.signal_set"), region, tir.const(2**40, "uint64"), -1)
    with pytest.raises((TypeError, ValueError)):
        T.verify_distributed_ir(_function(flag, tir.Evaluate(raw_signal)))


def test_auto_warp_specialization_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="CUDA"):
        T.sm_specialize_scope(auto_ws=True)


def test_thread_varying_peer_cannot_hide_behind_let_alias():
    base = tir.decl_buffer((4,), "float32", name="base")
    tx, peer = tir.Var("tx", "int32"), tir.Var("peer", "int32")
    iv = tir.IterVar(tvm.ir.Range(0, 64), tx, tir.IterVar.ThreadIndex, "threadIdx.x")
    remote = T.remote_view(base, peer)
    body = tir.LetStmt(peer, tx + 1, tir.BufferStore(remote, tir.const(1, "float32"), [0]))
    with pytest.raises(T.DistributedIRError, match="uniform"):
        T.normalize_remote_views(_function(base, tir.AttrStmt(iv, "thread_extent", 64, body)))


def test_uniform_let_peer_remains_bound_after_normalization():
    base = tir.decl_buffer((4,), "float32", name="base")
    rank, peer = tir.Var("rank", "int32"), tir.Var("peer", "int32")
    remote = T.remote_view(base, peer)
    body = tir.LetStmt(peer, rank + 1, tir.BufferStore(remote, tir.const(1, "float32"), [0]))
    result = T.normalize_remote_views(_function(base, body, rank))
    assert isinstance(result.body, tir.LetStmt)
    assert result.body.var.same_as(peer)
    assert tir.analysis.verify_ssa(result)
    T.verify_distributed_ir(result)


def test_remote_metadata_survives_real_json_serialization():
    base = tir.decl_buffer((2, 4), "float32", strides=(8, 1), elem_offset=3,
                           data_alignment=64, name="base")
    remote = T.remote_view(base, 1)
    original = _function(base, tir.BufferStore(remote, tir.const(1, "float32"), [0, 0]))
    restored = tvm.ir.load_json(tvm.ir.save_json(original))
    tvm.ir.assert_structural_equal(original, restored)
    result = T.normalize_remote_views(restored)
    T.verify_distributed_ir(result)
    tvm.ir.assert_structural_equal(result, tvm.ir.load_json(tvm.ir.save_json(result)))
