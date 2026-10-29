"""Real TIR-level tests for the M1 distributed DSL.

These are deliberately compiler-IR tests, not device communication tests:
M2 owns HIP/ROC_SHMEM lowering and hardware ordering litmus coverage.
"""

import unittest

import tvm
from tvm import ir, tir

import tilelang
from tilelang import language as T
from tilelang.language.distributed import DistributedIRError


def _buffer(name: str, *, dtype: str = "float32", shape=(2, 4), strides=(8, 1), offset=3) -> tir.Buffer:
    data = tir.Var(name + "_data", ir.PointerType(ir.PrimType(dtype), "global"))
    return tir.decl_buffer(
        shape, dtype, name=name, data=data, strides=strides, elem_offset=offset,
        scope="global", data_alignment=64, offset_factor=1,
    )


def _func(body: tir.Stmt, base: tir.Buffer, *scalars: tir.Var) -> tir.PrimFunc:
    return tir.PrimFunc([base.data, *scalars], body, buffer_map={base.data: base})


def _calls(node, op_name: str):
    found = []

    def visit(item):
        if isinstance(item, tir.Call) and isinstance(item.op, ir.Op) and item.op.name == op_name:
            found.append(item)

    tir.stmt_functor.post_order_visit(node, visit)
    return found


def _decl_buffers(node):
    found = []

    def visit(item):
        if isinstance(item, tir.DeclBuffer):
            found.append(item.buffer)

    tir.stmt_functor.post_order_visit(node, visit)
    return found


class DistributedM1IRTest(unittest.TestCase):
    def test_registered_ops_are_explicit_and_opaque(self):
        opaque = (
            "tl.get_rank", "tl.get_num_ranks", "tl.remote_address", "tl.fence_sys",
            "tl.signal_set", "tl.tileop.wait", "tl.tileop.put_block", "tl.tileop.get_block",
            "tl.tileop.st", "tl.tileop.ld",
        )
        for name in opaque:
            effect = tir.op.Op.get(name).get_attr("TCallEffectKind")
            self.assertIsNotNone(effect, name)
            self.assertEqual(int(effect), 3, name)
        self.assertEqual(int(tir.op.Op.get("tl.remote_view_token").get_attr("TCallEffectKind")), 5)

    def test_remote_view_normalizes_unique_typed_aliases_and_offset_once(self):
        base = _buffer("Stage")
        first, second = T.remote_view(base, 1), T.remote_view(base, 2)
        func = _func(tir.SeqStmt([tir.Evaluate(T.copy(base, first)), tir.Evaluate(T.copy(base, second))]), base)

        with self.assertRaises(DistributedIRError):
            T.verify_distributed_ir(func)
        normalized = T.normalize_remote_views(func)
        T.verify_distributed_ir(normalized)

        aliases = _decl_buffers(normalized.body)
        self.assertEqual(len(aliases), 2)
        self.assertFalse(aliases[0].data.same_as(aliases[1].data))
        self.assertEqual([int(alias.elem_offset.value) for alias in aliases], [3, 3])
        self.assertEqual([str(alias.dtype) for alias in aliases], ["float32", "float32"])
        self.assertEqual([list(alias.shape) for alias in aliases], [list(base.shape), list(base.shape)])
        self.assertEqual([list(alias.strides) for alias in aliases], [list(base.strides), list(base.strides)])
        self.assertEqual([alias.data_alignment for alias in aliases], [base.data_alignment, base.data_alignment])
        addresses = _calls(normalized.body, "tl.remote_address")
        self.assertEqual(len(addresses), 2)
        self.assertEqual([int(call.args[1].value) for call in addresses], [1, 2])
        self.assertFalse(_calls(normalized.body, "tl.remote_view_token"))

    def test_same_remote_view_multiple_accesses_get_ssa_distinct_bindings(self):
        base = _buffer("Stage")
        remote = T.remote_view(base, 1)
        func = _func(tir.SeqStmt([tir.Evaluate(T.copy(base, remote)), tir.Evaluate(T.copy(base, remote))]), base)
        normalized = T.normalize_remote_views(func)
        aliases = _decl_buffers(normalized.body)
        self.assertEqual(len(aliases), 2)
        self.assertFalse(aliases[0].data.same_as(aliases[1].data))
        self.assertEqual(len(_calls(normalized.body, "tl.remote_address")), 2)
        tvm.tir.analysis.verify_well_formed(tvm.IRModule({"main": normalized}))

    def test_remote_false_is_ordinary_local_alias(self):
        base = _buffer("Stage")
        local = T.remote_view(base, 7, remote=False)
        func = _func(tir.Evaluate(T.copy(base, local)), base)
        normalized = T.normalize_remote_views(func)
        self.assertFalse(_calls(normalized.body, "tl.remote_address"))
        self.assertFalse(_decl_buffers(normalized.body))
        T.verify_distributed_ir(normalized)
        int_local = T.remote_view(base, tir.const(-1, "int32"))
        self.assertFalse(_calls(_func(tir.Evaluate(T.copy(base, int_local)), base).body, "tl.remote_view_token"))

    def test_uniform_scalar_loop_and_blockidx_peers_bind_inside_scope(self):
        base = _buffer("Stage")
        peer = tir.Var("peer", "int32")
        scalar_func = _func(tir.Evaluate(T.copy(base, T.remote_view(base, peer))), base, peer)
        self.assertTrue(_calls(T.normalize_remote_views(scalar_func).body, "tl.remote_address"))

        i = tir.Var("i", "int32")
        loop_body = tir.Evaluate(T.copy(base, T.remote_view(base, i + 1)))
        loop_func = _func(tir.For(i, 0, 2, tir.ForKind.SERIAL, loop_body), base)
        loop_normalized = T.normalize_remote_views(loop_func)
        self.assertIsInstance(loop_normalized.body, tir.For)
        self.assertTrue(_calls(loop_normalized.body, "tl.remote_address"))

        bx = tir.Var("x", "int32")
        binding = tir.IterVar((0, 4), bx, iter_type=tir.IterVar.ThreadIndex, thread_tag="blockIdx.x")
        block_body = tir.Evaluate(T.copy(base, T.remote_view(base, bx + 1)))
        block_func = _func(
            tir.For(bx, 0, 4, tir.ForKind.THREAD_BINDING, block_body, thread_binding=binding), base
        )
        self.assertTrue(_calls(T.normalize_remote_views(block_func).body, "tl.remote_address"))

    def test_thread_varying_unknown_and_nonuniform_loop_peers_are_rejected(self):
        base = _buffer("Stage")
        tx = tir.Var("tx", "int32")
        thread_binding = tir.IterVar((0, 32), tx, iter_type=tir.IterVar.ThreadIndex, thread_tag="threadIdx.x")
        body = tir.Evaluate(T.copy(base, T.remote_view(base, tx + 1)))
        thread_func = _func(
            tir.For(tx, 0, 32, tir.ForKind.THREAD_BINDING, body, thread_binding=thread_binding), base
        )
        with self.assertRaisesRegex(DistributedIRError, "not block-uniform"):
            T.normalize_remote_views(thread_func)

        unknown = tir.Var("unknown", "int32")
        unknown_func = _func(tir.Evaluate(T.copy(base, T.remote_view(base, unknown))), base)
        with self.assertRaisesRegex(DistributedIRError, "not block-uniform"):
            T.normalize_remote_views(unknown_func)

        i = tir.Var("i", "int32")
        nonuniform_loop = tir.For(
            i, tx, 2, tir.ForKind.SERIAL, tir.Evaluate(T.copy(base, T.remote_view(base, i)))
        )
        nonuniform_func = _func(nonuniform_loop, base)
        with self.assertRaisesRegex(DistributedIRError, "not block-uniform"):
            T.normalize_remote_views(nonuniform_func)
        with self.assertRaisesRegex(TypeError, "integer rank"):
            T.remote_view(base, tir.const(1.5, "float32"))
        with self.assertRaisesRegex(TypeError, "scalar integer"):
            T.remote_view(base, tir.Var("lanes", "int32x4"))

    def test_unsupported_remote_load_position_fails_closed(self):
        base = _buffer("Stage")
        remote = T.remote_view(base, 1)
        tmp = tir.Var("tmp", "float32")
        func = _func(tir.LetStmt(tmp, tir.BufferLoad(remote, [0, 0]), tir.Evaluate(0)), base)
        with self.assertRaisesRegex(DistributedIRError, "no legal pointer binding scope"):
            T.normalize_remote_views(func)

    def test_copy_and_scalar_primitives_validate_types_defaults_and_intervals(self):
        src, dst = _buffer("Src"), _buffer("Dst")
        put = T.put_block(src[0, 0], dst[0, 0], 4, dst_pe=3)
        self.assertEqual([int(arg.value) for arg in put.args[2:]], [4, 3])
        get = T.get_block(src[0, 0], dst[0, 0], 4, src_pe=2)
        self.assertEqual([int(arg.value) for arg in get.args[2:]], [4, 2])
        with self.assertRaisesRegex(ValueError, "row"):
            T.put_block(src[0, 3], dst[0, 0], 2)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            T.get_block(src[0, 0], dst[0, 0], tir.const(-1, "int32"))
        with self.assertRaisesRegex(ValueError, "unit stride"):
            T.put_block(_buffer("Bad", strides=(8, 2))[0, 0], dst[0, 0], 1)
        with self.assertRaisesRegex(ValueError, "index must be non-negative"):
            T.put_block(src[-1, 0], dst[0, 0], 1)
        with self.assertRaisesRegex(ValueError, "axis 0"):
            T.put_block(src[2, 0], dst[0, 0], 1)
        with self.assertRaisesRegex(TypeError, "scalar integer"):
            T.put_block(src[0, 0], dst[0, 0], tir.const(1.5, "float32"))
        dynamic = tir.Var("count", "int32")
        with self.assertRaisesRegex(ValueError, "explicit range proof"):
            T.put_block(src[0, 0], dst[0, 0], dynamic)
        raw_src, raw_dst = T.region(src[0, 0], "r", 1, 4), T.region(dst[0, 0], "w", 1, 4)
        with self.assertRaisesRegex(ValueError, "row"):
            T.put_block(raw_src, raw_dst, 5)

        load_dst = _buffer("LoadDst", dtype="float32")
        ld = T.ld(src[0, 0], load_dst[0, 0])
        st = T.st(dst[0, 0], tir.const(7, "float32"))
        self.assertEqual([int(arg.value) for arg in ld.args[2:]], [0, 1, -1])
        self.assertEqual([int(arg.value) for arg in st.args[2:]], [0, 1, -1])
        with self.assertRaises(TypeError):
            T.ld(src[0, 0], tir.const(0, "float32"))
        with self.assertRaises(NotImplementedError):
            T.st(dst[0, 0], tir.const(1, "float32"), cache="ca")
        with self.assertRaises(NotImplementedError):
            T.ld(src[0, 0], load_dst[0, 0], cache="multimem")
        for multimem in (T.multimem_ld_reduce, T.multimem_st, T.multimem_red, T.multimem_tma_store,
                          T.multimem_signal, T.multimem_signal_add):
            with self.assertRaisesRegex(NotImplementedError, "multimem"):
                multimem(src, dst)
        with self.assertRaises(ValueError):
            T.st(dst[0, 0], tir.const(1, "float32"), scope="cta")
        with self.assertRaises(TypeError):
            T.ld(src[0, 0], _buffer("WrongDType", dtype="int32")[0, 0])
        with self.assertRaises(TypeError):
            T.st(_buffer("Half", dtype="float16")[0, 0], tir.const(1, "float16"))

    def test_sync_defaults_and_invalid_input_validation(self):
        flag = _buffer("Ready", dtype="uint32", shape=(4,), strides=(1,), offset=0)
        wait = T.wait_eq(flag[0], 1)
        self.assertEqual([int(arg.value) for arg in wait.args[0::1] if isinstance(arg, tir.IntImm)][-3:], [-1, 0, 0])
        with self.assertRaises(TypeError):
            T.wait_ge(_buffer("NotFlag")[0, 0], 1)
        with self.assertRaises(ValueError):
            T.wait_eq(flag[0], -1)
        with self.assertRaises(ValueError):
            T.signal_set(flag[0], 1 << 32)
        with self.assertRaises(ValueError):
            T.wait_eq(flag[0], 1, scope=1)  # raw integer enum cannot expand M1 scope
        raw_float_flag = _buffer("RawFloat")
        raw_wait = tir.call_intrin(
            "handle", tir.op.Op.get("tl.tileop.wait"), 0,
            T.region(raw_float_flag[0, 0], "r", 1, 1), 1, -1, 0, 0,
        )
        with self.assertRaisesRegex(DistributedIRError, "uint32"):
            T.verify_distributed_ir(_func(tir.Evaluate(raw_wait), raw_float_flag))
        raw_float_peer = tir.call_intrin(
            "handle", tir.op.Op.get("tl.tileop.wait"), 0,
            T.region(flag[0], "r", 1), 1, tir.const(1.5, "float32"), 0, 0,
        )
        with self.assertRaisesRegex(DistributedIRError, "integer rank"):
            T.verify_distributed_ir(_func(tir.Evaluate(raw_float_peer), flag))

    def test_public_v2_prim_func_frontend_builds_distributed_ir(self):
        """Exercise public ``@T.prim_func`` construction, not the parser proxy."""

        @T.prim_func
        def public_distributed(
            src: T.Tensor((4,), T.float32),
            dst: T.Tensor((4,), T.float32),
            flag: T.Tensor((1,), T.uint32),
        ):
            remote_dst = T.remote_view(dst, 1)
            with T.sm_specialize_scope(auto_ws=False):
                T.copy(src, remote_dst)
                T.fence_sys()
                T.signal_set(flag[0], 1)
                T.wait_eq(flag[0], 1)
                T.put_block(src[0], dst[0], 4, dst_pe=1)
                T.get_block(src[0], dst[0], 4, src_pe=1)
                T.st(dst[0], T.float32(1))
                T.ld(src[0], dst[0])

        with self.assertRaises(DistributedIRError):
            T.verify_distributed_ir(public_distributed)
        normalized = T.normalize_remote_views(public_distributed)
        T.verify_distributed_ir(normalized)
        self.assertTrue(_calls(normalized.body, "tl.remote_address"))


if __name__ == "__main__":
    unittest.main()
