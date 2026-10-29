"""Independent M2 review cases: public compilation and explicit bindings.

These tests do not replace multi-PE GPU execution.  Run with the isolated old
environment and its matching native build.  M0 candidate tests are unrelated.
"""

import unittest
from unittest.mock import patch

import tilelang
from tilelang import language as T
from tilelang.distributed.ipc_binding import bind_ipc, prepare_ipc_primfunc
from tvm import ir, tir


def copy_function():
    @T.prim_func
    def copy_peer(src: T.Tensor((16,), "float32"), dst: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=64):
            remote = T.remote_view(dst, (T.get_rank() + 1) % T.get_num_ranks())
            T.copy(src, remote)

    return copy_peer


def primitive_compile_function():
    # Compilation coverage only, not an application publication protocol.
    @T.prim_func
    def primitive_compile(src: T.Tensor((16,), "uint32"),
                          dst: T.Tensor((16,), "uint32"),
                          flag: T.Tensor((1,), "uint32")):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            peer = (T.get_rank() + 1) % T.get_num_ranks()
            T.put_block(src, dst, 16, peer=peer)
            T.fence_sys()
            T.sync_threads()
            if tx == 0:
                T.signal_set(flag[0], 1, peer=peer)
                T.wait_eq(flag[0], 1)
                T.wait_ge(flag[0], 1)
                T.st(dst[0], T.uint32(7), scope="sys", sem="release", peer=peer)
                T.ld(dst[0], src[0], scope="sys", sem="acquire", peer=peer)
            T.get_block(dst, src, 16, peer=peer)

    return primitive_compile


def scalar_compile_function(dtype, scope, semantics):
    store_semantics = "release" if semantics == "acquire" else semantics

    @T.prim_func
    def scalar_compile(src: T.Tensor((1,), dtype), dst: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=64):
            if T.get_thread_binding() == 0:
                T.ld(src[0], dst[0], scope=scope, sem=semantics, peer=1)
                T.st(dst[0], T.cast(3, dtype), scope=scope, sem=store_semantics, peer=1)

    return scalar_compile


def rank_only_function():
    @T.prim_func
    def rank_only(dst: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=64):
            if T.get_thread_binding() == 0:
                dst[0] = T.get_rank()
                dst[1] = T.get_num_ranks()

    return rank_only


def rank_expression_function():
    @T.prim_func
    def rank_expression(dst: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=64):
            if T.get_thread_binding() == 0:
                if T.get_num_ranks() == 4:
                    dst[T.get_rank()] = (T.get_rank() + 1) % T.get_num_ranks()

    return rank_expression


def padded_copy_function():
    @T.prim_func
    def padded_copy(src_handle: T.handle, dst_handle: T.handle):
        src = T.match_buffer(src_handle, (2, 4), "float32", strides=(8, 1), elem_offset=3)
        dst = T.match_buffer(dst_handle, (2, 4), "float32", strides=(8, 1), elem_offset=3)
        with T.Kernel(1, threads=64):
            remote = T.remote_view(dst, 1)
            T.put_block(src[1, 2], remote[1, 2], 2)

    return padded_copy


def loop_copy_function():
    @T.prim_func
    def loop_copy(src: T.Tensor((16,), "uint32"), dst: T.Tensor((16,), "uint32")):
        with T.Kernel(1, threads=64):
            for epoch in T.serial(3):
                T.put_block(src, dst, 16, peer=1)
                T.fence_sys()
                T.sync_threads()

    return loop_copy


class IndependentM2Review(unittest.TestCase):
    def test_binding_does_not_mutate_original(self):
        original = copy_function()
        before = ir.save_json(original)
        bound = bind_ipc(original, ["dst"])
        prepared = prepare_ipc_primfunc(bound, target_kind="hip")
        self.assertEqual(ir.save_json(original), before)
        self.assertEqual(len(prepared.params), len(original.params) + 3)
        descriptor = prepared.buffer_map[prepared.params[-3]]
        self.assertEqual(str(descriptor.dtype), "uint64")
        self.assertEqual([int(i) for i in descriptor.shape], [6])
        self.assertEqual([str(p.dtype) for p in prepared.params[-2:]], ["int32", "int32"])
        self.assertTrue(ir.structural_equal(
            prepared, prepare_ipc_primfunc(prepared, target_kind="hip")))

    def test_source_only_is_real_lower_without_compiler_process(self):
        func = bind_ipc(copy_function(), ["dst"])
        with patch("subprocess.run", side_effect=AssertionError("unexpected compilation")), \
                patch("subprocess.check_output", side_effect=AssertionError("unexpected device probing")):
            artifact = tilelang.lower(func, target="hip -mcpu=gfx936",
                                      enable_device_compile=False, enable_host_codegen=False)
        source = artifact.kernel_source
        self.assertIn("ipc", source)
        self.assertNotIn("remote_view_token", source)
        self.assertNotIn("rocshmem", source.lower())
        self.assertNotIn("dushmem", source.lower())
        self.assertNotIn("/public/home/baoboyuan/dcu_ipc", source)
        self.assertEqual(len(artifact.params), 5)

    def test_missing_remote_allocation_binding_fails(self):
        # Binding src does not authorize addressing another allocation (dst).
        func = bind_ipc(copy_function(), ["src"])
        with self.assertRaises(Exception):
            tilelang.lower(func, target="hip -mcpu=gfx936")

    def test_all_primitives_source_only(self):
        func = bind_ipc(primitive_compile_function(), ["src", "dst", "flag"])
        artifact = tilelang.lower(func, target="hip -mcpu=gfx936")
        for helper in ("put_block", "get_block", "fence_sys", "store_release",
                       "load_acquire", "wait_eq_acquire", "wait_ge_acquire"):
            self.assertIn(helper, artifact.kernel_source)
        self.assertEqual(len(artifact.params), 8)

    def test_non_hip_target_fails(self):
        with self.assertRaisesRegex(ValueError, "HIP"):
            prepare_ipc_primfunc(bind_ipc(copy_function(), ["dst"]), target_kind="cuda")

    def test_rank_queries_survive_without_remote_memory_operation(self):
        artifact = tilelang.lower(bind_ipc(rank_only_function(), ["dst"]),
                                  target="hip -mcpu=gfx936")
        self.assertIn("__tl_ipc_rank", artifact.kernel_source)
        self.assertIn("__tl_ipc_world_size", artifact.kernel_source)

    def test_rank_expressions_are_real_abi_uses(self):
        artifact = tilelang.lower(bind_ipc(rank_expression_function(), ["dst"]),
                                  target="hip -mcpu=gfx936")
        self.assertIn("__tl_ipc_rank", artifact.kernel_source)
        self.assertIn("__tl_ipc_world_size", artifact.kernel_source)

    def test_padded_remote_copy_preserves_physical_offset(self):
        artifact = tilelang.lower(bind_ipc(padded_copy_function(), ["dst"]),
                                  target="hip -mcpu=gfx936")
        # elem_offset 3 + row 1 * stride 8 + column 2 = element 13.
        self.assertIn("src[13]", artifact.kernel_source)
        self.assertGreaterEqual(artifact.kernel_source.count("[13]"), 2)
        self.assertIn("put_block<float>", artifact.kernel_source)

    def test_divergent_cooperative_copy_fails(self):
        @T.prim_func
        def partial_participation(src: T.Tensor((16,), "float32"), dst: T.Tensor((16,), "float32")):
            with T.Kernel(1, threads=64):
                tx = T.get_thread_binding()
                if tx == 0:
                    T.put_block(src, dst, 16, peer=1)

        # Only lane zero would otherwise copy element zero and leave the rest
        # untouched. A cooperative primitive must not silently accept this.
        with self.assertRaisesRegex(Exception, "block-uniform|cooperative|all threads"):
            tilelang.lower(bind_ipc(partial_participation, ["dst"]), target="hip -mcpu=gfx936")

    def test_divergent_explicit_cta_barrier_fails(self):
        @T.prim_func
        def partial_sync(dst: T.Tensor((16,), "float32")):
            with T.Kernel(1, threads=256):
                if T.get_thread_binding() == 0:
                    T.sync_threads()

        with self.assertRaisesRegex(Exception, "block-uniform|cooperative|all threads|thread-varying"):
            tilelang.lower(bind_ipc(partial_sync, ["dst"]), target="hip -mcpu=gfx936")

    def test_thread_local_condition_cannot_guard_cooperative_work(self):
        @T.prim_func
        def indirect_partial_sync(dst: T.Tensor((16,), "float32")):
            with T.Kernel(1, threads=256):
                gate = T.alloc_local((1,), "int32")
                gate[0] = T.get_thread_binding()
                if gate[0] == 0:
                    T.sync_threads()

        # A local load hides the direct threadIdx reference from a shallow
        # visitor but is not a proof that all threads participate.
        with self.assertRaisesRegex(Exception, "block-uniform|cooperative|all threads|thread-varying"):
            prepare_ipc_primfunc(bind_ipc(indirect_partial_sync, ["dst"]), target_kind="hip")

    def test_let_alias_cannot_hide_divergent_cta_barrier(self):
        tx = tir.Var("tx", "int32")
        gate = tir.Var("gate", "int32")
        dst = tir.decl_buffer((16,), "float32", name="dst")
        barrier = tir.Evaluate(tir.call_intrin("int32", "tir.tvm_storage_sync", "shared"))
        body = tir.LetStmt(gate, tx + 1, tir.IfThenElse(gate == 1, barrier, None))
        thread = tir.IterVar(None, tx, tir.IterVar.ThreadIndex, "threadIdx.x")
        body = tir.AttrStmt(thread, "thread_extent", 256, body)
        func = tir.PrimFunc([dst.data], body, buffer_map={dst.data: dst})
        with self.assertRaisesRegex(Exception, "block-uniform|cooperative|all threads|thread-varying"):
            prepare_ipc_primfunc(bind_ipc(func, ["dst"]), target_kind="hip")

    def test_thread_varying_loop_cannot_contain_cta_barrier(self):
        tx = tir.Var("tx", "int32")
        index = tir.Var("index", "int32")
        dst = tir.decl_buffer((16,), "float32", name="dst")
        barrier = tir.Evaluate(tir.call_intrin("int32", "tir.tvm_storage_sync", "shared"))
        body = tir.For(index, 0, tx, tir.ForKind.SERIAL, barrier)
        thread = tir.IterVar(None, tx, tir.IterVar.ThreadIndex, "threadIdx.x")
        func = tir.PrimFunc([dst.data], tir.AttrStmt(thread, "thread_extent", 256, body),
                           buffer_map={dst.data: dst})
        with self.assertRaisesRegex(Exception, "block-uniform|cooperative|all threads|thread-varying"):
            prepare_ipc_primfunc(bind_ipc(func, ["dst"]), target_kind="hip")

    def test_uniform_rank_guard_retains_cta_barrier(self):
        @T.prim_func
        def uniform_sync(dst: T.Tensor((16,), "float32")):
            with T.Kernel(1, threads=256):
                if T.get_rank() % 2 == 0:
                    for _ in T.serial(3):
                        T.sync_threads()

        artifact = tilelang.lower(bind_ipc(uniform_sync, ["dst"]), target="hip -mcpu=gfx936")
        self.assertIn("__syncthreads();", artifact.kernel_source)

    def test_unknown_and_colliding_bindings_fail(self):
        func = copy_function()
        with self.assertRaises(ValueError):
            bind_ipc(func, ["not_an_argument"])
        with self.assertRaises(ValueError):
            bind_ipc(func, {"src": "shared_descriptor", "dst": "shared_descriptor"})
        with self.assertRaises(ValueError):
            prepare_ipc_primfunc(bind_ipc(func, {"dst": "src"}), target_kind="hip")

    def test_ambiguous_buffer_name_fails(self):
        ba = tir.decl_buffer((16,), "float32", name="duplicate")
        bb = tir.decl_buffer((16,), "float32", name="duplicate")
        a, b = ba.data, bb.data
        func = tir.PrimFunc([a, b], tir.Evaluate(0), buffer_map={a: ba, b: bb})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            bind_ipc(func, ["duplicate"])


if __name__ == "__main__":
    unittest.main()
