"""Real generalized DSL bind/lower smoke tests (requires the DCU build env)."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.distributed.ipc_binding import bind_ipc

from examples.distributed.fusion_generalized.allgather_gemm import allgather_gemm
from examples.distributed.fusion_generalized.contract import make_plan
from examples.distributed.fusion_generalized.gemm_allreduce import gemm_allreduce
from examples.distributed.fusion_generalized.gemm_reduce_scatter import gemm_reduce_scatter


def _bound(kind, generator, plan):
    bindings = {"ready": "ipc_view_ready"}
    if kind == "ag":
        bindings["A_shard"] = "ipc_view_a_shard"
    else:
        bindings["partial"] = "ipc_view_partial"
    if kind == "ar":
        bindings["out"] = "ipc_view_out"
    return bind_ipc(generator(plan), bindings)


@pytest.mark.parametrize("kind,generator,marker", [
    ("ag", allgather_gemm, "ipc_ag_half8_i"),
    ("rs", gemm_reduce_scatter, "ipc_half8_i"),
    ("ar", gemm_allreduce, "ipc_ar_half8_i"),
])
@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_real_generalized_dsl_binds_and_lowers_to_verified_ipc_vectors(kind, generator, marker,
                                                                        world_size, dtype):
    plan = make_plan(kind, 1031, 521, 259, world_size=world_size, dtype=dtype)
    bound = _bound(kind, generator, plan)
    with tvm.target.Target("hip -mcpu=gfx936"):
        with tvm.transform.PassContext(config={
            "tl.enable_dcu_ipc_hoist": True,
            "tl.enable_dcu_ipc_vectorize": True,
        }):
            artifact = tilelang.lower(
                bound, target="hip -mcpu=gfx936", enable_device_compile=False, enable_host_codegen=False,
            )
    source = artifact.kernel_source
    assert marker in source
    assert "uint4" in source
    assert "tl::ipc::peer<std::byte>" in source
    assert source.find("tl::ipc::peer<std::byte>") < source.find(marker)
    if dtype == "bfloat16":
        assert "bfloat16" in source or "__bf16" in source
        # The real BF16 vector-to-FP32 lowering must not assume bfloat16x2
        # has .x/.y members: dcu_hip defines it as a compiler vector.
        assert "((bfloat16x2*)(&(" not in source
        if kind != "ag":
            assert "((bfloat16_t*)(&(" in source
    else:
        assert "float16" in source or "half" in source


@pytest.mark.parametrize("kind,generator,marker", [
    ("ag", allgather_gemm, "ipc_ag_half8_i"),
    ("rs", gemm_reduce_scatter, "ipc_half8_i"),
    ("ar", gemm_allreduce, "ipc_ar_half8_i"),
])
def test_unaligned_kp768_ag_and_persistent_multitile_shapes_still_lower(kind, generator, marker):
    # AG K=513 gives K_p=768, exercising fixed 65536-element chunks that do
    # not align to K_p.  The larger N also forces persistent multi-tile work.
    plan = make_plan(kind, 1031, 1801, 513 if kind == "ag" else 65, world_size=8,
                     dtype="bfloat16")
    assert plan.k_physical == (768 if kind == "ag" else 128)
    with tvm.target.Target("hip -mcpu=gfx936"):
        with tvm.transform.PassContext(config={
            "tl.enable_dcu_ipc_hoist": True,
            "tl.enable_dcu_ipc_vectorize": True,
        }):
            source = tilelang.lower(
                _bound(kind, generator, plan), target="hip -mcpu=gfx936",
                enable_device_compile=False, enable_host_codegen=False,
            ).kernel_source
    assert marker in source and "uint4" in source


@pytest.mark.parametrize("kind,generator,marker", [
    ("ag", allgather_gemm, "ipc_ag_half8_i"),
    ("rs", gemm_reduce_scatter, "ipc_half8_i"),
    ("ar", gemm_allreduce, "ipc_ar_half8_i"),
])
@pytest.mark.parametrize("world_size,k", [
    # Compact boundary matrix: every fusion kind and both PE contracts are
    # lowered, while collectively covering K=1/7/64/65.
    (4, 1), (4, 64), (8, 7), (8, 65),
])
def test_tiny_k_persistent_pipeline_still_lowers_to_half8(kind, generator, marker,
                                                           world_size, k):
    plan = make_plan(kind, 3, 5, k, world_size=world_size, dtype="bfloat16")
    assert plan.k_physical == (256 if kind == "ag" else 128)
    with tvm.target.Target("hip -mcpu=gfx936"):
        with tvm.transform.PassContext(config={
            "tl.enable_dcu_ipc_hoist": True,
            "tl.enable_dcu_ipc_vectorize": True,
        }):
            source = tilelang.lower(
                _bound(kind, generator, plan), target="hip -mcpu=gfx936",
                enable_device_compile=False, enable_host_codegen=False,
            ).kernel_source
    assert marker in source and "uint4" in source
