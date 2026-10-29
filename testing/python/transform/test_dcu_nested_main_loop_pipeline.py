"""Regression coverage for nested tile traversal around a DCU GEMM K loop."""

import importlib.util
import re
from pathlib import Path

import tilelang
from tilelang import tvm


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "examples" / "distributed" / "gemm_rs_prototype" / "python_schedule_probe.py"
TARGET = "hip -mcpu=gfx936"


def _probe_module():
    spec = importlib.util.spec_from_file_location("_dcu_nested_main_loop_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lower_probe(variant: str) -> str:
    probe = _probe_module()
    func = probe.make_probe(variant, 1024, 256, 256, 1)
    with tvm.target.Target(TARGET):
        artifact = tilelang.lower(
            func, target=TARGET, enable_device_compile=False, enable_host_codegen=False
        )
    return artifact.kernel_source


def _assert_four_stage_dcu_pipeline(hip: str) -> None:
    # Four-stage pipeline is still selected, including the register/warp path.
    assert hip.count("__global__") == 1
    assert "tl::cp_async_gs" in hip
    assert "warp_idx < 4" in hip
    # Waits use one K pipeline's static commit groups; they must not use the
    # enclosing tile traversal body's aggregate commit count (the old 20/10).
    assert "tl::cp_async_wait<4>()" in hip
    assert "tl::async_gld_fence(4)" in hip
    assert "tl::async_gld_fence(2)" in hip
    assert "tl::async_gld_fence(0)" in hip
    assert "tl::async_gld_fence(20)" not in hip
    assert "tl::async_gld_fence(10)" not in hip
    # The first prefetch pair must be made visible before LDS reads/MMAC.
    first_copy = hip.index("tl::cp_async_gs")
    first_barrier = hip.index("tl::wave_barrier", first_copy)
    first_lds_read = min(
        index for index in (hip.find("ds_read", first_copy), hip.find("buf_dyn_shmem", first_copy))
        if index >= 0
    )
    assert first_barrier < first_lds_read


def test_serial_hoisted_outer_tile_loop_keeps_k_pipeline_boundary():
    hip = _lower_probe("serial-hoisted")
    # The source-level serial tile traversal remains a real generated loop;
    # the compiler must not require a flat Python schedule to get correct waits.
    assert re.search(r"for \(int tile_row\s*=", hip)
    _assert_four_stage_dcu_pipeline(hip)


def test_flat_four_stage_baseline_remains_unchanged():
    _assert_four_stage_dcu_pipeline(_lower_probe("flat"))


def test_other_nested_tile_traversals_keep_the_same_k_pipeline_boundary():
    for variant in ("serial-inner-alloc", "persistent-hoisted"):
        _assert_four_stage_dcu_pipeline(_lower_probe(variant))
