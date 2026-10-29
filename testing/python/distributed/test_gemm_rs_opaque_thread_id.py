"""Source-level coverage for GEMM+RS's opaque communication thread id."""

import importlib.util
from pathlib import Path
import sys

from tilelang import tvm
import tilelang


TARGET = "hip -mcpu=gfx936"
PROTOTYPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "distributed" / "gemm_rs_prototype"


def _load_module(name: str):
    path = PROTOTYPE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source(bound) -> str:
    with tvm.target.Target(TARGET):
        artifact = tilelang.lower(
            bound,
            target=TARGET,
            enable_device_compile=False,
            enable_host_codegen=False,
        )
    return artifact.kernel_source


def _assert_opaque_thread_coverage(module) -> None:
    """Check that the historic 512-thread linear RS traversal is complete."""
    assert module.THREADS == 512
    covered = set()
    for tx in range(module.THREADS):
        for linear in range(tx, module.TILE_M * module.TILE_N, module.THREADS):
            assert linear not in covered
            covered.add(linear)
    assert covered == set(range(module.TILE_M * module.TILE_N))


def test_gemm_rs_keeps_gemm_codegen_and_uses_opaque_thread_rs_mapping():
    original_modules = {name: sys.modules.get(name) for name in (
        "gemm_reduce_scatter",
        "gemm_reduce_scatter_tile_grid",
    )}
    try:
        serial = _load_module("gemm_reduce_scatter")
        tile_grid = _load_module("gemm_reduce_scatter_tile_grid")
        for module in (serial, tile_grid):
            _assert_opaque_thread_coverage(module)
            _, bound = module.build_bound(1024, 512, 256)
            source = _source(bound)
            # The generic GEMM codegen remains intact while the owner-side
            # communication loop keeps its historic opaque coordinate.
            assert "tl::cp_async_gs<16" in source
            assert "__builtin_amdgcn_mmac_f32_16x16x16f16" in source
            assert "__launch_bounds__" not in source
            assert "tl::ipc::opaque_thread_id(" in source

            # Restrict the structural checks to the owner-side RS section so
            # ordinary GEMM thread indexing cannot satisfy them accidentally.
            rs_source = source.rsplit("if (__tl_ipc_rank ==", 1)[-1]
            assert "int comm_tx = tl::ipc::opaque_thread_id(((int)threadIdx.x));" in rs_source
            assert "tl::get_warp_idx_sync(64)" not in rs_source
            assert "tl::get_lane_idx(64)" not in rs_source
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_matched_gemm_async_a_source_uses_lane_intrin_but_keeps_raw_shared_destination():
    """The 512-thread matched GEMM must not expose raw tx sublanes to LLVM."""
    original_modules = {name: sys.modules.get(name) for name in (
        "gemm_reduce_scatter",
        "gemm_reduce_scatter_scheduled",
    )}
    try:
        _load_module("gemm_reduce_scatter")
        scheduled = _load_module("gemm_reduce_scatter_scheduled")
        _, bound = scheduled.build_bound(4096, 4096, 1024, total_blocks=80, comm_blocks=32)
        source = _source(bound)
        assert "tl::ipc::opaque_thread_id(((int)threadIdx.x))" in source
        a_copy_lines = [line for line in source.splitlines()
                        if "tl::cp_async_gs<16" in line and "a_dcu_res" in line]
        assert a_copy_lines
        for line in a_copy_lines:
            # The first cp_async argument is the LDS destination: it must
            # retain raw tx.  The source side after a_dcu_res is lane-only,
            # preventing tx bit 8 from reaching the unsafe max-256 combine.
            assert "threadIdx.x" in line.split("a_dcu_res", 1)[0]
            source_offset = line.split("a_dcu_res", 1)[1]
            assert "tl::get_lane_idx(64)" in source_offset
            assert "threadIdx.x" not in source_offset
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
