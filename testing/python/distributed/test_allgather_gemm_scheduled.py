"""Schedule and source-only checks for the four-PE AllGather + GEMM prototype."""

import importlib.util
from pathlib import Path
import types

import pytest
import tilelang
from tilelang import tvm


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/distributed/ag_gemm_prototype/allgather_gemm_scheduled.py"
BENCHMARK = SCRIPT.with_name("benchmark_allgather_gemm.py")


def load_example():
    spec = importlib.util.spec_from_file_location("allgather_gemm_scheduled_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_benchmark():
    spec = importlib.util.spec_from_file_location("allgather_gemm_benchmark_test", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _is_persistent_tile_loop(loop):
    value = loop.annotations.get("tl.dcu_persistent_tile")
    return value == 1 or getattr(value, "value", None) == 1


def _rewrite_ag_no_tile_base(func, *, loop_min=None, loop_extent=None,
                              dynamic_0_or_1_extent=False):
    """Return a real AG+GEMM TIR negative case without changing its frontend.

    The marker's body is retained, including the actual pipelined GEMM and
    global-to-shared copies.  Only its induction variable is replaced by zero,
    making A/B resource bases block-uniform rather than tile-varying.  This is
    deliberately invalid unless the traversal is an explicit ``min=0,
    extent=1`` loop.
    """
    changed = 0
    unknown = tvm.tir.Var("unknown_extent", "int32") if dynamic_0_or_1_extent else None
    extra_params = []
    if isinstance(loop_extent, tvm.tir.Var):
        extra_params.append(loop_extent)

    def post(node):
        nonlocal changed
        if isinstance(node, tvm.tir.For) and _is_persistent_tile_loop(node):
            changed += 1
            zero = tvm.tir.const(0, node.loop_var.dtype)
            body = tvm.tir.stmt_functor.substitute(node.body, {node.loop_var: zero})
            return tvm.tir.For(
                node.loop_var,
                zero if loop_min is None else loop_min,
                (tvm.tir.Min(tvm.tir.Max(unknown, tvm.tir.const(0, "int32")),
                             tvm.tir.const(1, "int32")) if unknown is not None
                 else (node.extent if loop_extent is None else loop_extent)),
                node.kind,
                body,
                node.thread_binding,
                node.annotations,
            )
        return None

    body = tvm.tir.stmt_functor.ir_transform(
        func.body, None, post, ["tir.For", "tir.IfThenElse"],
    )
    assert changed == 1
    if dynamic_0_or_1_extent:
        return tvm.tir.PrimFunc(
            [*func.params, unknown], body, func.ret_type, func.buffer_map, func.attrs,
        )
    if extra_params:
        return tvm.tir.PrimFunc(
            [*func.params, *extra_params], body, func.ret_type, func.buffer_map, func.attrs,
        )
    return func.with_body(body)


def _lower_ag_tir(func):
    return tilelang.lower(
        func, target="hip -mcpu=gfx936", enable_device_compile=False, enable_host_codegen=False,
    )


@pytest.mark.parametrize("s,comm", [(2048, 32), (4096, 32), (8192, 24), (16384, 16)])
def test_fixed_chunks_cover_each_gathered_shard_once_and_ready_layout_is_disjoint(s, comm):
    mod = load_example()
    cfg = mod.schedule_parameters(s, s // 4, s)
    assert cfg["comm_blocks"] == comm
    assert cfg["rows_per_chunk"] * s == mod.CHUNK_ELEMENTS
    assert cfg["chunks_per_row_tile"] * cfg["rows_per_chunk"] == mod.TILE_M
    assert cfg["chunks_per_source"] * mod.CHUNK_ELEMENTS == (s // 4) * s
    assert cfg["total_chunks"] == 4 * cfg["chunks_per_source"]

    # Each rank traverses its own source first and rotates through the other
    # three.  The physical ready index remains a one-writer permutation of all
    # source chunks despite that local traversal order.
    for rank in range(4):
        physical = []
        sources = []
        for chunk in range(cfg["total_chunks"]):
            source_offset, source_chunk = divmod(chunk, cfg["chunks_per_source"])
            source_peer = (rank + source_offset) % 4
            sources.append(source_peer)
            physical.append(source_peer * cfg["chunks_per_source"] + source_chunk)
        assert sources[0] == rank
        assert sorted(physical) == list(range(cfg["total_chunks"]))

        tile_rows_per_source = (s // 4) // mod.TILE_M
        tiles_per_source = tile_rows_per_source * cfg["tile_cols"]
        tiles = []
        for logical_tile in range(cfg["tile_count"]):
            offset, within = divmod(logical_tile, tiles_per_source)
            source = (rank + offset) % 4
            tiles.append((source * tile_rows_per_source + within // cfg["tile_cols"],
                          within % cfg["tile_cols"]))
        assert tiles[0][0] // tile_rows_per_source == rank
        assert sorted(tiles) == [(row, col) for row in range(cfg["tile_rows"])
                                 for col in range(cfg["tile_cols"])]

    chunks = set(range(cfg["total_chunks"]))
    peer_done = set(range(cfg["peer_done_base"], cfg["peer_done_base"] + 4))
    block_done = set(range(cfg["block_done_base"], cfg["block_done_base"] + comm))
    assert not chunks & peer_done
    assert not chunks & block_done
    assert not peer_done & block_done
    assert max(chunks | peer_done | block_done) + 1 == cfg["flag_words"]


def test_half8_mapper_is_a_bijection_of_a_fixed_65536_half_chunk():
    covered = {
        i * 512 + tx
        for i in range(128)
        for tx in range(512)
    }
    assert covered == set(range(65536))
    for new_i in range(16):
        for tx in range(512):
            lanes = [new_i * 4096 + tx * 8 + lane for lane in range(8)]
            assert lanes == list(range(lanes[0], lanes[0] + 8))
            assert lanes[0] % 8 == 0


@pytest.mark.parametrize(
    ("s", "comm_blocks"),
    [(2048, 32), (2048, 64), (4096, 32), (8192, 24), (16384, 16)],
    ids=["s2048-default", "s2048-single-tile", "s4096", "s8192", "s16384"],
)
def test_real_allgather_lowering_uses_one_hoisted_half8_copy_per_chunk(
    tmp_path, s, comm_blocks,
):
    mod = load_example()
    mod.source_only(s, s // 4, s, tmp_path, comm_blocks=comm_blocks, vectorized=True)
    raw = (tmp_path / "raw.tir").read_text()
    source = (tmp_path / "allgather_gemm_scheduled.hip").read_text()
    # S=2048 has one output tile per active compute CTA for default and
    # comm=64 role splits.  It must keep an explicit one-trip marked loop;
    # the backend's no-tile-base exception intentionally accepts no dynamic
    # 0/1 proof.
    assert "tl.dcu_persistent_tile" in raw
    if s == 2048:
        assert "for _once in T.serial(1, annotations={\"tl.dcu_persistent_tile\": 1})" in raw
    assert 'extern "C" __global__ void' in source
    assert "ipc_ag_half8_i" in source
    assert "opaque_thread_id" in source
    assert "tl::ipc::peer<std::byte>(ipc_view_a_shard" in source
    lines = source.splitlines()
    pointer_lines = [
        i for i, line in enumerate(lines)
        if "tl::ipc::peer<std::byte>(ipc_view_a_shard" in line
    ]
    vector_line = next(i for i, line in enumerate(lines) if "ipc_ag_half8_i" in line)
    copy_lines = [line for line in lines if "*(uint4*)" in line and "remote_source" in line]
    # The AG matcher has exactly one remote payload source and one gathered-A
    # destination, unlike AR's four-way reduction.  Looking at the payload
    # line avoids accepting unrelated GEMM vector operations.
    assert len(pointer_lines) == 1
    assert len(copy_lines) == 1
    assert max(pointer_lines) < vector_line
    assert "can_vectorize_" not in source


@pytest.mark.parametrize(
    "case",
    ["multiple", "unknown", "nonzero-min", "dynamic-0-or-1"],
    ids=["constant-base-multiple-trips", "unknown-extent", "nonzero-min",
         "dynamic-bounded-0-or-1-is-not-explicit"],
)
def test_constant_base_persistent_resource_requires_explicit_one_trip_loop(case):
    mod = load_example()
    _, bound = mod.build_bound(2048, 512, 2048, comm_blocks=32)
    kwargs = {
        "multiple": {"loop_extent": tvm.tir.const(2, "int32")},
        "unknown": {"loop_extent": tvm.tir.Var("unknown_extent", "int32")},
        "nonzero-min": {"loop_min": tvm.tir.const(1, "int32")},
        "dynamic-0-or-1": {"dynamic_0_or_1_extent": True},
    }[case]
    invalid = _rewrite_ag_no_tile_base(bound, **kwargs)
    with pytest.raises(tvm.error.TVMError, match="no resource base depends on its loop variable"):
        _lower_ag_tir(invalid)


def test_source_only_scalar_switch_leaves_the_ag_copy_scalar(tmp_path):
    mod = load_example()
    mod.source_only(2048, 512, 2048, tmp_path, vectorized=False)
    source = (tmp_path / "allgather_gemm_scheduled.hip").read_text()
    assert "ipc_ag_half8_i" not in source
    assert "opaque_thread_id" in source


def test_benchmark_jit_forwards_ag_vectorize_config(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    benchmark = load_benchmark()
    captured = []
    vector_source = "\n".join([
        "void* remote_source = tl::ipc::peer<std::byte>(ipc_view_a_shard, 0);",
        "for (int ipc_ag_half8_i = 0; ipc_ag_half8_i < 16; ++ipc_ag_half8_i)",
        "*(uint4*) A_dcu_res = *(uint4*) remote_source;",
    ])

    class Kernel:
        source = vector_source

        def get_kernel_source(self):
            return self.source

    def fake_compile(*args, **kwargs):
        captured.append(kwargs)
        return Kernel()

    import tilelang
    monkeypatch.setattr(tilelang, "disable_cache", lambda: None)
    monkeypatch.setattr(tilelang, "compile", fake_compile)
    monkeypatch.setattr(benchmark, "build_bound", lambda *args, **kwargs: (None, object()))
    monkeypatch.setattr(benchmark, "_check_adapter_init", lambda kernel: None)
    monkeypatch.setattr(benchmark.RsTileLangRun, "_require_full_residency", lambda run, kernel: None)
    for scalar in (False, True):
        if scalar:
            Kernel.source = "for (int scalar_copy = 0; scalar_copy < 128; ++scalar_copy)"
        run = types.SimpleNamespace(
            args=types.SimpleNamespace(m=2048, n=512, k=2048, total_blocks=80,
                                       comm_blocks=32, scalar=scalar, export_hip=None),
            rank=0, residency={}, residencies={},
        )
        assert isinstance(benchmark.Run._compile(run), Kernel)
    assert [call["pass_configs"] for call in captured] == [
        {"tl.enable_dcu_ipc_vectorize": True},
        {"tl.enable_dcu_ipc_vectorize": False},
    ]
