"""Independent scheduling and real code-generation checks for GEMM+AllReduce."""
import importlib.util
from pathlib import Path
import types

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/distributed/gemm_ar_prototype/gemm_allreduce_scheduled.py"
BENCHMARK = SCRIPT.with_name("benchmark_gemm_allreduce.py")


def load_example():
    spec = importlib.util.spec_from_file_location("gemm_ar_scheduled_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_benchmark():
    spec = importlib.util.spec_from_file_location("gemm_ar_benchmark_test", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("m,comm", [(2048, 48), (4096, 24), (8192, 16), (16384, 8)])
def test_reference_schedule_has_unique_producers_and_owners(m, comm):
    mod = load_example()
    cfg = mod.schedule_parameters(m, m)
    assert cfg["comm_blocks"] == comm
    compute = 80 - comm
    count = (m // 256) ** 2
    produced = [tile for block in range(compute) for tile in range(block, count, compute)]
    assert sorted(produced) == list(range(count))
    consumed = []
    for rank in range(4):
        owned = [rank + 4 * job for lane in range(comm)
                 for job in range(lane, count // 4, comm)]
        assert all(tile % 4 == rank for tile in owned)
        consumed.extend(owned)
        assert sorted((rank + step) % 4 for step in range(4)) == list(range(4))
    assert sorted(consumed) == list(range(count))
    tile_ready = set(range(count * 4))
    peer_done = set(range(cfg["peer_done_base"], cfg["peer_done_base"] + 4))
    block_done = set(range(cfg["block_done_base"], cfg["block_done_base"] + comm))
    assert not tile_ready & peer_done
    assert not tile_ready & block_done
    assert not peer_done & block_done
    assert max(tile_ready | peer_done | block_done) + 1 == cfg["flag_words"]


def test_half8_mapping_is_coalesced_and_covers_each_tile_exactly_once():
    addresses = [(chunk * 512 + tx) * 8 + v
                 for chunk in range(16) for tx in range(512) for v in range(8)]
    assert addresses == list(range(65536))
    for tx in range(512):
        assert (tx * 8) % 8 == 0
        assert (tx * 8) // 256 == (tx * 8 + 7) // 256


@pytest.mark.parametrize("m", [2048, 4096, 8192, 16384])
def test_real_allreduce_lowering_has_vector_loads_stores_and_ipc(tmp_path, m):
    mod = load_example()
    mod.source_only(m, m, m // 4, tmp_path, total_blocks=80,
                    comm_blocks=None, vectorized=True)
    source = (tmp_path / "gemm_allreduce_scheduled.hip").read_text()
    assert source.count('extern "C" __global__ void') == 1
    assert "opaque_thread_id" in source
    assert "uint4" in source, "the verified IPC pass must emit actual 16-byte accesses"
    # Inspect the payload operations, not an unrelated GEMM uint4.  The pass
    # must perform one uint4 pull for each partial peer, bind the one rounded
    # vector result once, then use that result for exactly four output pushes.
    lines = source.splitlines()
    pull_lines = [line for line in lines if "*(uint4*)" in line and "remote_partial" in line]
    push_lines = [line for line in lines if "*(uint4*)" in line and "remote_out" in line]
    assert len(pull_lines) == 4
    assert len(push_lines) == 4
    for peer in range(4):
        assert sum(f"remote_partial{peer}_remote_ptr_" in line for line in pull_lines) == 1
        assert sum(f"remote_out{peer}_remote_ptr_" in line for line in push_lines) == 1
    assert source.count("uint4 ipc_ar_reduced_f16x8 =") == 1
    assert all("= ipc_ar_reduced_f16x8;" in line for line in push_lines)
    assert "can_vectorize_" not in source
    assert "tl::ipc::" in source
    assert "__syncthreads()" in source
    # We require real IPC ABI parameters, not embedded HIP replacing the DSL.
    assert "ipc_view_partial" in source and "ipc_view_out" in source
    assert "ipc_view_ready" in source


def test_source_only_scalar_switch_disables_only_the_ar_vectorize_pass(tmp_path):
    mod = load_example()
    mod.source_only(2048, 2048, 512, tmp_path, total_blocks=80,
                    comm_blocks=48, vectorized=False)
    source = (tmp_path / "gemm_allreduce_scheduled.hip").read_text()
    assert "ipc_ar_half8_i" not in source
    assert "for (int _tmp_2 = 0; _tmp_2 < 128; ++_tmp_2)" in source
    assert "opaque_thread_id" in source


def test_benchmark_jit_forwards_ar_vectorize_pass_config(monkeypatch):
    """JITKernel owns lowering context, so pass_configs must be explicit."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    benchmark = load_benchmark()
    captured = []
    vector_source = "\n".join([
        "for (int ipc_ar_half8_i = 0; ipc_ar_half8_i < 16; ++ipc_ar_half8_i)",
        *[f"*(uint4*) remote_partial{peer}_remote_ptr_ = x;" for peer in range(4)],
        "uint4 ipc_ar_reduced_f16x8 = x;",
        *[f"*(uint4*) remote_out{peer}_remote_ptr_ = ipc_ar_reduced_f16x8;" for peer in range(4)],
    ])

    class Kernel:
        def __init__(self, source):
            self.source = source

        def get_kernel_source(self):
            return self.source

    def fake_compile(*args, **kwargs):
        captured.append(kwargs)
        enabled = kwargs["pass_configs"]["tl.enable_dcu_ipc_vectorize"]
        return Kernel(vector_source if enabled else "for (int _tmp_2 = 0; _tmp_2 < 128; ++_tmp_2)")

    import tilelang

    monkeypatch.setattr(tilelang, "disable_cache", lambda: None)
    monkeypatch.setattr(tilelang, "compile", fake_compile)
    monkeypatch.setattr(benchmark, "build_bound", lambda *args, **kwargs: (None, object()))
    monkeypatch.setattr(benchmark, "_check_adapter_init", lambda kernel: None)
    monkeypatch.setattr(benchmark.RsTileLangRun, "_require_full_residency", lambda run, kernel: None)
    for scalar in (False, True):
        fake_run = types.SimpleNamespace(
            args=types.SimpleNamespace(m=2048, n=2048, k=512, total_blocks=80,
                                       comm_blocks=48, scalar=scalar, export_hip=None),
            rank=0, residency={}, residencies={},
        )
        assert isinstance(benchmark.Run._compile_tilelang(fake_run), Kernel)
        assert fake_run.residencies["tilelang"] == {}
    assert [call["pass_configs"] for call in captured] == [
        {"tl.enable_dcu_ipc_vectorize": True},
        {"tl.enable_dcu_ipc_vectorize": False},
    ]
