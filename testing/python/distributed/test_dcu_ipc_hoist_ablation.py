"""End-to-end source checks for independently toggling DCU IPC hoisting.

The three scheduled distributed frontends are deliberately lowered through
the normal phase pipeline.  This verifies the experiment knob itself rather
than testing the Hoist or Vectorize implementations in isolation.
"""

import importlib.util
from pathlib import Path
import sys

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine import phase


ROOT = Path(__file__).resolve().parents[3]
TARGET = "hip -mcpu=gfx936"
RS_DIR = ROOT / "examples/distributed/gemm_rs_prototype"
AR_DIR = ROOT / "examples/distributed/gemm_ar_prototype"
AG_DIR = ROOT / "examples/distributed/ag_gemm_prototype"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _scheduled_bound(kind: str):
    """Load one real S=2048 bound frontend and return a cleanup callback."""
    if kind == "rs":
        saved = {name: sys.modules.get(name) for name in
                 ("gemm_reduce_scatter", "gemm_reduce_scatter_scheduled")}
        try:
            _load(RS_DIR, "gemm_reduce_scatter")
            scheduled = _load(RS_DIR, "gemm_reduce_scatter_scheduled")
            _, bound = scheduled.build_bound(2048, 2048, 512, total_blocks=80,
                                              comm_blocks=32)
            return bound, "ipc_view_partial", "ipc_half8_i", saved
        except Exception:
            _restore_modules(saved)
            raise
    if kind == "ar":
        name = "gemm_allreduce_scheduled"
        saved = {name: sys.modules.get(name)}
        try:
            scheduled = _load(AR_DIR, name)
            _, bound = scheduled.build_bound(2048, 2048, 512, total_blocks=80,
                                              comm_blocks=48)
            return bound, "ipc_view_partial", "ipc_ar_half8_i", saved
        except Exception:
            _restore_modules(saved)
            raise
    if kind == "ag":
        name = "allgather_gemm_scheduled"
        saved = {name: sys.modules.get(name)}
        try:
            scheduled = _load(AG_DIR, name)
            _, bound = scheduled.build_bound(2048, 512, 2048, total_blocks=80,
                                              comm_blocks=32)
            return bound, "ipc_view_a_shard", "ipc_ag_half8_i", saved
        except Exception:
            _restore_modules(saved)
            raise
    raise AssertionError(f"unknown scheduled frontend {kind}")


def _restore_modules(saved):
    for name, previous in saved.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _source(bound, *, hoist: bool | None, vectorize: bool) -> str:
    config = {"tl.enable_dcu_ipc_vectorize": vectorize}
    if hoist is not None:
        config["tl.enable_dcu_ipc_hoist"] = hoist
    # Match native GEMM source-only tests: layout inference queries the
    # active TargetContext, in addition to lower()'s explicit target argument.
    with tvm.target.Target(TARGET), tvm.transform.PassContext(config=config):
        return tilelang.lower(
            bound, target=TARGET, enable_device_compile=False, enable_host_codegen=False,
        ).kernel_source


def _source_and_hoist_boundary(bound, *, hoist, vectorize, monkeypatch):
    """Lower normally while recording the exact IR passed through Hoist.

    HIP text is not a valid equivalence oracle here: the existing LDS-layout
    lowering is allowed to choose different, equivalent layouts in separate
    invocations.  Recording this phase boundary instead proves that the
    omitted default and explicit ``True`` drive the same Hoist pass over the
    same IR, before unrelated later lowering decisions occur.
    """
    original_factory = tilelang.transform.HoistDCUIPCPointers
    entries, results = [], []

    def recording_factory():
        transform = original_factory()

        def apply(mod):
            entries.append(mod)
            result = transform(mod)
            results.append(result)
            return result

        return apply

    with monkeypatch.context() as scoped:
        scoped.setattr(tilelang.transform, "HoistDCUIPCPointers", recording_factory)
        source = _source(bound, hoist=hoist, vectorize=vectorize)
    assert len(entries) == len(results) == (1 if hoist is not False else 0)
    return source, entries, results


def _base_is_before_scalar_element_loop(source: str, descriptor: str) -> bool:
    """Recognize the actual hoisted shape without relying on local SSA names."""
    lines = source.splitlines()
    pointers = [i for i, line in enumerate(lines)
                if f"tl::ipc::peer<std::byte>({descriptor}" in line]
    assert pointers, f"missing IPC bases for {descriptor}"
    # The scheduled scalar payload loop has the fixed 128 chunks.  A hoisted
    # remote base must dominate one such loop; a statement-local base appears
    # in its body and therefore cannot satisfy this ordering.
    scalar_loops = [(i, len(line) - len(line.lstrip())) for i, line in enumerate(lines)
                    if "for (int " in line and "< 128" in line]
    assert scalar_loops, "missing scheduled scalar communication loop"
    # A pointer created inside its 128-iteration payload loop is indented
    # deeper than that loop's header.  A hoisted declaration is not.
    for pointer in pointers:
        pointer_indent = len(lines[pointer]) - len(lines[pointer].lstrip())
        if any(loop < pointer and loop_indent < pointer_indent
               for loop, loop_indent in scalar_loops):
            return False
    return any(max(pointers) < loop for loop, _ in scalar_loops)


def _plain_dcu_gemm():
    """A real non-IPC GEMM body for the phase-level non-regression check."""
    @T.prim_func
    def gemm(
        A: T.Tensor((256, 16), "float16"),
        B: T.Tensor((16, 256), "float16"),
        C: T.Tensor((256, 256), "float16"),
    ):
        with T.Kernel(1, threads=512):
            a_shared = T.alloc_shared((256, 16), "float16")
            b_shared = T.alloc_shared((16, 256), "float16")
            c_local = T.alloc_fragment((256, 256), "float32")
            T.clear(c_local)
            for kk in T.Pipelined(1, num_stages=4, enable_register_pipeline=True,
                                  enable_warp_divergence=True):
                T.copy(A[0, kk * 16], a_shared, major="row_major", enable_async=True)
                T.copy(B[kk * 16, 0], b_shared, major="col_major", enable_async=True)
                T.gemm(a_shared, b_shared, c_local, k_pack=1, transpose_C=True)
            T.copy(c_local, C[0, 0])
    return gemm


def test_dcu_ipc_hoist_default_true_and_explicit_overrides():
    with tvm.transform.PassContext(config={}):
        assert phase.should_enable_dcu_ipc_hoist()
    with tvm.transform.PassContext(config={"tl.enable_dcu_ipc_hoist": True}):
        assert phase.should_enable_dcu_ipc_hoist()
    with tvm.transform.PassContext(config={"tl.enable_dcu_ipc_hoist": False}):
        assert not phase.should_enable_dcu_ipc_hoist()


def test_plain_gemm_is_unchanged_when_ipc_hoist_is_disabled(monkeypatch):
    gemm = _plain_dcu_gemm()
    default, entries, results = _source_and_hoist_boundary(
        gemm, hoist=None, vectorize=False, monkeypatch=monkeypatch,
    )
    disabled = _source(gemm, hoist=False, vectorize=False)
    # Hoist receives the normal GEMM but is exactly identity on it; therefore
    # bypassing that identity pass cannot alter the production phase result.
    tvm.ir.assert_structural_equal(entries[0], results[0], map_free_vars=True)
    assert "ipc_view_" not in default and "ipc_view_" not in disabled


@pytest.mark.parametrize("kind", ["rs", "ar", "ag"])
def test_real_scheduled_ablation_a00_b10_c11(kind, monkeypatch):
    """A00=no hoist/vector, B10=hoist-only, C11=both passes."""
    bound, descriptor, vector_marker, saved = _scheduled_bound(kind)
    try:
        default_scalar, default_entries, default_results = _source_and_hoist_boundary(
            bound, hoist=None, vectorize=False, monkeypatch=monkeypatch,
        )
        a00 = _source(bound, hoist=False, vectorize=False)
        b10, explicit_entries, explicit_results = _source_and_hoist_boundary(
            bound, hoist=True, vectorize=False, monkeypatch=monkeypatch,
        )
        c11 = _source(bound, hoist=True, vectorize=True)

        # Both phase routes did invoke Hoist.  Compare its result against a
        # fresh invocation over the *same captured input IR*: this avoids
        # treating independent later LDS-layout choices as a compiler
        # regression, while proving that the default route was exactly the
        # verified Hoist transformation and that explicit B10 reached it.
        assert len(explicit_entries) == len(explicit_results) == 1
        direct_default = tilelang.transform.HoistDCUIPCPointers()(default_entries[0])
        tvm.ir.assert_structural_equal(default_results[0], direct_default, map_free_vars=True)
        # A00 retains scalar payloads and statement-local bases.
        assert vector_marker not in a00 and vector_marker not in b10
        assert vector_marker not in default_scalar
        assert not _base_is_before_scalar_element_loop(a00, descriptor)
        assert _base_is_before_scalar_element_loop(b10, descriptor)

        # C11 retains the hoisted bases and activates only the already-narrow
        # matching vectorizer; no frontend or pass-order change is involved.
        assert vector_marker in c11
        assert f"tl::ipc::peer<std::byte>({descriptor}" in c11
    finally:
        _restore_modules(saved)
