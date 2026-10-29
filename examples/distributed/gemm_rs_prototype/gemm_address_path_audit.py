"""Name-only address-path audit for the scheduled GEMM + ReduceScatter DSL.

This is an isolated diagnostic, not a scheduling or code-generation change.
It builds the existing ``gemm_reduce_scatter_scheduled`` PrimFunc twice:

* ``original`` retains its DSL-produced lower-case ``a``/``b`` buffer names.
* ``uppercase-ab`` rebuilds the same factory from its inspected Python DSL
  AST after renaming only the nested PrimFunc's two input buffers and all of
  their references to ``A``/``B``.

The second form exists solely to test the historical ``A``/``A_1`` and
``B``/``B_1`` recognition in InjectDCUAsyncCopyIdxen.  It is *not* hand-edited
HIP and does not modify the GEMM schedule, pipeline, IPC binding, or the
``tl::ipc::opaque_thread_id`` communication coordinate.

By default this writes source-only evidence under ``build-old``.  ``--benchmark``
uses the established benchmark runner in this process only: it temporarily
replaces the imported scheduled module's ``build_bound`` with the selected
variant and then invokes ``benchmark_gemm_reduce_scatter.main``.  The normal
runner and all production transforms remain untouched.

For root's raw-lane comparison, set the template root *before starting
Python*, for example::

  TL_TEMPLATE_PATH="$PWD/build-old/address-path-audit/templates-raw" \\
    python examples/distributed/gemm_rs_prototype/gemm_address_path_audit.py \\
      --variant uppercase-ab --source-only --m 2048 --n 2048 --k 512

The script records ``TL_TEMPLATE_PATH``, the selected ``intrin.h`` hash, and
the declared lane mode in each variant's metadata.  It neither changes the
real template tree nor disables the native-lane guard in the normal build.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import linecache
import os
from pathlib import Path
import re
import sys
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BUILD_ROOT = (REPO_ROOT / "build-old").resolve()
DEFAULT_OUTPUT = BUILD_ROOT / "address-path-audit" / "name-only"
TARGET = "hip -mcpu=gfx936"
VARIANTS = ("original", "uppercase-ab")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _under_build_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(BUILD_ROOT)
    except ValueError as error:
        raise ValueError(f"audit output must be under {BUILD_ROOT}, got {resolved}") from error
    return resolved


class _RenameNestedPrimFuncInputs(ast.NodeTransformer):
    """Rename only ``a``/``b`` in scheduled's nested DSL PrimFunc.

    The outer Python factory and all scheduling/control-flow code stay as its
    inspected source.  In particular this never parses or edits generated
    TIR/HIP.  Keeping the transformer scope-aware makes a/b spellings outside
    the decorated kernel ineligible for modification.
    """

    def __init__(self, names: dict[str, str]):
        self.names = names
        self._inside_target = False

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802 (AST API spelling)
        is_target = node.name == "fused_gemm_rs_scheduled"
        previous = self._inside_target
        if is_target:
            self._inside_target = True
        node = self.generic_visit(node)
        self._inside_target = previous
        return node

    def visit_arg(self, node: ast.arg):  # noqa: N802 (AST API spelling)
        if self._inside_target and node.arg in self.names:
            node.arg = self.names[node.arg]
        return node

    def visit_Name(self, node: ast.Name):  # noqa: N802 (AST API spelling)
        if self._inside_target and node.id in self.names:
            node.id = self.names[node.id]
        return node


def _ast_fingerprint(tree: ast.AST) -> str:
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _scheduled_factory_variant(variant: str):
    """Return a Python DSL factory and its auditable source provenance.

    ``T.prim_func`` uses Python source inspection.  The transformed AST is
    therefore compiled under a unique linecache filename before it is called;
    the decorator still sees real Python DSL source rather than manually
    authored TIR.  The function globals are copied from the original module,
    retaining its imported DSL names and all module constants.
    """
    import gemm_reduce_scatter_scheduled as scheduled

    factory = scheduled.gemm_reduce_scatter_scheduled
    source = inspect.cleandoc(inspect.getsource(factory)) + "\n"
    parsed = ast.parse(source, filename=inspect.getsourcefile(factory) or "<scheduled-dsl>")
    if variant == "original":
        return factory, source, True
    if variant != "uppercase-ab":
        raise ValueError(f"unknown variant {variant!r}")
    renamed = copy.deepcopy(parsed)
    renamed = _RenameNestedPrimFuncInputs({"a": "A", "b": "B"}).visit(renamed)
    renamed = ast.fix_missing_locations(renamed)
    restored = copy.deepcopy(renamed)
    restored = _RenameNestedPrimFuncInputs({"A": "a", "B": "b"}).visit(restored)
    restored = ast.fix_missing_locations(restored)
    if _ast_fingerprint(parsed) != _ast_fingerprint(restored):
        raise RuntimeError("name-only inverse AST transform changed the scheduled DSL")

    generated_source = ast.unparse(renamed) + "\n"
    source_hash = hashlib.sha256(generated_source.encode("utf-8")).hexdigest()[:16]
    filename = f"<gemm-address-path-audit-uppercase-ab-{source_hash}.py>"
    lines = generated_source.splitlines(keepends=True)
    linecache.cache[filename] = (len(generated_source), None, lines, filename)
    namespace = dict(factory.__globals__)
    namespace["__name__"] = factory.__module__
    # The audit module itself enables postponed annotations, while the
    # scheduled DSL module does not.  Do not inherit that unrelated future
    # flag: nested T.Tensor annotations must be evaluated inside the factory,
    # exactly as they are in the original DSL source.
    code = compile(generated_source, filename, "exec", dont_inherit=True)
    exec(code, namespace, namespace)  # pylint: disable=exec-used
    transformed = namespace.get(factory.__name__)
    if not callable(transformed):
        raise RuntimeError("AST diagnostic did not define the scheduled DSL factory")
    if factory.__closure__ is not None:
        raise RuntimeError("scheduled DSL factory unexpectedly has a closure; refuse to drop it")
    return transformed, generated_source, True


def build_variant(m: int, n: int, k: int, *, total_blocks: int, comm_blocks: int,
                  variant: str):
    """Build the scheduled DSL from the selected Python-source factory."""
    from tilelang.distributed.ipc_binding import bind_ipc

    factory, _, _ = _scheduled_factory_variant(variant)
    raw = factory(
        m, n, k, total_blocks=total_blocks, comm_blocks=comm_blocks,
    )
    bound = bind_ipc(raw, {"partial": "ipc_view_partial", "ready": "ipc_view_ready"})
    return raw, bound


def _lane_metadata(lane_mode: str) -> dict[str, object]:
    template_root_text = os.environ.get("TL_TEMPLATE_PATH")
    template_root = Path(template_root_text).resolve() if template_root_text else None
    helper = template_root / "tl_templates" / "dcu_hip" / "intrin.h" if template_root else None
    if lane_mode == "raw-template" and helper is None:
        raise ValueError("--lane-mode raw-template requires TL_TEMPLATE_PATH before Python starts")
    if lane_mode == "raw-template" and not helper.is_file():
        raise FileNotFoundError(f"raw-template intrin helper is missing: {helper}")
    return {
        "lane_mode": lane_mode,
        "tl_template_path": str(template_root) if template_root else None,
        "intrin_h": str(helper) if helper and helper.is_file() else None,
        "intrin_h_sha256": _sha256(helper) if helper and helper.is_file() else None,
        # These are deliberately only provenance fields.  The script passes no
        # compiler override and makes no production template/library mutation.
        "compile_environment": {
            name: os.environ.get(name)
            for name in ("TL_TEMPLATE_PATH", "CXXFLAGS", "HIPCC_FLAGS", "TILELANG_DEV_BUILD_ROOT")
            if os.environ.get(name) is not None
        },
    }


def _count_paths(text: str) -> dict[str, int]:
    return {
        "dcu_async_copy": text.count("dcu_async_copy("),
        "dcu_async_copy_idxen": text.count("dcu_async_copy_idxen("),
        "make_dcu_resource": text.count("make_dcu_resource("),
        "opaque_thread_id": text.count("opaque_thread_id"),
    }


def _rank_tag() -> str:
    """Keep multi-rank audit artifacts disjoint without initializing MPI here."""
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            if not value.isdigit():
                raise ValueError(f"invalid {name} for audit artifact path: {value!r}")
            return f"rank{value}"
    return "rank-local"


def export_source_evidence(m: int, n: int, k: int, *, total_blocks: int, comm_blocks: int,
                           variant: str, output: Path, lane_mode: str) -> Path:
    """Lower without device compilation and persist raw/bound/device/HIP evidence."""
    import tilelang
    import tvm

    output = _under_build_root(output) / variant / _rank_tag()
    output.mkdir(parents=True, exist_ok=True)
    factory, dsl_source, ast_roundtrip_verified = _scheduled_factory_variant(variant)
    raw, bound = build_variant(
        m, n, k, total_blocks=total_blocks, comm_blocks=comm_blocks, variant=variant,
    )
    artifact = tilelang.lower(
        bound, target=tvm.target.Target(TARGET), enable_device_compile=False, enable_host_codegen=False,
    )
    raw_text = raw.script()
    bound_text = bound.script()
    device_text = artifact.device_mod.script()
    hip_text = artifact.kernel_source
    files = {
        "raw.tir": raw_text,
        "bound.tir": bound_text,
        "device.tir": device_text,
        "gemm_reduce_scatter_scheduled.hip": hip_text,
        "scheduled_dsl.py": dsl_source,
    }
    for name, text in files.items():
        (output / name).write_text(text, encoding="utf-8")

    counts = _count_paths(device_text)
    # This expectation is the causal assertion the diagnostic exists to test;
    # it fails rather than silently relabeling an unchanged address path.
    if variant == "original" and counts["dcu_async_copy_idxen"]:
        raise RuntimeError("original lower-case variant unexpectedly used idxen")
    if variant == "uppercase-ab" and not counts["dcu_async_copy_idxen"]:
        raise RuntimeError("uppercase name-only variant did not restore idxen")
    if counts["opaque_thread_id"] != 1:
        raise RuntimeError("name-only variant changed the opaque communication coordinate")

    document = {
        "schema": "gemm-address-path-audit-v1",
        "purpose": "name-only A/B address-generation causality diagnostic",
        "variant": variant,
        "shape": {"m": m, "n": n, "k": k},
        "schedule": {"total_blocks": total_blocks, "comm_blocks": comm_blocks},
        "generator": "scheduled Python DSL PrimFunc; uppercase variant is a scoped Python-AST name-only transform",
        "dsl_factory": f"{factory.__module__}.{factory.__name__}",
        "dsl_ast_inverse_roundtrip_verified": ast_roundtrip_verified,
        "algorithm_changed": False,
        "ipc_opaque_thread_coordinate_preserved": True,
        "device_path_counts": counts,
        "lane_provenance": _lane_metadata(lane_mode),
        "files": {name: {"sha256": _sha256(output / name)} for name in files},
    }
    (output / "metadata.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"address-path audit source: {output}", flush=True)
    return output


def _arg_value(arguments: Iterable[str], option: str, default: int) -> int:
    arguments = list(arguments)
    for index, item in enumerate(arguments):
        if item == option:
            if index + 1 == len(arguments):
                raise ValueError(f"{option} needs a value")
            return int(arguments[index + 1])
        prefix = option + "="
        if item.startswith(prefix):
            return int(item[len(prefix):])
    return default


def _has_option(arguments: Iterable[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def _string_arg(arguments: Iterable[str], option: str, default: str) -> str:
    arguments = list(arguments)
    for index, item in enumerate(arguments):
        if item == option:
            if index + 1 == len(arguments):
                raise ValueError(f"{option} needs a value")
            return arguments[index + 1]
        prefix = option + "="
        if item.startswith(prefix):
            return item[len(prefix):]
    return default


def run_benchmark(benchmark_args: list[str], *, variant: str, source_output: Path) -> None:
    """Invoke the existing runner with a process-local selected build_bound."""
    import benchmark_gemm_reduce_scatter as benchmark
    import gemm_reduce_scatter_scheduled as scheduled

    if _arg_value(benchmark_args, "--total-blocks", 80) != 80:
        raise ValueError("name-only audit is restricted to the existing 80-CTA matched schedule")
    if _arg_value(benchmark_args, "--comm-blocks", 32) != 32:
        raise ValueError("name-only audit is restricted to the existing 48/32 CTA split")
    if _string_arg(benchmark_args, "--schedule", "matched") != "matched":
        raise ValueError("name-only audit may only use --schedule matched")
    if _string_arg(benchmark_args, "--implementation", "tilelang") != "tilelang":
        raise ValueError("name-only audit may only use --implementation tilelang")
    if _has_option(benchmark_args, "--diagnostic-hip"):
        raise ValueError("name-only audit rejects --diagnostic-hip; it must compile the DSL-generated variant")
    if not _has_option(benchmark_args, "--schedule"):
        benchmark_args.extend(("--schedule", "matched"))
    if not _has_option(benchmark_args, "--implementation"):
        benchmark_args.extend(("--implementation", "tilelang"))
    if not _has_option(benchmark_args, "--export-hip"):
        benchmark_args.extend(("--export-hip", str(source_output / "benchmark-hip")))
    if not _has_option(benchmark_args, "--json-prefix"):
        benchmark_args.extend(("--json-prefix", f"address-path-{variant}"))

    original_build_bound = scheduled.build_bound

    def audit_build_bound(m: int, n: int, k: int, *, total_blocks: int = 80, comm_blocks: int = 32):
        return build_variant(
            m, n, k, total_blocks=total_blocks, comm_blocks=comm_blocks, variant=variant,
        )

    scheduled.build_bound = audit_build_bound
    old_argv = sys.argv
    try:
        sys.argv = [str(HERE / "benchmark_gemm_reduce_scatter.py"), *benchmark_args]
        benchmark.main()
    finally:
        sys.argv = old_argv
        scheduled.build_bound = original_build_bound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="original")
    parser.add_argument("--source-only", action="store_true",
                        help="export raw/bound/device/HIP evidence without a GPU compilation")
    parser.add_argument("--benchmark", action="store_true",
                        help="run the established matched benchmark through a process-local build_bound patch")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lane-mode", choices=("protected", "raw-template"), default="protected",
                        help="provenance label; raw-template requires pre-set TL_TEMPLATE_PATH")
    args, benchmark_args = parser.parse_known_args()
    if not args.source_only and not args.benchmark:
        args.source_only = True
    if args.benchmark and not benchmark_args:
        raise ValueError("--benchmark requires explicit benchmark arguments (for example --m/--n/--k)")

    # Shape options are intentionally parsed from the runner-compatible tail,
    # so the same --m/--n/--k spelling works for evidence and benchmark modes.
    m = _arg_value(benchmark_args, "--m", 2048)
    n = _arg_value(benchmark_args, "--n", 2048)
    k = _arg_value(benchmark_args, "--k", 512)
    total_blocks = _arg_value(benchmark_args, "--total-blocks", 80)
    comm_blocks = _arg_value(benchmark_args, "--comm-blocks", 32)
    output = _under_build_root(args.output)
    source_output = export_source_evidence(
        m, n, k, total_blocks=total_blocks, comm_blocks=comm_blocks,
        variant=args.variant, output=output, lane_mode=args.lane_mode,
    )
    if args.benchmark:
        run_benchmark(benchmark_args, variant=args.variant, source_output=source_output)


if __name__ == "__main__":
    main()
