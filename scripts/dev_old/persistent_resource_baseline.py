"""Capture/compare unmarked GEMM codegen without changing the generators.

Run before and after the persistent-only backend change using the same old
environment. Exact HIP equality is required, not a normalized text comparison.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from check_paths import ROOT, check_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    check_paths()
    import tilelang
    from tilelang import tvm
    spec = importlib.util.spec_from_file_location("baseline_native_gemm", ROOT / "examples/dcu/example_gemm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(ROOT / "examples/distributed/gemm_rs_prototype"))
    import python_schedule_probe as probe
    cases = {}
    for m, n, k in ((512, 768, 256), (1024, 512, 512), (2048, 2048, 512),
                    (4096, 4096, 1024), (19200, 5120, 5120)):
        cases[f"native-m{m}-n{n}-k{k}"] = module.matmul.get_tir(m, n, k, 256, 256, 16)
    for variant in ("flat", "serial-inner-alloc"):
        cases[f"unmarked-{variant}"] = probe.make_probe(variant, 4096, 4096, 1024, 16)
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, func in cases.items():
        with tvm.target.Target("hip -mcpu=gfx936"):
            lowered = tilelang.lower(func, target="hip -mcpu=gfx936",
                                     enable_device_compile=False, enable_host_codegen=False)
        hip = lowered.kernel_source
        (args.output / f"{name}.hip").write_text(hip)
        (args.output / f"{name}.tir").write_text(lowered.device_mod.script())
        record = {"sha256": hashlib.sha256(hip.encode()).hexdigest(), "bytes": len(hip.encode())}
        if args.compare:
            expected = (args.compare / f"{name}.hip").read_text()
            record["exact_hip_equal"] = hip == expected
            if hip != expected:
                raise AssertionError(f"unmarked HIP changed: {name}; artifacts retained in {args.output}")
        records[name] = record
        print(name, record, flush=True)
    report = {"status": "PASS", "cases": records, "baseline": str(args.compare) if args.compare else None}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
