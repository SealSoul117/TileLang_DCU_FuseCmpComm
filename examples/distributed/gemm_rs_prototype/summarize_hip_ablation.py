"""Summarize rank-max HIP events and static ISA metadata (never mix the two)."""
import argparse
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {}
    for shape in (2048, 4096):
        shape_dir = args.root / str(shape)
        results[str(shape)] = {}
        for directory in sorted(shape_dir.iterdir()):
            if not directory.is_dir() or directory.name == "src":
                continue
            records = [json.loads(path.read_text()) for path in sorted(directory.glob("*rank*.json"))]
            if len(records) != 4:
                continue
            assert {r["rank"] for r in records} == {0, 1, 2, 3}
            assert all(r["shape"] == records[0]["shape"] for r in records)
            assert all(r["verification"][phase]["status"] == "passed"
                       for r in records for phase in ("pre_timing", "post_timing"))
            samples = [r["timing"]["samples_ms"] for r in records]
            assert len({len(s) for s in samples}) == 1
            maximum = [max(round_samples) for round_samples in zip(*samples)]
            result = {
                "median_max_pe_ms": statistics.median(maximum),
                "min_max_pe_ms": min(maximum), "max_max_pe_ms": max(maximum),
                "shape": records[0]["shape"], "repeats": len(maximum),
                "input_mode": records[0]["input_mode"],
                "verification": [r["verification"] for r in records],
                "implementation": records[0]["implementation"],
                "source": records[0].get("diagnostic_hip"),
                "performance_interpretation": (
                    "repeat2 retained, not used for attribution: overlapping root-owned GPU workload observed; see background-load.txt"
                    if directory.name.endswith("_repeat2") else
                    "primary run; see experiment README for stage dependencies and single-run limitations"
                ),
            }
            assembly = shape_dir / "src" / f"{directory.name}.s"
            if assembly.exists():
                text = assembly.read_text()
                result["static_isa"] = {
                    key: int(re.search(r"\." + key + r":\s*(\d+)", text)[1])
                    for key in ("sgpr_count", "vgpr_count", "sgpr_spill_count", "vgpr_spill_count", "private_segment_fixed_size")
                }
                result["static_isa"]["mbcnt_instructions"] = len(re.findall(r"^\s+v_mbcnt", text, re.M))
            results[str(shape)][directory.name] = result
            print(f"{shape} {directory.name:24s} {result['median_max_pe_ms']:.6f} ms")
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
