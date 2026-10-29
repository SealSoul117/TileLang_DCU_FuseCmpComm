"""Summarize a four-PE run: median of per-iteration slowest-PE times."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(directory):
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("gemm-ar-*-rank*.json"))]
    if len(records) != 4 or sorted(r["rank"] for r in records) != list(range(4)):
        raise ValueError(f"{directory}: expected exactly ranks 0..3")
    first = records[0]
    for record in records:
        for key in ("shape", "grid", "input_mode", "repeats", "implementation"):
            if record[key] != first[key]:
                raise ValueError(f"inconsistent {key}")
        for phase in ("pre_timing", "post_timing"):
            for variant in first["timing"]:
                if record["verification"][phase][variant]["status"] != "passed":
                    raise ValueError(f"unverified {phase}/{variant}")
    result = {"shape": first["shape"], "grid": first["grid"], "input_mode": first["input_mode"],
              "repeats": first["repeats"], "metric": "median of elementwise maximum over four PEs",
              "timing": {}}
    for variant in first["timing"]:
        all_samples = [r["timing"][variant]["samples_ms"] for r in records]
        if any(len(s) != first["repeats"] for s in all_samples):
            raise ValueError("incomplete samples")
        worst = [max(values) for values in zip(*all_samples)]
        result["timing"][variant] = {"median_ms": statistics.median(worst),
                                      "minimum_ms": min(worst), "maximum_ms": max(worst),
                                      "rankmax_samples_ms": worst}
    if "handwritten" in result["timing"] and "tilelang" in result["timing"]:
        result["tilelang_over_handwritten"] = (
            result["timing"]["tilelang"]["median_ms"] /
            result["timing"]["handwritten"]["median_ms"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps({str(d): summarize(d) for d in args.directories}, indent=2))
