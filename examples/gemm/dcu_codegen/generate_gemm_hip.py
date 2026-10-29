#!/usr/bin/env python3
"""Generate and optionally compile the DCU FP16 GEMM from the TileLang example."""

import argparse
import importlib.util
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXAMPLE = REPO / "examples/gemm/example_gemm_intrinsics_dcu.py"


def load_example():
    spec = importlib.util.spec_from_file_location("tilelang_dcu_gemm", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--arch", default="gfx936")
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    if args.m % 128 or args.n % 128 or args.k % 32:
        parser.error(
            "this specialized kernel requires M/N multiples of 128 and K a multiple of 32")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"gemm_m{args.m}_n{args.n}_k{args.k}_{args.arch}"
    source_path = args.output_dir / f"{stem}.hip"
    library_path = args.output_dir / f"{stem}.so"

    example = load_example()
    kernel = example.tl_matmul(args.m, args.n, args.k, "float16", "float16", "float32")
    source_path.write_text(kernel.get_kernel_source(), encoding="utf-8")
    print(f"generated {source_path}")

    if not args.no_compile:
        subprocess.run([
            "hipcc", "-std=c++17", "-O3", f"--offload-arch={args.arch}",
            "-shared", "-fPIC", f"-I{REPO / 'src'}", str(source_path),
            "-o", str(library_path),
        ], check=True)
        print(f"compiled  {library_path}")


if __name__ == "__main__":
    main()
