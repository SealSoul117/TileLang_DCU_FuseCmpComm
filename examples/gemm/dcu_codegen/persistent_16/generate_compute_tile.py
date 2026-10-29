#!/usr/bin/env python3
"""Generate a callable FP32 TileLang split-K GEMM tile for the persistent kernel."""

import argparse
import importlib.util
from pathlib import Path

import tilelang
from tilelang import tvm


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EXAMPLE = REPO / "examples/gemm/example_gemm_intrinsics_dcu.py"


def load_example():
    spec = importlib.util.spec_from_file_location("tilelang_dcu_gemm", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def extract_kernel(source):
    marker = 'extern "C" __global__'
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    end = None
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise RuntimeError("could not locate the end of the generated HIP kernel")
    return source[:start], source[start:end]


def make_helper(source):
    includes, kernel = extract_kernel(source)
    signature_end = kernel.index("{")
    body = kernel[signature_end + 1:-1]
    body = body.replace(
        "  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];\n", "")
    body = body.replace(
        "  const dim3 blockIdx = tl::rasterization2DRow<10>();\n", "")
    body = body.replace("(int)blockIdx.x", "logical_bx")
    body = body.replace("(int)blockIdx.y", "logical_by")
    if "blockIdx." in body:
        raise RuntimeError("unhandled blockIdx reference in generated kernel")

    # The stock kernel scatters a tile into a global row-major MxN output.
    # Communication needs the completed tile to be one contiguous reduce
    # operand, so redirect the single vector store into a 128x128 tile-major
    # staging buffer. The persistent wrapper passes C at the tile's base.
    output_stores = [
        line for line in body.splitlines() if "*(float4*)(C +" in line
    ]
    if len(output_stores) != 1:
        raise RuntimeError(f"expected one generated C store, got {len(output_stores)}")
    old_store = output_stores[0]
    _, rhs = old_store.split(" = ", 1)
    new_lhs = (
        "    *(float4*)(C + (((i_10 * 1024) + "
        "((((int)threadIdx.x) >> 5) * 128)) + "
        "((((int)threadIdx.x) & 31) * 4)))"
    )
    body = body.replace(old_store, new_lhs + " = " + rhs)

    return (
        "#pragma once\n"
        + includes
        + "\n__device__ __forceinline__ void tilelang_compute_tile(\n"
        + "    half_t* __restrict__ A, half_t* __restrict__ B,\n"
        + "    float* __restrict__ C_tile, int logical_bx, int logical_by,\n"
        + "    uchar* buf_dyn_shmem) {\n"
        + "  float* C = C_tile;\n"
        + body
        + "}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--local-k", type=int, default=1024)
    parser.add_argument("--arch", default="gfx936")
    parser.add_argument("--output", type=Path, default=HERE / "generated_compute_tile.h")
    args = parser.parse_args()

    if args.m % 128 or args.n % 128 or args.local_k % 32:
        parser.error("M/N must be multiples of 128 and local K a multiple of 32")

    example = load_example()
    # The outer @tilelang.jit decorator preserves the simplified PrimFunc builder
    # in __wrapped__. Calling it avoids compiling a temporary shared library.
    func = example.tl_matmul.__wrapped__(
        args.m, args.n, args.local_k, "float16", "float32", "float32")
    target = tvm.target.Target(f"hip -mcpu={args.arch}")
    with target:
        artifact = tilelang.lower(func, target=target)
    args.output.write_text(make_helper(artifact.kernel_source), encoding="utf-8")
    print(f"generated {args.output}")


if __name__ == "__main__":
    main()
