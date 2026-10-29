#!/usr/bin/env python3
"""Run a generated TileLang HIP GEMM library using only numpy and HIP runtime."""

import argparse
import ctypes
import os
import time
from pathlib import Path

import numpy as np


def hip_check(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation} failed with HIP error {code}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    dtk_root = Path(os.environ.get("DTKROOT", os.environ.get("ROCM_PATH", "/opt/dtk")))
    hip = ctypes.CDLL(str(dtk_root / "lib/libamdhip64.so"))
    hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hip.hipFree.argtypes = [ctypes.c_void_p]
    hip.hipDeviceSynchronize.restype = ctypes.c_int

    # Exactly representable values make a full-output correctness check cheap.
    a = np.full((args.m, args.k), np.float16(0.03125), dtype=np.float16)
    b = np.full((args.n, args.k), np.float16(0.0625), dtype=np.float16)
    c = np.empty((args.m, args.n), dtype=np.float16)
    device = [ctypes.c_void_p() for _ in range(3)]

    try:
        for pointer, size in zip(device, (a.nbytes, b.nbytes, c.nbytes)):
            hip_check(hip.hipMalloc(ctypes.byref(pointer), size), "hipMalloc")
        for pointer, host in zip(device[:2], (a, b)):
            hip_check(hip.hipMemcpy(pointer, ctypes.c_void_p(host.ctypes.data), host.nbytes, 1),
                      "hipMemcpy H2D")

        library = ctypes.CDLL(str(args.library.resolve()))
        library.call.argtypes = [ctypes.c_void_p] * 4
        library.call.restype = ctypes.c_int

        for _ in range(args.warmup):
            hip_check(library.call(*device, ctypes.c_void_p()), "TileLang call")
        hip_check(hip.hipDeviceSynchronize(), "warmup synchronize")
        start = time.perf_counter()
        for _ in range(args.repeat):
            hip_check(library.call(*device, ctypes.c_void_p()), "TileLang call")
        hip_check(hip.hipDeviceSynchronize(), "benchmark synchronize")
        elapsed_ms = (time.perf_counter() - start) * 1000 / args.repeat

        hip_check(hip.hipMemcpy(ctypes.c_void_p(c.ctypes.data), device[2], c.nbytes, 2),
                  "hipMemcpy D2H")
        expected = np.float16(args.k * float(a[0, 0]) * float(b[0, 0]))
        if not np.all(c == expected):
            raise AssertionError(
                f"incorrect GEMM: expected {expected}, observed [{c.min()}, {c.max()}]")
        tflops = 2 * args.m * args.n * args.k / elapsed_ms / 1e9
        print(f"PASS: all {c.size} outputs equal {expected}")
        print(f"{elapsed_ms:.4f} ms, {tflops:.2f} TFLOP/s")
    finally:
        for pointer in device:
            if pointer.value:
                hip.hipFree(pointer)


if __name__ == "__main__":
    main()
