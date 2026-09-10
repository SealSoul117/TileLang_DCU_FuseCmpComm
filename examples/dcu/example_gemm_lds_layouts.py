"""DCU GEMM LDS layout smoke tests: SIMPLE / SWIZZLE / SLAB / AUTO.

Covers:
  - 16x16  → SIMPLE (too small for ds_read_m32x16)
  - 16x32  → A SIMPLE, B SWIZZLE (ds_read ok, slabs < 4)
  - 128x32 → A SLAB, B SWIZZLE (the mixed-tile case)
  - 128x128 → SLAB (slabs >= 4)

Also forces each mode via T.gemm(lds_layout=...). Forced SLAB on a
tile with slabs<4 is expected to degrade to SWIZZLE.
"""

import os
import sys
import traceback

import tilelang
import tilelang.language as T
import torch

from example_gemm_bf16_NT import (
    evaluate_correctness,
    is_correct,
    print_correctness_report,
    print_verdict,
)

tilelang.disable_cache()


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, lds_layout, threads,
           dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, k * block_K], A_shared, major="row_major")
                T.copy(B[k * block_K, bx * block_N], B_shared, major="col_major")
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    k_pack=1,
                    policy=T.GemmWarpPolicy.FullRow,
                    lds_layout=lds_layout,
                )
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


CASES = [
    # name, M, N, K, bM, bN, bK, threads, lds_layout, note
    # 16-row tiles use 1 warp so warp_rows >= 1 (FullRow + 256 threads would collapse M).
    ("simple-16x16-auto", 16, 16, 16, 16, 16, 16, 64, "auto", "AUTO → SIMPLE both"),
    ("simple-16x16-force", 16, 16, 16, 16, 16, 16, 64, "simple", "forced SIMPLE"),
    ("swizzle-16x32-auto", 16, 32, 32, 16, 32, 32, 64, "auto", "AUTO → A SIMPLE, B SWIZZLE"),
    ("swizzle-16x32-simple", 16, 32, 32, 16, 32, 32, 64, "simple", "forced SIMPLE on 16x32"),
    ("swizzle-16x32-force", 16, 32, 32, 16, 32, 32, 64, "swizzle", "forced SWIZZLE"),
    ("mixed-128x32-auto", 128, 32, 128, 128, 32, 128, 256, "auto", "AUTO → A SLAB, B SWIZZLE"),
    ("mixed-128x32-swizzle", 128, 32, 128, 128, 32, 128, 256, "swizzle", "forced SWIZZLE both"),
    ("mixed-128x32-simple", 128, 32, 128, 128, 32, 128, 256, "simple", "forced SIMPLE both"),
    ("mixed-128x32-slab", 128, 32, 128, 128, 32, 128, 256, "slab", "forced SLAB; B degrades to SWIZZLE"),
    ("slab-128x128-auto", 128, 128, 128, 128, 128, 128, 256, "auto", "AUTO → SLAB both"),
    ("slab-128x128-force", 128, 128, 128, 128, 128, 128, 256, "slab", "forced SLAB"),
    ("slab-128x128-swizzle", 128, 128, 128, 128, 128, 128, 256, "swizzle", "forced SWIZZLE on large tiles"),
]


def run_case(name, M, N, K, bM, bN, bK, threads, lds_layout, note, seed=0):
    print("=" * 76)
    print(f"CASE {name}")
    print(f"  shape MNK=({M},{N},{K})  block=({bM},{bN},{bK})  threads={threads}  lds_layout={lds_layout}")
    print(f"  {note}")
    print("=" * 76)
    try:
        kernel = matmul(M, N, K, bM, bN, bK, lds_layout, threads)
    except Exception as e:
        print(f"  COMPILE ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    if os.environ.get("DUMP_SRC"):
        src = kernel.get_kernel_source()
        path = f"/tmp/{name}.hip"
        open(path, "w").write(src)
        print(f"  dumped kernel to {path} ({len(src.splitlines())} lines)")

    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    try:
        c = kernel(a, b)
    except Exception as e:
        print(f"  RUNTIME ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    ref_c = a @ b
    ref_fp32 = a.float() @ b.float()
    max_abs = (c.float() - ref_fp32).abs().max().item()
    print(f"  max_abs vs fp32 = {max_abs:.6f}")

    metrics = evaluate_correctness(c, ref_c, ref_fp32, K=K)
    print_correctness_report(metrics, name=name)
    passed, failures = is_correct(metrics)
    print_verdict(passed, failures)
    return passed


def main():
    selected = sys.argv[1:]
    cases = CASES
    if selected:
        cases = [c for c in CASES if c[0] in selected or any(s in c[0] for s in selected)]
        if not cases:
            print("No matching cases. Available:")
            for c in CASES:
                print(" ", c[0])
            sys.exit(2)

    results = []
    for case in cases:
        ok = run_case(*case)
        results.append((case[0], ok))

    print()
    print("#" * 76)
    print("SUMMARY")
    print("#" * 76)
    n_fail = 0
    for name, ok in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"  [{mark}] {name}")
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
