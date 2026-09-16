import tilelang
import tilelang.language as T

tilelang.disable_cache()

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0): #enable_register_pipeline=True, enable_warp_divergence=True):
                T.copy(A[by * block_M, k * block_K], A_shared, major="row_major")
                T.copy(B[k * block_K, bx * block_N], B_shared, major="col_major")
                T.gemm(A_shared, B_shared, C_local, k_pack=1, transpose_C=True, policy=T.GemmWarpPolicy.FullRow,)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm

import torch
torch.cuda.set_device(1)
def main():
    M = 64
    N = 128
    K = 64
    kernel = matmul(M, N, K, 64, 128, 64)

    import torch

    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    c = kernel(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)
    print("CUDA Source:")
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")
    print(f"GEMM TFLops: {2 * M * N * K / (latency * 1e9)}")

    torch.testing.assert_close(c, ref_c, rtol=1e-3, atol=1e-3)
    print("All check passed.")

    # Get CUDA Source


    # benchmark



if __name__ == "__main__":
    main()
