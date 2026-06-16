import tilelang
import tilelang.language as T

tilelang.disable_cache()

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    M = 10240
    N = 10240
    K = 10240
    kernel = matmul(M, N, K, 256, 256, 16)

    import torch

    a = torch.randn(M, K).cuda().half()
    b = torch.randn(N, K).cuda().half()

    c = kernel(a, b)

    ref_c = a @ b.T

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

    torch.testing.assert_close(c, ref_c, rtol=1e-1, atol=1e-1)
    print("All check passed.")

    # Get CUDA Source


    # benchmark



if __name__ == "__main__":
    main()
