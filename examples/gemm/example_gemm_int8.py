import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

def torch_matmul(A, B):
    ref_c = torch._int_mm(A, B)
    return ref_c

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.int8, accum_dtype=T.int32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_local[i, j]

    return gemm


def main():
    M = 10240
    N = 10240
    K = 10240
    kernel = matmul(M, N, K, 256, 256, 32)

    import torch

    a = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 127, (K, N), device="cuda", dtype=torch.int8)
    # a = torch.randn(M, K).cuda().half()
    # b = torch.randn(K, N).cuda().half()

    c = kernel(a, b)

    ref_c = torch._int_mm(a, b)

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
    profiler = kernel.get_profiler()
    ref_latency = profiler.do_bench(torch_matmul)
    print(f"Reference Latency: {ref_latency}ms")

    torch.testing.assert_close(c, ref_c, rtol=1e-1, atol=1e-1)
    print("All check passed.")

    # Get CUDA Source


    # benchmark



if __name__ == "__main__":
    main()
