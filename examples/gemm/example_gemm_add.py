import tilelang
import tilelang.language as T

tilelang.disable_cache()

def ref_gemm_add(A, B, bias):
    # 1. half * half -> 触发 rocBLAS 的 Matrix Core 核心（内部使用 FP32 累加）
    # 乘完后立即转为 float32，防止 PyTorch 默认把结果截断回 half，对齐 TileLang 的写回逻辑
    ref_c = A @ B + bias
    return ref_c

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared, major="row_major")
                T.copy(B[k * block_K, bx * block_N], B_shared, major="col_major")
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_local[i, j] + bias[bx * block_N + j]
            # T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    M = 7200
    N = 64
    K = 5120
    kernel = matmul(M, N, K, 256, 256, 16)

    import torch

    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()
    bias = torch.randn(N).cuda().half()

    c = kernel(a, b, bias)

    # ref_c = a @ b + bias

    # print("c:")
    # print(c)
    # print("ref_c:")
    # print(ref_c)
    print("CUDA Source:")
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()
    ref_latency = profiler.do_bench(ref_gemm_add)
    print(f"Reference Latency: {ref_latency}ms")
    print(f"Reference TFLops: {2 * M * N * K / (ref_latency * 1e9)}")
    latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")
    print(f"GEMM TFLops: {2 * M * N * K / (latency * 1e9)}")

    profiler.assert_allclose(ref_gemm_add, rtol=1e-1, atol=1e-1)
    print("All check passed.")

    # Get CUDA Source


    # benchmark



if __name__ == "__main__":
    main()
