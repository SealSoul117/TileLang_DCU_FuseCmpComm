import tilelang
import tilelang.language as T
import torch.nn.functional as F
import torch

tilelang.disable_cache()

def ref_gemm_gelu(A, B):
    # 1. half * half -> 触发 rocBLAS 的 Matrix Core 核心（内部使用 FP32 累加）
    # 乘完后立即转为 float32，防止 PyTorch 默认把结果截断回 half，对齐 TileLang 的写回逻辑
    matmul_result = A @ B
    
    # 2. 在 float32 精度下完成 GELU (Tanh) 计算
    ref_c = F.gelu(matmul_result, approximate='tanh')
    
    # 3. 直接返回 float32 结果
    return ref_c

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
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
                # 1. 从寄存器读取累加结果
                
                # 3. 将结果转换回目标数据类型，并写回全局内存 C
                # 注意全局坐标的映射: by * block_M + i 和 bx * block_N + j
                C[by * block_M + i, bx * block_N + j] = C_local[i, j] * 0.5 * (1.0 + T.tanh(0.79788456 * (C_local[i, j] + 0.044715 * C_local[i, j] * C_local[i, j] * C_local[i, j])))

    return gemm


def main():
    M = 8192
    N = 8192
    K = 8192
    kernel = matmul(M, N, K, 256, 256, 16)

    import torch

    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    c = kernel(a, b)

    matmul_result = a @ b
    ref_c = F.gelu(matmul_result, approximate='tanh')

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)
    print("CUDA Source:")
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()
    ref_latency = profiler.do_bench(ref_gemm_gelu)
    print(f"Reference Latency: {ref_latency}ms")
    print(f"Reference TFLops: {2 * M * N * K / (ref_latency * 1e9)}")
    latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")
    print(f"GEMM TFLops: {2 * M * N * K / (latency * 1e9)}")


    profiler.assert_allclose(ref_gemm_gelu, rtol=1e-1, atol=1e-1)
    print("All check passed.")

    # Get CUDA Source


    # benchmark



if __name__ == "__main__":
    main()
