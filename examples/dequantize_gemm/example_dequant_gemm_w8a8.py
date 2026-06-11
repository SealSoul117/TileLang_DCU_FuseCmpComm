import tilelang
import tilelang.language as T
from tilelang.autotuner import *
from tvm import tir
import itertools
import torch
import argparse

def get_configs():
    iter_params = dict(
        block_M=[64, 128],
        block_N=[64, 128],
        block_K=[64, 128, 256],
        num_stages=[2, 3],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


def matmul_int8xint8_bf16(M, N, K, tune=False):
    @tilelang.jit(out_idx=[4])
    def kernel_func(block_M, block_N, block_K, num_stages, threads):
        # 1. 形状定义：B 现在是 (K, N) 布局，没有 transpose
        A_shape = (M, K)
        B_shape = (K, N)  
        scale_a_shape = (M,) 
        scale_b_shape = (N,) 
        
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_K, block_N) # 对应 B 的 (K, N) 布局

        assert K % block_K == 0

        @T.prim_func
        def main(
            A: T.Tensor(A_shape, "int8"),
            B: T.Tensor(B_shape, "int8"),
            scale_a: T.Tensor(scale_a_shape, "float32"),
            scale_b: T.Tensor(scale_b_shape, "float32"),
            C: T.Tensor((M, N), "bfloat16"),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                # 分配 Shared Memory (仅用于输入 A 和 B)
                A_shared = T.alloc_shared(A_shared_shape, "int8")
                B_shared = T.alloc_shared(B_shared_shape, "int8")
                
                # 2. 寄存器分配：一个用于 INT32 累加，一个用于最终的 BF16 结果
                # 取消了 C_shared 的分配，直接在寄存器中完成所有后处理
                C_local_int32 = T.alloc_fragment((block_M, block_N), "int32")
                C_local_bf16 = T.alloc_fragment((block_M, block_N), "bfloat16")

                # 仅对输入 Shared Memory 开启 Swizzle 避免 Bank Conflict
                # T.annotate_layout(
                #     {
                #         A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                #         B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                #     }
                # )

                T.clear(C_local_int32)
                
                # K 维度的 Pipelined 循环
                for k in T.Pipelined(K // block_K, num_stages=num_stages):
                    # 3. 加载数据：B 的索引变为 [k * block_K, bx * block_N]
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    
                    # 4. GEMM 计算：移除 transpose_B=True
                    # A_shared: (block_M, block_K), B_shared: (block_K, block_N) -> C_local: (block_M, block_N)
                    T.gemm(A_shared, B_shared, C_local_int32)
                
                # 5. Epilogue (反量化)：直接在寄存器中计算，不经过 Shared Memory
                for i, j in T.Parallel(block_M, block_N):
                    val_fp32 = C_local_int32[i, j].astype("float32")
                    scaled_val = val_fp32 * scale_a[by * block_M + i] * scale_b[bx * block_N + j]
                    C_local_bf16[i, j] = scaled_val.astype("bfloat16")
                    
                # 6. 写回全局内存：直接从寄存器 (Fragment) Copy 到 Global Memory
                T.copy(C_local_bf16, C[by * block_M, bx * block_N])

        return main

    if tune:
        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tilelang.jit(out_idx=[4])
        def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, threads=None):
            return kernel_func(block_M, block_N, block_K, num_stages, threads).prim_func
        return kernel()
    else:
        def kernel(block_M, block_N, block_K, num_stages, threads):
            return kernel_func(block_M, block_N, block_K, num_stages, threads)
        return kernel


def ref_program(A, B, scale_a, scale_b):
    """PyTorch 参考实现，用于验证正确性"""
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)
    # B 现在是 (K, N)，直接 matmul 即可，不需要 .T
    C = torch.matmul(A_f, B_f) 
    
    if scale_a.dim() == 1: scale_a = scale_a.unsqueeze(1)
    if scale_b.dim() == 1: scale_b = scale_b.unsqueeze(0)
        
    C = C * scale_a * scale_b
    return C.to(torch.bfloat16)


def main(m=512, n=512, k=512, tune=False):
    total_flops = 2 * m * n * k
    
    # 生成随机测试数据
    A = torch.randint(-128, 127, (m, k), dtype=torch.int8, device="cuda")
    # 注意：B 的形状现在是 (K, N)，代表未转置的权重矩阵
    B = torch.randint(-128, 127, (k, n), dtype=torch.int8, device="cuda") 
    scale_a = torch.rand(m, dtype=torch.float32, device="cuda") + 0.1
    scale_b = torch.rand(n, dtype=torch.float32, device="cuda") + 0.1

    if not tune:
        kernel = matmul_int8xint8_bf16(m, n, k, tune=tune)(
            block_M=256, block_N=256, block_K=32, num_stages=4, threads=512
        )
        profiler = kernel.get_profiler()
        
        profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
        print("All checks pass.")

        latency = profiler.do_bench(warmup=50)
        print(f"Tilelang Latency: {latency:.4f} ms")
        print(f"Tilelang TFLOPS: {total_flops / latency * 1e-9:.2f}")

    else:
        best_result = matmul_int8xint8_bf16(m, n, k, tune=tune)
        print(f"Best latency: {best_result.latency:.4f} ms")
        print(f"Best config: {best_result.config}")
        print(f"Best TFLOPS: {total_flops / best_result.latency * 1e-9:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2430, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=5120, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=5120, help="Matrix dimension K")
    parser.add_argument("--tune", action="store_true", default=False, help="Enable tuning")
    args = parser.parse_args()

    main(args.m, args.n, args.k, args.tune)