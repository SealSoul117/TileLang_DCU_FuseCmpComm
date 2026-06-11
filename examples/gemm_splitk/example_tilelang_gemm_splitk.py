import tilelang
import tilelang.language as T
tilelang.disable_cache()

@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, split_k, dtype=T.float16, accum_dtype=T.float32, out_dtype=T.float32):
    splitK = K // split_k

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=512) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=4):
                T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(C[by * block_M + i, bx * block_N + j], C_local[i, j])

    return main


def main():
    M = 4096
    N = 4096
    K = 4096
    block_M = 256
    block_N = 256
    block_K = 16
    split_k = 2

    kernel = matmul(M, N, K, block_M, block_N, block_K, split_k)

    import torch

    torch.random.manual_seed(42)
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()
    c = torch.zeros(M, N).cuda().float()
    kernel(a, b, c)
    print("Done")

    ref_c = a @ b

    torch.testing.assert_close(c, ref_c.to(c.dtype), rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    main()
