"""Single-card persistent GEMM for the explicit DCU tile-resource contract.

This is intentionally a compute-only diagnostic: it has no IPC views, queue,
or communication role.  The marked ``job`` loop is the only opt-in site; its
``A``/``B`` spelling also exercises the existing idxen operand recognition.
"""
import tilelang.language as T


TILE_M = 256
TILE_N = 256
BLOCK_K = 16
THREADS = 512


def persistent_gemm(m: int = 2048, n: int = 2048, k: int = 512, *, blocks: int = 16):
    if m % TILE_M or n % TILE_N or k % (4 * BLOCK_K):
        raise ValueError("m/n must be tile aligned and k a four-stage K multiple")
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    tile_count = (m // TILE_M) * (n // TILE_N)

    @T.prim_func
    def kernel(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        C: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(blocks, threads=THREADS) as block_idx:
            A_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
            B_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
            C_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
            # The resource pass rejects K loops and non-GEMM loops; this outer
            # tile traversal is deliberately the one and only annotation.
            for job in T.serial(
                block_idx, tile_count, blocks,
                annotations={"tl.dcu_persistent_tile": 1},
            ):
                tile_row = job // (n // TILE_N)
                tile_col = job % (n // TILE_N)
                T.clear(C_local)
                for kk in T.Pipelined(
                    k // BLOCK_K,
                    num_stages=4,
                    enable_register_pipeline=True,
                    enable_warp_divergence=True,
                ):
                    T.copy(A[tile_row * TILE_M, kk * BLOCK_K], A_shared,
                           major="row_major", enable_async=True)
                    T.copy(B[kk * BLOCK_K, tile_col * TILE_N], B_shared,
                           major="col_major", enable_async=True)
                    T.gemm(A_shared, B_shared, C_local,
                           k_pack=BLOCK_K // 16, transpose_C=True)
                T.copy(C_local, C[tile_row * TILE_M, tile_col * TILE_N])

    return kernel
