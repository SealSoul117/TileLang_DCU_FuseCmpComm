"""Diagnose cp_async on real ablate_01e patterns."""
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()
tilelang.testing.set_random_seed(0)


def plain(tag, disable_safe=False, num_stages=0):
    configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    if disable_safe:
        configs[tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS] = True

    @tilelang.jit(out_idx=[5], pass_configs=configs)
    def qk_gemm(batch_size, UQ, UKV, heads, dim, max_seqlen_q, max_seqlen_k, block_M=128, block_N=128, threads=128):
        q_shape = [1, UQ, heads, dim]
        k_shape = [1, UKV, heads, dim]
        v_shape = [1, UKV, heads, dim]
        s_shape = [1, UQ, heads, UKV]
        dtype = T.float16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            Q_unpad: T.Tensor(q_shape, dtype),
            K_unpad: T.Tensor(k_shape, dtype),
            V_unpad: T.Tensor(v_shape, dtype),
            cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
            cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
            Scores: T.Tensor(s_shape, accum_dtype),
        ):
            with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                batch_idx = bz
                head_idx = by
                q_start_idx = cu_seqlens_q[batch_idx]
                k_start_idx = cu_seqlens_k[batch_idx]
                T.copy(
                    Q_unpad[0, q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, head_idx, :],
                    Q_shared,
                    major="row_major",
                )
                loop_range = T.ceildiv(max_seqlen_k, block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        K_unpad[0, k_start_idx + k * block_N : k_start_idx + (k + 1) * block_N, head_idx, :],
                        K_shared,
                        major="row_major",
                    )
                    T.fill(acc_s, 0)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        transpose_C=True,
                    )
                    for i, j in T.Parallel(block_M, block_N):
                        Scores[0, q_start_idx + bx * block_M + i, head_idx, k_start_idx + k * block_N + j] = acc_s[i, j]
                T.fill(V_shared, 0)
                T.fill(O_shared, 0)

        return main

    kernel = qk_gemm(1, 128, 128, 4, 64, 128, 128)
    src = kernel.get_kernel_source()
    print(f"=== {tag} ===")
    print(
        f"cp_async_gs={src.count('cp_async_gs')} idxen={src.count('cp_async_gs_idxen')} "
        f"vec_s={'vec_s' in src} wait={'cp_async_wait' in src}"
    )
    for key in ["tl::cp_async_gs", "for (int vec_s", "Q_unpad["]:
        i = src.find(key)
        if i >= 0:
            print("  snippet:", repr(src[i : i + 280].replace("\n", " | ")))
            break
    open(f"/tmp/{tag}.hip", "w").write(src)
    return src


def plain_static(tag, disable_safe=False, num_stages=0):
    """Same tile sizes but q/k start fixed to 0 (no cu_seqlens in address)."""
    configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    if disable_safe:
        configs[tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS] = True

    @tilelang.jit(out_idx=[5], pass_configs=configs)
    def qk_gemm(batch_size, UQ, UKV, heads, dim, max_seqlen_q, max_seqlen_k, block_M=128, block_N=128, threads=128):
        q_shape = [1, UQ, heads, dim]
        k_shape = [1, UKV, heads, dim]
        v_shape = [1, UKV, heads, dim]
        s_shape = [1, UQ, heads, UKV]
        dtype = T.float16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            Q_unpad: T.Tensor(q_shape, dtype),
            K_unpad: T.Tensor(k_shape, dtype),
            V_unpad: T.Tensor(v_shape, dtype),
            cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
            cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
            Scores: T.Tensor(s_shape, accum_dtype),
        ):
            with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                head_idx = by
                # Force analyzer-visible constant offsets (still read cu_seqlens to keep signature).
                _ = cu_seqlens_q[0]
                _ = cu_seqlens_k[0]
                T.copy(
                    Q_unpad[0, bx * block_M : (bx + 1) * block_M, head_idx, :],
                    Q_shared,
                    major="row_major",
                )
                loop_range = T.ceildiv(max_seqlen_k, block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        K_unpad[0, k * block_N : (k + 1) * block_N, head_idx, :],
                        K_shared,
                        major="row_major",
                    )
                    T.fill(acc_s, 0)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        transpose_C=True,
                    )
                    for i, j in T.Parallel(block_M, block_N):
                        Scores[0, bx * block_M + i, head_idx, k * block_N + j] = acc_s[i, j]
                T.fill(V_shared, 0)
                T.fill(O_shared, 0)

        return main

    kernel = qk_gemm(1, 128, 128, 4, 64, 128, 128)
    src = kernel.get_kernel_source()
    print(f"=== {tag} ===")
    print(
        f"cp_async_gs={src.count('cp_async_gs')} idxen={src.count('cp_async_gs_idxen')} "
        f"vec_s={'vec_s' in src} wait={'cp_async_wait' in src}"
    )
    for key in ["tl::cp_async_gs", "for (int vec_s", "Q_unpad["]:
        i = src.find(key)
        if i >= 0:
            print("  snippet:", repr(src[i : i + 280].replace("\n", " | ")))
            break
    open(f"/tmp/{tag}.hip", "w").write(src)
    return src


if __name__ == "__main__":
    for fn, tag, kwargs in [
        (plain, "A_dyn_safe_s0", dict(disable_safe=False, num_stages=0)),
        (plain, "B_dyn_nosafe_s0", dict(disable_safe=True, num_stages=0)),
        (plain, "C_dyn_safe_s1", dict(disable_safe=False, num_stages=1)),
        (plain, "D_dyn_nosafe_s1", dict(disable_safe=True, num_stages=1)),
        (plain_static, "E_static_safe_s0", dict(disable_safe=False, num_stages=0)),
        (plain_static, "F_static_safe_s1", dict(disable_safe=False, num_stages=1)),
    ]:
        try:
            fn(tag, **kwargs)
        except Exception as e:
            print(f"=== {tag} === FAIL: {type(e).__name__}: {e}")
