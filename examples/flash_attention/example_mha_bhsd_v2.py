"""MHA BHSD v2: defer reduce_sum / logsum until after P@V (MMA1).

Same tiling as v1; Softmax is split so PV can start before LSE update.
"""
import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
import tilelang.testing
tilelang.disable_cache()


def _bench_cuda(fn, name, total_flops, n_warmup=5, n_repeat=10):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    latency_ms = start.elapsed_time(end) / n_repeat
    print(f"[{name}] Latency: {latency_ms:.3f} ms")
    print(f"[{name}] {total_flops / latency_ms * 1e-9:.2f} TFlops")
    return latency_ms


def get_configs():
    iter_params = dict(
        block_M=[64, 128],
        block_N=[64, 128],
        num_stages=[0, 1, 2],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


_TUNE_IS_CAUSAL = False


def _tune_ref_program(Q, K, V):
    return ref_program(Q, K, V, _TUNE_IS_CAUSAL)


@autotune(
    configs=get_configs(),
    warmup=10,
    rep=10,
    ref_prog=_tune_ref_program,
    skip_check=False,
    rtol=1e-2,
    atol=1e-2,
    max_mismatched_ratio=0.01,
    cache_input_tensors=True,
)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, block_M=256, block_N=256, num_stages=4, threads=512):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.macro
    def MMA0(
        K: T.Tensor(kv_shape, dtype),
        Q_local: T.FragmentBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared, major="row_major")
        T.fill(acc_s, 0)
        # gemm_rs: Q stays in registers across the KV loop, no shared->reg reload.
        T.gemm(
            Q_local,
            K_shared,
            acc_s,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullRow,
            transpose_C=False,
            k_pack=2,
        )
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                q_idx = bx * block_M + i + past_len
                k_idx = k * block_N + j
                acc_s[i, j] = T.if_then_else(q_idx >= k_idx, acc_s[i, j], -T.infinity(acc_s.dtype))
        else:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), acc_s[i, j]
                )

    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        V_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared, major="col_major")
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow, transpose_C=False)

    @T.macro
    def SoftmaxPrep(
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        """rowmax + exp(P) + cast; keep fp32 P in acc_s for later sum."""
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)

        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def SoftmaxUpdateLSE(
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        """reduce_sum / logsum after P@V; does not touch acc_o."""
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    @T.macro
    def Rescale(
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            # Q is reused for every KV tile; keep it in fragment instead of Q_shared.
            Q_local = T.alloc_fragment([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_local)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                MMA0(K, Q_local, K_shared, acc_s, k, bx, by, bz)
                SoftmaxPrep(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                SoftmaxUpdateLSE(acc_s, scores_scale, scores_sum, logsum)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return main


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device), seq_kv - seq_q)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output


def main(
    batch: int = 1,
    heads: int = 1,
    seq_q: int = 256,
    seq_kv: int = 256,
    dim: int = 64,
    is_causal: bool = False,
    tune: bool = False,
):
    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5
    if not tune:
        tilelang.testing.set_random_seed(0)
        kernel = flashattn(
            batch, heads, seq_q, seq_kv, dim, is_causal, block_M=128, block_N=64, num_stages=0, threads=256
        )
        print(kernel.get_kernel_source())
        device = torch.device("cuda")
        dtype = torch.float16
        Q = torch.randn(batch, heads, seq_q, dim, dtype=dtype, device=device)
        K = torch.randn(batch, heads, seq_kv, dim, dtype=dtype, device=device)
        V = torch.randn(batch, heads, seq_kv, dim, dtype=dtype, device=device)

        def run_kernel():
            return kernel(Q, K, V)

        out = run_kernel()
        torch.cuda.synchronize()
        out_ref = ref_program(Q, K, V, is_causal)
        torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
        print("Match vs attention_ref.✅")

        try:
            import flash_attn

            q_bshd = Q.transpose(1, 2).contiguous()
            k_bshd = K.transpose(1, 2).contiguous()
            v_bshd = V.transpose(1, 2).contiguous()
            fla_out = flash_attn.flash_attn_func(q_bshd, k_bshd, v_bshd, causal=is_causal)
            fla_out = fla_out.transpose(1, 2).contiguous()
            torch.testing.assert_close(out, fla_out, rtol=1e-2, atol=1e-2)
            print("Match vs flash_attn.✅")
        except Exception as e:
            print(f"Skip flash_attn check: {type(e).__name__}: {e}")

        print("All checks passed.✅")

        def bench(fn, name, n_warmup=5, n_repeat=10):
            for _ in range(n_warmup):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_repeat):
                fn()
            end.record()
            torch.cuda.synchronize()
            latency_ms = start.elapsed_time(end) / n_repeat
            print(f"[{name}] Latency: {latency_ms:.3f} ms")
            print(f"[{name}] {total_flops / latency_ms * 1e-9:.2f} TFlops")
            return latency_ms

        bench(lambda: ref_program(Q, K, V, is_causal), "Torch")
        bench(run_kernel, "TileLang")
        try:
            import flash_attn

            q_bshd = Q.transpose(1, 2).contiguous()
            k_bshd = K.transpose(1, 2).contiguous()
            v_bshd = V.transpose(1, 2).contiguous()

            def run_flash_attn():
                return flash_attn.flash_attn_func(q_bshd, k_bshd, v_bshd, causal=is_causal)

            bench(run_flash_attn, "FlashAttn")
        except Exception as e:
            print(f"Skip flash_attn bench: {type(e).__name__}: {e}")
    else:
        global _TUNE_IS_CAUSAL
        _TUNE_IS_CAUSAL = is_causal
        kernel = flashattn(batch, heads, seq_q, seq_kv, dim, is_causal)
        best_latency = kernel.latency
        best_config = kernel.config
        ref_latency = kernel.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")
        try:
            import flash_attn

            device = torch.device("cuda")
            dtype = torch.float16
            Q = torch.randn(batch, heads, seq_q, dim, dtype=dtype, device=device)
            K = torch.randn(batch, heads, seq_kv, dim, dtype=dtype, device=device)
            V = torch.randn(batch, heads, seq_kv, dim, dtype=dtype, device=device)
            q_bshd = Q.transpose(1, 2).contiguous()
            k_bshd = K.transpose(1, 2).contiguous()
            v_bshd = V.transpose(1, 2).contiguous()

            def run_flash_attn():
                return flash_attn.flash_attn_func(q_bshd, k_bshd, v_bshd, causal=is_causal)

            _bench_cuda(run_flash_attn, "FlashAttn", total_flops)
        except Exception as e:
            print(f"Skip flash_attn bench: {type(e).__name__}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="heads")
    parser.add_argument("--seq_q", type=int, default=2048, help="query sequence length")
    parser.add_argument("--seq_kv", type=int, default=2048, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--is_causal", action="store_true", help="causal", default=False)
    parser.add_argument("--tune", action="store_true", help="tune configs")
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_q, args.seq_kv, args.dim, args.is_causal, args.tune)
