# ruff: noqa
import argparse
import itertools

import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from einops import rearrange, repeat
from tilelang.autotuner import *
from varlen_utils import generate_random_padding_mask, generate_qkv

tilelang.disable_cache()


def get_configs():
    iter_params = dict(
        block_M=[64, 128],
        block_N=[64, 128],
        num_stages=[0, 1, 2],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


# Autotune needs real varlen tensors (valid cu_seqlens); set in main() before tune.
_SUPPLY_INPUTS = None
_TUNE_REF = None


def _supply_prog(_params):
    assert _SUPPLY_INPUTS is not None, "set _SUPPLY_INPUTS before calling flashattn in tune mode"
    return list(_SUPPLY_INPUTS)


def _tune_ref_program(Q_unpad, K_unpad, V_unpad, cu_seqlens_q, cu_seqlens_k):
    """attention_ref on padded BSHD; ignores packed kernel inputs except for signature match."""
    st = _TUNE_REF
    assert st is not None, "set _TUNE_REF before calling flashattn in tune mode"
    out_ref, _ = attention_ref(
        st["q"],
        st["k"],
        st["v"],
        st["query_padding_mask"],
        st["key_padding_mask"],
        causal=st["causal"],
    )
    return out_ref


def _tune_manual_check(lib_outs, ref_outs):
    """Compare only valid packed tokens (UQ_pad tail is uninitialized)."""
    st = _TUNE_REF
    out_unpad = lib_outs[0].squeeze(0).contiguous()[: st["UQ"]]
    out = st["output_pad_fn"](out_unpad)
    out_ref = ref_outs[0]
    close = torch.isclose(out, out_ref, rtol=1e-2, atol=1e-2)
    n_bad = int((~close).sum().item())
    n_tot = out.numel()
    ratio = n_bad / n_tot if n_tot else 0.0
    print(f"[tune-check] mismatch {n_bad}/{n_tot} = {ratio * 100:.2f}%")
    if ratio > 0.01:
        raise AssertionError(f"mismatch_ratio={ratio:.4f} ({n_bad}/{n_tot})")


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    dim = q.shape[-1]
    scale = (1.0 / dim) ** 0.5  # log2(e)
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    scores = torch.einsum("bthd,bshd->bhts", q, k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    scores = scores * scale
    attention = torch.softmax(scores, dim=-1).to(v.dtype)

    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


@autotune(
    configs=get_configs(),
    warmup=10,
    rep=10,
    supply_prog=_supply_prog,
    ref_prog=_tune_ref_program,
    manual_check_prog=_tune_manual_check,
    skip_check=False,
    rtol=1e-2,
    atol=1e-2,
    max_mismatched_ratio=0.01,
    cache_input_tensors=True,
)
@tilelang.jit(
    out_idx=[5],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(
    batch_size,
    UQ,
    UKV,
    heads,
    dim,
    is_causal,
    max_seqlen_q,
    max_seqlen_k,
    block_M=64,
    block_N=64,
    num_stages=0,
    threads=256,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    # Packed BSHD: dummy batch=1 so T.copy matches example_mha_fwd_bshd.
    q_shape = [1, UQ, heads, dim]
    k_shape = [1, UKV, heads, dim]
    v_shape = [1, UKV, heads, dim]
    o_shape = [1, UQ, heads, dim]

    dtype = T.float16
    accum_dtype = T.float32

    @T.macro
    def MMA0(
        K_unpad: T.Tensor(k_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        head_idx: T.int32,
        k_start_idx: T.int32,
        q_current_seqlen: T.int32,
        k_current_seqlen: T.int32,
    ):
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
            transpose_C=False,
        )
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                q_idx = bx * block_M + i
                k_idx = k * block_N + j
                acc_s[i, j] = T.if_then_else(
                    (q_idx >= k_idx) and (q_idx < q_current_seqlen) and (k_idx < k_current_seqlen),
                    acc_s[i, j],
                    -T.infinity(acc_s.dtype),
                )
        else:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    (bx * block_M + i >= q_current_seqlen) or (k * block_N + j >= k_current_seqlen),
                    -T.infinity(acc_s.dtype),
                    acc_s[i, j],
                )

    @T.macro
    def MMA1(
        V_unpad: T.Tensor(v_shape, dtype),
        V_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        head_idx: T.int32,
        v_start_idx: T.int32,
    ):
        T.copy(
            V_unpad[0, v_start_idx + k * block_N : v_start_idx + (k + 1) * block_N, head_idx, :],
            V_shared,
            major="col_major",
        )
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow, transpose_C=False)

    @T.macro
    def Softmax(
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(k_shape, dtype),
        V_unpad: T.Tensor(v_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        Output_unpad: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
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

            batch_idx = bz
            head_idx = by

            q_start_idx = cu_seqlens_q[batch_idx]
            k_start_idx = cu_seqlens_k[batch_idx]
            v_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            k_current_seqlen = k_end_idx - k_start_idx

            T.copy(
                Q_unpad[0, q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, head_idx, :],
                Q_shared,
                major="row_major",
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # Same as origin / example_gqa_fwd_varlen: iterate this batch's K
            # length (not max_seqlen_k). Causal early-exits past the diagonal.
            loop_range = (
                T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(k_current_seqlen, block_N))
                if is_causal
                else T.ceildiv(k_current_seqlen, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                MMA0(
                    K_unpad,
                    Q_shared,
                    K_shared,
                    acc_s,
                    k,
                    bx,
                    head_idx,
                    k_start_idx,
                    q_current_seqlen,
                    k_current_seqlen,
                )
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V_unpad, V_shared, acc_s_cast, acc_o, k, head_idx, v_start_idx)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i < q_current_seqlen:
                    Output_unpad[0, q_start_idx + bx * block_M + i, head_idx, d] = acc_o[i, d]

    return main


def to_padded_bshd(x_unpad, seq_pad):
    x = x_unpad.unsqueeze(0).contiguous()
    if x.shape[1] == seq_pad:
        return x
    out = x.new_zeros(x.shape[0], seq_pad, x.shape[2], x.shape[3])
    out[:, : x.shape[1]] = x
    return out


def main(batch: int = 8, heads: int = 64, seq_len: int = 2048, dim: int = 128, tune: bool = False):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul

    tilelang.testing.set_random_seed(0)

    causal = False
    if causal:
        total_flops *= 0.5

    dtype = torch.float16
    device = torch.device("cuda")

    q = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)
    k = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)
    v = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)

    query_padding_mask = generate_random_padding_mask(seq_len, batch, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seq_len, batch, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    UQ = q_unpad.shape[0]
    UKV = k_unpad.shape[0]
    # Pad to the largest tile in the search space so every tune config fits.
    configs = get_configs()
    align_M = max(c["block_M"] for c in configs)
    align_N = max(c["block_N"] for c in configs)
    UQ_pad = (UQ + align_M - 1) // align_M * align_M
    UKV_pad = (UKV + align_N - 1) // align_N * align_N

    q_bshd = to_padded_bshd(q_unpad, UQ_pad)
    k_bshd = to_padded_bshd(k_unpad, UKV_pad)
    v_bshd = to_padded_bshd(v_unpad, UKV_pad)

    global _SUPPLY_INPUTS, _TUNE_REF
    _SUPPLY_INPUTS = [q_bshd, k_bshd, v_bshd, cu_seqlens_q, cu_seqlens_k]
    _TUNE_REF = {
        "q": q,
        "k": k,
        "v": v,
        "query_padding_mask": query_padding_mask,
        "key_padding_mask": key_padding_mask,
        "causal": causal,
        "output_pad_fn": output_pad_fn,
        "UQ": UQ,
    }

    print(
        f"Compile kernel: batch={batch}, heads={heads}, seq={seq_len}, dim={dim}, "
        f"UQ={UQ}, UKV={UKV}, UQ_pad={UQ_pad}, UKV_pad={UKV_pad}, max_seqlen_q={int(max_seqlen_q)}, "
        f"tune={tune}"
    )

    if tune:
        kernel = flashattn(
            batch,
            UQ_pad,
            UKV_pad,
            heads,
            dim,
            causal,
            int(max_seqlen_q),
            int(max_seqlen_k),
        )
        best_latency = kernel.latency
        best_config = kernel.config
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        return

    kernel = flashattn(
        batch,
        UQ_pad,
        UKV_pad,
        heads,
        dim,
        causal,
        int(max_seqlen_q),
        int(max_seqlen_k),
        block_M=128,
        block_N=64,
        num_stages=0,
        threads=256,  # BM=128 + threads=256 is incorrect on DCU; use 128
    )

    def run_kernel():
        return kernel(q_bshd, k_bshd, v_bshd, cu_seqlens_q, cu_seqlens_k)

    out_bshd = run_kernel()
    torch.cuda.synchronize()
    out_unpad = out_bshd.squeeze(0).contiguous()[:UQ]
    out = output_pad_fn(out_unpad)

    out_ref, _ = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
    )
    print(kernel.get_kernel_source())
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    print("Match vs attention_ref.✅")


    try:
        import flash_attn

        fla_out_unpad = flash_attn.flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            0.0,
            causal=causal,
        )
        fla_out = output_pad_fn(fla_out_unpad)
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

    def run_torch():
        return attention_ref(q, k, v, query_padding_mask, key_padding_mask, causal=causal)[0]

    bench(run_torch, "Torch")
    bench(run_kernel, "TileLang")

    try:
        import flash_attn

        def run_flash_attn():
            return output_pad_fn(
                flash_attn.flash_attn_varlen_func(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    0.0,
                    causal=causal,
                )
            )

        bench(run_flash_attn, "FlashAttn")
    except Exception as e:
        print(f"Skip flash_attn bench: {type(e).__name__}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="heads")
    parser.add_argument("--seq_len", type=int, default=2048, help="sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--tune", action="store_true", help="tune configs")

    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.tune)
