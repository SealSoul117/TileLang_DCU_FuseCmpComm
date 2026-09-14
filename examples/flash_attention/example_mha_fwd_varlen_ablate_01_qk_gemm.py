# ruff: noqa
"""Ablation 01: varlen path, only QK GEMM (Q @ K^T).

Shared buffers match the original example (no extra score shared):
  Q_shared [block_M, dim], K_shared [block_N, dim],
  V_shared [block_N, dim], O_shared [block_M, dim]
Scores are written from fragment directly to global.
"""
import argparse
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from varlen_utils import generate_random_padding_mask, generate_qkv

tilelang.disable_cache()


@tilelang.jit(
    out_idx=[5],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
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
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            k_current_seqlen = k_end_idx - k_start_idx

            T.copy(
                Q_unpad[0, q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, head_idx, :],
                Q_shared,
                major="row_major",
            )

            loop_range = T.ceildiv(max_seqlen_k, block_N)
            for k in T.Pipelined(loop_range, num_stages=0):
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
                T.copy(
                    acc_s,
                    Scores[
                        0,
                        q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M,
                        head_idx,
                        k_start_idx + k * block_N : k_start_idx + (k + 1) * block_N,
                    ],
                )

    return main


def pad_bshd(x_unpad, seq_pad):
    x = x_unpad.unsqueeze(0).contiguous()
    if x.shape[1] == seq_pad:
        return x
    out = x.new_zeros(x.shape[0], seq_pad, x.shape[2], x.shape[3])
    out[:, : x.shape[1]] = x
    return out


def main(batch=1, heads=4, seq_len=128, dim=64):
    tilelang.testing.set_random_seed(0)
    dtype = torch.float16
    device = torch.device("cuda")
    block_M = block_N = 128

    q = torch.randn(batch, seq_len, heads, dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_len, heads, dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq_len, heads, dim, dtype=dtype, device=device)
    qmask = generate_random_padding_mask(seq_len, batch, device, mode="full")
    kmask = generate_random_padding_mask(seq_len, batch, device, mode="full")
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
        _,
        _,
    ) = generate_qkv(q, k, v, qmask, kmask, kvpacked=False)

    UQ, UKV = q_unpad.shape[0], k_unpad.shape[0]
    UQ_pad = (UQ + block_M - 1) // block_M * block_M
    UKV_pad = (UKV + block_N - 1) // block_N * block_N
    q_bshd = pad_bshd(q_unpad, UQ_pad)
    k_bshd = pad_bshd(k_unpad, UKV_pad)
    v_bshd = pad_bshd(v_unpad, UKV_pad)

    print(f"[ablate01 qk_gemm] UQ={UQ} UKV={UKV} pad=({UQ_pad},{UKV_pad})")
    kernel = qk_gemm(
        batch, UQ_pad, UKV_pad, heads, dim, int(max_seqlen_q), int(max_seqlen_k), block_M=block_M, block_N=block_N
    )
    scores = kernel(q_bshd, k_bshd, v_bshd, cu_seqlens_q, cu_seqlens_k)
    torch.cuda.synchronize()
    scores = scores[0, :UQ, :, :UKV]  # [UQ, heads, UKV]

    ref = torch.einsum("thd,shd->hts", q_unpad.float(), k_unpad.float()).permute(1, 0, 2)  # [UQ, H, UKV]
    # einsum thd,shd->hts gives [H, UQ, UKV]
    ref = torch.einsum("thd,shd->ths", q_unpad.float(), k_unpad.float())
    diff = (scores.float() - ref).abs()
    print("tile", scores[0, 0, :8])
    print("torch", ref[0, 0, :8])
    print(f"max/mean abs {diff.max().item():.6f} {diff.mean().item():.6f}")
    torch.testing.assert_close(scores.float(), ref, rtol=1e-2, atol=1e-2)
    print("ablate01 QK GEMM PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim)
