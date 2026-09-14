# ruff: noqa
"""Ablation 00d: Q roundtrip with T.Parallel G->S (no T.copy), original shared size."""
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
def q_roundtrip(batch_size, UQ, UKV, heads, dim, max_seqlen_q, max_seqlen_k, block_M=128, block_N=128, threads=128):
    q_shape = [1, UQ, heads, dim]
    k_shape = [1, UKV, heads, dim]
    v_shape = [1, UKV, heads, dim]
    o_shape = [1, UQ, heads, dim]
    dtype = T.float16

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(k_shape, dtype),
        V_unpad: T.Tensor(v_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        Output: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            batch_idx = bz
            head_idx = by
            q_start_idx = cu_seqlens_q[batch_idx]

            for i, d in T.Parallel(block_M, dim):
                Q_shared[i, d] = Q_unpad[0, q_start_idx + bx * block_M + i, head_idx, d]
            for i, d in T.Parallel(block_M, dim):
                Output[0, q_start_idx + bx * block_M + i, head_idx, d] = Q_shared[i, d]

            T.fill(K_shared, 0)
            T.fill(V_shared, 0)
            T.fill(O_shared, 0)

    return main


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

    def pad(x_unpad, seq_pad):
        x = x_unpad.unsqueeze(0).contiguous()
        if x.shape[1] == seq_pad:
            return x
        out = x.new_zeros(x.shape[0], seq_pad, x.shape[2], x.shape[3])
        out[:, : x.shape[1]] = x
        return out

    print("[ablate00d Q roundtrip Parallel load]")
    kernel = q_roundtrip(
        batch, UQ_pad, UKV_pad, heads, dim, int(max_seqlen_q), int(max_seqlen_k), block_M=block_M, block_N=block_N
    )
    out = kernel(pad(q_unpad, UQ_pad), pad(k_unpad, UKV_pad), pad(v_unpad, UKV_pad), cu_seqlens_q, cu_seqlens_k)
    torch.cuda.synchronize()
    out = out[0, :UQ]
    ref = q_unpad
    diff = (out.float() - ref.float()).abs()
    print("tile", out[0, 0, :8])
    print("torch", ref[0, 0, :8])
    print(f"max/mean abs {diff.max().item():.6f} {diff.mean().item():.6f}")
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    print("ablate00d Q roundtrip Parallel load PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim)
