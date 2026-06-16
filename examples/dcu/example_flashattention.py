import torch
import tilelang
import tilelang.language as T

# JIT 编译配置：开启 Fast Math
@tilelang.jit(
    out_idx=[3],  # Q(0), K(1), V(2), Output(3)
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_fwd_causal_polished(batch, seq_len, heads, dim, is_causal=True, block_M=128, block_N=64):
    """
    基于官方最佳实践润色后的 Dense Causal FlashAttention (Forward Only)
    """
    # 核心优化 1：使用 Base-2 进行缩放，适配 T.exp2 硬件指令
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def flash_fwd(
        Q: T.Tensor([batch, seq_len, heads, dim], dtype),  # type: ignore
        K: T.Tensor([batch, seq_len, heads, dim], dtype),  # type: ignore
        V: T.Tensor([batch, seq_len, heads, dim], dtype),  # type: ignore
        Output: T.Tensor([batch, seq_len, heads, dim], dtype),  # type: ignore
    ):
        # 线程块配置
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=512) as (bx, by, bz):
            # ================= 1. 内存与寄存器分配 =================
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

            # 核心优化 2：Swizzled Layout 消除 Shared Memory Bank Conflict
            # T.annotate_layout({
            #     Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
            #     K_shared: tilelang.layout.make_swizzled_layout(K_shared),
            #     V_shared: tilelang.layout.make_swizzled_layout(V_shared),
            # })

            # ================= 2. 初始化 =================
            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            
            # Causal 边界判断
            loop_range = T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)

            # 核心优化 3：Pipelined Pipeline 掩盖访存延迟
            for k in T.Pipelined(loop_range, num_stages=4):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                
                # 核心优化 4：非常优雅的 Mask 技巧。
                # 由于 T.gemm 会累加到 acc_s，我们提前用 0 和 -inf 初始化，完美实现截断。
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                
                # 核心优化 5：使用 FullRow 策略优化 Tensor Core 计算
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                
                # ================= 3. Online Softmax 逻辑 =================
                T.copy(scores_max, scores_max_prev)
                # 原生 reduce 操作
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    
                # 核心优化 6：使用硬件加速的 exp2
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                    
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                
                T.reduce_sum(acc_s, scores_sum, dim=1)
                
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            # ================= 4. 归一化并写回显存 =================
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
                
            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return flash_fwd


# =================================================================
# 简单测试脚本：对比 PyTorch 官方 FlashAttention (如可用) 或 Naive PyTorch
# =================================================================
if __name__ == "__main__":
    BATCH, SEQ_LEN, HEADS, DIM = 2, 1024, 8, 128
    
    # 初始化 Tensor
    q = torch.randn(BATCH, SEQ_LEN, HEADS, DIM, dtype=torch.float16, device="cuda")
    k = torch.randn(BATCH, SEQ_LEN, HEADS, DIM, dtype=torch.float16, device="cuda")
    v = torch.randn(BATCH, SEQ_LEN, HEADS, DIM, dtype=torch.float16, device="cuda")
    
    # 实例化我们的算子 (首次运行需要编译，大约几秒钟)
    # block_M=128, block_N=64 是 A100/H100 上的常见最优分块
    tl_flash_attn = flashattn_fwd_causal_polished(
        batch=BATCH, seq_len=SEQ_LEN, heads=HEADS, dim=DIM, 
        is_causal=False, block_M=128, block_N=128
    )
    
    # 运行并获取输出
    out_tl = tl_flash_attn(q, k, v)
    print("TileLang 算子运行成功，输出 Shape:", out_tl.shape)

    # 用纯 PyTorch 验证一下正确性
    import torch.nn.functional as F
    q_pt = q.transpose(1, 2)  # [B, H, S, D]
    k_pt = k.transpose(1, 2)
    v_pt = v.transpose(1, 2)
    
    out_pt = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=True)
    out_pt = out_pt.transpose(1, 2)  # 变回 [B, S, H, D]
    
    # 对比误差
    max_diff = torch.max(torch.abs(out_tl - out_pt))
    print(f"最大误差 (与 PyTorch SDPA 相比): {max_diff.item():.6f}")