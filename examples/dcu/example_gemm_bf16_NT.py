import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()


def evaluate_correctness(src_bf16, ref_bf16, ref_fp32, K,
                         eps_dtype=2 ** -8, atol_scale=3.0, rtol_const=0.05,
                         top_k=100, grid_size=16):
    """评估 GEMM kernel 的正确性。

    Args:
        src_bf16: 待测 kernel 输出 (2D tensor)
        ref_bf16: torch 同精度参考输出 (2D tensor)
        ref_fp32: torch fp32 金标准 (2D tensor)
        K: GEMM 的 K 维度 (用于计算 atol 理论界)
        eps_dtype: 数据类型机器精度 (bf16=2^-8, fp16=2^-10)
        atol_scale: 绝对误差地板 scale 系数 (atol = sqrt(K) * eps * scale)
        rtol_const: 相对误差天花板 (默认 0.05 = bf16 量化裕度)
        top_k: 取 top-K 最坏元素求平均
        grid_size: 错误位置空间网格每维最大分区数 (默认 16×16)

    Returns:
        dict, 包含全局指标、边界指标和错误位置统计
    """
    assert src_bf16.shape == ref_bf16.shape == ref_fp32.shape, "shape mismatch"
    H, W = src_bf16.shape
    eps = 1e-12

    ref_v = ref_fp32.double().reshape(-1)
    src_v = src_bf16.double().reshape(-1)
    torch_v = ref_bf16.double().reshape(-1)

    # ---------- 全局指标 ----------
    cos_src = (src_v @ ref_v) / (src_v.norm() * ref_v.norm() + eps)
    cos_torch = (torch_v @ ref_v) / (torch_v.norm() * ref_v.norm() + eps)

    rel_src = (src_v - ref_v).abs() / (ref_v.abs() + eps)
    rel_torch = (torch_v - ref_v).abs() / (ref_v.abs() + eps)

    rms_src = rel_src.pow(2).mean().sqrt()
    rms_torch_v = rel_torch.pow(2).mean().sqrt()
    median_src = rel_src.median()
    median_torch_v = rel_torch.median()

    # ---------- 边界指标 ----------
    max_rel = rel_src.max()
    actual_k = min(top_k, rel_src.numel())
    top_k_mean = rel_src.topk(actual_k).values.mean()

    # 混合判据: |diff| > atol + rtol * |ref|
    #   atol 用 sqrt(K)*eps 理论界, 覆盖小元素
    #   rtol 用固定常数, 覆盖大元素
    atol = (K ** 0.5) * eps_dtype * atol_scale
    rtol = rtol_const
    diff = (src_v - ref_v).abs()
    allowed = atol + rtol * ref_v.abs()
    bad_mask_1d = diff > allowed
    bad_mask_2d = bad_mask_1d.reshape(H, W)
    bad_ratio = bad_mask_2d.float().mean()

    # 超容忍倍数: diff / allowed, > 1 即坏点; 此值替代 max_rel/top_k_mean 作为判定依据
    excess = diff / allowed
    max_excess = excess.max()
    actual_k_excess = min(top_k, excess.numel())
    top_k_mean_excess = excess.topk(actual_k_excess).values.mean()

    row_bad_ratio = bad_mask_2d.float().mean(dim=1)
    col_bad_ratio = bad_mask_2d.float().mean(dim=0)
    bad_rows = torch.where(row_bad_ratio > 0.5)[0]
    bad_cols = torch.where(col_bad_ratio > 0.5)[0]

    # ---------- 错误位置统计 ----------
    bad_idx_1d = torch.nonzero(bad_mask_1d).flatten()
    bad_count = int(bad_idx_1d.numel())

    # 全局最坏点: excess = diff / allowed 最大处 (无论是否判定为坏点)
    worst_flat = int(excess.argmax().item())
    worst_point = {
        "row": worst_flat // W,
        "col": worst_flat % W,
        "src": src_v[worst_flat].item(),
        "ref": ref_v[worst_flat].item(),
        "diff": diff[worst_flat].item(),
        "allowed": allowed[worst_flat].item(),
        "excess": excess[worst_flat].item(),
    }

    # 粗粒度网格: 把 H×W 分成 grid_r×grid_c 个区域, 统计每格坏点数与密度
    grid_r = min(grid_size, H)
    grid_c = min(grid_size, W)
    bin_r = (H + grid_r - 1) // grid_r
    bin_c = (W + grid_c - 1) // grid_c
    dev = bad_mask_2d.device
    grid_hist = torch.zeros(grid_r, grid_c, dtype=torch.long, device=dev)
    grid_density = torch.zeros(grid_r, grid_c, device=dev)

    top_bad_rows = []
    top_bad_cols = []
    worst_bad_points = []
    if bad_count > 0:
        # 最坏坏点坐标 (仅统计 excess > 1 的坏点, 按 excess 降序)
        excess_bad = excess[bad_idx_1d]
        n_worst = min(top_k, bad_count)
        topv, topi = excess_bad.topk(n_worst)
        worst_rows = bad_idx_1d[topi] // W
        worst_cols = bad_idx_1d[topi] % W
        worst_bad_points = [
            (int(r), int(c), float(e))
            for r, c, e in zip(worst_rows.tolist(), worst_cols.tolist(), topv.tolist())
        ]

        # 行/列坏点数直方图 top-10 (坏点最多的行列)
        row_counts = bad_mask_2d.sum(dim=1)
        col_counts = bad_mask_2d.sum(dim=0)
        top_bad_rows = sorted(
            ((int(r), int(row_counts[r].item())) for r in
             torch.nonzero(row_counts).flatten().tolist()),
            key=lambda x: x[1], reverse=True)[:10]
        top_bad_cols = sorted(
            ((int(c), int(col_counts[c].item())) for c in
             torch.nonzero(col_counts).flatten().tolist()),
            key=lambda x: x[1], reverse=True)[:10]

        # 网格直方图 (边缘格面积可能偏小, 用实际元素数归一化密度)
        bin_idx = ((torch.arange(H, device=dev).unsqueeze(1).expand(H, W) // bin_r)
                   * grid_c + (torch.arange(W, device=dev).unsqueeze(0).expand(H, W)
                               // bin_c)).reshape(-1)
        grid_area = torch.bincount(bin_idx, minlength=grid_r * grid_c).float().clamp(min=1)
        grid_hist = torch.bincount(bin_idx[bad_mask_1d],
                                   minlength=grid_r * grid_c).reshape(grid_r, grid_c)
        grid_density = grid_hist.float() / grid_area.reshape(grid_r, grid_c)

    return {
        # 全局
        "cos_src_vs_fp32": cos_src.item(),
        "cos_torch_vs_fp32": cos_torch.item(),
        "rms_rel_src": rms_src.item(),
        "rms_rel_torch": rms_torch_v.item(),
        "median_rel_src": median_src.item(),
        "median_rel_torch": median_torch_v.item(),
        # 边界
        "max_rel_src": max_rel.item(),
        "top_k_mean_rel_src": top_k_mean.item(),
        "max_excess": max_excess.item(),
        "top_k_mean_excess": top_k_mean_excess.item(),
        "top_k": actual_k,
        "K": K,
        "atol": atol,
        "rtol": rtol,
        "atol_scale": atol_scale,
        "eps_dtype": eps_dtype,
        "bad_pixel_ratio": bad_ratio.item(),
        "bad_row_indices": bad_rows[:20].tolist(),
        "bad_col_indices": bad_cols[:20].tolist(),
        "bad_row_ratio_max": row_bad_ratio.max().item(),
        "bad_col_ratio_max": col_bad_ratio.max().item(),
        "bad_row_count": int(bad_rows.numel()),
        "bad_col_count": int(bad_cols.numel()),
        # 错误位置统计
        "H": H,
        "W": W,
        "bad_count": bad_count,
        "worst_point": worst_point,
        "worst_bad_points": worst_bad_points,
        "top_bad_rows": top_bad_rows,
        "top_bad_cols": top_bad_cols,
        "grid_r": grid_r,
        "grid_c": grid_c,
        "bin_r": bin_r,
        "bin_c": bin_c,
        "grid_hist": grid_hist.tolist(),
        "grid_density": grid_density.tolist(),
    }


def print_correctness_report(m, name="kernel"):
    """格式化打印评估结果(含中文解释)。"""
    sep_thick = "=" * 76
    sep_thin = "-" * 76

    print(sep_thick)
    print(f"  Correctness Report: {name}")
    print(sep_thick)

    print("  [全局指标 - 反映整体计算正确性]")
    print(sep_thin)

    print("  cos-sim vs fp32  (余弦相似度: 越接近 1 说明方向越一致)")
    print(f"    torch  : {m['cos_torch_vs_fp32']:.8f}")
    print(f"    {name:<7s}: {m['cos_src_vs_fp32']:.8f}")
    print()

    print("  median rel       (中位相对误差: 反映「典型元素」的精度)")
    print(f"    torch  : {m['median_rel_torch']:.8f}")
    print(f"    {name:<7s}: {m['median_rel_src']:.8f}")
    print()

    print("  RMS rel          (均方相对误差: 对大误差敏感, 异常点会拉爆)")
    print(f"    torch  : {m['rms_rel_torch']:.8f}")
    print(f"    {name:<7s}: {m['rms_rel_src']:.8f}")

    print(sep_thin)
    print("  [边界指标 - 识别少量元素严重错误的情形]")
    print(sep_thin)

    print("  max rel error    (最大相对误差: 远大于 1 强烈暗示位置错位)")
    print(f"    {name}: {m['max_rel_src']:.4f}")
    print()

    print(f"  top-{m['top_k']} mean rel  (最坏 K 个元素的平均相对误差: 量化「最坏一小撮」)")
    print(f"    {name}: {m['top_k_mean_rel_src']:.4f}")
    print()

    print("  max excess       (最大「超容忍倍数」: diff / (atol+rtol×|ref|), > 1 即坏点)")
    print(f"    {name}: {m['max_excess']:.4f}")
    print()

    print(f"  top-{m['top_k']} mean excess (最坏 K 个元素的平均超容忍倍数)")
    print(f"    {name}: {m['top_k_mean_excess']:.4f}")
    print()

    print(f"  bad pixel ratio  (坏点率 = |diff| > atol + rtol × |ref| 的元素占比)")
    print(f"    atol = sqrt(K) × eps × {m['atol_scale']} = sqrt({m['K']}) × {m['eps_dtype']:.4e} × {m['atol_scale']} = {m['atol']:.6f}")
    print(f"    rtol = {m['rtol']}")
    print(f"    {name}: {m['bad_pixel_ratio']*100:.4f}%")
    print()

    print("  坏点空间分布     (用于定位是否为边界 bug)")
    print(f"    行坏点率最大: {m['bad_row_ratio_max']*100:.2f}%")
    print(f"    列坏点率最大: {m['bad_col_ratio_max']*100:.2f}%")
    print(f"    >50% 坏点行数: {m['bad_row_count']}, 前 20 个索引: {m['bad_row_indices']}")
    print(f"    >50% 坏点列数: {m['bad_col_count']}, 前 20 个索引: {m['bad_col_indices']}")

    print(sep_thin)
    print("  [错误位置统计 - 坏点具体在哪些行列]")
    print(sep_thin)

    print("  全局最坏点       (excess = diff/(atol+rtol×|ref|) 最大处, 无论是否判定为坏点)")
    wp = m["worst_point"]
    print(f"    row = {wp['row']}, col = {wp['col']}")
    print(f"    src = {wp['src']:.6f}, ref = {wp['ref']:.6f}, "
          f"diff = {wp['diff']:.6f}, allowed = {wp['allowed']:.6f}, excess = {wp['excess']:.3f}")
    print()

    if m["bad_count"] == 0:
        print("  坏点总数: 0, 无位置统计输出")
    else:
        print(f"  坏点总数: {m['bad_count']} ({m['bad_pixel_ratio']*100:.4f}%)")
        print()

        print(f"  top-{len(m['worst_bad_points'])} 最坏坏点坐标 (按 excess 降序):")
        for r, c, e in m["worst_bad_points"]:
            print(f"    row = {r}, col = {c}   excess = {e:.3f}")
        print()

        print("  坏点最多行 top-10 (row, count):", m["top_bad_rows"])
        print("  坏点最多列 top-10 (col, count):", m["top_bad_cols"])
        print()

        # ASCII 密度网格: 单色渐变, 越深的字符表示该区域坏点越多 (sequential 编码)
        ramp = " .:oO#@"
        hist = m["grid_hist"]
        gmax = max((max(r) for r in hist), default=0)
        print(f"  坏点空间分布网格 ({m['grid_r']}×{m['grid_c']}, "
              f"每格约 {m['bin_r']}×{m['bin_c']} 元素):")
        for row_hist in hist:
            chars = []
            for v in row_hist:
                if v == 0:
                    chars.append(ramp[0])
                else:
                    idx = 1 + int(round((v / gmax) * (len(ramp) - 2)))
                    chars.append(ramp[min(idx, len(ramp) - 1)])
            print("    |" + "".join(chars) + "|")
        print(f"    图例: '{ramp[0]}' = 0 坏点, '{ramp[1]}'..'{ramp[-1]}' = 0..{gmax} 坏点/格")
    print(sep_thick)


def is_correct(metrics,
               max_bad_row_count=0,
               max_bad_col_count=0,
               max_bad_pixel_ratio=0.001,
               max_rms_ratio=3.0,
               max_excess_cap=3.0,
               max_top_k_excess=1.5):
    """对 evaluate_correctness 的输出做通过/失败判定 (面向边界异常)。

    所有条件必须同时满足 (AND) 才算 PASS。
    返回 (passed: bool, failures: list[str])
    """
    failures = []

    # A. 坏点空间分布: 没有任何 >50% 坏点的行/列
    if metrics["bad_row_count"] > max_bad_row_count:
        failures.append(
            f"bad_row_count = {metrics['bad_row_count']} > {max_bad_row_count}  "
            f"(rows w/ >50% bad pixels, 强烈的边界 bug 信号)"
        )
    if metrics["bad_col_count"] > max_bad_col_count:
        failures.append(
            f"bad_col_count = {metrics['bad_col_count']} > {max_bad_col_count}  "
            f"(cols w/ >50% bad pixels, 强烈的边界 bug 信号)"
        )

    # B. 散布坏点率
    if metrics["bad_pixel_ratio"] > max_bad_pixel_ratio:
        failures.append(
            f"bad_pixel_ratio = {metrics['bad_pixel_ratio']*100:.4f}% > "
            f"{max_bad_pixel_ratio*100:.4f}%"
        )

    # C. RMS 比值: 揭示「少量大误差」
    rms_ratio = metrics["rms_rel_src"] / max(metrics["rms_rel_torch"], 1e-12)
    if rms_ratio > max_rms_ratio:
        failures.append(
            f"rms_rel ratio = {rms_ratio:.2f}x > {max_rms_ratio:.2f}x  "
            f"({metrics['rms_rel_src']:.6f} vs torch {metrics['rms_rel_torch']:.6f})"
        )

    # D. 最大超容忍倍数: 单点位置错位的硬证据 (混合判据口径)
    if metrics["max_excess"] > max_excess_cap:
        failures.append(
            f"max_excess = {metrics['max_excess']:.4f} > {max_excess_cap}  "
            f"(单点 diff 超容忍倍数过大, 极可能位置错位)"
        )

    # E. top-K 最坏元素平均超容忍倍数
    if metrics["top_k_mean_excess"] > max_top_k_excess:
        failures.append(
            f"top-{metrics['top_k']} mean excess = {metrics['top_k_mean_excess']:.4f} > "
            f"{max_top_k_excess}"
        )

    return (len(failures) == 0, failures)


def print_verdict(passed, failures):
    sep_thick = "=" * 76
    print(sep_thick)
    if passed:
        print("  Verdict: [PASS]")
    else:
        print("  Verdict: [FAIL]")
        print("  Failures:")
        for f in failures:
            print(f"    - {f}")
    print(sep_thick)


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            # T.use_warp_divergence(True)
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, k * block_K], A_shared, major="row_major")
                T.copy(B[k * block_K, bx * block_N], B_shared, major="col_major")
                T.gemm(A_shared, B_shared, C_local, k_pack=1, policy=T.GemmWarpPolicy.FullRow,)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    M = 128
    N = 32
    K = 128
    kernel = matmul(M, N, K, 128, 32, 128)

    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    c = kernel(a, b)
    # c = (a.cpu() @ b.cpu().T).cuda()
    ref_c = a @ b

    profiler = kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")
    print(f"GEMM TFLops: {2 * M * N * K / (latency * 1e9)}")
    print(kernel.get_kernel_source())

    ref_fp32 = (a.float() @ b.float())
    metrics = evaluate_correctness(c, ref_c, ref_fp32, K=K)
    print_correctness_report(metrics, name="tilelang-NT")

    passed, failures = is_correct(metrics)
    print_verdict(passed, failures)


if __name__ == "__main__":
    main()
