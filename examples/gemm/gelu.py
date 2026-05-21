import torch
import time

M, N, K = 8192, 1024, 4096 
device = "cuda"  # ROCm 下 PyTorch 依然沿用 "cuda" 作为设备字串

# 输入保持为 float16 (half)，用以触发 AMD Matrix Core 的 HGEMM 硬件加速
A = torch.randn(M, K, device=device, dtype=torch.float16)
B = torch.randn(K, N, device=device, dtype=torch.float16)

warmup = 10
repeat = 100

def ref_gemm_gelu(A, B):
    # 1. half * half -> 触发 rocBLAS 的 Matrix Core 核心（内部使用 FP32 累加）
    # 乘完后立即转为 float32，防止 PyTorch 默认把结果截断回 half，对齐 TileLang 的写回逻辑
    C_float = torch.mm(A, B).to(torch.float32)
    
    # 2. 在 float32 精度下完成 GELU (Tanh) 计算
    C_float = torch.nn.functional.gelu(C_float, approximate="tanh")
    
    # 3. 直接返回 float32 结果
    return C_float

# --- Warmup ---
for _ in range(warmup):
    C = ref_gemm_gelu(A, B)
    torch.cuda.synchronize()

# --- Timing ---
start = time.time()
for _ in range(repeat):
    C = ref_gemm_gelu(A, B)
torch.cuda.synchronize()
end = time.time()

# --- 计算性能指标 ---
avg_time_sec = (end - start) / repeat
avg_time_ms = avg_time_sec * 1000

# GEMM 的总浮点运算次数 (2 * M * N * K)
flops = 2 * M * N * K
tflops = (flops / avg_time_sec) / 1e12

# 打印输出，完美适配你的第一版日志解析脚本
print(f"Matrix dimensions: M={M}, N={N}, K={K}")
print(f"Avg time: {avg_time_ms:.3f} ms")
print(f"Ref TFlops: {tflops:.6f}")