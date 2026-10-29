# 四 PE AllGather + GEMM 原型

此目录只包含 TileLang 前端、受限 IPC 优化路径与四 rank 运行器；不是通用
AllGather 实现。请在旧树隔离环境中运行：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
python examples/distributed/ag_gemm_prototype/allgather_gemm_scheduled.py \
  --m 2048 --n 512 --k 2048 --output /tmp/ag-source
```

支持的形状为 `M=K=S`、`N=S/4`，其中 `S` 为 2048、4096、8192 或
16384。固定 80 个 512-thread CTA；通信 CTA 数依次为 32、32、24、16。
通信以固定的 65536 个 FP16 元素为一个 chunk（不是固定 256 行），每个
256 行 GEMM 输入 tile 在读取前等待覆盖其行的全部 chunk-ready 标志。

每个 rank 的 `A_shard[S/4,S]` 和 ready flags 使用 FineGrained IPC
allocation；收集后的 `A[S,S]`、`B[S,S/4]` 与 `out[S,S/4]` 使用
DeviceDefault allocation。rank 本地 shard/source rows 最先遍历，随后轮转
另三个 source。通信 writer 在每个 chunk 后执行 system fence、barrier 和
thread-0 ready signal；结束时的 peer-done 握手保证远端 `A_shard` 生命周期
在下一轮重用前已经安全。

前端故意写成 `opaque_thread_id` 上的标量 128×512 copy。验证过 IPC ABI 的
`HoistDCUIPCPointers` 仅将循环不变的远端 source base 留在 chunk 范围内；
`VectorizeDCUIPCCommunication` 随后只对已证明的完整、连续、16-byte 对齐
copy 重写为 `ipc_ag_half8_i`/`uint4`。它不提供运行时 fallback，也不影响
普通 GEMM、RS 或 AR 路径。`--scalar` 显式关闭该 pass。

四 rank 运行和导出真实 JIT HIP：

```bash
# 在本站申请同一节点的四张 DCU；请在拿到的交互 shell 内执行下一段。
srun --account=ncic --partition=bw1000 --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=64G --time=01:00:00 --gres=dcu:bw1000:4 --pty bash

cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
timeout --signal=TERM 300s mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python examples/distributed/ag_gemm_prototype/benchmark_allgather_gemm.py \
  --m 2048 --n 512 --k 2048 --warmup 5 --repeats 20 --timeout-seconds 120 \
  --json-dir build-old/ag-gemm-run --export-hip build-old/ag-gemm-run/hip
```

runner 将 vectorize 配置直接传给 `tilelang.compile(pass_configs=...)`，并在
计时前验证导出的 JIT HIP 包含一个 hoisted remote `A_shard` base 和一个
half8 copy。它还检查 80 CTA/512 thread/64 KiB 的完整驻留条件。每个 epoch
在计时外 poison gathered-A 和 output，然后仅 local HIP synchronize 后做
完整 gathered-A 和 GEMM output 校验，最后才进行跨 PE sync。

默认 dyadic 输入的 CPU oracle 不做 16K 矩阵乘：它依据 source、global row、
global column、`q=k%4` 和 epoch 的整数公式直接计算输出；gathered-A 与输出
均按 FP16 bitwise 比较。`--continuous` 是 `S<=2048` 的 NumPy matmul
诊断路径，使用 `rtol=atol=1e-3`。每 rank JSON 保留原始 HIP-event samples；
跨 rank 汇总须先逐轮取 max，再计算 median。

## 本轮验证

四种尺寸均已在四卡 BW1000 上完成真实 JIT 编译、完整 gathered-A/output 校验，
2048 的 scalar 对照与 continuous 输入也通过。相关回归测试共 77 项通过。
生成代码、计时和边界说明见
[AG+GEMM 集成报告](../../../docs/compiler_internals/distributed_fusion_ipc_ag_gemm_review.md)。
这是一版已经验证 PASS 生效的固定角色原型，不是已完成动态调度或性能调优的最终版本。

后续 2048 性能修复已完成：同分配两轮约 0.137 ms，手写约 0.136 ms，默认 48/32 不变；
新库相关回归 82 项通过，四尺寸四卡正确性重新验证通过。

最小前端 API 用法如下；实际执行仍必须由四 rank runner 提供 IPC allocation
和 ABI 参数，不能把这个函数当作单卡 GEMM 调用。

```python
import tilelang
from examples.distributed.ag_gemm_prototype.allgather_gemm_scheduled import build_bound

raw, bound = build_bound(2048, 512, 2048, total_blocks=80, comm_blocks=32)
kernel = tilelang.compile(
    bound, target="hip -mcpu=gfx936", execution_backend="ctypes",
    pass_configs={"tl.enable_dcu_ipc_vectorize": True},
)
```

角色固定为 compute CTA 与 communication CTA 两组；没有添加手写 HIP、动态
work queue/M3 counter 或动态任务分配。`S=2048` 只有 16 个 output tiles，少于
48 个 compute CTA，因此仍有空闲 compute CTA。小尺寸性能修复使用显式范围判断
和带 `tl.dcu_persistent_tile` 标记的恒定一次 tile 循环；多 tile 的配置保留原调度。
该写法既复用 tile-local 资源基址路径，又避免动态 0/1 次外层循环引起的严重寄存器溢出。
默认 compute/comm 数、四级 K 流水及通信协议均未改变。

具体消融、同场手写对照和验证边界见 [2048 性能排查](AG2048_PERFORMANCE_FIX.md)。
