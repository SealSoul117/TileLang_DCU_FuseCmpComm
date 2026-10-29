# TileLang 三种融合算子：性能对比与优化汇总

补充：当前三个算子的指针外提 / half8 向量化开关对照实测，见
[2026-10-14 IPC PASS 消融报告](IPC_PASS_ABLATION.md)。这是同场开关实验，
不应与下文历史手写基线混算优化收益。

更新日期：2026-10-13。AG 2048 已更新为小尺寸修复后的同场复测；其余行整理已有实测记录。
“目前 TileLang 版本”指当前标量通信前端、后端自动指针外提与 half8 向量化的实现，
不是历史手工改写 HIP 的消融实验，也不是早期未命中向量化 PASS 的版本。

## 1. 如何阅读这些数据

环境均为单节点四卡 BW1000 / gfx936、DTK 26.04。FP16 输入/输出、FP32 GEMM 累加；
固定 80 个 Block、每 Block 512 线程、64 KiB LDS，GEMM tile 为 256×256×16、四级流水。
这里的“手写版”指手工组织调度、通信与同步的融合 HIP；其 GEMM 计算体也来自 TileLang，
不是 rocBLAS + RCCL 串行基线。

下表耗时单位均为 **ms，越小越好**；`TL/手写 = TileLang 耗时 ÷ 手写耗时`，大于 1
表示 TileLang 记录值更慢。不同算子的分片和输出不同，不能直接横比三个算子的耗时。

| 对比 | TileLang 记录 | 手写记录 | 可比性 |
|---|---|---|---|
| GEMM + AllReduce | 2026-10-13，job 10109 | 同作业、同轮交替测试 | 同条件对照 |
| GEMM + ReduceScatter | 2026-10-06，job 9827，最终 half8 JIT | 2026-09-25，job 9259 | 历史参考，非同场 A/B |
| AllGather + GEMM，2048 | 2026-10-13，job 10112，修复后 | 同作业、同分配交替测试 | 同条件对照，算法调度仍不同 |
| AllGather + GEMM，其余尺寸 | 2026-10-13，job 10111 | 2026-09-25，job 9259，原始 IPC 参考版 | 历史参考，且调度不同 |

统计口径也需区分：

- TileLang AR/RS：30 warmup、30 有效样本，`median_iteration(max_PE(event_ms))`。
  AR 同轮手写对照使用同一分配、输入和计时方法。
- TileLang AG：2048 最新对照为 30 warmup、30 有效样本；其余尺寸为 5 warmup、20 样本。
  同样先逐次取四 PE 最大值，再取中位数。
- 历史手写 RS/AG：5 轮，每轮 5 warmup、30 样本，
  `median_round(max_PE(mean_iteration(event_ms)))`。
- 均测 GPU 算子区间，包含融合 kernel 内的计算、通信、等待和完成协议；不包含初始化、
  分配、编译、输入上传、flags reset、host MPI barrier 和 CPU 校验。
  **不是含准备操作的完整 Python API 延迟。**

因此，RS 和 AG 非 2048 行的比值只描述记录值之比，不能作为严格、稳定的性能差距或优化加速比。
RS 最新 half8 性能记录是 10 月 6 日；后续 AR/AG 扩展检查了 RS 代码生成回归，
但没有在 10 月 13 日重新完成四尺寸 RS 性能测试。

## 2. 三个算子的性能

### 2.1 GEMM + AllReduce：最新同条件对照

每个 PE 计算本地 `[M,K] × [K,N]`，四个 partial 归约后，每个 PE 得到完整 `[M,N]`。
本表 `M=N=S，K=S/4`；对照手写 `dcu_fused_gemm/gemm_ar/v2` 的 `fused_vector`。
compute/comm 数对 TileLang 和该手写对照相同。

| 本地 GEMM M/N/K | compute/comm | TileLang 自动 half8 | 手写 fused_vector | TL/手写 | TileLang 多耗时 |
|---|---:|---:|---:|---:|---:|
| 2048/2048/512 | 32/48 | 0.184400 | 0.176239 | 1.046× | 4.63% |
| 4096/4096/1024 | 56/24 | 0.554960 | 0.534000 | 1.039× | 3.93% |
| 8192/8192/2048 | 64/16 | 1.980641 | 1.938320 | 1.022× | 2.18% |
| 16384/16384/4096 | 72/8 | 10.880397 | 10.252638 | 1.061× | 6.12% |

结论：本轮 AR 已接近手写参考，记录差距约 2.2%～6.1%，但尚未完全持平。
从 Python 显式 half8 改成由 PASS 自动生成后，性能变化约在 ±0.3% 内；
这一步主要让前端更简单，不应宣称产生了新的显著加速。

来源：[AR 向量化报告](../../../docs/compiler_internals/distributed_fusion_ipc_ar_vectorize_review.md)，
[真实 JIT 计时目录](../../../build-old/ipc-ar-vectorize-20261013/after-enabled/perf/)。

### 2.2 GEMM + ReduceScatter：当前 half8 与历史手写参考

每个 PE 计算 `[S,S/4] × [S/4,S]`，随后归约并输出本 PE 的连续行分片 `[S/4,S]`。
TileLang 与历史手写参考均使用下列固定 compute/comm 分工及 owner 交错计算顺序。

| 本地 GEMM M/N/K | compute/comm | TileLang 自动 half8 | 历史手写 IPC | 记录值 TL/手写 |
|---|---:|---:|---:|---:|
| 2048/2048/512 | 48/32 | 0.135439 | 0.120421 | 1.125× |
| 4096/4096/1024 | 48/32 | 0.374720 | 0.330186 | 1.135× |
| 8192/8192/2048 | 72/8 | 1.448397 | 1.270847 | 1.140× |
| 16384/16384/4096 | 76/4 | 8.941585 | 7.572880 | 1.181× |

结论：最新 half8 记录比历史手写值多约 12.5%～18.1%；不是同条件差距结论。
大尺寸已经实际命中向量化，不再使用此前约 2.799/20.820 ms 的未向量化结果描述当前版本。

来源：[RS half8 报告](../../../docs/compiler_internals/distributed_fusion_ipc_vectorize_review.md)，
[最终 half8 原始记录](../../../build-old/ipc-vector8-20261006/summary.json)（只取 `m*-final`），
[历史手写性能报告](../../../../dcu_fused_gemm/gemm_rs/PERFORMANCE.md)，
[历史 RS 原始汇总](../../../../dcu_fused_gemm/gemm_rs/results/final_compare/summary.json)。

### 2.3 AllGather + GEMM：2048 修复后同场对照，其余尺寸历史参考

每个 PE 持有 A 的行分片 `[S/4,S]`、B 的列分片 `[S,S/4]`；先按块收集完整 A，
就绪行块与 GEMM 重叠执行，输出本地列分片 `[S,S/4]`。

| 本地 GEMM M/N/K | TileLang compute/comm | TileLang 自动 half8 | 手写 IPC（口径见上） | TL/手写 |
|---|---:|---:|---:|---:|
| 2048/512/2048 | 48/32 | 0.137439 | 0.135520 | 1.014× |
| 4096/1024/4096 | 48/32 | 0.427199 | 0.390373 | 1.094× |
| 8192/2048/8192 | 56/24 | 1.662638 | 1.564676 | 1.063× |
| 16384/4096/16384 | 64/16 | 13.439430 | 12.145517 | 1.107× |

这个表有额外的算法调度差异：

- 两边初始 Block 数相同，但 TileLang 始终固定角色；手写参考采用动态 GEMM 队列，
  通信 Block 完成 gather 和 reader-done 协议后，会加入计算。
- TileLang 每个 128 KiB chunk 有独立 single-writer ready flag，计算方逐个等待；
  手写版对每个 256 行块使用 SYSTEM ACQ_REL counter，计算方等待完成计数。
- TileLang 2048 仍只有 16 个输出 tile、部分 compute Block 空闲。现在通过显式范围判断
  和带标记的恒定一次循环复用资源基址路径，并消除原动态 0/1 次循环导致的严重 spill；
  并未改变 48/32 配比。旧 0.615919 ms 是修复前记录，不能再代表当前版本。

2048 同分配对照独立两轮为：修复后 0.137199/0.137439 ms，手写 0.135760/0.135520 ms，
差距约 1.1%～1.4%。修复前同场为 0.612800/0.612719 ms，前后加速约 4.46×。
完整消融、机器码和复测入口见 [2048 性能修复报告](AG2048_PERFORMANCE_FIX.md)。

4096～16384 的记录值较接近历史参考，但没有同场比较，不能据此宣称已经追平手写版。

来源：[AG 集成报告](../../../docs/compiler_internals/distributed_fusion_ipc_ag_gemm_review.md)，
[TileLang 四尺寸原始记录](../../../build-old/ipc-ag-gemm-20261013/after/)，
[历史手写报告](../../../../dcu_fused_gemm/gemm_rs/PERFORMANCE.md)，
[历史 AG 原始汇总](../../../../dcu_fused_gemm/ag_gemm/results/final_compare/summary.json)。

#### 补充：手写 AG 后续 v2 调优结果

另有手写 `ag_gemm/v2` 的冻结 `queue_pull` 配置，不能把上表原始 IPC 参考当成唯一或最新
手写版本。以下保留其独立主确认数据，不与 TileLang 数据组成同条件加速比：

| S | v2 冻结配置 | v2 手写耗时 ms |
|---:|---|---:|
| 2048 | 128 KiB，comm32 | 0.138090 |
| 4096 | 512 KiB，comm32 | 0.463055 |
| 8192 | 512 KiB，comm16 | 1.394465 |
| 16384 | 512 KiB，comm24 | 11.943035 |

该组为 3 seeds / 5 rounds / 30 iterations；4096 独立复测为 0.370527 ms，
说明环境波动不能忽略。没有挑选主确认和复测中的最小值拼成一张“最佳性能”表。
来源：[手写 AG v2 最终报告](../../../../dcu_fused_gemm/ag_gemm/v2/FINAL_REPORT.md)。

## 3. 目前实际采用的优化

### 3.1 单 kernel 分工与 tile/chunk 级计算通信重叠

三个 TileLang 算子均只有一次融合 kernel 启动，使用计算 Block / 通信 Block 分工。
AR/RS 在 GEMM tile 就绪后立即规约，不等所有 GEMM 完成；AG 在相应 256 行输入就绪后
立即开始 GEMM，不等整个 AllGather 完成。80 Block 全驻留是保证等待方与生产方能前进的
前提，runner 会检查 occupancy；这不是可任意超额启动的通用 persistent 调度器。

- **RS**：按 owner 交错计算 tile，使四个 owner 更早获得待规约数据；只由负责该输出
  分片的 owner 拉取四 PE partial 并规约，避免每个 PE 重复计算完整输出。
- **AR**：owner 为 `tile % 4`，owner 拉取并规约，随后主动 push 到四 PE 输出，
  对应 ReduceScatter + owner-push AllGather；当前不是让每个 PE 各自拉取四份数据做全量归约。
- **AG**：128 KiB 固定块并行拉取、local-rank-first 行分片顺序；gathered A 只被本地
  GPU 使用，采用 DeviceDefault，远端可读 A_shard 与 flags 使用 FineGrained。

这些是算法、调度和内存配置，不是编译器 PASS 自动选择出来的参数。
当前通信基于仓库内置 HIP IPC，不依赖 rocSHMEM/dushmem；没有接入 SDMA 通信路径。

### 3.2 保留四级 GEMM 流水与按 tile 构造资源基址

复用现有 `T.copy(..., enable_async=True)`、四级 `T.Pipelined`、`T.gemm`。
后端识别真正的内层 K 循环，避免把外层 persistent tile 遍历误当成 K 流水边界。

对显式标记 `tl.dcu_persistent_tile` 的循环，在每个 tile 内构造 A/B resource base，
让 K 循环里的地址只含 tile 内局部偏移，复用原生 GEMM 的资源地址生成路径。
不改变未标记的普通 GEMM；保留 A/B 命名约定，没有顺带修复 idxen 对名称的依赖。
AG 2048 的有效计算 Block 使用带标记的恒定一次循环；后端仅对显式 `min=0, extent=1`
允许无 tile-varying 基址的窄例外，避免动态 0/1 次循环的高 spill 路径。

代码：[K 循环识别](../../../src/transform/inject_utils.cc)、
[persistent resource lowering](../../../src/transform/lower_dcu_resource.cc)。

### 3.3 通信线程索引采用已验证的汇编绕行

通信使用 `tl::ipc::opaque_thread_id(tx)` 的空内联汇编屏障，保留真实线程编号，
绕过本环境“实际启动 512 threads，但采用 max-threads=256 编译配置”的索引优化问题。
它不是 warp/lane 自动重构，也不插入线程同步。此前的 persistent 专用 GEMM lane
保护已清理，不将它列为当前仍启用的优化。

代码：[IPC helper](../../../src/tl_templates/dcu_hip/ipc.hpp)。

### 3.4 通信 peer 指针外提 PASS

`HoistDCUIPCPointers` 将内层元素循环中重复的远端基址解析移动到循环外，保留在
正确的 tile/chunk、条件和同步作用域内：

- RS：四个输入 peer 指针；AR：四个 partial 指针及四个 output 指针。
- AG：一个随外层 chunk 改变、但在当前元素循环内不变的 source-peer 指针。
- AG 扩展支持词法外层 serial-loop/thread-binding 变量决定的 peer；仍不把当前
  元素索引、内存读取或调用误认成纯不变量，不跨 wait/fence/barrier 外提。

代码：[HoistDCUIPCPointers](../../../src/transform/hoist_dcu_ipc_pointers.cc)。

### 3.5 通信索引简化与外层范围分析

Python 表达简单线性遍历；既有 Simplify 在识别 opaque thread 范围后化简
除法、取模、循环次数及能证明冗余的边界判断，没有另建复杂的 serial-loop 自动重构机制。
向量化 PASS 重放外层 block/loop/Let/分支/assume 范围事实，使大尺寸也能证明索引合法，
避免“仅小尺寸命中向量化”的旧问题。

代码：[Simplify](../../../src/transform/simplify.cc)、
[索引简化报告](../../../docs/compiler_internals/distributed_fusion_ipc_index_simplify_review.md)。

### 3.6 由后端生成 FP16×8 向量访存

前端仍写 `for linear in T.serial(comm_tx, 65536, 512)`。后端把每线程 128 轮标量工作
重排成 16 轮、每轮连续 8 个 half，完整且不重复地覆盖原 65536 元素，生成 `uint4`
load/store（16 字节）：

| 算子 | 每个向量迭代的通信数据操作 |
|---|---|
| GEMM + RS | 四路 half8 load，按原顺序 FP32 累加，转 FP16 后写本地输出 |
| GEMM + AR | 四路 half8 load，FP32 累加，绑定一次 FP16 向量结果，复用到四路 output store |
| AG + GEMM | 一路远端 half8 load，一路本地 gathered-A half8 store |

匹配器验证 ABI、范围、连续性和 16-byte 对齐；当前不添加设备侧向量化安全 guard 或
标量 fallback。不能在编译期证明的候选保持原样；调用者仍须保证分配有效、对齐、
输入输出不重叠及生命周期。**未删除 IPC helper 的既有检查，也没有删掉同步协议。**

代码：[VectorizeDCUIPCCommunication](../../../src/transform/vectorize_dcu_ipc_communication.cc)。
这是 gfx936/四 PE/512 threads 下 RS、AR、AG 已验证模式的窄优化，不是通用向量化。
示例默认启用；直接 JIT 调用必须传：

```python
kernel = tilelang.compile(
    bound,
    target="hip -mcpu=gfx936",
    execution_backend="ctypes",
    pass_configs={"tl.enable_dcu_ipc_vectorize": True},
)
```

不能只设置外层 PassContext 就认定真实 JIT 已启用；还需核验导出的通信 HIP。

## 4. 当前结论与未完成事项

1. **AR 最接近手写参考**：已有同条件数据，差距约 2.2%～6.1%。
2. **RS 已补齐大尺寸 half8**：记录值接近历史参考，但仍需同作业复测量化实际差距。
3. **AG 功能与两个 PASS 已验证**：四尺寸完整 gathered-A/output 校验、2048 continuous
   与 scalar 对照通过；2048 的退化循环性能问题已修复，同场两轮与手写差距约 1.1%～1.4%。
4. TileLang AG 尚未移植手写的动态队列、通信 CTA 转计算角色、row-counter 与 v2 chunk
   调优；这不是已完成的 M3。AR 仍有 SGPR/scratch/spill 差异，尚未完成针对性优化。
5. 保留既有验证边界：RS 4096 continuous 的原容差问题没有在本轮修复；部分融合 HIP
   存在 A/B LDS 布局生成非确定性，不能据零点几百分点波动作单因素归因。

入口：[GEMM+RS](../gemm_rs_prototype/gemm_reduce_scatter_scheduled.py)、
[GEMM+AR](../gemm_ar_prototype/gemm_allreduce_scheduled.py)、
[AG+GEMM](allgather_gemm_scheduled.py)。
AG 编译运行方法见 [README](README.md)。
