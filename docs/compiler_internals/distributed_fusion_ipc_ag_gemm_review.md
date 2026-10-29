# TileLang AllGather + GEMM：标量前端与 IPC PASS 集成

> 后续更新：本文保留初次集成时的代码边界与性能记录。2048 小尺寸随后已修复，
> 默认 48/32 下由约 0.613 ms 降至 0.137 ms；此前“不标记小尺寸循环”的说明不再代表当前实现。
> 见 [2048 性能排查与修复](../../examples/distributed/ag_gemm_prototype/AG2048_PERFORMANCE_FIX.md)。

日期：2026-10-13。开发、构建及验证均在 `TileLang_DCU_old`，没有 git commit。
本模块由用户指定的 gpt-5.6-terra/high 子代理实现，主代理设计并独立 review；没有继续分派。

## 算子与边界

入口是 `examples/distributed/ag_gemm_prototype/allgather_gemm_scheduled.py`。
四 PE，每个 PE 持有 A 的行分片 `[S/4,S]` 和 B 的列分片 `[S,S/4]`；
在每个 PE 收集完整 A `[S,S]`，生成本地输出 `[S,S/4]`。FP16 输入/输出，FP32 累加。
S 支持 2048、4096、8192、16384，GEMM tile 为 256×256×16，四级流水。

沿用现有 AR/RS 原型的固定 80 CTA、512 threads、64 KiB LDS 方案。
四尺寸 compute/comm CTA 分别为 48/32、48/32、56/24、64/16。
两类 CTA 不动态转换角色；不是手写 AG 的动态 work queue，也不代表 M3 完成。
2048 只有 16 个输出 tile，因此部分计算 CTA 空闲；每 CTA 至多一 tile，前端不添加
`tl.dcu_persistent_tile` 标记。其余尺寸保留多 tile 外层循环的既有资源基址处理。
没有修改 persistent resource 后端或放松其标记合法性检查。

通信每 chunk 65536 个 half（128 KiB），各通信 CTA round-robin 分工，先本 PE 行分片，
随后轮转其他 PE。每个 chunk 单 writer 发布独立 ready flag；计算 CTA 等齐一个
256 行输入区域的全部 chunk，再开始该 tile 的 GEMM。每个 payload writer 都执行
system fence，再由 barrier 汇合，thread 0 release 发布；消费侧 acquire wait、fence、barrier。
全部通信 CTA 的本地 done 与跨 PE reader-done 握手保护源 A_shard 的生命周期。
不依赖尚未实现的 SYSTEM ACQ_REL row-counter RMW 接口。

A_shard/ready 使用 FineGrained，gathered A、B、output 使用 DeviceDefault。
全驻留检查是运行前置条件，不能推广到任意超额 grid 或跨节点。

## 必要的编译器改动

1. `src/transform/hoist_dcu_ipc_pointers.cc`：AG 的 source peer 依赖外层 chunk loop
   与 blockIdx，但在内层 128 轮元素循环中不变。旧 PASS 的真实 source-only HIP
   在元素循环内重复计算 `remote_source_remote_ptr_0`，见
   `build-old/ag-gemm-20261013/source-scalar-2048/`。现在仅接纳词法外层的 trivial
   serial loop 变量与 thread-extent 变量作为不变量，仍拒绝当前元素变量、Load、Call；
   不跨 barrier、wait、未知副作用或条件进行外提。
2. `src/transform/vectorize_dcu_ipc_communication.cc`：原匹配器只识别 RS/AR 的规约。
   新增单个已外提远端指针 → 一个本地参数 buffer 的直接 copy 模式；复用 ABI 校验、
   外层范围事实和线程坐标变换。检查源 PE 属于 [0,4)、完整 128×512 标量覆盖、
   `base + linear`、源/目标范围、连续 8 lane 和 16-byte 对齐，生成 16 轮 half8 copy。

上述改动在没有测试的情况下也仍是实际 AG 优化路径所需，不是为了满足测试修改原有接口。
不改 HIP codegen、IPC runtime、原生 GEMM、AR/RS 前端或流水线顺序。
仍是 gfx936、四 PE、512 threads 的窄优化：编译期不能证明时保留标量，不增加运行时
安全分支。调用者仍须保证 allocation 有效、16-byte 对齐、输入输出不重叠及生命周期。

前端只写标量遍历：

```python
remote_source = T.remote_view(A_shard, source_peer)
opaque_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
for linear in T.serial(opaque_tx, 65536, 512):
    A[global_row + linear // k, linear % k] = remote_source[
        shard_row + linear // k, linear % k]
```

真实 HIP 等价于以下结构（偏移表达式简写，非复制手写 HIP）：

```cpp
auto remote_source = tl::ipc::peer<std::byte>(descriptor, source_peer, rank, world, 0);
for (int ipc_ag_half8_i = 0; ipc_ag_half8_i < 16; ++ipc_ag_half8_i) {
    *(uint4*)(A + destination_base + ipc_ag_half8_i * 4096 + opaque_tx * 8) =
        *(uint4*)(remote_source_half + source_base + ipc_ag_half8_i * 4096 + opaque_tx * 8);
}
```

runner 显式传 `tilelang.compile(pass_configs={"tl.enable_dcu_ipc_vectorize": True})`，
在执行前核验实际 JIT HIP，而非只看 source-only 或外层 PassContext。
`--scalar` 使用同一前端，仅关闭向量化 PASS，指针外提仍启用。

## 验证记录

Slurm job 10111，dcu0，四卡 BW1000 / gfx936 / DTK 26.04。
主代理证据根目录：`build-old/ipc-ag-gemm-20261013/`。

- `before/`：编译器源码/二进制快照、AR/RS 四尺寸与普通/未标记 GEMM 基线。
- `audit_ag.py` / `after/audit/`：四尺寸分别捕获两个 PASS 前后 TIR；两者确实改写
  通信部分，GEMM 分支逐一通过结构相等检查。每个尺寸生成单个融合 kernel。
- `after/m2048`、`m4096`、`m8192`、`m16384`：真实 JIT HIP、四 rank JSON 和运行日志。
  四尺寸完整 gathered A 与 output 的 FP16 bitwise 校验通过，包含 poison 后 epoch 0、
  epoch 1，以及计时后的再次检查；运行前确认 80 CTA 可完全驻留。
- `after/scalar2048`：同一前端关闭向量化的四卡对照，通过全量校验。
- `after/continuous2048`：连续随机输入、完整 NumPy FP32 GEMM 参考校验通过；
  gathered A 仍按位验证，包含 epoch 0/1 和计时后检查。
- `tests-final.log`：77 项通过，包括向量化、外提、AG/AR 调度与 JIT 开关、索引化简、
  persistent resource、M2 独立接口回归。初次新增 hoist 测试的 walker 漏遍历
  `AttrStmt`，仅修测试遍历器；没有为测试修改生产语义或弱化断言。
- `after/native/manifest.json`：7 份普通/未标记 GEMM HIP 与本轮修改前基线逐字相同；
  `gemm-smoke.log`：两种普通 GEMM 的真实 GPU 数值检查通过。
- `after/regression/`：AR/RS 各四尺寸的向量化 PASS 输入、输出 TIR 与基线逐字相同，
  通信 HIP 也相同。**不是所有最终 HIP 都逐字相同**：AR 2048/16384 与 RS 16384
  出现原有 A/B LDS 布局次序翻转。同一最终二进制再次生成（`regression-repeat/`），
  AR 16384、RS 8192 也会翻转，证明存在同版本的生成非确定性。保留原始差异，
  不通过删行、归一化比较或额外修改 GEMM 后端掩盖它。
- `paths-final.log`：Python、TileLang、动态库、模板和缓存均来自 old 隔离环境，
  未加载 develop；`build-final.log`：TileLang 与内置 IPC runtime 构建通过。

默认 dyadic oracle 独立按 source/global row/global col/epoch 的整数公式计算，
不是以 kernel 结果作为参考。另提供小尺寸 continuous NumPy GEMM 校验；
gathered A 始终按位比较，只有 continuous GEMM 输出使用 rtol=atol=1e-3。

本轮快速计时：5 warmup、20 repeats，每轮取四 PE HIP event 最大值再取中位数；
不包含分配、清空、MPI barrier、CPU 校验，单位 ms。

| M | N_local | K | compute/comm | 融合耗时 |
|---:|---:|---:|---:|---:|
| 2048 | 512 | 2048 | 48/32 | 0.6159 |
| 4096 | 1024 | 4096 | 48/32 | 0.4272 |
| 8192 | 2048 | 8192 | 56/24 | 1.6626 |
| 16384 | 4096 | 16384 | 64/16 | 13.4394 |

2048 scalar 对照为 0.7673 ms，默认 half8 为 0.6159 ms；这两次顺序短测不是充分的
交替对照实验，不将差值全部归因于向量化，也不推广成稳定加速比。

这些是初步数据，不是手写版本的同条件性能对比。尤其 2048 调度利用率低且未使用多 tile
资源路径，不能据此认定尺寸与性能单调变化，也未归因其具体瓶颈。后续若追求手写性能，
需要单独对齐调度并测量；本模块不自动扩展到动态调度实现。
