# GEMM+AllReduce 通信指针外提扩展

日期：2026-10-12。仅修改 `TileLang_DCU_old`，未提交 git commit。
实现由指定的 gpt-5.6-terra/high 子代理完成，主代理负责设计、review、构建及独立验证。

## 原因与修改

真实输入为 `examples/distributed/gemm_ar_prototype/gemm_allreduce_scheduled.py`。
原有 `HoistDCUIPCPointers` 未能处理其通信循环，原因有两项：

1. 原实现只处理 serial 循环开头连续出现的指针 Let；AR 的四个 output 指针在归约之后，
   并不属于这段前缀。
2. AR 的八元素归约在这个阶段仍是小型 unrolled For，且由
   `pragma_unroll_explicit=False` 包裹；原实现拒绝任何嵌套 For/AttrStmt。

本轮仅修改生产文件 `src/transform/hoist_dcu_ipc_pointers.cc`，扩展已有 pass，
不新增 pass，不移动编译阶段，不修改 AR/RS Python 算子、IPC runtime、手写 HIP 或 GEMM 后端。
即使删除所有测试，这个改动仍是让真实 AR 路径自动外提指针所必需的，不是测试接口适配。

- 从 serial 循环直接、无条件的 SeqStmt 子节点提取合格地址 Let，允许它们不连续。
  仅移动地址生产者；每个 Let 的 payload body 留在原顺序位置。
- 允许常量起点非负、常量 extent 在 0～16、无线程绑定/附加 annotation 的
  unrolled/vectorized payload For；递归审计其中所有操作。
- 仅允许指向该循环变量、值为字面量 0/1 的 `pragma_unroll_explicit` 包装。
  其他 AttrStmt 仍拒绝；不从该包装或嵌套循环内部抽取指针。
- 条件表达式与分支均检查 opaque/未知调用；条件内部或嵌套 payload 循环内的
  remote-address producer 会阻止本层外提。

原有 ABI-ready/binding、descriptor 不可变性、peer 循环不变性和 zero-trip guard 全部保留。
仍不跨越 wait/signal/fence/barrier、任意未知调用、serial 嵌套循环、allocation 等边界。
没有把 `remote_address` 注册成 pure，也没有删除 helper 的运行时检查。

## 生成代码的变化

以下为结构示意，实际 HIP 在证据目录中：

```cpp
for (/* 原 owned tile 循环 */) {
    wait_tile(); fence_sys(); __syncthreads();
    // 四个 partial peer 指针 + 四个 output peer 指针
    void* p0 = peer(partial_view, 0);  // p1、p2、p3 同理
    void* o0 = peer(output_view, rank);  // 其余三个输出 PE 同理
    for (int chunk = 0; chunk < 16; ++chunk) {
        // 原来的四次 half8 load、FP32 sum、四次 half8 store
    }
    fence_sys(); __syncthreads();
}
```

四个尺寸均为每线程、每 owned tile 从 `8×16=128` 次源码级地址解析降为 8 次。
不是每个 CTA 只解析一次，也没有将 payload 读取提前到 ready 之前。
四个 remote load 和四个 remote store 仍为 `uint4`/16-byte 向量操作。

## 验证与限制

- `tilelang_module` 已重新 build，路径隔离检查通过，使用 old 的 Python、TVM 和原生库。
- 指针外提正反例及 AR 调度/codegen：26 项通过；索引化简、向量化、persistent resource、
  M2 独立接口回归：31 项通过。
- AR 四个尺寸的完整 dyadic partial/output pre/post 检查全部通过。
  AR 2048 continuous 补充检查通过，3 warmup / 5 repeats，原容差不变。
- RS 4096 matched dyadic 四卡补充检查通过。
- AR/RS 各四个尺寸在该 pass 前后的 GEMM compute branch 均结构相等。
  RS 四个尺寸的新旧 pass 输入及输出 TIR 文本也分别完全相同。
- 5 份原生 GEMM + 2 份未标记 persistent GEMM 的完整 HIP 与修改前逐字一致。

一次更强的“RS 最终 HIP 必须逐字相同”审计在 4096 上失败，失败日志保留，未删除断言。
独立比较确认该案例在外提 pass 前后 TIR 均与旧版相同，差异出现在后续 A/B LDS 排列
（0 与 32768 byte 的位置交换）。原外提验收中也记录过共享内存分配不确定性。
因此不能声称所有融合算子的最终 HIP 都逐字不变，也不将下表全部收益严格归因于地址解析次数。
本轮没有为消除文本差异修改共享内存分配器。

RS 4096 continuous 测试未通过 `rtol=atol=1e-3`：rank 2 的 4194304 个输出中有
4 个元素超出容差，违规元素最大绝对误差 0.001953。使用保存的**修改前**动态库独立复测，
同一 rank 也出现 4 个超差元素，报告的误差相同；这与历史报告的既有问题一致。
未放宽容差、未修改 reference，不将 dyadic 通过表述为任意随机输入均正确。
旧库首次复测因隔离目录缺少既有 Cython wrapper 导入失败，补齐同一 old 构建的 wrapper
后重试；两个日志均保留，只有第二次才是有效数值复测。

16384 新 HIP 的 ISA 仍为 `NumSgprs=104, NumVgprs=256, ScratchSize=200`，
与外提前相同。外提解决重复 peer 解析，不意味着 GEMM spill/reload 已解决。

## 同节点性能

Slurm job 10061，dcu0 四卡，DTK 26.04；512 threads/CTA、80 CTAs。
四个尺寸的 compute/comm blocks 分别为 32/48、56/24、64/16、72/8，保持既有参数。
每实现 30 warmup / 30 repeats，逐轮先取四 PE kernel event 最大值，再取中位数；单位 ms。
每组 TileLang/手写参考共享输入和 allocation，交替测量。手写代码未修改。

| M=N, K | 外提前 TileLang | 外提后 TileLang | 耗时下降 | 同轮手写参考 | 比手写多耗时 |
|---|---:|---:|---:|---:|---:|
| 2048, 512 | 0.233600 | 0.184400 | 21.1% | 0.176720 | 4.3% |
| 4096, 1024 | 0.636799 | 0.553520 | 13.1% | 0.532320 | 4.0% |
| 8192, 2048 | 2.330799 | 1.984719 | 14.8% | 1.935999 | 2.5% |
| 16384, 4096 | 15.433276 | 10.901673 | 29.4% | 10.222870 | 6.6% |

这是本次端到端实测，不是固定 HIP 的单变量归因实验；不承诺其他输入/拓扑上的固定收益。

## 复现与证据

在计算节点启动新进程：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
python examples/distributed/gemm_ar_prototype/gemm_allreduce_scheduled.py \
  --m 4096 --n 4096 --k 1024 --output build-old/ar-hoisted-inspect
python -m pytest -q testing/python/transform/test_hoist_dcu_ipc_pointers.py
```

四卡性能复现沿用 `examples/distributed/gemm_ar_prototype/run_compare.sh`。
runner 关闭 kernel cache，确保真实重新编译；未删除用户缓存，已加载旧库的进程需要重启。

证据根目录 `build-old/ipc-ar-hoist-20261012/`：

- `before/`：原 pass、原动态库、基线 HIP/TIR、性能与原生 GEMM 基线。
- `after/perf/`：四尺寸各 PE JSON、完整验证日志、HIP、rank-max 汇总。
- `after/audit-complete/`：八个融合案例的真实 pass 前后 TIR 与 compute 分支结构检查。
- `after-audit.log`：保留的 RS 最终 HIP 逐字比较失败记录；`compare_ir.py` 为独立 IR 比较。
- `after/native/manifest.json`：七份原生/未标记 HIP 的逐字比较。
- `targeted-tests.log`、`regression-tests.log`、`ar-continuous.log`、`rs-dyadic.log`。
- `rs-smoke.log`、`rs-smoke-before-r2.log`：新旧库 continuous RS 数值问题对照。
- `after-m16384.s`：实际优化后 HIP 的设备汇编。

最终库 `build-old/lib/libtilelang_module.so` SHA256：
`b316c139f07fee31e6b2cf444dd276bc342f284c480f520715270ae55aa1f3d8`。
本轮停止在 AR 指针外提扩展，不继续 M3 或其他优化。
