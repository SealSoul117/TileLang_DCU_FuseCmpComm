# GEMM + ReduceScatter（四 PE 有界调度候选与历史诊断）

## 直接生成当前优化版 HIP

在计算节点上选择 old 环境后运行：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
python examples/distributed/gemm_rs_prototype/gemm_reduce_scatter_scheduled.py
```

默认生成 2048/2048/512 的源码，输出到
`examples/distributed/gemm_rs_prototype/generated/scheduled/gemm_reduce_scatter_scheduled.hip`。
入口使用 `PassContext` 显式开启 `tl.enable_dcu_ipc_vectorize`，结合后端现有的
指针外提和索引简化生成 half8 通信；这是源码生成，不是四卡运行或性能测试。
`lower()` 直接重新生成源码，不读取 JIT kernel 缓存。原有输出目录中的同名文件会更新，
如需保留旧结果请用 `--output` 指定新目录。

总 Block 为 80 且未指定 `--comm-blocks` 时，四个已测 M/N/K 尺寸分别使用：
2048/2048/512 → 48/32，4096/4096/1024 → 48/32，8192/8192/2048 → 72/8，
16384/16384/4096 → 76/4（计算/通信 Block）。其他尺寸仍默认 32 个通信 Block，
不是自动调优；显式 `--comm-blocks` 始终优先。

```bash
python examples/distributed/gemm_rs_prototype/gemm_reduce_scatter_scheduled.py \
  --m 8192 --n 8192 --k 2048 --output /tmp/gemm-rs-8192-half8
```

这里只调整该示例的生成入口；全局向量化开关、`build_bound()` 的默认参数及 benchmark
入口均未改变。自行调用 `tilelang.compile()` 时仍需传入
`pass_configs={"tl.enable_dcu_ipc_vectorize": True}`。
运行生成代码时，仍须遵守四 PE IPC 协议及实际地址 16 字节对齐、有效范围、输出不与源
重叠的契约；该优化不加入运行时对齐检查。以上配置是目前已验证的配置，不是任意尺寸的最优保证。

2026-10-06 地址路径对照见 [GEMM_ADDRESS_PATH_AUDIT.md](GEMM_ADDRESS_PATH_AUDIT.md)：
历史诊断发现，未标记 persistent 路径与原生 GEMM 的 idxen、tile resource base、local row
分解并不相同。现在显式 opt-in 的 tile-resource 分解已在 marker 路径对齐；使用范围、拒绝条件和 CPU 验证见
[PERSISTENT_RESOURCE_BACKEND.md](PERSISTENT_RESOURCE_BACKEND.md)。通信汇编绕行未改。

同日后续的 persistent 专用 lane 清理已验收：通过资源地址检查的 gfx936 一维 512 线程
tile loop 不再调用 native lane helper；普通/未标记 GEMM 保持原行为。四卡正确性通过，
4096 静态 mbcnt 从 337 降至 0。范围、同机性能对照及已知限制见上述报告的 lane 清理节。

本轮当前入口是 `gemm_reduce_scatter_scheduled.py` 与 benchmark 的
`--schedule matched`：它以 TileLang PrimFunc 生成固定 80 CTA 的 GEMM+RS 候选，并包含本轮
为实际地址错误所作的 DCU resource-lowering 修复，以及上述已验证 scope 内的普通 lane
坐标生成。`InjectWarpIdx` 的 warp 路径保留，未启用该 scope 的 DCU 路径仍使用
`get_lane_idx` 的 native wave64 保护；GEMM+RS 的通信循环则保留历史 `opaque_thread_id(tx)` 后的
512-thread 线性元素遍历，避免把通信坐标交给同一套代数重写。未恢复 launch-bounds 或 DS 指令修改。它是有界诊断，不是 M3
或通用 persistent 实现。

此前 wave/lane 版本的 single-CTA 顺序诊断路径已在四 PE `4096x4096x1024` dyadic full pre/post 检查
通过。它仍只用于对照、生成 HIP 和失败复现；不应将 single 的结果解读为 matched 调度的验证，
也不应据此作性能结论。

## 2026-10-03：恢复汇编通信索引（当前版本）

已撤回上轮未通过的 serial 循环自动识别、logical-tid 重构和 lane 缓存。
`InjectWarpIdx` 恢复到改动前的版本；重建后的 `libtilelang_module.so` 与
`build-old/lane-step1/libtilelang_module.before.so` 的 SHA-256 相同。
已有 GEMM sublane 保护、四级 K 流水、80 CTA 的 48 compute / 32 comm 调度和
ready/completion 协议均保留。

三个 Python 入口（scheduled、single、flat）的通信改为：

```python
comm_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
for linear in T.serial(comm_tx, TILE_M * TILE_N, THREADS):
    row = linear // TILE_N
    col = linear % TILE_N
    # 原有四 PE 读取、FP32 求和、FP16 写回
```

`src/tl_templates/dcu_hip/ipc.hpp` 中的 helper 使用此前验证过的
`asm volatile("" : "+v"(tid))`。它不是内存屏障，也不替代通信同步；
`call_extern` 使该值对 TIR 不透明，空汇编读写约束阻止 LLVM 沿用原始线程编号的范围推断。
不透明值在元素循环前取得，循环内不重新调用 helper。GEMM 仍使用原有线程索引路径。

本轮 Slurm job 9594，dcu1，四卡 gfx936，使用 old 的环境、模板和重建二进制：

- 相关编译/源码回归：12 passed。
- `thread_index_step1.py --opaque`：65536 和 1003 元素，32/64 位索引均通过；
  所有元素正确，512 个尾部 sentinel 无损坏。此前 raw 索引的 1003 场景有 21 个错误元素
  和 21 个越界 sentinel 写入，原失败证据保留在 `build-old/lane-step1/`。
- 普通 GEMM：`512x768x256`、`1024x512x512` 通过原 `rtol=atol=1e-3` 检查。
- matched：`2048x2048x512`、`4096x4096x1024` 四 PE 的 dyadic full partial/RS
  pre/post 均通过；前者的 continuous full pre/post 也通过原容差。
- single 和 flat：`1024x512x256` 四 PE dyadic full partial/RS 通过。

matched 的记录值为 0.706719 ms（2048）和 1.477678 ms（4096）：5 warmup、30 repeat，
每轮先取四 PE 最大值，再取中位数，只计 HIP event kernel 时间。这是恢复后的当前基线，
不代表已实现指针外提、边界消除、向量化或 lane 缓存。4096 continuous 本轮未重新验收，
此前与手写共同出现的严格容差边界问题仍保留，不宣称已修复。

本轮 HIP、JSON、日志和路径隔离证据在 `build-old/asm-comm-restore/`。
在已分配的计算节点上可复现：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
python scripts/dev_old/thread_index_step1.py --opaque --output build-old/asm-index-check
mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python examples/distributed/gemm_rs_prototype/benchmark_gemm_reduce_scatter.py \
  --implementation tilelang --schedule matched --m 4096 --n 4096 --k 1024 \
  --warmup 5 --repeats 30 --json-dir build-old/asm-rs-check \
  --export-hip build-old/asm-rs-check/hip
```

## 此前 wave/lane 版本验收结果（历史记录）

下表均为同机四 PE `gfx936`、matched 的固定 80 CTA（48 compute / 32 comm）配置；full
pre/post 表示本地 tile-major FP16 partial 与连续 row-shard ReduceScatter output 均按 runner
原有严格检查。时间为 dyadic 输入、5 warmup + 30 repeat、每轮先取四 PE 最大值再取 median，
仅记录本轮测量，不能外推为通用性能结论。

| 路径与 shape | dyadic full pre/post | continuous full pre/post | dyadic median（ms） |
| --- | --- | --- | --- |
| matched `2048x2048x512` | 四 PE 通过 | 四 PE 通过 | TileLang `0.7979989946`；手写 `0.1222399995` |
| matched `4096x4096x1024` | 四 PE 通过 | partial 四 PE 通过；RS 不通过原有 `rtol=atol=1e-3` | TileLang `1.6288779974`；手写 `0.3422390074` |
| single `4096x4096x1024` | 四 PE 通过 | 未在此处宣称 | 未作为性能路径测量 |

`4096` continuous 的严格 RS 失败没有放宽容差：rank 0 有 2 个、rank 2 有 4 个元素超过该
阈值。手写基线在同一输入下出现相同的失败；对 rank 0/2 分别完整比较 16,777,216 个 partial
元素与 4,194,304 个 RS shard 元素，TileLang 与手写输出均 bitwise 相同。因此它是当前
CPU-reference 容差边界的已知事实，而不是 matched continuous 全通过的结论。

源码回归为 28 passed；普通 GEMM 两个尺寸及 `56 compute / 24 comm` 的 4096 dyadic
配置也已通过。仅使用 `old/build-old` 重建产物，路径隔离检查通过。HIP、JSON、连续输入 failure dump 和手写对照均保存在
`build-old/rs-schedule-audit/`（例如 `tl-cont-dump/`、`hand-cont-dump/`、
`lane-fixed2048/`、`lane-fixed4096/`）；这些为本树内审计证据，不是可移植性保证。

以下 single/flat 的“推荐”、数值结果、失败分析和“没有性能基准”等内容均为 **2026-10-01
历史诊断记录**。它们保留为失败证据，不描述 matched，也不构成 matched 的正确性或性能结论。

## 2026-10-01 single/flat 诊断记录

当时推荐的是 [直接 tile-grid 版本](gemm_reduce_scatter_tile_grid.py)：一个 CTA 计算一个
tile，默认 `M=1024,N=256,K=256` 的四 PE dyadic 运行中，四个 rank 的 local partial 与
ReduceScatter output 均为 bitwise CPU oracle match。它仅在本机四张 gfx936、最多八个
CTA/PE 的资源前提下验证。已通过的 shape 是 `M=1024,N=256,K=256` 与
`M=2048,N=256,K=512`（均为 dyadic、partial 与 RS bitwise）；后者也通过 continuous
`rtol=atol=1e-3`。这不是任意 `N=256` 或任意 `K` 的承诺。`M=1024,N=512,K=256` 的
ReduceScatter 已实测失败，见下文。
原 `gemm_reduce_scatter.py` 的单 CTA 顺序遍历版本仍保留作生成 HIP 与失败复现。上述是
2026-10-01 旧运行的数值失败记录；本轮 native lane 修复后，它的四 PE
`4096x4096x1024` dyadic full pre/post 已通过，但仍不是推荐入口或性能基线。

每个 PE 都持有本地 `A[M,K]`、`B[K,N]`。FP16 输入经 `T.gemm` 在 FP32 fragment 中累加，
每个 PE 将每个 tile 独立写入自己的 FP16 `partial[tile,256,256]` 槽。每个线程执行 system fence，CTA
barrier 后 thread 0 用 release signal 向该 tile 的连续行 shard owner 发布
`ready[tile, source]`。

owner 是连续行切片的拥有者：默认 `M=1024` 时四个 tile row 分别归 PE 0--3；例如
`M=2048` 时 PE 0 得到 row tile 0--1、PE 1 得到 2--3，依此类推。owner 的 thread 0
acquire-wait 四个 source ready 槽，CTA barrier 后所有线程从四个 remote partial 读取
同一元素，先转 FP32 求和，最后转 FP16 写本 PE 的 `[M/4,N]` row-major output。

`partial` 的每个 tile 都有独立槽，直到 host collective 完成之前绝不覆盖。因此本内核
没有 ACK，也不 reset ready flags；host 只能在全部 PE 的 kernel 完成、没有远端 reader
以后，才可统一释放或复用 IPC allocation。

两种调度的前提不同，不能混用。保留的顺序版每 PE 只有一个 CTA，并按相同 tile 次序推进；
因此 owner 等待同一 tile 的 publication 不需要假设另一个 owner CTA 已调度。推荐 flat
tile-grid 版每 PE 至多有八个 CTA、每 CTA 一个 tile；它依赖本机 gfx936 的资源足以让这批
CTA 同时驻留，才可保证每个等待 owner 都有对应 producer CTA 在运行。它不保证超额 grid、
任意 CTA 数或 persistent 调度的活性。

## 2026-10-02：逐项 HIP 性能归因实验

本轮是 **HIP 诊断实验，不是已落地的 TileLang 后端优化**。原始 scheduled.py、
compiler、模板和旧结果均未被诊断变体替换。`make_hip_ablation.py` 从上一轮已检查的
HIP 导出受限变体；`--diagnostic-hip` 明确绕过 TileLang lowering，只编译独立的
C ABI launch wrapper。不能把这些手工变换后的 HIP 称为 TileLang 自动生成结果。

Slurm job 9568，dcu2，四 PE/gfx936；80 CTA（48 compute + 32 comm）、512 threads、
64 KiB 动态 LDS。所有变体保持原有 tile ownership/compute order、FP16 partial、
FP32 reduction、FP16 output、ready publication、reader-safe handshake 和 host reset。
分配/编译/初始化/reset/host sync 不计入 HIP event；5 warmup + 30 repeats，
每轮先取四 PE 最大耗时，再对 30 轮取中位数。手写基线也在同节点重测。

首轮 dyadic 数据（单位 ms）：

| 变体 | 2048×2048×512 | 4096×4096×1024 |
|---|---:|---:|
| v0 原始生成 HIP | 0.786239 | 1.624559 |
| v1 外提 payload peer 指针及 descriptor 检查 | 0.434560 | 1.045919 |
| v1b 标量不变，仅简化有效域内的索引/边界判断 | 0.255680 | 0.839439 |
| v2 通信循环对齐手写 uint4 路径 | 0.232960 | 0.825200 |
| v3 在 v2 上只计算一次 native lane | 0.162080 | 0.457600 |
| v4 在 v3 上恢复 A 的 idxen | 0.162320 | 0.416800 |
| v5 在 v4 上改成每 tile 的 A/B resource base | 0.162480 | 0.418640 |
| 独立分支：v0 只缓存 lane | 0.715119 | 1.423118 |
| 独立控制：v2 换为完整手写 compute_tile | 0.149120 | 0.369760 |
| 完整手写基线 | 0.122640 | 0.344240 |

进一步将 signal 的 peer 地址解析外提，并一次检查整个有效 flag span：
v4 的 2048/4096 分别为 0.160160/0.412800 ms；reference_compute 分别为
0.148240/0.365599 ms。此处收益较小，不能用它解释最初的倍数差距。

解释边界：

- v1b 是 v1 的分支；v2 从 v1 构造，但同时实现 v1b 的简化和向量通信。
  所以 v1→v2 **不是纯向量化收益**。v1b→v2 更接近向量访存及其线程布局的额外收益。
- v3→v4 对 4096 有收益，对 2048 无明显收益；v5 没有显示明显收益。
- lane 的收益受通信是否优化影响，不能把独立分支收益直接相加。
- 338 条静态 mbcnt 在 v3 中降为 2 条，但 scratch/private segment 反而增加。
  性能依然明显提升，因此不能仅以 spill count 判定主瓶颈，静态条数也不是动态执行次数。
- reference_compute 同时替换整个 GEMM 实现，只是剩余差距的控制实验，不能归因给某一条指令。
- 向量路径复现手写从零累加的方式；不承诺 NaN/带符号零与原标量路径严格 IEEE 逐位等价。

所有上表变体均在四 PE 上通过计时前后全量 dyadic partial 和 RS 的逐位检查。
2048 的 v3 连续随机输入通过原 rtol=atol=1e-3；4096 的 v4 保留原容差失败：
rank0 的 2 个、rank2 的 4 个 RS 元素。保存的 rank0/rank2 全量 partial 和 RS shard
与上一轮手写 continuous dump 逐位相同（差异数 0）；未放宽容差、未将失败计入性能表。

原始源码、ISA、四 rank JSON、日志在 `build-old/rs-ablation/`。
`summarize_hip_ablation.py` 会校验四 rank 的 pre/post 状态，再聚合样本。
诊断入口的 descriptor/payload 参数顺序有独立的 mock ABI 单测；默认 TileLang 路径不变。

复测异常必须保留：`*_repeat2` 的六组数据仍在原目录和 summary 中，不用于上表归因。
第二轮出现手写版也同时明显变慢（2048: 0.276160 ms，4096: 3.558398 ms）。
停止本测试后，rocm-smi 仍显示四卡全部 100% 利用率，并列出 root 所有的 Python 3.11
KFD 进程（各约 25 GiB 显存）；这些进程于计算节点时间 10:41:36 UTC 启动，早于异常
复测，而第一轮主实验和信号地址补充实验在此前完成。证据保存在
`build-old/rs-ablation/background-load.txt`。不能把异常复测解释成代码退化或静默丢弃；
第一轮是单批 30 次结果，小幅差异仍需在无并发负载时复测。没有终止外部进程。

本轮原有 scheduled/source 回归和新 ABI 单测共 6 项通过；old 路径隔离检查通过。
Slurm 9568 已释放。没有修改生产 compiler/模板，没有自动 commit。

复现示例（已申请四卡，工作目录为 old）：

```bash
source scripts/dev_old/env.sh
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
python examples/distributed/gemm_rs_prototype/make_hip_ablation.py \
  --source build-old/rs-schedule-audit/lane-fixed4096-hip/tilelang-matched-m4096-n4096-k1024-rank0.hip \
  --m 4096 --n 4096 --k 1024 --output build-old/rs-ablation/4096/src
mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python examples/distributed/gemm_rs_prototype/benchmark_gemm_reduce_scatter.py \
    --implementation tilelang --schedule matched \
    --diagnostic-hip "$PWD/build-old/rs-ablation/4096/src/v4_idxen.hip" \
    --m 4096 --n 4096 --k 1024 --warmup 5 --repeats 30 \
    --json-dir build-old/rs-ablation/4096/v4_idxen
python examples/distributed/gemm_rs_prototype/summarize_hip_ablation.py \
  build-old/rs-ablation --output build-old/rs-ablation/summary.json
```

## matched 有界调度候选

`gemm_reduce_scatter_scheduled.py` 是和手写参考调度对齐的独立候选：固定 80 个 CTA，
其中 48 个计算 CTA 交错四个连续 row-shard owner 的 tile，32 个通信 CTA 以每线程一次
`opaque_thread_id(tx)` 和 512-thread 线性元素遍历执行 owner-local 标量 ReduceScatter。
它仍是有界诊断，不是 M3 或通用 persistent 实现；RS
没有改写成手写参考的 `uint4` 向量路径。

它只面向同机四 PE `gfx936`，`K >= 64` 且为 64 的倍数。matched runner 将 A/B/out 分配为
本地 DeviceDefault，将 partial/flags 分配为 FineGrained IPC；flags 包含 tile-ready、peer
completion 和 comm-CTA completion 三段。启动前会查询实际 CU 数和
`512 threads / 64 KiB` 的 kernel occupancy，不能让全部 80 CTA 同时驻留即直接失败，不能以
超额 grid 等待代替该检查。每轮仍须在 host collective 完成后才 reset flags/复用 workspace。

只导出 source：

```bash
python examples/distributed/gemm_rs_prototype/gemm_reduce_scatter_scheduled.py \
  --m 2048 --n 2048 --k 512 --total-blocks 80 --comm-blocks 32 \
  --output examples/distributed/gemm_rs_prototype/generated/scheduled
```

四 PE runner/benchmark（在已分配 DCU 节点，使用 old 的隔离环境）：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python examples/distributed/gemm_rs_prototype/benchmark_gemm_reduce_scatter.py \
    --implementation tilelang --schedule matched --m 2048 --n 2048 --k 512 \
    --total-blocks 80 --comm-blocks 32 --input-mode dyadic
```

顺序 single-CTA 原型继续保留为原始诊断/失败复现路径；不要将它的结果混同为 matched 调度的
验证或性能结论。

## 2026-10-01 single/flat：直接 tile-grid 的 HIP 与数值检查

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
python examples/distributed/gemm_rs_prototype/gemm_reduce_scatter_tile_grid.py \
  --output examples/distributed/gemm_rs_prototype/generated/tile_grid_flat
```

该命令只执行 source-only lowering：

```python
tilelang.lower(bound, target="hip -mcpu=gfx936",
               enable_device_compile=False, enable_host_codegen=False)
```

它会在本目录 `generated/tile_grid_flat/` 写入原始 TIR、绑定 IPC descriptor 后的 TIR、lower 后
device TIR 和生成的 `gemm_reduce_scatter_tile_grid.hip`。验收产物均由编译器生成，未手改；
若要手工实验 HIP，建议复制到另一个文件并重新验证，保留原产物作对照。
已数值通过的多 tile row-shard 形状为：

```bash
python examples/distributed/gemm_rs_prototype/gemm_reduce_scatter_tile_grid.py --m 2048 --n 256 --k 512 \
  --output examples/distributed/gemm_rs_prototype/generated/tile_grid_flat_m2048_n256_k512
```

在有 DCU 的节点，才可追加 `--compile` 请求真实 ctypes/HIP 编译：

```bash
python examples/distributed/gemm_rs_prototype/gemm_reduce_scatter_tile_grid.py --compile \
  --output examples/distributed/gemm_rs_prototype/generated/tile_grid_flat
```

### 四 PE runner

先以一个交互 shell 申请本机四 DCU（本站资源名为 `dcu:bw1000`）：

```bash
srun --account=ncic --partition=bw1000 --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=64G --time=01:00:00 --gres=dcu:bw1000:4 --pty bash
```

在该 shell 内运行（不是裸 `srun --ntasks=4 python ...`）：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
timeout --signal=TERM 150s mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python examples/distributed/gemm_rs_prototype/run_gemm_reduce_scatter.py \
    --flat-tile-grid --input-mode dyadic
```

这个 runner 的默认 shape 为 `M=1024,N=256,K=256`。它使用 rank 不同的 `[-8,8]/16` FP16
输入，先要求每个 rank 的 tile-major FP16 partial bitwise 等于 CPU oracle，再要求其连续
row shard bitwise 相等；每项同时打印最大绝对误差、NaN 数与 bit mismatch 数。通过日志是
`generated/run_flat_tile_grid_dyadic/launcher.log`，以及独立复测
`build-old/logs/gemm-rs-prototype-flat-default-run-review.log` 和
`build-old/logs/gemm-rs-prototype-flat-m2048-dyadic-review.log`。

## 保留的单 CTA 顺序版本

`gemm_reduce_scatter.py` 保持原始“每 PE 一个 CTA 顺序遍历 tile”的设计，用于观察它的真实
融合 HIP、source-only 和 `--compile`。默认命令仍是：

```bash
python examples/distributed/gemm_rs_prototype/gemm_reduce_scatter.py
```

它会在所选 `generated/` 子目录保存 raw/bound/device TIR 与 HIP。该版本的数值失败见
“当前失败分析”；不要把它当作推荐 run 命令。

`--sync-copy` 是仅为诊断提供的无流水线、每步等待变体：每个 K block 使用 `T.serial`，
并在 copy 与 GEMM 前后 CTA barrier。HIP 后端仍可将这些 copy 降为 `cp_async_gs`，但每步
有 `cp_async_wait<0> + wave_barrier + __syncthreads`，不能把它称为“最终 HIP 不含 async”。
默认仍是官方 DCU GEMM 同款的 512 threads、
4-stage `T.Pipelined`、tile `256x256`、`block_K=16` 路径。这里的 512-thread launch
沿用本机 gfx936/DTK 的既有 `--gpu-max-threads-per-block=256` 工具链配置；它不是通用 HIP
可移植性声明，也只应在本机 profile 验证。

## 已知限制

- 这是 bounded 单 kernel 融合演示。顺序版每 PE 一个 CTA；flat 版每 PE 至多八个 CTA，
  依赖本机足够驻留资源，不保证超额 grid 的活性。两者均没有 compute/communication 动态
  角色、work queue、ACK/epoch 复用，未设计或验证计算通信重叠，也未性能调优，不能代表最终性能。
- 仅针对单机四 PE、`gfx936`、FineGrained IPC `partial` 与 `ready` 的 M2 已验证组合。
  不承诺跨节点、任意 GPU、任意 PE 数或完整生命周期 ABI。
- 远端 partial 保持 FP16；remote half BufferLoad 会通过真实 source-only lowering 检查。
  每个远端 load 先单独写入 local FP16 标量，再转 FP32 求和，避免把 remote alias 包在
  Cast 或直接放进一个 Add（现有 normalizer 不支持这些表达式）。如果该 lowering 被当前旧树拒绝，应保留
  报错并报告，而不是修改生产接口或悄悄改为 FP32 partial。
- 提供 correctness runner，但没有性能基准；其结果不能解释为性能数据。
- flat tile-grid 没有 compute/communication 动态角色、ACK/epoch 复用或 persistent work queue。
  `partial` 每 tile 独立且 host `ctx.sync()` 后才回读/close；它只证明这一轮 bounded launch，
  不代表 flags/workspace 可在 kernel 内复用。
- “最多八 CTA/PE”只是本机资源边界，并不表示所有 tile shape 都已正确。尤其 `N=512`
  虽仍在该数目内，却已观察到 RS 错误；不要将 flat tile-grid 当作通用 `N`/shape 实现。

## 当前失败分析（保留，不掩盖）

默认 pipelined 版本已完成 source-only lowering 和实际 HIP/LLVM 编译，但首次四 PE
`--input-mode dyadic` 数值运行失败在**本地 partial**核验：四个 rank 都出现 NaN，且有
57,109--84,683 个 bit mismatch；因此尚未把它称为 GEMM+ReduceScatter 正确，也没有继续将
输出 mismatch 归咎于 IPC。失败日志保留于
`generated/run_default_dyadic/run.log`。

当前只读代码审计的强候选解释是 outer serial tile loop 会被 DCU `IsDCUMainLoop` 递归识别为
主 GEMM loop，导致后续 async-load fence 注入进入 main-loop phase 太早；与纯 GEMM HIP 相比，
该失败 HIP 的 GEMM prologue 缺少预期 `wave_barrier()/cp_async_wait<4>()`。这不是已证明的
编译器结论，本原型不修改生产 pass；`--sync-copy` 仅用于验证这个示例级 schedule 假设。

该 `--sync-copy` 变体也实际执行但仍在 local partial 验证失败（无 NaN、最大绝对误差
9.39--10.17、每 rank 261,924--261,955 个 bit mismatch），日志为
`generated/run_default_dyadic_sync_copy/run.log`。故它也不能作为正确性原型或 RS 验证。

早期 `gemm_reduce_scatter.py --tile-grid`（extent-one outer loop）在真实 source-only lowering
的 `Simplify` pass 报内部 `owner` range 更新冲突，日志为
`generated/default_tile_grid/source-only.log`。推荐 flat 版本不是这一技巧：它是独立文件，
直接用二维 `T.Kernel(tile_cols, tile_rows)`，没有 outer tile `For`。

flat tile-grid 的 `M=1024,N=512,K=256` dyadic 独立复测中，四个 rank 的 **partial 均
bitwise 正确**，但 ReduceScatter 全部失败（最大绝对误差 16.25--18.91、47,851--63,971 个
bit mismatch）。日志为 `build-old/logs/gemm-rs-prototype-flat-n512-run-review.log`。根因尚未
定位；本原型不修改编译器或加入示例级 workaround。
