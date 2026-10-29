# GEMM + AllReduce TileLang 原型

这是一个固定四 PE 的 GEMM+AllReduce 原型，对齐手写 `fused_vector` 的几何和协议：80 个 CTA、每 CTA 512 线程、`256x256x16` GEMM tile、`tile % 4` owner、FP16 tile-major partial，以及向四个 PE 的完整 row-major 输出复制。它是本机 gfx936 上的有界实验，不是通用 collective 或自动调优实现。

`gemm_allreduce_scheduled.py` 是 TileLang 算子；`benchmark_gemm_allreduce.py` 在同一份分配、输入、reset 和计时条件下运行 TileLang、手写参考或二者交替的比较。通信保持 PE0、PE1、PE2、PE3 的 FP32 加法顺序，最后转换回 FP16。

每个 PE 输入各自的 `A_r[M,K]`、`B_r[K,N]`，先将 FP32 GEMM accumulator 转成 FP16 partial，再对四份 partial 做 FP32 SUM 并转回 FP16。每个 PE 最终都得到完整的 `[M,N]` 输出，不是 ReduceScatter 的四分之一输出。

## 环境隔离与源码生成

只能使用 old 树的环境，避免 develop checkout 的 Python、TVM 或编译器混入：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
```

在分配到的计算节点生成 2048 的 TileLang HIP/TIR：

```bash
python examples/distributed/gemm_ar_prototype/gemm_allreduce_scheduled.py \
  --m 2048 --n 2048 --k 512 --total-blocks 80 --comm-blocks 48 \
  --output build-old/gemm-ar/source2048
```

输出目录包含 `raw.tir`、`bound.tir`、`device.tir` 与
`gemm_allreduce_scheduled.hip`。这一步只生成源码，不等同于四卡正确性或性能结论。

当前默认的固定 80 CTA 分配如下（计算 CTA / 通信 CTA）：

| M=N | K | compute / comm |
| ---: | ---: | ---: |
| 2048 | 512 | 32 / 48 |
| 4096 | 1024 | 56 / 24 |
| 8192 | 2048 | 64 / 16 |
| 16384 | 4096 | 72 / 8 |

可用 `--comm-blocks` 在 1..79 内做实验，但它不是已验证的自动调优结果。算子固定 80 CTA、512 线程与 64 KiB LDS；runner 会查询实际 occupancy。若 80 个 CTA 不能同时驻留，直接失败，不能用超额 grid 等待来替代 full-residency 条件。

## 只运行 TileLang

下例在已经获得四 GPU 的 Slurm 分配内运行。每个 rank 写一份 JSON；使用
`summarize.py` 后的指标是每个 iteration 四 rank event 时间的最大值，再取 median。

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python examples/distributed/gemm_ar_prototype/benchmark_gemm_allreduce.py \
  --implementation tilelang --m 2048 --n 2048 --k 512 \
  --warmup 30 --repeats 30 --input-mode dyadic \
  --json-dir build-old/gemm-ar/tilelang-m2048 \
  --export-hip build-old/gemm-ar/tilelang-m2048/hip
python examples/distributed/gemm_ar_prototype/summarize.py \
  build-old/gemm-ar/tilelang-m2048
```

计时前、计时后都会 poison 完整 `partial` 与 `out`，重新 launch 后复制回完整 device 数据检查。默认 dyadic 输入要求 local tile-major partial 和完整 output 都与 CPU oracle **bitwise** 相同；连续随机输入可用 `--input-mode continuous` 作为补充检查。检查先仅作本地 device sync，随后才进行跨 PE sync，因此也覆盖 kernel 内的完成握手，而非用 host 同步掩盖漏写。

## 与手写参考同条件比较

先构建适配器；它保留手写 HIP 的原始实现，并只把既有入口接到同一 IPC allocation/stream ABI：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
bash examples/distributed/gemm_ar_prototype/build_reference.sh \
  /public/home/baoboyuan/dcu_fused_gemm/gemm_ar/v2 \
  build-old/gemm-ar/reference
```

随后在同一类四卡分配内执行比较。`run_compare.sh` 对每个 shape 用 30 warmup、30 repeat，TileLang 与手写变体按轮次反转顺序，且 pre/post 都分别完整校验：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
bash examples/distributed/gemm_ar_prototype/run_compare.sh \
  build-old/gemm-ar/compare build-old/gemm-ar/reference
```

该脚本默认跑四个表中 shape；可在命令末尾附加单个 `M`（例如 `2048`）缩小范围。原始每-rank HIP event samples 和 verification 记录都保存在 JSON，`summarize.py` 输出 rank-max median，不能只取某一 rank 的 median。

## 生成与后端边界

当前通信前端是 `for linear in T.serial(comm_tx, TILE_M * TILE_N, THREADS)`：四路标量 FP16 load、FP32 顺序求和、一次 FP16 舍入、四路标量 output push。2026-10-13 已扩展 `VectorizeDCUIPCCommunication`，由后端自动生成 half8 访存，不再由 Python 显式写 `T.vectorized(8)`。四 PE、512 threads、256×256 tile 是当前已验证的窄模式，不代表任意 AR 都会自动匹配。

`source_only` 默认开启该 PASS；`--scalar` 使用同一份前端并关闭 PASS。benchmark 必须通过 `tilelang.compile(..., pass_configs={"tl.enable_dcu_ipc_vectorize": True})` 将开关传入 JIT，而不是只设置外层 PassContext。计时前会检查真实 JIT HIP 是否包含四路 `uint4` load/store 和单次归约结果绑定；不满足就报错，不把标量结果当成向量性能。直接调用示例：

```python
_, bound = build_bound(4096, 4096, 1024)
kernel = tilelang.compile(
    bound, target="hip -mcpu=gfx936", execution_backend="ctypes",
    pass_configs={"tl.enable_dcu_ipc_vectorize": True},
)
```

以下 v4 说明和 2026-10-11/12 表格为显式 half8 前端的历史记录；当前版本与其通信协议、调度、归约顺序一致，只有访存分组交给了后端。

v4 在固定四 PE 的 rank 范围以及 `owned` 的范围处使用 `T.assume`。其中 `owned` 的提示**仅在** `for owned in T.serial(...)` 已进入的循环体内成立，不把空闲通信 CTA 假定成有工作。v4 source-only 结果表明四个 partial payload 的 `condval/zero-fill` guard 与四个远端 output store guard 都已消失；每个 chunk 内的 peer lookup 仍未外提到 tile 边界。

2048 和 16384 的 v4 ISA 都确认了四个 remote load 与四个 remote store 为 `dwordx4`。但 v4 分别使用 `NumSgprs=104`、`NumVgprs=256`、`ScratchSize=200`，手写 `fused_vector` 为 `52`、`256`、`0`；GEMM 循环可见 folded spill/reload。寄存器/SGPR 与 scratch 是明确的生成差异，但不能据此把全部性能差距归因为单一原因。

通信索引仍使用既有 `opaque_thread_id(tx)`，以保留 512-thread 线性遍历与 GEMM 索引路径的隔离。其底层空 asm 读写约束只阻止 LLVM 复用原始 thread id 的范围推断，不是内存屏障；可见性仍完全依赖 fence、barrier、wait 与 signal。`256x256` / 512-thread GEMM 路径沿用本机已有的 gfx936 lowering；参考库的构建也沿用 `--gpu-max-threads-per-block=256` 这一既有工具链绕行。这不是一般 HIP 可移植性或任意线程形状的承诺。

曾尝试用显式 typed `tl.remote_address(tl.region(...), peer)` 基址把 peer table 查询外提至 tile 边界。该形式能构造，但当前 TVM 版本在混合该显式基址与 `remote_view` 的规范化时崩溃；本原型没有为此修改通用 frontend 或 backend，保持已经通过原始 v3 检查的高层 `remote_view` 路径。

本原型没有修改 TileLang 后端、既有 ReduceScatter 实现或手写 HIP，也没有创建 commit。v5 曾只交换 compute/communication 分支的文本顺序；ISA 变为 `ScratchSize=216` 且无改善，已完全撤回，当前源码恢复为 v4。

## v4 首轮性能与验证（2026-10-11，外提扩展前）

下表均为四 PE rank-max median，单位 ms。它是当前 v4 的已测结果，不宣称为最终最优实现。

| M=N, K | TileLang v4 | 手写参考 |
| --- | ---: | ---: |
| 2048, 512 | 0.237040 | 0.179280 |
| 4096, 1024 | 0.645920 | 0.541120 |
| 8192, 2048 | 2.340078 | 1.956158 |
| 16384, 4096 | 15.481127 | 10.182834 |

独立验收共 9 项测试通过；四个 shape 的完整 dyadic partial/output pre/post 检查均通过。表中各 shape 均为每个实现 30 warmup、30 repeat；2048 的 continuous 补充检查使用 3 warmup、5 repeat，也在 `rtol=atol=1e-3` 下通过，不用于上表性能结论。每组比较的两个实现共享输入和 allocation，保持相同轮次与 full verification 条件。

本轮在 Slurm job 10058 / dcu0 上完成，环境路径检查为 old 树。结果保存在仓库内：

- `build-old/gemm-ar-20261011/final-v4/`：四个尺寸的原始日志、每 PE JSON、rank-max 汇总和真实 JIT 导出的 HIP。
- `build-old/gemm-ar-20261011/continuous2048-v4/`：连续随机输入的补充正确性检查。
- `build-old/gemm-ar-20261011/gemm-ar-final-tests.log`：最终 9 项测试结果。
- `build-old/gemm-ar-20261011/tilelang-v4-m{2048,16384}.s` 与 `handwritten-m{2048,16384}.s`：相同架构和优化选项下的指令对照。

该轮最终源码重新生成的 16384 HIP 与上表实测 JIT 的 HIP 逐字节一致。当时待查的是 GEMM spill/reload，以及窄指针外提 pass 未覆盖的 AR 通信形式；该轮未扩大后端修改范围。

## 2026-10-12：后端自动外提 AR peer 指针

上述“指针仍在 chunk 内”描述的是首轮状态。现已扩展已有 `HoistDCUIPCPointers`，本文件对应的 Python 算子无须修改。四个输入 peer 和四个输出 peer 的基址均在 ready 等待之后、chunk 循环之前解析；每线程、每 owned tile 的源码级解析次数从 128 降到 8，half8 访存与同步协议不变。

同一 dcu0 四卡、30 warmup / 30 repeats 的 rank-max median（ms）：

| M=N, K | 外提前 | 外提后 | 同轮手写参考 |
| --- | ---: | ---: | ---: |
| 2048, 512 | 0.233600 | 0.184400 | 0.176720 |
| 4096, 1024 | 0.636799 | 0.553520 | 0.532320 |
| 8192, 2048 | 2.330799 | 1.984719 | 1.935999 |
| 16384, 4096 | 15.433276 | 10.901673 | 10.222870 |

四尺寸完整 dyadic 检查通过；2048 continuous 补充检查通过。原生/未标记 GEMM 七份 HIP 逐字不变。16384 的 scratch 仍为 200 bytes，本轮未解决 spill/reload。部分融合 HIP 存在下游 A/B LDS 排列差异，因此以上是端到端实测，不是严格单变量归因。

实现边界、57 项测试、RS 既有 continuous 容差问题与复现证据见[外提扩展报告](../../../docs/compiler_internals/distributed_fusion_ipc_ar_pointer_hoist_review.md)。产物保存在 `build-old/ipc-ar-hoist-20261012/`；新进程重新编译即可使用更新后的 old 动态库。

## 2026-10-13：标量前端、PASS 自动 half8

当前 Python 无须手动写八元素分组。沿用相同参数与四卡测试方法，默认启用后端向量化的结果如下（ms）：

| M=N, K | 旧显式 half8 | PASS 自动 half8 | 同轮手写参考 |
| --- | ---: | ---: | ---: |
| 2048, 512 | 0.184719 | 0.184400 | 0.176239 |
| 4096, 1024 | 0.553360 | 0.554960 | 0.534000 |
| 8192, 2048 | 1.985678 | 1.980641 | 1.938320 |
| 16384, 4096 | 10.856307 | 10.880397 | 10.252638 |

性能基本持平，变化均在约 ±0.3% 内，不宣称稳定加速。四尺寸完整 dyadic 校验、2048 continuous 补充校验及 64 项测试通过。RS 四份生成 HIP、原生/未标记 GEMM 七份 HIP 与基线逐字相同。

注意 JIT 必须显式传 `pass_configs`，不能只设置外层 PassContext；当前 benchmark 已修正并核验实际生成代码。早期未正确传开关的标量误测保留在证据目录 `after/perf/`，不用于本表。最终有效产物在 `build-old/ipc-ar-vectorize-20261013/after-enabled/perf/`。详见[AR 自动向量化报告](../../../docs/compiler_internals/distributed_fusion_ipc_ar_vectorize_review.md)。
