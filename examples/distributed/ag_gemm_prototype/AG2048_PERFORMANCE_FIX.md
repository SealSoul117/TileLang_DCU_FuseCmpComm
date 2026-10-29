# AG + GEMM 2048/512/2048：小尺寸性能排查

日期：2026-10-13。对象是 AllGather + GEMM（AG），不是 GEMM + AllReduce。
本轮单节点四卡 BW1000/gfx936，DTK 26.04，Slurm job 10112；仅使用 old 树。

## 结论与修改范围

主要问题不是 compute/comm Block 比例，而是只有 16 个输出 tile 时的代码生成路径：

1. 旧前端在 tile 数不超过 compute Block 数时取消 persistent 标记，未进入已有的
   tile-local A/B 资源基址及 direct-lane 路径。
2. 仅补回该标记仍不够。`T.serial(block_idx, 16, 48)` 被生成成动态 0/1 次外层循环；
   HIP 编译器把大量 GEMM 输出地址计算移到计算之前，增加存活寄存器并产生严重 scratch spill。
3. 保持原 48 compute / 32 comm 分工，改为 `block_idx < 16` 条件下显式执行一次带标记
   的 GEMM tile。恒定一次循环可消除，实测机器码没有 scratch spill，性能明显恢复。

前端仍是 TileLang，通信仍由既有 pointer-hoist/half8 PASS 生成。没有替换成手写 HIP，
没有减少校验或同步，没有改为动态队列，也没有改变 tile=256×256、K=16、四级流水、
80 CTA、512 threads、64 KiB LDS、FP16 输入/输出及 FP32 累加。

后端最终只增加一个窄例外：显式 `tl.dcu_persistent_tile` 循环即使资源基址不依赖
循环变量，只要 **min 是常量 0、extent 是常量 1**，也可走原资源构造路径。
仍要求真实 GEMM K 循环、向量 async global-to-shared copy、合法资源基址及块一致控制流。
动态 0/1 范围推导不在最终实现中；排查中试过的词法范围分析已撤回。
未标记的原生 GEMM 不使用该例外；没有改 `InjectWarpIdx` 或引入新的 lane 保护。

## Block 扫描：为什么不是简单多分配通信 Block

以下是修复前前端，同一作业内依次运行，30 warmup / 30 样本。
统计为每个样本先取四 PE event 的最大值，再取中位数，单位 ms。

| comm | compute | 耗时 | 是否进入原 persistent 资源路径 |
|---:|---:|---:|---|
| 16 | 64 | 0.623120 | 否 |
| 32 | 48 | 0.613119 | 否 |
| 48 | 32 | 0.613599 | 否 |
| 64 | 16 | 0.623119 | 否 |
| 65 | 15 | 0.206320 | 是 |
| 72 | 8 | 0.222720 | 是 |

64→65 的跃变对应 `tile_count > compute_blocks` 条件翻转；此时某个 Block 才开始计算
第二个 tile，旧前端才添加 persistent 标记。不能把整个跃变解释为增加一个通信 Block。

## 消融和机器码证据

| 版本 | compute/comm | 耗时 ms | VGPR | ScratchSize |
|---|---:|---:|---:|---:|
| 原始前端（同分配交替第 2 轮） | 48/32 | 0.612800 | 256 | 1260 |
| 补资源路径，但保留动态 0/1 次循环（同轮） | 48/32 | 0.420960 | 256 | 1172 |
| 上述版本只增加 `block_idx < 16` 条件 | 48/32 | 0.421359 | 未单独统计 | 未单独统计 |
| 条件 + 显式恒定一次循环（诊断） | 48/32 | 0.144559 | 246 | 0 |
| 手写参考（同分配交替第 2 轮） | 48/32 | 0.136000 | 未单独统计 | 未单独统计 |

最终生产版本的独立 ISA 编译结果为 VGPR=245、SGPR=54、ScratchSize=0，
静态 Folded Spill/Reload 均为 0，见 `isa/final32.s`；上表恒定一次行是先前诊断变体。

中间版本与诊断版本是因果排查，不是最终稳定加速比。第一次交替测试有明显波动
（原始 0.628399、中间版本 0.522080、手写 0.141359 ms）；未将它作为稳定代表值。

`ScratchSize` 是 hipcc 机器码元数据中的每线程 scratch 大小（字节），不是通信字节数。
静态 spill/reload 计数不能直接换算成动态耗时。原始/中间/恒定一次的静态 Folded Spill
分别为 314/292/0；补资源路径消除了旧 `get_lane_idx`/mbcnt 计算，但仍保留大量 spill。

关键机器码对应关系：中间版本先从 kernel 参数 `0x20` 载入 `out` 到 `s[16:17]`，
随后在 GEMM 主循环前形成大量加到该指针上的地址，写入 scratch；恒定一次版本不再有
这些 spill。这支持“退化外层循环导致地址提前计算和寄存器存活期变长”的解释。
没有做逐条指令动态 stall profiling，因此不把全部差距精确归因到某条指令。

ISA 编译使用与真实 JIT 一致的主要参数：`-O3 --offload-arch=gfx936
--gpu-max-threads-per-block=256`，实际 launch 仍是经过校验的 512 threads。

## 数据与复现

### 最终生产版本，同分配交替对照

每轮独立启动四 rank，三种实现复用同一组数据分配，按样本交替反转执行顺序。
每种实现均为 30 warmup / 30 样本，统计仍是 `median_iteration(max_PE(event_ms))`。

| 独立轮次 | 原始 TileLang ms | 修复后 TileLang ms | 手写 ms | 修复后比手写多 |
|---|---:|---:|---:|---:|
| final-paired1 | 0.612800 | 0.137199 | 0.135760 | 1.06% |
| final-paired2 | 0.612719 | 0.137439 | 0.135520 | 1.42% |

修复前后加速约 **4.46×**，耗时降低约 **77.6%**。这两轮可复现了接近手写的结果，
不再使用旧的约 0.616 ms 描述当前 2048 默认版本。其余尺寸的历史计时没有在这里替换成
不同样本数量的短测数据。

原始目录：[本轮 artifacts](../../../build-old/ag2048-tuning-20261013/)。

- `c16/`～`c72/`：旧前端 Block 扫描的 HIP 和四 rank JSON。
- `paired1/`、`paired2/`：原始/中间修复/手写三方同分配交替对照。
- `idle/`、`constant-one/`：两个控制流消融；后者 30 warmup / 30 样本。
- `isa/c32.s`、`fixed32.s`、`c65.s`、`constant-one.s`：对应机器码。
- `before/`：修改前源码、二进制和普通 GEMM 基线。
- `compare_same_alloc.py`、`handwritten_adapter.cpp`：独立同分配对照入口。
- `final-paired1/`、`final-paired2/`：最终生产前端和新库的两轮同分配对照，含真实 JIT HIP。

所有计时都只覆盖融合 GPU 算子；初始化、flags reset、MPI barrier 和 CPU 校验在计时外。
手写对照使用 `dcu_fused_gemm/ag_gemm/fused_ag_gemm_m2048_n512_k2048_ipc.hip`，
复用相同 A_shard/A/B/output 分配、输入和事件计时。由于协议不同，它有独立 ready 区；
仍保留动态计算队列及通信 Block 完成后参与计算，不能称两个实现调度完全相同。

在已分配四卡的 shell 中运行当前生产版本：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python examples/distributed/ag_gemm_prototype/benchmark_allgather_gemm.py \
  --m 2048 --n 512 --k 2048 --warmup 30 --repeats 30 \
  --json-dir build-old/ag2048-confirm --export-hip build-old/ag2048-confirm
```

默认仍为 48/32，不需要靠把 comm 增加到 65 来触发快速路径。

## 验证与回归

- 新库 `tilelang_module` 与 `tilelang` 均已重新构建；路径检查通过，未加载 develop 树。
- 最终两轮同分配对照：三种实现均通过 epoch 0/1 的完整 gathered-A/output 校验、
  poison 检查和计时后校验。另用连续随机输入，对三种实现均进行了完整 NumPy FP32
  matmul 参考校验，沿用原 runner 的 1e-3 容差；未放宽校验。
- AG 4096/8192/16384 均重新真实 JIT、四卡运行并通过上述 dyadic 全量校验。
  此轮仅 2 warmup / 3 样本作回归烟测，计时约 0.431/1.656/13.424 ms，
  不替代此前正式性能表。
- 最终 7 个相关测试文件 **82 项通过**（`final-tests.log`）：涵盖 persistent 资源、
  指针外提、向量化、索引简化、AG/AR 前端和 IPC 独立 review。
  新门禁负例拒绝无 tile-varying 基址的多次循环、未知 extent、非零 min 和动态 0/1 次循环。
- 四个 AG 尺寸均核验指针外提与 half8 PASS 确实改变通信、没有改写 GEMM 分支。
  4096/8192/16384 两个 PASS 前后的 TIR 与修改前逐字一致。
- AR/RS 各四尺寸向量化 PASS 前后的 TIR 与本轮修改前一致；未修改两个通信优化 PASS。
- 原生 GEMM 两个尺寸真实 GPU smoke 通过。额外用保存的旧二进制与最终新二进制，在
  `LowerSharedGlobalCopy` 和 `InjectWarpIdx` 边界比较 7 份普通/未标记 GEMM：
  输入、输出 TIR 和这一组最终 HIP 全部逐字一致，见 `native-boundary-old/new`。
  初次不带 instrument 的对照中，一份未标记 serial 样例出现过既有 A/B LDS 分配顺序变化；
  原始失败日志保留在 `final-native.log`，没有删除或把那次结果标成通过。

最终生产变更仅涉及 AG 前端、`src/transform/lower_dcu_resource.cc` 的显式一次循环窄例外、
AG 测试以及相关文档。编译临时实验和 ISA 保存在 build-old，不替代正常 JIT 输入。
未提交 git commit，未修改手写参考或 develop 仓库。
