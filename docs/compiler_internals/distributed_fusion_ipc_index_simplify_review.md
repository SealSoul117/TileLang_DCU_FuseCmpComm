# 通信索引简化验收报告（2026-10-06）

## 结论与范围

本轮在 `TileLang_DCU_old` 完成通信索引简化；**不新增 pass，而是窄范围扩展既有
`tl.transform.Simplify`**。保留通信的空汇编线程索引绕行，不实现向量化，不改变 IPC 协议、
规约顺序、GEMM 生成逻辑或 Python 算法。不是 M3 完成声明。

实现由指定的 gpt-5.6-terra/high 子代理完成，未继续分派；主代理负责设计、节点构建、
独立审计、正确性/性能验证和最终 review。没有 git commit。

## 为什么原有 Simplify 不够

真实 scheduled GEMM+RS 使用 `tl::ipc::opaque_thread_id(threadIdx.x)`。helper 的值仍是
线程索引，但空汇编刻意阻断下游编译器错误利用 256 线程上限；不能改回原始线程变量。
现有 Simplify 不知道这个 opaque 返回值的范围，因此保留了多余的 guard 和符号循环次数。
这是实际代码生成路径的问题，即使没有新增测试也需要处理。

独立 Analyzer 探针确认：

- `(linear // 256) * 256 + linear % 256` 原本就能化简，不能算作本轮新增收益。
- 获得 `0 <= comm_tx < 512` 后，已有 Analyzer 可以证明 row 等下界判断恒真。
- 普通表达式 Simplify 仍保留 `(66047 - comm_tx) // 512`，但 `ConstIntBound` 返回
  `[128,128]`，`CanProveEqual(expr,128)` 也成立。因此需要很小的单值物化补充，而非新 pass。

## 实现安全边界

唯一生产源码修改是 `src/transform/simplify.cc`：

1. 仅匹配 ABI-ready、具有 IPC binding、HIP 且精确 `gfx936` 的 PrimFunc。
2. 从当前 `thread_extent(threadIdx.x)` 的正 int32 常量取得实际线程数；**不使用
   target 的 `max_num_threads=256`**。当前融合 launch 为 512。
3. Let/Call/实参严格匹配 int32 标量、精确 helper 名称及当前线程 Var identity。
4. 只在该 Let 的词法 body 中给 Analyzer 建立返回值范围；保留 opaque Call，
   不发射 assume、LLVM 范围元数据、launch-bounds，不构造 warp/lane 替代索引。
5. 仅在该范围内、含此 Let Var 的 serial For extent 上，对纯标量 signed integer
   执行单值范围和等值证明两项检查，再替换成常量。
6. For 使用 CopyOnWrite 保留 step、annotations、thread binding、span 等字段，
   保留 Parent 的迭代变量记账。未知或不支持的嵌套 threadIdx.x scope 不继承外层线程范围。

未给 rank/world 注入范围；不能证明的 tile guard 保留。未触发条件的普通路径维持原行为。
IPC helper、指针外提 pass、idxen pass、warp/lane 模板和原生 GEMM 示例均未修改。

## 实际 HIP 变化

以下为去掉无关地址展开后的示意，完整 HIP 在实验目录中。

修改前：

```cpp
int comm_tx = tl::ipc::opaque_thread_id((int)threadIdx.x);
if (comm_tx < 65536) {
  // 已外提的四个 peer 指针
  for (int i = 0; i < ((66047 - comm_tx) >> 9); ++i) {
    if (0 <= i * 2 + (comm_tx >> 8) && rank_tile_is_valid) {
      // peer payload load
    }
    // 输出写入另带可证明冗余的非负行判断
  }
}
```

修改后：

```cpp
int comm_tx = tl::ipc::opaque_thread_id((int)threadIdx.x);
// 同样四个 peer 指针，仍在 wait/fence/barrier 之后
for (int i = 0; i < 128; ++i) {
  if (rank_tile_is_valid) {
    // 同样的 peer payload load，linear = i * 512 + comm_tx
  }
  // 删除已经证明冗余的输出下界判断
}
```

没有跨同步边界移动数据访问，也没有改变 FP32 加法次序及 FP16 写回。

## 验证记录

实验目录：`build-old/ipc-index-simplify-20261006/`。
节点 dcu0，Slurm 作业 9810，DTK 26.04，四张 gfx936；Python、模板、IPC runtime、
TileLang 动态库均来自 old 隔离环境，没有使用 develop 或旧路径的 TileLang。

- 新增 6 项专项测试：512/256 高半线程、helper/target/ABI/dtype 门控、未知范围、
  非整除尾部、副作用、有效 512 launch 的零 trip/单线程尾部、词法隔离和真实 HIP。
- 原有 99 项 M0/M1/M2、语言、布局及后端回归。
- 实际流水中 4 次 Simplify：启用与禁用新范围知识时，GEMM compute branch 结构一致。
- 独立枚举 512 个线程 × 128 次循环：65536 个 tile 元素恰好覆盖一次，包括高半线程。
- 最终 2048/4096 真正 JIT 的 HIP：compute/wait 前缀、completion 后缀与基线逐字一致，
  改动只在通信循环。
- 四 PE 2048²×512、4096²×1024，计时前后验证均通过；额外 2048 连续随机输入、
  4096 的 24 comm / 56 compute CTA 配置通过。
- 四 PE M2 block litmus、remote copy/get 和两项原生 GEMM GPU smoke 通过。

原生代码生成比较保留了一个真实失败记录：第七项 `unmarked-serial-inner-alloc` 的 A/B LDS
半区交换导致首次文本比较不一致。**修改前的隔离二进制独立重跑也产生了完全相同的新布局**；
7 份新输出与该旧二进制重跑输出全部逐字一致。没有归一化偏移，也没有放宽原比较断言。
该非确定性仍存在，本轮不修复它。

指针外提的集成测试相应更新：实际 512 launch 现在可证明零 trip guard 恒真，因此验证
128 次循环、helper 保留及同步/指针/访问次序；原有合成动态 zero-trip 安全测试全部保留。
这没有删减其安全验收目标，也没有为测试增加生产接口。

## 性能

以下是**完整融合 GEMM+ReduceScatter kernel**，不是纯通信时间。使用 matched 调度，
80 CTA = 48 compute + 32 comm，512 threads，四级流水；输入为 dyadic，5 次预热、30 次采样。
每次取四 PE HIP-event 时间的最大值，再对 30 个值取中位数；不计编译、初始化及主机分配。

| 每 PE GEMM M/N/K | 修改前 | 最终修改后 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 2048 / 2048 / 512 | 0.352080 ms | 0.213199 ms | 39.45% |
| 4096 / 4096 / 1024 | 0.637759 ms | 0.450319 ms | 29.39% |

首轮 after 为 0.2048795 / 0.454159 ms，最终表采用补齐 For 元数据保留后的二进制复测，
不从两轮中挑最低值。4096 最终单次采样最大值约 0.735 ms，所有样本均保留；表中是中位数。

首轮 2048 JIT 另有既有 LDS 布局交换。为隔离归因，独立诊断对照固定修改前计算代码、
仅接入新通信循环：**0.346799 → 0.2047995 ms，下降 40.95%**，四 PE 验证通过。
该实验显式使用 `--diagnostic-hip`，不是替代真正的编译器 JIT 验收。
最终 2048 JIT 恰好与基线计算布局一致，也通过逐字前缀/后缀检查。

4096 ISA 元数据：NumSgprs=104、NumVgprs=256、ScratchSize=212、Occupancy=1，前后相同；
codeLen 从 23224 降为 22968 字节。不能把本轮收益宣称为减少寄存器压力或消除 spill。

## 复现与限制

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
cmake --build build-old --target tilelang_module -j 6
python -m pytest -q testing/python/transform/test_simplify_dcu_ipc_indices.py
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python examples/distributed/gemm_rs_prototype/benchmark_gemm_reduce_scatter.py \
  --implementation tilelang --schedule matched --m 4096 --n 4096 --k 1024 \
  --warmup 5 --repeats 30 --json-dir build-old/ipc-index-simplify-recheck
```

以上 GPU 命令应在已申请四卡的计算节点运行。原始 rank JSON、生成 HIP、ISA、构建/测试日志、
`review_codegen.py`、`review_compute_scope.py` 和 `summary.json` 均保留在实验目录。
最终动态库 SHA256：`78e839d93b802c6d822414df0d7063ee72fdcfc555aefd0da5c015be9bd10e2c`。

此优化只覆盖已验证 profile 和 helper 契约，未声称所有平台适用。此前已记录的 4096 连续
随机输入在 1e-3 阈值下少量 FP16 数值边界问题不由本轮解决；本轮没有放宽该阈值。
仍保留未知 rank/tile 判断；向量化和后续优化待用户另行确认。

最终 review：主代理通过。最终二进制统一运行 **105 项测试全部通过**（24.85 s，
`after/final-105.log`）；上述四卡验收、真实代码隔离审计和性能归因检查通过。
本轮停止于通信索引简化，不进入向量化或其他模块。
