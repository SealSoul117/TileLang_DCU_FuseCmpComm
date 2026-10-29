# GEMM+AllReduce：由后端 PASS 生成 half8 通信访存

日期：2026-10-13。仅在 `TileLang_DCU_old` 开发、构建、验证；未提交 git commit。
实现子代理沿用用户指定的 gpt-5.6-terra/high，不再分派；主代理设计并独立 review。

## 完成的功能

AR 前端不再显式写 `T.vectorized(8)`、八元素局部数组或 chunk 循环，而是：

```python
comm_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)
for linear in T.serial(comm_tx, TILE_M * TILE_N, THREADS):
    p0[0] = remote_partial0[tile, linear // TILE_N, linear % TILE_N]
    # p1/p2/p3 同理
    reduced[0] = T.cast(T.cast(p0[0], "float32") + T.cast(p1[0], "float32")
                       + T.cast(p2[0], "float32") + T.cast(p3[0], "float32"), "float16")
    remote_out0[...] = reduced[0]
    # 另外三个 output peer 同理
```

改动的生产文件：

- `src/transform/vectorize_dcu_ipc_communication.cc`：扩展既有 PASS，新增 AR 专用匹配分支，
  复用 RS 的外层范围分析和线程坐标变换；不改原 RS 的重写逻辑。
- `examples/distributed/gemm_ar_prototype/gemm_allreduce_scheduled.py`：上述标量前端，
  opaque thread id 放在每个 owned tile 的 ready wait/fence/barrier 之后；计算调度和同步协议不变。
- `examples/distributed/gemm_ar_prototype/benchmark_gemm_allreduce.py`：正确传递 JIT 开关，
  并在计时之前核验真实 HIP 是否生成所请求的向量模式。
- `tilelang/engine/phase.py` 仅更新 RS/AR 注释，PASS 顺序未改。

没有修改指针外提 PASS、HIP codegen、IPC runtime、原生 GEMM、RS 算子或手写 HIP。
不修改生产源码就无法将 AR 的多输出标量循环自动转成 half8；本工作是用户要求的真实功能，
不是为了满足测试而添加接口。

## 匹配条件与生成方式

只接受 gfx936、IPC ABI-ready/binding、512 threads 的既有窄 profile：
8 个已经外提的地址 Let，128 轮标量元素循环，循环内恰好 4 pull＋1 local reduce＋4 push。
输入 peer 严格为 0/1/2/3，输出严格为 rank、(rank+1)%4、(rank+2)%4、(rank+3)%4。
输入属于同一个 descriptor，输出属于另一个 descriptor；四路输出使用同一个 FP16 reduced 值。
按词法作用域重放 `tl.assume` 中的范围事实，要求能证明 rank 在 [0,4)；不从名称推断范围，
也不将 sibling 的假设带入当前作用域。

此外验证临时量使用次数、索引相同、源/目标范围、16-byte 对齐、8 lane 连续性。
源地址必须是 `source_base + linear`；目标完整矩阵地址必须是
`output_base + (linear // 256) * row_stride + linear % 256`，不是错误的 `base + linear`。

复用原有映射：

```text
old_i  = 8 * new_i + comm_tx // 64
old_tx = 8 * (comm_tx % 64) + lane
linear = 4096 * new_i + 8 * comm_tx + lane
```

其中 new_i∈[0,16)、comm_tx∈[0,512)、lane∈[0,8)，完整且无重复地覆盖原 128×512 域。
输出为 4 个 FP16x8 load、原顺序的 FP32x8 加法、一次 FP16x8 舍入，显式 Let 绑定
`ipc_ar_reduced_f16x8`，四路 store 复用此结果，避免重复 load/求和。
生成的真实 HIP 每轮恰好四个 remote `uint4` load 和四个 remote `uint4` store。
8 个 peer 指针仍在元素循环外、owned-tile 内、ready 等待之后。

这是编译期匹配/证明，不增加设备侧运行时安全检查或 fallback。仍沿用已批准的 caller 契约：
allocation 有效、16-byte 对齐、输入输出不重叠、通信数据按原协议发布、allocation 存活至所有使用完成。
无法匹配或证明的代码不重写，保持标量。不是通用多 PE / 任意 tile 的 AllReduce 自动向量化。

## JIT 开关的实际修正

`source_only` 使用 PassContext 调用 `tilelang.lower`，该开关直接有效。
但 `tilelang.compile` 会由 JITKernel 内部新建 PassContext；只设置外层上下文不足以启用 PASS。
benchmark 必须使用：

```python
kernel = tilelang.compile(
    bound, target="hip -mcpu=gfx936", execution_backend="ctypes",
    pass_configs={"tl.enable_dcu_ipc_vectorize": not args.scalar},
)
```

`--scalar` 和默认模式共享同一份标量前端，只改变这个后端开关。
示例内部的 `gemm_allreduce_scheduled`/`build_bound` 不再接受无意义的 `vectorized` 参数；
`source_only` 的选择参数和 CLI `--scalar` 保留。直接 JIT 调用应采用上面的 `pass_configs`。

首次 source-only 已通过，但 JIT 未显式传配置，误测了标量版本；主代理 review 不通过后退回修正。
误测数据完整保留在 `after/perf/`，**不用于下表向量化结论**。
benchmark 现在会在计时前检查向量循环、每个 peer 唯一的一次 uint4 load/store、单一 reduced 绑定；
配置与生成代码不符立即报错。新增测试覆盖 True/False 配置实际传入 compile。

首轮测试还发现 synthetic Attr key 错写为 `tir.tilelang_assume`，已按真实常量修为 `tl.assume`，
未为测试修改生产语义；旧测试绑定特定 `p0[8]` 的文本断言更新为更准确的每 peer 访存检查。
未删除原安全义务或放宽数值容差。

## 性能与正确性

同一个 Slurm job 10109 / dcu0，4 DCU，DTK 26.04，路径检查全部指向 old。
512 threads、80 CTAs，四个尺寸 compute/comm 分别为 32/48、56/24、64/16、72/8。
每个实现 30 warmup、30 repeats，每轮取四 PE event 最大值再取中位数；单位 ms。
旧显式 half8 前端及 benchmark 已冻结保存，旧/新各自交替比较同一手写参考。

| M=N, K | 旧显式 half8 | PASS 自动 half8 | 耗时变化 | 同轮手写参考 |
|---|---:|---:|---:|---:|
| 2048, 512 | 0.184719 | 0.184400 | -0.17% | 0.176239 |
| 4096, 1024 | 0.553360 | 0.554960 | +0.29% | 0.534000 |
| 8192, 2048 | 1.985678 | 1.980641 | -0.25% | 1.938320 |
| 16384, 4096 | 10.856307 | 10.880397 | +0.22% | 10.252638 |

结论：本次样本四尺寸变化均在约 ±0.3% 内，性能基本持平，不宣称有稳定加速。
自动版本比同轮手写参考多耗时约 2.2%～6.1%。

- 真正启用 PASS 的四尺寸全量 dyadic partial/output pre/post 检查通过。
- 新标量前端 2048 四卡 smoke 通过；启用 PASS 的 2048 continuous 补充检查通过。
- AR/RS 各四尺寸在向量化 PASS 前后 GEMM compute branch 结构相等。
- RS 四尺寸的新旧 PASS 输入/输出 TIR 与最终 HIP 均逐字相同。
- 普通/未标记 GEMM 的 7 份 HIP 与基线逐字相同。
- 最终 64 项测试通过，构建成功，记录见 `tests-final.log`、`build-final.log`；包含向量化、外提、AR 调度、
  JIT 开关、索引化简、persistent resource 和 M2 接口回归。
- 16384 真正向量版本 ISA 为 `NumSgprs=104, NumVgprs=256, ScratchSize=200`，与旧显式版本相同。
  不宣称本次修复了既有 GEMM spill/reload。

不扩大本次正确性结论到所有随机输入。历史 RS 4096 continuous 的容差问题未在本轮修复。
完整融合代码仍可能受到既有 A/B LDS 排列非确定性的影响；不以少量样本中的零点几百分点作性能归因。

## 复现与产物

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
python examples/distributed/gemm_ar_prototype/gemm_allreduce_scheduled.py \
  --m 4096 --n 4096 --k 1024 --output build-old/ar-auto-half8
# 同一标量前端，关闭向量化 PASS
python examples/distributed/gemm_ar_prototype/gemm_allreduce_scheduled.py \
  --m 4096 --n 4096 --k 1024 --scalar --output build-old/ar-scalar
```

四卡对比继续使用原 `run_compare.sh`。重新启动进程以加载新动态库；benchmark 禁用 kernel cache，
用户旧缓存未删除。

证据根目录：`build-old/ipc-ar-vectorize-20261013/`。

- `before/`：冻结的旧源码/动态库/benchmark，原生 GEMM、AR/RS IR 与旧显式 half8 性能。
- `after-enabled/perf/`：**最终有效**四尺寸 JIT HIP、逐 rank 数据、校验和汇总。
- `after/audit/`：实际 PASS 前后 TIR、HIP、GEMM 不变性和 RS 回归证明。
- `after/native/`：7 份原生/未标记 GEMM 精确比较。
- `after/perf/`：早期 JIT 未传开关的标量误测，保留但不作向量化结论。
- `scalar-smoke/`、`after/continuous/`、`final-smoke/`：补充设备验证。
- `after-enabled-m16384.s`：最终有效向量版本的 ISA。

最终库为 `build-old/lib/libtilelang_module.so`，SHA256：
`6f0dc0eb22e209b259d2fab8cc6c6292fb0f3f8ee531bec00a5e2775c4ddac60`。
本轮只完成 AR 自动向量化，不继续后续模块或其他优化。
