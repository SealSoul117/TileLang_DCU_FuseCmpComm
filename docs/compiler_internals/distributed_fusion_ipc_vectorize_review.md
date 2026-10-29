# IPC 通信向量访存 pass：实现、验收与历史记录

日期：2026-10-06。仅修改 `TileLang_DCU_old`，未修改 develop 或手写融合算子，未提交 git commit。

## 结论

### 当前状态：外层范围分析与 half8（2026-10-06）

按用户要求，将原有 opt-in pass 从 half2 扩展为 half8，并修复大尺寸未命中的范围分析。
实际功能依据：此前 8192/8192/2048（72 compute + 8 comm）和
16384/16384/4096（76 compute + 4 comm）的输出地址依赖外层 tile 循环，旧 pass
只绑定内部 128 轮循环和 512 线程的范围，不能证明输出索引合法，因此保留了标量路径。
这不是为满足测试扩展生产接口；删除测试后，这两种真实生成路径仍需要此修改。

实现变化：

- 在候选通信语句处重放其词法范围内的 blockIdx 范围、serial For 范围、纯算术 Let
  绑定和 If 的 then/else 条件。每个候选使用独立 Analyzer；离开作用域恢复事实栈，
  ConstraintContext 严格逆序释放，不假设额外的 rank/world 常量。
- 按先前已知的分支事实推导保守的有限循环范围。进入循环体时可利用 extent > 0，
  因此允许 2048 配置中其他通信 CTA 的零工作量；区间计算使用 __int128 检查，
  不允许范围宽度或端点在 TIR 索引类型中截断、溢出。
- 映射改为 `old_i = 8*new_i + comm_tx//64`、
  `old_tx = 8*(comm_tx%64) + lane`，其中 new_i 为 0..15、lane 为 0..7。
  仍完整且唯一地覆盖 128*512=65536 个原始元素。
- 使用 `Ramp(..., 1, 8)`，复用既有 HIP printer，生成 uint4 load/store；每个元素仍
  按原顺序进行四 PE 的 FP32 加法，再转回 FP16。没有修改 HIP printer 或 GEMM 后端。
- 证明所有八个地址连续、起点为 8 个 half 的倍数，buffer alignment 必须至少 16 字节
  且是 16 的倍数。调用方需保证实际地址 16 字节对齐、范围有效且可表示、输出不与源重叠。
  不新增运行时 vectorization guard/fallback；不能证明合法的候选在编译期保持原样。

开关仍为 `tl.enable_dcu_ipc_vectorize=True`，默认 false；匹配范围仍限定为已绑定 IPC ABI、
gfx936、512 线程、四个外提 IPC 源指针的 scheduled RS normal form，不是通用通信向量化。
重新生成时应避免使用旧 JIT 缓存；本轮真实 JIT 验证显式调用 `tilelang.disable_cache()`。

主代理验收：Slurm job 9827，dcu0，四卡 PCI 地址 5e:00.0、05:00.0、e9:00.0、c1:00.0。
最终 native SHA-256 为 `f56442d87ccc35b6d979927931a41a29b7b48c2b290d824bcf840ba5a806b48b`。
仅使用 old 的 Python、模板和 native 库，路径隔离检查通过。

- 26 项专项测试通过：向量化、指针外提、索引简化、persistent resource、嵌套流水。
  新增/更新了八元素映射、四尺寸真实 lowering、分支/Let 范围、作用域泄漏、未知范围、
  int32/int64 端点溢出、16 字节对齐以及不越界但只满足 half2 对齐的负例。
- 四尺寸每个 PE 均经过计时前后完整 GEMM partial 与 RS 输出校验，dyadic 输入逐位一致。
- 同一次 pass 的前后 IR 审计确认四尺寸的 GEMM 分支和有序同步调用结构一致。
- 原生 GEMM 两个设备 smoke 用例通过；16 份最终 HIP 均有 half8 循环且无向量 guard，
  相同编译参数生成的 ISA 均含四条 `global_load_dwordx4` 及 `global_store_dwordx4`。
- 两个扩展回归文件为 3 passed / 2 failed，仍为此前记录的旧单行 T.serial 断言和旧 GEMM
  lane/a_dcu_res 路径断言；本次未修改、跳过或删除这些既有测试。

最终真实 JIT 性能：80 CTA、512 线程、64 KiB shared、四级流水；预热 30 次，计时 30 次。
指标为每轮四个 PE 的 HIP event 耗时最大值，再取中位数；不包含分配、编译、flag reset
和 host MPI barrier。参数、启动器、完整校验与之前四尺寸测试一致。

| 本地 M/N/K | compute/comm blocks | 最终 half8 JIT 中位数 ms |
| --- | ---: | ---: |
| 2048/2048/512 | 48/32 | 0.135439 |
| 4096/4096/1024 | 48/32 | 0.374720 |
| 8192/8192/2048 | 72/8 | 1.448397 |
| 16384/16384/4096 | 76/4 | 8.941585 |

这些是 PASS 自动生成的代码，不是手工修改 HIP 的结果。16384 的本轮样本仍有波动；
不使用最小值宣称收益，也不宣称完全追平手写版。上一轮当前 JIT 的大尺寸约为
2.799/20.820 ms，属于不同作业的参考值，不是本轮同场 A/B。

证据目录：`build-old/ipc-vector8-20261006/`，最终结果选择 `m*-c*-final/`。
`summary.json` 包含逐轮样本、四 rank 汇总及 16 份 ISA 统计；`scope-final.log`、
`focused-final-r2.log`、`native-gemm-final.log`、`legacy-regression.log`、`paths-final.json`
分别记录 IR 保持性、专项测试、原生 GEMM、既有失败及路径隔离。初次编译的 Optional
命名空间错误与新增测试的 IfThenElse 构造错误日志均保留；后者只修测试构造，未因此修改
生产逻辑。第一次 half8 设备结果也保留，最终表格来自收紧 alignment %16 后重新构建的复测。

### 历史状态：移除 half2 运行时检查（2026-10-06）

用户已要求暂时移除向量化 pass 加入的运行时安全检查。显式开启时，pass 直接生成
64 轮 half2 向量循环，不再生成 `can_vectorize_half2` guard 或 128 轮标量回退；调用方
需保证 4 字节对齐、有效且可表示的分配范围，并保证输出与输入不重叠。开关仍默认关闭。
下文关于 guard、错位回退及其设备验收的结论仅为历史记录，不描述当前 unchecked 契约。
此前对历史性能回退的归因性判断已由后续诊断撤回；下文性能数据仅保留为当时的观测记录，
不能据此推断当前 unchecked 路径的性能或根因。

本次主代理 review 与验证（Slurm job 9817）：old 后端增量构建和路径隔离通过；
向量化、指针外提、索引简化、persistent resource、嵌套流水共 25 项专项测试通过。
真实 TileLang JIT 的 2048/2048/512、4096/4096/1024 各四 PE，运行前后 GEMM partial
及 ReduceScatter 输出全量检查均通过；八份 HIP 均无向量化 guard 或标量 fallback。
同一次 pass 前后 IR 比较证明两个尺寸的 GEMM 分支及有序 IPC 同步调用结构不变；
原生 GEMM 的两个设备 smoke case 通过。未做新的性能验收。

扩展回归另有两项既有失败：`test_gemm_rs_scheduled.py` 的单行 `T.serial` 源码断言，
以及 `test_gemm_rs_opaque_thread_id.py` 对旧 GEMM lane 路径的断言。用保存的旧 native
二进制复测仍有同样失败；本次未修改、跳过或删除这些测试。完整初次回归为 28 passed / 2 failed。
日志和新 HIP 位于 `build-old/ipc-vectorize-unguarded-20261006/`。

`ipc.hpp::can_vectorize_half2` 的 helper 定义保留以兼容历史导出的 HIP，但当前 pass
不再调用它。旧 HIP 仍含原检查，需重新生成才能得到本次无检查版本。运行时不满足上述
对齐/有效范围/不重叠前提的输入不再承诺安全回退；编译期负例检查与原有 IPC 检查不变。

以下为修改前的历史结论，已由上文当前状态更新覆盖：新增
`VectorizeDCUIPCCommunication`，但**不能作为已取得收益的优化启用**。
四 PE 正确性测试通过；完整 GEMM+RS 和同一 HIP 启动器的对照均出现明显性能回退。
第一轮 review 要求补充安全性检查；修正后第二轮性能验收未通过，按约定停止继续调优。
最终保留为研究用 opt-in pass，`tl.enable_dcu_ipc_vectorize` **默认 false**。

## 实现范围

- `src/transform/vectorize_dcu_ipc_communication.cc`：仅识别 ABI-ready、gfx936、512 线程、
  已外提四个 IPC 指针的 scheduled RS FP16 pull/reduce/store normal form。
- 插入位置：指针外提、展开及现有 Simplify 之后，host/device split 之前。
- 保留 512 个线程，将 128 轮单 half 处理改为 64 轮双 half；映射为
  `old_i=2*new_i+comm_tx//256`、`old_tx=2*(comm_tx%256)+lane`。
  两 lane 始终落在原来的合法线程/循环域内，覆盖原来的 65536 个元素。
- 使用 TIR `Ramp(...,1,2)`，复用已有 HIP printer，生成 `uint1` 32-bit load/store；
  每 lane 仍按原顺序以 FP32 做 `((p0+p1)+p2)+p3`，最后转回 FP16。
- 局部 Analyzer 检查连续性、偶数偏移、范围及输出唯一性；限制 uniform 控制/基址、
  temp 使用、descriptor 身份、存储 scope 等。不识别的形式保持原样。
- `src/tl_templates/dcu_hip/ipc.hpp` 中的 `can_vectorize_half2` 检查五个基址的 4-byte
  对齐、可表示地址区间及输出与各只读源区间不重叠；不满足就执行原始 scalar For。
  这不是针对任意 VMM/物理别名的通用证明，仍限定于当前 IPC 分配契约。
- 保留原来的 opaque asm 线程索引。没有改变 GEMM codegen、通信算法或同步协议。
  opaque scope 后续 fence/barrier 等语句原位保留，不移过副作用。

显式研究开关：

```python
kernel = tilelang.compile(
    bound_func,
    target="hip -mcpu=gfx936",
    execution_backend="ctypes",
    pass_configs={"tl.enable_dcu_ipc_vectorize": True},
)
```

默认不传该项，继续走此前的标量通信路径。此 pass 不是通用向量化能力，也不表示 M3 完成。

## 已完成的设备与源码检查

全部证据保存在 `build-old/ipc-vectorize-20261006/`，Slurm job 9813、dcu0、四张 gfx936。

- 独立穷举验证 65536 元素的映射双射、双 half 相邻及行边界。
- 地址 guard 的 16 个设备边界用例通过，包括错位、区间重叠、相邻区间、空地址和溢出。
- 独立真实 TileLang DSL 四 PE 测试：正常输出和偏移一个 half 的输出都逐元素精确正确，
  高半区线程覆盖及前后 canary 检查通过。生成源码确有 `uint1` load/store；错位走标量回退。
- 完整 2048/2048/512 和 4096/4096/1024 GEMM+RS：计时前后四 PE 全量校验通过。
- pass 前后真实 IR 中，compute 分支和所有 IPC 同步调用结构保持相同。
- 原生 GEMM 两个设备 smoke case 通过，old 环境路径隔离通过。
- 最终默认路径的 105 项回归测试全部通过；最终 opt-in 的四 PE 对齐/错位设备测试再次通过。
- 新增 4 项专项测试全部通过，包含映射、非 IPC no-op、6 种拒绝场景、fence 保持，
  以及真实 scheduled 算子的默认关闭/显式开启生成检查。这不等于任意通信程序的通用安全性证明。
- 关闭开关后的真实 JIT 2048 算子恢复至 **0.202799 ms**，计时前后四 PE 校验通过，
  四份生成源码均不包含 `can_vectorize_half2`。默认用户路径没有保留这次性能回退。
- 原生/未标记源代码七项比较：首次六项逐字相同，末项出现既有 LDS 半区互换。
  保留初次失败；保存的旧二进制再次生成七项，与新二进制产物全部逐字相同，
  证明该差异不是新 pass 引入。未归一化偏移，也未放宽比较。

## 性能：不能宣称收益

同一 allocation，80 CTA = 48 compute + 32 comm、512 threads、四级流水，dyadic 输入；
5 次 warmup、30 次计时。指标是每轮四 PE HIP event 耗时最大值，再取中位数，不是四卡均值。

| 完整 TileLang JIT 算子 M/N/K | 原标量 ms | 启用向量 pass ms |
| --- | ---: | ---: |
| 2048/2048/512 | 0.203520 | 0.846878 |
| 4096/4096/1024 | 0.455121 | 1.079518 |

为排除不同启动路径的影响，另用**同一 diagnostic HIP launcher**，分别执行保存的旧源码
和新生成源码（不是拿手工改写 kernel 冒充 TileLang 结果）：

| 2048/2048/512 对照 | 中位数 ms | 最小/最大 ms |
| --- | ---: | ---: |
| 旧标量源码 | 0.198400 | 0.196639 / 0.207520 |
| 新向量源码 | 0.898240 | 0.301120 / 1.490559 |

新向量版本不仅没有收益，还明显变慢并抖动。不能把这一结果解释为单纯的 JIT 启动器差异，
也不能用较好的最小样本代替中位数宣称加速。向量 load/store 已生成，不代表降低总通信字节量，
更不保证融合 kernel 的寄存器分配、访存指令及执行效率一定改善。
**本轮尚未定位性能回退的确切根因**，停止于报告，不继续改算法或展开下一项优化。

## 可查看的文件

- 专项测试：`testing/python/transform/test_vectorize_dcu_ipc_communication.py`。
- `after/pair-audit-rank0.hip`：短小的真实向量路径及标量回退示例。
- `after/rs4096/hip/`：完整融合算子生成代码。
- `summary.json`、各 PE JSON、`control-*-retest.log`：原始计时和验证记录。
- `after/scope-review-final.log`、`after/vector-tests-final-r3.log`、`after/regression-final.log`：最终构建后的检查结果。
  前两次专项测试构造失败的日志亦保留，未修改生产逻辑去适配测试。

后续若继续，应先固定同一 GEMM 指令/资源布局，对通信段生成的机器指令、寄存器/spill、
缓存属性及动态行为做定点对照；这些只是待检验方向，当前未将任何一个认定为根因。
