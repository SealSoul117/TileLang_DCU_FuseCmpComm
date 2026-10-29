# GEMM 地址路径与 native lane 保护的因果诊断

日期：2026-10-06。仅使用 `TileLang_DCU_old`；Slurm job 9806，dcu1，四张 gfx936。
这是地址路径诊断，不是新的生产后端实现。没有修改生产 pass、模板、编译参数或二进制。
结束时 old 路径隔离检查通过，四卡均空闲、无残留 KFD 进程；job 9806 已释放。

## 对照条件

固定 matched 调度：80 CTA，48 compute / 32 comm，512 threads，256×256×16 tile，
四级 K 流水，编译仍使用 `--gpu-max-threads-per-block=256`。通信始终保留
`tl::ipc::opaque_thread_id` 的空汇编绕行，未改 ready/completion 协议、规约顺序或布局。

“关闭保护”使用 `build-old/address-path-audit/templates-raw` 的隔离模板视图，
只将 `get_lane_idx` 还原为 `threadIdx.x % warp_size`，其余模板链接到 old。
当前 pass 中生成 `get_lane_idx` 调用的改写仍然存在；这不是宣称整个 pass 已回退到上游。
生产 `src/tl_templates/dcu_hip/intrin.h` 始终保留 native `mbcnt` 保护。

## 为什么复用同一 GEMM 后端却没有走同一地址路径

### 1. 小写 a/b 没有进入 A 的 idxen 路径

`src/transform/inject_dcu_async_copy_idxen.cc` 的 `IsABufferName/IsBBufferName`
只识别 `A/A_1`、`B/B_1`。`GemmDimCollector` 和 `FillGemmDimInfoFromBufferMap`
由此提取 K 与元素字节数；小写名未提取 K，`BuildMatrixParams` 在 `k_dim <= 0` 时禁用转换。

注意：融合版的 copy maps、operand=0/1、use_idxen 标注仍在，并非丢失 GEMM operand 映射。
缺的是上述维度提取步骤。只在 Python DSL 中将参数 a/b 改成 A/B，真实 lower 就从
24 个普通 async copy 变成 12 个 A idxen + 12 个 B 普通 async copy。
诊断脚本使用 scoped Python AST rename，并逆向比较 AST；不做 TIR 文本回读，
不依赖此前撤销的 raw-region 扩展。

### 2. persistent owner 交错调度改变 resource base 和 copy offset 的分解

原生 `examples/dcu/example_gemm.py` 每个 block 的 tile 坐标就是 bx/by；
4096×4096×1024 导出的 A resource 是 `A + blockIdx.y * 262144`，
idxen 的 row 只含 tile 内的 warp/lane 坐标。

融合版的 tile 坐标依赖外层 job 循环：

```python
job = block_idx + iteration * 48
tile = (job % 4) * tiles_per_rank + job // 4
```

`LowerSharedGlobalCopy` 的 collector 把外层循环变量也加入 `loop_vars_`，
`VariableEliminator` 构造 resource 候选基址时将它们置零。
当前 `FinalizeResourceBases` 还要求所有 copy 相对该候选的偏移可证明非负；
不能证明时回退到矩阵基址。这个安全回退来自此前的实际 resource-offset 正确性修复，
不能为了模仿原生路径直接删除它。

实际导出结果是 kernel 入口的 `A + 0` / `B + 0`，tile 坐标留在每条 copy 的偏移中。
因此同样调用 `T.copy/T.gemm` 并不意味着生成的地址分解与普通 GEMM 相同。

## 实机因果对照

以下主对照均为 4096×4096×1024，dyadic 输入，完整 CPU oracle 逐位检查。
融合版通过项检查四 PE 的 full partial 和 RS output，包含计时前后检查。
失败项在 partial 阶段退出，不作性能比较。

| 版本 | native lane 保护 | 结果 |
| --- | --- | --- |
| 原生 `example_gemm.py` | 关 | 全矩阵通过 |
| 小写输入，flat tile-grid 纯 GEMM | 关 | 全矩阵通过 |
| 小写输入，普通 sequential persistent 纯 GEMM，16 blocks | 关 | 全矩阵通过 |
| 只保留融合版 owner 交错调度，移除全部 IPC 的纯 GEMM | 关 | 失败，4,192,800 个错误元素 |
| 上一项完全相同 HIP | 开 | 全矩阵通过 |
| 当前小写输入 matched 融合版 | 开 | 四 PE full pre/post 通过 |
| 当前小写输入 matched 融合版 | 关 | 四 PE partial 失败 |
| 仅改名 A/B，恢复 idxen | 开 | 四 PE full pre/post 通过 |
| 仅改名 A/B，恢复 idxen | 关 | 四 PE partial 失败，与小写失败输出逐位相同 |
| idxen + 每 tile resource base，但仍以 global-row 减 tile-row 表达局部坐标 | 关 | 四 PE partial 失败 |
| idxen + 每 tile resource base + 直接生成纯 local-row | 关 | 四 PE full pre/post 逐位通过 |

原生与 flat/sequential 控制说明：不能把“小写 a/b”或“有外层循环”单独等同于必然算错。
去除 IPC 后仍复现则说明，错误不是通信协议或规约误差导致的。

4096 小写失败版的四 rank 错误数分别为 4,192,802 / 4,192,739 / 4,192,799 / 4,192,707。
将 tile-major partial 恢复成矩阵后，全量输出精确符合以下错误行映射：

```python
wrong_row = (row | 1024) if (row & 16) else row
actual[row, :] == expected[wrong_row, :]  # 四 rank 全部逐位相同
```

仅恢复 idxen 的四份失败输出与小写版逐位相同。机器码不含 `v_mbcnt`，且出现将原始
thread id 与 block bits 合并的 `v_lshl_or_b32`；例如 idxen 版本先计算
`(thread_id << 2) | (block_id << 10)` 再用包含 bit 10 的 `0xce0` 掩码。
实际 thread id 的 bit 8 因 launch512 而可能非零，移位后污染 row 的 bit 10。
这与上述实测错误行映射一致。

移动 resource 后若仍生成“global row - tile row”，机器码仍会先作上述错误合并再相减。
所以只验证 C++ 数学等价还不够；需在交给 HIP 编译器前生成纯 tile-local 坐标。

2048×2048×512 的 uppercase/raw 对照已四 PE 通过；不能由小尺寸通过外推4096也安全。

## 对齐后是否需要额外保护

在上述两种 shape 的 dyadic 实验中，**不再需要 native lane 保护**：local-row 诊断版
使用原始 `threadIdx.x % 64`，两者均四 PE full partial/RS pre/post 逐位通过。
当前 matched/未对齐版仍需要，不能直接删除生产保护。

补充验证：

- 2048 continuous：四 PE full partial/RS pre/post 通过原 `rtol=atol=1e-3`。
- 4096 continuous：四 PE partial 通过原容差；RS 仍出现此前的严格容差边界失败，
  **不宣称 full continuous 通过**。rank 0/2 保存的全部 partial（各16,777,216元素）
  和 output（各4,194,304元素）分别与历史 `hand-cont-dump`、`tl-cont-dump`
  全量逐位比较，均零差异。未放宽容差、未改变 oracle。
- `local-row-kernel.s` 中没有 `v_mbcnt`，原失败路径的
  `thread_id << 2 | block_id << 10` 加 `0xce0` 掩码也已消失。
- 本轮只做短采样：1 warmup + 5 repeats、每次取四 PE 最大值后中位数。
  当前未对齐保护版4096为1.469601 ms；对齐关闭保护的HIP诊断4096为1.224161 ms，
  2048为0.659360 ms。这里同时改了地址路径与保护，不能把全部收益归因于删除保护，
  也不是新的正式性能基线。

关键对齐不是在 HIP 中留下等价减法，而是直接生成：

```cpp
// 每次外层 tile 迭代构造；tile_row 为 block-uniform。
auto A_res = tl::make_wave_buffer_resource<11>(A + tile_row * 256 * 1024);
// 完全不含 tile_row/blockIdx/outer-loop 坐标。
int row_local = (lane >> 3) * 32 + warp_idx * 4 + ((lane & 7) >> 1);
tl::cp_async_gs_idxen<16, /*原 LDS offset*/>(dst, A_res, k_column_bytes, row_local);
```

上例为解释用伪代码；完整可编译诊断在
`build-old/address-path-audit/uppercase-raw4096/tile-base/local-row-rank0.hip`。
替换前，诊断脚本严格匹配全部12处源 row 表达式，并枚举所有合法 job × 512线程，
证明 `global_row == tile_row * 256 + local_row`：4096覆盖131,072对，2048覆盖32,768对。
B resource 同样按 tile 构造，byte offset 减掉 tile-col 基址；未改 B 的运算/布局。

这证明的是指定 gfx936/DTK/shape/调度下的路径对齐能避开已复现问题，
不是对 launch512/max256 组合在任意 kernel 上安全的保证，也不是新的编译器实现已经完成。
后续真正的后端工作应当是：可靠提取 GEMM operand 的形状信息，
按外层 tile 作用域构造 resource，并在 TIR 中完成 local offset 分解/化简。
不能重新引入有 carry/取模时不正确的基址相减，也无需重启通信 serial-loop 自动识别。

### 复现 local-row 诊断

在已分配四卡并加载 old 环境的节点执行（输入 HIP 先由改名 DSL 诊断导出）：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
python scripts/dev_old/gemm_tile_base_audit.py \
  --source build-old/address-path-audit/uppercase-raw4096/hip/tilelang-matched-m4096-n4096-k1024-rank0.hip \
  --output build-old/address-path-audit/uppercase-raw4096/tile-base/local-row-rank0.hip \
  --m 4096 --n 4096 --k 1024 --local-row
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
TL_TEMPLATE_PATH="$PWD/build-old/address-path-audit/templates-raw" \
  mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python examples/distributed/gemm_rs_prototype/benchmark_gemm_reduce_scatter.py \
  --implementation tilelang --schedule matched \
  --diagnostic-hip "$PWD/build-old/address-path-audit/uppercase-raw4096/tile-base/local-row-rank0.hip" \
  --m 4096 --n 4096 --k 1024 --warmup 1 --repeats 5 \
  --json-dir build-old/address-path-audit/localrow-raw4096
```

隔离模板不是另一份安装；除了 intrin.h，其余文件链接到 old/src。
如果该实验目录不存在，需要先创建等价的隔离模板视图，不能把环境指向 develop。

## 诊断入口与证据

- `gemm_address_path_audit.py`：原始与 A/B 改名的真实 Python DSL lower/JIT 对照。
- `scripts/dev_old/gemm_address_pure_probe.py`：原生、flat、serial、owner-interleaved 的无 IPC 控制。
- `scripts/dev_old/gemm_tile_base_audit.py`：明确标注 HIP-only 的 resource/offset 对齐实验，
  **不能称作后端已经实现的优化**。
- 所有日志、HIP、TIR、JSON、失败数组和 ISA：`build-old/address-path-audit/`。

未修改文件的 SHA-256（与本轮开始前相同）：

```text
inject_warp_idx.cc       2304fec776b69b590e50a6d404059ca499ef5a88bd7a1e4fd3411e4e436cf5e3
dcu_hip/intrin.h         c1314567e0ce9533190a8ecd1ecb68314b22e4113b0e9057ef65971fececbe3b
libtilelang_module.so    6ed3d1953ca81671dc46460a4f6b035d53d74b163f515785935b1d4f8e308c3b
```
