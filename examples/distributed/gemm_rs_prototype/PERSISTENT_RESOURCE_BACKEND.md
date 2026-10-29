# 显式 persistent GEMM tile-resource 后端

本文记录 `tl.dcu_persistent_tile` 的窄范围契约。它只修正 owner-interleaved GEMM 的
global-to-shared 地址分解；不实现 work queue、通信/计算动态角色或 M3。

## 前端用法

仅将输出 tile 的外层遍历标记为：

```python
for job in T.serial(block_idx, tile_count, compute_blocks,
                    annotations={"tl.dcu_persistent_tile": 1}):
    # tile_row/tile_col，随后是四级 K pipeline 的 T.copy/T.gemm
```

`gemm_reduce_scatter_scheduled.py` 只在 compute `job` loop 使用该标记；通信 serial loop
没有标记，`tl::ipc::opaque_thread_id` 汇编调用不变。`gemm_persistent_resource.py` 是无 IPC 的
单卡计算入口，用于单独检查 A/B 的地址路径。

输入 operand 采用大写 `A`/`B`，以便沿用现有 idxen A/B 维度识别；本模块不修改 idxen pass。

## 后端契约与拒绝条件

未标记 PrimFunc 仍走原 `LowerSharedGlobalCopy` 分支，不由本模块猜测 persistent loop。标记的
loop 必须是 serial、block-uniform，且其 body 必须包含一个真实的 innermost DCU GEMM K loop。
资源只针对标记 loop body 的向量化 async global-to-shared copy 建立。

对每个 source Buffer，后端在该 tile loop 内、tile-coordinate Let 之后、K loop 之前构造
`tl.make_dcu_resource`。基址保留 outer tile/block-uniform 坐标；offset 从原始标量地址精确
相减并在绑定范围下化简。只有在 residual 非负且不再经 Let 依赖 tile/block 时才发射。已验证的
residual 被直接缓存给 copy 替换，不能由无范围 Analyzer 再次化简，从而不拆开 floor/mod carry。

显式拒绝：K/非 GEMM marker、无 tile-dependent base、thread-dependent 或含 load/opaque-call 的
enclosing control、thread/inner-K/不纯 base、无法消去 tile/block residual、marker 内定义 global
source Buffer，以及无法让依赖 Let 支配资源 Let 的结构。这些拒绝不会回退成“看似优化成功”的
普通地址表达式。

## 验证入口

CPU TIR 回归：

```bash
pytest -q testing/python/transform/test_dcu_persistent_resource.py \
          testing/python/transform/test_dcu_resource_offset_split.py
```

第一项从真实 compute-only GEMM lower 后，以 Var 身份和 Let 展开检查 A/B resource 位于 marker
loop 内；普通 copy 的 `args[3]` 与 idxen row 的 `args[4]` 都不依赖 marker/block 变量。第二项
保留未标记 resource-offset fallback 的 floor/mod carry 与非负地址重建，并仅更新 scheduled 集成
断言：该例已显式 opt-in，故应使用 A/B tile resource，而不应再断言旧的 B+0 fallback。

source-only 示例：

```bash
python examples/distributed/gemm_rs_prototype/gemm_reduce_scatter_scheduled.py \
  --m 4096 --n 4096 --k 1024 --total-blocks 80 --comm-blocks 32 \
  --output build-old/persistent-resource-source
```

GPU 结果、raw-helper 隔离和普通 GEMM byte-exact 基线由验收记录维护；这里不把它们外推为通用
launch512/max256 或任意 persistent schedule 的安全结论。

## persistent tile 的 lane 保护清理（2026-10-06 验收通过，非 M3）

resource/local-offset proof 成功后，后端仅在同一 tile `For` 上附加内部
`tl.dcu_persistent_resource_verified=1`。它不是前端接口：单独写 marker 绝不会关闭保护。
`InjectWarpIdx` 同时看到原 `tl.dcu_persistent_tile=1` 和该 proof token，且当前 target 明确为
gfx936、线程形状为一维 `threadIdx.x=512`（`y/z=1` 视为未启用）时，才在该 loop body 把
lane source 设为 `threadIdx.x % 64`。现有 `get_warp_idx_sync`/readfirstlane warp 路径保留。

任何 loop 外、未标记 sibling、只有 marker 没有 proof token、非 512 或多维线程形状，在原有
DCU InjectWarpIdx 路径中都继续生成 `get_lane_idx(64)`/native mbcnt 保护；非 DCU target 则保持
该 pass 原本的不处理行为。通信的
`opaque_thread_id` 和 `intrin.h` 均不修改。该例外基于 tile-local 地址已经在 TIR 中证明且
gfx936 一维512实际路径的单独验证；它不表示一般 persistent 调度、其他设备或 M3 已完成。

### 本次 lane 清理的验收结果

第二轮 review 通过。实现仍由 `gpt-5.6-terra / high` 子代理完成，主代理负责设计、
审查、构建与设备验收；未提交 Git commit、未进入通信指针外提等后续优化。
第一轮的三个新测试预期错误已修正：warp 的 `FloorDiv(warp_idx, 1)` 用结构和 Analyzer
验证；gfx942 验证原 pass 的 no-op；HIP warp helper 按实际双参数签名检查。
没有为这些测试错误修改生产实现，也没有削弱旧回归断言。

生产修改仅有两处：

- `src/transform/lower_dcu_resource.cc`：证明通过后，在同一 tile loop 上记录内部 verified token。
- `src/transform/inject_warp_idx.cc`：仅在已验证 profile 的 loop body 生成普通 lane 表达式；
  loop bounds 不进入新 scope，退出 loop 后恢复旧行为，内层不支持的线程维度也恢复旧行为。

无需修改现有 Python 示例，仍只使用 `tl.dcu_persistent_tile`；用户不应自行设置内部 token。
不再需要 `templates-raw` 隔离环境。本次所有 GPU 测试使用正常 `scripts/dev_old/env.sh`
选择的生产模板，`intrin.h`、idxen 与原生 GEMM 示例的 SHA-256 均与修改前相同。

环境：Slurm job 9808，dcu0，四卡 gfx936，DTK 26.04，old 的 `.venv-dcu/build-old/src`。
最终模块库 SHA-256：

```text
6f474b05071f30ac57e2d332741393b4145b7187fffdb7f62d41b62f26af62ab
```

验收结果：

- 14 项专项测试通过（包含新增 scoped lane 测试及之前的资源/流水/warp 测试），61 项旧回归通过。
- 5 组原生 GEMM + 2 组 unmarked 对照的 HIP，与本次修改前快照全部逐字一致。
- 2048/4096 的通信分支 HIP 与本次修改前逐字相同；没有改通信算法、索引汇编或同步协议。
- 4096 融合 HIP 不再调用 `get_lane_idx`，设备汇编中的静态 `v_mbcnt` 从 337 条降为 0；
  `get_warp_idx_sync` 保留。静态指令数不等于动态执行次数。
- 四 PE matched、48 compute / 32 comm：2048×2048×512 和 4096×4096×1024 的
  dyadic full partial/RS pre/post 逐位检查通过；2048 continuous 通过原容差。
- 四 PE 4096、56 compute / 24 comm 的 dyadic full pre/post 通过。
- 纯 persistent GEMM：16 blocks 下 2048/4096 的 dyadic 和 continuous 均通过；
  15 blocks 的 2048 dyadic 也通过。
- 原生 GPU smoke：512×768×256、1024×512×512 continuous 全元素检查通过。

同节点本批对照，5 warmup、30 repeat；每轮取四 PE 最大 HIP event 时间，再取中位数。
只统计 kernel，排除 host prepare/reset/sync，不是端到端时间：

| M×N×K | 清理前 ms | 清理后 ms | 耗时下降 |
| --- | ---: | ---: | ---: |
| 2048×2048×512 | 0.725600 | 0.629120 | 13.30% |
| 4096×4096×1024 | 1.495678 | 1.282878 | 14.23% |

这是一批同机结果，不外推至其他尺寸/设备。4096 汇编的 private segment 从 20 增至
200 bytes，SGPR/VGPR 仍为 104/256，因此不把收益解释成所有资源用量都下降。
既有 4096 continuous 融合 RS 的严格容差边界本轮未重新验收、不宣称修复；下方历史失败
及手写对照证据继续保留。

证据目录：`build-old/persistent-lane-cleanup-20261006/`，包含 `before/`、`after/` 的
HIP/四 rank JSON/ISA，`build2.log`、`review2-tests.log`、`legacy-regression.log`、
`pure-gpu.log`、`native-gpu.log` 和 `final-paths.json`。相关专项可复现为：

```bash
source scripts/dev_old/env.sh
python -m pytest -q \
  testing/python/transform/test_dcu_persistent_lane_scope.py \
  testing/python/transform/test_dcu_persistent_resource.py \
  testing/python/transform/test_dcu_resource_offset_split.py \
  testing/python/transform/test_dcu_nested_main_loop_pipeline.py \
  testing/python/transform/test_inject_warp_idx.py
```

## 2026-10-06 tile-resource 后端验收记录（lane 清理前）

本模块第二轮 review 通过，未进入后续模块、未创建 Git commit。实现由
`gpt-5.6-terra / high` 子代理完成，主代理负责设计、源码审查、构建与节点验证。
第一轮专项测试发现测试本身错误地要求每个 operand 都随 tile iteration 变化：
2048 的 N tile 数为 8，而 blocks=16 时 B 列恒为 `block_idx % 8`。
没有为该断言修改生产代码；改为同时精确检查 blocks=15（A/B 都变化）和
blocks=16（A 变化、B 合法恒定），两者都保留作用域及全部局部坐标断言。

环境为 old 的 `.venv-dcu` / `build-old` / `src`，Slurm job 9807，dcu1，四卡 gfx936，
DTK 26.04。所有编译与执行都来自真实 TileLang PrimFunc，没有替换为手改 HIP。
最终模块库 SHA-256：

```text
d9b27d11f28798797613ace03f246f40d024c137e843b4851707a6c2577a4aa8
```

### 普通 GEMM 隔离

本轮相对于开始时的工作树，仅在 `lower_dcu_resource.cc` 添加独立 persistent 分支及
marker 入口；旧 collector/injection/replacer 的实现未变。
`inject_dcu_async_copy_idxen.cc`、`inject_warp_idx.cc`、`dcu_hip/intrin.h` 的 SHA-256
与本轮开始时相同。没有修改原生 `examples/dcu/example_gemm.py`，也没有移除全局 lane 保护。

5 组原生 GEMM 的最终 HIP 与修改前逐字相同（不是规范化比较）：

- 512×768×256、1024×512×512；
- 2048×2048×512、4096×4096×1024；
- 原例默认的 19200×5120×5120。

额外的 unmarked flat 对照也逐字相同。另一个 unmarked serial-inner-alloc 对照需要单独说明：
首次 baseline 使用 A/B 一种 LDS 排列，最终生成使用另一种排列，因此最初的严格比较确实报错，
记录保留在 `final-unmarked.log`，没有放宽比较断言或删除失败文件。
随后在隔离目录加载**修改前保存的二进制**，同一进程重复生成，第三次复现了与最终结果
逐字相同的另一种排列。7 份最终 HIP 与 `before/same-process2/` 中的 7 份旧二进制产物
逐字匹配。旧代码的 `MergeSharedMemoryAllocations` 在同 start/size 时用 Var 指针地址
排序，是这种布局非确定性的代码线索；本轮不修改此 pass。因此不能把“7 份匹配”解释为
该旧诊断例在所有进程/生成次序下都有唯一的文本输出。

### 回归与设备结果

- 最终专项测试：11 passed，包括 resource scope、idxen row 的 Var identity/Let 展开、
  非法 marker/控制流/外层 impure alias、既有 offset carry、嵌套 K pipeline 和 warp 检查。
- 既有 M0/M1/frontend/layout 等回归：61 passed；不代表待修的 M0 IPC 契约已完成。
- 原生 GEMM 两组 continuous GPU smoke：512×768×256、1024×512×512，通过原
  `rtol=atol=1e-3` 全元素检查。
- 默认模板、四 PE、48 compute / 32 comm：2048×2048×512 continuous 与
  4096×4096×1024 dyadic，完整 partial/RS pre/post 通过。
- 隔离 raw lane helper、四 PE：2048/4096 dyadic 的完整 partial/RS pre/post 通过；
  最终二进制另复跑 4096 的 48/32 和 56/24 配置，均通过。
- 隔离 raw lane helper、纯 persistent GEMM：2048/4096 各自的 dyadic（bitwise）和
  continuous（原容差）全部通过；15 blocks 的 2048 dyadic 也通过。

raw helper 仅用单条命令的 `TL_TEMPLATE_PATH` 指向
`build-old/address-path-audit/templates-raw`，将 `get_lane_idx` 恢复为普通线程取模；
生产模板及普通 GEMM 完全不改。这是验证地址路径本身的隔离实验，不是新的默认模式。

4096 continuous 融合 RS 的已知严格容差边界问题仍存在：rank 0 有 2 个、rank 2 有 4 个
RS 元素超阈值。两 rank 保存的完整 partial（各 16,777,216 元素）和 RS shard
（各 4,194,304 元素）与历史手写版逐位相同，bit mismatch 为 0。
未放宽容差，也不宣称该场景验收通过或问题已解决。

本轮以正确性和生成隔离为目标；1 warmup / 5 repeat 的短采样不作为正式性能结论。

### 复现与证据

在已申请的计算节点中：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
cmake --build build-old --target tilelang_module -j 6
python scripts/dev_old/check_paths.py
python -m pytest -q \
  testing/python/transform/test_dcu_persistent_resource.py \
  testing/python/transform/test_dcu_resource_offset_split.py \
  testing/python/transform/test_dcu_nested_main_loop_pipeline.py \
  testing/python/transform/test_inject_warp_idx.py

export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python examples/distributed/gemm_rs_prototype/benchmark_gemm_reduce_scatter.py \
  --implementation tilelang --schedule matched --m 4096 --n 4096 --k 1024 \
  --warmup 1 --repeats 5 --json-dir build-old/persistent-resource-check \
  --export-hip build-old/persistent-resource-check/hip
```

证据根目录：`build-old/persistent-resource-backend-20261006/`。

- `before/`：修改前源码/二进制、最初 HIP、隔离旧二进制重复生成记录。
- `final/unmarked/`：最终 7 份 unmarked HIP；比较工具为
  `scripts/dev_old/persistent_resource_baseline.py`，始终要求 exact equality。
- `build4.log`、`review2-tests.log`、`final-legacy-regression.log`、`final-paths.json`。
- `final-native-gpu.log`、`final-pure-gpu.log`、`final-default2048-cont.log`、
  `final-default4096.log`、`final-raw4096.log`、`final-raw4096-c24.log`。
- `raw4096-cont/failure/`：保留的严格容差失败数据，与
  `build-old/rs-schedule-audit/hand-cont-dump/` 的手写输出作逐位比较。
