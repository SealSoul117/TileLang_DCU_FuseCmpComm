# M0：DCU 分布式融合契约

M0 冻结后续 DSL、HIP、调度、运行时和 JIT 共享的数据契约。本模块只使用 Python
标准库：不导入 `tilelang/__init__.py`，不编译代码、不分配 ROC_SHMEM 内存，也不实现
设备端同步或运行时行为。

```text
tilelang/distributed/spec.py      FusionSpec、SymmetricBufferDesc、
                                  DistributedKernelABI、RuntimeBinding
tilelang/distributed/schedule.py  SchedulePlan 及映射记录
```

它们由 `tilelang.distributed` 再导出。CPU 测试通过 `importlib` 直接装载该小包，避开
带原生依赖的 TileLang 包根初始化。

## 语义规格

`
()` 的 v1 默认值为 `M=N=4096`、`K_local=1024`、`P=4`、tile `(128,128)`、
输入 fp16、累加/通信/输出 fp32、AllReduce 循环 owner、19 通信 block、`batch_tiles=1`、
tile-major 输出和 `collective_host` 完成。M0 仅接受 `gemm` 加 `allreduce` 或
`reduce_scatter`。AllGather 被明确拒绝：它需要输入 readiness 协议，不能伪装为 v1 的
输出 ready 协议。

物理输出布局和数学结果分布独立。AR 要求 `tile_major` 与 `replicated`；RS 要求
tile-major、`row_sharded`、连续行 owner 以及 `compute_tile_order='owner_interleaved'`。
首条 v1 lowering 仅支持完整的 `128x128` M/N tile 和 `K_local % 32 == 0`。
`tile_extent` 仍记录短边 tile 的映射范围，但 `validate_v1_lowering()` 和 v1 ABI 会拒绝
尚未实现 padding/guard 的 GPU lowering，M0 不声称边界 tile 已可执行。

## Buffer 描述符与 ABI

`SymmetricBufferDesc` 不可变且可 JSON 序列化。`strides` 的单位是**元素**；
`byte_offset`、`nbytes` 是**字节**；`alignment` 是 allocation base 的保证对齐。view 的
有效对齐为 `gcd(base_alignment, byte_offset)`（offset 为零时即 base alignment）。v1 要求
FP32 staging/output 的有效对齐至少 16 字节（float4），ready 至少 4 字节。

`nbytes` 是底层 allocation 容量而非 view 大小，检查：

```text
byte_offset + ((sum((shape[i] - 1) * strides[i]) + 1) * sizeof(dtype)) <= nbytes
```

因此非零 offset 被保留并验证。不同 `allocation_id` 从不被当作连续 heap。ABI v1 明确
拒绝 workspace alias：staging、output、ready 必须有不同的 allocation ID。

`DistributedKernelABI` 是不可变的版本 1 JSON manifest，包含匹配的 spec/schedule、
persistent launch（grid、256 threads/block、动态 LDS）、能力要求、工作区布局和对称参数
绑定、ready reset 生命周期及 collective-host 完成契约。factory、直接构造和 JSON reload
都会从 spec/schedule 重新推导并验证：

- staging：FP32 `[max_owned_tiles, P, tile_m, tile_n]`；
- ready：uint32 `[tile_count, P]`；
- AR output：FP32 `[tile_count, tile_m, tile_n]`；
- RS output：FP32 `[max_owned_tiles, tile_m, tile_n]`，只有
  `output_valid_tile_count(rank)` 前缀有效。

`output_tile_to_local_offset(tile_id, rank)` 返回本 rank 的 tile-major 输出槽；RS 上非 owner
tile 返回 `None`。ABI 要求 `grid_blocks - comm_blocks == schedule.compute_blocks`，保留
`rocshmem_ipc`/`system_scope_atomics`，且各 v1 工作区均有读写权限。

`RuntimeBinding` 单独保存 rank、context handle、opaque pointer 与 allocation generation。
`compile_identity()` 排除 allocation ID、pointer、generation、rank、context；
`agreement_identity()` 保留 allocation ID 与布局、排除短暂运行时值。v1 禁止 alias，故纯 ID
重命名不会混淆 alias 与 non-alias cache artifact。

JSON reader 使用精确 schema。未知/缺失字段、非 object、非法 JSON、伪造 `min_nbytes`、在
整数位填 bool，以及常见嵌套类型错误（例如列表 `collective_kind` 或嵌套列表 permissions）
均抛 `DistributedConfigError`。

M0 记录但不执行的生命周期：

```text
ready 清零 -> 本地 stream 完成 -> 跨 PE barrier
-> launch -> 本地 kernel 完成 -> 跨 PE barrier -> output 可消费
```

本地 launch event 不是 collective completion；最终 barrier 前不得复用工作区。

## 调度映射与 ready 公式

`SchedulePlan(spec, compute_blocks)` 是 producer/consumer 唯一的映射来源：

```text
work_id <-> tile_id <-> (bx, by)
tile_id <-> (bx, by)  由 tile_to_coords / coords_to_tile 双向验证
tile_id -> (owner, owned_index) -> owner_slot
tile_id -> (producer_block, producer_iteration, batch_start_work)
owner-local tile lists -> owner-interleaved consumer_tile_order
```

`coords_to_tile(bx, by)` 仅接受非 bool 整数，严格检查 `0 <= bx < tiles_x` 和
`0 <= by < tiles_y`，并与 `tile_to_coords` 互为逆映射。

v1 AR 的手写参考公式为：

```text
owner       = tile_id % P
owned_index = tile_id // P
owner_slot  = owned_index * P + source
producer_block     = work_id % compute_blocks
producer_iteration = work_id // compute_blocks
batch_start_work   = producer_block
                   + (producer_iteration // batch_tiles) * batch_tiles * compute_blocks
```

ready 位于 owner 的 ready allocation：

```text
ReadyKey(owner, batch_start_work, source)
local ready offset = batch_start_work * P + source
```

owner 是远端 allocation 的选择器，不参与本地 offset。AR 中 `work_id == tile_id` 只是巧合，
不能推广到 RS。`ready_key` 验证 batch start/source/owner 的非 bool 整数和范围；
`notifications_for_batch` 对同一 producer batch 的每个不同 owner 仅发一次，尾 batch 只含
`work_id < tile_count` 的工作。

RS 的连续行 owner 和 producer/consumer owner-interleaved 顺序均由 owner-local tile list
推导；`owned_index` 是本地 list 序号，绝非 compute work ID。四个等长 owner 行组的前四个
producer owner 是 `[0,1,2,3]`；AR 保留 row-major。没有拥有行的 rank 不会得到伪造 slot：
`tile_from_owned_index` 返回 `None`，`consumer_for_lane` 为空。M0 的统一 ready key 不声称
与历史手写 RS local-tile-indexed ready array 字节兼容。

## CPU 验证

```bash
python3 -m unittest testing/python/distributed/test_m0_contracts.py
python3 -m pytest testing/python/distributed/test_m0_contracts.py  # 安装 pytest 时
```

15 项 CPU 测试覆盖两种执行顺序下 work/tile/coords/slot 双射、batch 1/2/4/8、producer 公式、
owner 去重通知与 ready coverage、RS 本地输出 offset、idle/owner 尾部和 mapping-only edge guard；
还覆盖 offset/span、非法配置、严格 JSON、ABI round trip、workspace 篡改、alias/capability/
permission/alignment 拒绝、runtime generation 及 cache/agreement identity。

M0 未测试 HIP codegen、ROC_SHMEM 映射、设备内存序或 collective 执行；它们属于 M1 之后的
显式验收工作。
