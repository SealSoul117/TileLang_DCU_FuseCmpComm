# M0 逐文件实现详解：把分布式融合的约定变成可校验的数据

日期：2026-09-21。对应主仓库 HEAD：`b2bc99b0a3ab03cb5ffce731db87048931e011a6`，以及当前未提交的 M0 实现。
本报告按当前文件行号定位；以后代码修改可能移动行号。它是源码教学报告，不替代历史验收记录。

## 1. 先理解 M0 到底做了什么

我们已经有手写的 GEMM + AllReduce kernel，其中隐含了许多约定：

- 哪个计算 Block 算哪个 tile？
- tile 的结果由哪个 PE 归约？
- source PE 的中间结果写进 owner 的哪个 staging 槽？
- 一个 ready 标志对应一个 tile，还是一批 tile？
- 输出是普通二维行主序，还是 tile-major？
- 本卡 kernel 结束以后，其他卡是否还可能向本卡写数据？

如果编译器、kernel 生成器和运行时各自抄一套公式，很容易“单独看都正确，组合后却不一致”。
**M0 的工作就是把这些约定写成一份不可变、可序列化、可检查的公共契约。**

M0 不会生成 HIP，不会执行 GEMM，不会申请显存，也不会调用 ROC_SHMEM。
运行 M0 得到的是 Python 对象、整数映射和 JSON 文本，不是 GPU kernel。

### 1.1 它在整体流程中的位置

```text
算子的需求：尺寸、PE 数、tile 大小、AR/RS
       │
       ▼
FusionSpec：数学含义、布局和算法选项
       │
       ▼
SchedulePlan：work/tile/owner/slot/ready 的统一映射
       │
       ▼
DistributedKernelABI：启动参数、workspace、绑定和生命周期清单
       │
       ├── 编译身份：以后用于识别可复用的编译结果
       ├── 跨 PE 身份：以后用于核对各进程是否使用相同约定
       └── 运行绑定校验：检查 rank、allocation ID 和 generation
       
以上是 M0 已实现的对象与校验。
下面是后续接线，而不是当前已自动运行的流程：
       └── 算法 DSL/调度生成器 → M1 TIR → HIP lowering → runtime → DCU
```

M0 与 M1 目前没有自动转换桥梁：构造 `FusionSpec` 不会自动得到 M1 `PrimFunc`。

### 1.2 阅读前需要的几个词

| 词 | 本项目中的意思 |
| --- | --- |
| PE / rank | 参加通信的一个进程/编号；实验中通常每个 PE 对应一张 DCU |
| tile | 把大矩阵结果切成的小矩形，如 128×128 |
| compute Block | persistent kernel 中负责反复计算 tile 的线程块 |
| owner | 负责收齐并归约某个 tile 的 PE |
| source | 当前这份局部 GEMM 结果来自哪个 PE |
| staging | owner 上暂存各 source 局部结果的工作区 |
| ready | 标记某个 producer batch 已经发布的 uint32 槽 |
| ABI | 编译器与运行时共同遵循的接口清单；这里首先表现为 Python/JSON manifest |
| generation | 分配的生命周期编号；防止误用同名但已被替换的内存 |

## 2. M0 新增/修改文件总清单

| 文件与代码入口 | 类型 | 主要作用 |
| --- | --- | --- |
| [tilelang/distributed/spec.py](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:19) | 新增 | 配置、对称 Buffer 描述、workspace、ABI、运行绑定、严格序列化 |
| [tilelang/distributed/schedule.py](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:23) | 新增 | producer/consumer 映射、owner、ready、输出槽及尾部处理 |
| [tilelang/distributed/__init__.py](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/__init__.py:8) | 新增 | 统一导出上述公共接口，没有调度或通信执行逻辑 |
| [testing/python/distributed/test_m0_contracts.py](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:21) | 新增 | 不依赖 TVM 原生库的 15 项 CPU 契约测试 |
| [.gitignore](/public/home/baoboyuan/TileLang_DCU/.gitignore:24) | 修改，共享文件 | 放行 `tilelang/distributed/` 和 `testing/python/distributed/`，防止已有 `dist*/` 规则忽略源码 |
| [docs/compiler_internals/distributed_fusion_m0.md](/public/home/baoboyuan/TileLang_DCU/docs/compiler_internals/distributed_fusion_m0.md:1) | 新增说明 | 面向使用者的契约、单位、映射和测试说明 |
| [docs/compiler_internals/distributed_fusion_m0_review.md](/public/home/baoboyuan/TileLang_DCU/docs/compiler_internals/distributed_fusion_m0_review.md:1) | 新增验收记录 | 各轮问题、补修内容、测试证据与源码 hash |
| 本报告 `distributed_fusion_m0_implementation_report.md` | 本次新增文档 | 面向首次阅读实现的逐文件、逐功能讲解 |

`.gitignore` 后来还被 M1 补充；M1 的 language 目录例外和 `build-m1/` 规则不属于 M0 算法实现。
总计划 `distributed_fusion_dcu_plan.md` 是实施依据，不是 M0 新增的执行逻辑。
原有 TVM/CK 子模块检出差异、虚拟环境、GEMM 示例不应误算为 M0 的功能改动。

## 3. spec.py：从用户配置到 ABI 清单

### 3.1 基础校验：所有入口先讲同一种“语言”

入口：[DistributedConfigError](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:25)、[_is_int](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:33)、[_strict_object](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:37)。

这一层解决的是“不要让错误配置带着歧义进入后续代码”：

- `DistributedConfigError` 是统一的配置异常，继承 `ValueError`。
- `_is_int` 特意排除 bool，因为 Python 的 `isinstance(True, int)` 为真，但矩阵尺寸 `True` 显然不合法。
- `_strict_object` 要求 JSON object 的字段集合精确匹配 schema，既不缺字段，也不静默忽略未知字段。
- [_parse_json](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:52) 将 JSON 解析错误转为配置错误。
- [_canonical_dtype](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:61) 把 `float16/float32` 统一为 `fp16/fp32`。
- [_tuple_of_ints](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:70) 将 shape/strides 等变成合法的整数 tuple。
- [_json_dump](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:86) 固定 key 排序和分隔符，为稳定 hash 提供确定的文本。

这不是形式主义：如果两个进程对同一个字段的解释不同，后面很可能表现为等待不结束或远端地址错误。

### 3.2 FusionSpec：回答“究竟要计算什么”

字段定义：[FusionSpec](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:90)。
构造校验：[FusionSpec.__post_init__](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:127)。

默认值对应第一条手写 v1 路径：

| 配置组 | 默认值/含义 |
| --- | --- |
| 尺寸 | M=N=4096，K_local=1024，4 PE |
| tile | 128×128 |
| 数值类型 | 输入 FP16，accumulator/通信/输出 FP32 |
| 算法 | `direct_push_owner_reduce_v1` |
| collective | AllReduce |
| owner | `cyclic`，按 tile 编号循环分配 |
| 执行顺序 | `row_major` |
| 输出 | 物理 `tile_major`；数学分布 `replicated` |
| 通信资源 | 19 个 comm Blocks，batch_tiles=1 |
| 完成语义 | `collective_host` |

`@dataclass(frozen=True)` 表示构造后不能随意修改字段。要换配置，应创建新对象；这可以避免某个模块偷偷改变公共约定。
构造时用 `object.__setattr__` 做 dtype/tuple 规范化，并不等于允许外部随意修改。

AR 与 RS 不是只改一个字符串：

| collective | owner_policy | compute_tile_order | 数学输出分布 |
| --- | --- | --- | --- |
| AllReduce | cyclic | row_major | replicated |
| ReduceScatter | contiguous_rows | owner_interleaved | row_sharded |

例如只写 `FusionSpec(collective_kind="reduce_scatter")` 会因为其他字段仍是 AR 默认值而报错。
这是有意要求配置完整，防止数学语义、访问顺序和物理布局混为一谈。

AllGather 在当前契约里明确拒绝，因为 AG+GEMM 需要描述“输入数据何时可用”，而这里首先冻结的是“GEMM 输出何时可归约”。

### 3.3 配置能描述边界，不等于 GPU 已支持边界

入口：[requires_edge_guards](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:200) 和 [validate_v1_lowering](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:204)。

M0 的纯数学映射可以描述 M=130 这样的短边 tile；这有利于检验映射规则。
但第一条 lowering 契约要求：

- M/N 不出现不完整 tile；
- K_local 能被 32 整除；
- tile_shape 恰好为 128×128。

`FusionSpec` 可以先构造，`DistributedKernelABI` 构造时再调用 `validate_v1_lowering()` 拒绝尚不可生成的配置。
所以“映射测试通过”和“GPU epilogue 支持该尺寸”是两回事。

[FusionSpec 序列化](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:221) 将字段显式写入 dict，回读仍走构造校验，而不是从 JSON 绕过规则。

### 3.4 SymmetricBufferDesc：描述一块内存，不申请这块内存

定义：[SymmetricBufferDesc](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:267)；
校验：[__post_init__](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:292)。

这里必须区分三个单位：

- `shape`：各维元素数；
- `strides`：元素单位的步长；
- `byte_offset/nbytes/alignment`：字节单位。

`nbytes` 是整个 backing allocation 的容量，不只是逻辑元素总字节数。
`allocation_id` 是符号身份，不是实际地址；不同 ID 不会被视为同一大 heap 的相邻段。

相关派生量在 [itemsize / effective_alignment / view_span_nbytes](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:335)：

```text
max_element_offset = Σ (shape[i] - 1) × strides[i]
view_span_nbytes   = (max_element_offset + 1) × sizeof(dtype)
必须满足：
byte_offset + view_span_nbytes <= nbytes
```

具体例子来自 CPU 测试：shape=(2,3)、strides=(5,1)、FP32、byte_offset=16。

- 第一行元素偏移为 0、1、2；第二行为 5、6、7。
- 最大偏移为 7，因此覆盖跨度为 8×4=32 字节，不是 6×4=24 字节。
- allocation 至少需要 16+32=48 字节；给 47 会被拒绝。
- base alignment=256 不代表这个 view 仍有 256 字节对齐。
  偏移后的保证对齐为 gcd(256,16)=16，适合后续 float4。

[structural_dict](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:352) 可以保留或去掉 allocation ID，且排除 generation；
[to_dict/from_dict](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:368) 的完整持久化形式则保留 generation。
这是后面两类身份 hash 的基础。

### 3.5 辅助数据类：把 ABI 的每一部分都单独定义

| 代码块 | 功能与关键规则 |
| --- | --- |
| [LaunchConfig](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:394) | grid Blocks、每 Block 线程数、动态 shared memory 字节数；这里只校验字段，不查询 CU/occupancy |
| [WorkspaceRequirement](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:422) | 某角色需要的 dtype/shape/strides/offset；`min_nbytes` 从布局推导 |
| [WorkspaceRequirement.from_dict](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:466) | 不信任 JSON 自报的 min_nbytes，重新计算并比较 |
| [BufferBinding](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:500) | 将 argument 名、workspace role 和 Buffer 描述符关联起来 |
| [ResetProtocol](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:532) | ready 每轮清零、本地 stream 同步、launch 前跨 PE barrier、复用时机 |
| [CompletionContract](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:568) | 本地 kernel 完成之后还需跨 PE barrier 才算 collective_host 完成 |
| [RuntimeBinding](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:600) | 当前 rank、context handle、allocation 对应的指针字符串与 generation |

`ResetProtocol` 和 `CompletionContract` 只是记录和校验“必须做什么”，没有真正执行 memset、stream synchronize 或 barrier。

`RuntimeBinding` 同样不访问 GPU：`"0x100"` 在这里是 opaque string，不会被转换或解引用。
其构造检查 pointer/generation 两组 ID 唯一且集合相同，并排序为稳定 tuple。

### 3.6 工作区必须由 schedule 推导，不能由调用者自己宣称

核心入口：[_expected_workspace_layouts](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:483)。

设 P 为 PE 数，T 为总 tile 数，O 为各 owner tile 数的最大值，E=tile_m×tile_n：

| 工作区 | shape | 用途 |
| --- | --- | --- |
| staging | [O,P,tile_m,tile_n] | 同一个 owner tile 为每个 source 留一个独立槽 |
| AR output | [T,tile_m,tile_n] | 每个 PE 保存所有最终 tile |
| RS output | [O,tile_m,tile_n] | 每个 PE 只使用自己拥有的有效前缀 |
| ready | [T,P] | 按 producer batch start 与 source 寻址，预留统一容量 |

默认 4096×4096、P=4 时，T=1024，O=256，每 PE 的契约容量为：

- staging：256×4×128×128×4 = 67,108,864 字节，即 64 MiB；
- AR output：1024×128×128×4 = 64 MiB；
- ready：1024×4×4 = 16 KiB。

这些只包括三种 workspace，不包括 A/B，也不是本次新测得的显存使用量。
本报告生成时已用当前 M0 代码复算上述数字。

### 3.7 DistributedKernelABI：总装并再检查一次

定义/校验：[DistributedKernelABI](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:655)。
便利工厂：[DistributedKernelABI.v1](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:750)。

工厂的流程是：

```text
检查 spec 与 schedule 配套
    → 检查 v1 尺寸支持范围
    → 为 staging/output/ready 确定不同 allocation ID
    → 推导 WorkspaceRequirement
    → 为每个 requirement 构造 SymmetricBufferDesc 和 BufferBinding
    → 构造 LaunchConfig
    → 进入 DistributedKernelABI.__post_init__ 做统一校验
```

统一校验的重点：

1. ABI version 必须是整数 1。
2. schedule 必须来自同一份 spec。
3. grid_blocks - comm_blocks 必须等于 compute_blocks；首期每 Block 为 256 threads。
4. capabilities 必須声明 `rocshmem_ipc` 与 `system_scope_atomics`。
   这只是能力需求声明，**不是已经探测硬件支持**。
5. workspace 必须与 schedule 重新推导出的布局完全一致。
6. 每种 workspace 恰好有一个 binding；dtype、shape、strides、offset、容量匹配。
7. workspace 可读可写；staging/output 有效对齐至少 16 字节，ready 至少 4 字节。
8. 三种 workspace 的 allocation ID 不得相同，首版明确禁止 alias。

因此，把 ready 的 requirement 和 binding 一起伪造成 4 字节也不能蒙混通过：
实现比较的不只是“两份输入互相一致”，还会与 schedule 的真实需求比较。

`v1(grid_blocks=80)` 的 80 是当前工厂默认值，不是查询硬件的结果，更不是所有 DCU 的固定 CU 数。
后续资源检查仍需由专门模块完成。

### 3.8 序列化、两种身份，以及运行绑定

代码入口：[ABI 序列化](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:804)、
[compile_identity](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:858)、
[agreement_identity](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:862)、
[bind_runtime](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/spec.py:866)。

| 内容 | 完整 JSON | 编译身份 | 跨 PE 一致性身份 |
| --- | --- | --- | --- |
| spec/schedule/launch/布局/协议 | 有 | 有 | 有 |
| allocation ID | 有 | 去掉 | 保留 |
| runtime_generation | 有 | 去掉 | 去掉 |
| rank/context/真实指针 | 单独放在 RuntimeBinding | 不进入 | 不进入 |

两类 identity 都是对规范化 JSON 求 SHA-256：

- 编译身份用于将来判断“代码能否复用”；换一批实际内存不应该重新编译。
- 一致性身份用于将来判断“各 PE 是否认为同一个 ID 代表同一个布局”；这里保留 ID。

目前只是提供这些身份计算函数，尚未改造真实 JIT cache，也没有自动执行跨 PE hash 交换。

`bind_runtime()` 名字容易误导：当前只检查 nranks、allocation ID 集合和 generation 是否匹配，
返回 `None`，不修改 ABI，不把指针注入 kernel，也不会连接 communicator。

## 4. schedule.py：保证生产者和消费者用同一套地址公式

### 4.1 三种结果记录

定义：[ProducerAssignment](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:23)、
[ReadyKey](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:39)、
[ConsumerAssignment](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:56)。

- ProducerAssignment：一个 work 最终计算哪个 tile、归哪个 owner、写哪个槽、属于哪个 batch。
- ReadyKey：远端 owner 选择器 + 本地 ready 数组下标。
- ConsumerAssignment：一个 owner 的某条通信 lane 在第几次迭代处理哪个 tile。

它们都是不可变记录，不是启动线程或执行通信的对象。

### 4.2 四种编号千万不要混用

[work_to_tile / tile_to_work](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:120)、
[tile_to_coords / coords_to_tile](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:133)：

- `tile_id`：逻辑二维结果按 tile 行主序编号；
- `work_id`：计算任务在执行顺序中的编号；
- `owned_index`：该 tile 在 owner 自己的列表中的序号；
- `producer_block`：哪个计算 Block 执行这项 work。

基础公式：

```text
tiles_x = ceil(N/tile_n)
tiles_y = ceil(M/tile_m)
tile_count = tiles_x × tiles_y

bx = tile_id % tiles_x
by = tile_id // tiles_x
tile_id = by × tiles_x + bx
```

AR 的 work_id 恰好等于 tile_id；RS 不相等。不能把 AR 的巧合写成所有算法的前提。

[tile_extent / tile_is_full](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:145) 返回短边有效范围；
[_check_tile/_check_work/_check_rank](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:108) 集中检查编号范围。

### 4.3 owner、owner_slot 和输出槽

入口：[owner_of_tile](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:160)、
[owner_tiles/owned_index](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:176)、
[owner_slot](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:222)。

AR：

```text
owner = tile_id % P
owned_index = tile_id // P
owner_slot = owned_index × P + source
```

owner_slot 是 staging 中的“tile 槽编号”，不是 float 元素下标，更不是字节地址。
槽内元素 j 的元素偏移为 `owner_slot × tile_elements + j`。
因为不同 source 使用不同槽，所以 source 之间不需要竞争写同一 payload 地址。

RS 用商和余数把连续 tile 行分给 PE；有余数时前几个 PE 多拥有一行。
`owned_index` 从 owner 的 tile 列表求得，而不是从 work_id 取模猜出。

[tile_from_owned_index](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:194) 对无效尾部返回 None；
[output_tile_to_local_offset](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:202)：

- AR：每 PE 输出槽就是 tile_id；
- RS：owner 上是 owned_index，非 owner 返回 None；
- [output_valid_tile_count](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:218) 告诉 runtime 有效前缀长度。

### 4.4 producer 分配与 batch ready：最容易写错的部分

核心实现：[producer_for_work](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:227)。

```python
producer_block = work_id % compute_blocks
producer_iteration = work_id // compute_blocks
batch_first_iteration = (producer_iteration // batch_tiles) * batch_tiles
batch_start_work = producer_block + batch_first_iteration * compute_blocks
```

一个 Block 的相邻两次工作相差 compute_blocks，所以它的一批 work 通常不是连续整数。

[batch_work_items](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:254) 检查 batch 是否属于这个 Block、是否对齐，
并只返回小于 tile_count 的工作；
[producer_batches](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:273) 枚举这个 Block 的有效批次。

[ready_key](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:282) 的公式为：

```text
owner：选择访问哪张卡的 ready allocation
storage_index = batch_start_work × P + source
```

owner 不再乘进本地 ready 下标，因为它已通过“选择哪个 PE 的 allocation”表达了。

[notifications_for_batch](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:301) 对本 batch 涉及的 owner 去重；
[ready_for_tile](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:312) 让消费者反查同一 producer batch 的 key。

例子：P=2，compute_blocks=2，batch_tiles=4，Block 0 的首批 work 为：

```text
work/tile = [0,2,4,6]
owner     = [0,0,0,0]
source    = 1
```

这里只需要给 owner 0 的 ready[1] 发布一次，而不是四次。
它的语义是“这个 source 的整个 producer batch 对此 owner 的数据都已准备好”。
因此后续生成器必须先完成整批 payload，再通知；M0 只给 key，不替设备端执行这个顺序。

### 4.5 通信消费者怎样分工

入口：[_owner_interleaved_tiles](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:316)、
[consumer_tile_order](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:332)、
[consumer_for_lane](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:344)。

`_owner_interleaved_tiles` 先取各 owner 的列表，再按各列表的第 0 项、第 1 项……轮流交错；
短列表用完后跳过，不制造假 tile。

`consumer_for_lane(owner,lane)` 在这个 owner 的本地列表上按：

```text
owned_index = lane, lane + comm_blocks, lane + 2×comm_blocks, ...
```

分配通信工作。这和 compute Block 的 work_id 分配是两套有明确连接关系的映射，不是同一个循环。

[SchedulePlan 序列化](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/schedule.py:353) 只存 spec 与 compute_blocks；
映射由规则重算，不在 JSON 中再保存一套可能失配的映射表。

## 5. 两个从头走到尾的例子

### 5.1 默认 AR：tile 1023、source 3

从 `FusionSpec()` 和 `SchedulePlan(spec,61)` 开始：

| 步骤 | 当前代码得到的值 |
| --- | --- |
| tile 网格 | 32×32 |
| tile 坐标 | (bx,by)=(31,31) |
| owner / owned_index | 3 / 255 |
| source 3 的 staging 槽 | 255×4+3=1023 |
| compute Block / iteration | 1023%61=47 / 1023//61=16 |
| batch=1 的 batch_start_work | 1023 |
| 等待/发布位置 | owner 3 的 ready[1023×4+3]，即 ready[4095] |
| AR 最终输出槽 | 每个 PE 的 output[1023] |

这里的 output[1023] 表示整个 tile，不是单个 float。
上述值与 [默认参考公式测试](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:41) 一致，本报告生成时也独立复算过。

### 5.2 RS：为何 work_id 和 tile_id 必须分开

M=512、N=384、P=4，因此有 4 行×3 列共 12 个 tile，每个 owner 拥有一行：

```text
owner 0: [0,1,2]
owner 1: [3,4,5]
owner 2: [6,7,8]
owner 3: [9,10,11]

计算顺序：
work_id: 0  1  2  3  4  5  6  7   8  9  10 11
tile_id: 0  3  6  9  1  4  7  10  2  5  8  11
```

取 tile 3、source 2、compute_blocks=5、batch_tiles=2：

- tile 3 对应 work 1；
- owner=1、owned_index=0、owner_slot=2；
- producer Block=1，batch_start_work=1；
- ready 在 owner 1 的 ready[1×4+2]=ready[6]；
- RS 结果写 owner 1 的 output[0]，不是 output[3]，其他 PE 不拥有这个 tile。

这同时展示了逻辑布局、访问顺序、归属和物理槽四者如何协作。
更换执行顺序不应该更换数学结果归属。

## 6. 其余文件与测试怎样保证这些约定

### 6.1 tilelang/distributed/__init__.py

[导入](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/__init__.py:8) 和 [__all__](/public/home/baoboyuan/TileLang_DCU/tilelang/distributed/__init__.py:23)
只是把 spec/schedule 类集中导出。不要把这个包名与 M1 的 `tilelang/language/distributed` 混淆：

- 前者是主机端的契约和整数映射；
- 后者是 DSL 到 TIR 的构造接口。

### 6.2 test_m0_contracts.py：每组测试检查什么

[_load_contract_package](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:21) 用 importlib 将该小包单独加载，
避免普通 `import tilelang.distributed` 先执行 TileLang 根包、再要求 TVM/HSA 动态库。
**M0 源码无原生依赖，不代表普通 Python 包导入一定绕过根包。**

| 测试入口 | 验证内容 |
| --- | --- |
| [default_v1_mapping](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:41) | 默认参数与手写公式一致 |
| [coords_to_tile](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:52) | 坐标双向映射与非法编号 |
| [batch_formula](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:69) | 非连续 batch、owner 去重、尾批次 |
| [enumerated_mapping](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:83) | 多种 P/Block/batch 下 slot 唯一、tile 不丢 |
| [reduce_scatter_rows](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:113) | 连续行归属与消费者交错 |
| [edge_and_inactive](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:142) | 边界 tile、无 tile PE、空闲 Block/lane |
| [rs_compute_order](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:172) | 生产者也按 owner-interleaved，且 work/tile 双射 |
| [ready_coverage](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:191) | 每次等待都被一次合法通知覆盖，payload 属于该 batch |
| [buffer_units](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:236) | 非零 offset、带 padding 的跨度和容量 |
| [invalid_configuration](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:256) | AG、非法 batch、不完整 RS 配置拒绝 |
| [abi_identity_runtime](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:270) | JSON 回读、不可变性、两种 hash、generation |
| [edge_lowering](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:301) | 不把 mapping-only edge 当作已可执行 |
| [rs_workspace](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:307) | RS 有效前缀、伪造 workspace 拒绝 |
| [abi_rejections](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:340) | alias/capability/权限/对齐/tile 不匹配拒绝 |
| [strict_schema](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m0_contracts.py:359) | 未知字段、伪造 min_nbytes、bool、嵌套非法类型 |

### 6.3 文档、ignore 和验证边界

[原接口说明](/public/home/baoboyuan/TileLang_DCU/docs/compiler_internals/distributed_fusion_m0.md:1) 是使用契约摘要；
[原验收报告](/public/home/baoboyuan/TileLang_DCU/docs/compiler_internals/distributed_fusion_m0_review.md:1) 保留历次 review 事实，
包括经用户批准补修后的通过结论。本报告没有覆盖或重写那些历史记录。

历史 M0 15 项测试通过；M1 最终完整套件中也重新包含并通过了这 15 项。
CPU 测试只证明已检查的配置/映射行为，不能证明实际系统原子能力、GPU 数值正确、死锁自由或性能。

当前实现也不是高性能 GPU scheduler：例如 owner 列表可通过 Python 枚举重算，
`tile_to_work` 某些路径使用列表搜索。它首先是可靠的参考规则和生成器输入，不在 GPU 每个元素循环里执行。

## 7. 建议阅读顺序与一句话总结

第一次阅读建议按以下顺序：

1. FusionSpec 的字段与配置检查；
2. 本报告两个实例，再回到 SchedulePlan 对照公式；
3. SymmetricBufferDesc 的单位和跨度；
4. _expected_workspace_layouts 和 ABI.__post_init__；
5. identity、RuntimeBinding 和测试；
6. 再读 M1 报告，理解如何把通信意图写成 TIR。

**M0 的交付物是“大家必须一致遵守的规则”，不是执行这些规则的设备 kernel。**

