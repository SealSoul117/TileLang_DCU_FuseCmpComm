# TileLang_DCU 分布式融合：IPC v2 实施计划

日期：2026-10-01。状态：**计划已更新，old 已重建并通过既有回归；新增 M0 IPC v2
候选第二轮 review 未通过，按用户最新指示暂缓；M2 第二轮 review 通过，等待用户确认，
未进入 M3；legacy M0/M1 保留。**

本轮结束更新：old 已完成重建、实际路径隔离、61 项既有/上游回归及两项 DCU GEMM
数值检查。新增 M0 IPC v2 候选第二轮 review 未通过，不能作为已冻结契约。用户随后明确
授权绕过该待办开始 M2：本模块以真实 runtime allocation 和显式参数绑定为边界，不依赖
未通过的 M0 schema，也不顺带修补 M0。见 [升级报告](distributed_fusion_old_upgrade_20261001.md) 和
[M0 IPC 审查报告](distributed_fusion_m0_ipc_review.md)。M2 的真实通信 DSL lowering、内置
IPC 组件安装和四 PE Block publication 已通过，详见
[M2 审查记录](distributed_fusion_m2_review.md)。完整 wheel 尚未验证；以下其余模块仍是
实施目标，不是已完成功能清单。

本文是 TileLang_DCU_old 分布式融合的唯一实施计划。本次用户授权的可写范围仅为 old；
TileLang_DCU_develop、dcu_ipc、dcu_fused_gemm 在本轮均为只读证据/参考，不能复制它们
生成的 HIP 文本或覆盖其文件。主代理已将 old fast-forward 到上游
e19ab375aaee0b364e1d28c838fab5ea8db939ce；构建和既有回归已完成。TVM gitlink 仍为
79ed747，用户子模块不动；升级前本地 patch/source 备份位于
.cache/tilelang-old-upgrade-20261001.C13o6g，stash 保留。新 DCU 示例已随上游进入 old，
不另做功能 backport。

## 1. 范围、版本隔离与历史冻结

旧计划的 ROC_SHMEM/v1 设想已经过时，不能只换库名。新路径以 dcu_ipc 的 HIP IPC
映射、每 allocation 的设备端 peer 基地址表和内联 DeviceView 地址计算为基础。本计划
新增独立 **IPC v2 profile**，不改变、扩展或重释 v1/legacy ABI1。

| 项目 | 冻结 v1（历史） | IPC v2（本计划） |
|---|---|---|
| 通信后端 | 原 M0/M1 的 ROC_SHMEM 设计记录 | dcu_ipc HIP IPC；不链接 ROC_SHMEM/dushmem/RCCL |
| ABI | legacy ABI1，保持原样 | IPCKernelABI ABI2，独立 Python 描述与绑定 |
| 首个参考 | 不重新承诺其扩展 | 四 PE、batch=1 的 AR/RS/AG |
| 生成方式 | 历史提案 | TileLang IR/宏生成；不得拼接手写 HIP 文本 |
| 完成 | 历史契约 | 区分 output ready、无远端 reader、flags 可复用 |

- M0/M1 的 v1 文本保留为历史，不迁移成 IPC v2 要求，也不修改 ABI1。
- 2026-09-23 的撤回仅是对 utils/copy 扩展及两项测试的**文本回读**；58 pass 只是
  历史结果。禁止重引入该扩展、两项测试，或把该数字称为当前验收。
- 保留旧测试不变量；不能为 IPC 功能弱化、删除、跳过或伪造旧测试。
- v2 首期只承诺新的四 PE、batch=1 reference；既有/历史 1 PE、2 PE 和 batch
  扩展不在承诺内，需后续单独设计、实现、验收。

M2 已复用 M1 DSL/IR 接入 IPC GPU backend，并验证单机四 PE / gfx936 / FineGrained
payload 与 flags 的 Block 发布；不声称全平台内存模型保证或完整融合算子已经集成。
显式 HIP target 的正常 lowering 已验证 source-only，不调用设备编译器或设备探测；
手写 .hip 或“只跳过最终 subprocess”的脚本仍不能代替这一能力。跨节点、SDMA、
VMM、任意超额 persistent grid、自动 MPI 初始化不在首期。

## 2. 已审计事实和设计影响

### 2.1 dcu_ipc

dcu_ipc 已完成 I0--I6 和线程级 S1。应用拥有 MPI_Init_thread(MPI_THREAD_FUNNELED)；
init_mpi、symm_alloc、sync_all、symm_free、finalize 须由所有 PE 同序集体调用。首期以
单一活跃 context 使用，不能用模块全局状态覆盖别的 communicator。

现有 DeviceView 是可平凡复制的 inline kernel 参数：

~~~text
local_base: byte*                 peer_bases_device: byte* const*
nbytes: size_t                    my_pe / npes: int
allocation_id / generation: u64
~~~

peer_ptr<T>(view, byte_offset, peer) 的 offset 是 allocation-relative **bytes**，由 peer
base 加 offset 形成 typed 指针；它不是全局对称 VA，不能假设两个 allocation 连续。对称
分配可为 FineGrained 或 DeviceDefault，memory kind 属于 profile 契约，不能让编译器猜测。
地址 helper 是 inline，故仅为它不要求外部 device function 或 RDC；但 Python/PIC .so bridge
仍须单独做编译、加载、符号审计。

S1 的 fence_sys、release store、acquire wait 仅实测 **FineGrained payload +
FineGrained flags、每槽单 writer、单活跃调用线程**。它不证明 CTA 其他线程的写可被
thread0 发布，也没有 Block 协作、ACK/epoch 复用或 DeviceDefault payload 的通用语义。
v2 不得以 S1 替代 Block publication 验证。

### 2.2 新参考 profile

| profile | 局部计算 | 通信/输出 | 调度与生命周期 |
|---|---|---|---|
| AR v2 | FP16 GEMM、FP32 accumulate，写 local tile-major FP16 partial | cyclic owner pull 四份 partial，FP32 sum，转 FP16 后 push 到所有 PE 的 row-major output | owner=tile_id%P；device handshake 后本卡返回；reset 前已有 collective boundary |
| RS | 同一 local partial | 连续 row-shard owner pull/sum，只写本地 [M/P,N] row-major | producer 是 owner-interleaved，输出归属仍连续 |
| AG | row-sharded A[M/P,K]、column-sharded B[K,N_local] | pull 完整 A 后算本地 [M,N_local] | 256 行一组、65536 half/chunk；SYSTEM ACQ_REL counter；通信 CTA 完成后 join compute queue |

参考共同使用 512 threads、64 KiB dynamic LDS、256x256 tile、block_K=16、4-stage pipeline。
--gpu-max-threads-per-block=256 而实际 launch 512 是 DTK/gfx936 已测绕行，只是
reference-only 风险与完整验证前提；绝非可移植默认值或一般 HIP 合法性结论。

AG 的完整 A 是仅本地读写的 DeviceDefault；可被 peer 访问的 input/flags 是
FineGrained。row counter、queue、completion flags 的槽位和含义必须分离。其 SYSTEM
ACQ_REL RMW 是 M1 的未来实现项，不能因 S1 被列为已完成。

参考内核虽有通信完成 handshake，flags reset 仍要求上轮 collective boundary。因此 ABI
和测试必须分别表示：(a) output ready，(b) no more remote readers，(c) flags/workspace
reusable；任一个均不能代替另两个。

## 3. M0：独立纯 Python IPC v2 契约

后续在 old 内新增彼此独立的纯 Python 模块 tilelang/distributed/ipc_spec.py 与
tilelang/distributed/ipc_schedule.py；不修改 legacy spec.py/schedule.py。它们不导入 HIP、
MPI 或 dcu_ipc，也不宣称已有 GPU backend。公开类建议为：

~~~text
IPCFusionSpec              # 算子、shape、dtype/layout、PE 数、profile、约束
IPCSchedulePlan            # work/tile/owner/slot/ready/queue 的纯映射
IPCKernelABI(version=2)    # 参数、DeviceView binding、launch、completion/reset
~~~

legacy ABI1 不动。M0 首个 validator 只接受 nranks=4、batch_tiles=1 和 reference 精度：
FP16 input/partial/output、FP32 accumulate/reduce。不能用“无 local partial”作 blanket
hard gate，因为 AR/RS v2 有 local partial；应拒绝未声明的 batch、PE 数、对齐、memory
kind、launch、layout 组合，而非静默退化。

### 3.1 IPCFusionSpec 和 memory roles

规格包含 profile in {ar_v2, rs_v2, ag_v2}、nranks、M/N/K、rank-local input shard、tile、
threads、LDS、stages、每个 precision/layout/rounding、owner/queue 策略和 completion/reset
policy。以下 roles 是固定枚举且各 profile 必须完整绑定：

~~~text
localpartial   仅本 PE 写，供 AR/RS owner pull 的 tile-major FP16 partial
sharedinput    peer 可读的 FineGrained 输入（AG A）
gatheredA      完整 A；DeviceDefault，仅本地读写
output         声明为 row-major 的结果及其 writer/reader
flags          FineGrained ready/counter/queue/completion/reset slots
~~~

### 3.2 IPCKernelABI ABI2

每个 binding 记录 role、argument position、DeviceView allocation id/generation、memory kind、
dtype、logical shape/strides、alignment、allocation-relative offset_bytes、nbytes、local/peer
permissions、layout、precision 和 rounding。元素数量与 bytes 永不混用。manifest 还记录：

- grid/threads/dynamic LDS、device profile、512-launch workaround 风险位；
- arithmetic/communication precision、tile-major partial 到 row-major output 的转换点；
- IPCSchedulePlan 公式与 specialization hash；
- release/acquire/CTA barrier publication relation、每 slot writer/reader、completion
  handshake、reset 前 collective boundary；
- output-ready、no-more-remote-readers、flags-reusable 三个独立条件；
- hidden descriptor/context bindings 的生命周期。

ABI2 只描述已有 dcu_ipc::DeviceView，不改 C++ struct、不为 inline helper 引入 RDC。
descriptor binding 必须核对 id、generation、kind、size、alignment、offset，不能只检查指针非空。

### 3.3 IPCSchedulePlan

同一个不可变对象提供 producer/consumer 映射，至少包含：

~~~text
tile_id <-> (tile_row,tile_col) <-> logical (bx,by)
AR: owner=tile_id%P、source localpartial slot、cyclic consumer
RS: row-shard owner、连续 output row、owner-interleaved producer
AG: A row group -> chunk range -> counter；ready row -> queue claim -> GEMM tile
~~~

M0 CPU 测试枚举小尺寸有效计划，验证双射、范围、唯一 writer、consumer coverage、row/chunk
count、queue 互斥、alignment/offset 和非法配置。该 PASS 只是 schema/数学验证，不代表 GPU
atomic、Block 发布或 persistent 活性。

## 4. 实现模块、隔离路径和验收

### M1：复用现有 DSL/IR，补 IPC metadata 和 counter

legacy M1 的公开 DSL 到 TIR、typed remote view/copy/signal/wait、normalize/verify 已完成
验收；这些能力并不绑定 ROC_SHMEM，IPC v2 必须优先复用，不能以“新 v2 DSL”重造或撤销
它们。IPC adapter 只按真正缺口补 memory role、DeviceView descriptor provenance、ABI2
binding、schedule/completion metadata，以及 AG 所需 SYSTEM ACQ_REL counter RMW；最终远端
地址仍是 DeviceView+byte offset。无能力时明确诊断。

M1 的回归要求是公开 DSL 到 TIR、normalize/verify 和 ABI/plan JSON 往返，连同参数 binding、
role/layout/byte-offset 保留。文本打印/回读不再是要求，2026-09-23 对文本回读的撤回继续
有效。scope/effect 跨 pass 安全和 divergent-barrier 不是 M1 已证明的事实，归入 M2/M3
及最终集成门禁。AG counter RMW 在本模块才是未来实现，不能提前标注完成。

### M2：IPC device adapter 与 Block litmus

2026-10-01 用户追加要求：所需 HIP IPC 源码直接纳入本仓库，随 TileLang 构建和安装，
不得链接用户目录中单独构建的 dcu_ipc 二进制。允许将 M5/M6 的最小 context/allocation
桥接、PIC runtime 与安装规则提前到 M2；完整 profile ABI、DLPack、分布式 compile_once
和缓存仍留在后续模块。DTK/HIP 和 MPI 是系统工具链依赖，不是另行安装 IPC 的步骤。
普通 GEMM 不应强制加载 MPI；通信 runtime 按需加载。内置源保留来源说明。

M2 必须把既有 DSL 经真实 TIR lowering 接到内置设备 helper，不能只给出手写 HIP 测试。
覆盖运行期 rank/world size、remote_view/普通 copy、typed put/get、fence、signal/wait 和
既有 scoped scalar 支持集；无法实现的输入必须明确拒绝，不能退化为本地访问。
代码导出、真实多 PE 执行、安装产物均须检查；不以完整 AR/RS/AG 融合算子作为本模块范围。

新增 HIP adapter/helper，使用内置的 IPC inline peer_ptr/ptr，于元素循环外绑定 typed
peer pointer，保留 allocation-relative byte offset。不得 include/link ROC_SHMEM，或假设
allocation 连续。先实现真正 Block-level litmus：

~~~text
all payload writers store -> each writer system fence -> CTA barrier
  -> thread0 release publish
consumer thread0 acquire wait -> CTA barrier -> whole CTA reads payload
~~~

它必须有随机 payload、延迟、重复 reset、错误 PE/slot 记录和 timeout，且独立于 GEMM。
M2 同时验证 release/acquire/CTA effect 作为可见副作用不会在 IPC lowering 与后续 pass
之间丢失或穿越其同步边界。仅 M2 通过后才可声称目标组合验证了 Block publication；
S1 仍仅是 S1。

2026-10-01 验收：第二轮正式 review 通过。16 项独立检查、24 组标量真实 HIP 编译、
最终 LLVM 同步指令检查、四 PE copy/get、全 12 方向 32 epoch × 8 batch Block litmus、
已安装 IPC 组件独立执行、61 项既有回归及两组 GEMM 数值检查通过。整包 wheel 的
TVM Python/native revision 一致性尚未验证，ctypes 持久缓存亦不在本次支持范围。
M0 候选不修改，M3 必须等待用户另行确认。

#### 2026-10-06：IPC remote pointer 的局部外提（验收通过，非 M3 完成）

`LowerDCUIPC` 已把已 `bind_ipc` 且 ABI-ready 的远端 view 规范化为
`tl.ipc.remote_address(descriptor, peer, rank, world)`。在 scheduled RS 的通信元素循环中，
同一 tile 内四个 payload peer 的 DeviceView/peer-table 基址解析原先逐元素重复。新增
`HoistDCUIPCPointers` 只在 `StorageRewrite` 后、`SplitHostDevice` 前检查该专用 Let normal
form，并仅把 serial 元素循环**直接前缀**的、循环不变 remote-address Let 提到该循环之外；
payload remote load/store、条件输出和循环索引一律留在原位置。它不是 generic LICM，也不
注册 remote_address 为 pure，不修改普通 GEMM、Python 算子、通信算法或手写 HIP。

门控要求 function 同时具有 bind_ipc 写入的 ABI-ready/binding 属性，descriptor 必须以
参数 Var 身份同时出现在 buffer_map 中。peer 可为常量、纯标量参数或已支配的纯外层 Let，
但不能依赖元素 loop、BufferLoad 或 opaque call；rank/world 必须是 ABI 参数。descriptor
在 TIR 中一旦被写、BufferLoad、别名或作为任何未白名单 operand 逃逸即全函数不外提。
`remote_address`、signal/wait/put/get/st/ld 的固定 read-only descriptor operand 是唯一的
ABI 白名单。这个狭窄“launch 期间 descriptor 不变”前提来自 runtime：allocation 只在 peer
table 完整后 publish，runtime bridge 只上传并缓存该 allocation 的 DeviceView 直到 free；
它不推广为普通可写 pointer 的不变性结论。

只要目标 loop 包含 fence、wait、signal、barrier、unknown/opaque call、nested loop、
allocation 或 While，pass 就不外提。payload-only `If` 可保留，但 pass 不从 If 内抽取 Let，
从而不改变条件执行。未知正 extent 会生成 `if (extent > 0)` 包住外提 Lets 和原 For，且
只有 min/extent 本身没有 BufferLoad/Call 时才允许此重复读取，保持 zero-trip 时原本不发生
address helper runtime check 的语义。该首版不跨 tile/通信分支、不向量化、不缓存 lane 或
payload 数据，也不实现 M3 的动态角色、queue、生命周期或 completion。

第二轮正式 review 通过：8 项新增检查、30 项 M2/后端回归、61 项既有回归；7 份普通/未标记
GEMM 的 HIP 逐字一致，实际 pass 前后 compute branch 对象保持不变。四 PE matched RS、
M2 Block litmus、remote copy/get 和原生 GEMM GPU smoke 通过。2048²×512 与 4096²×1024
融合 kernel 的同节点中位耗时分别从 0.628159/1.273439 ms 降到 0.350240/0.642880 ms。
4096 固定 HIP、仅移动四个指针声明的独立对照支持该性能归因。详细条件、限制与复现见
[通信指针外提验收报告](distributed_fusion_ipc_pointer_hoist_review.md)。本轮停止于此，不推进后续优化。

#### 2026-10-12：外提 PASS 覆盖 GEMM+AllReduce（非 M3 完成）

扩展上述首版的两个限制：地址 Let 不再要求位于 serial 循环的连续前缀，但必须仍为
直接、无条件 SeqStmt 子节点；允许经完整效应审计的小型常量 unrolled/vectorized payload
循环及精确匹配的 `pragma_unroll_explicit` 包装。不得从条件或嵌套循环内部抽取地址，
仍保留 descriptor、peer、zero-trip 与同步边界检查。AR 的四个输入与四个输出 peer 指针
现在均位于 owned-tile 内、ready 等待之后、chunk 循环之前，Python 算子与 pass 顺序不变。

四尺寸完整 dyadic 四卡验证通过，26 项针对性测试与 31 项回归通过；7 份普通/未标记 GEMM
HIP 逐字不变。同节点 AR 从 0.234/0.637/2.331/15.433 ms 降到
0.184/0.554/1.985/10.902 ms。验证范围、未解决事项及最终 HIP 的 LDS 排列差异详见
[AR 通信指针外提扩展报告](distributed_fusion_ipc_ar_pointer_hoist_review.md)。

#### 2026-10-06：IPC opaque thread 通信索引化简（已通过最终 review，非 M3 完成）

scheduled RS 的通信元素循环保留
`comm_tx = tl::ipc::opaque_thread_id(threadIdx.x)`：该空汇编 helper 继续隔离通信坐标，不能把
它替换为原始 thread var，也不能向 TIR/HIP 发射 `assume`、范围 annotation 或 launch-bounds。
普通 `Simplify` 会因该 call 非 pure 而不绑定 `comm_tx`，所以即便实际是 512 线程，原先仍保留
`comm_tx < 65536`、非负 row/output guard 和符号 trip count；这不是 rank/world 的编译期事实。

本轮只扩展既有 `tl.transform.Simplify`，不新增 pass，也不改 phase、DSL、IPC 算法、signal/wait/
fence、规约顺序、GEMM 或 lane/warp 路径。仅同时满足 ABI-ready 且有 binding、HIP `gfx936`、当前
`thread_extent` 为可表示的正 `int32` 常量、并且 Let 精确为
`call_extern("tl::ipc::opaque_thread_id", 当前 threadIdx.x Var)` 的 `int32` 标量形式时，Analyzer
才在该 **Let body 的词法范围**按 Var identity 使用 `[0, launch_extent)`。未知/零/负/错误 dtype
extent、For thread-binding、嵌套动态 `threadIdx.x`、不同 helper、非 Var 实参、非 HIP/非 gfx936/
缺 ABI 属性均回到原 Simplify 行为；target 的 `max_num_threads=256` 绝不能替代真实 launch 的 512。

既有 Analyzer 负责消去经该局部事实证明的恒真条件。它不会自动把所有单值 `FloorDiv` 改常量，
因此只对同一 scope 内、含该 Let Var 的 serial For extent，在表达式为纯标量 signed integer、
`ConstIntBound` 是单值且 `CanProveEqual` 同时成立时物化常量；没有手写除法 rewrite。rank/world
没有范围注入，rank-dependent tile predicate 必须保留；非整除尾部、含 BufferLoad/opaque call 的
extent 也不折叠。实际 scheduled 形状可据此把 `(66047 - comm_tx) // 512` 证明为 `128`，但这不
推广为固定 rank/world 或通用通信优化。

专项 CPU/真实 source-only 验证覆盖 helper/参数/target/ABI 门控、512 launch 对 target-256 的高半
线程、scope identity、动态/非法 extent、非整除尾部、零 trip 与 side effect、rank guard 及原 helper
保留。最终统一运行 105 项测试全部通过；四 PE scheduled RS、2048 连续输入、4096 的 24 comm
配置、M2 block litmus、remote copy/get 与原生 GEMM GPU smoke 通过。最终 2048/4096 JIT 的
compute/wait/completion HIP 与修改前逐字一致；原生未标记示例的既有 LDS 布局非确定性已用
旧二进制独立复现，未放宽比较断言。融合 kernel 中位数为 0.352080→0.213199 ms 和
0.637759→0.450319 ms；固定计算代码的 2048 对照支持通信索引优化归因。
详见[通信索引简化验收报告](distributed_fusion_ipc_index_simplify_review.md)。本轮不据此标记 M3
完成，不声称向量化、通信协议变化或一般设备支持；停止于本项优化。

#### 2026-10-06：scheduled RS 通信 half2 访问（实施中，非 M3 完成）

本项只处理 `HoistDCUIPCPointers` 之后的 scheduled RS 通信元素循环；不是通用
`VectorizeLoop`，不改变普通 GEMM、GEMM compute、IPC 算法、signal/wait/fence/barrier、
`opaque_thread_id` 空 asm 或 FP32 加法的左结合顺序。候选必须同时具有 ABI-ready/binding、HIP
gfx936、`threadIdx.x=512`，并精确匹配四个已外提 `remote_address`、128 次 serial 的四条
predicated FP16 pull、四个单元素 local temp 和一次 FP32 `p0+p1+p2+p3` 到 FP16 output 的
normal form。任一 alias、未知 scope/shape/stride/offset/alignment、buffer predicate、opaque/
副作用、跨元素依赖、非 unit serial step、非 uniform 控制流或非 unit y/z 均保持原样。

fast path 用新 `j in [0,64)` 和原 `comm_tx in [0,512)` 的两 lane 映回旧域：
`old_i=2*j+comm_tx//256`、`old_tx=2*(comm_tx%256)+lane`。因此不是错误地扩大旧 tx 到
1023；两个 lane 分别对应旧的 `(old_i,old_tx)` 与其相邻元素。Analyzer 必须证明每个 source
是 `base+old_i*512+old_tx`、output 是从实际 row stride 取出的
`base+(linear//256)*row_stride+linear%256`，并证明两个 lane 的 source/output 索引相邻、
lane0 偶数、source 在原 rank predicate 下及 output 在全域内均不越界。这给出 128×512
旧 writer 的覆盖和唯一性，不假设 runtime rank/world 范围。

TIR 的 Buffer alignment=64 只是静态前提，不能据不同 Var 或 device-function `noalias`
推断 remote peer allocation 与 output 不重叠。故 fast path 每 tile 调用无 payload access、无
同步的 `tl::ipc::can_vectorize_half2`：检查五个非空完整 buffer range 的 4-byte 对齐、
非零/可表示区间和 output 与任一 remote range 的不重叠，采用无加法溢出的差值比较；失败时
执行保留的原 scalar For。remote-remote 可重叠，因为它们均只读。通过后仅生成
`Ramp(base,1,2)` FP16 load/store 与 lane-wise FP32x2 左结合求和，复用现有 HIP vector printer；
本项不引入通用指针 alias 假设或修改 codegen。当前默认关闭，只能以 PassContext
`tl.enable_dcu_ipc_vectorize=true` 显式研究性启用；未设置时 phase 不调用该 pass，普通及既有
IPC 生成保持原路径。初次等价验证虽通过，但同 launcher 的真实性能未达到验收，故不作为默认
优化、M3 完成或一般向量化能力宣称；仅保留严格 matcher、scalar fallback 和专项验证供后续
单独研究。最终 105 项既有回归和 4 项新增专项测试通过；四 PE 正确性与错位回退验证通过，
但性能回退，按两轮 review 约定暂停。详见[向量访存实验报告](distributed_fusion_ipc_vectorize_review.md)。

#### 2026-10-13：AR 标量前端接入既有 half8 向量化 PASS

在既有 RS half8 实现上增加 AR 专用匹配分支：四个输入 peer 基址、四个输出 peer 基址，
四次标量读取、按原 FP32 顺序归约并舍入一次、向四个 rank-rotated peer 写相同结果。
前端只写 serial 元素循环；后端证明源连续布局、目标完整矩阵的行跨度、范围、对齐、
线程映射全覆盖及临时量唯一使用，再生成 half8 load/store。使用词法作用域内的 rank/owned
assume，不将事实泄漏给 sibling；不加设备侧运行时安全检查或 fallback。
未匹配的模式保持原样，RS 路径与普通 GEMM 不变，不新增 PASS 或改变 pass 顺序。

真实 JIT 必须用 `tilelang.compile(pass_configs={"tl.enable_dcu_ipc_vectorize": True})` 传递开关；
仅包外层 PassContext 不会覆盖 JIT 内部新建的配置。本轮同时修正 AR benchmark，并在计时前
核验真实生成 HIP，防止把未启用 PASS 的标量代码计作向量性能。测试及同节点性能见
[AR 自动向量化报告](distributed_fusion_ipc_ar_vectorize_review.md)。这仍不是 M3 完成。

#### 2026-10-13：TileLang AllGather + GEMM 原型（非 M3 完成）

新增 `examples/distributed/ag_gemm_prototype/`，以现有通信 DSL 和 `T.gemm` 生成单个
四 PE AG+GEMM kernel；不拼接手写 HIP。A 行分片、B 列分片，支持
M=K=2048/4096/8192/16384、N_local=M/4。固定 80 CTA、512 threads，通信 CTA
分别为 32/32/24/16。复用 AR/RS 的固定角色，而不是手写 AG 的动态 compute queue；
每 65536-half chunk 使用独立 single-writer ready flag，替代尚未接入 DSL 的 row counter。
因此不将该原型记为 M3 动态调度或通用 AG counter 接口完成。

前端通信为标量线性遍历。既有 Hoist PASS 扩展至词法外层 serial/thread 变量决定的
不变 peer；Vectorize PASS 新增严格单远端到本地 copy 模式，证明完整范围和对齐后
生成 half8。真实 JIT 显式传递 `pass_configs` 并检查 HIP 中指针外提、uint4 copy 生效。
四尺寸四卡 full dyadic gathered-A/output、poison 双 epoch 与计时后检查通过；
2048 continuous 和 scalar 对照通过。详细范围、回归和运行证据见
[AG+GEMM 集成报告](distributed_fusion_ipc_ag_gemm_review.md)。

同日追加小尺寸性能修复：AG 2048/512/2048 保留 48 compute / 32 comm，
有效计算 CTA 使用范围判断 + 显式恒定一次的标记 tile 循环；资源 lowering 仅补充
`min=0, extent=1` 常量循环的窄例外，仍保留原 GEMM/async-copy/一致性门禁。
消除动态 0/1 次循环引起的大量 spill 后，同场两轮约 0.137 ms，手写约 0.136 ms。
没有新增动态调度、通用范围分析或改变未标记原生 GEMM。
见 [小尺寸性能报告](../../examples/distributed/ag_gemm_prototype/AG2048_PERFORMANCE_FIX.md)。

### M3：persistent 动态角色和资源

实现 compute/communication 动态角色及 work queue，不把 80 CTA、四 PE 或 comm 数硬编码进
codegen。roles、owner/interleaved order、queue claim、CTA barrier 均来自 IPCSchedulePlan，
barrier 参与条件必须 block-uniform。AG 通信 CTA 完成 gather 和 reader completion 后，和原
compute CTA 一起领取本地 GEMM tile；queue 不得与 row-ready 或 completion 共槽。

先以 tag/synthetic payload 验证活性、角色覆盖、尾任务、无 divergent barrier、LDS 生命周期；
集成时验证 normalize、layout/pipeline/DCU pass 不跨 scope/effect/sync 边界移动 IPC 操作。
再查 registers、spill、LDS、occupancy、persistent 全驻留。512-launch workaround 仅能在
gfx936 专用 profile 完整验证，不得变成 portable default。

#### 2026-10-06：显式 tile-resource 地址分解（局部后端工作，非 M3 完成）

为修正 owner-interleaved persistent GEMM 的实际地址路径，`LowerSharedGlobalCopy` 增加了
仅由 `T.serial(..., annotations={"tl.dcu_persistent_tile": 1})` 触发的局部处理；它不猜测
任意 serial loop，不标记既有 GEMM，也不识别通信 serial loop。未标记 PrimFunc 保持原有
collector/injection 分支，普通 native GEMM 的生成契约不变。

标记必须包住真实的、内含 DCU GEMM K loop 的 block-uniform 输出 tile 遍历。资源 Let 在该
tile loop 内、tile 坐标 Let 之后且 K pipeline 之前建立；A/B 继续使用既有 uppercase
idxen 识别。每条 g2s copy 从原始 scalar source offset 减同一精确 resource base，随后用
绑定范围的 Analyzer 同时证明 residual 非负、且不再（经 uniform Let 展开后）依赖 tile 或
block 坐标；通过后的简化 residual 被缓存并直接写入 TIR，避免无范围二次化简改变
floor/mod carry。无法证明、base 含 load/不纯 call/thread/inner-K 状态、marker 误落 K/非
GEMM loop、控制流可能 thread-dependent，或 source buffer 在 marker 内部定义时都明确拒绝。

该局部工作不实现 work queue、CTA 动态角色、IPC 生命周期或 M3 的任何完成条件；通信
`opaque_thread_id` 路径及 native mbcnt/warp 保护均不因此修改。修改动机是已复现的
4096²×1024 raw-lane A 行错误，以及 tile-base + 直接 local-row 在四 PE dyadic 的通过证据，
不是为满足文本/单元测试而改变普通 GEMM。

本局部功能于 2026-10-06 第二轮 review 通过：11 项专项、61 项既有回归，原生 GEMM
5 组 exact-HIP 对照及两组 GPU smoke，通过真实 JIT 的 persistent GEMM/四 PE RS
验证。额外 unmarked persistent 的旧 LDS 排列非确定性和 4096 continuous RS 的已知
容差边界均保留，不据此扩大验收结论。详细作用域、复现命令及证据见
[persistent resource 验收记录](../../examples/distributed/gemm_rs_prototype/PERSISTENT_RESOURCE_BACKEND.md)。
本功能通过不改变 M3 的待实现状态。

#### 2026-10-06：verified tile 的 scoped lane 清理（验收通过，非 M3 完成）

在上述 resource/local-offset proof 已成功的同一个 marker loop 内，已实现对 gfx936、
一维 `threadIdx.x=512` 的窄 profile 让 `InjectWarpIdx` 用 `threadIdx.x % 64` 生成 lane 坐标，
以消除该已对齐 persistent body 的 `get_lane_idx`/mbcnt 调用；`get_warp_idx_sync` 仍保留。
唯一门控是 frontend marker 加 LowerSharedGlobalCopy 成功写入的内部 verified token，二者缺一
不可启用。loop 外 sibling、普通/未标记 GEMM、通信 opaque thread id、非 gfx936、非 512 或
active y/z 维度均保留原行为（非 DCU target 原本不执行该 pass）。全局 HIP template、
idxen 与普通 warp pass 语义不变。

这是对已验证生产地址路径的局部清理，第二轮 review 已通过：14 项专项、61 项旧回归、
7 份 unmarked exact-HIP、纯 persistent GEMM 与四 PE matched GPU 验证通过；4096 静态
mbcnt 从 337 降为 0。不创建 lane cache，不重构通信，也不构成 M3 的动态角色、queue
或生命周期实现。详细结果见上述 persistent resource 验收记录的 lane 清理节；停止在本项。

### M4：TileLang GEMM compute macro

从 examples/dcu/example_gemm.py 的 T.gemm 路径提炼可组合 TileLang 宏：FP16 input、FP32
fragment accumulate、tile=256x256、block_K=16、4 stages、512 threads、64 KiB LDS，并接受 ABI2 typed
destination/layout。禁止解析、替换、拼接 reference .hip 的 blockIdx、内嵌 compute_tile 或
HIP 文本。

分别验收普通 row-major GEMM、AR/RS local tile-major partial、AG row-major local output；
用非均匀数据和 CPU/rocBLAS 对照 layout、rounding、offset。新宏不等于既有 JIT example
已经具备 source-only。

### M5：运行时绑定和生命周期

复用 dcu_ipc 的 MPI 所有权、collective 顺序、DeviceView lifetime、memory kind 与 sync_all；
TileLang 不自行 MPI_Init/Finalize，也不篡改 caller context。binding 前检查 context/rank/device、
allocation id/generation、view live、相同 ABI/config hash、四 PE 同机、role memory kind。
view/allocation 在 no-more-remote-readers 且本地 stream 完成前不得 free/reuse。

一轮生命周期是：上轮 reader-safe -> collective boundary -> reset flags/queue ->
collective boundary -> launch -> device handshake 输出 output-ready 与 reader-safe ->
调用方按 ABI 使用/释放。local kernel return、HIP event 或 output-ready 任一项都不单独许可 reset。

Python Tensor/DLPack 导出是借用 view：保留 allocation owner，外部 view 不得自行 free，且
必须在 stream 完成和 reader-safe 后才允许 allocator 释放。分布式 out_idx 不得继续默认
走普通 torch.empty；需有对称/本地 role-aware output allocation 策略和明确借用生命周期。

### M6：构建、source-only、capability cache

实现时才新增 IPC manifest、source-only exporter、PIC bridge smoke test。cache key 至少有
ABI2、profile、shape/layout/dtype/rounding、schedule hash、mcpu、DTK/HIP compiler、dcu_ipc
ABI identity；进程必须重新绑定本地 DeviceView，不能缓存 peer VA。IPC profile 无 ROC_SHMEM
链接；纯单卡 GEMM 不增加 MPI/IPC 链接或运行时依赖。

source-only 必须从真实 PrimFunc/IR 生成 HIP、ABI manifest、审计用编译命令，不创建 JIT
adapter、不在导出时暗中编译；实现前始终是待办。

保留原计划的 compile_once 职责：可由组内 root 编译、原子写 capability cache，再把成功或
异常传播给所有 rank；各 rank 各自加载 artifact 并绑定本地 context/DeviceView。先允许每
进程独立编译并验证，再实现跨 rank 协调；root 失败、cache/ABI 不匹配必须全组明确失败。

### M7：AR/RS/AG ABI2 profiles

1. **AR v2**：local FP16 tile-major partial，cyclic owner pull 四份、FP32 sum、FP16
   row-major push all；验证 output-ready 和 reader-safe。
2. **RS v2**：同一 partial/precision，连续 rows owner、owner-interleaved producer，
   只写 local [M/P,N]；不广播。
3. **AG v2**：A[M/P,K]、B[K,N_local]；256-row 组按 65536 half chunks pull，SYSTEM
   ACQ_REL counter 发布，compute acquire 后逐 row 推进；gatheredA 是 local DeviceDefault，
   input/flags FineGrained，comm roles join compute queue。

每 profile 明示 localpartial、peer readers、output writer、completion signal 和 reset
boundary；不得混用 reference 的 slot/flag formula。

### M8：验收、性能与报告

最终集成仍要求 M0 CPU schema 验收；本次用户允许 M2 独立于待修 M0 先实施。
设备验证顺序是 M1 IR -> M2 Block litmus -> M3 synthetic persistent -> 新四 PE
batch=1 end-to-end。保持 legacy 测试不变量；不为“测试通过”修改原有源码。端到端测试需
rank-dependent 随机/解析输入、poison、多轮、延迟一个 producer/reader、独立 partial/
workspace/output/flags 校验，并在无预先跨 PE host barrier 情况下验证所声明 completion。

仅可在同一 PE 数、shape、shard、precision、layout、rounding、计时边界、reset/completion
包含关系下比较。三类基线固定为：(1) 当前手写 IPC reference，(2) 同一 GEMM 与同一通信
算法的串行版，(3) rocBLAS+RCCL。报告 kernel 区间和包含 prepare/reset/规定 collective
completion 的完整 end-to-end 两种时间、max-PE 统计、register/spill/LDS/occupancy、设备/
编译器、profile/schedule hash。建议生成版同批 reference 的 max-PE median 不慢于 10%；
它不是历史绝对门槛。RCCL FP16 与本 profile 的 FP32 中间归约可有不同 rounding/add order，
但数学同义，报告必须声明相应误差界，不能要求所有基线逐位相同。历史数字、不同 layout、
不同计时口径、未等待 readers 的程序不得用于性能结论。

## 5. Lowering 顺序和完成门禁

~~~text
IPCFusionSpec / IPCSchedulePlan / IPCKernelABI(ABI2)
  -> ValidateIPCProfile（roles、kinds、shape/layout、completion/reset）
  -> bind typed DeviceView descriptors and provenance
  -> TileLang GEMM macro + IPC effects
  -> existing layout/pipeline/DCU passes（不得跨 release/acquire/CTA 边界）
  -> lower IPC view/sync/counter to dcu_ipc inline helpers
  -> VerifyIPC lowering（byte offsets、uniform barriers、role/slot coverage）
  -> HIP source + ABI2 manifest + optional actual source-only export
~~~

远端 access 不得被降成无 alias 字符串 call，flags 也不得被当作普通可重排 load/store。
memory-kind、Block litmus、PIC bridge、occupancy、completion 任一验证失败即停在该 profile；
不能以手写 HIP fallback 或放宽同步宣称完成。

## 6. 交付判据

- [ ] ABI1/v1 冻结，ABI2 纯 Python M0 的 schema/公式/错误诊断通过。
- [x] M2 Block publication litmus 独立通过；S1 不再被误作其替代（限上述目标组合）。
- [ ] compute macro 由 TileLang IR 生成，未拼接 reference HIP。
- [ ] AR/RS/AG 以四 PE、batch=1、同精度/layout/计时契约通过，并报告 kernel 与完整
  end-to-end 两种区间及三类同语义基线。
- [ ] output-ready、reader-safe、flags-reusable 有不同字段、测试、生命周期边界。
- [x] IPC build 无 ROC_SHMEM；纯 GEMM 不加载 MPI/IPC；正常 source-only lowering 已验证。
- [ ] 报告不把历史 58 pass/性能说成当前结果，也不夸大 gfx936 workaround 的可移植性。

本文件记录实施设计与顺序，完成状态以文件顶部及独立验收报告为准。本轮保留了未通过
review 的 M0 IPC 候选与失败测试，没有创建 Git commit。本轮 M2 已通过并报告，
等待确认，不自动进入 M3。
