# M2 审查记录（第二轮通过）

日期：2026-10-01。当前状态：第一轮退回项已修正，主代理第二轮独立复测通过。
本轮停止于 M2；M0 仍暂缓，M3 未开始，无 Git commit。

## 范围和依据

用户明确暂缓 M0 IPC 候选问题，授权先做 M2，并要求通信源码纳入 TileLang 安装。
本轮不修改 M0，不将其候选 schema 作为已通过契约。实现子代理使用
gpt-5.6-terra / high，禁止继续分派；主代理负责设计与独立审查。

实施路径为已有通信 DSL → 显式 runtime 参数绑定 → 真实 TIR lowering → HIP 内联设备
实现。内置 HIP IPC runtime 单独构建、按需加载，不依赖原 dcu_ipc 目录的二进制。
用户仍需提供 DTK/HIP、MPI 工具链；这与另装 IPC 组件不同。

## 原有生产源码修改的必要性门禁

| 位置 | 实际用户功能与未修改时的失败路径 |
|---|---|
| 编译入口/通信 lowering | M1 仅注册并验证 IR；现有通信 op 没有设备实现，普通 lower/compile 不能执行这些调用。必须接入正常编译路径。 |
| HIP codegen/设备模板 | 将通信 op 生成真实地址访问和系统级同步，而非未定义函数或本地访问；独立于测试是否存在。 |
| JIT 参数封装 | 新增 descriptor/rank/world 是真实 kernel 参数；如果 adapter 仍拿原函数签名，launcher 与设备函数参数不一致。 |
| HIP 编译头文件搜索 | 生成 HIP 引用了仓库内置 IPC helper，正常 JIT 必须能找到随包提供的头文件，不能要求用户手工指定外部 IPC 路径。 |
| CMake/安装规则 | 用户明确要求安装 TileLang 同时构建 IPC；仅引用外部库或手工单独编译不满足要求。 |
| 既有通信 IR 校验 | cooperative put/get 只执行线程 0 时会漏掉其余元素，CTA barrier 的不一致参与会挂起。对 If/For/While 使用 Let-aware uniformity 白名单，无法证明则拒绝；这是实际执行正确性要求，不是文本回读扩展或任意控制流的完整证明。 |

具体实现路径见下方第二轮记录和实现报告。不为测试工具扩展 utils.py/copy_op.py，
不改变曾撤回的文本回读要求。

## 验收清单

- [x] 真实公开 DSL、正常 lower/compile 路径，不拼接参考融合 HIP。
- [x] rank/world size 来自运行期参数；绑定依据活跃 allocation，不依赖 M0 候选。
- [x] remote_view、copy、put/get 的 provenance 和偏移检查通过（padded stride/elem_offset 为 source-only 验证）。
- [x] peer 指针计算不在每次 block copy 的逐元素循环内重复；动态 peer 保持运行期表达式。
- [x] release/acquire、每 writer fence、CTA 同步在最终 LLVM 中保留。
- [x] 多 writer 四 PE / 全 12 方向，非均匀 payload、延迟、重复使用/reset、保护区与 timeout。
- [x] 已覆盖错误目标、缺失绑定和不支持控制流/布局的明确拒绝，不静默变为本地访问。
- [x] 内置 runtime 的 MPI 所有权、allocation 与 descriptor 生命周期检查通过。
- [x] CMake 组件安装产物可独立找到自身库和头文件，无外部 dcu_ipc 二进制依赖（非整包 wheel 验收）。
- [x] 普通 GEMM 不加载 IPC/MPI；既有 61 项回归及真实 GEMM 数值回归。

## 已知独立待办

开始实现前，主代理在 Slurm 9505 / dcu0 四卡 allocation 中复测：路径隔离 PASS、
既有测试 61 passed（9.74 秒）、512×768×256 与 1024×512×512 两组生成 GEMM 的
真实 HIP 执行均 PASS（最大绝对误差 0.00390625）。日志：
`build-old/logs/m2-baseline.log`。这是基线，不是 M2 验收结果。

M0 IPC 候选的第二轮失败测试保留，不计入本模块通过项，也不为得到全绿而删除。
当前 old 开发构建按前轮隔离方案使用匹配的 TVM 79ed747，用户 submodule checkout
仍为另一版本。整个 wheel 的 TVM Python/native 一致性必须单独核查；组件安装通过
不等于已证明整个 wheel 正确。

## 正式审查轮次

### 实现中独立组件检查（不是 M2 整体验收）

已使用 CMake 的 `dcu_ipc` component 安装到
`build-old/m2-install-review/tilelang`。独立程序
`testing/python/distributed/m2_install_review.cpp` 仅使用该目录的 include/lib 和系统
MPI/HIP 编译；Slurm 9505 中四进程执行 exit=0，四 PE 均打印
`PASS installed-component`。检查包括：

- 集体分配的大小不一致被拒绝，随后正常分配成功。
- rank/world、byte offset、越界和非法 peer 检查。
- 活跃 allocation 阻止 finalize；释放后原 descriptor 不再有效。
- C ABI 第二次创建 context、上传 DeviceView、正常释放。
- runtime finalize 不结束调用者的 MPI。

日志：`build-old/logs/m2-installed-smoke.log`。安装后的 ELF NEEDED 为 MPI、HIP 和系统
C++/C 库，没有外部 dcu_ipc 库。MPI 的 CUDA-aware/xpmem/hcoll/ucc 可选组件产生警告，
本次 fallback 仍通过；日志未删除或隐藏。此结果不包含设备通信 kernel，也不是 wheel 验收。

后续扩大组件检查时发现 descriptor 上传沿用了 caller 当前 GPU，而非 context GPU；
保留失败日志 `m2-installed-extended-diagnostic.log`。修复 DeviceGuard 和线程检查后，
新增的 context-device 分配、caller device 恢复、缓存命中也拒绝非 owner 线程检查全部
通过，四 PE exit=0；日志 `m2-installed-extended-retest.log`。

公开 remote_view + T.copy 的完整 normal lowering/source-only 已通过；开发中独立
测试 7 passed（3.81 秒），日志 `m2-independent-development-retest.log`。独立测试早期
有一项 fixture 错用未声明 PointerType 的 Var，已只修正测试自身；这不是生产缺陷。
上述是部分能力检查，其他 primitive 的正向编译和四 PE Block litmus 尚待完成，
不能依据这 7 项结果宣称整个 M2 正确。

后续独立检查：

- 8 项初始编译/绑定检查通过（`m2-independent-strict.log`）；另发现仅查询 rank/world
  的函数在 SplitHostDevice 后丢失参数，已保留失败日志和直接表达式复测结果。
  仍继续检查 rank 在算术、条件及索引中的完整重写，不能仅覆盖直接赋值。
- `int32/uint32/float32 × gpu/sys × weak/volatile/relaxed/acquire-release`，
  共 24 组实际 ctypes/HIPCC 编译通过；日志 `m2-scalar-matrix-compile.log`。
  这是编译覆盖，不等于 24 组设备内存序均得到硬件验证。
- 带 stride=8、elem_offset=3 的二维 remote copy 导出保留两端 element 13 的物理偏移；
  日志 `m2-padded-diagnostic.log`，这是 source-only 检查。
- runtime 修改后重新构建并安装，四 PE 独立程序再次通过，增加错误线程 free 被拒绝且
  原 descriptor 保持有效检查；日志 `m2-installed-free-retest.log`。
- 实现代理的四 PE public DSL `remote_view + T.copy` 已运行通过。更强的多 epoch、
  全 12 方向 Block litmus 仍在编译阶段触发 native TIR SIGSEGV；保留
  `m2-block-litmus-compile-retest.log` 与 `m2-block-litmus-compile-retest2.log`。
  当前不能宣称 Block publication 验收通过。

定位结果：同一复杂函数的 `normalize_remote_views` 单独通过，但新增 Python
`_LowerIPC` / PrimFunc 重建阶段崩溃。已批准改为仓库内原生 C++ lowering；Python 仅
负责显式绑定/参数构造，原生遍历负责 opaque 通信 op、remote alias 作用域和所有表达式
位置的 rank/world 参数替换。该调整解决真实通信函数无法编译的问题，不改变 litmus
为手写 HIP，也不缩减多轮通信要求。旧作业 9505 正常释放，后续使用 9510 / dcu0 四卡。

独立 Python runtime 四 PE 检查亦通过：缓存 descriptor 的非 owner 线程请求被拒绝，
释放后 allocation 和 DeviceView 均拒绝取地址，runtime 不结束 caller MPI。
日志 `m2-python-runtime-review.log`；普通 import 及导入惰性公开 API 不加载 MPI/IPC，
见 `m2-lazy-import-review.log`。

原生 lowering 构建后，主代理独立 11 项检查通过（4.09 秒，
`m2-native-independent-review.log`），包含 rank 算术/条件/索引与 padded offset。
进一步将实际生成 HIP 编译为 LLVM，系统 fence、release store、acquire load 存在，
但 **CTA barrier 缺失**（`m2-native-llvm-review.log`）。根因是既有
`CodeGenTileLangHIP::PrintStorageSync` 对 shared/shared.dyn 的输出被注释。
这不是测试工具问题：实际多 writer 发布协议会缺少必要 rendezvous。已批准在显式 IPC
绑定函数的原生 lowering 中保留该同步为不透明 IPC barrier，并在 HIP 发出真正 CTA
同步；不得删除 LLVM 检查，不改普通 GEMM 的既有同步策略。修复后须重新验证设备 litmus。

另外独立审查发现 bridge finalize 不能按本地 descriptor cache 提前返回，否则只在部分
PE 上传描述符时会造成集体调用不对称。修正为由 core 的集体 live-allocation 检查裁决，
重新安装后四 PE 扩展测试通过：仅 rank 0 创建 descriptor，所有 rank 一起收到 Busy，
继续 free 后可再分配、创建 descriptor 并正常 finalize。日志
`m2-installed-finalize-review.log`，测试没有删除可选 MPI 插件警告。

### 第一轮正式 review：退回

主路径已通过独立 copy/get 四 PE 执行、32 epoch × 8 batch 非零 canary 压力测试、
24 种标量真实 HIP 编译、12 项初始独立检查、61 项旧回归和两组 GEMM 数值检查。
日志分别为 `m2-independent-copy-get-review.log`、`m2-independent-stress-32x8.log`、
`m2-native-scalar-matrix-review.log`、`m2-final-regression.log`。

阻塞项 R1-1：只扫描原始 thread Var 的 if 检查不足以证明参与者一致。
`Let(gate = threadIdx.x + 1); if gate == 1: CTA barrier` 被 prepare 接受，实际会使
只有线程 0 执行 barrier；独立失败证据为 `m2-formal-r1-let-alias.log`。
要求追踪 Let 依赖、拒绝无法证明 uniform 的 load 条件，并检查 cooperative/barrier
所在循环的迭代条件。保留 rank/blockIdx 与正常 serial uniform 控制流的合法路径。
另一个 local-condition case 当前得到 undefined 内部错误，也应在进入该路径前正确拒绝。
测试构建曾漏传 IfThenElse 的 else=None，已仅修复测试自身，不作为生产缺陷。

文档项 R1-2：安装示例须指定用户可写 prefix（实际默认是 /usr/local），MPI 环境应在
首次 configure/build 之前设置；明确 public scope 关键字 gpu/sys、地址绑定粒度和 get
的真实执行结果。整包 wheel、持久 cache 和 M0 候选的限制不能省略。

两轮正式 review 仍未通过则报告并暂停；通过后也报告并等待用户确认，
不自动进入 M3、不自动 git commit。

### 第二轮正式 review：通过

R1-1 已改为 Let-aware `_is_block_uniform` 检查：If 条件、包含 cooperative copy/CTA
barrier 的 For(min/extent/step)、While condition 必须可证明 block-uniform；BufferLoad
和未知条件 fail-closed，rank/blockIdx/正常 serial uniform 路径保留。
独立测试包含 Let 别名隐藏的线程条件、本地 load 条件、线程变化循环边界和合法 rank
条件/serial 循环。R1-2 的首次配置 MPI 环境、用户可写 prefix、gpu/sys 和 copy/get
执行说明均已修正。

主代理在 Slurm 9510 / dcu0 / 四张不同 BDF 的 gfx936 上执行第二轮检查：

| 检查 | 结果 | `build-old/logs/` 下日志 |
|---|---|---|
| 独立绑定、lowering、uniformity、source-only 检查 | 16 passed，exit=0 | `m2-formal-r2-independent.log` |
| 实际生成 HIP → LLVM 同步指令审计 | 系统 fence、release/acquire、真实 CTA barrier 调用存在，exit=0 | `m2-formal-r2-llvm.log` |
| 既有回归、路径隔离、两组 GEMM | 61 passed；两组 GEMM PASS，最大绝对误差 0.00390625；exit=0 | `m2-formal-r2-regression.log` |
| 全 12 方向 Block publication 压力复测 | 4 PE × 32 epoch × 8 batch，全部 PASS，exit=0 | `m2-formal-r2-stress-32x8.log` |

此前独立四 PE copy/get、已安装组件生命周期、Python runtime、24 种标量 HIP 编译及
惰性导入检查仍有效；相关代码在本轮修改中未改变。最终生产路径位于
`tilelang/distributed/ipc_binding.py`、`src/transform/lower_dcu_ipc.cc`、HIP codegen、
`src/tl_templates/dcu_hip/ipc.hpp` 和 `src/runtime/dcu_ipc`，通过正常 lower/JIT 接入。
实现与重现命令见 [实现报告](distributed_fusion_m2_implementation_report.md)。

通过范围仅限本报告覆盖的单机 gfx936/FineGrained 组合；未证明 DeviceDefault payload、
跨节点或任意平台内存模型。完整 wheel 的 TVM Python/native 匹配仍待核查，不能将
CMake 组件安装通过表述为整包 pip 安装已经通过。ctypes 持久缓存未实现；M0 候选及
其失败测试保留。没有完成或开始 M3、融合 GEMM 和完整 AR/RS/AG 生成。
