# M1 逐文件实现详解：把分布式通信意图变成可检查的 TIR

日期：2026-09-21。对应主仓库 HEAD：`b2bc99b0a3ab03cb5ffce731db87048931e011a6`，
以及当前未提交的 M1 实现。代码链接指向当前快照的起始行；后续编辑可能移动行号。
本报告按“先理解流程，再看核心源码，最后看兼容与测试”组织，逐一覆盖新增/修改文件。

2026-09-23 更新：按用户授权，撤回 `utils.py`/`copy_op.py` 的文本回读专用扩展，
删除依赖这些扩展的测试。当前保留正常 DSL 构造与 IR 校验，不再承诺打印文本可回读；
删除理由及验收范围变化见第 7.3、8.1 节。第 11.4 节的 60 项结果仍是历史记录。

## 1. M1 解决的问题，以及它还没有解决的问题

以前我们在 HIP 中直接写远端地址访问、ready 发布和等待。要让 TileLang 正式生成这些操作，
首先必须让编译器能够表达并识别它们，而不是仅把一段 HIP 字符串粘到 GEMM 后面。

M1 做了三件事：

1. 提供 `T.remote_view`、`T.signal_set`、`T.wait_eq` 等 Python DSL 接口。
2. 把这些调用记录为有类型、有参数、有副作用属性的 TIR 节点。
3. 提供远端 Buffer 规范化和合法性检查，拒绝当前不能安全处理的形式。

**M1 的输入是 Python DSL/PrimFunc，输出是可检查、可序列化的 IR；不是已经能通信的 HIP kernel。**

没有在本模块实现的事情包括：`rocshmem_ptr` 设备调用、通信 helper、设备原子指令、
persistent Block 调度、显存映射、最终 JIT/runtime 接线和四卡数值/性能验证。

### 1.1 先补齐几个编译器概念

| 概念 | 不需要编译器背景的理解 |
| --- | --- |
| DSL | 用 Python 语法描述将来在设备上执行的程序，而不是此刻就执行设备通信 |
| TIR | 编译器内部的程序树，包含循环、读写、调用、Buffer 等对象 |
| PrimFunc | 一份低层函数的 IR 对象；它仍不是已加载的 GPU 函数 |
| Buffer | 类型、形状、步长和指针关系的描述；创建 Buffer 对象不等于分配 GPU 内存 |
| Var | IR 中的符号变量，不一定已有运行时值 |
| Call / intrinsic | 编译器认识的操作调用；需要后续 lowering 才能成为设备代码 |
| Evaluate | 把一个表达式调用作为语句执行，其返回值不用 |
| LetStmt | 在一个词法范围内给变量绑定值；例如“这一段里 p 代表该远端地址” |
| DeclBuffer | 声明某个指针应按什么 dtype/shape/stride 看待，不负责 malloc |
| AttrStmt | 给一段语句加编译器标记，如“这是显式 specialization 作用域” |
| pass / lowering | 遍历并改写程序树，逐步把高层操作变成目标机器支持的操作 |
| FFI | Python 与 C++ 库之间的函数、对象桥接层；不是 GPU 的卡间通信 |

### 1.2 M0、M1 与已有编译链的关系

```text
M0：FusionSpec / SchedulePlan / ABI
    描述配置、布局、调度公式；目前尚未自动生成下面的 DSL
                       │ 后续生成器将负责连接
                       ▼
Python 的 @T.prim_func 程序
  ├── 普通 T.copy / T.gemm 等已有操作
  └── M1 的 remote_view / signal / wait / fence 等操作
                       ▼
未规范化 PrimFunc：含 tl.remote_view_token
                       ▼  显式调用 normalize_remote_views
typed pointer + LetStmt + DeclBuffer + 分布式 Call
                       ▼  显式调用 verify_distributed_ir
通过当前 M1 的类型、peer 和规范化检查
                       │
             此处是 M1 目前的交付边界
                       │ 后续接入
                       ▼
布局推导 / tile-op lowering / 目标优化 / HIP codegen
                       ▼
hipcc + ROC_SHMEM device/runtime + 实际 DCU launch
```

已有普通编译入口可以从 [engine.lower.lower](/public/home/baoboyuan/TileLang_DCU/tilelang/engine/lower.py:234) 阅读：
它调用 PreLowerSemanticCheck、LowerAndLegalize、OptimizeForTarget，再走代码生成。
**本次对 lower.py 的修改只是 FFI 接口来源调整，没有把 M1 normalize/verify 插入这条流水线。**
因此，“导入后存在 T.signal_set”不能理解成“现有 JIT 已经可以执行它”。

## 2. 新增/修改文件清单：先分清功能和兼容层

### 2.1 分布式功能及前端接线

| 文件/入口 | 类型 | 新增或修改的职责 |
| --- | --- | --- |
| [tilelang/language/distributed/__init__.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/__init__.py:3) | 新增 | 汇总分布式接口与 normalize/verify |
| [tilelang/language/distributed/common.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:22) | 新增 | rank/world、作用域、typed remote view、打印 hook |
| [tilelang/language/distributed/comm.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:13) | 新增 | typed block copy、带 scope/sem 的 scalar load/store |
| [tilelang/language/distributed/sync.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:14) | 新增 | uint32 signal、wait、system fence |
| [tilelang/language/distributed/ir.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:21) | 新增 | remote alias 规范化、peer 一致性与原语验证 |
| [tilelang/language/distributed/multimem.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/multimem.py:9) | 新增 | 暴露同名接口，但明确拒绝不支持的 CUDA multimem |
| [src/op/distributed.cc](/public/home/baoboyuan/TileLang_DCU/src/op/distributed.cc:18) | 新增 | C++ 原生 op 身份、参数个数、副作用和打印名注册 |
| [tilelang/language/__init__.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/__init__.py:103) | 修改 | 导出 region；从第 122 行起将分布式接口接到 T.* |
| [tilelang/language/utils.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/utils.py:7) | M1 修改已撤回 | region 保持原有字符串访问模式，不扩展整数输入 |
| [tilelang/language/copy_op.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/copy_op.py:53) | M1 修改已撤回 | 保持原有普通 copy 路径，不接受回读专用的 raw-region 输入 |

### 2.2 构建、兼容、测试与文档

| 文件/入口 | 类型 | 作用 |
| --- | --- | --- |
| [tilelang/_ffi_compat.py](/public/home/baoboyuan/TileLang_DCU/tilelang/_ffi_compat.py:16) | 新增 | 为 Python/C++ 交互选择一致的 FFI 注册表 |
| 第 9 节逐文件列出的既有 Python 文件 | 修改 | 将旧/新 FFI 混用统一到上述入口；不是重写 GEMM/通信算法 |
| [CMakeLists.txt](/public/home/baoboyuan/TileLang_DCU/CMakeLists.txt:18) | 修改 | 可选跳过 submodule 自动更新，保护已有检出 |
| [tilelang/env.py](/public/home/baoboyuan/TileLang_DCU/tilelang/env.py:33) | 修改 | 允许独立 build-m1 目录加载 |
| [.gitignore](/public/home/baoboyuan/TileLang_DCU/.gitignore:22) | 修改，共享 | 忽略 build-m1 产物；放行 language/distributed 源码 |
| [testing/python/distributed/test_m1_ir.py](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | 新增 | M1 本身的 10 项真实 IR 测试；两项完整文本回读测试已删除 |
| [testing/python/distributed/ir/test_m1_review.py](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:10) | 新增 | 主代理维护的独立探针，参数化展开为 25 项 |
| [testing/python/distributed/ir/test_ffi_compat.py](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_ffi_compat.py:10) | 新增 | 新进程中验证真实 FFI 注册与容器 |
| [docs/compiler_internals/distributed_fusion_m1.md](/public/home/baoboyuan/TileLang_DCU/docs/compiler_internals/distributed_fusion_m1.md:1) | 新增说明 | API、IR 形式、支持范围 |
| [docs/compiler_internals/distributed_fusion_m1_build.md](/public/home/baoboyuan/TileLang_DCU/docs/compiler_internals/distributed_fusion_m1_build.md:1) | 新增说明 | 匹配 TVM/FFI 的独立构建、测试环境 |
| [docs/compiler_internals/distributed_fusion_m1_review.md](/public/home/baoboyuan/TileLang_DCU/docs/compiler_internals/distributed_fusion_m1_review.md:1) | 新增记录 | 正式 review、测试证据、已知限制 |
| 本报告 `distributed_fusion_m1_implementation_report.md` | 本次新增 | 逐文件源码讲解 |

M0 的 source/test 在 M1 验收期间保持不变；只是作为回归测试重新运行。
TVM/CK 子模块在 git status 中显示的检出差异是既有状态，不是本模块新改写其算法。

## 3. common.py：创建通信上下文符号、词法作用域和远端视图

### 3.1 辅助检查及跨 TVM 表示适配

代码块：

- [_is_rank_dtype](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:22)：根据 dtype 的稳定字符串表示识别标量整数，排除浮点、向量和 bool/uint1 类型。
- [_buffer_type_for_decl](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:32)：适配 Buffer 类型字段在不同 TVM Python 接口中的表示；此实现针对普通 default Buffer，不是通用特殊 Buffer 类型转换器。
- [_static_integer](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:43)：提取 Python int/TIR IntImm，供本地 peer=-1 等判断使用。
- [_validate_peer_type](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:53)：构造时的 peer 类型检查；完整的 block-uniform 检查在 ir.py 中做。
- [_zero_indices / _full_region](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:102)：构造 base Buffer 的完整 provenance region。
- [_offset_dtype](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:111)：偏移 dtype 辅助函数；当前主路径直接使用 base_offset.dtype，未依赖此辅助函数。

这里的 peer 检查不是 runtime rank 范围检查，也不查询当前有几张卡。
动态 peer 必须由后续 ABI/runtime 提供正确值。

### 3.2 get_rank/get_num_ranks：不能把编译进程 rank 写死

入口：[get_rank](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:77) 和 [get_num_ranks](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:82)。

核心逻辑很短：

```python
return tir.call_intrin("int32", tir.op.Op.get("tl.get_rank"))
```

调用它得到的是“运行时应该从通信上下文读取 rank”的 int32 IR 表达式，
不是当前 Python 进程查询出的整数。这样将来同一份代码才能在多个 PE 复用。

M1 只建立这个符号操作；如何从 context 取实际 rank，尚需 runtime/device lowering 实现。

### 3.3 sm_specialize_scope：标记一段代码，不分配 CU

入口：[sm_specialize_scope](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:87)。

`auto_ws=False` 返回真正的 TIR attribute frame：

```python
return _tir_attr(0, "tilelang.sm_specialize", 0)
```

放在 `with` 中时，它包住内部语句，形成词法 AttrStmt。
`auto_ws=True` 明确抛 NotImplementedError，不能把 CUDA 自动 warp specialization 冒充为 HIP 支持。

这个 scope 不会替你选择 compute/comm Blocks，不会读取 CU 数，也没有证明分支中的 barrier 安全；
这些仍是后续调度和验证工作。

### 3.4 remote_view：为什么返回 Buffer，而不是一个裸指针

入口：[remote_view](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:116)。

既有 T.copy、布局推导和 epilogue 操作需要的是 Buffer。
如果只返回一段 `"rocshmem_ptr(...)"` 字符串，编译器无法直接知道：

- 指针指向 float32 还是 float16；
- 是几维、每维多大、stride 是什么；
- 当前 view 与哪个本地 allocation 对应；
- 访问哪个 peer；
- 原有 elem_offset 是否已经加过。

因此 remote_view 保留真实 Buffer，按以下步骤构造：

1. 输入必须是 global Buffer；不会创建 `remote_global` 这类现有 storage pass 不认识的 scope。
2. 检查 peer 的基本类型。
3. 如果 remote=False 或常量 peer=-1：返回同底层 data、同 offset 的本地 alias，不产生远端 token。
4. 否则把 base region、peer、原 offset、shape、strides 存进 `tl.remote_view_token`。
5. 暂时将 `elem_offset` 写成 `base_offset + token`，其余 dtype/shape/stride/alignment 等继续保留。

这段 `base_offset + token` 是**未规范化的前端标记**，不是设备上真的要加的地址。
它必须在后续 normalize 时被消费掉。

[remote_view_token](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:61) 和
[remote_address](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/common.py:72) 是公开可访问的打印/回读 hook；
它们并没有执行远端地址解析。

### 3.5 M0 的 byte_offset 与 M1 的 elem_offset 不是同一个单位

M0 描述符的 byte_offset 以字节计；TIR Buffer.elem_offset 以元素计。
例如 FP32 view 的 elem_offset=3 表示 12 字节。

当前没有 M0 descriptor → M1 Buffer 的自动转换器。
将来连接两者时必须显式处理单位，不能直接把 M0 的 12 写成 TIR 的 elem_offset=12。

## 4. ir.py：把“带标记的视图”变成合法、可检查的指针绑定

这个文件是 M1 的核心转换逻辑。它不是 HIP codegen，也没有新建并注册一个自动执行的 C++ pipeline pass。

### 4.1 识别标记和 provenance

| 代码入口 | 做什么 |
| --- | --- |
| [DistributedIRError](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:21) | 给 M1 IR 诊断统一错误类型 |
| [_op_name / _same](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:25) | 识别 op；用 TVM 对象身份比较，而不是名字相等 |
| [_remote_token](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:35) | 遍历 Buffer.elem_offset，找远端 token；多个 token 报错 |
| [_remote_buffers](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:51) | 收集语句里 BufferLoad/BufferStore 涉及的 remote Buffer，按对象身份去重 |
| [_region_buffer](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:68) | 从 tl.region 中取得 Buffer |
| [原语检查辅助函数](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:74) | peer 整数类型、uint32 范围、scalar dtype 和 scope/sem 白名单 |

provenance 的意思是“这个远端视图是以哪个本地 Buffer 为对应基准”。
它在 IR 中以 region 保存，不靠一个 Python 全局字典偷偷关联，因此能跟随序列化和打印。

### 4.2 _make_alias：新指针、原布局、offset 只加一次

入口：[_make_alias](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:105)；
配套记录：[_RemoteAlias](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:130)。

`_make_alias` 创建一个新的 typed Var：

```python
pointer_type = ir.PointerType(ir.PrimType(buffer.dtype), "global")
data = tir.Var(..., pointer_type)
```

然后 decl_buffer 使用这个新 data，但保留原 shape/strides/alignment/offset_factor/axis_separators，
把 elem_offset 恢复为 token 保存的原 offset。

假设 base 为 FP32、shape=(2,4)、strides=(8,1)、elem_offset=3。
规范化后的远端访问 [1,2] 应表示：

```text
peer allocation base + (3 + 1×8 + 2) × 4 bytes
                    = peer allocation base + 52 bytes
```

`remote_address` 的契约是 allocation base，不是已经偏移 12 字节的 view 指针，
所以原 offset 只在 Buffer 索引中应用一次。

### 4.3 _AliasNormalizer：真正改写语句的地方

类入口：[_AliasNormalizer](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:146)。

| 方法 | 逻辑 |
| --- | --- |
| [_entry](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:156) | 为某个语句中的 remote view 创建 fresh typed Var 与 alias |
| [_entries_in](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:169) | 收集当前访问语句所需的别名，同一语句内去重 |
| [_active_entry](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:181) | 从当前作用域的临时栈中查别名 |
| [_wrap](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:188) | 用 LetStmt + DeclBuffer 包住改写后的访问语句 |
| [_rewrite_expr](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:201) | 递归重建受支持的 Call/BufferLoad，替换远端 Buffer |
| [visit_buffer_load_](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:226) | 通用访问路径的保护：没有合法绑定作用域时拒绝 remote load |
| [visit_buffer_store_](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:241) | 改写 store 的目标及表达式，再包住绑定 |
| [visit_evaluate_](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:256) | 改写调用语句（如 T.copy），再包住绑定 |
| [mutator 装饰](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:268) | 将 Python visitor 接入 TVM 的 IR 遍历机制 |

核心结构为以下伪 IR，省略了具体 printer 语法：

```text
原始：
  copy(local_buffer, remote_view_buffer_with_token)

规范化后：
  Let remote_ptr = tl.remote_address(region(base_buffer), peer)
    DeclBuffer remote_alias(
        data=remote_ptr,
        dtype/shape/strides/alignment=原值,
        elem_offset=原值)
      copy(local_buffer, remote_alias)
```

LetStmt 是“绑定地址表达式”，DeclBuffer 是“声明怎样解释地址”。
两者都不等于在 GPU 上申请 staging，也没有立即调用 rocshmem_ptr。

临时 `_active_entries` 只是一次遍历的局部重写状态；最终信息都在 IR 中，
不依赖它在编译结束后继续存活。

### 4.4 为什么同一 view 用两次，也要避免复用同一个 Var 定义

编译器的 SSA 约定要求一个 Var 对象不能在多个位置重复定义。
第一版曾在两个兄弟 LetStmt 中定义同一个 Var，即使名字看似一样，也会破坏这个约定。

当前 `_entry` 每次合法语句绑定都创建 fresh Var：

```text
语句 1：Let p0 = remote_address(...) → copy(..., alias0)
语句 2：Let p1 = remote_address(...) → copy(..., alias1)
```

这保证结构正确，但暂时可能解析同一个 peer 地址多次。
**M1 没有实现性能上的 rocshmem_ptr 循环外提或公共地址绑定合并。**
不能把“SSA 修复”解读为“已经做到最优地址复用”。

### 4.5 为什么绑定不能一律放到函数开头

如果 peer 是 serial loop 的 i 或 Let 定义的 peer，那么它只在相应作用域里有意义。
把远端指针绑定提升到 PrimFunc 根部，就会引用尚未定义的变量。

因此 `_wrap` 把绑定放在包含该次 Evaluate/BufferStore 的语句周围，留在原 loop/Let 内部。
当前尚不支持的情况会明确拒绝：

- remote load 位于 Let 的 value；
- remote load 用来决定 if/assert/loop bound；
- 任意包含 remote load 的复合算术表达式不属于当前通用支持范围。

前置拒绝在 [_has_remote_load](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:434) 和
[_reject_unbindable_remote_loads](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:447)。
`_rewrite_expr` 的有限重建路径也针对匹配 TVM 的 Python mutator/Call 重建兼容问题；
它不是一个可以处理任意 TIR 表达式的通用优化器。

### 4.6 peer 必须 block-uniform：同一个 Block 不能各线程选不同 peer

入口：[_collect_bindings](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:271)、
[_is_block_uniform](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:319)、
[_verify_uniform_peers](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:376)。

检查分两步：

1. 收集函数整数参数、Let 绑定、serial/unrolled loop 的 min/extent/step、
   blockIdx/threadIdx 的实际 IterVar.thread_tag。
2. 从 peer 表达式追溯这些变量，保守判断它是否在 Block 内一致。

| peer 来源 | 当前处理 |
| --- | --- |
| 常量、整数函数参数 | 接受 |
| get_rank/get_num_ranks | 接受 |
| blockIdx 或其算术表达式 | 接受 |
| bounds 和 step 都一致的 serial/unrolled loop 变量 | 接受 |
| Let alias | 继续追溯其定义值 |
| threadIdx 或依赖它的 Let/loop | 拒绝 |
| BufferLoad、未知 Call、未绑定变量 | 拒绝 |

不是看变量名字是不是 `tx`，而是看真实线程绑定元数据。
即使把 threadIdx 改名为 `owner`，也不能绕过检查。

这里还修过一个实际崩溃：匹配 TVM 的 For.step 属性存在但可以为 None。
`getattr(node,"step",默认值)` 在属性存在时不会替换 None。
现在 [显式 step=None 处理](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:298) 将隐式步长解释为 1，再遍历；
也检查 thread-varying step，避免仅检查 min/extent。

注意它只检查 peer 值是否 uniform，**没有证明所有控制流和 barrier 的参与线程集合安全**，
也没有验证动态 peer 的 0≤peer<P 范围。后续模块仍须补齐相关运行和资源约束。

### 4.7 对外两个入口：normalize 不等于 verify

[normalize_remote_views](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:479)：

```text
检查 PrimFunc 类型
  → 检查 peer uniformity
  → 拒绝没有合法绑定位置的 remote load
  → 遍历并改写函数体
  → 检查没有遗留 remote token
  → 返回新 PrimFunc
```

[verify_distributed_ir](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:496)：

```text
检查 PrimFunc 类型
  → 检查 peer uniformity
  → 默认要求 remote view 已规范化
  → 按 op 检查 region、dtype、参数个数、uint32 值、scope/sem、typed copy
  → 失败则报告；成功返回 None
```

[assert_remote_views_normalized](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:417) 是单独的前端标记检查入口。
它目前没有被自动插入所有 codegen 路径，调用方仍需显式使用。
`require_normalized=False` 用于检查前端 IR，不代表未规范化形式可以执行。

这里的 verifier 也不是所有 TIR 正确性的完整证明器；例如 SSA 验证在独立测试中还调用了 TVM 的 verify_ssa。
不要把“verify_distributed_ir 返回”理解成“硬件运行必定正确”。

## 5. comm.py：typed block copy 和带内存语义的 scalar 访问

### 5.1 put_block/get_block：size 是元素，不是字节

入口：[put_block](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:138) 和
[get_block](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:168)。

概念方向：

- put：本地 src → dst_pe 的 dst；
- get：src_pe 的 src → 本地 dst；
- 默认 peer=-1 为本地约定；
- 不附带 ready 发布，也不表示一个完整 collective。

目前它们只构造四参数 IR：
`(src_region,dst_region,size,peer)`。
还没有决定最终映射成哪一种 ROC_SHMEM 设备 API，所以不能说它们现在就在执行 getmem_wg。

### 5.2 为什么要这么多 copy 校验辅助函数

| 代码入口 | 主要逻辑 |
| --- | --- |
| [_is_region / _load](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:19) | 识别 raw region；整 Buffer 转成零起点 BufferLoad |
| [_is_integer_scalar](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:31) | size/index 必须是标量整数 |
| [_validate_indices](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:41) | 检查已知的负数及各维静态越界 |
| [_as_static_int](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:52) | 提取可静态确定的长度和起点 |
| [_require_contiguous_run](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:62) | 限定末维 stride=1、静态长度、可证明的末维区间 |
| [_contiguous_region](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:96) | 非末维 extent=1、末维 extent=size，显式编码访问区域 |
| [_typed_size](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:106) | 拒绝负长度/浮点/bool 等非法元素数 |
| [_region_load](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:115) | 从 raw region 提取真正的 endpoint |
| [validate_block_copy_regions](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:121) | raw/printer 路径也重新检查 dtype、连续区间和 region extent |

以 shape=(2,4)、strides=(8,1) 的 padded Buffer 为例：

- 从 [0,0] copy 4 个元素可表示一行连续区间；
- 从 [0,3] copy 2 个元素跨越逻辑行尾，必须拒绝；
- 不允许把“有 padding 的二维张量”误当作平坦连续数组；
- 动态 size 或无法确定的末维起点/extent，当前没有范围证明机制，拒绝而不猜测。

因此 put/get_block 不是任意 strided tensor copy。
要描述完整多维 tile copy，已有 T.copy 是另一层接口；不能随意将两者互换。
对其他动态维的检查也不能解释为完整的运行时边界保护。

### 5.3 st/ld：表达作用域和语义，不在 Python 中加载远端数据

入口：[st](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:233)、
[ld](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:267)。

两者均生成五参数 IR：

```text
st(dst_region, value, sem_id, scope_id, peer)
ld(src_region, dst_region, sem_id, scope_id, peer)
```

特别注意：这里 `T.ld(src,value)` 的 value 是一个可写 Buffer/BufferLoad 目标，
不是“Python 返回一个 float”；IR 调用的结果类型是 handle。

白名单在 [scope/sem/dtype 表](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:13)：

- scope：gpu=1、sys=2；
- st：weak=0、volatile=1、release=3、relaxed=4；
- ld：weak=0、volatile=1、acquire=2、relaxed=4；
- dtype：int32、uint32、float32；
- cache/na/nc 选项不支持，显式报错。

[_memory_mode](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:194) 负责前端字符串模式；
[_validate_scalar_buffer](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:206) 和
[_value_matches_dtype](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:212) 检查类型；
[_raw_memory_mode](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/comm.py:227) 检查打印回读的整数模式。

st/ld 的 raw region 分支使用 printer 的位置参数顺序，不是普通 DSL 的参数语义。
例如回读分支里名为 `na` 的 Python 形参临时承接 peer，这只是为了重建已打印的五参数 op；
正常用户代码仍应使用 scope/sem/dst_pe/src_pe 的公开签名。

## 6. sync.py：把“发布—等待”表示成有副作用的操作

### 6.1 校验 helpers 与默认值

| 代码入口 | 功能 |
| --- | --- |
| [WaitScope/WaitSemantics](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:14) | 首期只定义 SYS 和 ACQUIRE |
| [_enum](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:22) | 枚举、字符串以及回读整数统一成 op 参数 |
| [_uint32_load](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:35) | flag 必须是 uint32 BufferLoad |
| [_scalar_region](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:43) | 标志访问范围每维为 1 |
| [_uint32_value](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:47) | 静态值必须落在 0…2^32−1，动态表达式必须是 uint32 |
| [_region_flag](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:66) | raw region 也不能绕过 flag dtype 检查 |

wait 的 SYS/ACQUIRE 编码都为 0，与 comm.py 的 scalar scope 编码不是同一套枚举。
不能跨 op 随意复用裸数字。

### 6.2 fence_sys、signal_set、wait_eq/ge 的构造过程

- [fence_sys](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:74)：构造零参数 `tl.fence_sys`。
- [signal_set](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:83)：检查 flag/value/peer，构造 `(write_region,value,peer)`。
- [_wait](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:96)：将公开接口转交给统一构造函数。
- [wait](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:102)：构造六参数 `(relation,read_region,expected,peer,scope,semantics)`。
- [wait_eq](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:123) 使用 relation=0；
  [wait_ge](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/sync.py:130) 使用 relation=2。

一个很具体的修复位于 [wait 的 peer 参数检查](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/ir.py:402)：
wait 的 peer 在 args[3]，最后一项是 semantics；不能用通用的 args[-1] 检查 peer。

### 6.3 M1 表达内存序意图，不证明硬件内存序

后续正确协议应区分以下动作：

```text
producer：
各 writer 写 payload
  → 各 writer system fence
  → Block barrier
  → thread0 发布 ready

consumer：
thread0 acquire wait
  → Block barrier
  → 各线程读取 payload
```

这是算法/后端应实现并验证的协议，不是调用一个 signal_set 就会自动补齐的动作。
fence 不是 CTA barrier；wait 不是所有线程自动同步；signal 也不会自动发布其他 writer 的数据。
当前没有生成实际自旋循环或 atomic 指令，更没有在 M1 完成跨 DCU litmus 测试。

## 7. C++ 注册、Python 导出和打印回读怎样配合

### 7.1 src/op/distributed.cc：告诉原生编译器“这些操作存在”

注册宏：[TL_REGISTER_DISTRIBUTED_OP](/public/home/baoboyuan/TileLang_DCU/src/op/distributed.cc:18)。

每条注册提供：

1. 全局 op 名；
2. 参数个数；
3. `TCallEffectKind`；
4. `TScriptPrinterName`。

| 原生 op | 参数数 | 主要意义 |
| --- | --- | --- |
| tl.get_rank / tl.get_num_ranks | 0 | 动态上下文读取 |
| tl.remote_view_token | 可变 | 前端元信息标记 |
| tl.remote_address | 2 | provenance + peer 的地址表达式 |
| tl.fence_sys | 0 | system fence 意图 |
| tl.signal_set | 3 | flag region、值、peer |
| tl.tileop.wait | 6 | 关系、flag、expected、peer、scope、semantics |
| tl.tileop.put_block/get_block | 4 | 两个 region、元素数、peer |
| tl.tileop.st/ld | 5 | scoped scalar 访问 |

注册代码分别从 [rank/remote op](/public/home/baoboyuan/TileLang_DCU/src/op/distributed.cc:25) 和
[sync/comm op](/public/home/baoboyuan/TileLang_DCU/src/op/distributed.cc:38) 开始。

除 token 使用 embed-info，其他都为 opaque effect。
意思是让理解该属性的编译器逻辑知道“这不是可以当作普通纯算式消除/合并的计算”。
但仅注册 opaque 不足以证明所有已有 pipeline pass 都不会跨同步边界移动访存；
完整流水线验证仍未完成。

文件中没有 HIP 模板展开、rocshmem 调用或完整 collective 算法。

### 7.2 两级 __init__.py 导出

[distributed 包导出](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/__init__.py:3) 汇总函数，
[language 总入口](/public/home/baoboyuan/TileLang_DCU/tilelang/language/__init__.py:122) 将其导出为常见的 `T.*`。

这一步只提供名字，既不会自动调用 normalize，也不会自动插入 runtime 初始化。
`tilelang.distributed`（M0）和 `tilelang.language.distributed`（M1）是不同层，不应混用。

### 7.3 utils.py 和 copy_op.py 的扩展已撤回：为什么删除相关测试是可接受的

正常 DSL 可能写 `T.copy(A,B)`，打印后的底层 IR 则可能写成：

```text
T.copy(T.region(A[...],1,...), T.region(B[...],2,...), ...)
```

过去为重新解析这种打印文本，曾让 `region` 接受整数访问模式，并让 `copy`
直接接收已编码 region。2026-09-23 已按用户要求撤回这两个扩展，两文件恢复为
Git HEAD 内容；`T.region` 的导出仍保留，调用时使用原有 `"r"`/`"w"`/`"rw"`。

正常编译路径直接从 DSL 构造 TIR，再交给 normalize/verify 和后续 lowering，
不要求经过“打印 → 重新解析”的中转。因此删除只验证这一辅助路径的测试是可接受的，
不应仅为了测试而扩大通用 API 的输入契约。具体调整如下：

- 删除 `test_all_m1_primitives_print_parse_roundtrip`。
- 删除 `test_ordinary_local_dsl_parse_print_regression`。
- 删除原 `test_sync_defaults_and_print_parse_roundtrip` 最后的 remote-copy 回读断言；
  默认值和非法输入校验保留，重命名为 `test_sync_defaults_and_invalid_input_validation`。
- 删除仅供这些测试使用的 `_RoundtripNamespace`、`_from_source` 和 parser import。
- 两项独立 raw wait/signal 安全检查改用原有字符串模式构造 region，保留原断言。

这属于用户授权的验收范围调整：原计划“解析/打印 round-trip”中的文本回读要求
不再作为当前 M1 的验收条件，不能把删除测试描述成该能力仍然通过。原生注册、类型/
边界、副作用、SSA、peer 一致性、元数据、公开前端及 JSON 序列化检查继续保留，
也没有放宽后续通信正确性和硬件内存序的验收要求。

### 7.4 multimem.py：为什么有接口却一调用就报错

[MultimemReduceOp](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/multimem.py:9) 定义常见归约操作枚举；
[_unsupported](/public/home/baoboyuan/TileLang_DCU/tilelang/language/distributed/multimem.py:18) 是统一 capability error；
第 25 行起的六个同名 API 都指向它。

这样从 CUDA/tilescale 迁移的代码会得到明确“不支持”，而不是静默变成普通 load/store，
导致用户以为硬件执行了跨多个 PE 的 multimem collective。
这不是 multimem 的软件性能实现，也不是对所有未来 DCU 平台能力的永久判断。

## 8. 一个完整的 M1 示例：从 Python 到规范化 IR

实际公开前端测试位于 [test_public_v2_prim_func_frontend_builds_distributed_ir](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py)。

下面是同一思路的教学简化，**只构造 IR，不是一份可 launch 的分布式 kernel**：

```python
from tilelang import language as T

@T.prim_func
def describe_copy(
    src: T.Tensor((4,), T.float32),
    dst: T.Tensor((4,), T.float32),
):
    remote_dst = T.remote_view(dst, 1)
    with T.sm_specialize_scope(auto_ws=False):
        T.copy(src, remote_dst)

normalized = T.normalize_remote_views(describe_copy)
T.verify_distributed_ir(normalized)
print(normalized.script())
```

逐步解释：

1. 函数注解创建 src/dst 的 TIR Buffer 参数，不分配实际数组。
2. remote_view 保留 dst 布局，把 peer=1 和 provenance 放进 token。
3. T.copy 记录源读 region 和目标写 region；暂时仍是高层 tile copy。
4. sm_specialize_scope 给内部 copy 加属性，不决定哪个 Block 执行。
5. normalize 找到目标 Buffer 的 token，创建独立的 global float32 指针 Var。
6. 在 copy 周围绑定 `remote_address(region(dst),1)` 并声明 alias。
7. verify 确认 peer 为 block-uniform、没有未处理 token。
8. script 打印当前 IR；并没有调用 hipcc 或 GPU launch。

真实测试还构造 fence/signal/wait/put/get/st/ld，并做 normalize/verify；
它证明这些公开接口能通过实际 v2 builder 构造 IR，不等于该测试里的语句顺序就是完整的多线程通信协议。

### 8.1 打印与序列化的当前边界

`script()` 可用于查看 IR，但当前不保证打印文本经 `from_source` 回读；过去仅用于
此目的的测试 namespace 已删除，不能继续按旧说明调用。正常构造使用上面的公开
`@T.prim_func` 接口。保留的独立 JSON roundtrip 测试验证 metadata 不依赖 Python
全局表，但 JSON 往返不等价于文本打印/解析保证。

## 9. FFI 兼容层：为何看起来修改了很多不相关文件

### 9.1 _ffi_compat.py 的实际选择流程

本次环境中，用户已有 TVM 检出与主仓库要求的版本不同，Python 调用也混用了
`tvm.ffi`、`tvm._ffi` 和外置 `tvm_ffi`。
旧内置 FFI 与独立外置 FFI 可能拥有不同原生类型注册表，混载可能导致类型解释错误甚至进程崩溃。

新增 [_loaded_tvm_ffi](/public/home/baoboyuan/TileLang_DCU/tilelang/_ffi_compat.py:16) 的逻辑：

```text
查看当前已加载 tvm 的 ffi
  ├── 有完整旧接口 register_func/_init_api/... → 使用它
  └── 没有可用旧接口 → 导入匹配的外置 tvm_ffi
      ↓
统一导出 register_object / get_global_func / Array / Map
      ↓
包装 register_global_func 与 init_ffi_api
```

[后端选择](/public/home/baoboyuan/TileLang_DCU/tilelang/_ffi_compat.py:27) 不修改 sys.modules 来伪造包；
[register_global_func](/public/home/baoboyuan/TileLang_DCU/tilelang/_ffi_compat.py:46) 同时支持 decorator 和 f= 两种调用；
[init_ffi_api](/public/home/baoboyuan/TileLang_DCU/tilelang/_ffi_compat.py:58) 初始化指定 C++ 函数 namespace。

匹配新版 TVM 中的 `tvm.ffi` 可能只是 `tvm_ffi` 的重导出别名。
“两个 Python 模块名同时存在”不一定等于“有两个独立原生注册表”，测试使用实际选中的模块判断。

这层是 Python/C++ 兼容设施，不是 ROC_SHMEM 通信后端；
它也不能单独修复不同 TVM C++ API 的全部差异。

### 9.2 逐文件兼容改动清单

下面每行都对应真实源码 diff。除特别说明外，只替换 FFI 导入/调用来源，
保留原来的函数名、算法体与 backend 选择逻辑。

| 修改文件 | 对应代码块 | 功能与修改意义 |
| --- | --- | --- |
| [tilelang/_ffi_api.py](/public/home/baoboyuan/TileLang_DCU/tilelang/_ffi_api.py:3) | import 与第 6 行 init_ffi_api | 从统一注册表暴露 `tl` namespace 的 C++ API |
| [tilelang/transform/_ffi_api.py](/public/home/baoboyuan/TileLang_DCU/tilelang/transform/_ffi_api.py:3) | import 与第 6 行初始化 | 同样处理 `tl.transform`；没有在这里新增 M1 pipeline pass |
| [tilelang/language/ast/_ffi_api.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/ast/_ffi_api.py:21) | 第 23 行初始化 | 使用同一桥接层暴露 `script.ir_builder.tir` |
| [tilelang/carver/analysis.py](/public/home/baoboyuan/TileLang_DCU/tilelang/carver/analysis.py:7) | get_global_func 导入；第 119 行原有 NormalizePrimFunc 查询 | 保证分析代码从同一注册表取函数；没有改 tile 搜索算法 |
| [tilelang/contrib/hipcc.py](/public/home/baoboyuan/TileLang_DCU/tilelang/contrib/hipcc.py:12) | 第 94 行 HIP compile callback 注册 | 编译回调接入统一 FFI；没有新增 ROC_SHMEM device-link |
| [tilelang/contrib/nvcc.py](/public/home/baoboyuan/TileLang_DCU/tilelang/contrib/nvcc.py:14) | 第 322、387、533、546、563 行注册器 | libdevice、compute version、BF16/FP8/TMA 查询改用统一注册；不是实现 CUDA 通信 |
| [tilelang/contrib/rocm.py](/public/home/baoboyuan/TileLang_DCU/tilelang/contrib/rocm.py:25) | 第 105、129、231 行注册器 | ROCm link、bitcode 路径、架构查询回调统一注册；另补文件尾换行 |
| [tilelang/engine/callback.py](/public/home/baoboyuan/TileLang_DCU/tilelang/engine/callback.py:3) | 第 15/26 行 register_global_func(f=...) | HIP/CUDA 源码后处理回调统一注册 |
| [tilelang/engine/lower.py](/public/home/baoboyuan/TileLang_DCU/tilelang/engine/lower.py:11) | 第 59/128 行注册；第 187、202、217 行起的 backend 查询 | 统一编译回调和 host/device/source-only codegen 查询；没有接入 M1 normalize/verify |
| [tilelang/ir.py](/public/home/baoboyuan/TileLang_DCU/tilelang/ir.py:4) | 第 9 行起全部 register_object 装饰器 | Fill、AtomicAdd、Copy、Conv2DIm2Col、两类 GemmWarpPolicy、Gemm/GemmSP、FinalizeReducer、Parallel、Reduce、CumSum、Region、ReduceType 的 Python 包装统一注册；节点算法没有改写 |
| [tilelang/language/frame.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/frame.py:4) | 注册函数导入；第 110 行 LetFrame 装饰器沿用该别名 | 保证已有 LetFrame Python/C++ 对象一致 |
| [tilelang/language/kernel.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/kernel.py:8) | register_object 来源；第 94 行 KernelLaunchFrame | 保留原 launch frame 行为；未在这里实现 persistent compute/comm 分工 |
| [tilelang/language/warpgroup.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/warpgroup.py:4) | register_object 来源；第 9 行 WarpSpecializeFrame | 既有对象注册兼容，不代表 HIP 自动 WS 已支持 |
| [tilelang/language/parser/operation.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/parser/operation.py:22) | DataType/DataTypeCode 导入 | 从 `tvm.runtime` 获取，避免依赖旧 FFI 私有路径；表达式运算规则未改 |
| [tilelang/language/v2/builder.py](/public/home/baoboyuan/TileLang_DCU/tilelang/language/v2/builder.py:7) | Array/Map 导入；第 395 行 isinstance | 容器类型判断使用与当前注册表一致的类，避免前端不认识原生容器 |
| [tilelang/layout/layout.py](/public/home/baoboyuan/TileLang_DCU/tilelang/layout/layout.py:4) | 第 11 行 tl.Layout 注册 | 保留 Layout 实现，只统一对象注册入口 |
| [tilelang/layout/fragment.py](/public/home/baoboyuan/TileLang_DCU/tilelang/layout/fragment.py:5) | 第 12 行 tl.Fragment 注册 | 保留 Fragment 布局实现 |
| [tilelang/tileop/async_copy/__init__.py](/public/home/baoboyuan/TileLang_DCU/tilelang/tileop/async_copy/__init__.py:5) | 第 7 行 AsyncCopyPy 注册 | 去掉旧 `tvm._ffi` 来源；异步 copy 算法未改；另补尾换行 |
| [tilelang/tileop/gemm/__init__.py](/public/home/baoboyuan/TileLang_DCU/tilelang/tileop/gemm/__init__.py:8) | 第 18/25 行 infer_layout/lower 注册，第 60 行 GemmPy 注册 | GEMM Python callback 和对象统一注册；没有改 MMAC 计算逻辑 |
| [tilelang/tileop/gemm_sp/__init__.py](/public/home/baoboyuan/TileLang_DCU/tilelang/tileop/gemm_sp/__init__.py:10) | 第 15/21 行 callback，第 28 行 GemmSPPy | 稀疏 GEMM 同类兼容调整 |

这解释了为什么 M1 的 diff 涉及 GEMM、CUDA helper 和 layout 等文件：
TileLang 根导入会接触这些模块，必须保证整个被导入链使用一致的桥接设施。
**修改文件名里有 gemm，不表示本轮已完成新的 fused GEMM 算法。**

## 10. 构建与环境改动逐项解释

### 10.1 CMakeLists.txt

新增选项在 [TILELANG_SKIP_SUBMODULE_UPDATE](/public/home/baoboyuan/TileLang_DCU/CMakeLists.txt:18)，默认 OFF。
开启后跳过原有 `git submodule update --init --recursive`，不在配置阶段自动切换用户依赖版本。

新 C++ 文件通过原来已有的 [file(GLOB TILE_LANG_SRCS)](/public/home/baoboyuan/TileLang_DCU/CMakeLists.txt:143) 中 `src/op/*.cc` 收集，
不是本次又新增一套独立编译器 target。重新配置并构建后才会进入原生库。
已有 [tilelang_module target](/public/home/baoboyuan/TileLang_DCU/CMakeLists.txt:263) 负责产生动态库。

### 10.2 tilelang/env.py

[开发构建目录覆盖](/public/home/baoboyuan/TileLang_DCU/tilelang/env.py:33) 增加 `TILELANG_DEV_BUILD_ROOT`：

- 未设置：仍使用原来的 build；
- 设置：要求绝对路径；
- 从该目录的 lib 和 tvm 子目录加载产物。

它只改变开发模式库搜索位置，不改变算法、目标架构或 Slurm 资源。
也不会自己选择正确 TVM Python 源码；仍要按构建说明设置匹配环境。

### 10.3 .gitignore

[build-m1 忽略规则](/public/home/baoboyuan/TileLang_DCU/.gitignore:22) 避免提交原生产物。
[language/distributed 例外](/public/home/baoboyuan/TileLang_DCU/.gitignore:27) 防止宽泛 dist*/ 忽略新源码。
M0 的 distributed 和 testing 例外保留，不删除用户文件。

### 10.4 为什么要独立 build-m1，而不是直接切换原子模块

用户原有 TVM：`883e96b42ae0df40c2f7194cc932bbcd9d0c5627`。
主仓库 gitlink 所需 TVM：`79ed747db67e60d3a1889d8afd33473bc2424ade`。

旧版本缺少当前 TileLang 所用的一些对象宏、For.step/ForFrame 接口。
只改 Python FFI 入口不足以让全部 C++ 源码兼容。

实际采用的隔离环境：

| 位置 | 用途 |
| --- | --- |
| `/public/home/baoboyuan/.cache/tilelang-m1/tvm-79ed747` | 匹配 TVM 的独立 worktree，不切换用户原目录 |
| `/public/home/baoboyuan/.cache/tilelang-m1/python` | 匹配 FFI Python 包，版本 0.1.3.dev11+gae346ec92 |
| `TileLang_DCU/build-m1` | 原生 libtilelang_module、TVM 和 Cython wrapper 产物 |
| `TileLang_DCU/.venv-dcu` | 原有用户虚拟环境；构建期补充 pytest/Z3 等依赖 |

这些是依赖与构建产物，不是 M1 新增的设备算法文件。
旧 build 曾用于诊断，其中 TVM 库有过重建；不能假定旧 build 未动，也不能将它当成本轮验收产物。

完整命令保存在 [构建步骤](/public/home/baoboyuan/TileLang_DCU/docs/compiler_internals/distributed_fusion_m1_build.md:31) 和
[测试环境](/public/home/baoboyuan/TileLang_DCU/docs/compiler_internals/distributed_fusion_m1_build.md:67)。
原生 target 与 Cython wrapper 都需要构建。登录节点缺少完整 HSA 链接目标，因此当时在计算节点加载真实库；
这不代表测试执行了四卡通信。

## 11. 三份新增测试文件分别在测什么

### 11.1 test_m1_ir.py：实现自身的 10 项验收

辅助构造和 IR 遍历见 [IR helpers](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py)；不再包含文本 parser namespace。

| 测试入口 | 检查内容 |
| --- | --- |
| [registered_ops](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | 真实原生 op registry 及 opaque/embed-info 属性 |
| [remote_alias_offset](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | 两个 peer 的 distinct typed Var，非零 offset、strides、alignment 保留 |
| [repeated_view](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | 同一 view 多次访问不重复使用绑定 Var |
| [local_fallback](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | remote=False / peer=-1 不生成 remote_address |
| [uniform_peers](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | 整数函数参数、serial loop、真实 blockIdx 绑定 |
| [nonuniform_rejection](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | threadIdx、未知变量、不一致 loop 和非法 peer 类型拒绝 |
| [unsupported_position](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | remote load 位于 Let value 时拒绝 |
| [copy_scalar_validation](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | copy 大小/连续区间、scalar dtype、cache、multimem 限制 |
| [sync_defaults_and_invalid_input_validation](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | 默认值与非法 raw wait 校验；回读部分已删除 |
| [public_v2_frontend](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/test_m1_ir.py) | 真实公开 @T.prim_func 构造，不用 parser proxy 替代 |

### 11.2 test_m1_review.py：父代理独立探针

辅助 PrimFunc 构造位于 [_function](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:10)。
该文件不是子代理修改自己的断言来通过测试，关键组如下：

| 代码块 | 检查内容 |
| --- | --- |
| [SSA](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:14) | 用 TVM verify_ssa 检查重复 view |
| [wait peer](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:26) | 确认检查的是 args[3]，不是 semantics |
| [size/start](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:36) | 参数化负数、浮点、bool、负起点、越界 |
| [dtype/peer](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:52) | scalar load 类型不匹配、浮点 peer |
| [raw wait/modes](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:68) | raw TIR 不能绕过 flag dtype 和 scope 检查 |
| [uint32/local](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:83) | 溢出 payload 拒绝，TIR IntImm(-1) 本地退化 |
| [step/padded interval](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:99) | thread-varying step、动态 padded 行起点拒绝 |
| [raw signal](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:119) | uint64 大值不能静默缩成 uint32 |
| [auto_ws](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:127) | CUDA 自动 WS 显式拒绝 |
| [Let 派生](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:132) | threadIdx 经 Let 不得伪装 uniform；uniform Let 的合法绑定保留 |
| [JSON 序列化](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_m1_review.py:154) | 前端 token、规范化 alias/provenance 经真实 JSON 往返保留 |

参数化展开后共 25 项，并不是 25 个不同的 def。

### 11.3 test_ffi_compat.py：验证的不是一个假的注册表

入口：[真实子进程回归](/public/home/baoboyuan/TileLang_DCU/testing/python/distributed/ir/test_ffi_compat.py:10)。

测试创建新的 Python 进程，真实导入 tvm 和 tilelang，然后：

1. 检查实际选择的 FFI 模块；
2. 注册 decorator 形式和 f= 形式两个函数；
3. 经 get_global_func 调回来，结果都为 42；
4. 创建真实原生 Array/Map 并读取；
5. 子进程退出码必须为 0。

这种测试也能暴露导入期崩溃；不是仅检查 Python 文件能否通过语法解析。

### 11.4 历史完整验收结果及证据范围

2026-09-20 在 dcu0 上的两次完整结果：

| 组成 | 数量 |
| --- | --- |
| M0 契约回归 | 15 |
| M1 自身 IR 测试 | 12 |
| 父代理独立探针 | 25 |
| FFI 新进程回归 | 1 |
| 既有普通前端与 LetInline 回归 | 7 |
| 合计 | 60 |

两次均 60 passed、0 failed、0 skipped。证据：
[首次完整 JUnit](/public/home/baoboyuan/TileLang_DCU/build-m1/m1-final-review.xml:1)、
[重复完整 JUnit](/public/home/baoboyuan/TileLang_DCU/build-m1/m1-final-repeat.xml:1)。

本次写报告时读取并核对了这些 XML，没有把历史结果描述成今天新跑的四卡实验。
既有普通回归文件
[frontend_v2](/public/home/baoboyuan/TileLang_DCU/testing/python/language/test_tilelang_language_frontend_v2.py:1)、
[LetInline](/public/home/baoboyuan/TileLang_DCU/testing/python/transform/test_tilelang_transform_let_inline.py:1)
只是选取并运行，未因 M1 修改其实现。

2026-09-23 撤回回读扩展并清理测试后，主代理在 dcu1 使用相同 TVM/FFI/build-m1
环境重新验收：M0 15 项、M1 核心 10 项、独立探针 25 项、FFI 与普通回归 8 项，
合计 **58 passed、0 failed、0 skipped（7.70 秒）**。证据：
[清理后 JUnit](/public/home/baoboyuan/TileLang_DCU/build-m1/m1-test-cleanup-20260923.xml)。
该结果不包含已删除的文本回读测试，不能与历史 60 项结果视为相同覆盖范围。

## 12. 把两个模块连起来理解，但不要把计划当成已完成

以下是将来的职责分配，不是当前自动执行的调用链：

| 问题 | M0 提供什么 | M1 提供什么 | 仍待后续完成 |
| --- | --- | --- | --- |
| 我是谁、总共多少 PE | spec.nranks 与 RuntimeBinding 校验 | rank/world intrinsic | 通信 context 实际注入 |
| tile 写谁、写哪个槽 | owner/owner_slot/batch 公式 | remote_view、copy/store 表达 | schedule 到 DSL 的生成器、远端指针解析 |
| 数据好了通知谁 | ready_key | signal_set/fence/wait | writer 协议、系统原子和内存序验证 |
| 工作区多少、何时可复用 | workspace/reset/completion 契约 | IR 保留 Buffer provenance | 实际分配、映射、barrier、生命周期 |
| 哪些 Block 计算/通信 | compute/comm 数量和映射 | specialization 词法标记 | persistent 调度生成与资源安全 |

特别要记住：

- M0 的 `bind_runtime` 目前是校验，不是绑定硬件；
- M1 的 `remote_address` 目前是符号，不是已解析的设备指针；
- M1 的 scope 目前是标记，不是调度器；
- opaque effect 是必要信息，不是完整内存序正确性证明；
- 通过 IR 测试，不等于融合 kernel 已经生成、通信或跑得更快。

## 13. 最实用的阅读路线

如果只想先抓住核心，按这个顺序看：

1. 第 8 节的最小例子与真实 @T.prim_func 测试；
2. common.py 的 remote_view；
3. ir.py 的 normalize_remote_views → _entry → _make_alias → _wrap；
4. _collect_bindings / _is_block_uniform；
5. sync.py 与 comm.py 对照原生 op 参数表；
6. 第 7.3 节的回读扩展撤回理由与当前验收边界；
7. 最后阅读 FFI/构建兼容附录，避免一开始陷进环境问题。

**M1 的核心价值是：通信意图不再藏在 HIP 字符串里，而成为能被序列化、检查和继续 lowering 的程序结构。
下一步才是让这些结构真正变成经过验证的 DCU 通信代码。**
