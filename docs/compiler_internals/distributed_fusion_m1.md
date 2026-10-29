# M1：分布式 DSL 与 TIR 原语

日期：2026-09-20；测试范围更新：2026-09-23。本文是 `distributed_fusion_dcu_plan.md` 的 M1 实现说明，
只覆盖 DSL、可检查 TIR 与原生 op 注册；不包含 ROC_SHMEM/HIP device helper、
GPU 通信、persistent 调度或融合 GEMM（分别属于 M2 及后续模块）。

## 公开 DSL

`tilelang.language`（即 `T`）和 `tilelang.language.distributed` 都导出以下
接口：

- `get_rank()`、`get_num_ranks()`：返回动态运行时 intrinsic，不读取 Python
  编译进程的 rank。
- `sm_specialize_scope(auto_ws=False)`：真实 `with` 词法 `AttrStmt` frame。
  HIP M1 只接受 `False`；`True` 明确报 CUDA automatic warp-specialization
  capability 不支持，且不宣称分配 SM。
- `fence_sys()`、`signal_set(flag, value, peer=-1)`、
  `wait_eq/ge(value, expected, peer=-1, scope=SYS, semantics=ACQUIRE)`：flag
  限定为 `uint32`，wait/signal/fence 是具有 opaque effect 的 TIR op。
  M1 不把这表示为已验证的 HIP acquire/release 硬件实现，也不把 fence 当作
  CTA barrier；该证明和设备实现属于 M2。
- `put_block(src, dst, size, dst_pe=-1)`、
  `get_block(src, dst, size, src_pe=-1)`：`size` 是同 dtype 的元素数量，
  不是字节数；接口不隐含 ready 信号。首期只接受一段已证明连续的 typed run：
  末维 stride 必须为 1，所有静态 index 均需在界内、静态长度不得跨越行尾。
  M1 不接受动态长度，因为尚无可携带的 ``0 <= start`` 与
  ``start + size <= extent`` 范围证明；不会把它默默当成跨行平坦 copy。
- `st(dst, value, scope='gpu', sem='weak', na=False, dst_pe=-1)`、
  `ld(src, value, scope='gpu', sem='weak', na=False, nc=False, src_pe=-1)`：
  与 tilescale 的签名/默认值对齐。M1 HIP 白名单只接受 `gpu/sys` 与显式
  语义集合；`na`、`nc` 或 `cache`/multimem 请求显式拒绝。`ld` 的 value 必须是
  可写的 Buffer/BufferLoad，且 dtype 与 source 相同。
- `remote_view(buffer, peer, remote=True)`：只接受 global Buffer。`remote=False`
  （以及常量 `peer=-1`）是普通本地 alias；其余情况返回仍有原 dtype、shape、
  strides、alignment、offset factor 与 elem offset 的 typed view。

`put_block/get_block` 的 `peer=` 是兼容别名；新代码使用方向明确的
`dst_pe/src_pe`。不支持的 PTX cache/multimem 功能绝不会静默映射到 HIP。
与 tilescale 同名的 `multimem_ld_reduce`、`multimem_st`、`multimem_red`、
`multimem_tma_store`、`multimem_signal`、`multimem_signal_add` 在 M1 表面可见，
但都立刻抛出明确的 capability error；它们不会生成 PTX、call_extern 或本地伪替代。

## remote view 的两段 TIR 表示

未规范化 view 的 `elem_offset` 带有 `tl.remote_view_token`。token 的参数有
`tl.region(base)` provenance、peer、原 view 的 elem offset，以及 shape/stride
元数据；因此它是可序列化的 IR，而非 Python 全局 alias 表或 `call_extern`
字符串。未规范化 token 不允许直接进入本地 codegen。

`normalize_remote_views(PrimFunc)` 将每个 remote Buffer **身份**转换为独立：

```text
Let(remote_ptr_N: PointerType(dtype, "global"),
    tl.remote_address(tl.region(base), peer),
    DeclBuffer(remote_alias_N[data=remote_ptr_N,
                              shape/strides/alignment=原 view,
                              elem_offset=原 base offset],
               use(remote_alias_N)))
```

`remote_address` 的契约是返回 allocation base，别名 Buffer 的 `elem_offset`
只应用一次。于是同一 base 的不同 peer（甚至同 peer 的不同 view identity）不会
共用 data Var，也不会把 nonzero offset 加两次。base/alias region 与 provenance
仍在 IR 中可见，普通 `T.copy` 可消费规范化后的 alias。

绑定被放在包含实际 `Evaluate`/`BufferStore` 访问的最小语句内，避免 loop-local、
Let-local 或 blockIdx peer 被错误地提升到 PrimFunc 根部而变成自由 Var。这不是
M2 所需的循环外地址外提优化。remote load 若出现在 M1 无法合法包裹的 Let value、
if condition 等位置，会早报错，绝不退化成 local access。

`verify_distributed_ir` 默认要求已规范化；
`assert_remote_views_normalized` 可单独用作 lowering 前门禁。

## peer 的 block-uniform 验证

M1 采用保守白名单：常量、函数 scalar 参数、`get_rank/get_num_ranks`、带
`thread_tag=blockIdx.*` 的线程绑定、以及 bounds 本身 block-uniform 的 serial/
unrolled loop var（和其算术/Let alias）可用。带 `thread_tag=threadIdx.*` 的变量、
其 Let alias、非均匀 loop bound、BufferLoad、未知 Call 与未绑定 Var 都拒绝。
判断以 `IterVar.thread_tag` 和 Var 身份为准，不以名称猜测。

这只证明 block-uniform，不证明跨设备硬件内存序。

## 原生注册与副作用

`src/op/distributed.cc` 注册 rank/world、remote token/address、fence/signal/
wait、block copy 及 scalar ld/st。除只携带 frontend 信息的 token（embed-info）
外，均标记 `TCallEffectKind=kOpaque`，使后续 pass 能够识别其不是纯表达式；完整的
跨 pass 不重排证明与最终 lowering 验收不属于 M1。region 参数使读写 Buffer 信息
保留在 TIR，而不是隐藏在不透明 extern 名称中。

分布式 op 的打印名称和部分重建 hook 仍保留，但不保证打印出的 TIR 文本能通过
`T.region`/`T.copy` 重新解析。按用户要求，已撤回 `utils.py` 的整数访问模式支持和
`copy_op.py` 的 raw-region 回读分支；普通 DSL 构造 IR 不依赖这两项扩展。

## 测试

`testing/python/distributed/test_m1_ir.py` 覆盖：

- op registry/effect；sync 默认值；非法 scope/type/cache/multimem；
- typed block copy 的大小、dtype、stride、跨行和负长度；
- `remote=False` 本地退化；非零 offset/stride/alignment 保留；同 base 双 peer
  alias 的不同 typed data Var；
- function scalar、serial-loop 与实际 blockIdx peer；threadIdx、其派生/未知
  peer 及非均匀 loop 的拒绝；无合法绑定位置的 remote load 拒绝；
- 规范化 remote `T.copy`、原语构造与校验，以及公开 `@T.prim_func`/v2 builder
  构造（scope、通信原语、remote view/copy）后再 normalize/verify。

### 2026-09-23：删除文本回读测试的范围与理由

按用户授权，删除 `test_all_m1_primitives_print_parse_roundtrip`、
`test_ordinary_local_dsl_parse_print_regression`，以及原
`test_sync_defaults_and_print_parse_roundtrip` 中的 remote-copy 文本回读部分。
最后一项保留同步默认值和非法输入检查，并改名为 `test_sync_defaults_and_invalid_input_validation`。
只供回读使用的 `_RoundtripNamespace`/`_from_source` 也一并删除。

这次删除是可接受的范围调整：正常路径是公开 DSL 直接构造 PrimFunc，再做
normalize/verify 和后续 lowering，并不需要先打印成文本再读回来。不应仅为这一
辅助测试能力扩大已有 `region`/`copy` API 的输入契约。核心的原生 op、副作用、
类型/边界、SSA、peer 一致性、offset/strides、公开前端构造和真实 JSON 序列化
测试继续保留。两项 raw wait/signal 安全检查仅改用原有 `"r"`/`"w"` 构造 region，
不再依赖整数模式，校验断言不变。

代价也明确保留：原计划中“解析/打印 round-trip”的文本回读验收项，经此次用户
授权不再作为当前 M1 的验收条件，不能再宣称它通过。JSON 序列化仍被测试，但
不是文本回读的等价替代。此调整不豁免后续设备代码、同步语义或通信正确性验收。

在原生库可加载的 DCU 节点运行前，先按
[`distributed_fusion_m1_build.md`](distributed_fusion_m1_build.md) 的“测试环境”章节
导出匹配的独立 TVM、FFI 与 `build-m1/tvm` 路径；不能回退到仓库内旧 build。随后运行：

```bash
srun --jobid=<作业号> --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --exact \
  env ROCM_PATH=/public/software/dtk/26.04 \
  LD_LIBRARY_PATH=/public/software/dtk/26.04/lib:/public/software/dtk/26.04/hip/lib:/public/software/dtk/26.04/llvm/lib \
  .venv-dcu/bin/python -m pytest -q testing/python/distributed/test_m1_ir.py
```

此测试是实际 TVM/TileLang TIR 测试，不是 mock；它不运行 M2 device helper，
也不能作为 acquire/release litmus 通过的声明。
