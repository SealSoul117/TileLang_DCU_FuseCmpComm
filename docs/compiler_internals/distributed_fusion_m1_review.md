# M1 实现与验收报告

日期：2026-09-20。依据：`distributed_fusion_dcu_plan.md` 的 M1 章节。

2026-09-23 更新：用户授权撤回通用 `region`/`copy` 的文本回读扩展并删除依赖测试。
下面的 60 项结果保留为 2026-09-20 历史记录，不代表调整后的测试数量；更新见末节。

**状态：M1 实现完成，第二轮主代理独立验收通过。已暂停后续模块，未进入 M2。**

## 分工与范围

- 实现子代理使用 `gpt-5.6-terra`、`high`，不继续分派。
- 主代理负责设计、构建协调、独立验收；不执行 Git commit。
- 本模块只实现分布式 DSL、TIR 原语及其规范化/校验。没有启动 M2，
  没有实现或验证 ROC_SHMEM 设备 helper、硬件内存序或完整融合 GEMM。
- 按用户要求，M1 完成报告后暂停后续模块；两轮正式 review 仍未通过则先汇报。

## 已实现内容

- `tilelang/language/distributed/{common,comm,sync,ir}.py`：rank/world、
  specialization 词法作用域、typed remote view、block copy、scalar load/store、
  uint32 signal/wait、fence，以及独立的 remote view 规范化与校验入口。
- `src/op/distributed.cc`：原生 op 注册和副作用分类。除前端信息 token 外均为
  opaque；这并不代表完整编译流水线已通过不重排验证。
- remote view 保留 Buffer/region provenance、dtype、shape、strides、offset 和
  alignment；规范化使用 typed pointer + LetStmt + DeclBuffer，不使用 Python 全局表。
- HIP 首期不支持的 CUDA multimem/cache/automatic warp specialization 显式报错。
- 真实 IR 专项测试、父代理独立边界测试与中文接口文档已加入并通过验收。

## 构建与环境

当前主仓库与原检出 TVM 不匹配，旧 TVM 缺少新对象宏、For step 和 ForFrame 接口。
没有切换用户子模块，改用主仓库 gitlink 指定的匹配版本独立构建：

| 项目 | 验收环境 |
| --- | --- |
| TileLang HEAD | `b2bc99b0a3ab03cb5ffce731db87048931e011a6` |
| 匹配 TVM | `79ed747db67e60d3a1889d8afd33473bc2424ade`，`0.23.dev0` |
| 匹配 FFI | `ae346ec92a3c386f1376064ae086aae72947c329`，`0.1.3.dev11+gae346ec92` |
| 原生产物 | `build-m1/` 的 `tilelang_module`、`tilelang_cython_wrapper` |
| Python / DTK | `.venv-dcu` Python 3.10.20 / DTK 26.04 |
| 运行环境 | Slurm 单卡 dcu0，8 CPU、32 GiB；M1 无需四卡通信 |

原生构建、真实 TileLang 根导入及 `tl.get_rank` 原生注册查询已通过。
新增可选 `TILELANG_DEV_BUILD_ROOT` 隔离加载入口，以及统一 Python FFI 兼容入口，
避免旧内置 FFI 与新外置 FFI 的独立注册表在同一进程混用。

完整复现环境和命令见 [M1 隔离构建环境](distributed_fusion_m1_build.md)。
原有 `build/` 曾用于诊断，TVM 库已重建；它不是本次通过构建的 M1 产物。

## Review 和测试记录

第一轮正式静态 review **未通过**并退回修改，主要问题为：重复 remote view
使用可能重复定义同一 Var，wait 的 peer 参数校验位置错误，以及 size/index、
raw region、scalar dtype、uint32 溢出和 loop step uniformity 检查不足。
以上问题已修正。子代理在真实 TVM 上完成补修、自测后提交冻结候选；主代理再开展
第二轮独立 review，而不是只采用子代理的测试结论。

真实测试还暴露了匹配 TVM 的 API 差异，包括 dtype/Buffer 类型接口、raw intrinsic
须使用 Op 对象、以及可选 `For.step=None`。最后一项原先被传入原生遍历器而导致
进程崩溃；主代理独立确认属性取值，修正为隐式步长 1 后，serial-loop/blockIdx
用例与完整套件均通过。没有通过跳过这些用例隐藏崩溃。

父代理在 2026-09-20 已经独立完成（历史版本）：

| 检查 | 历史结果 |
| --- | --- |
| M0 冻结契约回归 | 15 passed，`build-m1/m0-regression.xml` |
| FFI + 普通 DSL/LetInline 回归 | 8 passed，`build-m1/m1-compat-regression.xml` |
| M1 自身 IR 专项 | 12 passed |
| 主代理独立边界/序列化探针 | 25 passed |
| 第二轮正式 review 完整合跑 | **60 passed，0 failed，0 skipped**，6.54 秒 |
| 完整合跑结果 | `build-m1/m1-final-review.xml` |
| 新 Python 进程重复完整合跑 | **60 passed，0 failed，0 skipped**，6.30 秒，`build-m1/m1-final-repeat.xml` |

独立探针包含：同一 remote view 重复使用仍满足 SSA，wait 检查真正的 peer 参数，
非法大小/起点、dtype、uint32 溢出、raw region 绕过、thread-varying loop step、
threadIdx 经 Let 派生的 peer 拒绝、uniform Let 绑定保留、auto_ws=True 拒绝，
以及规范化前后真实 JSON 序列化。源码复核确认元数据和非零 offset 保留，绑定没有
提升到 loop/Let 之外，未添加伪 storage scope 或隐式本地通信替代。

未使用 mock 原生 op、跳过实际库加载或用语法检查冒充 IR 验收。
FFI 回归修正了一处测试自身的布局判定错误：新版 `tvm.ffi` 重导出 `tvm_ffi`
类型，不能仅用 `Map` identity 推断加载了两个独立注册表。

公开 `@T.prim_func` 的 v2 builder 分布式构造测试继续保留。2026-09-20 还测试了
通过专用 namespace 的真实 TVM 文本回读；2026-09-23 已按用户授权删除该辅助路径
测试及 namespace，不再将文本回读列为当前验收保证。

完整验收复现：先按构建文档导出环境，再在计算节点执行：

```bash
python -m pytest -q \
  testing/python/distributed/test_m0_contracts.py \
  testing/python/distributed/test_m1_ir.py \
  testing/python/distributed/ir \
  testing/python/language/test_tilelang_language_frontend_v2.py::test_argument \
  testing/python/language/test_tilelang_language_frontend_v2.py::test_expr \
  testing/python/language/test_tilelang_language_frontend_v2.py::test_dtype_str_repr \
  testing/python/language/test_tilelang_language_frontend_v2.py::test_marco_return \
  testing/python/language/test_tilelang_language_frontend_v2.py::test_prim_func_generator \
  testing/python/transform/test_tilelang_transform_let_inline.py \
  --junitxml=build-m1/m1-test-cleanup-20260923.xml
```

## 验收边界与已知限制

- M1 的规范化/校验是独立入口，尚未接入最终设备 lowering 流水线。不能将这些
  原语直接当作已可执行的 HIP 通信 kernel；原子内存序 litmus、系统可见性和最终
  跨 pass 不重排验证仍属于 M2 及后续模块。
- remote pointer 在语句内合法绑定；循环外 `rocshmem_ptr` 外提是后续设备实现的
  要求，本模块不声称已经满足该性能要求。
- 首期 remote view 支持显式 Call/region 与直接 BufferLoad/BufferStore 的语句访问。
  含远端 load 的任意复合算术表达式、Let value/控制条件等尚不通用支持，会报错，
  不会静默变成本地 load。`T.copy` 和写远端 epilogue 所需的基础形式已经测试。
- block copy 首期要求静态元素数及可证明的末维连续区间，不支持无范围证明的动态
  size/末维起点。scalar load/store 白名单为 int32/uint32/float32。
- 只验证当前匹配 TVM/FFI 的 DCU 开发环境，未声称全仓库 CUDA、所有旧 TVM 布局或
  既有 GEMM 性能回归全部通过。没有四卡通信和性能测试。

详细接口见 [M1 实现说明](distributed_fusion_m1.md)。按用户要求，交付本报告后
暂停，不开始 M2，不自动提交 Git commit。

## 保留情况

总计划、M0 spec/schedule/tests 的 SHA256 与恢复前一致。用户原有子模块内部干净，
其检出版本保持不变：TVM `883e96b42ae0df40c2f7194cc932bbcd9d0c5627`；
composable_kernel `1c45ca35dd5c215e0c1db1f40f01556f467f52a8`。
原有 `.venv-dcu/`、GEMM 生成示例等用户文件未删除，没有 Git commit。

## 2026-09-23：撤回辅助回读能力后的验收更新

用户明确授权删除依赖 `utils.region` 整数模式和 `copy` raw-region 扩展的测试。
删除两项完整测试（所有原语文本回读、普通 copy 文本回读），删除同步默认值测试中
最后的 remote-copy 回读部分，并移除专用 parser namespace。同步默认值和非法输入
断言保留；两项 raw wait/signal 探针仅将 region 的整数模式改为 `"r"`/`"w"`，继续
验证 dtype 和 uint64 溢出不能绕过检查。没有将失败测试改为 skip/xfail。

此删除是可接受的：正常 DSL → TIR → normalize/verify 路径无需打印后重新解析，
不应仅为辅助文本回读测试修改通用 `region`/`copy` API。但它确实缩小了验收范围：
原计划的文本 print/parse round-trip 项经用户授权不再作为当前 M1 的验收条件，
不再声称已保证文本回读。公开前端、核心 IR 正确性与 JSON 往返仍被检查；JSON
序列化不是文本回读的替代证明，后续设备/通信验收要求不变。

主代理检查删改范围后，在 dcu1、Slurm 作业 9200（单卡、2 CPU、8 GiB）使用原先
匹配的 TVM/FFI/build-m1 环境，重新执行完整回归命令：

| 当前测试组 | 结果 |
| --- | --- |
| M0 契约 | 15 passed |
| M1 核心 IR | 10 passed |
| 独立边界/序列化探针 | 25 passed |
| FFI 与普通前端/LetInline | 8 passed |
| 合计 | **58 passed，0 failed，0 skipped，7.70 秒** |

证据：`build-m1/m1-test-cleanup-20260923.xml`。历史 60 项 XML 未覆盖。
`utils.py` 与 `copy_op.py` 仍与 Git HEAD 一致，未为了测试重新引入兼容分支；
`git diff --check` 通过。当前结论是调整后验收范围内通过，仍未进入 M2、未提交 commit。
