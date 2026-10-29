# old 上游同步、重建和路径隔离

日期：2026-10-01。上游同步、原生构建及路径隔离已完成；IPC M0 补齐尚未通过 review，未进入 M2。

## 源码与保留范围

- `git fetch origin` 确认 `origin/develop` 最新为
  `e19ab375aaee0b364e1d28c838fab5ea8db939ce`，old 已从 `b2bc99b` fast-forward 到该提交。
- 这是上游已有提交的快进，不是新建 Git commit；没有推送。
- 上游代码与本地 M0/M1 改动合并后，仅 GEMM Python 注册入口存在文本冲突：保留
  上游 `layout_map=None` 参数和传递，同时保留本地统一 FFI 注册入口。
- 未改变用户已检出的 TVM、composable_kernel 子模块版本。
  使用 gitlink 所需的独立 TVM `79ed747` worktree 构建，FFI 为 `ae346ec92` 对应包。
- `utils.py`、`copy_op.py` 与更新后的上游一致；没有重新引入被撤回的文本回读扩展。
- develop、dcu_ipc 和手写融合参考目录不作修改。

实际功能所需的一处额外修复：`src/tl_templates/dcu_hip/copy.h:104` 将 `int` 地址分量
显式转换成 `uint32_t` 后初始化地址向量。最新上游尚未包含 develop 工作树中的这处修改。
修复前用 **old 的头文件**编译已有真实 GEMM device 源码，DTK 报
`non-constant-expression cannot be narrowed from type 'int' to 'uint32_t'`，
包含 `cp_async_gs_idxen<16,...>` 实际实例化失败；证据为
`build-old/logs/header-probe-before.log`。这影响用户请求的正常 GEMM HIP 编译，即使
删除所有测试也仍需修复；不是为测试扩展原有接口，未屏蔽诊断。

恢复材料位于 `/public/home/baoboyuan/.cache/tilelang-old-upgrade-20261001.C13o6g/`：
`local-before.patch` 是升级前 tracked diff，`source-before.tar.gz` 保存 tracked 源码和
未跟踪的项目源码/文档（不含虚拟环境及子模块）。另保留升级前 tracked 改动的 Git stash，
消息为 `old upstream upgrade 2026-10-01 preserve tracked M0 M1 changes`；改动已 apply，
不要再次盲目 apply。既有 build/build-m1 产物未删除，不是本轮验收产物。

## 重建与隔离入口

在计算节点、分配到的 DCU 资源内执行：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
bash scripts/dev_old/build.sh
source scripts/dev_old/env.sh
python scripts/dev_old/check_paths.py
python scripts/dev_old/gemm_smoke.py
```

本轮作业为 Slurm 9504、dcu0、单卡、8 CPU、32 GiB；从空目录 `build-old/`
重新配置、编译 `tilelang_module` 与 `tilelang_cython_wrapper`，不复用迁移前的 CMake cache。
构建日志为 `build-old/logs/configure.log`、`build-old/logs/build.log`。

`env.sh` 显式选择 old 的源码、`.venv-dcu`、`build-old`、模板头文件与独立 kernel cache；
覆盖 Python/TVM 路径，并从 PATH 排除其他 TileLang checkout。独立 TVM/FFI 属于声明的
外部依赖，不来自 develop。虚拟环境激活脚本、editable 路径及 TVM worktree 的改名前路径
已修复；dmlc-core/rang 依赖链接只指向 old。不要把旧文档中的原目录名当作新运行命令。

`check_paths.py` 检查实际 Python 模块位置、进程已加载的 native 库、单一 FFI registry、
CMake cache 和编译命令，不只检查环境变量。任何 develop 或改名前路径进入这些实际路径
都视为失败。

`gemm_smoke.py` 从 **old** 的 `examples/dcu/example_gemm.py::matmul` 获取真实 PrimFunc，
通过 HIP ctypes JIT 编译，实际启动 DCU kernel。开发虚拟环境的 Torch 是 CPU 版本，因此
此检查用 HIP 分配/拷贝和 NumPy FP32 参考，不伪装成 Torch GPU 测试。检查非方阵
512×768×256 和 1024×512×512；生成的 HIP 保存在 `build-old/gemm-smoke/`。
保留上游 DTK/gfx936 特殊编译参数，不据两项 smoke 宣称所有尺寸/架构性能已验证。

## 验收状态

新原生库构建成功，包括 `build-old/lib/libtilelang_module.so`、
`build-old/tvm/libtvm.so`、`build-old/lib/tilelang_cython_wrapper.cpython-310-x86_64-linux-gnu.so`。
这是原生开发构建产物，不是新发布的 pip wheel。

| 检查 | 本轮结果 | 证据 |
| --- | --- | --- |
| 原 M0/M1/FFI/前端回归 + 上游新增 rewind layout | 61 passed，0 failed，0 skipped | `build-old/old-upgrade-regression.xml`、`build-old/logs/regression.log` |
| Python/native 库实际路径 | PASS；无 develop 或改名前源码路径，单一 FFI registry | `build-old/logs/path-isolation.json` |
| GEMM 512×768×256 | 393216 元素全量数值检查 PASS | `build-old/logs/gemm-smoke.log` |
| GEMM 1024×512×512 | 524288 元素全量数值检查 PASS | 同上 |

两项 GEMM 与 NumPy FP32 GEMM 转 FP16 参考比较，rtol=atol=1e-3，最大绝对差
均为 0.00390625；这是按相对/绝对误差共同判断的通过，不是逐位相同。保存 device HIP、
完整 ctypes wrapper 和编译后的 `kernel.so`，均位于 `build-old/gemm-smoke/` 对应形状目录。
没有进行四卡通信或性能对比，也没有据两项 smoke 宣称完整新融合编译器已实现。

首轮回归曾因改名前 pytest 字节码的 `co_filename` 仍指向 `TileLang_DCU`，出现
`inspect.getsourcelines: could not get source code`。通过独立
`PYTHONPYCACHEPREFIX=build-old/pycache` 修复路径隔离，未修改编译器/测试断言。
失败日志及 XML 保留为 `*before-pycache-isolation*`。
GEMM harness 初次保存 wrapper 使用了不适用于当前 ctypes adapter 的 accessor；改为
已有 `adapter.lib_code` 后运行通过，没有为 harness 改生产 API；首次记录为
`build-old/logs/gemm-smoke-first-harness-error.log`。

构建仍有 Z3 头文件返回值警告；HIP 编译仍有模板宏重定义和返回值警告，日志保留，
未屏蔽警告，不宣称 warning-free。

IPC M0 v2 第二轮 review 未通过并已暂停，详见
[独立审查报告](distributed_fusion_m0_ipc_review.md)。该部分尚未补齐；旧契约不受影响。
