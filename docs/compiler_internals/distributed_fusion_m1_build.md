# M1 隔离构建环境

日期：2026-09-20。测试结论见 `distributed_fusion_m1_review.md`；本文记录环境，
不代替验收结果。

## 为什么使用独立目录

主仓库 `b2bc99b0a3ab03cb5ffce731db87048931e011a6` 记录的 TVM 是
`79ed747db67e60d3a1889d8afd33473bc2424ade`，用户当前实际检出是
`883e96b42ae0df40c2f7194cc932bbcd9d0c5627`。后者缺少当前 TileLang 使用的
新版对象宏、For step 和 ForFrame 接口。仅适配宏不能解决 TIR API 差异。

没有切换用户已有 TVM/CK 子模块，也没有提交 Git commit。M1 使用：

| 项目 | 本次路径/版本 |
| --- | --- |
| TileLang 源码 | `/public/home/baoboyuan/TileLang_DCU` |
| 匹配 TVM 的独立 worktree | `/public/home/baoboyuan/.cache/tilelang-m1/tvm-79ed747` |
| TVM commit | `79ed747db67e60d3a1889d8afd33473bc2424ade` |
| TVM FFI commit | `ae346ec92a3c386f1376064ae086aae72947c329` |
| 独立 FFI Python 包 | `/public/home/baoboyuan/.cache/tilelang-m1/python` |
| FFI Python 版本 | `0.1.3.dev11+gae346ec92` |
| 独立原生产物 | `/public/home/baoboyuan/TileLang_DCU/build-m1` |
| Python | `.venv-dcu/bin/python`，3.10.20 |
| DTK | `/public/software/dtk/26.04` |

独立 TVM 的 dmlc-core/rang 使用指向原有同 commit 依赖的只读用途软链接；FFI
及其 dlpack/libbacktrace 在独立目录初始化。原有 `build/` 曾用于旧依赖构建诊断，
其中 TVM 库已重建；它不是本次 M1 的验收产物。

## 配置与构建

在有完整 DTK/HSA 动态库的计算节点上运行。先激活已有虚拟环境：

```bash
cd /public/home/baoboyuan/TileLang_DCU
source .venv-dcu/bin/activate
export TVM_ROOT=/public/home/baoboyuan/.cache/tilelang-m1/tvm-79ed747
export ROCM_PATH=/public/software/dtk/26.04
export PATH="$ROCM_PATH/llvm/bin:$PATH"
export CPATH="$ROCM_PATH/include:$ROCM_PATH/hip/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S . -B build-m1 \
  -DTILELANG_SKIP_SUBMODULE_UPDATE=ON \
  -DUSE_CUDA=OFF -DUSE_ROCM="$ROCM_PATH" \
  -DUSE_LLVM=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-m1 --target tilelang_module tilelang_cython_wrapper -j 6
```

`TILELANG_SKIP_SUBMODULE_UPDATE=ON` 必须保留，防止配置时自动改变用户的子模块。
`CPATH` 用于该环境下 TVM 的 HIP 头文件搜索，`llvm/bin` 提供链接所需的 `ld.lld`。
已有虚拟环境另外安装了 `z3-solver==5.1.0.0`、`pytest==8.4.2`、`setuptools-scm`。

精确匹配 FFI 包的构建命令为：

```bash
CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install \
  --no-deps --no-build-isolation \
  --target /public/home/baoboyuan/.cache/tilelang-m1/python \
  "$TVM_ROOT/3rdparty/tvm-ffi"
```

该包已安装，正常复测无需重复安装。构建工具依赖 scikit-build-core、Cython、
setuptools-scm；不要将不同版本 FFI 的 Python 扩展和原生注册表混装。

## 测试环境

在上述环境基础上设置：

```bash
export TILELANG_DEV_BUILD_ROOT=/public/home/baoboyuan/TileLang_DCU/build-m1
export TVM_IMPORT_PYTHON_PATH="$TVM_ROOT/python"
export TVM_LIBRARY_PATH="$TILELANG_DEV_BUILD_ROOT/tvm"
export PYTHONPATH="/public/home/baoboyuan/.cache/tilelang-m1/python:$TVM_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

python -m pytest -q \
  testing/python/distributed/test_m0_contracts.py \
  testing/python/distributed/test_m1_ir.py \
  testing/python/distributed/ir
```

`TILELANG_DEV_BUILD_ROOT` 是可选的绝对路径覆盖，未设置时仍使用原先的 `build/`。
禁用的是 **可选** Torch C-DLPack 扩展的自动 JIT，不是禁用 TVM 原生库、M1 op
或任何验收检查。M1 是 IR 测试，不依赖该张量交换加速扩展。

登录节点上的 HSA 库链接最终指向计算节点才有的 `/opt/hyhal`，因此这里仍需
计算节点进行真实库加载。M1 不需要四卡通信分配；单卡计算节点环境即可。
