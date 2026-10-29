# 内置 HIP-IPC 通信运行时（M2）

本目录将用户授权的 `dcu_ipc` Context/Allocation/MPI bootstrap 源码内置到
TileLang；构建和运行**不**读取、链接或安装外部 `~/dcu_ipc` 二进制。保留的
`include/dcu_ipc` 与 `src/{bootstrap_mpi,context,allocation,metadata}` 是该
授权接口源的直接内置副本，`runtime_bridge.cc` 只增加 TileLang 的窄 C ABI 和
device-resident `DeviceView` 描述符上传。

构建条件：系统 HIP/DTK 和系统 MPI（例如 HPC-X）是前置依赖，运行时不会初始化
或终结 MPI。调用方须先以 `MPI_THREAD_FUNNELED` 或更强等级初始化 MPI；每个进程
只能有一个 active Context，`symm_alloc/free/sync/finalize` 必须由所有 rank 按相同
顺序集体调用。CMake 默认构建 `tilelang_dcu_ipc_runtime`，可用
`cmake --install build-old --prefix "$PWD/build-old/install" --component dcu_ipc`
单独安装 header 与 shared library；不要假设对默认 `/usr/local` 有写权限。

Python 的 `tilelang.distributed.ipc_runtime.IPCContext` 是惰性 ctypes 封装；它不
引入 torch/DLPack，也不接管 MPI。`DeviceAllocation.device_view()` 返回可直接传给
`tilelang.compile(..., execution_backend="ctypes")` 的普通 `uint64[6]` device
descriptor，配合 `bind_ipc` 追加的 `rank/world_size` 参数。

当前硬件验证范围是单机 4 PE、gfx936、FineGrained payload/flags。它不宣称 wheel
打包已验证，也不替代 M0 的待定 IPC ABI 设计。GPU device helper 对 peer、rank、
world、起始地址范围和对齐 fail-closed。block copy 的完整区间仍以 DSL 的静态 typed
region 校验及调用方 allocation 容量为前提；它是 CTA cooperative copy，发布协议仍需
调用方显式执行每-writer fence、CTA barrier 与 thread-zero release/acquire。

## 可复现实测

首次配置必须在 MPI 环境已设置后完成；下面既不依赖已有 CMake cache，也把安装写到
用户可写目录。`scripts/dev_old/build.sh` 已包含对应的 `cmake -S/-B` 配置选项；如需
手动配置，应复用其 `-DTILELANG_SKIP_SUBMODULE_UPDATE=ON`、`-DUSE_ROCM` 和 Python
解释器选项。

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
export MPI_HOME=/public/software/hpcx/2.20/ompi-slurm-pmix
export OPAL_PREFIX=$MPI_HOME PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
scripts/dev_old/build.sh
cmake --install build-old --prefix "$PWD/build-old/install" --component dcu_ipc
```

`testing/python/distributed/m2_dsl_block_litmus.py` 是公开 DSL 的 4 PE 验收入口：
它调用 `bind_ipc` 后走普通 `tilelang.compile(..., execution_backend="ctypes")`，不
包含手写通信 HIP kernel。测试覆盖 256 线程、3 个不等长 typed copy（四 rank 的全 12
方向）、每 writer 系统 fence + CTA barrier + thread-zero release、thread-zero acquire
wait + CTA barrier + 全 CTA weak payload read、epoch ACK 复用、旧值 volatile 预读，以及
每 epoch/slot-tail/allocation guard 的随机 uint32 载荷校验。默认 3 epoch x 2 host reset
batch；可用环境变量放大压力。

```bash
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python testing/python/distributed/m2_dsl_block_litmus.py
# 压力：M2_LITMUS_EPOCHS=32 M2_LITMUS_BATCHES=8 后执行同一命令
```

该脚本显式关闭 ctypes 持久缓存：当前后端没有 host source 可写入缓存，这不是 M2
试图解决或声明支持的功能。
