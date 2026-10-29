# M2 HIP-IPC 实现报告

状态：2026-10-01 第二轮正式审查通过；主代理已独立复测。完整记录见
[M2 审查记录](distributed_fusion_m2_review.md)。没有创建 Git commit，也没有进入
M3、调度、GEMM 宏或 M0 修补；后续模块等待用户确认。

## 实际执行路径

M2 的公开入口是 `tilelang.distributed.bind_ipc(func, views)`。它仅声明已有 global
Buffer 到公开 `uint64[6]` `DeviceView` 参数的绑定，不偷偷替换用户原有参数；随后用户
照常调用 `engine.lower` 或 `tilelang.compile`。

1. `engine.lower` 在正常 LowerAndLegalize 管线前调用
   `prepare_ipc_primfunc`：M1 `remote_view` 先规范化和校验，追加普通 descriptor、rank、
   world-size 参数，并拒绝非 HIP、同名冲突或未绑定 base。
2. 内建 C++ `tl.transform.LowerDCUIPC` 在 `SplitHostDevice` 前重写整个 TIR 表达式树：
   `get_rank/get_num_ranks` 变为真实 ABI Var；remote provenance 解析为 descriptor；
   `remote_address`、put/get、signal/wait、fence、scalar ld/st 变为带 descriptor/rank/world
   操作数的 `tl.ipc.*` effectful call。这样设备函数真实使用这些参数，host/device split
   不会删掉它们。
3. 对已 `bind_ipc` 的函数，C++ pass 将显式 shared storage-sync 变为
   `tl.ipc.block_barrier`。HIP backend 输出 `__syncthreads()`；普通 GEMM 的既有
   storage-sync 策略不变。Python IR verifier 对含 cooperative put/get 或 IPC CTA barrier
   的 If/For/While 要求 block-uniform 控制条件，追踪 Let 别名；无法证明的 BufferLoad
   等条件明确拒绝，而不是仅扫描原始 threadIdx。
4. HIP codegen 调用 `src/tl_templates/dcu_hip/ipc.hpp`：每个 block copy 在其元素循环外
   绑定 peer 地址；动态 peer/rank/world 保留为运行期表达式。codegen 先物化可能
   写临时变量的表达式再输出 C++ 调用，避免动态 `%` 表达式落进半个调用文本。系统
   release/acquire、fence 和 CTA barrier 的 LLVM 检查见审查日志。
5. ctypes launch 将普通 `uint64[6]` device descriptor 和 rank/world 传给上述 kernel。
   `IPCContext` 的 C ABI 由内置 runtime 生成 descriptor；它没有任何 `~/dcu_ipc` 外部
   二进制依赖。

## Runtime 与安装

`src/runtime/dcu_ipc` 内置了用户授权的 Context/Allocation/MPI bootstrap 源码及
headers；其 11 个核心文件保持与参考接口源逐字一致。`runtime_bridge.cc` 只提供
TileLang C ABI、DeviceView 上传/缓存和生命周期桥接。bridge 使用 core 的
`DeviceGuard`、live allocation、owner-thread 与 collective gate：descriptor 仅在 context
选择的 device 上分配，错误线程/cache-hit 被拒绝；free 先完成 core collective 再释放
descriptor；finalize 不会因只有部分 PE 上传 descriptor 而本地提前返回、导致其他 PE
卡在 collective。

默认 CMake target `tilelang_dcu_ipc_runtime` 使用系统 HIP 和 MPI（未启用 HIP language
编译），作为 PIC shared library 安装。最小隔离安装检查：

首次构建必须在 MPI 环境已经设置后配置，不能假定已有 CMake cache：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
export MPI_HOME=/public/software/hpcx/2.20/ompi-slurm-pmix
export OPAL_PREFIX=$MPI_HOME PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
scripts/dev_old/build.sh  # 内含 cmake -S . -B build-old 的首次配置
cmake --install build-old --prefix "$PWD/build-old/install" --component dcu_ipc
```

调用者拥有 `MPI_Init*`/`MPI_Finalize`；每进程仅一个 active context；
`symm_alloc/free/sync/finalize` 必须所有 rank 同序集体调用。运行时验证的 allocation
payload/flag 为 FineGrained。普通 `tilelang.distributed` 导入保持 lazy，不会加载 MPI/IPC。

## 可复现验证

构建与四 PE 运行需在拥有 DCU 的节点执行（以下为 HPC-X 示例）：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
export MPI_HOME=/public/software/hpcx/2.20/ompi-slurm-pmix
export OPAL_PREFIX=$MPI_HOME PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
scripts/dev_old/build.sh  # 首次调用会 cmake -S/-B；随后也可安全重建
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python testing/python/distributed/m2_dsl_remote_copy_runner.py
M2_LITMUS_EPOCHS=32 M2_LITMUS_BATCHES=8 \
  mpirun -np 4 --oversubscribe --bind-to none --stdin none \
  python testing/python/distributed/m2_dsl_block_litmus.py
```

`m2_dsl_remote_copy_runner.py` 先以公开 `remote_view + T.copy` 写 next peer，再以公开
`remote_view + T.get_block` 从 next peer 拉回本地；两阶段均经 `bind_ipc → tilelang.compile
→ ctypes`。`m2_dsl_block_litmus.py` 是公开 DSL 生成的 4 PE block publication litmus：
256 threads、4 rank 全 12 个方向、191/223/251 不等长 typed copy、最后 wave 延迟、每
writer fence、CTA barrier、thread-zero release/acquire wait、ACK epoch 复用、消费者
旧值 volatile 预读及 barrier 后 whole-CTA weak payload read。它在每 epoch 校验随机
uint32 载荷，检查 slot tail 与各 allocation 的非零 canary guard，并进行多个 host reset
batch。自测日志包括：

- `build-old/logs/m2-block-litmus-random-4pe.log`：默认随机载荷四 PE 通过；
- `build-old/logs/m2-block-litmus-random-stress-4pe.log`：32 epoch × 8 batch 通过；
- `build-old/logs/m2-independent-stress-32x8.log`：独立 canary 压力复测四 PE 通过；
- `build-old/logs/m2-native-independent-barrier-review.log`、
  `m2-native-llvm-barrier-review.log`：公开管线、动态 peer、full-expression rank、
  divergent barrier 拒绝，以及 HIP/LLVM 系统语义检查；
- `build-old/logs/m2-installed-finalize-review.log`：partial descriptor upload 下的
  collective Busy finalize/recovery 验证。
- `build-old/logs/m2-independent-copy-get-review.log`：独立 4 PE 公开
  `remote_view + T.copy` 和反向 `T.get_block` GPU 复测通过。

最终第二轮独立复测：

- `build-old/logs/m2-formal-r2-independent.log`：16 项通过。
- `build-old/logs/m2-formal-r2-llvm.log`：真实生成 HIP 的 LLVM 中保留系统 fence、
  release/acquire 及实际 `llvm.amdgcn.s.barrier` 调用。
- `build-old/logs/m2-formal-r2-regression.log`：61 项既有回归、路径隔离及两组 GEMM
  数值检查通过，GEMM 最大绝对误差均为 0.00390625。
- `build-old/logs/m2-formal-r2-stress-32x8.log`：四 PE、全 12 方向、32 epoch × 8 batch
  通过。上述四个独立命令均 exit=0。

ctypes backend 当前没有可持久保存的 host source；这些通信 runner 显式
`tilelang.disable_cache()`，不把该 backend 的磁盘 cache 当作 M2 支持范围，也没有改动
生产 cache。

## 支持范围与明确限制

- 已验证 profile 是单机、4 PE、gfx936、FineGrained payload/flag；不是跨节点或任意
  ROCm/硬件保证。
- 支持已注册的 rank/world、remote view、put/get、fence、signal/wait eq/ge、及
  `int32/uint32/float32` scalar ld/st 的 weak/volatile/relaxed/acquire/release、gpu/sys
  组合。weak/volatile 不是同步；跨 CTA 发布必须按 litmus 的 fence/barrier/release/acquire
  协议编写。
- block copy 是 CTA cooperative；仅接受静态、连续 typed region。descriptor helper 检查
  peer/rank/world、起始元素范围和对齐；完整区间仍由 DSL 静态 region 校验与调用方
  allocation 容量共同保证。
- M2 不实现 M3 persistent 调度、融合 GEMM、广泛 DLPack/cache/export、MPI 初始化托管或
  M0 schema 修复。M0 IPC 候选为 deferred，非废弃；本实现不依赖其未通过的契约。
- CMake component 安装已经验证。整包 wheel 尚未声称可用：其 Python package 固定的
  TVM checkout 与当前 native build 所用 TVM revision 不同，本轮没有混改用户的第三方
  checkout。
