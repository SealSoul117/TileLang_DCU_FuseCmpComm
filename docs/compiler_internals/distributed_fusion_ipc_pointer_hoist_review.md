# IPC 通信指针外提：实现与验收

日期：2026-10-06。工作目录仅为 `TileLang_DCU_old`；第二轮正式 review 通过。
实现由指定的 gpt-5.6-terra/high 子代理完成，主代理负责设计、构建、独立审查及设备验证。
第一轮因新增测试构造错误和覆盖不足退回，第二轮补齐通过。未自动 git commit，不推进其他优化或 M3。

## 实际功能与修改范围

原 scheduled GEMM+RS 的通信元素循环每次迭代都重新调用四次
`tl::ipc::peer<std::byte>(ipc_view_partial, peer, rank, world, 0)`。
这重复读取和校验 DeviceView/peer table，而 allocation 在该 launch 内保持存活且描述符不变。
即使不保留任何测试，这条真实生成路径也需要后端转换才能自动消除重复解析。

本轮生产修改只有：

- `src/transform/hoist_dcu_ipc_pointers.cc`：新增专用 `HoistDCUIPCPointers` PrimFunc pass。
- `tilelang/transform/__init__.py`：注册对应 Python pass 入口。
- `tilelang/engine/phase.py`：在 `StorageRewrite` 后、`UnrollLoop`/`SplitHostDevice` 前自动执行。

不修改 Python 算子、HIP codegen、IPC helper、op 的 opaque 属性、GEMM 地址生成、lane/warp、
通信汇编索引、信号协议或算法。`signal_set` 内部的地址解析也不属于本轮优化。

当前真实生成的结构可简写为：

```cpp
wait_tile(); fence_sys(); __syncthreads();  // 原有协议，顺序不变
int comm_tx = tl::ipc::opaque_thread_id(threadIdx.x);
if (comm_tx < 65536) {                    // 保持零次循环不执行地址校验
    void* p0 = tl::ipc::peer<std::byte>(view, 0, rank, world, 0);
    void* p1 = tl::ipc::peer<std::byte>(view, 1, rank, world, 0);
    void* p2 = tl::ipc::peer<std::byte>(view, 2, rank, world, 0);
    void* p3 = tl::ipc::peer<std::byte>(view, 3, rank, world, 0);
    for (/* 原元素循环 */) {
        // 原有边界判断、远端 load、加法、output store，均留在此处
    }
}
```

这是每 tile、每线程缓存四个解析后的基址，不是把整个通信分支的地址统一提前，也不是提前读取 payload。
当前 256×256 tile、512 线程对应每线程 128 次元素迭代，地址 helper 调用由 4×128 次减少至 4 次。
helper 的 descriptor/peer/基址检查保持不变；不能将它表述成删除了每个元素的边界检查。

## 安全边界

只接受同时具有 binding/ABI-ready 属性、且 descriptor 的参数 Var 与 buffer_map 身份吻合的函数。
descriptor 写入、读取、别名或非 ABI 白名单参数逃逸使整个函数保持原状。
稳定性依据是内置 runtime 的 immutable peer table 发布及 DeviceView 一次上传、缓存至 free 的生命周期；
调用方仍必须在所有 GPU/远端用户完成前保持 allocation 存活，不能修改该元数据或将 payload 与其别名绑定。

仅外提 serial 元素循环直接前缀中的不变地址 Let。peer 允许常量、纯标量参数、纯外层 Let；
不允许当前循环变量、循环内局部变量、BufferLoad 或 opaque call 依赖。
不跨条件、线程、分配、嵌套循环、assert/block、wait/signal/fence/barrier 或未知调用。
payload-only 条件语句可保留，但不会从条件内抽出指针绑定。
不确定正数的 extent 用 `extent > 0` 保护；min/extent 本身含内存读取或调用时不做该转换。
保留原 Let Var 和 Buffer.data 身份，远端数据访存及其同步顺序不变。

## 验证结果

- 新增 `testing/python/transform/test_hoist_dcu_ipc_pointers.py`：8 项通过，包含多种正反例及真实 scheduled codegen。
- M2 独立接口与 persistent/DCU 后端：30 项通过；M0/M1、语言层及既有回归：61 项通过。
- 主代理额外检查实际 pass 前后 GEMM compute branch `same_as`；零 extent 的 guard 确为 false、
  动态 guard 确为 `extent > 0`，含 assertion 的循环保持对象身份；全部通过。
- 5 组原生 GEMM 加 2 组未标记 persistent：HIP 与优化前逐字一致。
- 四 PE 真正重新 JIT 的 matched GEMM+RS：2048²×512、4096²×1024 dyadic 输入通过；
  2048 continuous 输入、4096/24 comm blocks 通过；最终 guard 版二进制再次运行 4096 通过。
- 四 PE M2 Block publication litmus（3 epochs × 2 batches、12 个方向）、remote copy/get 全部通过。
- 原生 GEMM 512×768×256、1024×512×512 continuous GPU smoke 通过。

## 性能

同一个 dcu0 四卡 Slurm job 9809，80 CTAs＝48 compute＋32 communication，512 threads/CTA。
每组 5 次 warmup、30 次测量；先对每轮四 PE 的 HIP event 取最大值，再取中位数。
只计 kernel，不计编译、分配、数据初始化、flag reset 或 host 跨 PE 同步。
两组均检查计时前后的完整 dyadic 输出。

| M×N×K | 优化前 | 后端自动外提后 | 耗时下降 |
|---|---:|---:|---:|
| 2048×2048×512 | 0.628159 ms | 0.350240 ms | 44.24% |
| 4096×4096×1024 | 1.273439 ms | 0.642880 ms | 49.52% |

既有共享内存分配器存在 A/B LDS 排列非确定性，因此另做了固定 HIP 的单变量对照：
以本次自动生成的 4096 HIP 为基础，仅把四个指针声明放回循环内，其他内容不变
（文件尾空行不计）；两边均通过四卡完整输出检查。
循环内解析为 **1.278638 ms**，外提为 **0.635200 ms**，耗时下降 **50.32%**。
这组手工调整仅用于性能归因，保存在 build artifact 中，不属于编译器实现或 Python 算子。

这些是本次同条件样本，不承诺其他尺寸/拓扑的固定加速比。
既有 4096 continuous RS 在 rtol=atol=1e-3 下的少量边界元素问题，本轮未修复且未重测；
不将 dyadic 通过扩大为所有随机输入均通过。

## 产物与复现

证据根目录：`build-old/ipc-pointer-hoist-20261006/`。
`before/` 保留旧动态库、7 组原生 HIP、逐 PE JSON 和实际中间 TIR；
`after/` 保留真实生成 HIP、GPU/pytest 日志；`control/` 保留固定 HIP 对照；
`summary.json` 和 `summarize.py` 提供聚合数据及校验；`final-paths.json` 检查 old 路径隔离。

最终构建库为 `build-old/lib/libtilelang_module.so`，SHA256：
`cad80e2271efc861ef4a082dca8f678f924ed1e488c7311c304c5ca310a9f24f`。

在计算节点、old 环境下可查看真实自动生成的 HIP：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
source scripts/dev_old/env.sh
python examples/distributed/gemm_rs_prototype/gemm_reduce_scatter_scheduled.py \
  --m 4096 --n 4096 --k 1024 \
  --output build-old/ipc-pointer-hoist-inspect
python -m pytest -q testing/python/transform/test_hoist_dcu_ipc_pointers.py
```

本轮已重新 build。新进程、新编译会自动执行 pass；性能 runner 已关闭 kernel cache，
没有删除用户旧缓存，也不保证此前缓存的旧 kernel 会自动重新编译。
