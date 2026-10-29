设备端 epoch GEMM + AllReduce
============================

保留 gemm_allreduce_epoch.py 的 TileLang 算子定义及设备端同步协议。
生成入口位于 examples/distributed/generate_epoch.py，只保存生成的 HIP 源码。
需要已构建的本仓库 TileLang、原仓库声明的 TVM/FFI 和 gfx936 HIP 工具链。

在仓库根目录运行：
  python -m examples.distributed.generate_epoch --kind ar --pe 8 --m 2048 4096 --output build/epoch-ar
4 PE 使用 --pe 4；输出目录必须尚不存在。
生成步骤会编译算子，但不会启动多卡计算；成功生成不等于数值验证通过。
默认 BF16、N=2048、K=512，沿用原来的编译参数及代码检查。
本目录不再打包下游应用的桥接库、manifest、运行适配器或性能测试结果。

调用契约：必须遵守
------------------
- 每个工作区的 ready 在初始化、全组尚未开始计算时清零一次，以后不清零。
- 所有 PE 的对应调用使用同一个非零 epoch，严格递增，范围 1..0xffffffff。
- 不能复用旧 epoch，不能回绕到 0；耗尽时在全组静止后重建工作区。
- 首版由主机本地计数、作为 kernel 参数传递，不需要每次广播计数。
- C ABI 必须按 uint32 传递 epoch；测试覆盖 0x80000000 和 0xffffffff。
- 同一工作区必须按 stream 依赖顺序使用，不能并发 forward。切换 stream 必须显式
  wait_event，且事件记录在上一调用输出读取/crop 之后；这不是跨 PE 屏障。
- 输入 A/B 必须在本地 stream 上准备完成；IPC partial/out/ready 使用已验证的
  fine-grained 内存模式。初始化和退出时的 IPC 生命周期协调仍不可省。
- 不支持使用固定 epoch 的 graph replay；图重放需另行设计设备端序号管理。
- 本 kernel 的 out 是可复用 IPC 工作区。调用者须在同一 stream 上消费或复制旧输出，
  然后才能启动下一次使用该工作区的调用。不能把这个裸工作区当作永久独立结果。

为什么不需要入口/出口 runtime 屏障
--------------------------------
计算 Block 写本地 partial，发布 tile_ready[tile,rank]=epoch；owner 等齐所有 PE
对应 tile 的本轮信号后才 Pull/归约/向各 PE Push 输出。
通信 Block 完成自身全部 payload 访问后发布 block_done=epoch；协调线程收齐后向
每个 PE 发布 peer_done=epoch，并等待本地收齐全部 PE 的完成信号。

每个信号槽只有一个 writer；wait_ge 避免错过已推进的完成状态，但它本身不构成
缓冲区复用证明。复用安全还依赖以下三条：
1. 本卡 kernel 结束说明所有 owner 已完成上一轮读取，故 partial 可以被本卡复用。
2. 向 PE q 写下一轮输出前，owner 必须收到 q 的下一轮 tile_ready；这证明 q 已经
   在本地 stream 上完成上一轮输出消费/crop，进入下一轮 GEMM。
3. 本轮 flag 不清零，因此不会擦除先到的下一轮信号。done 信号单调推进，不会回退。
不要删除完成握手，也不要把这些条件外推到无序多 stream、并发 forward 或别的算法。

