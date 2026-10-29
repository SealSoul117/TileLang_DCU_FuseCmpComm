设备端 epoch AllGather + GEMM / GEMM + ReduceScatter
==================================================

保留 allgather_gemm_epoch.py、gemm_reduce_scatter_epoch.py 的算子定义。
默认 BF16、8 PE，AG N=768/K=2048，RS N=2048/K=512。
构建 TileLang 后，在仓库根目录分别运行（输出目录必须尚不存在）：
  python -m examples.distributed.generate_epoch --kind ag --m 2048 4096 --output build/epoch-ag
  python -m examples.distributed.generate_epoch --kind rs --m 2048 4096 --output build/epoch-rs
生成入口只保存 HIP 源码，沿用原编译参数、预取同步及向量化检查。
生成步骤会编译算子但不启动多卡计算，不提供下游应用的导出包或运行适配。

RS 数据流
  普通 Torch A -> GEMM -> IPC partial -> owner pull/reduce -> 普通 Torch output。
  最后的 block_done / peer_done 握手保证所有远端读取完成后才复用 partial。
  每次调用不用 pack_a、crop、清零 flags，也不用 runtime 前后屏障。

AG 数据流
  普通 Torch A_shard -> 通信 blocks 的向量 push -> 每张卡的 IPC gathered A。
  每个 source 写不重叠的行分片，每个块负责固定 65536 元素 chunk。
  每线程一次向量搬运 8 个 BF16（16 bytes），完成后发布目标端 ready(epoch)。
  计算 blocks 等待本 tile 所需 chunks，然后执行原 GEMM，直接写 Torch output。
  A 和 A_comm 两个参数必须传同一块本地 IPC 分配：
    A 保留普通 GEMM 本地 buffer 的编译布局；
    A_comm 仅用于 IPC remote_view。不要把 A 本身直接绑定为 remote_view。
  首次试验绑定 A 本身后改变了 GEMM A 的异步加载路径并数值失败，
  已保留失败日志；双视图避免修改 GEMM 后端。

AG 复用协议
  在 flag_words 后追加 P 个 arrival flags。
  第一个通信 block 发布本 PE 的 arrival(epoch) 到所有 PE。
  通信 blocks 等待所有 PE arrival >= epoch 后才能写 gathered A。
  这表示每个 PE 在其有序 stream 上已完成上一次整个 kernel，
  因而上次所有 GEMM readers 都已结束。仅使用 comm-done 不足以保证这一点。
  然后以 chunk ready、block_done、peer_done 完成当前调用的通信协议。

共同约束
  flags 只在集体初始化时清零。各 PE 的调用顺序、shape、epoch 必须一致。
  每个 workspace 固定 device/stream，不支持并发调用或 CUDA/HIP graph replay。
  uint32 epoch 从 1 递增，不允许回绕；耗尽需所有 PE 停止后重建。
  所有 persistent blocks 必须同时可驻留，runtime 检查 occupancy。
  输出是新建 Torch tensor，非复用 IPC 输出槽；保留旧输出不会被后续调用覆盖。
  初始化/退出仍有集体通信。稳态每算子一次 fused launch，首次权重打包除外。
  “零复制”指移除接入层 pack/crop，不是取消 AG 的必要数据搬运。

编译结构约束与排错记录
  不要在 compute/comm 分支之前另加独立的入口握手语句序列。
  当前 InsertAsyncGldFence 只在 seq_nesting_ == 0 时补充 AB 预取对同步；
  独立入口语句会使计算嵌套在下一层，导致预取部分少四处 wave_barrier。
  本次直接将入口握手放入通信分支，保留原计算 IR 结构。
  修正后 M2048 的计算 HIP 与旧版逐行对比，只有 wait_eq(1) -> wait_ge(epoch)
  和 ready 指针变量名不同；未修改 GEMM 后端。
  generate_epoch.py 检查四处预取对同步及 128-bit push，缺失则拒绝导出。
  本地分片单独复制、每 wave acquire、Default payload 的试验未解决上述问题；
  最终保持 FineGrained payload，所有分片统一走 push。诊断库不用于正式构建。

