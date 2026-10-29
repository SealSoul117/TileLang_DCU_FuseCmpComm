# M0 IPC v2 契约

**当前状态：候选，第二轮 review 未通过，已暂停。** 下文描述设计与已有实现意图，
不是所有约束已被正确实现的保证。尤其 JSON 类型/重复 key、原生 binding 宽度及权限
校验仍有缺口；完整结果见 [主代理审查报告](distributed_fusion_m0_ipc_review.md)。

日期：2026-10-01。M0 仅提供纯 Python CPU 元数据，不执行 MPI、HIP 分配、原生指针、
atomic、GPU lowering 或硬件验证。它新增 ipc_spec 和 ipc_schedule，不改 legacy ABI1。

IPCFusionSpec 只接受四 PE、batch=1、整 256x256 tile、block_K=16 的 AR、RS、AG profile。
m/n/k 是每 PE GEMM 尺寸；AR/RS 的 global shape 是 [m,n,4k]，AG 是 [m,4n,k]。
CPU 能描述其他对齐尺寸不代表实机支持。FP16 输入/输出、FP32 GEMM；AR/RS 有 FP32
reduce 和 FP16 partial，AG 没有 partial/reduce。数学是 A@B，不转置，输出 row-major。

manifest 固化 launch、precision、rounding、memory requirements、DeviceView 字段语义及
output-ready、no-more-remote-readers、flags-reusable 三个条件。派生 layout/offset/
precision 由 spec 重算，JSON 篡改会失败。M0 不实现 AG counter RMW、CTA 发布或队列互斥。
flags 只能在全部 PE 本轮完成后的 host collective 边界后 reset；kernel return 不能单独许可
复用。

CPU 测试覆盖 12 份 reference 配置、独立映射公式、逆映射、AG fail-closed、manifest 篡改
与 binding 检查。M1/M2 才实现 counter、Block publication 和 IPC GPU lowering。
