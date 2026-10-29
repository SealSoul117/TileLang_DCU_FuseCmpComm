# GEMM + ReduceScatter 代码生成原型：独立审查

日期：2026-10-01。用户要求用现有 TileLang 接口尝试生成融合 HIP。本轮不是 M3/M4 验收，
不修改或扩展生产编译器、通信接口及 IPC runtime。实现代理为 gpt-5.6-terra / high，
没有继续分派；主代理负责设计、独立检查与最终审查。没有 Git commit。

## 结论与范围

推荐 `gemm_reduce_scatter_tile_grid.py`：直接二维 grid，每 CTA 一个 tile；GEMM 和 RS 在
同一个 `__global__` 中。四 PE 的局部 A/B 相乘得到 FP16 partial，各 tile 的连续行分片
owner 等待四个源发布后，读取四份 partial，FP32 求和并转 FP16 写入本地 `[M/4,N]`。
这是部分形状已验证的演示，不是通用形状或高性能 persistent 实现。

实际环境：Slurm 9511、dcu0、四张 gfx936，每张 80 CU；old 隔离环境及原有 DTK 26.04。
只承诺以下已测用例，不把“最多 8 CTA/PE”解释为通用正确性或调度保证。

| 形状 M/N/K（K 为每 PE 的本地 K） | 输入/标准 | 主代理结果 |
|---|---|---|
| 1024/256/256 | 随机整数除以 16，FP16；partial 与 RS 逐位比较 | 四 PE 全通过，最大绝对误差 0 |
| 2048/256/512 | 同上 | 四 PE 全通过，最大绝对误差 0 |
| 2048/256/512 | 连续随机 FP16；rtol=atol=1e-3 | 四 PE 全通过；partial 最大误差 0.0009765625，RS 0.001953125 |
| 1024/512/256 | 随机整数除以 16，逐位比较 | partial 全通过，但 RS 全失败；不能当可用形状 |

独立运行日志均在仓库根目录的 `build-old/logs/`：

- `gemm-rs-prototype-flat-default-run-review.log`
- `gemm-rs-prototype-flat-m2048-dyadic-review.log`
- `gemm-rs-prototype-flat-m2048-run-review.log`
- `gemm-rs-prototype-flat-n512-run-review.log`

## 独立代码检查

- 推荐版经过正常 `bind_ipc → tilelang.lower/compile`，没有拼接或手改 HIP。
- source-only 审计将 subprocess.run/check_output/check_call 设为报错，仍正常导出；
  只有一个融合设备函数、五个原始参数和四个追加参数。检查日志：
  `build-old/logs/gemm-rs-prototype-flat-source-audit-review.log`。
- 实际 DTK clang 编译生成 LLVM；确认 MMAC、系统 fence、release store、acquire load、
  真正的 CTA barrier 均保留。产物 `build-old/gemm-rs-prototype-review/flat-default.ll`，
  编译日志 `build-old/logs/gemm-rs-prototype-flat-llvm-review.log`。工具链/模板 warnings
  原样保留，编译退出码 0，不宣称无 warning。
- partial 每 tile 独立、不在 kernel 内覆盖或 reset；所有 PE 的 kernel 完成且主机
  collective 结束后再释放，所以本示例不需要 in-kernel ACK。不能推广为通用复用协议。
- 既有 M2 独立检查基线 16 passed；本轮未改 M2 生产源码。

## 保留的问题

1. 原单 CTA 外层 serial tile 循环版本能生成和编译，但本地 partial 含 NaN/Inf。
   主代理独立复现退出码 1：`build-old/logs/gemm-rs-prototype-independent-pipelined-run.log`。
   `IsDCUMainLoop` 递归识别外层循环、流水线阶段跟踪和缺失 prologue wait 是候选原因，
   尚未完成因果定位，未修改这些 pass。
2. 无流水线/每步等待变体也未通过 partial；extent-one 外层循环技巧在 Simplify 阶段失败。
   README 保留具体日志，不以删除或放宽检查掩盖。
3. N512 的 partial 逐位正确，但 RS 最大误差 16.25--18.91。根因未定位。
   已检查 LLVM，输出索引仍包含 `threadIdx & 255` 对应的 and 指令；不能声称该 mask
   已被 LLVM 优化删除。512-thread launch 与既有 256 编译配置只是应审计的风险之一。
4. 所有执行结果限于本机已测组合；不宣称跨节点、任意 shape/PE 数、性能收益或
   persistent 计算/通信角色调度已经完成。M0 仍暂缓，未推进后续模块。

## 正式审查轮次

第一轮要求修正文档：不能把顺序版的单 CTA 活性解释套到 flat grid；补全实际 Slurm
资源命令、明确两个有效形状、将成功产物与失败变体分开。第二轮已修正，主代理审查
通过，范围为“可查看并复现的有限形状原型及如实保留的失败证据”，不是通用融合算子。

原有 `.gitignore` 的 `dist*/` 会忽略 `examples/distributed`，本轮仅放行这个原型源码目录；
`generated/` 仍忽略但文件保留在工作区。没有执行 git add/commit。
