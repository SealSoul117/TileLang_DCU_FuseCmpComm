# M0 IPC v2：主代理审查报告

日期：2026-10-01。**状态：第二轮 review 未通过，按用户规则暂停；M0 IPC 契约尚未补齐。**

## 分工和完成边界

实现子代理为 `gpt-5.6-terra`、推理强度 high，没有继续分派。
主代理给出接口、参考映射、生命周期与验证设计，独立阅读代码并编写复现测试。
没有 Git commit，没有进入 M1 counter 扩展、M2 或后续模块。

新候选保存在 `tilelang/distributed/ipc_spec.py`、`ipc_schedule.py`，由 `__init__.py`
导出。AR/RS 的 owner/partial/ready/output 映射和 AG chunk/queue 映射已具备；
只包含 CPU 元数据，不代表实现了跨卡 kernel、原生 runtime 或原子同步。
legacy `spec.py`、`schedule.py`、`test_m0_contracts.py` 与升级前备份 SHA256 一致。

## 第一次 review

主代理发现超出 HIP `int K` 范围的 `k=1<<40` 被接受；另外要求补齐：
manifest 原生类型及派生字段、严格 JSON 类型/重复 key 拒绝、runtime binding 上下文与
原生宽度检查、peer 读写权限、完成协议、负例测试及可维护的代码格式。
候选退回修订，未通过第一次 review。

## 第二次 review

已修正大尺寸 K/部分派生索引上限，以及完成协议的文字方向；普通映射测试通过。
但是前次指出的多项 ABI 校验仍未实现，独立复现包括：

| 问题 | 复现 | 当前错误行为 |
| --- | --- | --- |
| JSON 类型可混淆 | `full_resident: true` 改成 `1`，或 `threads: 512` 改成 `512.0` | Python 字典相等比较仍接受 |
| 重复 JSON key | 同一文档先出现 `abi_version:99`，再出现 `abi_version:2` | 后者覆盖前者，未拒绝 |
| native 字段溢出 | flags 的 allocation_id 或 offset/capacity 设为 `1<<80` | 正数及大小比较通过，无法装入原生 uint64/size_t |
| rank 期望类型错误 | rank=0 时传 `expected_rank=False` | `False==0` 绕过校验 |

另外，peer_access 仍只是布尔值，不能描述 remote read/write 权限；manifest 还缺少完整
DeviceView 字段类型、参数角色映射及结构化发布/重置协议。identity 仅由 spec/version
生成，尚未包含完整 manifest。runtime 期望上下文参数仍可省略。实现方测试仅有三个方法，
尚不足以覆盖要求的 binding/类型/权限负例。

证据：`build-old/logs/m0-ipc-review-round2.log`。合跑 28 个测试方法
（legacy 15 + 实现方 IPC 3 + 主代理 IPC 10），四个方法失败，共六个失败子例；
不是 28 项全部通过。失败断言保留，未改为 skip/xfail，也没有放宽预期。

CPU-only 复现（无需 DTK/GPU）：

```bash
cd /public/home/baoboyuan/TileLang_DCU_old
/public/software/python/3.10.20/bin/python3 -m unittest \
  testing/python/distributed/test_m0_contracts.py \
  testing/python/distributed/test_m0_ipc_contracts.py \
  testing/python/distributed/test_m0_ipc_review.py
```

依据“review 两轮仍未通过先汇报”，停止第三轮实现。保留候选和失败证据，等待用户决定
是否授权继续补修；不能把当前 ABI2 当作已冻结的 M2 输入。
