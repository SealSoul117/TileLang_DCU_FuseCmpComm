# M0 验收报告

日期：2026-09-20。依据：`distributed_fusion_dcu_plan.md` 第 3 节及相关下游契约要求。

当前状态：**M0 已通过用户授权的第三轮补修验收。**

## 1. 分工、范围与流程

- 实现子代理：`gpt-5.6-terra`，推理强度 `high`，不得继续分派。
- 主代理负责设计、独立检查和最终验收；不自动执行 Git commit。
- 一次仅实现一个模块。用户已授权 M0 通过并报告后进入 M1，M1 报告后暂停后续模块。
- 前两轮验收未全部通过时已经暂停并汇报；本次第三轮补修得到用户明确批准。
- 保留原有子模块修改、虚拟环境及示例目录，不将其算作本模块改动。

M0 交付 Python 契约、序列化和纯 Python 调度映射。它不实现 HIP 代码生成、设备通信或运行时。

## 2. 交付文件

- `tilelang/distributed/spec.py`：`FusionSpec`、对称 buffer 描述、ABI、运行时绑定及校验。
- `tilelang/distributed/schedule.py`：work/tile/坐标、owner/source slot、batch ready、输出及消费者映射。
- `tilelang/distributed/__init__.py`：接口导出。
- `testing/python/distributed/test_m0_contracts.py`：不依赖 GPU 的契约测试。
- `docs/compiler_internals/distributed_fusion_m0.md`：模块接口说明。
- `.gitignore`：增加精确例外，避免原有 `dist*/` 规则把 `distributed/` 源码和测试忽略掉。

## 3. 第一轮 review：退回修改

初版 9 项 CPU 单测通过，但主代理独立复现以下问题：

1. RS 只交错消费者顺序，计算生产者仍按行顺序访问 owner。M=512、N=384、P=4 时，前四个
   work 的 owner 是 `[0,0,0,1]`，不符合计划要求的 `[0,1,2,3]`。
2. ABI 仅比较用户给出的 workspace 和 binding 是否一致，未与 schedule 的实际需求比较。
   将 ready 的两份描述同时缩到一个 uint32 后，4 字节 ready buffer 会被接受。
3. staging、output、ready 使用相同 allocation ID 且 offset 都为零仍可通过，存在重叠风险。
4. 缺少必要通信 capability、只读 staging 也能通过；对齐含义与首版 tile 支持范围不够明确。
5. 非法配置诊断不统一，存在忽略未知字段、丢弃伪造派生容量、把 bool 当整数等问题。
6. 测试需要补充实际通知与等待覆盖、空闲 block、尾批次以及上述反例。

主代理将这些问题退回原子代理，没有进入下一模块。

## 4. 第二轮 review：主要问题已修复，仍未验收通过

独立确认的修复：

- RS 计算顺序已改为 `owner_interleaved`，上述例子的前四个 owner 为 `[0,1,2,3]`。
- RS output 采用 owner-local tile-major 布局，以最大 owned tile 数作对称容量，分别声明每 PE
  有效前缀；上述例子的 output 为 `[3,128,128]`，不是完整的 `[12,128,128]`。
- workspace 布局从 spec/schedule 推导，过小 ready、workspace allocation 别名、缺失必要
  capability、只读 staging 均被拒绝。
- 区分 allocation-base alignment 与 offset 后的有效 view alignment。
- 首版 lowering 明确要求 128×128 tile，并拒绝尚未实现的非对齐矩阵边界。

主代理复跑命令：

```bash
python3 -m unittest testing/python/distributed/test_m0_contracts.py -v
python3 -m py_compile tilelang/distributed/__init__.py \
  tilelang/distributed/spec.py tilelang/distributed/schedule.py \
  testing/python/distributed/test_m0_contracts.py
git diff --check
```

结果：**14 项单测通过，语法和已跟踪文件的空白检查通过**；新增文件另行直接阅读。

另用独立 Python 参考公式验证了 **1,280 组配置**，全部通过：

- tile 网格 `(行,列)`：`(1,1)、(1,3)、(2,2)、(3,2)、(5,3)`。
- PE 数：`1、2、4、7`；计算 block：`1、2、7、20`；通信 block：`1、3`。
- batch：`1、2、4、8`；两种算法：AR 与 RS。
- 覆盖 work/tile 双射、坐标、owner、source slot、真实通知唯一性、每次等待的 payload
  覆盖、空闲 lane、尾批次及每 PE 输出索引。此枚举是独立审核命令，不是额外的仓库单测方法。

第二轮仍有两项缺口：

1. 只有 `tile_to_coords`，缺少 `(bx,by) -> tile_id` 的逆映射接口和校验。
2. 部分非法类型仍泄漏原始 `TypeError`：`collective_kind=[]`、`permissions=[[]]`，以及
   `ready_key(None,0,0)`，没有统一为 `DistributedConfigError`。

这些是契约完整性与错误诊断问题，不是上述合法输入映射检查发现了数值错误。按用户的两轮限制，
主代理当时停止实施并报告，M1 未启动。

## 5. 用户授权的第三轮补修

本轮仅补齐两项缺口及其回归测试，并将模块说明和验收报告改为中文。

补修结果：

- 新增 `SchedulePlan.coords_to_tile(bx, by)`，接受非 bool 整数、校验边界，返回
  `by * tiles_x + bx`，与 `tile_to_coords` 互为逆映射。
- `FusionSpec` 在字符串枚举判断前验证字段类型，permissions 在集合查找前检查元素类型，
  `ready_key` 在取模前检查 batch start。未用宽泛异常捕获掩盖内部错误。
- 模块接口说明和本验收报告均已改为中文。

主代理独立验收：

1. 复跑同一 unittest 命令：**15 项通过**。
2. 在 16 种 tile 网格、各自对齐和短边两种矩阵尺寸上检查 **308 对坐标双射**，全部通过。
3. 对语义枚举、permissions、ready key 和坐标接口执行 **637 项非法输入检查**，全部按约定
   抛出 `DistributedConfigError`，包括上一轮的三个 TypeError 反例。
4. `py_compile`、`git diff --check` 通过；直接检查新增测试及中文说明。
5. 总计划文件 SHA-256 未变；没有 Git commit，没有改动已有 HIP kernel。

**最终结论：本轮两项剩余缺口已补齐，M0 在 CPU 契约层通过验收。**
后续可按用户授权进入 M1；不能将本结论解释为设备通信或融合代码生成已完成。

本轮最终审核快照 SHA-256：

- `spec.py`：`82baade9256521bc1854d143745368df013ec963e0710a064f2703fbea931199`。
- `schedule.py`：`924d738436be9d8807e569a0c5c4780ed65d14cc59bae1fbf8112444fe5b2a3d`。
- `test_m0_contracts.py`：`8ff8ed458691baf548dfc55482e0206372accd954b91557845fb8033fab57392`。

## 6. 参考版本与证据边界

- TileLang_DCU HEAD：`b2bc99b0a3ab03cb5ffce731db87048931e011a6`。
- 总计划 SHA-256：`ec9c08cebb08237099c9cbc1725517f049fad22fd8702a6dbb7fedb89a67ce07`。
- 手写 AR v1 SHA-256：`89aecc768aeb96f85dce0d579342385bea0300d6e09ca81a19d6c4636e9c0624`。
- 手写 RS SHA-256：`cbf4e3610d59cc064fc48e428e5697391b2faebf3101354ad81299e1505c085a`。

第二轮审核快照 SHA-256：

- `spec.py`：`3b53784eb00ac6c453a2550dc00cab8ec84687ed7c47347b9ba9610101fc21c7`。
- `schedule.py`：`3d0249ca4c7ab74aa8d043ccc64be178d1ed86e536608799d909b149d9bc0aa9`。
- `test_m0_contracts.py`：`5ae05bd8058a44ab7ff2a04b127189520dff1579189e6f96b0435f0d0fcb2ca4`。

M0 的 CPU 测试不证明 HIP 生成正确、ROC_SHMEM 地址可用、跨设备内存顺序成立、persistent kernel
不会死锁、占用率满足要求，也不提供 DCU 数值或性能结论。这些仍属于后续模块验收。
