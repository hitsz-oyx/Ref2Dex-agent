| 结果解读表（实测） | 含义 | 决定 |
|---|---|---|
| **A 未就绪**：训练 unknown 批次 0；legacy 模型入口 mask=True 为 **256/256**；部署 missing mask=True 为 **0/256**；三组 goal 张量均非 bit-exact | 模型有 mask 消费能力，当前训练数据入口没有 unknown 编码 / 路由 | 需要改代码与明确配方变更；本轮只列清单，不启用 |
| B 全量规则的已记录参数 / ≤reset 绑定未达门槛；Shadow native 参数仅 **3/1,064（0.282%）母序列、30/6,000（0.5%）窗口** | 有小范围原生正例，不能泛化成全量规则；完整 waypoint 可认证仍为 0 | 不将 B 升格成 ≥80% 的合法候选；全量补造规则标 **NOT_DEPLOY_LEGAL** |
| 固定零位移与 missing 的 delta 字节相等，但 mask / goal points 不相等 | “常量为零”不等于 unknown | 不能以 zero delta 冒充训推统一 |
| C 原信封：**1,064 母 rollout、2.423h 对象流、134,000,640–536,002,560 B** | 保留有效计划条件需要有来源/执行绑定的新记录方案 | 只引用 brief32，不重算、不采集；等机主裁决 |
| 当前 A/B 都不能直接开训，A 可作代码变更候选、C 可作新录候选 | 不能宣称三条路线永远不可行，也不能宣布已解锁 | **0 updates，不改生产 / 配方 / status** |


## 证据：范围和方法

`P-20261023-goal-unlock-readiness` / `brief36-goal-unlock-20261023-r1`。仅作 CPU 入口取证与对既有产物的计数；不重复 brief31/32/35 inventory、provenance 或 r9 replay，不新读全量原始信号。写盘预计 ≤16 MiB，上限 1 GiB，写前实测空间。训练 0 updates，无 GPU，不改生产、config 或 goal_missing_policy。

取证分开记录：训练模型本身是否接受 false mask、真实训练数据路径是否生成该编码、部署是否生成同编码。只有三项都成立才能称 A 就绪。数值零但 mask=True 不算 unknown。B 的规则族包括常量/hold、固定世界目标或位移、接近方向、几何接触目标及已有事前计划；规则选择本身也不能在缺记录时冒充原任务意图。


### A：实际模型入口张量

证据文件：[receipt.json](brief36-goal-unlock-20261023-r1/receipt.json)、[完整 dtype/shape/数值 JSON](brief36-goal-unlock-20261023-r1/boundary-tensors.json)、[保留原始位模式的 NPZ](brief36-goal-unlock-20261023-r1/boundary-tensors.npz)。读取 NPZ 不需要 pickle。

训练侧使用真实 V6Windows、从生产 `train_pretrain.py` 导入的 Steps、DataLoader / collate、同样的 `.to(device)` 后调用真实 SharedWAMV6 类。保持实际 B1 full config 的参数维度、normalizer 和 precision 配置，仅使用不执行的 geometry 构造 seam，避免为输入取证加载 / 运行 CUDA 点云 backbone。SharedWAMV6.forward_pre_hook 捕获真正交给模型的 batch 后抛专用停止异常，**在任何 network forward、geometry、loss、backward、optimizer 构造 / step 之前停止**。

该探针复用既有 event-free reader 分支，**没有给训练 sample 赋值 / 清零任何 goal 字段**。训练入口的 35 个 collate 字段与 direct-reader + 同一 collate 逐项相等。它是可复现的只读训练数据路径入口探针，不冒充已经运行完整 train_pretrain.main 或 DDP；当前 curated main 本来就会在创建 optimizer 前拒绝，没有可供截获的 unknown goal batch。CPU 工程 batch=8，不产生 optimizer update，也不代表生产吞吐。

部署侧直接运行 OnlineV6Runtime.predict → online → collate → `_trajectory` → 同一 SharedWAMV6 类的 forward_pre_hook；调用 `goal=None`，再作一个“目标就是当前 pose”的有效零位移负控制。8 条合成 live history 仅用于部署 API 输入形状；每条 geometry query 与捕获训练样本完全相同，故按相同 query / dtype 对照。没有新 rollout、模型预测或 τ 指标，也没有把这个负控制实现成 B 规则。

| 真实模型入口 | goal_object_delta | goal_object_points | object_goal_valid |
|---|---|---|---|
| legacy 训练数据路径 | float32 `[8,32,3]`，768 元素非零，范围 **[-0.4459224,0.39823306] m** | float32 `[8,32,3]`，768 非零 | bool `[8,32]`，**256 True / 0 False** |
| 部署 missing：goal=None | float32 `[8,32,3]`，**全 +0** | float32 `[8,32,3]`，**全 +0** | bool `[8,32]`，**0 True / 256 False** |
| 部署有效目标=当前 pose | float32 `[8,32,3]`，**全 +0** | float32 `[8,32,3]`，等于查询点，768 非零 | bool `[8,32]`，**256 True / 0 False** |
| 当前 curated 训练路径 | **没有模型入口张量** | **没有模型入口张量** | reader 提前拒绝；模型调用 **0** |

训练 8 条实际 sample_id 及第一 query 的 delta（m；完整精度在 NPZ）：

| sample_id `[sequence,slot,tick]` | delta[0] |
|---|---|
| [676,0,92] | [-0.001990166, 0.008611336, 0.001238011] |
| [434,0,226] | [-0.070538878, -0.240530163, 0.106975108] |
| [40,0,185] | [0.002708919, -0.000072067, 0.009447142] |
| [817,0,117] | [-0.049493369, 0.035597850, -0.021445051] |
| [577,0,53] | [0.124597184, 0.077874996, 0.298689663] |
| [923,0,382] | [-0.012983723, 0.064498216, 0.203539565] |
| [27,0,547] | [0.000038874, 0.000193607, -0.000108017] |
| [190,0,147] | [0.000063645, 0.000078705, 0.000200335] |

这些是生产 Steps 本次实际抽出的 8 个 source1 样例，**不是全量新普查，也不宣称覆盖九个 source**。其 mask 行为与所有样本共享的 reader 分支一致；r7 覆盖计数来自已有 receipt。

逐位比较同时要求 dtype / shape / uint8 位模式相同（包括 signed zero，不只浮点近似）：

| 比较 | delta | points | valid |
|---|---|---|---|
| 实际 legacy 训练 vs 部署 missing | **False** | **False** | **False** |
| 部署有效零位移 vs 部署 missing | **True** | **False** | **False** |
| 训练 unknown vs 部署 unknown | **null** | **null** | **null**：训练侧没有此输出，不能伪造再宣称相等 |

当前 curated reader 的实测拒绝是 `scheduler waypoint cache binding absent for curated tau; refusing realized-endpoint fallback before training`。

**代码级证据**（v6 路径相对本 Task；v5 完整位置另列）：

- `tools/run/train_pretrain.py:46–50,125–134,215–217,297–309`：真实 Steps / reader 参数 / DataLoader / `.to(device)` / 模型调用；没有 goal-unknown 选项。curated 分支先执行 dataset[0]，拒绝早于模型及 optimizer。
- `src/ptv3_shared_wam_v6/data.py:15–16,54–66`：reader 有 event_input_policy，没有 goal_input_policy；curated 缺 cache 就拒绝，其他路径继承 super().__getitem__，不生成 unknown goal。
- `src/task/ptv3-shared-wam-v5/src/ptv3_shared_wam_v5/data.py:60–61,155–157`：legacy goal 是 future_flow[-1] / all-True mask，collate 只 stack，不改变编码。
- `src/ptv3_shared_wam_v6/online_runtime.py:53–54,82–84,145–158,265–269`：missing 转 float32 zero delta + bool False，points 也零填，真实 collate 后断言再送模型。
- `src/ptv3_shared_wam_v6/model.py:138–147`：模型读取 goal_valid，清理无效 query，并把 goal token mask 拼入共享 attention；**消费者支持 mask，不等于训练生产者已生成 missing 编码**。

**明确回答：否，当前真实训练与部署在 goal 缺失 / 常量条件下尚未统一。** 部署 missing 有零填充 / False，训练没有可输出同编码的入口；有效常量（尤其零位移）仍是 True mask，不能当 missing。模型能消费一个 false mask 的事实不足以把 A 判为就绪。

### B：从既有产物计数，不生成 goal

B 只读取 brief32 的 persisted recording_layers / metadata、brief35 的 mother-provenance / receipt、r7 与固定 anchor JSON。没有重新扫描 8,482 个原 artifact，没有重放 scheduler，没有读取全量 raw pose/force，也没有实现任何候选规则。计数区分：原料存在、参数明示存在、完整规则可认证；几何 / 手位姿 / 时钟“可用”不能代替期望目标的规则参数。

| 规则族 | 必需的任务规则 / 参数 | 已记录参数候选（母序列 / 窗口） | 完整 waypoint 可认证 | 结论 |
|---|---|---:|---:|---|
| hold 在初始位置 / 常量零位移 | 选择 hold 的任务声明 / goal_rule_id，目标参考系与有效性 | **0 / 0** 明示规则；不是因为向量0难以计算 | **0 / 0** | 在本记录合同下新设 hold 意图：**NOT_DEPLOY_LEGAL** |
| 观测 reset pose + 固定 SE(3) 位移 | displacement / 坐标系 / 旋转约束 / anchor 定义 | **0 / 0** 明示通用参数 | **0 / 0** | 不能拿手工 +z 或按对象/手特化的距离补意图：**NOT_DEPLOY_LEGAL** |
| 沿接近方向位移 | 哪只手、方向定义与符号、距离 / 终点 / 姿态 | **0 / 0** 明示参数 | **0 / 0** | 初始 hand/pose 只能提供观测原料；距离和意图未录：**NOT_DEPLOY_LEGAL** |
| 物体几何→接触区 / 目标点 | 期望区域、side/patch、offset、hand role、姿态 / 终态 | **0 / 0** 明示参数 | **0 / 0** | normals / proximity labels 不等于事前接触目标：**NOT_DEPLOY_LEGAL** |
| **Shadow 原生构造常量世界目标** | **nominal object_start_pose + goal_displacement → goal_init_state** | **3 / 30**，规则参数来自冻结 collector 正证据 | **0 / 0** 完整既定 scheduler-waypoint 绑定 | 原生目标参数是正例；覆盖仅 **0.282% 母序列、0.5% 窗口**，不能推广 |
| 预先声明 reference / TaskSpec | source 身份、source_role、reset 顺序、对应 scheduler 路径 | brief35 可认证 **0 / 0** | **0 / 0** | 直接继承“不确定6000”，不重新普查或强改成非法 |

表中的“0”是既有导出目录内相应明示规则 / 参数或认证的计数，不是证明原始世界里绝无任何隐藏参数。未解码的 opaque 注释与缺日志仍不能升级为已录意图。所有规则函数都可在数学上设计得因果，但这不认证它们是这批目标执行的原任务。

Shadow 正例不能被抹掉：brief35 已存 3 条 `native_goal_position_evidence`，各 `(100,3)` 全程固定 `[0,0,0.8999999761581421]`；给定的冻结 commit `6141c34e` 和其 runtime copy 表明构造先于执行、reset 恢复 init 值。这些参数不需要发明。但 **constructor nominal anchor 与观测到的 post-reset object pose 不是同一来源**，因此不能把它偷换成“对任意样本的首帧 object pose 加0.2m”。也没有已录的同一部署 scheduler-waypoint 转换 / 帧映射 / goal type 绑定可直接复用。

≥80% 母序列门槛至少要 **852/1,064**；native 参数正例只有 **3**，到该门槛仍差 **849** 母序列，扩到全量则差 **1,061 母序列 / 5,970 窗口** 的原生任务规则。**没有 B 规则满足门槛**。若为补满这些记录而发明 displacement、接触区、方向或“默认 hold”，就是用新意图替换原合同，当前状态为 **NOT_DEPLOY_LEGAL**；本轮不实现、不启用。

### 历史 τ 可比性：只量输入身份影响

没有训练或新 τ 推理，因此新旧 EPE / τ 差值 **null，不能估计或宣称改善**。能量化的是合同改动覆盖范围：

- 若以后裁决 A 全量启用，6,000 条训练样本的 **192,000 个 goal_valid query 位**将由旧 all-True 变 False；固定 512 锚点的 **512/512 行、16,384 位** goal 条件身份会改变。旧输入 / cache receipt 不能当新条件下的测量。
- B 的 native 参数候选仅 30 条；固定 512 锚点中涉及这 3 个母序列的行有 **10/512**。这只是候选参数交集，**不是 10 条已解锁 waypoint 或新 τ 数值**；其余 502 行不能靠该规则认证。若推广 B 到全量，补造新规则会改变训练任务定义，历史条件身份不能继续沿用。
- 历史 B1 仍只能作旧配方参考。换 goal / event 来源后，误差变化不能解释为事件消融的因果效应。新条件下的模型 / B1 对照与聚类 CI 若要评估，需要另行冻结同一输入口径；本轮没有运行。

### C：只引用 brief32

[brief32-cost-envelope-20261023.json](brief32-cost-envelope-20261023.json) 的原值为：**1,064 母 rollout、1,067 对象流、2.423333h @30Hz 对象流、134,000,640–536,002,560 B 原始载荷**，另有 instruction / snapshot / geometry 开销。它是原有规模信封，不是新测吞吐或恢复成功保证。本轮直接复制原字段到 receipt.C，`estimates_recomputed=false`、`new_collection=false`，没有重新做采集估算。

## 我对哪条路可行的判断（含反例）

**A 是我建议优先交机主裁决的工程变更路线，但当前未就绪。** 它不需要猜期望位移或接触目标，能如实表达“goal 不可得”；仍须明确改变冻结的 goal_missing_policy，并把训练和部署所有模型入口接到同一缺失编码。不能因 API 已支持 None 就在现有配方下开训。

最小实现清单与规模（本轮不实施）：

| 位置 | 所需改动 | 最小规模 |
|---|---|---|
| `data.py` | 显式 goal policy、spawn/serde 传递、curated 分支在获批 unknown 策略下生成 zero delta/points + False；手物/contact 监督保留 | `__init__ / __getstate__ / __getitem__` **3 个现有函数** |
| `online_runtime.py` | checkpoint policy 校验并在 predict/online 全路径统一强制 unknown；即使 caller 提供有效 waypoint 也不能意外恢复不同训练合同；复用现有 sanitizer | **约3个现有入口函数**，含 forecast-only 共同构造路径 |
| `train_pretrain.py` | reader_options 对 TRAIN/DEV 传递策略、admission 与初始化/恢复身份绑定；没有 goal cache 时仅对明确新政策放行 | `main` **1 个函数**；必要的 recipe/admission 校验随之更新 |
| 新配方 / 身份 | 明确改写 goal_missing_policy，冻结新 condition identity / recipe / panel 输入合同 | **1 份明确配方变更**，不改旧冻结配方冒名续用 |
| tests | 训练与部署 post-collate bit-exact（None、有效输入覆盖、零位移反例）、监督不变与恢复拒绝旧身份 | **至少2组合同测试**；模型参数 / executor 接口 **0 改动** |

所以不是只翻一个 status；至少涉及 **3 个生产文件约7个函数 + 配方身份与测试**。本轮只交付审计工具、文档和其 tests，没有实施上述任何一项。

两个不能忽略的反例：

1. **零位移有效目标并非 unknown**：部署负控制实测 delta 逐位全零，与 None 相同，但 valid 为256 True，points 为查询点；None 则 valid全False、points全零。只清 delta 而保留 True 不能修合同。
2. **因果且确定不等于原意图**：同一 reset 观测与几何可以构造向左、向右或 hold 多种因果目标；若规则/参数未录，挑其中一种就是新任务声明。另一方面，Shadow 3 条原生固定目标是无需发明的反例，故不能笼统说“B每一个候选的每个参数都缺失”；准确结论是 **B 没有覆盖≥80%且可直接绑定同一 waypoint 语义的候选**。

若机主坚持保留有效、可认证的计划 goal，C 是已有规模预算的新录路线；若裁决接受 goal 不可得的统一语义，则先完成 A 的代码与配方变更。**本轮不能推出“只有新录才可能”**，因为 A 是可实施但尚未获启用裁决的路线。三路均未被启用；唯一一次 τ 训练保持 **0 updates**。

## 复现命令

冻结代码 HEAD：`2d4c07c341b33d8bbf7fa5e8cd7affadc1fd911a`。环境 Python 3.8.20 / torch 2.4.1+cu121 / NumPy 1.24.4，`CUDA_VISIBLE_DEVICES=''`；实测入口取证与旧产物计数 **3.138s**。7 项新测试通过（1.82s）。既有普查 `prior_censuses_rerun=false`；写前实测 `/home2` 可用 **147,013,210,112 B**。政策文件未改，SHA256仍 `7bfcd6e569bb205b88439a9d2bf056eff8421198a03fe614c69d1538d6308044`；全部 prior 输入 hash 在 receipt 中。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=src/task/ptv3-shared-wam-v6/tmp/brief36-tests-reproduce \
  src/task/ptv3-shared-wam-v6/tests/test_goal_unlock_readiness.py
# 7 passed in 1.82s

# 必须选不存在的新目录；不会优化、启用新策略或执行 network forward。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/audit_goal_unlock_readiness.py \
  --output src/task/ptv3-shared-wam-v6/docs/audit/brief36-goal-unlock-20261023-r1-reproduce
```

## 写盘字节数

预估上限 **16 MiB**、硬上限 **1 GiB**；receipt + 完整 tensor JSON / NPZ 实际 **202,448 B**。本轮完整交付 6 文件逻辑大小 **238,215 B**、磁盘分配 **249,856 B**（不含 Git object、工程测试临时 fixture 与他线写入）；无 checkpoint / 视频 / 轨迹副本，不删历史 outputs。
