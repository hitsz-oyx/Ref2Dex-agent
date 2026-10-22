# 结果解读表

| 结果 | 实测数字 | 含义 | 决定 |
|---|---|---|---|
| 6,000 全可重放、等价与无前视通过，且与旧 endpoint 差异很大 | 未发生；合法条件 0/6,000，差异未定义 | 当前不能以条件修正解释改进 | 门槛(2)未达成，不启动冒烟或封顶训练 |
| 覆盖通过但差异为数值噪声级 | 未发生；max_abs_delta、分位数、方向一致性均为 null | 不能把“没有条件”解释成“条件相等” | 不作 B1 近似同语义或 B2 条件修正归因 |
| 只有部分窗口可重放，需点名缺失字段 | 实际为 0 可重放、6,000 不可重放；1,067 个 sequence/slot 组 | 全部缺任务意图；另有 84 个窗口的起点至 t 状态有效性前缀有缺口 | 报最小合同改动，由机主决定；不改 config、不训练 |
| 必须读取未来观测才能恢复条件 | 相同 ≤t 观测对应两种合法指令，条件分别 −7 mm、+7 mm，最大差 14 mm | 仅观测历史不能确定部署指令；把观测未来提升为指令违反约束 | 停止观测到指令的重建路线；没有准入任何这类条件 |
| 冒烟 step / checkpoint / identity 失败 | 未运行；0 optimizer updates、0 checkpoint bytes | 门槛(3)未验证 | 不能宣称基础设施已通过真实训练验收 |

本轮结论：**BLOCKED_CAUSAL_TASK_INTENT_UNAVAILABLE**。这不是继续等待一份
“已存储的 scheduler 路径”。部署调度器是一个消费任务意图的状态机，不是从实测历史
自动推断任务意图的规划器。没有事先已知任务意图，也没有生成该意图的因果输入时，
仅在录制状态上调用同一调度器，无法唯一恢复部署条件。已有 r8 的训练/推理语义差异
背景直接沿用，没有重跑旧 endpoint 哈希审计。

## 身份与盘点

- experiment_id：`P-20261023-scheduler-cache-identities`；task：`ptv3-shared-wam-v6`。
- run_id：`tau-curated-20261023-r9`；branch：`main`。
- 实际审计代码：`5939b405e823f5574a518e68e6fda0136b26706a`。
- 开工时 `r9/` 不存在，v6 范围没有未提交改动；r8 实现及 receipt 已存在，直接复用。
  未触碰执行器轴、PROJECT_MEMORY、curated_sampling 或训练 config。
- 环境：`/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python`，
  Python 3.8.20 / NumPy 1.24.4；CPU 元数据与 NumPy 调度器工程检查，无模型推理或训练。
  实际 executable、platform、版本完整保存在 receipt；未使用任何 GPU。

## 新证据：重放所需的输入不能从观测中凭空产生

部署 `goal_scheduler.OrderedGoalScheduler.__init__` 明确消费
`reference_poses, reference_times_s, object_points`；构造时就计算整条指令路径的
`path_points / arc / milestones`。`update_schedulers(..., fast_geometry=True)`
之后消费实测 pose/time/hand11，沿该指令推进。`inspire_shadow.BatchedInspireInputs.reset`
在 episode 开始时接收独立 `reference`；调度器更新不负责创造它。

本轮实际枚举所有 1,067 个 sequence/slot 组的数组文件、sequence record 与 meta 字段，
同时检查 curated/parent manifest：没有已知任务指令、task goal、task spec 或
instruction producer 字段。receipt 保存每组实际字段表，并逐条列出 6,000 个排除项。
没有把“需要输出的 `scheduler_reference_poses` 文件不存在”硬编码成唯一判断条件。
原始源数据中是否另有可绑定的任务意图，本轮没有穷举；本结论范围是当前 curated
录制合同及其 metadata，不能外推为所有源数据永远不可能补齐。

工程反例使用两份**独立合成任务指令**：≤t 的 live pose/hand 完全相同，指令本身
≤t 的前缀也相同；事前已知的后续任务分别朝 −x、+x。在 t=3，真实部署调度器发出
均值 `−0.007000000216066837 m` 和 `+0.007000000216066837 m` 的 x 条件，最大差
`0.014000000432133675 m`。任何仅消费观测前缀的确定性生成器都无法区分这两个任务。
这不是实际 6,000 条的 scheduler-vs-legacy 差异，也没有进入训练 identity 集合。

独立任务指令可在 episode 前描述后续目标；无前视要求限制的是 live 观测。
若改为把录制 `object_poses[t+1:]` 当该指令，则其未来来自标签，直接违反本轮约束。
只截取 `object_poses[:t+1]` 再当指令，也是在把观测提升为任务意图，且把路径终点
改成当前时刻，不能声称等价于部署时已知任务。两种办法均未使用。

调度器实际输出是**一个** `goal_pose / goal_delta`，带 `.8s` horizon；并不输出
24 个未来时刻的 scheduler waypoint。需要 K24 标签不等于需要读取未来实测状态
来运行 24 次 scheduler update。本轮没有制造这样的未来条件路径。

## replay helper 修复和边界测试

修复 [scheduler_cache_identity.py](../../src/ptv3_shared_wam_v6/scheduler_cache_identity.py)：

- 旧 helper 用 `reference_poses[t]` 冒充 live pose；现在每 tick 从独立
  `object_poses[t, slot]` 读取实测位姿。
- 旧 helper 整体转换并检查整段 joints，读取了未来尾段；现在逐 tick 只读当下
  pose、joints 和可选有效性 mask，不因未来 NaN 或截断而改变输出。
- 可显式选择 hand_index；已知 pose/hand11 无效时直接拒绝，不跨缺口伪造状态。
- 保留缺 instruction 必须报错的行为；没有改变任何生产数据 reader。

[测试](../../tests/test_scheduler_cache_identity.py)证明：固定独立指令时，观测 tail
截断、置 NaN、以及禁止整体数组转换/未来索引的 guard 均不影响条件字节；live 与
reference 不相同时仍使用 live anchor。原有 `_goal_from_pose` sanitizer 逐位等价、
跨 source 复合 identity、错误 endpoint 语义拒绝测试一起通过：**10 passed,
5 deselected，3.45s**。

这些是合成独立指令下的 helper/边界测试。**没有**合法的录制条件，因此
6,000 条代表子集的逐位等价与无前视验收均为 NOT_RUN，不能用合成通过替代门槛(2)。

## receipt v2 与语义相容性

[receipt v2](../../../../../outputs/ptv3-shared-wam-v6/tau-curated-20261023-r9/scheduler-cache-identity-receipt.json)
及 [完整性检查](../../../../../outputs/ptv3-shared-wam-v6/tau-curated-20261023-r9/integrity-check.json)
覆盖全部原始 6,000 个 admitted TRAIN identity，逐条键与 r7 一致，无遗漏或重复排除。

| 项目 | 实测 |
|---|---:|
| 可重放 / 不可重放 | 0 / 6,000 |
| scheduler 条件哈希唯一数 / 重复数 | 0 / 0 |
| 条件哈希重复率、复合 identity 碰撞数 | null / null（无准入条件，不能报 PASS） |
| scheduler-vs-legacy max_abs_delta、分位数、方向余弦 | 全为 null |
| 显著不一致窗口清单、是否为合法规划意图 | null（无法比较，无可解释的准入意图） |
| TRAIN / resume optimizer updates | 0 / 0 |
| checkpoint bytes | 0 |

逐 source 窗口数：s0=96、s1=5,632、s2=96、s4=30、s5=18、s6=66、s8=12、s9=4、s10=46。
旧 realized-endpoint 的 6,000 个唯一哈希直接引用 r8，没有重复生成。
无法区分“条件差异很大”与“数值噪声级”；不能据此给 B2 做条件修正的因果归因，
也不能把旧 `step011000` 认证为同语义基线。它继续保留旧契约身份。

## 缺失字段和最小合同改动

| 输入 / 绑定 | 当前情况 | 最小动作 |
|---|---|---|
| `OrderedGoalScheduler.reference_poses` 所代表的**事前已知任务意图** | 6,000/6,000 缺；不允许用观察轨迹替代 | 绑定独立 task instruction，或提供因果 task-goal producer 及其意图输入；不要求预存 scheduler waypoint |
| `reference_times_s` | 可按既定 30Hz 指令时间基生成，不是单独存储 blocker | 与获准的指令一并明确时间基与 reset 起点 |
| `object_points / interaction_points` | 已有 frozen query/surface 几何 | 复用原 geometry；不增加模型输入 |
| live `object_pose / hand11 / time` | 当前窗口 H4 合格不保证从 episode 起点可连续重放 | 明确已知 reset，记录连续有效状态，或提供该时刻的完整 scheduler+interaction snapshot |
| `hand_index` 与部署 hand11 的对应 | 录制有两手，当前合同未绑定 scheduler 使用哪手 | 按 source/sequence 明确绑定，不因结果好坏选手 |
| `goal_type / SchedulerConfig` | 本轮请求可固定 `se3_pose / SchedulerConfig()` | 在新合同中显式绑定，不从未来运动猜测 |
| `execution_ready / lost_control` | 本轮指定的 batch update 使用默认 None / False | 不是缺失 blocker；不要求虚构外部 readiness/force 字段 |

状态前缀的附加问题：26 个窗口的 `object_pose_valid[0:t+1]` 含无效帧；76 个窗口
两手都没有从起点到 t 连续有效的 hand11（这里依据录制有效性 mask，不是新物理认证）。
两者并集 **84** 个窗口：s6=26、s8=12、s10=46；其余 5,916 仅通过该项 mask 前缀检查，
没有因而获得合法任务指令或完整 replay admission。

即使只重置到每个窗口 H4 起点，也会丢失 scheduler 的 pointer、last_pose/time、
progress/recovery 状态和 interaction 历史；没有部署对应 reset/snapshot 不能声称
同源重放。本轮没有跨缺口补值、回填未来或自设 reset 来提高覆盖。

需要机主决定的是**意图输入合同**：提供可在部署前得到的任务意图来源/生成规则，
还是另行改变“示教路径作为已知任务指令”的研究定义。后者与本轮禁止把
`object_poses` 当指令的约束冲突，本轮没有自行采用。不是要求机主批准某个网络或 reward。

## 复现命令与写盘字节数

审计纯读取输入，输出路径必须不存在，避免覆盖已有 receipt。以下命令可在不存在的
同级输出文件重新生成；原 r9 证据保留。环境不需要 GPU。

```bash
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/audit_causal_scheduler_inputs.py \
  --curated-manifest outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json \
  --output outputs/ptv3-shared-wam-v6/tau-curated-20261023-r9/reproduced-receipt.json

PYTHONPATH=src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src:src/task/wam-rl-policy-v3/src \
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q -p no:cacheprovider \
  src/task/ptv3-shared-wam-v6/tests/test_scheduler_cache_identity.py \
  src/task/wam-rl-policy-v3/tests/test_batched_geometry.py \
  -k 'scheduler or replay or instruction or identity or train_and_infer_goal_tensors_match_bitwise or future_endpoint_semantics_are_rejected_by_both_paths'
```

写盘前 `/home2` 可用 `148,654,813,184 bytes`。r9 两个 JSON 的逻辑大小合计
**4,601,799 bytes**，实际分配 **4,608,000 bytes**，低于 2G；没有模型、cache 或 checkpoint
输出。代码/Markdown 与解释器产生的小量字节码不计入上述 r9 产物数值。
