---
schema: ref2dex.probe.v2
probe_id: P-20261023-reset-bound-goal-channel
experiment_id: P-20261023-reset-bound-goal-channel
date: 2026-10-23
task: ptv3-shared-wam-v6
branch: main
git_commit: 23a11ca629156c71d663a7c1d8f1c7cad93cebc0
classification: Blocker
status: BLOCKED
run_id: reset-goal-channel-20261023-brief45-r3
claim_id: goal-channel-contract
hypothesis_family: reset-bound-goal-channel
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: qualify the reset-declared task-level goal before a matched task-goal versus unknown comparison
decision_changed_if_negative: retain unknown until provenance exists; failed execution does not establish task information uselessness
---

# P-20261023-reset-bound-goal-channel

**brief45 修正：要接回的是任务级最终目标；局部 waypoint 只作诊断。**
撤回原“把调度器 waypoint 变成 goal 通道”的方向表述，保留来源、落盘和因果重放机制。
修正实现已通过工程验证，旧数据可认证覆盖仍 **0/6,000**；干净对照未运行，B−A=null。
本轮没有 v6 更新、新物理 rollout、checkpoint 或批量模型推理。

## 1. 契约：任务级目标与局部 waypoint 分开

[完整合同](../../audit/reset-bound-goal-contract-20261023.md)。任务级目标是场景/任务规定的
**最终空间 waypoint + 明确阶段语义**，reset 时可得。本实现显式绑定
`task_level_terminal_waypoint / task_completion_stage_no_arrival_deadline`，
规则必须声明 `target_stage=task_completion`；不是 t+24 实测目标，没有固定到达 deadline。
当前仅支持事前声明的 anchor + fixed_world_translation 最终任务；不替一般旧轨迹发明意图。

训推共用 `ResetWaypointChannel`：从 reset 来源一次生成 `task_goal_pose`，后续当前观测
只用于坐标变换，不改变最终目标。`update` 返回任务级 goal 的三字段；局部规划 pose 放在
`planner_waypoint_diagnostic`，不替代主条件。这里的局部诊断来自 OrderedGoalScheduler
路径子模块，不能冒充完整 TaskScheduler.local_goal_pose 或抓取/释放阶段反馈。

未来真实 pose 角色、未来时间戳、自身执行冒充独立录制、跨 reset，以及重算 hash 后替换
最终目标/参考路径均拒绝。当前物体 pose 合法作 reset anchor；显式任务 hold 的零 delta
有效，与 unknown 的 false mask 不同。来源 metadata 仍需可信 collector 记录，hash 不能
证明伪造来源真实。非零平移/旋转下与真实 v6 online/collate 数值逐位一致；新语义须显式
传入，旧 checkpoint 的 unknown policy、生产 reader 与缓存身份门未因此自动放行。

## 2. 可认证落盘及6,000窗口覆盖

运行代码 `23a11ca6`；[新归档 receipt](../../audit/task-goal-channel-20261023-brief45-receipt.json)。
完整产物 `outputs/ptv3-shared-wam-v6/reset-goal-channel-20261023-brief45-r3/`：

- `synthetic-execution/instruction.json`：规则、任务最终 pose、阶段、来源、reset/clock绑定。
- `synthetic-execution/waypoints.jsonl`：4个唯一 tick，任务 goal pose/delta、单列局部诊断、
  observed_at_ns / available_at_ns / decision_at_ns、instruction/reset绑定。
- 完全相同的 tick 重试逐位返回且不追加；冲突重试拒绝；已有目录拒绝覆盖。写入预算拒绝
  不推进 scheduler/tick。每执行指令≤1MiB、产物≤16MiB，磁盘余量<20GiB停止。
- `coverage.csv` 沿冻结 r7 顺序 join brief35 的6,000行，不重抽样、不追认旧 reset。

| 旧窗口认证项 | 实测 |
|---|---:|
| 可认证合法任务级目标 | **0/6,000（0%）** |
| 缺记录无法判定 | **6,000/6,000** |
| 确证候选来自本次未来 | **0/6,000**，不能把缺证当非法 |
| 显式 source_role / available_before_reset 证据命中 | **0 / 0** |
| 派生来源、角色、目标执行/reset、事前冻结、独立性证据缺失 | **各6,000/6,000**，重叠缺口不可相加 |
| Shadow 原生任务参数候选 | **3母序列/30窗口**；完整认证仍0 |

来源窗口数 s0=96、s1=5,632、s2=96、s4=30、s5=18、s6=66、s8=12、s9=4、s10=46。
旧缺证是接回任务级目标的 blocker；合成正控制只证明工程机制可执行，不计入覆盖率。
真实 collector 尚未接入 wrapper，不能声称已有真实新执行认证，也不能改名旧未来标签。

补证需找回真实历史来源/reset证据，或在新执行时记录；新执行不能倒贴旧人类轨迹。
原规模成本信封：1,064母 rollout、1,067对象流、261,720对象 tick，30Hz累计2.423对象流小时；
raw payload134,000,640–536,002,560B，另加几何/指令/日志。旧墙钟0.83–5.37h未经本轮实测。
新录数据若更换对照行，须在观察模型结果前重新冻结协议。

## 3. 干净对照与局部诊断格

[机器可读预注册](../../../configs/reset_goal_vs_unknown_prereg_v1.json) 已修订为v2。
A=unknown；B=合法任务级最终目标。相同TRAIN6,000、TRAIN512/DEV3356面板、模型/预算/
初始化/样本顺序，event均unknown；同K8 noise、native-capture整组配对bootstrap。
当前来源未过门，A/B没有运行；DEV认证尚未运行，值是null，不填0冒充普查。
固定权重输入诊断不能回答合法条件是否可学习；后续匹配学习预算只是预注册，不是训练授权。

| 条件 | 当前证据/状态 | 用途 |
|---|---|---|
| A unknown goal | 规范零填充+false mask；未运行本次配对 | 主对照 |
| B合法任务级目标 | 覆盖0/6,000；未运行 | 主问题 |
| C局部 waypoint | 沿用下述旧反例；新预算0 | 诊断，不能算本次干净三臂 |

[旧审计](P-20261022-wam-v6-goal-chain-audit.md) healthy4371 / DEV prefix8 原表：
oracle **14.709/10.537mm**，planner **22.980/11.287mm**，GT锚下恶化 **8.271mm**。
这两列是 **GT/own-object contact distance**（GT confidence>.25），不是hand EPE或成功率。
原表同DEV前8帧hand EPE亦从15.797升到21.122mm。旧planner用了完整录制路径作oracle
reference、event沿用录制标签，不能把这组数塞入修正合同的B，也不能假称与新A/B同预算。

若要补测C，必须先写出“为什么这次局部 waypoint 会更好”及“旧反例为何不适用”的机制
理由；当前没有该理由，故不分配新模型/物理预算。四个合成scheduler tick仅检验工程隔离。

## 4. 结果解释与执行边界

离线前8帧hand改善≥5%且配对95%CI全<0、contact hand改善≥5%、object恶化≤5%仅作
离线筛选，不是物理成功验收。模型实际消费的token/mask、时序和尺度诊断必须单列；未测
原因保留unresolved。CI跨0报不确定，不能报等效或追预算求显著。

即使任务级目标接回后成功率不提升，也只能写 **“该通道未通过执行落点”**，不得写
“任务信息无用”。拇指关节限位、approach未完成、约82mm残差等执行瓶颈与τ是否知道
最终目标是独立问题。本轮没有测成功率差异，`task_success_B_minus_A=null`。

## 环境与复现

Conda graspenv，Python3.8.20 / NumPy1.24.4，CPU来源/几何工程审计；没有科学训练或推理。
定向回归23 passed；扩大到v6全套及通道回归为249 passed / 1 failed：未修改的
`test_missing_scheduler_binding_fails_before_reading_any_labels` 绕过构造函数后缺
`goal_input_policy`，对应reader也未修改，不能称全套通过。

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/task/wam-rl-policy-v3/src:src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
 --basetemp=tmp/brief45-goal-tests \
 src/task/wam-rl-policy-v3/tests/test_reset_goal_channel.py \
 src/task/ptv3-shared-wam-v6/tests/test_reset_goal_certification.py \
 src/task/ptv3-shared-wam-v6/tests/test_v6_goal_contract.py

# 保留r1/r2/r3原始产物；输出必须是新目录。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/certify_reset_goal_channel.py \
 --output outputs/ptv3-shared-wam-v6/reset-goal-channel-20261023-brief45-reproduce
```

原r1运行代码92563d9a与旧receipt保留，证明当时局部waypoint工程合同的状态；它不认证本次
任务级合同，也不被重写成新成功结果。r2为修正未提交时的工程演练；r3绑定已提交实现。
