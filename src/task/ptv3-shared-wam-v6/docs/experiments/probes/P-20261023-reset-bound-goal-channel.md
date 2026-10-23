---
schema: ref2dex.probe.v2
probe_id: P-20261023-reset-bound-goal-channel
experiment_id: P-20261023-reset-bound-goal-channel
date: 2026-10-23
task: ptv3-shared-wam-v6
branch: main
git_commit: 92563d9ad53a7bc9e593d8318d6a2b6156de4521
classification: Blocker
status: BLOCKED
run_id: reset-goal-channel-20261023-r1
claim_id: goal-channel-contract
hypothesis_family: reset-bound-goal-channel
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: certify the same scheduler goal producer before a paired waypoint versus unknown comparison
decision_changed_if_negative: retain unknown conditioning and quantify missing bindings and new-recording cost without claiming task intent useless
---

# P-20261023-reset-bound-goal-channel

**交付：契约可执行，来源写盘链通过工程验证；旧数据合法覆盖仍为0/6,000，干净对照已预注册但未运行。**
没有 v6 更新、新物理 rollout、checkpoint、模型替换或旧标签重命名。

## 第一步：契约与负控制

[完整合同](../../audit/reset-bound-goal-contract-20261023.md)：空间 waypoint + 有序几何阶段，
无 endpoint deadline。唯一 waypoint 产生器是原 `OrderedGoalScheduler / update_schedulers`，
训练重放与新执行共用 `ResetWaypointChannel`。本实现是显式 reset anchor + 声明的固定位移
规则，不发明旧任务意图；不宣称一般 TaskScheduler 全反馈链已认证。

未来真实 pose 角色、未来时间戳（即使保留合法角色）、本执行冒充独立参考、跨 reset、
换 hash 后塞入未来路径、跳历史与未来 live 观测均被拒绝。当前物体 pose 合法；显式 hold
规则可生成合法当前 pose goal：delta=0、points=query、valid=True，和 A 的 unknown 不同。
训练与部署共用 producer，并在平移/旋转非零情况下与真实 v6 online/collate 三字段逐位相同。

这是来源与时间合同的可执行检查，不是凭 pose 数值猜它来自未来；伪造所有 metadata
不能由 hash 自动揭穿，真实认证仍要求 collector 边界可信记录。

## 第二步：落盘与覆盖率

实际运行代码 `92563d9a`；[小型归档 receipt](../../audit/reset-goal-channel-20261023-receipt.json)。
本轮相关回归 **31 passed in 2.33s**，repo verify PASS。测试与产物均绑定该版本。
收尾时共享工作区另一会话正在把同一代码改成 task_level_terminal_waypoint；本轮没有
覆盖、提交或认证这些并发修改，不能将本轮结果用于其新语义。
完整产物在 `outputs/ptv3-shared-wam-v6/reset-goal-channel-20261023-r1/`：

- `synthetic-execution/instruction.json`：每项 source_role、available_before_reset、source hash、
  observed/available 时间、execution/reset/clock，以及 rule/anchor/query/surface 内容。
- `synthetic-execution/waypoints.jsonl`：4次 scheduler update 的 mode、进度索引、goal_pose/delta、
  时间与 instruction/reset 绑定；保存后重新读回重放逐位相同。明确 synthetic，不认证旧窗口。
- `coverage.csv`：沿原 r7 顺序严格 join brief35 的6,000行；不重查原始来源，也不重抽样。
- `receipt.json`：覆盖率、环境、代码版本、成本及未执行的对照状态。

| 旧窗口认证项 | 实测 |
|---|---:|
| 可认证合法 | **0/6,000 (0%)** |
| 缺记录无法判定 | **6,000/6,000** |
| 确证候选 waypoint 来自本次未来 | **0/6,000**（不把缺记录算非法） |
| source_role / available_before_reset 显式证据命中 | **0 / 0** |
| 派生来源、角色、目标执行/reset、事前冻结、独立性证据缺失 | **各6,000/6,000** |
| Shadow 原生规则参数候选 | **3母序列/30窗口**，完整 waypoint 认证仍0 |

来源窗口数：s0=96、s1=5,632、s2=96、s4=30、s5=18、s6=66、s8=12、s9=4、s10=46。
判据不按 source 或物体放宽。Shadow 冻结 `6141c34e` 的 nominal anchor + 固定位移、
reset 恢复是合法先例；新生成链保留 nominal anchor，不以 post-reset 测量替换。
示例演练不是原三条 Shadow 的真实执行补证，原 collector 尚未接入 wrapper。

**代价**：仅补字段不能追认旧目标执行；需找回真实历史 reset/来源证据，或用新 collector
在新执行时绑定。原规模成本信封沿用 brief32：1,064母 rollout、1,067对象流、261,720对象
tick，30Hz累计2.423对象流小时；raw payload **134,000,640–536,002,560 B**，另加指令/
几何/日志。旧条件假设墙钟0.83–5.37h未经本轮实测，失败、重试、控制器资格与工程接入另计。
新录不保证产生原6,000窗口，不能将其贴回旧人类轨迹；若更换数据必须重新冻结对照。

生成链每次执行 instruction ≤1MiB、总记录≤16MiB，磁盘余量<20GiB停止；路径已有则拒绝
覆盖。没有改只读外部 collector、既有 checkpoint 或生产训练 reader 的 HOLD。

## 第三步：冻结的干净对照

[机器可读预注册](../../../configs/reset_goal_vs_unknown_prereg_v1.json)。同一6,000训练数据、
同一TRAIN512与原DEV3356面板、同一模型/初始化/优化预算，只有 goal 三字段及其显式 policy
不同；event 两端均 unknown，防止未来事件条件混入。K8使用相同8个 noise seeds，关联窗口
按 native capture 整组 paired bootstrap，主指标DEV前8帧hand EPE，contact hand与object作守卫。

先冻结现有权重的输入对照（每臂≤3,600s、总产物≤2GiB、GPU仅4/5/6/7）；这只能诊断当前
checkpoint 的消费，不能证明合法条件的可学习性。A checkpoint 的 unknown policy 会清除
goal，故 B 开跑前必须验证**真正模型消费**的 tokens/valid 没有被旧 policy 抹掉；不靠改输入
后结果相同来宣布无效。当前生产 B reader 尚无合法缓存入口，需先过100%来源/重放/边界门。

| 预注册结果 | 下一步及解释边界 |
|---|---|
| B前8帧hand至少改善5%，paired95%CI全<0；contact hand也改善≥5%，object恶化≤5% | 接回可认证 goal 通道；以相同合同设计后续学习验证，不改变本轮“不重训v6”限制 |
| CI跨0或指标方向冲突 | 报不确定/局部作用，不称等效，不扩预算追显著 |
| 无改善或更差 | 只能说当前checkpoint/架构/消费方式未用好该条件；不能说任务信息无用 |

负结果固定诊断：①**注入口**：post-collate与真实模型 token/mask是否收到B、是否被A policy
清除；②**时序**：按既有scheduler mode、confirmed/target index分层，报告当前stage到目标
的需求，不将stage解释成t+24；③**尺度**：delta与既有normalizer/position_scale=.05m之比、
饱和/异常值和输入token RMS。未测或无法区分的原因写 unresolved，不编造三者之一为根因；
这些诊断不能倒改验收、不按物体特化。

后续匹配学习的预算在预注册中单独冻结为每臂500更新/3,600s、global batch288、同初始
权重与新optimizer、同sample顺序；任一未完成则比较未完成。它**不是本轮训练授权**。
合法来源覆盖0使本轮无法获得科学上的正/负比较，B−A=**null**。DEV认证未运行，不能填0
当作已普查。禁止用已知录制物体路径oracle reference的Planner-G冒充部署评估。

## 环境与复现

Conda `graspenv`，Python3.8.20 / NumPy1.24.4；CPU只执行来源校验、scheduler几何和合同
测试，没有神经网络训练或批量推理。GPU未使用。

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/task/wam-rl-policy-v3/src:src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
 --basetemp=tmp/goal-contract-phase2-tests \
 src/task/wam-rl-policy-v3/tests/test_reset_goal_channel.py \
 src/task/ptv3-shared-wam-v6/tests/test_reset_goal_certification.py

# 输出目录必须不存在；原r1证据保留。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/certify_reset_goal_channel.py \
 --output outputs/ptv3-shared-wam-v6/reset-goal-channel-20261023-reproduce
```
