# P-20261002-cm-decision-interface

Family: CM decision interface；类型：Decision Probe；状态：COMPLETED；Probe 标签：UNPROMISING。

这个 Probe 区分：Cm 的短期模型 rollout 能否提供 direct-Q/Cup 没有的**相对动作排序**。
它不训练 actor、critic、V、Q 或 dynamics，也不启动 PPO。当前 baseline 是冻结的
rotation-Cup `pi0`；continuation `Q_base` 使用现有 physical-value V，direct-Q 作为
同一候选面的强对照。

在每个真实接触状态，先构造 Cup 周围 37 个归一化局部动作（原动作、每个维度 `±0.1`），
再保存完整候选面与两组分数。Cm 分数是三成员短 rollout 的均值减标准差，包含一步真实
局部 reward 和 `gamma * Q_base(predicted state)`；Cm 选择不活跃时回退原 Cup。候选面
形成后才以固定概率随机执行 Cup、direct-Q、Cm、打乱 direct-Q 排序和随机候选，连续
执行 10 步并保存实际状态、接触、clearance、动作和局部 reward。

工程 smoke 使用 96 环境、airplane、`windows_per_stratum=1`；正式 r1 使用
`windows_per_stratum=2`、最多 650 tick，clear/non-clear 各取两个窗口，每臂概率 0.2。
完整窗口要求没有 reset/terminal 污染。工程检查只要求
checkpoint/schema、有限 tensor、冻结参数和候选面/动作合同通过。

首轮只作 Probe 标签：

- 支持门：每个臂至少 24 个完整窗口；Cm 至少改变 24 个相对 Cup 的候选；
- 排序门：held 数据中 Cm 相对 Cup 的真实 10 步局部效用改善为正，并优于 direct-Q；
- 局部效用为 10 步高度变化加 tracker stable reward，另报末三步共同接触和 clearance；
- 任何门支持不足为 `UNCLEAR`；支持足够但排序/真实效用失败为 `UNPROMISING`；三个门
  同时通过才允许设计后续策略训练。

真实局部结果是随机候选的 on-policy/IPW 观察，不把单次窗口升级为完整成功率或稳定抓取
结论。普通 Cm 数据规模、PPO 和旧 residual 路线均不在本实验内。

## Native 结果

`r1` 使用 seed705、assignment20705、96 个环境、每个 clear/non-clear 层各两个窗口，
共 212 个完整窗口；五臂支持为 Cup/direct-Q/Cm/shuffled/random = 39/37/47/47/42，
没有 terminal 或 reset 污染。Cm 排序在 212 个状态都给出高于 Cup 的候选分数，且 Cm 臂
47/47 个窗口改变了候选动作；整体改变率为 171/212。这说明模型确实进入了决策接口。

真实 10 步结果没有通过效用门。末三步最小正抬升的均值为 Cup 34.42 mm、direct-Q
42.36 mm、Cm 11.97 mm；Cm 相对 Cup 的按环境 cluster bootstrap 90% 下界为
`-38.27 mm`。局部 reward 总和为 Cup 0.998、direct-Q 0.977、Cm 0.348，Cm 相对
Cup 的 90% 下界为 `-0.965`。Cm 的末三步共同接触比例较高（84.4% 对 Cup 71.8%），
但 clearance 比例较低（19.9% 对 36.8%，差值 90% 下界 `-28.70` 个百分点）。

因此支持门和动作覆盖门通过，真实局部排序门失败，Probe 定为 `UNPROMISING`。这只是
冻结模型的候选排序筛查，不能推导完整策略成功率或稳定抓取结论；按合同不启动 PPO，
不扩大普通 Cm 数据，也不继续在该接口上扫描阈值。完整统计见
[`results.json`](P-20261002-cm-decision-interface-results.json)，原始候选面和真实
窗口保存在 `src/task/CmResidual/research/decision_interface/output/`。
