# 先扩大真实控制权限，再学习对应后果

2026-10-02，HF12关闭后下一步Decision；尚未启动新科学slot。

问题：原2步候选+8步base的干预合同，是否本身不足以持续保留物体？
HF12改变guide所有权和共同reward后，on5130评价决策仍沿用初始推荐；
off45/5144已改变，不能说整个PPO执行管线无法改变动作。On非base片段
仅占.485%有效帧，候选总覆盖4.952%。这些事实使控制合同成为值得区分
的高层假设，但不证明稀疏控制是唯一原因，也不把17vs19当核心思想反证。

行动：停止旧selector的lambda/reward/epochs调整。首先实现物理可解释的
持续控制计划：选定专家在窗口内逐步根据观测出动作，以及抓取姿态保持
（用真实native PD逆变换保持固定wrist/finger目标，遵守关节/耦合限制）。
不增加随意±扰动。base仍是明确对照；保持候选不能凭名字假定有用。
先工程核对命令、native PD目标、末端截断、重新观察及冻结模型合同。

随后最小Decision Probe在共同处理前的接触/已抬升状态随机分配完整10步
计划，记录逐步实际命令、支持保留、丢失和末端高度，已抬升风险单独报告。
先问“可执行计划是否提供保留机会”，若没有机会则回到抓取几何/候选设计，
不再拟合一个没有好动作的Cm。如果存在机会，训练预测该真实10步计划的Cm，
输入须包含计划身份/参数，不能把旧2+8 checkpoint当成持续10步计划模型。
后续闭环可以只执行首2步便重新观察与决策，但预测的计划须覆盖整个窗口；
实际MPC结果独立验证，不能拿离线全10步标签冒充重新规划的实际后果。

新Probe的seed、propensity、对照、metric、阈值和停止条件需在首次采集前
固定在独立卡中，当前不预判正结果。不在这一决定里继续跑全任务成功率。
正信号才再接入可学习策略；最终仍需独立稳定抓取/掉落与multi-training-seed
Cm-on/off Validation。保持候选上线前核对whole-mesh/table clearance，
避免把净接触力或中心上抬直接解释为物体已被手抓离桌面。

成本：先小GPU工程smoke，新Decision Probe目标≤60min/8GiB、一张空闲GPU。
若无法在剩余deadline/预算内完成最小判别设计，则记录未完成范围，不压缩
label/配对/结论边界。MISSION claim、资源权限、外部read-only不变；
当前会话直接执行，无子代理、无新身份、无外部不可逆操作，无新授权边界。

Pre-science power check: r2engineering35clear windows/217total atquota2.
Choose twelve collection panels with64clear-first envs/32general and
already-clear.4hold/.4base/.04eachotherexpert allocation. This changes only
predeclared state sampling/allocation before science, allsevenoptions retained;
base duplicated null slots.2each. Reverify engineering, charge r2; no success
labels used for this decision, unchanged opportunity/support/risk gates.
