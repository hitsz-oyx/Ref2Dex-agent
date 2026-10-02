# P-20261003-cm-candidate-advantage

Family: CM candidate advantage；类型：Decision Probe；状态：RUNNING。

这个 Probe 区分候选动作覆盖/识别不足和 Cm 物理后果本身没有动作优势信息。每个合格
接触状态生成 HF16 的 8 个实际 native candidates：6 个专家动作、base-hold 和 fixed
Cup。状态观察后按 `p=1/8` 随机执行一个 candidate 一个 tick，再用同一 fixed Cup
continuation 执行 9 ticks。Cm 只计算触发时的一步物理预测；没有 future-state 特征、
task-value 训练或 actor/PPO 更新。

实验 ID：`P-20261003-cm-candidate-advantage`。计划使用 5 个显式 seed，每 seed 96
environment、每个初始 episode 最多 2 个窗口。held 分组沿用 HF16 的 motion/start
hash 分层。所有开始的窗口必须完整且没有 terminal/reset 污染。

预注册 screen：每个 arm held 至少 24 窗口、12 episode；Cm score 的排序和随机化
真实局部后果均优于 fixed Cup，且按 motion/start group 的 candidate-vs-Cup 90% 下界
为正，接触/clearance 风险差不超过 5 个百分点。支持不足记 `UNCLEAR`，支持充分但
排序或真实效用失败记 `UNPROMISING`。Probe 通过才允许下一步设计短 rollout value
conversion；不会因本 Probe 直接启动策略训练。

当前代码与完整结果将在 native 采集和独立审计完成后补写；中间输出保存在
`src/task/CmResidual/research/contact_consequence/output/P-20261003-cm-candidate-advantage-r1/`。
