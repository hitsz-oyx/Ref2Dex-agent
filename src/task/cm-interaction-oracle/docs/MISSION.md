这个分支的核心路线就是想看看加入了E(effect)和I(interaction)的两个的预测信息之后能不能对抓取产生帮助。先验证GT的E和I有帮助，然后再加入Cm

用户 ref11 调整 oracle E/I 预测阶段：暂停 native joint→execution forecast 支线，先把
真实 hand point-flow 作为 oracle planning action，检验 `(H,F_hand)→E/I`。
比较状态、真实端点流与 0→4/4→8 两段轨迹流，并使用 matched flow shuffle。
ref11 阶段不以任务Y、控制映射或候选nativearm作为模型输入或完成条件。
随后 ref12 在同一 oracle 合同上单独加入后续任务Y的链路检验。

本阶段完成条件是可靠的 oracle E/I 预测信息与清晰的同状态候选证据边界；
事后真实手运动可能包含物体响应，不能直接升级为前瞻因果planning结论。
真正 flow-space planning、desired flow 执行、matched Cm-on/off 训练策略收益
仍受根级 Mission 约束，在前置证据足够后分别推进。

用户 ref12 将当前阶段推进到 oracle flow→predicted E12/I14→Y 的链路检验。
Y 沿用 ref8 continuation 合同；source E/I 必须为按环境 cross-fitted/OOF 预测，
test 用 full-source 模型。保持 H、直接 flow、预测 E/I、GT E/I 和 hybrid matched
对照。配对采集及其工程检查仍延后，native execution predictor 仍暂停。
这一阶段只评价事后 oracle 表示链的任务预测信息，不升级为可部署 planner。

用户 ref13 将当前路线改为逆向必要性检验：先做同前缀候选的 Oracle-Y Utility
Gate，真实执行七个局部干预，区分短期 continuation Y 与稳定持有 Z。
这重新启用配对采集；通过后才做噪声容忍度与表示比较。继续暂停旧 execution
predictor、PPO 和进一步 Y/E/I MSE 调参，不强制 E/I 作为信息瓶颈。


用户 ref13_1 要求先纠正 one-shot 与 rolling short-Y 的适用范围：复用保存轨迹，
保持短窗口标签/utility，检查时序分叉和到真实失败的提前量。完成条件是完整
窗口、原标签精确重放与清晰的后续不同 H 边界；不得把轨迹查表/拼接写成滚动
干预的可执行收益或上限。新 rolling 交互验证与预测模型属于后续独立阶段。

用户 ref14 固定逆向闭环必要性主链：Z → GT-Z候选机会 → rollingGT-Y
控制 → Y精度容忍度 → 直接Y/物理瓶颈/hybrid → actualflow预测 → desiredflow
执行 → learnedplanner → 后续蒸馏/PPO。每个前置环节须先有决策价值。
更新后的 ref14_1 当前只推进真正same-current-state rollingGT-Y：保持原短Y、
utility与Z，每8步重新fork评估并实际执行组合选择，保存其真实动作前缀继续。
允许数学上等价的baseline最大utility剪枝；未模拟候选标签不补齐。
当前属于Probe，不能把旧one-shot候选Z上限当rolling上限；不得先改long-Y、
重开executionforecast、训练预测器或PPO。正向才为下一阶段噪声容忍度提供依据，
不自动完成全局Cm对训练所得策略的因果收益要求。


用户ref14_3在rollingGT-Y和noise gates之后授权actual hand flow→Y ranking
predictor阶段：保持Y/U，比较Direct、严格source-OOF E/I bottleneck与Hybrid，
按anchor隔离并检查shuffle；actualflow仍是事后oracle。离线gate未过不得进入
rolling learned-Y。此次固定Probe已完成但三臂未过70%screen；不改变Mission claim，
不重开execution、desiredflow、PPO/critic/reward，也不外推为E/I无效。
