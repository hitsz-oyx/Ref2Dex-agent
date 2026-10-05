这个分支的核心路线就是想看看加入了E(effect)和I(interaction)的两个的预测信息之后能不能对抓取产生帮助。先验证GT的E和I有帮助，然后再加入Cm

用户 ref11 明确调整当前阶段：暂停 native joint→execution forecast 支线，先把
真实 hand point-flow 作为 oracle planning action，检验 `(H,F_hand)→E/I`。
比较状态、真实端点流与 0→4/4→8 两段轨迹流，并使用 matched flow shuffle。
当前不以任务Y、控制映射或候选nativearm作为模型输入或完成条件。

本阶段完成条件是可靠的 oracle E/I 预测信息与清晰的同状态候选证据边界；
事后真实手运动可能包含物体响应，不能直接升级为前瞻因果planning结论。
真正 flow-space planning、desired flow 执行、matched Cm-on/off 训练策略收益
仍受根级 Mission 约束，在前置证据足够后分别推进。
