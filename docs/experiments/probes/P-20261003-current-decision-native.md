# P-20261003-current-decision-native

Family: HF27；类型：Decision Probe；状态：COMPLETED；标签：`UNPROMISING`。

本 Probe 使用 HF27 current-step ridge head，只在每个 H10 决策时刻读取当前 Cm 候选
预测、触发状态和 fixed-Cup reference；Cm 的 risk/contact/OOD 物理门保持冻结。五个
scientific seed（521--525）共 384 个完整窗口，六个 controller arm 的真实 propensity
为 `1/6`，所有 arm 均达到至少 48 窗口、12 episode、8 initial groups 的支持门；native
重放和 countercommand 误差均为 0。

calibrated arm 相对 strong fixed-Cup 的 IPW 局部保留高度点估计为 `+9.62 mm`，但
90% initial-group 区间为 `[-8.14,+33.45]`，没有通过预注册的区间为正门。掉落差异
`-1.56 pp` 和 joint 差异 `0` 点估计可接受；相对原 Cm 的 `+34.37 mm` 虽区间为正，
但 joint-force 差异 `-0.09375` 未通过控制门。校准器在 6000 个 decision frames 中
相对原 Cm 改变 269 个推荐（44.8%），因此失败不是“没有改变动作”。

结论只关闭这个 current-step selector/value conversion 配方，不声称 Cm 物理预测无效，
也不启动 PPO 或最终成功率 Validation。下一步需换更高层的 representation/planning
或重新设计可识别的动作优势数据合同。

补充的 known-propensity candidate-effect ranking audit 也未形成可用排序证据：Cm 原始
score 的高低四分位 effect gap 为 +34.65 mm，但组级 90% 区间
`[-152.77,+197.17]`；current calibrated recommendation 的 gap 为 +126.77 mm，
区间 `[-151.77,+319.90]`。这说明点估计不能替代同一状态动作优势的可识别证据。
