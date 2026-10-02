# Decision Memo：先校准物理预测到相对任务价值

当前要区分的是：HF16 的一步 motor-conditioned Cm 预测已经有信息，但任务效用失败，
究竟是动作覆盖不足，还是从物理后果到相对任务价值的转换错误。

关键证据是 HF16 slot1 的一步物理信息门通过，slot2 中 Cm 实际改变了 129/179 个
base 窗口；但 Cm 对强 fixed-Cup 的 H10 保留高度差为 -1.671 mm，区间跨零。上一轮
decision-interface 使用通用 V continuation 时，Cm 改动作覆盖达到 47/47，但真实
局部抬升和 reward 明显下降，进一步指向 value conversion，而不是“模型没有输出”。

选择最便宜的离线 Decision Probe：复用 HF16 的 903 个随机 H10 窗口，冻结 Cm 的真实
动作条件物理预测，只训练一个小型 ridge relative-value head。输入是候选相对 fixed-Cup
的预测高度、接触保持、释放风险、预测不确定性和当前物理状态；目标是实际 H10 保留
高度相对同一模型 fixed-Cup 预测的 `delta-U`。它预测的是动作后果的相对任务效用，
不是成功率分类器，也不替代 Cm 物理模型。

fit/cal/held 按 motion/start group 固定切分，使用
`sha256("9851/motion/start")` 的个位桶 `0--3 / 4--5 / 6--9`。held 上同时改善 Spearman 排序和 RMSE
才算转换值得进入 native Probe；任一条件失败就关闭这个 calibration 配方，不扩普通
Cm 数据、不扫 ridge/seed、不启动 PPO。若通过，再用同一 relative head 做一次有界
随机 native 候选执行，之后才考虑策略学习。
