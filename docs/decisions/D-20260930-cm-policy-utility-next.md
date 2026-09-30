# 2026-09-30：优先回答 Cm 策略价值

用户确认先获得真实策略上的 matched Probe，出现正向信号后优先完成正式验证；
期限为 2026-10-03 23:59（Asia/Shanghai）。本次暂时跳过代理工作流搭建，资源
和科学结论边界不变。复用 C1 底座，单 actor 蒸馏不作为默认优先交付。

当前阻碍：r7 的 ridge 校准修复虽达到区间覆盖门，但 holdout fallback 为
83.87%、阈值稳定性为 88.71%，未通过其余门槛；不能据此否定设计中的
128/128 MLP。六专家轨迹蒸馏最新现存 r2 fit manifest 为
FAILED/BrokenPipeError，没有可用于训练的轨迹。

选择：对原 scratch teacher-arbitration 设计补做一次固定 MLP 估计器可行性
检查。这是进入策略 Probe 的 Decision 前置门，不是策略收益实验，也不重开
HF01–HF05。只训练预声明 MLP，不换 seed、目标、阈值或宽度调参。
数据沿用有轴合同的 r7：fit 188、holdout 186；holdout 已被旧 screen 查看，
这里只作探索性筛查，不当作新的独立验证集。输入须过 canonical adapter，
只使用 pre-action observation、candidate action 和 candidate id。

预算：2 CPU，15 分钟，100 MiB，0 GPU。固定 seed 278，Adam lr=1e-3，
weight_decay=1e-4，1000 个 full-batch update。fit 用 seed 278 固定拆成
80% optimization / 20% calibration（episode disjoint）；归一化仅来自 optimization。
一步 delta 使用标准化 Huber loss，五步 contact 使用 q10/q50/q90 pinball loss。
不以结果选择 checkpoint；只在 calibration 子集上冻结 delta 各坐标的
absolute residual q90 半径及 contact q10 的单侧 q90 校正（最小为零），
随后一次评分 holdout，不用 holdout 修正区间。

延续原门：holdout contact q10 下界覆盖 >=0.90，delta 区间覆盖 >=0.80，
fallback <=0.50，0.55/0.60/0.65 teacher 选择稳定性 >=0.90。
设计 delta head 只有 3D 点估计，没有 delta quantile 输出，不能从 contact
quantile 获得米制位移区间；这里明确用上述独立 calibration 残差半径补齐，
不把 contact quantile span 当米制半径。若任一门失败，停止此 teacher-envelope
实现，不扫描门限；转回更高层机制选择。若通过，才修复轨迹采集并制定
matched Cm-on/off/placebo student 和物理 held-lift Probe；拟合成功不能
替代真实策略实验。技术错误或输入漂移立即停止，标为执行失败而非科学失败。

最终比较必须保持任务、底座、训练容量、更新数、seed 和评估条件匹配。
任何正式 claim 仍需正式 Validation。本决定不保证得到正结果。
