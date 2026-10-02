# Decision Memo：用随机 native candidate 建立可识别的 Cm 动作优势契约

当前需要决定的是：Cm 的一步物理后果预测是否能在真实候选动作之间提供可识别的
相对优势，还是此前失败主要来自候选覆盖和 task-value 标签契约。

已有 HF16 native panel 每个触发状态都生成 8 个实际 motor candidates，但此前的
whole-window controller assignment 只随机了 5 个控制律，不能给 8 个候选提供相同的
随机化支持。HF09 的 2+8 数据也没有使用 HF16 的 candidate panel。继续调 selector 或
value head 不能解决这个识别缺口。

本 Probe 固定同一 airplane、同一 strong fixed Cup continuation 和 HF16 的 8 个真实
native candidates。对合格接触状态在观察状态之后按 `p=1/8` 分配一个 candidate，执行
该 candidate 一个 simulator tick，随后固定 Cup 9 ticks。Cm 只用于触发时的一步物理
预测和诊断记录；不使用未来状态，不训练 task-value head，不把后果预测改名为成功率
预测器。每个窗口保存候选面、实际 PD、assignment、propensity、Cm score/retention/
release 和 H10 后果。

最便宜的判别条件：

* 每个 candidate 在 held group 至少 24 个窗口，至少 12 个 episode；否则 `UNCLEAR`；
* Cm 预测的 candidate 相对 fixed Cup 排序在 held 上的 Spearman 与 top-choice
  observed local utility 均为正，并且不劣于 shuffled physical predictions；
* fixed Cup 对照的共同局部结果按 motion/start group bootstrap，Cm 选择的真实
  candidate 相对 fixed Cup 下界为正；风险/接触不恶化超过预先固定的 5 个百分点。

这些是 Probe screen，不是最终稳定抓取或 policy utility 结论。失败时关闭单步候选
优势数据契约，不扫描阈值、ridge、seed，不启动 PPO；成功时才设计独立的短期 value
conversion 和 matched policy training。

首批 5 个 seed 完成后共有 195 个窗口，但 arm 支持为
`[34,30,14,24,24,23,19,27]`，因此支持门尚未满足。首批固定审计显示 candidate4
相对 fixed Cup 的点估计为正，但 group interval 很宽；这不是效用结论。只因支持不足，
追加一次预先限定的 support-completion：再用 3 个显式 seed，完全相同的候选生成、
`p=1/8` 分配、one-tick candidate + nine-tick fixed Cup continuation 和独立输出目录。
不根据首批分数改 candidate、模型、阈值或标签。追加后若任一 arm 仍少于 24 个窗口，
直接记 `UNCLEAR` 并关闭此数据契约；若支持通过，再按原定 group bootstrap 审计。

总成本上限仍为单 GPU native 60 分钟、输出 8 GiB；所有模型和专家冻结，输出只写本
仓库 owned research/output。
