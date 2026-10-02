# Decision Memo：训练 Cm 条件的二元 macro 选择策略

gated selector 的 native 失败关闭了冻结规则直接改动作的路线，但 raw 3 tick macro
仍有正向 score，说明策略可能需要同时观察历史和 Cm 的物理 consequence，自己学习
何时执行 macro、何时保持 Cup。此前六类 recovery PPO 被固定推荐先验压住，实际
argmax 没有改变；本路线不复用该先验。

选择一个有界策略学习 Probe：二元 categorical PPO，option 0 为 raw-top Cm candidate
重复 3 tick 后 fixed Cup 7 tick，option 1 为全程 fixed Cup。Cm-on 输入为当前历史、
候选动作差和冻结 Cm 的 score/std/retention/release；Cm-off 使用同形零物理通道，
网络初始化、actor proposal、reward、optimizer、rollout 数和 evaluation seed 全部
匹配。共同 reward 是接触支持的高度进展，并对末段 contact/clearance 丢失施加固定
惩罚；Cm 只提供动作后果预测特征，不预测最终成功率。

最小训练为 on/off 各 4 个完整 96-env rollout、每 rollout 后 PPO 更新；随后两个新
evaluation seed 做确定性 argmax。推进门要求：训练后实际二元 argmax 相对随机初始化
确实改变，Cm-on 的 stable hold 不低于 off 且 contact-loss 不恶化超过 2pp。若任一门
失败，关闭该 learned-macro 配方，不增加 rollout/epoch、调 reward 或扫描 seed；通过
才做正式多 seed Validation。

## 结果

Probe 已按固定预算完成。二元 actor 的实际 greedy 选择发生改变，且 Cm-on/off 的
选择分布明显不同；但两个 evaluation seed 汇总 stable success 为 Cm-on `7/192`
（3.65%）、Cm-off `11/192`（5.73%），未达到 stable-hold 门。acquisition 后
release 差为 `-0.06pp`，contact-loss 门通过，但单一失败门足以关闭路线。

结论：**UNPROMISING**。不增加训练轮数、epoch、reward 或 seed 扫描；Cm 一步物理
信息仍保留，下一次路线需要重新定义宏动作或更高层的价值/规划接口。机器结果见
`docs/experiments/probes/P-20261003-cm-learned-macro-policy-results.json`。
