# P-20261003-cm-value-adapter

## 决策问题

八个候选动作已经有真实 PD 差异，但随机后果数据是否足以让一个保留 Cm 物理
预测职责的 action-conditioned value adapter 学会局部任务 score？结果将决定是否
值得做新的 native A/B。

## 预注册方法

- 数据：固定 Cm、固定六专家和 `P-20261003-cm-candidate-advantage-r1` 的 287 个
  `p=1/8` 随机窗口；按独立 motion/start hash 留出，不读取 future state 作为输入。
- baseline：state + candidate PD target + candidate ID 的 ridge。
- Cm-on：baseline 加冻结 Cm 对每个候选的 score、ensemble std、retention、release，
  以及相对 fixed Cup 的同类物理量和完整候选 panel。
- 目标：实际 one-candidate H10 physical score。它是局部后果适配，不是最终成功率
  预测器。
- 固定 `Ridge(alpha=1)`，组留出用已知 propensity 计算 policy-vs-fixed IPW score，
  并用 motion/start cluster bootstrap 给区间。

## 判定

只有 Cm-on 的留出 policy score 下界为正，才启动新的 native A/B；否则关闭该具体
value-adapter 配方，不调 alpha/阈值/seed，不启动 PPO。
