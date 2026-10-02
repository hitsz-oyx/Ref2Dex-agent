# P-20261003-cm-gated-macro

## 目的

验证 3 tick Cm macro 的正向 score 是否能在复用已有安全规则后保留，避免把 contact
风险带入策略学习。

## 固定协议

- 使用冻结 Cm、native expert、actor 和 native PD bundle。
- 触发时取 Cm raw top；只有满足既有 `top-fixed > 2*ensemble_std`、candidate 无 OOD、
  retention 至少 fixed-.05、release 不超过 fixed+.05 才允许 macro。
- `p=.5` 随机分配 gated-Cm 与 fixed Cup。gated-Cm 通过时执行候选 3 tick 后 fixed
  Cup 7 tick；不通过时整段使用 fixed Cup。fixed arm 全程 fixed Cup。
- 新 seed 626–630，96 env，30 Hz，windows-per-episode=4；10,000 次 motion/start
  cluster bootstrap。

这是新的风险约束执行合同，不调整既有阈值。若 score 或任何风险门失败，关闭该
gated macro 配方，不启动 PPO；全部通过才训练小型 categorical policy。
