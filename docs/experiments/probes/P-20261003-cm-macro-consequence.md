# P-20261003-cm-macro-consequence

## 目的

检验 one-tick Cm candidate 的物理变化是否被 fixed-Cup continuation 抹平。该 Probe
使用一个固定的 3-tick Cm candidate macro + 7-tick fixed-Cup continuation，与 fixed
Cup A/B 随机比较，不调整模型、阈值或 seed。

## 固定协议

- Cm、native expert、baseline actor 和 native PD bundle 全部冻结。
- contact state 触发后计算八个候选的 Cm 物理 score，Cm arm 选择 raw top；fixed arm
  选择 candidate 7。策略在观察后以 `p=.5` 随机分配。
- Cm arm 连续执行触发时的 candidate raw command 3 tick，随后 fixed Cup 7 tick；
  fixed arm 全程 fixed Cup 10 tick。
- 新 seed、96 env、30 Hz、windows-per-episode=4；按 motion/start group 做
  10,000 次 bootstrap。
- 每侧至少 48 窗口和 20 episode；score 90% 下界为正，retention/contact/clearance
  90% 下界不低于 -0.05 才算 `PROMISING`。

失败只关闭这个宏动作转换配方；不把它外推成 Cm 核心假设或最终成功率结论。
