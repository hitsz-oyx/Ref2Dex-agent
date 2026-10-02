# P-20261003-cm-uncertainty-native

## 目的

检验离线已观察到的 `cm_uncertainty_fallback_2std` 是否能在新的真实 contact
state 上超过 fixed Cup。该 Probe 的结果决定是否值得启动 categorical option PPO。

## 固定协议

- Cm、六个 native expert 和 baseline checkpoint 全部冻结；不拟合、不更新 optimizer。
- 在观察到 contact state 后，固定规则为：只有 top candidate 比 fixed Cup 高出
  `2 * relative_std`，且 candidate 无 OOD、retention 至少为 fixed-.05、release
  不超过 fixed+.05 时才执行 top，否则执行 fixed Cup。
- 以独立 `p=.5` 随机分配 Cm fallback 与 fixed Cup；两者都执行 one-tick candidate
  后接 nine-tick fixed Cup continuation。
- 新 seed 为 613–617，96 env，30 Hz，windows-per-episode=4；统计按 motion/start
  group 做 10,000 次 bootstrap。
- 预设支持门：每侧至少 48 个窗口和 20 个 episode；score 的 90% 下界须大于 0，
  retention/contact/clearance 的 90% 下界不得低于 -0.05。

## 结果

五个 seed 均完成，合计 190 个窗口，Cm fallback 95、fixed Cup 95；20 个
motion/start groups，62 个 episode。支持门通过，但真实差异没有通过：

| 指标（Cm − fixed Cup） | 均值 | 90% cluster bootstrap |
| --- | ---: | ---: |
| score | -2.79 mm | [-15.74, +11.00] mm |
| first delta | +0.86 mm | [-1.58, +3.56] mm |
| retained | -0.095 | [-0.290, +0.063] |
| contact last 3 | -0.059 | [-0.244, +0.093] |
| clearance last 3 | -0.093 | [-0.295, +0.066] |

Cm fallback 改变 50/95 个 Cm 臂窗口（52.6%）。执行候选计数为
`expert0..expert5, base_hold, fixed_cup = [0,38,7,1,3,0,1,45]`。
冻结 checkpoint hash 为
`07537b1e9ae6b1e38b2a34367ca68436e1dfbe4c0c9585bba2c80c54f8c099a5`。

## 判定与后续

Probe 判定为 `UNPROMISING`：离线同批重放的正信号没有迁移到新 native state，且
保留和末段接触的点估计变差。按停止条件关闭该 uncertainty fallback 配方，不扫描
sigma、阈值或 seed，不启动 PPO。该结果不反驳 Cm 的一步物理预测能力，也不构成
最终任务成功率结论；它只说明当前固定 candidate ranking contract 没有可靠局部效用。

原始记录和审计结果见
[`P-20261003-cm-uncertainty-native-results.json`](P-20261003-cm-uncertainty-native-results.json)。
