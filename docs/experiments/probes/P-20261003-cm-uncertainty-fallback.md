# P-20261003-cm-uncertainty-fallback

这是 P-20261003 candidate-advantage 数据上的冻结模型机制审计，不重新拟合 Cm。

原预注册 selector 标记 `UNPROMISING`：287 个窗口中改变 282 个动作，score 相对
fixed Cup 的 IPW 90% 组区间 `[-7.3,+45.1] mm`。固定 uncertainty fallback 规则为：

```text
top score - fixed score > 2 * ensemble relative std
candidate OOD = false
retention(top) >= retention(fixed) - .05
release(top) <= release(fixed) + .05
```

否则执行 fixed Cup。该规则在同一 known-propensity 数据上的结果为：

| 指标 | IPW 增益 | 90% 组区间 |
|---|---:|---:|
| H10 retained score | +24.0 mm | [+7.0, +40.7] |
| retained | +0.401 | [+0.107, +0.675] |
| last-3 contact | +0.370 | [+0.033, +0.669] |
| last-3 clearance | +0.401 | [+0.107, +0.675] |

规则改变 143/287（49.8%）窗口；支持 287 窗口、25 motion/start groups、95 episode
IDs。结果标签为 `PROMISING_LOCAL_SIGNAL`，范围严格限于冻结模型的离线动作效应
审计。它还没有证明 direct-Q 比较、完整策略成功率或 Cm-on/off 学习收益。

下一步是新状态上的 native validation；成功后才训练 categorical option policy。
完整数字和 propensity/cluster bootstrap 审计见
`P-20261003-cm-candidate-advantage-results.json`。
