# P-20261003-cm-uncertainty-gated-candidate-v2

状态：`AUTHORIZED / RUNNING`（native 采集完成后补写结果）

这是对旧 candidate-advantage 线索的最小 corrected native Probe。它区分：冻结 Cm 的
ensemble uncertainty 是否能在真实接触状态上安全地选择一个短期物理后果更好的候选，
而不是继续扩大普通 Cm 数据或把 Cm 接到 actor 特征。

固定协议：6 个显式 simulator seed；每个环境最多 4 个窗口；同一 native PD checkpoint、
6 个专家候选、base-hold、fixed Cup 共 8 个候选；触发后按 `p=1/8` 随机执行一个候选
1 tick，再执行 fixed Cup 9 ticks。Cm 只用于冻结的短期 consequence score、ensemble
standard deviation、retention/release 和 OOD gate。每个窗口在触发时保存当前 motion、
start frame、rest height、episode id；manifest 另外保存 simulator seed、panel seed 和
`PYTHONHASHSEED`。

固定 fallback：

```text
top_score - fixed_score > 2 * ensemble_std
candidate_ood = false
retention(top) >= retention(fixed) - .05
release(top) <= release(fixed) + .05
```

预注册门：每个 arm 至少 24 个窗口、12 个 episode；fallback 相对 fixed Cup 的 score
组 bootstrap lower90 `> 0`；last-3 contact 与 clearance lower90 各 `>= -.05`；改变
比例在 `[.05,.80]`。支持不足记 `UNCLEAR`，支持充分但任一效用或风险门失败记
`UNPROMISING`。只有所有门通过，才允许下一步固定的小型 option-policy 设计；本 Probe
本身不启动策略训练。

输出目录：
`src/task/CmResidual/research/contact_consequence/output/P-20261003-cm-uncertainty-gated-candidate-v2/`

## 结果

6 个 simulator seed（710–715）完成 252 个窗口，22 个 motion/start groups；8 个 arm
支持为 `[25,28,26,32,37,34,35,35]`，episode/window provenance 唯一，v2 seed 字段和
trigger metadata 均通过审计。支持门通过，原始 Cm selector 仍然是 `UNPROMISING`。

固定 uncertainty fallback 改变 119/252（47.2%）窗口，但相对 fixed Cup 的 score 组
bootstrap 为 `-10.12 mm`，90% 区间 `[-30.69,+8.61] mm`；last-3 contact 为
`+0.014`，区间 `[-0.281,+0.258]`，clearance 为 `+0.049`，区间
`[-0.213,+0.277]`。score 区间跨零且两个风险下界均低于 `-.05`，因此本 corrected
Probe 为 `UNPROMISING`，关闭该 fallback 配方，不启动 option-policy、PPO、阈值扫描或
普通 Cm 数据扩展。完整 v2 审计见
[`P-20261003-cm-uncertainty-gated-candidate-v2-results.json`](P-20261003-cm-uncertainty-gated-candidate-v2-results.json)。
