# Decision Memo：重采 uncertainty-gated candidate consequence Probe

## 当前需要决定的问题

旧的 candidate-advantage 结果显示，固定的 uncertainty fallback 可能把 Cm 的短期
物理后果转成相对固定 Cup 的动作选择收益，但旧采集器在 reset 后复用了 motion/start
元数据，也没有把真实 simulator seed 与 panel seed 分开记录。因此旧的正向信号只能作为
路线线索，不能直接进入策略训练。

## 关键证据

冻结 Cm 的旧离线重放中，规则

```text
top_score - fixed_score > 2 * ensemble_std
candidate_ood = false
retention(top) >= retention(fixed) - .05
release(top) <= release(fixed) + .05
```

在 287 个窗口上改变 143 个动作，score 相对 fixed Cup 的组 bootstrap 90% 下界为
`+6.98 mm`；但该数据属于 v1 provenance，不能作为 fresh native 证据。原始 Cm
selector 本身仍冻结为失败，不能借此扩大数据或启动 PPO。

## 选择的行动

修正采集器和审计器后，运行一个新的 `P-20261003-cm-uncertainty-gated-candidate-v2`
Probe。保持冻结的 native PD checkpoint、8 个实际候选、观察后均匀 `p=1/8` 分配、
one-tick candidate 加 nine-tick fixed Cup continuation，只改变输出目录、显式 simulator
seed 和 provenance 合同。使用固定的 6 个 simulator seed，不扫描 sigma、阈值、候选集或
checkpoint；Cm 仍只预测短期物理后果，未训练 actor、success classifier 或 PPO。

## 判定与停止条件

每个候选至少 24 个窗口、至少 12 个唯一 episode，且每个窗口 id 唯一。fallback 的
score 相对 fixed Cup 的组 bootstrap lower90 必须大于 0；last-3 contact 和 clearance
的 lower90 不得低于 `-0.05`；动作改变比例必须在 `[.05,.80]`。任一支持或风险门失败，
关闭该 fallback 配方，不继续调阈值或增加普通 Cm 数据。全部通过只授权设计一个小型
categorical option policy Probe，仍需另行固定 Cm-on/off matched 训练合同。

## 成本和边界

最多 6 个单 GPU native 运行，每次 240 秒墙钟，使用现有 checkpoint 和 motion 数据，
不覆盖旧目录、不修改外部 baseline。该行动不改变 MISSION 的核心问题或最终 claim。

## 处置

Probe 已完成。252 个窗口满足支持门，但 fallback score lower90 为 `-30.69 mm`，
contact/clearance 风险门也失败，因此结果为 `UNPROMISING`。关闭该 fallback 配方；不启动
小型 option policy，不扫描 2-sigma 规则，也不扩大普通 Cm 数据。v1 的离线正向信号保留
为已撤回的路线线索，不能与本 v2 native 结果合并成正向证据。
