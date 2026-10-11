# Ref11 对照：之前尝试为什么没有成功

日期：2026-10-11。本文把用户提供的 `docs/user/ref/ref11.md`（PointWAM 的人类视频到共同 3D 场景/手轨迹预训练思路）与本 Task 已完成的 Probe 对齐。它是归因记录，不把 PointWAM 论文结果当成本地复现结果，也不修改用户提供的 ref 文件。

## 先给结论

本地失败的共同原因不是 PTv3、优化器或“视频不能学动力学”。原生 OakInk2/GRAB/ARCTIC 合同下，模型能学到动作相关效果；失败集中在外部数据进入模型之前：我们要求外部数据直接满足原生动作条件/刚体效果合同，或者只保留了 scene head，却没有构造 PointWAM 所需的共同 `scene + hand trajectory` 监督。

PointWAM 的关键顺序是：

```text
human video -> metric 3D current scene + current semantic hand
             -> future scene displacement + future hand displacement
             -> shared representation pretraining
robot demos  -> same representation + robot action retargeter
```

我们此前多数尝试实际做成了：

```text
video/sensor -> native H=4,K=24 or raw sensor forecast
             -> immediately compare native effect EPE/persistence
```

这两个目标不等价。

## 逐项归因

| 之前的尝试 | 实际合同 | 失败证据 | 归因 |
|---|---|---|---|
| EPIC/ObjectForesight 手+场景 | 想直接凑原生 `H=4,K=24` 的手/场景窗口 | 保守合同下 95 个候选 starts 没有一个合格的 H4+K24 手窗口；放宽为 point-only 后只有 78/71 个重叠窗口，只有 2 个 dev clip 达到 h8 支持，模型 h8 object macro `24.808 mm`，static `9.464 mm` | 主要是标签覆盖、点身份和几何质量不足；不是对视频预训练本身的否定。公开生成链还使用整段视频，不能宣称因果输入 |
| DROID/PointWorld H5 直接接原生模型 | 把 11 个时间点（1 context+10 future）压到原生 4 history+24 future；缺 verified point-kind、跨 clip identity、双手 semantic keypoints/action | 直接 native adapter 审计判为 incompatible；每个 clip 最多只能提供 7 个 future 点，且没有原生 28-step 合同 | 数据合同不兼容，训练前就应停止 |
| 官方 H5 `K_aux=10` scene-only Probe | 只用颜色/法线（或逐点广播的 context joint/gripper state）预测 scene flow；随机初始化、8 fit+2 held clips、500 updates | color/normal H10 `2.263 mm` 对 static `1.878 mm`；state H10 `2.183 mm` 对 `1.878 mm` | 这是 one-context、scene-only、单 episode 的小 Probe；没有 hand branch、没有共同 scene/hand loss，证据不足以测试 PointWAM 方法 |
| 官方 checkpoint transfer | 加载 PointWorld 官方权重，native 数据仍是 OakInk2/GRAB/ARCTIC | 实验卡已标为 `INVALID_IMPLEMENTATION / INCONCLUSIVE`；官方视频数据根本没进入训练；且源 checkpoint 是 31-D/15-mm/patch256，native 是 18-D/10-mm/patch128+temporal pooling | 测错了干预；shape-compatible 不等于 feature-semantic-compatible |
| DynamicWAM-style motion token | 在弱视频 point-only PTv3 上加一个全局历史 token | 只有两个合格 dev clip；token `13.726 mm`，matched control `12.949 mm` | 只测试了额外 token，仍没有 PointWAM 的当前手点、未来手 head、scene/hand 联合训练；仅关闭当前配方 |
| EgoTouch raw sensor | 预测未来 `24x512` raw bytes | held task macro actual future hand `4.607`，persistence `3.435` | 目标错了：这不是 pressure map/contact，也不是 scene/hand trajectory pretraining |
| EgoTouch processed pressure | 已补测 masked `(21,21)` pressure，H4/K8，past-only normalization | history/past-hand/future-hand arms 均在五个 held tasks 输给 persistence | 关闭的是“pressure future forecast/teacher recipe”；仍不是 PointWAM scene+hand 轨迹路线，也不是 robot action utility |
| RBO/CHORD/T-Rex | 分别是 measured FT、simulator contact wrench、per-fingertip F6 | RBO adapter 对齐 `PROMISING`，但单序列 forecast 输 persistence；CHORD 当前只有 simulator contact/wrench wiring smoke | 这些是独立的 calibrated interaction/wrench 路线，不能代替共同 scene/hand 预训练 |

## 已经被排除的错误归因

1. **不是原生模型完全不会用动作。** 原生三源训练保留了有效 checkpoint；早期 matched temporal Probe 在 6000 updates 的 validation 中，`H+A` h24 为 `15.63 mm`，`H` 为 `21.81 mm`，打乱动作为 `21.90 mm`。这不是最终 policy 结论，但说明 native action/effect 链路可学习。
2. **不是所有失败都来自代码 bug。** 弱视频 Probe 的独立复核没有发现确定的 indexing、future-input leakage、loss、checkpoint identity 错误；失败主要落在 label quality、support 和 source shift。早期若干启动错误已单独保留，不能混进科学负结果。
3. **不是“用了官方视频权重所以没用”。** checkpoint-transfer 那次确实测错了；后续 H5 data-only Probe 是从随机初始化开始，而且明确 `official_checkpoint_loaded=false`。它的负结果只限于短、单 episode、scene-only 配方。
4. **不是 raw EgoTouch 结果证明触觉无效。** ref10 已明确 raw bytes 与 processed pressure/contact map 是不同目标；processed pressure 的负结果也只关闭当前的 causal pressure forecast recipe。

## 与 PointWAM 的真正差距

### 1. 手轨迹不是可选附加输入

PointWAM 用 EgoDex/VITRA 的 metric 3D hand annotation/reconstruction，把 human hand 和 robot hand 都压到少量 semantic keypoints，再同时监督 future scene 和 future hand displacement。我们的 H5 适配没有 hand keypoints；把 7-D `joint_positions` 或 7-D `gripper_pose` 广播到每个 scene point，只是全局状态 feature，不会产生 hand surface geometry、手-场景共同运动或 future hand target。

### 2. 深度尺度和共同坐标没有被我们建立

PointWAM 用可投影的 3D hand 来校准 monocular depth scale，再用 intrinsics/camera pose 把 CoTracker tracks lift 到 metric world。我们的弱视频受 offline depth/camera contract、duplicate query、初始 object coverage 和跨 clip identity 限制；DROID H5 虽然有 metric scene flow，却没有与 native semantic hand/action 对齐的共同表示。

### 3. 预训练阶段和机器人阶段被混在了一起

PointWAM human stage **没有 action label**；action retargeter 只在 robot demos 阶段从头学习。我们曾把 human/video 数据直接塞到 action-conditioned native effect head，或用 future human hand 当作 action proxy。后者可作离线 oracle 对照，不能成为可控 robot action，也不能替代 robot-stage retargeter。

### 4. 规模与评估阶段不相称

PointWAM 论文描述的是约 1.15M human episodes、标准化 31-frame windows、60k human pretraining steps，再用 robot demonstrations fine-tune。我们已跑的 video data probes 主要是一个 DROID episode 的 8/2 clips、500 updates，或几个弱视频 clip；这些 Probe 适合判断数据链路是否通，不能承担同等能力的结论。

## 当前最准确的结论

- 已关闭：当前 EPIC/ObjectForesight weak-label point-only 配方、DROID H5 `H=4,K=24` 直接 adapter、官方 H5 color/normal 或 state-only 500-step scene auxiliary、当前 DynamicWAM-style token、当前 EgoTouch pressure forecast。
- 尚未关闭：有合格 metric hand + scene trajectory 的 PointWAM-style representation pretraining；跨多 interaction 的 RBO wrench auxiliary；T-Rex per-fingertip F6；simulator-native CHORD wrench/reward；以及任何 matched native Cm-on/Cm-off policy comparison。
- 因此“之前没成功”的最简解释是：**我们没有真正运行 PointWAM 的核心数据合同，而是在若干不完整的外部合同上直接做 native effect 预测或短期 persistence 比较。** 这些负结果是配方/输入边界证据，不是对 PointWAM 思路或视频数据的总否定。

## 证据索引

- [PointWAM 方法核查](2026-10-10-pointwam-human-video-usage.md)
- [EPIC/ObjectForesight readiness](../experiments/probes/P-20261010-video-data-readiness.md)
- [弱视频 point dynamics](../experiments/probes/P-20261010-video-point-dynamics.md)
- [DROID native adapter](../experiments/probes/P-20261010-droid100-native-adapter-contract.md)
- [DROID H5 auxiliary](../experiments/probes/P-20261011-official-video-aux.md)
- [DROID state auxiliary](../experiments/probes/P-20261011-official-video-state-aux.md)
- [误测的 checkpoint transfer](../experiments/probes/P-20261010-open-video-backbone-transfer.md)
- [EgoTouch pressure audit](2026-10-11-ref10-engineering-audit.md)
- [native shared-stats baseline](../experiments/probes/P-20261009-pointworld-main3-sharedstats.md)
