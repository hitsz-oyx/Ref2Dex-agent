# Ref10 engineering audit: EgoTouch pressure/contact 接入

日期：2026-10-11。Task：`cm-pointflow-effect-pretrain`。本审查只读当前 worktree、已存在的输出和源码；没有训练、下载或修改其他文件。

## 审查结论

ref10 的方向可以落地，但当前实现还没有走到 pressure-map 监督。现有 raw-sensor learner 读的是 `jq_pressure.json` 的左右各 256 个 byte 通道，拼成 512D 后除以固定 255；目标是未来 `24×512` raw sensor delta。它没有读取 `pressure_grids.npz`，也没有 contact head 或 point-cloud fusion。因此上一轮 raw sensor negative 不能评价 pressure/contact 目标。

已验证的十条 TRAIN recording 的压力、WiLoR hand 和 Vive 行号连续且一致；已经取得的 RGB 也与原始 frame IDs、30 Hz 相对 PTS 对齐。这只是文件级时钟一致性，不能推出传感器物理延迟、世界坐标或相机标定。WiLoR 文件中的 `aligned_to_vive=false`，手坐标和尺度不能直接作为机器人世界位置。

## 数据合同

固定数据版本：`zhouzhoujy/EgoTouch@cfdbb0ac31cc2af4247943820aa250575e7e6637`。

每条已核对 recording 的主要字段：

| 字段 | 实际形状/语义 | 可用于第一版 Probe |
|---|---|---|
| `jq_pressure.json` | 每帧 `frame_index, ts, sensor_left/right[256]`，byte 范围 0..255 | 只作 raw 对照；不能自行命名 tactile/bend 物理角色 |
| `pressure_grids.npz` | `left/right_pressure_grid[T,21,21]`，每手 217 个有限格，其余 NaN；另有 baseline/normalization 元数据 | 第一版直接作为 pressure target，保留 NaN mask |
| `wilor_hands.json` | `left/right_pos[T,21,3]`，行号与压力一致；部分 recording 有空/缺失 hand | 仅作 hand-local 条件；使用 finite mask，不能当 calibrated world pose |
| `vive_poses.json` | 与 pressure 行号和 ts 一致 | 目前只用于审计，不进入 target |
| `manual_contact_annotation.json` | 已搜索的 79 条 TRAIN 都是 false/false | 不能当全段 no-contact，也不能构造可靠 BCE 正负标签 |
| `chest.mp4` | 已审计样本 30 Hz，frame IDs 与 label 行数一致 | 可作为当前帧视觉条件；没有相机内外参，不能投影到 3-D 点云 |

十条 recording 的 raw/processed 输出位于
`outputs/cm-pointflow-effect-pretrain/egotouch-diverse-raw-20261010-r1/`；
两样本完整 schema、NaN mask、全记录 max 和候选映射复算位于
`outputs/cm-pointflow-effect-pretrain/egotouch-label-schema-20261010-r1/`
及
`docs/research/2026-10-10-egotouch-tactile-channel-contract.md`。

官方代码的 HDF5 converter 直接读取 `pressure_grids.npz`，因此第一版应沿用 processed grid，不从 256 raw 通道重新猜 mapping。公开实现和数据来源：
[TouchAnything converter](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/core/convert_to_hdf5.py)、
[EgoTouch fixed release](https://huggingface.co/datasets/zhouzhoujy/EgoTouch/tree/cfdbb0ac31cc2af4247943820aa250575e7e6637)。

## 当前可复用代码

- `tools/audit/audit_egotouch_label_schema.py`：读取原始行、检查 frame IDs/ts、hand/grid 形状、NaN mask 和元数据。
- `tools/run/prepare_egotouch_raw_sensors.py`：保留源行号和 timestamps，检查 raw byte 范围，生成 immutable `.npz`；但它只打包 raw sensor 与 hand，不打包 pressure grid。
- `src/oakink_wm/raw_sensors.py`：`sensor_window` 的 H4/K24 窗口、history-only hand scale、past-only 输入合同可复用；`SensorWindows` 当前强制 28 帧双手均有效。
- `tools/run/train_rawsensor_conditioning.py` 与 `train_rawsensor_visual_context.py`：可复用 matched arm、FIT-only 标准化、任务隔离和 persistence 基线；输出头目前固定为 `24×512`，需要新的 pressure head。
- `tests/test_raw_sensors.py`：已有 future sensor peak、future hand scale、缺 hand/非单调时钟测试，可作为新 dataset adapter 的合同模板。

## 发现的工程边界和潜在 bug

1. **目标错位。** 当前 learner 的 target 是 raw byte delta；它从未消费已经处理好的 21×21 pressure map。因此当前 `UNPROMISING` 只适用于 raw-sensor teacher recipe，不能关闭 ref10 pressure 路线。
2. **processed grid 不能按 441 个独立 taxel 计数。** 每手只有 217 个 finite cells，存在 raw index alias；右手还有 28 个空间修复格。target 必须带 `pressure_valid[T,2,21,21]`，loss 只能在 finite measured cells 上计算。
3. **未来归一化风险。** 两个样本的 grid 使用全 recording 的 shared `tactile_max/bend_max`，峰值出现在早于 history 的后续帧；将该 grid直接作为 history feature 会泄漏未来。第一版只把 grid 作为 future target，输入若包含 pressure 必须用 raw/固定 255 或 TRAIN-only 统计。不要按 recording 重新除以未来最大值。
4. **manual contact 不能做标签。** 当前 bounded search 的 false/false 可能是未标注或字段语义未知；不能以其生成 contact negative。可定义的第一版 contact 是 pressure-derived proxy（例如 measured cell 且 pressure>预注册阈值），必须报告它是软件派生 proxy，不是真实 3-D contact GT，并做阈值敏感性。
5. **hand validity 过严且会改变样本分布。** `prepare_egotouch_raw_sensors.py` 只保留 H4+K24 全双手 finite 的窗口；十条 recording 中 plush fit 只有 43 个 complete bimanual frames，因而零窗口。这适合当前 matched screen，却不应成为 pressure-only adapter 的硬要求。pressure target 可以先逐手训练，hand 条件按 side mask；否则大量真实触觉帧被静默丢弃。
6. **手坐标不是世界坐标。** `sensor_window` 使用每个窗口前 4 帧的 wrist-relative shape 和 history median palm scale，这个 past-only 处理是安全的，但丢弃了 wrist transport；不能把它和 Isaac 的 hand/world pose 直接拼接。
7. **当前候选 normalization group 未验证为 tactile/bend。** `train_rawsensor_conditioning.py` 依赖 mapping index 集合给 group A/B 打分；研究笔记已明确这些只是处理候选，不能把结果写成接触或弯曲通道收益。
8. **当前 RGB 对齐仍非物理同步。** 行号、PTS、30 Hz 一致只证明文件布局；没有 camera calibration，也没有 hand-to-camera extrinsics。不要把 RGB feature 投影成 3-D hand/object point。
9. **当前测试覆盖不到 pressure adapter。** 测试只覆盖 raw `sensor_window` 的未来隔离和缺手拒绝；没有 grid NaN loss、mask、threshold、global-max 泄漏或 side-specific missing-hand 行为测试。

## 最小可判别 Probe：pressure + contact target

目标是回答一个窄问题：在相同当前 hand/history 条件下，pressure-map future target 是否比 persistence 提供可学习且跨任务的 interaction signal。它不声称 calibrated force、3-D contact wrench 或机器人策略收益。

### 输入与窗口

- 采用已下载十条 recording，保留原 frame IDs 和 timestamps；先固定 H=4、K=8（约 0.27 s），避免把 K=24 误认为 pressure 标签足够长。
- 以每手独立窗口建样本；pressure target 允许单手有效，hand condition 使用 `hand_valid[T,side]`。不要要求双手共同 finite 才能保留窗口。
- 当前输入只用 past hand-local shape（可加当前 RGB feature 的单独 matched arm）；不把未来 hand 或未来 pressure 放入输入。
- 将 `pressure_grids.npz` 复制到 prepared pack，字段至少为：
  `pressure[T,2,21,21]`、`pressure_valid[T,2,21,21]`、`frame_index[T]`、`timestamps[T]`、`hand[T,2,21,3]`、`hand_valid[T,2]`、源文件 hashes、baseline/normalization 元数据。

### Target 和 loss

- pressure regression：只对 `pressure_valid` 的 cells 计算 Huber/L1；NaN 在 arithmetic 前替换为 0，并始终由 mask 排除。
- contact proxy：`contact_proxy = pressure_valid & (pressure >= tau)`，`tau` 在协议中预先固定（建议先扫 `tau=0` 和一个 TRAIN-only quantile，不能按每条 recording 的未来值选择）。它是 grid-level activation proxy，不是 manual/contact GT。
- 用两个 head：BCE/logit head 预测 proxy，pressure head 预测 ([0,1]) grid；pressure loss 可按 proxy 或 valid mask 加权，但要同时报告 all-valid 和 proxy-positive 两种误差。
- baseline：persistence、history-only MLP、history+hand MLP；所有 arm 共享窗口、mask、target 和随机种子。主要指标按 recording/task macro MAE、masked RMSE、proxy F1/PR-AUC（若 proxy prevalence 足够）报告。单次 fit gain 不升级为结论。

### 预先判定

Probe 仅形成 PROMISING/UNPROMISING/UNCLEAR：

- PROMISING：history+hand 在至少 3 个 held tasks 上相对 persistence 的 masked pressure h8 MAE 改善 >=10%，并且 proxy F1/PR-AUC 不劣于 persistence；不能有一个 held task 恶化超过 20%。
- 若 proxy 极度稀疏、所有 recording 没有足够 positive cells，判 UNCLEAR，不能改写成负结果。
- 若 pressure target 能学而 proxy 不稳定，只保留 pressure auxiliary，不接触 BCE。
- 只有 Probe 正向后，才设计与 native point encoder 的 fusion arm；先不把 EgoTouch pressure head并入最终 policy。

## Point-cloud fusion 的边界与最小接法

EgoTouch 当前没有经过审计的 object point tracks、camera intrinsics/extrinsics 或 world transform，所以不能把其 pressure map 直接对齐到 OakInk/Isaac 物体点云。最小可判别 fusion 应分两步：

1. **先做 hand-surface token fusion。** 将每手 21×21 pressure grid 编成固定 masked CNN/MLP token，和 21 个 hand-joint token 或当前 native hand/action token 做 gated cross-attention；只在 EgoTouch 内测 pressure prediction，不宣称世界点云。
2. **再做 robot matched fusion。** 在已有 native OakInk/GRAB/ARCTIC 数据中，使用真实 calibrated hand/object points 和 contact/effect target；把同维度 tactile token 接入 native scene token，比较 Cm on/off 的 matched trained-policy gate。EgoTouch 只能作为共享 pressure encoder 的辅助预训练来源，不能提供这条 world-point 对齐监督。

若需要一个最小接口，建议：
`PressureWindow.__getitem__ -> {hand_history, pressure_target, pressure_valid, contact_proxy, elapsed, frame_ids, side_valid}`；
`PressureHead.forward -> {pressure_logits[B,K,2,21,21], pressure_value[B,K,2,21,21]}`；
`masked_pressure_contact_loss(pred,target,mask)` 必须拒绝没有有效 cell 的样本，并不能读取任何 future normalization。

## 下一步顺序

1. 先写 pressure-only prepared adapter 和 mask/NaN/future-max tests；不改 raw pipeline 的既有结果。
2. 用十条已有本地 recording 做 H4/K8 的 persistence 对照与一个小 history+hand arm；记录 GPU/时间，保持 fit/held task 隔离。
3. 只有 pressure Probe 正向且 mask/clock 合同通过，才实现 hand-surface token fusion；native point-cloud fusion 另起 matched synthetic/robot experiment。
4. CHORD wrench 只在 native calibrated robot 数据上加入预测 head；不能从 EgoTouch pressure grid 推导 Newton wrench。

## 证据位置

- [EgoTouch schema probe](../experiments/probes/P-20261010-egotouch-label-schema.md)
- [raw sensor conditioning probe](../experiments/probes/P-20261010-rawsensor-hand-conditioning.md)
- [tactile channel contract](2026-10-10-egotouch-tactile-channel-contract.md)
- [raw sensor implementation](../../src/oakink_wm/raw_sensors.py)
- [raw preparation](../../tools/run/prepare_egotouch_raw_sensors.py)
- [raw tests](../../tests/test_raw_sensors.py)
