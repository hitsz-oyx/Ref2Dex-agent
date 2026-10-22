# τ 训练集质量与 curated manifest（2026-10-23）

## 结果解读表（先看这个）

| 同一 held-out DEV 集（3,356 窗口） | 当前可比基线 | 数据质量版训练后的判据 | 对下一步的决定 |
|---|---:|---|---|
| τ 预测误差：hand/object EPE，Oracle=t+24 终点 | healthy4371：15.797 / 14.722 mm；已有 flow8500：17.374 / 16.206 mm | hand 与 object 都下降至少 5% 才算预测改善 | 只改善预测而不改善长时程漂移，不能宣称 τ 更可靠；检查 goal/时序契约和 rollout |
| 长时程漂移：last-8 EPE；同一 DEV、同一 goal 条件 | healthy4371：96.409 / 28.203 mm；flow8500：101.245 / 31.933 mm | last-8 hand/object 都下降至少 5% 才算稳定性改善 | 改善说明数据/采样减少耗散或漂移，可继续同一结构；仅短期下降则不改变路线 |
| 两项都不改善 | 尚无修契约后的新权重 | 视为数据清洗不足或不是当前瓶颈 | 不扩大训练；先查条件语义、配对和长时程闭环 |

上表的 healthy4371/flow8500 数字来自既有同一 DEV goal-chain 审计；本轮没有训练新权重，因此**没有把任何指标改善归因于 curated manifest**。`last-8` 是 24 步预测的末 8 步漂移代理，不是物理耗散定理。完整结果解读必须在 goal 契约修好后，用同一 held-out 集、同一随机种子和同一评估代码重跑。

产物：[curated manifest（11 MB，37,112 个候选窗口，含每行准入理由和阶段）](../../../../../outputs/ptv3-shared-wam-v6/tau-curated-20261023-r1/curated-manifest.json)，生成器 [`curate_tau_training_set.py`](../../tools/audit/curate_tau_training_set.py)。

## 1. 来源、权重和阶段覆盖

本地准备的是工程子集，不是九源全量：1,177 条序列、360,671 帧，其中 GRAB 1,132 条（96.175%），其余八源合计45条；仍有38,349条计划序列没有准备。按当前 TRAIN 窗口，GRAB 是 33,272/33,672 = **98.812%**。这是“选择规则把所有计划 GRAB 都取入、其他源每个 split 最多三条完整序列”的直接结果，属于工程子集特性；作为 τ 主训练分布则是缺陷，因为模型几乎只看 GRAB 的时序和尺度。

| source | TRAIN 窗口（占比） | TRAIN 序列/物体 | DEV 窗口（占比） | DEV 序列/物体 |
|---|---:|---:|---:|---:|
| OakInk2 | 96 (0.285%) | 3 / 3 | 96 (2.861%) | 3 / 4 |
| GRAB | **33,272 (98.812%)** | 1,040 / 41 | 2,944 (87.723%) | 92 / 4 |
| ARCTIC | 96 (0.285%) | 3 / 5 | 96 (2.861%) | 3 / 6 |
| Shadow/UniDex | 30 (0.089%) | 3 / 1 | 30 (0.894%) | 3 / 1 |
| TACO | 18 (0.053%) | 3 / 2 | 21 (0.626%) | 3 / 1 |
| HOT3D | 66 (0.196%) | 3 / 3 | 86 (2.563%) | 3 / 2 |
| DexYCB | 12 (0.036%) | 3 / 2 | 0 (0%) | 0 / 0 |
| H2O | 4 (0.012%) | 3 / 2 | 7 (0.209%) | 3 / 2 |
| HOI4D | 78 (0.232%) | 3 / 2 | 76 (2.265%) | 3 / 2 |

当前阶段审计使用冻结 proximity sidecar：至少3个 query 的置信度阈值、`.8/.5` 进入/离开滞回、3 帧持续性；它是几何接近标签，不是力接触或抓稳真值。`close` 表示远离状态但 24 步内有 proximity 进入事件，`contact` 表示接近状态且 24 步内有离开事件，`hold` 表示接近且没有 24 步内离开，`approach` 表示远离且没有近端进入事件；事件/状态未知单独列出。

| 阶段 | TRAIN 窗口 / 占比 / 物体数 | DEV 窗口 / 占比 / 物体数 |
|---|---:|---:|
| approach | 1,388 / **4.122%** / 50 | 232 / **6.913%** / 14 |
| close | 2,813 / 8.354% / 53 | 283 / 8.433% / 16 |
| contact | 7,628 / 22.654% / 53 | 708 / 21.097% / 15 |
| hold | 21,751 / 64.597% / 55 | 2,048 / 61.025% / 20 |
| unknown | 92 / 0.273% / 12 | 85 / 2.533% / 8 |

严重缺失的是 **approach**：TRAIN 只有 4.12%，而 hold 占 64.60%；contact 本身并不缺。按软接触目标（未来 24 帧至少3个 query 达到 `.25`）统计，TRAIN 有接触目标 32,391（96.19%）、无接触 1,281（3.81%）；DEV 为 3,124（93.08%）和232（6.92%）。因此不能用“删掉无接触”补救，正确动作是身份无关的阶段采样和补充 far-to-contact 覆盖。

## 2. 窗口准入和退化样本

当前工程候选为 TRAIN 33,726、DEV 3,386；最终准入为 TRAIN 33,672、DEV 3,356。规则只有两项：

1. H4（`t-3..t`）至少一只手/一个关节在四帧均有效；
2. 选中物体槽位在 `t-3..t+24` 的 `object_pose_valid` 全部有效。

拒绝原因如下，分母是候选窗口：

| split | 候选→准入 | history 缺失 | object pose 无效 | 两者同时 |
|---|---:|---:|---:|---:|
| TRAIN | 33,726→33,672（拒54，0.160%） | 26（48.15% of rejected） | 12（22.22%） | 16（29.63%） |
| DEV | 3,386→3,356（拒30，0.890%） | 23（76.67%） | 7（23.33%） | 0 |

这条准入**没有**显式遮挡/可见性规则，也没有把 contact 缺失、目标越界或静止帧作为拒绝条件：

- 未来 contact target 无效 mask 仍留在 TRAIN 78 窗口（0.232%）和 DEV 78 窗口（2.324%）；历史 surface 无效分别留在 120（0.356%）和106（3.159%）。这是 mask 语义，不是伪造负接触。
- 无软接触目标的 1,281/232 窗口仍被保留为 approach/unknown 候选。
- “物体和手在 K24 内都小于2 mm 位移”的静止诊断只有 TRAIN 4、DEV 0；没有静止帧洪泛问题，但 hold 的高比例是真实的长保持/重复观测，不能为了指标删掉。
- 独立完整性审计发现 3,306 个无效物体位姿帧，按 `object_pose_valid=false` mask；它们没有进入上述准入窗口。

## 3. 配对、坐标、尺度和时序

手/物配对由 `hand_meta_identity`、`contact_meta_identity`、geometry manifest identity 绑定；独立重算 56,054 个 contact query，最大置信度差 **0.0002438815**，原始 11 点到重建 21 点最大对齐误差 **4.82e-6 m**。因此当前已知的主要噪声源不是随机错配，而是：

- 不同来源的坐标/尺度和时间合同不同，统一到当前物体锚点与 30 Hz H4/K24 后才可比较；
- HOI4D/source10 的奇数中间帧是合成时间点，contact supervision 按 mask 处理，不能当真实观测；
- 缺失 surface、joint 或 object pose 只由 validity mask 表示，不能解释成“没有接触”；
- 几何 proximity 不是测力接触，不能直接支持抓稳或耗散结论。

重复审计发现 exact `(sequence, slot, tick)` 重复为 **0**；TRAIN/DEV 序列交集为 **0**。但 K24 需要的时间重叠很高：TRAIN 有 140,160 对 `Δtick<24` 的近重复窗口，涉及 33,352/33,672（99.05%）窗口；DEV 为 13,040 对，涉及 3,182/3,356（94.82%）。这是有意的时序滑窗重叠，不是复制文件；评估时应按序列/捕获分组 bootstrap，不能把窗口数当独立样本数。

## 4. goal 语义修复、重算产物和代价

当前训练 reader 在 `sample_window` 中构造 `goal_object_delta = pose(t+24)-pose(t)`，即**实现出来的 t+24 终点**；部署 scheduler 当前发出的是 `spatial_path_waypoint`，没有到达时间保证。两者是不同随机变量，不能仅因为都是 32×3 就直接复用。

goal 合约修好后，必须新版本化并重新落盘/绑定：

1. TRAIN/DEV 每个 admitted window 的 scheduler-style goal、goal validity 和 event/task condition sidecar（37,028 个窗口）；
2. recipe manifest、TRAIN/DEV index/strata identities、cache bindings；
3. 只用 TRAIN 重算的 normalizer JSON；
4. 修复后的 train/inference contract、deployment adapter 和 contract tests；
5. 同一 held-out 面板的 baseline receipt，以及新语义的评估输入。

当前 6.03 GB 手/接触/事件 cache 不应覆盖。若只增加紧凑的 32×3 float32 goal sidecar，理论 payload 约 **13.6 MiB**，连同 validity、事件和 JSON 余量按 **<100 MiB** 估价；真正成本是一次 360,671 帧/37,028 窗口的 CPU/IO 重建和新的 provenance。若 scheduler goal 需要重放完整路径，按 1,177 条缓存逐序列重放，不能把旧 t+24 goal 数组冒充修复结果。旧 checkpoint 只能做 weights-only 初始化，不能 strict-resume 成“同分布继续训练”。

## 5. 不特化的训练配方

只改数据和采样，结构与损失保持 v6：joint21、H4、K24、surface778/object2048、strict FP32，以及原 trajectory/scene/contact/event losses。建议的新 sampler 是机制性规则：

- 先按九个 source ID 等概率抽 source；source 内按 admitted sequence 等概率抽序列；再在该序列内按四个阶段等概率抽可用阶段，阶段缺失时只在该序列已有阶段内归一化；
- 每个 batch 对同一序列设置固定上限，避免 1,132 条 GRAB 序列和少量其他源之间靠重复窗口形成隐含 object 权重；
- 不读取或比较 object name，不为那20个新物体设阈值，不按物体成功率删样本；所有门仍只有 H4 validity、future object validity 和显式 masks；
- DEV 保持序列不相交，且不重新拟合 normalizer；重叠窗口只作为同一 capture 的相关观测处理。

这会降低 GRAB 的采样占比、提高 approach 的可见度，但不会制造缺失的物理接触，也不会把本地子集升级成全量泛化证据。

## 6. 当前停止条件和复现

当前 `/home2` 约 141 GiB 可用；1,177 序列 cache 为 5.62 GiB，注册的 38,349 条未准备序列线性估计还需约177.4 GiB，并且要保留 20 GiB 停写线，因此不能在本机复制全量。goal 契约、`object_goal_valid` 和 train/inference 语义未修复前，训练保持 HOLD；不碰 `wam-rl-policy-v3/**`，不使用 GPU4/5/6/7。

生成审计 manifest（只读、约32秒 CPU、无 GPU、无写入源数据）：

```bash
cd /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent
python3 src/task/ptv3-shared-wam-v6/tools/audit/curate_tau_training_set.py \
  --output outputs/ptv3-shared-wam-v6/tau-curated-20261023-r1/curated-manifest.json
```

goal 修复并通过 contract tests 后，才允许建立新的 `P-20261023-tau-data-quality` run manifest；届时使用 GPU4/5/6/7，启动前检查四卡利用率、显存、磁盘和预计完成时间。新训练必须保存 environment、recipe identity、normalizer identity、GPU telemetry 和同一 DEV 的两项指标。
