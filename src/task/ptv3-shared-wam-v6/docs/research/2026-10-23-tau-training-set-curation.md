# τ curated TRAIN 首批（2026-10-23）

这轮只做 CPU/IO 审计和 manifest 生成，没有转换新序列，也没有启动训练。最终产物是
[6,000-window curated manifest](../../../../../outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json)
和 [固定 TRAIN 锚定样本](../../../../../outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/train-anchor.json)。
6,000 个窗口来自已有的 1,177 条本地序列；因此相对于当前本地 cache，**新增序列数为 0**。
manifest 同时登记了完整配方中待转换的 6,000 条 sequence key。raw payload 在当前工作区
可解析，但本轮没有启动转换；状态是 `READY_FOR_BATCH_CONVERSION`，所以不能把这 6,000
条序列计入有效覆盖。

## brief19 失败和实际 verdict

brief19 的第一次工具在读取没有 `windows[split]` 的序列时直接索引字典，触发了
`candidate_windows` `KeyError`。这次不掩盖该失败：修复后的工具按默认文件名或声明路径
解析 sidecar，缺失/损坏 sidecar 单独计数，不会变成准入窗口。

修复后保留准备阶段的实际 verdict：

| split | candidate | admitted | rejected | rejection counts |
|---|---:|---:|---:|---|
| TRAIN | 33,726 | 33,672 | 54 | history 26；object pose 12；两者 16 |
| DEV | 3,386 | 3,356 | 30 | history 23；object pose 7 |

准入规则与物体身份无关。H4 规则要求至少一只手/一个关节在 `t-3..t` 全部有效，
含义是输入历史可以解释；TRAIN/DEV 分别拒绝 26/23 条 history 缺失窗口。K24 规则要求
选中 slot 的 object pose 在 `t-3..t+24` 有效，含义是物体目标和 24 步标签可追溯；
TRAIN/DEV 分别拒绝 12/7 条单独 pose 缺失，另有 TRAIN 16 条同时违反两条。窗口选择阶段
只按机制相位、source、物体几何尺寸分箱和运动范围分箱轮转；object name 不参与门槛、排序
或配额。proximity phase 是几何接近 sidecar，不是力接触真值。

## 6k 首批的覆盖

curated TRAIN 恰为 6,000 个 admitted windows，覆盖 1,064 条已有序列、9 个 source、
61 个 `(source, object)` 几何实例。以下把序列数和有效覆盖分开：

| 量 | curated TRAIN |
|---|---:|
| 新增序列数（相对现有 1,177） | **0** |
| manifest 中代表的已有序列数 | 1,064 |
| approach / close / contact / hold / unknown 窗口 | 1,200 / 1,200 / 1,800 / 1,740 / 60 |
| 各阶段代表的序列数 | 1,028 / 1,049 / 1,050 / 1,057 / 17 |
| 各阶段代表的几何实例数 | 50 / 53 / 53 / 55 / 12 |
| geometry extent 分箱（low / mid / high） | 1,940 / 2,017 / 2,043 |
| object translation range（min / median / max） | 0.00005 / 0.0497 / 1.1839 m |
| object rotation range（min / median / max） | 0.033 / 13.54 / 179.76° |
| hand motion range（min / median / max） | 0.0020 / 0.3473 / 1.5120 m |

近重复窗口比例为 61.23%（同一 sequence/slot 内起点相差小于 24 帧）。选择时优先做
24 帧间隔轮转，但可用的长序列仍留下近重复；这些窗口不能按 6,000 个独立机制样本
解释。`approach/close/contact/hold` 的阶段配额是机制覆盖配额，不是成功率或物体价值权重。

固定锚定样本是 512 个已有 TRAIN admitted rows，锚点 ID 为
`94d9287bb6c650cc80e4345515e0589472786bd749981a0b25ea7475118ebc37`。它按稳定的
`sequence_key/slot/tick` 保存，扩集前后不重抽；新旧 checkpoint 必须在同一修正后的评估
接口和这套锚定样本上比较。

完整配方的 6,000 条 pending sequence 已按 source 做了确定性批次清单，payload 可用但没有转换。用户给出的
线性估算是 23.0 GiB；只读 header inventory 给出的保守上界为 96.50 GiB。本次检查可用空间约
139.1 GiB，按保守上界估算后仍约 42.6 GiB，超过 20 GiB 停写线。该 preflight 证明空间足够；
它不把未转换 payload 计为有效覆盖。

## exposure 和后续判据

上一轮 flow matching 的已运行步数是 11,380，global batch 是 288，估算窗口曝光
`11,380 × 288 = 3,277,440`；在 33,672 个 TRAIN windows 上约 97.37 次/窗口。若保持
相同 updates/batch 而只训练这份 6,000-window panel，均匀曝光约 546.24 次/窗口。反过来，
如果未来转换使窗口池扩大约 5 倍，同一 updates/batch 下每个窗口曝光约降到原来的 1/5；
扩集后 TRAIN 拟合变差不能直接读成“数据无效”。

固定锚定面板上只报告两列事实：**τ 预测质量是否改善**、**成功率是否改善**。撤回上轮
“预测误差↓但成功率不动 ⇒ 瓶颈不在 WAM 预测质量”的因果句；这两列事实不能单独给出
瓶颈归因，归因留给后续判别实验。

| τ 预测质量是否改善 | 成功率是否改善 |
|---|---|
| 本轮未测（未启动训练） | 本轮未测（未启动闭环评估） |

| 新有效覆盖 | 固定 TRAIN 锚定样本预测质量 | 解释和下一步 |
|---|---|---|
| ↑ | ↑ | 训练集质量确实提升；扩集后再谈泛化 |
| ↑ | 不变 | 瓶颈更可能在优化/条件/标签，下一步查 reader 条件与标签 |
| 不变（近重复占多数） | 任意 | 1,177 条已有 cache 已被重复采样主导，先改采样/去重 |

v6 goal 契约已在本日另一提交中修复并通过 targeted/full contract tests：`object_goal_valid`
不再硬编码，训练与推理统一为 `spatial_path_waypoint`。但 scheduler-conditioned cache
identities 尚未重建，因此本 manifest 仍为 `training_allowed=false`，本轮不启动长训；待
cache identity gate 通过后，才能用固定锚定样本、同一修正评估接口和新的环境 receipt 比较
checkpoint。v2 manifest 已由 `curated_sampling.py` 兼容读取，但 sampler 只接受这一审计状态，
不会把它当作可训练 recipe。
