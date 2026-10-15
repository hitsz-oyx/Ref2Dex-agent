# 九源数据独立复审：HOI4D 插值边界漏筛

2026-10-15，工程审计。用户要求复查已有九源审计后是否仍有异常。只读检查，不占 GPU，不改变原始数据、recipe、统计、正在训练的模型或 Viser。原始复核代码位于项目 `tmp/nine-source-reaudit-20261015-r1/`，完整证据位于 `outputs/ptv3-shared-wam-v3/nine-source-reaudit-20261015-r1/`。

## 结论与影响范围

**确认一个局部准入实现缺口：旧 strict 和新 expanded 配方各有 8 个 HOI4D TRAIN 窗，包含超过已授权 5 m/s 上限的插值手部半步，并实际参与监督。** 最大 8.110 m/s；DEV 未发现这种正时段漏筛。旧版/新版对应相同 payload、tick，仅 row 和全局 index 因 DexYCB 扩产而改变。

这 8 窗占 HOI4D 的 55,343 个 TRAIN 窗约 0.0145%；按当前来源采样权重，其单样本总概率约 8e-6，global512、80k 更新期望约 327 次抽中（旧 strict）。这不是已经发生次数的日志复算，也不能据此解释整体模型效果。它违反了既定保守质量准入，不等于独立证明动作物理上绝不可能。

本地训练状态文档和远端既有启动回执绑定旧 `nine-source-strict-quality-20261015-r1`；扩充 `nine-source-expanded-dexycb-20261015-r1` 是独立候选配方，没有在本次审计中替换运行输入。本次未实时查询远端最新步数。

## 可复核的真实训练窗口

下表为 **旧 strict** 的 row 和未过滤全局 index；它们均经过实际 `NineWindows.__getitem__` 重放，`blocked=false`、存在于 source10 的 TRAIN sampling pool，异常半步对应两个标签的 validity 均为 true。

| strict row / expanded row | tick | strict global index | 原生 15Hz 帧 | 30Hz 异常半步 | 原生 / 半步速度 m/s |
|---|---:|---:|---|---|---:|
| 36531 / 36431 | 283 | 4758893 | 153→154 | 306→307 | 8.110228 / 8.110228 |
| 36840 / 36740 | 555 | 4772888 | 289→290 | 578→579 | 7.250609 / 7.250609 |
| 37583 / 37483 | 251 | 4805711 | 137→138 | 274→275 | 6.147056 / 6.147054 |
| 37032 / 36932 | 227 | 4781387 | 125→126 | 250→251 | 6.065583 / 6.065582 |
| 37274 / 37174 | 291 | 4792411 | 157→158 | 314→315 | 5.993356 / 5.993358 |
| 37568 / 37468 | 299 | 4805276 | 161→162 | 322→323 | 5.805116 / 5.805112 |
| 36923 / 36823 | 267 | 4777062 | 145→146 | 290→291 | 5.742624 / 5.742623 |
| 36558 / 36458 | 259 | 4759813 | 141→142 | 282→283 | 5.649513 / 5.649513 |

第一例序列为 `ZY20210800001/H1/C20/N20/S381/s01/T1`，payload `outputs/3d-data-preparation/hoi4d-resampled30-20261015-r1/sequences/sequence-00231.npz`。窗口支持为 30Hz 帧 280..307；原生坏转换是 306..308，但插值帧 307 进入 forecast 第 24 帧。实际 `trajectory_target[23]-trajectory_target[22]` 的范数乘 30 为 8.110228 m/s。对应左侧槽 1、点 8，两个 future label 与当前起点均有效。

全部序列名、旧/新 global index、点坐标、validity、窗口和原生 frame position 保存在 `real-reader-halfstep-audit.json`；其 `counts` 与 `actually_supervised_counts` 都是旧/新各 8。

root 随后独立读取这 8 个 payload，重新计算窗口末两帧速度及有效标签支持，得到 `PASS_CONFIRMED_8_LEAKED_WINDOWS`，证据 `root-halfstep-recheck.json`；不是仅转述本报告或以测试通过替代数据语义核对。root 已接手候选筛选修复及版本化新配方准备，正在运行的旧输入仍保持冻结。

## 根因与修复边界

`tools/audit/audit_motion_sources.py` 以 `covered[start] & covered[end]` 统计 retained context；`tools/audit/prepare_strict_motion_admission.py` 又只按该统计的峰值选择需要重放的候选序列。上述第一例 `covered[306]=true`、`covered[308]=false`，因此整段 15Hz 转换未进入候选峰值；但 306→307 的插值半段已在实际标签内。筛选器不是把速度算错了，而是候选选择未覆盖所有影响窗口的插值依赖。

最小修复建议：以原始 `full_capture` 超阈值选择候选，再运行已有 `bad[lo:hi+1]` 和 H4/K24 重叠规则。这样继续用真实 15Hz 观测判断异常，不把插值当新观测，并保守覆盖边界。

不能直接把现有 strict recipe 当新 base 输入原准备工具、然后覆盖 `quality_admission`：该工具的 offset 来自 `Catalog` 已过滤后的索引，descriptor 却仍绑定原始 base identity，存在索引漂移风险。应从原始 manual 父配方重建，或在原始全局 index 空间合并原排除和新增排除，创建新版本。对正在运行的 subset-resume 保持数值 normalizer 的兼容边界；对独立新阶段如重新拟合统计，必须同步新准入集合。已有 checkpoint 曾经接触旧数据的事实不会因后来筛掉数据而消失。

最小回归用例应包含：

```python
# 15Hz observed306、308；307是二者线性插值，且手点都有效。
# 在306→308放置>5m/s位移；唯一候选窗口tick283，支持280..307。
assert covered[306] and not covered[308]
assert label_speed_30hz(306, 307) > 5
assert candidate_selected_from_full_capture()
assert tick283 not in newly_permitted_windows
# matched control: <5m/s时保留；阈值恰好5时按严格>处理。
```

## 其他检查结果

- 原始完整序列检查 40,161 条，4 CPU workers，138.5 秒；0 文件/有限性/时钟递增/SO(3) 错误，0 native 准入窗口跨 segment、缺目标 future 或无合格历史手的错误。此数量是 expanded `Catalog` 当前提供的序列，已经没有窗口且无 motion-quality 标记的元数据行不在该遍历内；它们没有训练曝光。不是声称检查了所有隔离历史或 TEST 原始文件。
- 共 45 条、86 窗与异常转换端点相碰。其中 8 条/8 TRAIN 窗为上述真实正时段重叠；另外 37 条/78 窗仅碰单端点，没有异常转换时段进入窗口。普通 30Hz 单端点触碰不能直接宣称坏跳变进入标签。若严格执行声明的“异常端点范围任意重叠都隔离”，需再排除 78 TRAIN/8 DEV 窗；这是包含上述 8 窗的总数，不是 86 个严重标签错误。
- expanded 与 strict 的其余八源全部允许 `(实际 payload, tick)` 集合 exact 一致，main3 的完整索引逐数组相等；计数与 recipe 的 TRAIN `source_counts` 相符。新增 DexYCB 没有绕过其他源排除。原 UniDex 飞离尾段、人工隔离和 HOI4D 初段隔离仍沿用先前集合；本次只发现上述边界缺口，未恢复原坏尾段。
- TRAIN/DEV 的 native `group_id` 和实际 payload 路径无重叠，main3 各来源 sequence 无 TRAIN/DEV 重叠。DexYCB 全部行按官方 s1 被试分配复算，200 条 TEST 序列名未进入 native 配方；HO-Cap 未出现在训练来源。这不是对所有数据集官方 split 定义的全新独立认证。
- DexYCB 元数据保留 699 TRAIN/100 DEV 序列；真正有准入窗为 **697 TRAIN/99 DEV**，仍为 3,096 TRAIN/382 DEV 窗。三条零窗序列：`20200820_144644`、`20201022_114441`（TRAIN），`20200928_155434`（DEV）。空行保留可追溯，不是训练泄漏。
- 每个 TRAIN 来源独立抽取 10,000 个实际 sampler ID，0 个已 blocked ID。这个检查确认旧排除的 sampler wiring；不能替代上面对漏标窗的语义核对。
- 新增六源每源 TRAIN/DEV 各取首尾实际准入窗，共 24 窗，独立 NumPy 从 raw hand/object pose/current anchor 重算标签；最大绝对误差 **1.64e-7 m**。2,638 份 native geometry 尺度最大对角线 1.173m；全序列已知手点的最大腕指跨度无 >0.5m 案例。未发现数量级单位错误，但少量端点/尺度审计不能证明每帧 CAD alignment 或手物语义都正确。

工程证据索引：`progress.json`、`index-split-audit.json`、`sequences.jsonl`、`findings.json`、`boundary-overlap-audit.json`、`real-reader-halfstep-audit.json`、`endpoint-sampling-audit.json`。初始 `findings.json` 的 `threshold_violating_windows` 使用“触碰任意坏端点”的保守定义，必须结合后两份半步复核区分真正监督重叠；不能直接将其 45 条称为未筛掉的异常运动。
