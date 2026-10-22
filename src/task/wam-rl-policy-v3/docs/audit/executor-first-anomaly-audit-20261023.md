# 执行器首次异常取证（2026-10-23）

本报告只读已有产物，不启动仿真。`first_limit_tick` 定义为保存轨迹中第一个实际关节位置越过 native lower/upper bound 的状态；`contact_relation` 使用 `measured_contact` 或有效接触/物体力达到 1 N 的最早证据。单次数值 IK 未运行，因此没有把“本次求解未找到”写成不可达证明。

三条执行链保持分开：WAM unseen40（v9，A1，40/40 最终 physical joint limit）、旧 source40 基线（A0，14/40 成功、12 回合终止限位）和 source-R4（A0，0/40，全部烧满 24,000 controls）。

追加的面板 gate 也未通过：source40 的 40 个 `(sequence, object)` 与 WAM v9 unseen40 的 8 个重复面板键交集为 **0**，所以不是同一初态面板；`source 成功 ∩ WAM 失败` 为空。没有可执行见证可做离线替代抓姿检查，也不启动短回放。该结果只说明两批不能作配对因果比较，不说明接近目标假说为真或为假。

## 分类汇总

|执行链|样本/终止标签|首限位阶段|接触前首限位|接触后首限位|无接触证据|首限位关节事件|限外目标事件|达到阶段后再限位|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|WAM unseen40|40/40|close=7, approach=33|20|20|0|49|0|9|
|source40|40/12|close=5, approach=7|6|5|1|18|0|2|
|source-R4|40/0|approach=31, close=8, lift=1|—|—|29|0|—|0|

WAM unseen40 的首限位关节事件为 `thumb_intermediate_joint` 24（下界）、`thumb_proximal_pitch_joint` 15（下界 10、上界 5）、`pinky_proximal_joint` 5（下界）、`thumb_distal_joint` 5（下界）；同一状态同时越界时按首个保存状态中的全部关节计数。source40 的对应事件为 18 个（拇指中节 8、拇指近节 6、其余环/小指/拇指远节 4）。两者所有首限位事件的 pre-limit target status 都是 `legal_inward_or_neutral`：WAM 49/49、source 18/18；限外目标事件均为 0。

原始 τ 的可用性也单独保留：WAM 40 例中有 30 例在首越界状态保存了 `tau_valid`，source40 这条 A0/source 链的 12 例均没有有效 τ（这是不同执行链的输入合同，不能补算成 WAM τ）。映射后的 `pd_targets_actual`、raw/adapter action、wrist target 和几何/接近字段逐例写入 JSON；WAM v9 旧 consumer 没有独立 `reachability` 字段，因此不能从它证明路径可达或不可达，source40 新 consumer 的 `reachability` 字段只能作为保存时的准入记录，仍不是全局可达性证明。

R4 的终止标签分布来自其自己的执行链：35/40 `stage_contact_not_certified`、5/40 `contact_without_lift`，无 physical joint-limit；其中 31 例最终仍在 approach、8 例 close、1 例 lift。严格按“没有进入 close 且没有接触力/测量接触证据”，29/40 属于从未建立接近/接触证据；9/40 有某种力或测量接触证据，5/40 达到 stage contact。它们不能与 WAM 的 40/40 最终限位合并解释。

## 逐例：WAM unseen40（A1）

下面的 `q→limit` 是首次保存越界状态，`pd` 是该状态前一个控制的映射后 PD 目标；完整原始 τ（若该 tick `tau_valid`）、动作向量、wrist target、接触力和前后 5 状态窗口见 JSON 产物。

|id|物体|tick/time(s)|阶段|首限位关节（方向）|q / excess(rad)|pd|接触关系|接触首证据(s)|
|---|---|---:|---|---|---:|---:|---|---:|
|p003-A1|banana|60/2.000|close|thumb_proximal_pitch_joint upper q=0.5759 e=0.0259 pd=0.3666; thumb_intermediate_joint lower q=-0.0160 e=0.0160 pd=0.2200|—|—|contact_before_first_limit|1.4|
|p009-A1|cylindermedium|46/1.533|close|thumb_proximal_pitch_joint lower q=-0.0075 e=0.0075 pd=0.2614|—|—|contact_before_first_limit|1.5|
|p015-A1|hammer|2/0.067|approach|thumb_intermediate_joint lower q=-0.0004 e=0.0004 pd=0.2039|—|—|contact_only_at_or_after_first_limit|1.4333333333333333|
|p021-A1|cubesmall|70/2.333|approach|thumb_proximal_pitch_joint lower q=-0.0027 e=0.0027 pd=0.3084|—|—|contact_only_at_or_after_first_limit|3.1333333333333333|
|p027-A1|flashlight|37/1.233|approach|thumb_intermediate_joint lower q=-0.0624 e=0.0624 pd=0.1042|—|—|contact_before_first_limit|1.1666666666666667|
|p033-A1|alarmclock|37/1.233|approach|pinky_proximal_joint lower q=-0.0095 e=0.0095 pd=0.4433; thumb_distal_joint lower q=-0.0017 e=0.0017 pd=0.1587|—|—|contact_only_at_or_after_first_limit|2.3|
|p039-A1|cup|60/2.000|approach|thumb_intermediate_joint lower q=-0.1212 e=0.1212 pd=0.1449|—|—|contact_before_first_limit|1.8333333333333333|
|p045-A1|gamecontroller|1/0.033|approach|thumb_intermediate_joint lower q=-0.0164 e=0.0164 pd=0.1665|—|—|contact_only_at_or_after_first_limit|1.0666666666666667|
|p051-A1|banana|59/1.967|approach|thumb_intermediate_joint lower q=-0.0326 e=0.0326 pd=0.1586|—|—|contact_before_first_limit|1.4|
|p057-A1|cylindermedium|46/1.533|close|thumb_proximal_pitch_joint lower q=-0.0081 e=0.0081 pd=0.3005|—|—|contact_before_first_limit|1.5|
|p063-A1|hammer|2/0.067|approach|thumb_intermediate_joint lower q=-0.0004 e=0.0004 pd=0.2039|—|—|contact_only_at_or_after_first_limit|1.4333333333333333|
|p069-A1|cubesmall|69/2.300|approach|thumb_proximal_pitch_joint lower q=-0.0652 e=0.0652 pd=0.2525|—|—|contact_only_at_or_after_first_limit|4.1|
|p075-A1|flashlight|37/1.233|approach|thumb_intermediate_joint lower q=-0.0987 e=0.0987 pd=0.1004|—|—|contact_before_first_limit|1.1666666666666667|
|p001-A1|alarmclock|37/1.233|approach|pinky_proximal_joint lower q=-0.0095 e=0.0095 pd=0.4431; thumb_distal_joint lower q=-0.0017 e=0.0017 pd=0.1585|—|—|contact_only_at_or_after_first_limit|2.4|
|p007-A1|cup|61/2.033|approach|thumb_proximal_pitch_joint upper q=0.6562 e=0.1062 pd=0.2072; thumb_intermediate_joint lower q=-0.0642 e=0.0642 pd=0.1243|—|—|contact_before_first_limit|1.8333333333333333|
|p013-A1|gamecontroller|1/0.033|approach|thumb_intermediate_joint lower q=-0.0164 e=0.0164 pd=0.1665|—|—|contact_only_at_or_after_first_limit|1.0666666666666667|
|p019-A1|banana|56/1.867|close|thumb_intermediate_joint lower q=-0.0692 e=0.0692 pd=0.1775|—|—|contact_before_first_limit|1.4|
|p025-A1|cylindermedium|46/1.533|close|thumb_proximal_pitch_joint lower q=-0.0082 e=0.0082 pd=0.3005|—|—|contact_before_first_limit|1.5|
|p031-A1|hammer|2/0.067|approach|thumb_intermediate_joint lower q=-0.0004 e=0.0004 pd=0.2039|—|—|contact_only_at_or_after_first_limit|1.4333333333333333|
|p037-A1|cubesmall|68/2.267|approach|thumb_proximal_pitch_joint lower q=-0.0144 e=0.0144 pd=0.2689|—|—|contact_only_at_or_after_first_limit|2.8666666666666667|
|p043-A1|flashlight|37/1.233|approach|thumb_intermediate_joint lower q=-0.0874 e=0.0874 pd=0.1070|—|—|contact_before_first_limit|1.1666666666666667|
|p049-A1|alarmclock|37/1.233|approach|pinky_proximal_joint lower q=-0.0095 e=0.0095 pd=0.4432; thumb_distal_joint lower q=-0.0017 e=0.0017 pd=0.1587|—|—|contact_only_at_or_after_first_limit|2.4|
|p055-A1|cup|59/1.967|approach|thumb_intermediate_joint lower q=-0.0113 e=0.0113 pd=0.1469|—|—|contact_before_first_limit|1.8333333333333333|
|p061-A1|gamecontroller|1/0.033|approach|thumb_intermediate_joint lower q=-0.0164 e=0.0164 pd=0.1665|—|—|contact_only_at_or_after_first_limit|1.0666666666666667|
|p067-A1|banana|60/2.000|approach|thumb_proximal_pitch_joint upper q=0.5648 e=0.0148 pd=0.3412; thumb_intermediate_joint lower q=-0.0658 e=0.0658 pd=0.2047|—|—|contact_before_first_limit|1.4|
|p073-A1|cylindermedium|46/1.533|close|thumb_proximal_pitch_joint lower q=-0.0082 e=0.0082 pd=0.3005|—|—|contact_before_first_limit|1.5|
|p079-A1|hammer|2/0.067|approach|thumb_intermediate_joint lower q=-0.0004 e=0.0004 pd=0.2039|—|—|contact_only_at_or_after_first_limit|1.4333333333333333|
|p005-A1|cubesmall|68/2.267|approach|thumb_proximal_pitch_joint lower q=-0.0046 e=0.0046 pd=0.2693|—|—|contact_only_at_or_after_first_limit|3.1|
|p011-A1|flashlight|37/1.233|approach|thumb_intermediate_joint lower q=-0.0824 e=0.0824 pd=0.1203|—|—|contact_before_first_limit|1.1666666666666667|
|p017-A1|alarmclock|37/1.233|approach|pinky_proximal_joint lower q=-0.0095 e=0.0095 pd=0.4431; thumb_distal_joint lower q=-0.0017 e=0.0017 pd=0.1585|—|—|contact_only_at_or_after_first_limit|2.3666666666666667|
|p023-A1|cup|61/2.033|approach|thumb_intermediate_joint lower q=-0.0691 e=0.0691 pd=0.1835|—|—|contact_before_first_limit|1.8333333333333333|
|p029-A1|gamecontroller|1/0.033|approach|thumb_intermediate_joint lower q=-0.0164 e=0.0164 pd=0.1665|—|—|contact_only_at_or_after_first_limit|1.0666666666666667|
|p035-A1|banana|60/2.000|approach|thumb_proximal_pitch_joint upper q=0.6275 e=0.0775 pd=0.3383|—|—|contact_before_first_limit|1.4|
|p041-A1|cylindermedium|46/1.533|close|thumb_proximal_pitch_joint lower q=-0.0082 e=0.0082 pd=0.3005|—|—|contact_before_first_limit|1.5|
|p047-A1|hammer|2/0.067|approach|thumb_intermediate_joint lower q=-0.0004 e=0.0004 pd=0.2039|—|—|contact_only_at_or_after_first_limit|1.4333333333333333|
|p053-A1|cubesmall|68/2.267|approach|thumb_proximal_pitch_joint lower q=-0.0296 e=0.0296 pd=0.2677|—|—|contact_only_at_or_after_first_limit|2.3|
|p059-A1|flashlight|37/1.233|approach|thumb_intermediate_joint lower q=-0.0349 e=0.0349 pd=0.1211|—|—|contact_before_first_limit|1.1666666666666667|
|p065-A1|alarmclock|37/1.233|approach|pinky_proximal_joint lower q=-0.0095 e=0.0095 pd=0.4429; thumb_distal_joint lower q=-0.0017 e=0.0017 pd=0.1583|—|—|contact_only_at_or_after_first_limit|2.3|
|p071-A1|cup|61/2.033|approach|thumb_proximal_pitch_joint upper q=0.5681 e=0.0181 pd=0.2569; thumb_intermediate_joint lower q=-0.1033 e=0.1033 pd=0.1542|—|—|contact_before_first_limit|1.8333333333333333|
|p077-A1|gamecontroller|1/0.033|approach|thumb_intermediate_joint lower q=-0.0164 e=0.0164 pd=0.1665|—|—|contact_only_at_or_after_first_limit|1.0666666666666667|

## 逐例：旧 source40 基线（A0）

|id|物体|tick/time(s)|阶段|首限位关节（方向）|q / excess(rad)|pd|接触关系|接触首证据(s)|
|---|---|---:|---|---|---:|---:|---|---:|
|p003-A0|eyeglasses|57/1.900|close|thumb_proximal_pitch_joint upper q=0.5586 e=0.0086 pd=0.2476; thumb_intermediate_joint lower q=-0.0482 e=0.0482 pd=0.1486|—|—|contact_before_first_limit|1.5|
|p015-A0|scissors|40/1.333|close|thumb_proximal_pitch_joint upper q=0.5738 e=0.0238 pd=0.3554; thumb_intermediate_joint lower q=-0.3527 e=0.3527 pd=0.2132|—|—|contact_before_first_limit|1.3|
|p036-A0|spherelarge|46/1.533|close|thumb_proximal_pitch_joint lower q=-0.0050 e=0.0050 pd=0.2627|—|—|contact_before_first_limit|1.4666666666666666|
|p001-A0|camera|51/1.700|approach|thumb_intermediate_joint lower q=-0.0089 e=0.0089 pd=0.1615|—|—|contact_before_first_limit|1.6666666666666667|
|p004-A0|fryingpan|1/0.033|approach|thumb_proximal_pitch_joint upper q=0.5587 e=0.0087 pd=0.2177|—|—|contact_only_at_or_after_first_limit|1.2333333333333334|
|p010-A0|phone|2/0.067|approach|thumb_intermediate_joint lower q=-0.0068 e=0.0068 pd=0.1429|—|—|contact_only_at_or_after_first_limit|1.4|
|p022-A0|cylinderlarge|66/2.200|close|ring_proximal_joint lower q=-0.0911 e=0.0911 pd=1.3121; thumb_proximal_pitch_joint lower q=-0.0135 e=0.0135 pd=0.2428; thumb_distal_joint lower q=-0.0047 e=0.0047 pd=0.1942|—|—|contact_before_first_limit|2.0666666666666664|
|p031-A0|piggybank|16/0.533|approach|pinky_proximal_joint lower q=-0.1454 e=0.1454 pd=0.5953; thumb_distal_joint lower q=-0.0028 e=0.0028 pd=0.0793|—|—|no_contact_evidence_in_saved_trace|—|
|p034-A0|pyramidsmall|2/0.067|approach|thumb_proximal_pitch_joint lower q=-0.0160 e=0.0160 pd=0.3115; thumb_intermediate_joint lower q=-0.0510 e=0.0510 pd=0.1869|—|—|contact_only_at_or_after_first_limit|2.5666666666666664|
|p011-A0|piggybank|1/0.033|approach|thumb_intermediate_joint lower q=-0.0035 e=0.0035 pd=0.1664|—|—|contact_only_at_or_after_first_limit|1.8666666666666667|
|p029-A0|mouse|1/0.033|approach|thumb_intermediate_joint lower q=-0.0042 e=0.0042 pd=0.1630|—|—|contact_only_at_or_after_first_limit|1.5|
|p035-A0|scissors|45/1.500|close|thumb_intermediate_joint lower q=-0.0574 e=0.0574 pd=0.1577|—|—|contact_before_first_limit|1.3|

## 机制判断与最便宜修法

|首次异常类型|证据|共同机制候选|最便宜的下一步|
|---|---|---|---|
|接触前越界（WAM 20/40；source 5/12，另有 source 1/12 无证据）|首越界时 q 已越界，但 pre-limit PD target 没有向限外；WAM 旧记录没有 reachability 字段|动态超调、初态/速度、坐标或 PD 映射异常均仍可能；现有记录不能把目标不可达单独坐实|先做离线状态/命令一致性和可达性审计；若要干预，优先限速/有界状态保护的共同机制，不逐物体改目标|
|接触后越界（WAM 20/40；source 6/12）|接触力/测量接触先于首越界；target 仍合法 inward/neutral|接触载荷、手指提前闭合或闭合后未卸载导致动态越界|先接入力/负载驱动的逐指卸载/重试与接触后闭合门控；保持同一 R 和后续逻辑|
|已达标后再限位|本三组被标记限位的回合中，没有 `hold`/最终成功先于首限位；R4 也无成功|当前证据不支持把“正常达标后收尾限位”当主类，但需保留该判据以免把未来 3/40 ever-success 与最终限位混淆|后续审计必须同时报告 ever-success、contact/lift/hold 首次时间和最终终止标签|
|限位标签但实际合法端位|这三组的 limit-labelled 记录首个 q 状态都实际越界；没有“合法 q 端位仅因标签失败”的证据|不能把触限标签直接当根因；目标越界、实际越界、合法机械端位三者已分开|继续使用 q、目标命令和阶段证据三分统计，禁止只看最终终止原因|

### 结论边界

R4 的 0/40 说明该链的主损失落在接触建立/推进：35/40 连 stage contact 都未达；40 回合合计跑满 24,000 controls（每回合 600），没有早停。它没有证明 WAM unseen40 的限位机制已经消失，也不能把 source-R4 的超时和 WAM 的最终限位合并成同一执行链机制。WAM/source 首限位前没有持续向限外 PD 目标，支持“需要查共同执行器动态/接触处理”的方向；它仍不能排除数据覆盖、换侧接近、目标可达性或坐标映射异常。

原始 JSON 审计产物：`outputs/wam-rl-policy-v3/executor-first-anomaly-audit-20261023.json`。审计工具：`src/task/wam-rl-policy-v3/tools/audit/audit_executor_first_anomalies.py`。
