---
schema: ref2dex.probe.v2
probe_id: P-20261022-wam-v6-goal-chain-audit
experiment_id: P-20261022-wam-v6-goal-chain-audit
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: c971bca689fa76845128cecb04c2dc43173f069c
claim_id: C1
hypothesis_family: HF-v6-goal-contract-identifiability
probe_index_in_family: 1
seed_pool: probe
seeds: [172,181,182,183,184,185,186,187,188]
decision_changed_if_positive: prioritize causal waypoint versus realized-endpoint conditioning and intent supervision before further model scaling
decision_changed_if_negative: inspect supervision and sequence generalization without assuming missing contact intent explains the errors
status: COMPLETED
run_id: goal-chain-audit-20261022-r1
---

# Goal contract, event-aligned identifiability and deployment-path consistency

**12:4x implementation update**: restoring complete task declarations and fixing RELEASE phase timing
changes the same 501 archived robot trajectories from 0 to 77 full-task scorer successes (9 inside the
safe prefixes). This is a scorer repair, not a new physical-policy result. New teacher/source-PD controls
remain 0/6 complete tasks, so P1 stays HOLD. See [the joint review package](../../audit/fix-review-20261022.md)
for diffs, concrete expectations, remaining intent/reset/action limits, and the conditional v6 retrain list.

Result: COMPLETED. FullDEV3356/K8: healthy OracleH/O15.80/14.72mm versus FM17.37/16.21; Planner21.12/25.55 versus22.79/27.60. Goal semantics mismatch verified; contact-region gains not robust after phase/same-window controls.
Decision: prioritize clarifying the spatial-goal/time contract and sequence generalization before scaling. Do not directly relabel t24 supervision with planner waypoints without defining the timing/target contract. No new training, loss, model or data changes. Three-executor-prior study remains design only.

这是 Decision Probe：区分 goal 契约错配、事件相位/样本筛选虚增方差，以及手物共同偏移；
使用已有权重/缓存的推理与 GT 统计，是当前能改变下一步决策的最便宜方法。

**独立核到**：语义契约不同；相同样本下换 goal 后误差增大；FM 在 TRAIN 部分误差改善、
DEV 变差且 C 头更好；GT τ 的几何接触 F1 高于模型；存在共同平移误差成分。
**仍属推测/待定**：直接用 Planner-G 重训的收益、精确条件下 τ 是否不可辨识、架构本身是否差、
前8帧闭环可控性。完整 TaskScheduler 缺抓稳/释放/支撑反馈；本轮 Planner-G 是实际路径子模块
逐序列回放，使用已知录制物体路径作为 oracle reference，不能冒充完整机器人部署评估。

完整产物：[数值报告](../../../../../../outputs/ptv3-shared-wam-v6/goal-chain-audit-20261022-r1/audit-report.md)、[逐项 CSV](../../../../../../outputs/ptv3-shared-wam-v6/goal-chain-audit-20261022-r1/decision-metrics.csv)、[运行完成记录](../../../../../../outputs/ptv3-shared-wam-v6/goal-chain-audit-20261022-r1/completion.json)。

Run: `goal-chain-audit-20261022-r1`; completed 2026-10-22T11:40:13.494180+08:00

Independently verified: the deployment path module supplies a spatial waypoint with no endpoint time; training supplies the realized t+24 pose. Goal interventions below use the exact unmodified OrderedGoalScheduler, with recorded object path supplied as a known oracle instruction and persistent causal observation replay. Full TaskScheduler is not reproduced: measured grasp/release/support feedback is unavailable. E_next is held at the same recorded label across arms. These are offline forecasting diagnostics, not closed-loop controllability or task-success measurements.

Interpretation: a substantial goal-conditioning mismatch and TRAIN→DEV generalization gap are measured. Improved C-head F1 does not imply improved tau/contact geometry. Contact-region variance benefits are not robust after event phase and identical-window controls. These checkpoint comparisons do not isolate flow matching causally: training step, objective and continuation differ.

## Paired goal interventions, prefix8

All DEV3356 windows/113 cached sequences/112 native captures; TRAIN5241/1064/1063; unadmitted prefixes905/905/905. TRAIN and unadmitted complete H4+K24 supports are disjoint across all local207579 TRAIN windows and aliases. TRAIN/DEV native capture intersection=0. DEV complete coverage contains overlapping windows; report847-support nonoverlap core separately. Remote ancestor exposure for unadmitted prefixes remains UNKNOWN.

|Model / panel|N windows|Oracle H/O mm|Planner H/O mm|Perturbed H/O mm|C-head F1 O/P/Pert|
|---|---:|---|---|---|---|
|healthy4371 / train_prefix|5241|10.559/7.552|17.838/20.909|18.987/21.027|0.673/0.611/0.671|
|flow8500 / train_prefix|5241|9.949/6.433|18.692/22.477|20.134/22.416|0.702/0.664/0.697|
|healthy4371 / heldout_sequence|3356|15.797/14.722|21.122/25.550|21.668/25.887|0.686/0.620/0.684|
|flow8500 / heldout_sequence|3356|17.374/16.206|22.786/27.604|23.511/28.030|0.709/0.674/0.704|
|healthy4371 / heldout_prefix_local_recipe|905|48.069/0.966|48.063/0.998|49.349/3.609|0.421/0.434/0.443|
|flow8500 / heldout_prefix_local_recipe|905|49.767/3.917|49.745/3.933|51.406/5.280|0.411/0.424/0.413|

C-head threshold=.8. For full DEV, H EPE N=1,109,388 joint-frames/draw, object EPE N=859,136 point-frames/draw; C F1 N=54,343,680 valid query-frames/draw. FM K8 seeds172,181–187; deterministic baseline K1. Query counts per draw are not multiplied into independent sample size. Captures are bootstrap units (500 draws, seed188).

|Nonoverlap DEV core|N|Oracle H/O mm|Planner H/O mm|
|---|---:|---|---|
|healthy4371|847|13.526/10.832|16.943/17.057|
|flow8500|847|15.235/12.384|18.672/18.647|

FM minus healthy, full DEV Oracle prefix8: hand +1.576mm, paired capture95%CI[1.449,1.701]; object +1.484mm,[1.334,1.658]; C-head F1 +.02347,[.01722,.03004]. These intervals describe this panel/checkpoint pair, not independent training replications.

|DEV prefix8 / model / goal|C centroid mm (N3010 windows/22181 frames)|GT/own-object contact distance mm (N3102 windows/14996006 queries)|GT/own geom F1|GT/own clearance proxy mm|
|---|---|---|---|---|
|healthy4371 / oracle|3.883|14.709/10.537|0.507/0.591|0.893/0.717|
|healthy4371 / planner|4.002|22.980/11.287|0.442/0.574|0.959/0.683|
|healthy4371 / perturbed|3.889|22.474/10.877|0.433/0.583|1.228/0.873|
|flow8500 / oracle|3.937|15.049/10.571|0.484/0.564|0.956/0.767|
|flow8500 / planner|4.165|23.527/11.615|0.421/0.543|1.000/0.701|
|flow8500 / perturbed|3.964|23.029/11.008|0.414/0.552|1.301/0.929|

C location=probability-weighted surface-contact centroid error on GT-positive frames, not full patch geodesic error. Contact-distance mask=GT confidence>.25 (soft positives, not exclusively physical contact). Clearance proxy=GT-negative C<.01, N3353 windows/25767438 query-frames. Geometry F1 uses exponential3mm radius/10mm falloff≥.8 (distance≤5.23mm), with exact query counts in JSON. Full geometry anchors share the same GT tick/mask. Predicted object means affine CAD extension of predicted32; it does not preserve nonrigid residual.

GT-tau reference on the SAME3356 DEV: soft-positive contact distance8.841mm; geometry F1 .970809; geometric contact-centroid1.181mm. Thus geometry can reproduce near-correct contact when given GT tau; its nonzero soft-positive distance is a label/geometry baseline. Oriented-depth proxy GT reference .922mm versus healthy1.038/FM1.033mm (N3433820 vertex-frames), so raw oriented depth is not physical penetration. See actual CAD occupancy below.

## Time and interaction strata (Oracle)

|Panel / band|N windows|Healthy H/O mm|FM H/O mm|
|---|---:|---|---|
|train_prefix / prefix8|5241|10.559/7.552|9.949/6.433|
|train_prefix / middle8|5241|27.867/15.813|24.142/14.014|
|train_prefix / last8|5239|40.974/16.055|36.162/17.221|
|heldout_sequence / prefix8|3356|15.797/14.722|17.374/16.206|
|heldout_sequence / middle8|3354|59.343/40.270|63.232/43.986|
|heldout_sequence / last8|3352|96.409/28.203|101.245/31.933|
|heldout_prefix_local_recipe / prefix8|905|48.069/0.966|49.767/3.917|
|heldout_prefix_local_recipe / middle8|905|226.410/1.618|231.195/9.075|
|heldout_prefix_local_recipe / last8|905|397.192/2.937|403.416/13.485|

|Panel / phase, prefix8|N H windows / joint-frames|N object windows / point-frames|Healthy H/O mm|FM H/O mm|
|---|---|---|---|---|
|train_prefix / pre_contact|355/64785|355/50080|8.246/3.034|8.284/3.501|
|train_prefix / contact_onset|803/131292|803/100128|10.950/5.534|10.602/5.376|
|train_prefix / hold|4450/1342845|4450/1025056|10.134/8.854|9.662/7.041|
|train_prefix / post_release|919/214158|919/163360|13.689/2.037|11.843/4.172|
|train_prefix / unknown|24/2016|24/3072|10.138/5.617|11.108/6.306|
|heldout_sequence / pre_contact|257/55650|257/43424|9.972/2.369|11.516/3.973|
|heldout_sequence / contact_onset|384/61068|384/46848|15.260/5.893|15.966/6.523|
|heldout_sequence / hold|2850/872424|2850/671520|15.007/17.930|16.627/19.337|
|heldout_sequence / post_release|485/114009|485/87328|25.314/2.050|27.038/4.544|
|heldout_sequence / unknown|75/6237|77/10016|9.606/4.959|11.262/6.281|
|heldout_prefix_local_recipe / pre_contact|573/192528|573/146688|49.086/0.823|50.112/3.653|
|heldout_prefix_local_recipe / contact_onset|0/0|0/0|NA/NA|NA/NA|
|heldout_prefix_local_recipe / hold|3/1008|3/768|13.118/32.960|15.492/34.461|
|heldout_prefix_local_recipe / post_release|329/110544|329/84224|46.617/0.925|49.479/4.099|
|heldout_prefix_local_recipe / unknown|0/0|0/0|NA/NA|NA/NA|

Onset=first6 recorded frames after a contact-entry transition; subsequent contact=hold. Post-release follows recorded exit; unknown labels reported separately. Full band×phase×goal×source counts, C scores and geometry proxies are in large-summary.json.

|DEV source / Oracle|N|Healthy H/O mm|FM H/O mm|C F1 healthy/FM|
|---|---:|---|---|---|
|0|96|6.239/1.061|8.756/3.743|0.763/0.763|
|1|2944|16.300/15.914|17.796/17.342|0.690/0.714|
|2|96|17.077/11.914|19.311/13.438|0.554/0.587|
|4|30|10.866/6.859|12.456/8.284|0.187/0.136|
|5|21|17.210/15.168|18.829/15.930|0.372/0.424|
|6|86|11.986/4.714|14.569/7.222|0.643/0.685|
|9|7|7.297/3.482|8.959/4.801|0.473/0.447|
|10|76|8.940/4.686|10.437/5.864|0.330/0.190|

## Identifiability: raw versus event alignment

Main disjoint bank6185 candidates/1064 cached sequences/1063 native captures. Expanded existing-TRAIN banks: seed182 N31619; seed183 N31576; both1064 sequences/1063 captures. Up to16 uniform+16 inside-horizon-event candidates per sequence. Candidate windows may overlap, but a variance group uses one independent native capture and each capture is used once per gate. Event-only subpopulation N19672/19747,~1062 captures. No new data or fitting.

Piecewise phase warp preserves t0 and t24, aligning E_next to the BASE group median. Aligned first8 refers to phase ticks, not actual executable first8. Recompute all groups on event-only subpopulation; do not compare differently selected raw/aligned cohorts.

|Seed / gate|Independent groups / windows|Raw→aligned H8 mm²|H24 mm²|O8 mm²|O24 mm²|
|---|---|---|---|---|---|
|182 / 20|0/0|NA|NA|NA|NA|
|182 / 50|14/69|1610.002→709.190|45970.622→34479.165|144.846→145.457|187.988→187.004|
|182 / 50_time3|9/45|3080.526→2740.459|51868.454→50713.854|28.849→32.198|65.660→67.587|
|182 / 100|57/266|1996.194→1606.129|34230.927→25276.580|346.089→399.987|761.444→766.469|
|182 / 100_time3|42/193|1627.438→1541.544|31873.673→30933.318|229.613→229.099|450.746→448.708|
|183 / 20|0/0|NA|NA|NA|NA|
|183 / 50|11/57|1454.789→568.452|48451.335→34402.745|99.662→111.986|203.766→203.127|
|183 / 50_time3|10/49|2486.439→2114.262|42620.989→41236.571|77.330→80.818|155.693→153.867|
|183 / 100|57/269|2102.504→1796.738|33418.328→27126.505|300.927→347.506|699.364→696.782|
|183 / 100_time3|45/206|1140.456→1020.154|26842.311→26172.297|199.073→195.921|477.658→472.275|

Conditions: identical CAD/hand-presence/E_next type; complete-link RMS radius20/50/100mm independently on observed H4 hand,64-point H4 object and32-point goal. time3 additionally matches an ALREADY supplied E_next timing within3frames. No exact-input irreducibility follows from finite-radius groups. All raw stages/label permutations and condition-width stats are in event-only-aligned-variance.json; all-TRAIN population in event-aligned-variance.json.

## Same-kept-window contact intent control

Every stage below restricts the previous partition to EXACTLY the current nonsingleton windows. This removes the effect of simply dropping rare-label samples. Singleton partitions have zero df, never zero-variance evidence.

|Seed / gate / added label|Same N|Aligned H8 reference→refined mm²|H24|O8|O24|H8 drop%, paired95%CI|
|---|---:|---|---|---|---|---|
|182 / 50 / object_region|44|709.855→742.232|29497.595→31352.567|100.161→102.716|177.691→182.919|-4.561 [-14.984, 5.862]|
|182 / 50 / hand_region|29|832.551→832.551|30036.564→30036.564|110.655→110.655|172.876→172.876|0.000 [0.000, 0.000]|
|182 / 50 / interaction|10|1656.450→1656.450|17341.999→17341.999|78.846→78.846|106.476→106.476|0.000 [0.000, 0.000]|
|182 / 50_time3 / object_region|28|2988.326→2867.636|53648.630→40762.463|39.115→43.367|74.957→83.365|4.039 [-2.942, 32.396]|
|182 / 50_time3 / hand_region|21|2228.960→2228.960|29991.884→29991.884|57.733→57.733|109.813→109.813|0.000 [0.000, 0.000]|
|182 / 50_time3 / interaction|8|536.374→536.374|24100.804→24100.804|88.909→88.909|184.667→184.667|0.000 [0.000, 0.000]|
|182 / 100 / object_region|189|1553.086→1626.840|25021.047→25472.248|399.615→411.078|727.489→747.231|-4.749 [-8.084, -0.990]|
|182 / 100 / hand_region|128|1641.625→1506.068|24306.506→24289.045|451.084→431.362|706.001→685.427|8.258 [-2.179, 15.400]|
|182 / 100 / interaction|55|929.506→929.506|23339.982→23339.982|375.432→375.432|549.878→549.878|0.000 [0.000, 0.000]|
|182 / 100_time3 / object_region|137|1004.463→894.638|29444.106→26703.869|172.214→179.991|341.820→371.578|10.934 [-6.238, 23.956]|
|182 / 100_time3 / hand_region|89|775.118→775.118|20418.459→20418.459|200.056→200.056|440.976→440.976|0.000 [0.000, 0.000]|
|182 / 100_time3 / interaction|39|917.144→917.144|19179.193→19179.193|268.793→268.793|438.280→438.280|0.000 [0.000, 0.000]|
|183 / 50 / object_region|37|790.088→754.282|40402.979→44864.702|96.879→104.577|165.902→173.159|4.532 [-5.602, 16.006]|
|183 / 50 / hand_region|18|395.432→395.432|21641.407→21641.407|132.019→132.019|231.972→231.972|0.000 [0.000, 0.000]|
|183 / 50 / interaction|6|551.878→551.878|12991.715→12991.715|147.831→147.831|323.888→323.888|0.000 [0.000, 0.000]|
|183 / 50_time3 / object_region|34|2462.762→2728.251|38393.240→39606.248|90.116→103.889|157.123→178.998|-10.780 [-23.001, 6.751]|
|183 / 50_time3 / hand_region|17|3675.260→3675.260|33673.828→33673.828|164.404→164.404|266.398→266.398|0.000 [0.000, 0.000]|
|183 / 50_time3 / interaction|6|646.599→646.599|9907.773→9907.773|287.300→287.300|511.677→511.677|0.000 [0.000, 0.000]|
|183 / 100 / object_region|200|1884.706→1879.340|25762.656→25104.712|319.242→335.554|608.214→569.916|0.285 [-8.896, 8.810]|
|183 / 100 / hand_region|136|1597.963→1653.684|23208.315→23325.128|361.753→356.838|574.879→570.965|-3.487 [-7.169, -0.227]|
|183 / 100 / interaction|62|917.521→917.521|16431.022→16431.022|467.440→467.440|790.383→790.383|0.000 [0.000, 0.000]|
|183 / 100_time3 / object_region|154|871.458→914.674|22756.911→22214.607|174.757→187.264|446.732→396.167|-4.959 [-11.842, 1.591]|
|183 / 100_time3 / hand_region|109|714.533→716.426|17954.688→18065.986|232.452→226.750|412.171→401.374|-0.265 [-2.568, 2.752]|
|183 / 100_time3 / interaction|44|899.950→899.950|13854.273→13854.273|304.326→304.326|503.458→503.458|0.000 [0.000, 0.000]|

At50mm object-region refinement: seed182 same44 windows, mean H4-hand/object/goal distances35.684/8.733/19.765→35.351/8.699/19.744mm; seed183 same37 windows35.066/8.794/18.018→36.095/10.156/19.737mm. H8 drops −4.56%[-14.98,5.86] and+4.53%[-5.60,16.01], no stable benefit. At100mm/time3 the apparent42% reduction becomes same-window+10.93%[-6.24,23.96] in seed182 and−4.96%[-11.84,1.59] in seed183. Do not claim contact intent is sufficient or tau intrinsically unidentifiable. Narrow support remains limited; Ref annotations unavailable.

## Watertight CAD occupancy / signed clearance at t8

All3356 DEV inferred with all3 goals and K8 FM; mapped watertight GRAB meshes cover2944 windows/92 native captures.412 non-GRAB windows lack mapped CAD assets and are explicitly excluded from this metric.128 sampled transported-skin vertices/window/draw: healthy376832 vertices; FM3014656 vertices across8draws. No singular object maps. Exact affine CAD containment; world signed-distance lower/upper bounds from canonical BVH SDF×min/max singular values. This audits a hand-skin proxy and an affine extension of the predicted32, not a complete physical robot hand.

|Model / goal / anchor|N windows|Certified..possible >1mm inside % of vertex-draws|Mean penetration depth bounds mm|Mean sampled minimum signed-clearance bounds mm|
|---|---:|---|---|---|
|healthy4371 / oracle / GT_object|2944|3.305..3.305|0.266..0.266|12.354..12.354|
|healthy4371 / oracle / own_object|2944|2.907..2.999|0.175..0.216|8.192..9.676|
|healthy4371 / planner / GT_object|2944|2.054..2.054|0.162..0.162|30.570..30.570|
|healthy4371 / planner / own_object|2944|2.376..2.420|0.132..0.154|10.673..11.643|
|healthy4371 / perturbed / GT_object|2944|2.349..2.349|0.187..0.187|23.043..23.043|
|healthy4371 / perturbed / own_object|2944|2.442..2.498|0.140..0.165|3.599..4.856|
|flow8500 / oracle / GT_object|2944|3.115..3.115|0.249..0.249|12.619..12.619|
|flow8500 / oracle / own_object|2944|2.888..2.992|0.171..0.213|6.925..10.279|
|flow8500 / planner / GT_object|2944|1.818..1.818|0.143..0.143|31.543..31.543|
|flow8500 / planner / own_object|2944|2.242..2.305|0.131..0.156|9.763..12.715|
|flow8500 / perturbed / GT_object|2944|2.199..2.199|0.179..0.179|23.722..23.722|
|flow8500 / perturbed / own_object|2944|2.494..2.565|0.144..0.173|2.535..4.872|

Positive sampled minimum signed clearance means separation, negative means penetration. It is evaluated at t8 only, not averaged prefix8; sampling128 skin vertices can miss extrema. Lower penetration under Planner does not prove quality: its GT-object clearance grows substantially, while absolute GT forecast errors worsen. Native forward controls and cube/CAD-registration/anisotropic-affine distance-bound controls PASS.

## Shared translation error, GT-fitted diagnostic at t8

|Model / goal|N windows|H/O EPE before mm|H/O after shared GT-fit shift mm|Common shift mm|Mean error direction cosine|Eligible draw count|
|---|---:|---|---|---|---|---:|
|healthy4371 / oracle|3354|33.215/29.044|28.171/16.598|23.921|0.448|2998|
|healthy4371 / planner|3354|46.121/54.956|38.655/23.393|43.105|0.569|2978|
|healthy4371 / perturbed|3354|47.783/56.103|39.481/23.751|44.367|0.598|3263|
|flow8500 / oracle|3354|35.911/31.971|30.612/18.244|25.231|0.433|25922|
|flow8500 / planner|3354|49.093/58.603|41.024/25.181|44.972|0.524|25736|
|flow8500 / perturbed|3354|51.088/60.083|42.109/25.708|46.841|0.598|26559|

Two DEV windows have no valid t8 hand target, so this control N3354. Per-window means at t8 differ from point-pooled prefix8 means. Fit uses (mean hand error+mean object error)/2 at that GT tick, diagnostic only, never a deployable correction. Planner FM: hand/object49.09/58.60→41.02/25.18mm after removing a44.97mm shared shift;80.49% of eligible error-vector pairs point into the same half-space. This supports a common-drift component while substantial hand residual remains.

## Reproduction and limits

Exact window identities/planner states: panels.json; overlap support-audit.json; training-only planner residuals planner-error-distribution.npz. Immutable batch sum/counts and per-draw variation: worker*/{healthy4371,flow8500}/*.npz. Main nested metrics/paired intervals large-summary.json; CAD endpoint arrays mesh-endpoints/; actual CAD/BVH results large-mesh-penetration.json; GT geometry reference GT-reference.json. Every process has launch*.json argv/runtime/commit/PID receipts. Scripts are task-local tools/audit/.

Core neural audit implementationc971bca6; launch HEAD can include unrelated shared-workspace commits. Later helpers/controls are recorded in their receipts. Healthy baseline4371 has no GT-object geometry branch; FM8500 has geometry_object_source=gt_dense2048. Both strict FP32, matching exact normalizer buffers and finite weights. Different objectives/steps mean this is an existing-checkpoint diagnostic, not causal proof that the FM architecture is worse.

GPU3 claim and GPU2 later claim were handled by resuming only this diagnostic; other jobs and GPU4/5/6/7 untouched. All owned GPU work has finished;0/3 were observed idle on release. Later final check shows other users on0/3,~3.1GiB each (PIDs2719937/2719940), untouched. No optimizer/training/PPO/loss changes/new data/GitHub push/credential access in this run. Outputs2.405GiB; filesystem146.03GiB free at final check, versus20GiB floor. No source data cleanup or migration.

Full same-executor predicted-tau/reference-retarget/no-prior study remains DESIGN ONLY in experiment card, awaiting owner confirmation. Offline EPE is not proof of controllability. Full TaskScheduler causal feedback replay and lifetime-unseen ancestor proof remain unavailable.

Checks:9 targeted tests PASS; native-forward max errors below1e-6m; GT metric identity/cube/affine depth/CAD registration PASS. Repository-wide verify still fails on6 PRE-EXISTING trajectory-policy cards missing schema fields, unrelated to this audit.

## Frozen methods and execution constraints

## Independently verified contract, and replay scope

`wam_rl_policy_v3/goal_scheduler.py:OrderedGoalScheduler.update` emits
`goal_semantics='spatial_path_waypoint'` and `reference_endpoint_time_s=None`.
Its positive horizon_s=.8 does NOT mean guaranteed arrival at t+.8. It uses
ordered milestones, observed path progress, proximity/motion feedback and bounded
lookahead (default3–25mm), recovering/timeout modes. V6's inherited
`ptv3_shared_wam_v5/data.py:sample_window` instead supplies the exact realized
pose at t+24, as32 same-ID point displacements in the current object frame.
These are different random variables. Direct replacement is not justified by
matching tensor dimensions; this audit measures the sensitivity, not a training fix.

Planner-G runs this exact, unmodified deployment path module persistently from
the first available observation through each selected tick, at30Hz; never reset
per window. Reference=complete recorded object path supplied as a known
instruction (oracle reference); only current/past poses and valid hand11 enter
`update`. Missing pose observations are skipped and counted; times remain real.
Right valid hand else left; hand11 indices0,1,5,9,13,17,4,8,12,16,20. Reports
retain modes/timeout/reference index. This is NOT full TaskScheduler replay:
the caches lack measured grasp-established, released, supported feedback. The
path module is used by TaskScheduler and deployment adapters, but these missing
inputs prevent certifying the complete deployment pipeline.

Oracle-G=unchanged t+24 endpoint. Perturbed-G=start from legal Planner-G, sample
paired SE(3) residual inv(Oracle-G)*Planner-G from TRAIN only, exclude the same
sequence, right-compose onto Planner-G, project with exact scheduler bounds.
This is a stress sensitivity arm matched to empirical TRAIN planning errors,
not an unbiased simulator of real robot planning error. E_next held identical
across arms to isolate goal; it is the recorded annotation, not separately
inferred deployment E_next. Planner causal replay does not make that label causal.

## Scale, splits, seeds and metrics

All frozen DEV3356 windows (all existing sources); report independently by
source and capture-cluster statistics. A separately flagged nonoverlap core
has disjoint inclusive[t-3,t+24] support28frames within capture. All-window
DEV coverage does not pretend3356 independent examples. TRAIN bank samples
up to16 disjoint supports from EVERY frozen TRAIN sequence (~1064); up to8
selected independently of their temporal order for neural evaluation. Unadmitted
prefixes: one random complete28-frame valid support avoiding EVERY local TRAIN
window in that recording. Complete remote ancestor admission manifest remains
unavailable: label `heldout_prefix_local_recipe`, never lifetime-unseen.

Selection/empirical perturbation seed181; fixed FM per-sample noise seeds
172,181..187,K8 with common random numbers across goal arms. Bootstrap/permutation188.
These are repeated forecast draws, NOT eight independent trained models.
Healthy deterministic K1; FM backbone evaluated once/condition then exactly the
existing4-step ODE repeated8times. Native-forward identity smoke required.
Report sample-mean accuracy and spread; no best-of-K claim. C head is independent
of sampled residual τ, so repeated C scores do not imply C sampling diversity.

All24 EPE, C-head queryF1 threshold.8 and probability-weighted GT-surface contact
centroid error, object best-SE3 residual; bands1–8/9–16/17–24. Phase evaluation
labels: pre-contact; first6 frames after GT event entry=onset; later contact=hold;
clear state following release=post-release; missing=unknown. No causal inference
from phase masks. Geometry prefix8 only, all2048 object queries, observed778/hand
bone-transport skin: distance on soft-positiveGT C>.25; geometricF1 exponential
3mm radius/10mm falloff threshold.8 (~5.23mm); contact centroid error on GT-positive
frames; oriented-nearest-point penetration PROXY and GT-direction-referenced
clearance PROXY on negativeC<.01. These proxies are not physical signed mesh
penetration. Own-object dense CAD uses observed-only affine extension of predicted32,
report affine residual and object rigid residual alongside both anchors. Compare
absolute hand/object EPE before interpreting small relative error as quality.

Identifiability follow-up: same raw H4-hand/H4-object/goal distances and E_next,
20/50/100mm complete-link neighborhoods; independent-capture groups. Add event-
aligned dispersion with fixed relative-event offsets and common valid support,
side by side with raw variance, plus contact-presence/region/finger/action labels.
Report neighborhood support/diameters and label-partition distances; singleton
groups contribute no variance degrees of freedom. Event timing control remains
necessary because E_next timing already enters v6. Ref annotations unavailable.

## Runtime and artifacts

Run root `outputs/ptv3-shared-wam-v6/goal-chain-audit-20261022-r1/`.
Entry `tools/audit/audit_goal_chain.py --phase prepare|bank|evaluate`.
Environment `/home2/wyy/miniconda3/envs/graspenv/bin/python`, strict FP32,
AMP/TF32 off. Two detached inference workers, physicalGPU2/3 exclusively (fresh
idle check required); never touch trainingGPU4/5/6 or GPU7, no process control
outside this run. Each worker allocator18GiB, ≥2GiB device headroom, initially
3h wall-time cap; completed immutable batch shards support resume. If GPU is
claimed, stop only owned diagnostic worker and reassign to an idle0–3 device.
CPU replay6 processes. No credential reads/writes, no push. No checkpoints made.
Estimated additional peak8GiB, initial disk149.45GiB available; floor20GiB.
Exact frozen commit, launch commands/PIDs/progress/ETA recorded in run receipts.


## Executor three-condition design — NOT LAUNCHED, awaits owner confirmation

Decision: does the learned prior improve executable task success versus the
existing smooth reference+geometric-retargeting prior, or only forecast scores?
Freeze one executor checkpoint, action frequency/control gains, observations,
hand/object morphology, reset states, task goals, episode timeout and safety
termination. Three condition providers: predictedτ from one fixed WAM checkpoint;
the owner's existing reference+geometry retargeter at matched replanning rate;
no trajectory prior (zero/masked condition with an explicitly compatible executor).
If the fixed executor was never trained with masked conditioning, no-prior is
an out-of-distribution ablation, NOT RL-from-scratch evidence; report that
separately. RL-from-scratch requires its own trained policy and matched compute,
so it cannot simultaneously be claimed a fixed-executor comparison.

Proposed data: immutable existing capture-disjoint task split,100 reset/task
instances across available objects and contact regimes,3 paired simulator seeds
per reset per arm=900 episodes. No selecting visually favorable demonstrations.
First engineering feasibility smoke5 resets/arm, then freeze100 before success
comparison. For each episode report full-task success, path/terminal achievement,
surface contactF1/penetration/contact-position error, longest continuous valid
hold duration and drop count; report paired task-cluster95%CIs and all failures.
Native robot contact/force vs human surface contact must be distinguished.

Budget proposal only: one idle authorized GPU, up to2h feasibility plus6GPUh
evaluation, max900 episodes×20s simulated time; estimate wall time from smoke
before authorization. Optional RL efficiency arm:3 seeds, success curves versus
environment transitions with equal observation/action/reward/reset/task mix and
fixed maximum10M transitions/seed; budget at most24GPUh total initially and stop
at matched target success or bound. Count all prior/executor training separately;
do not present inherited pretraining as free. No PPO, training or execution of
this design is authorized in this diagnostic run.


## Exact reproduction commands

```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
OUT=outputs/ptv3-shared-wam-v6/goal-chain-audit-20261022-r1
TOOL=src/task/ptv3-shared-wam-v6/tools/audit
$PY $TOOL/audit_goal_chain.py --output "$OUT" --phase prepare --workers 6 --seed 181
$PY $TOOL/audit_goal_chain.py --output "$OUT" --phase bank
CUDA_VISIBLE_DEVICES=3 $PY $TOOL/audit_goal_chain.py --output "$OUT" --phase evaluate --batch-size 64 --shard 0 --shards 2 --k 8 --memory-gib 18 --max-seconds 10400
CUDA_VISIBLE_DEVICES=0 $PY $TOOL/audit_goal_chain.py --output "$OUT" --phase evaluate --batch-size 64 --shard 1 --shards 2 --k 8 --memory-gib 18 --max-seconds 10800
CUDA_VISIBLE_DEVICES='' $PY $TOOL/prepare_event_enriched_bank.py --output "$OUT" --seed 182
CUDA_VISIBLE_DEVICES='' $PY $TOOL/prepare_event_enriched_bank.py --output "$OUT" --seed 183 --bank-name event-bank-seed183
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_event_aligned_variance.py --output "$OUT/event-bank" --seed 188
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_event_aligned_variance.py --output "$OUT/event-bank" --seed 188 --event-only
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_event_aligned_variance.py --output "$OUT/event-bank-seed183" --seed 188
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_event_aligned_variance.py --output "$OUT/event-bank-seed183" --seed 188 --event-only
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_same_support_variance.py --output "$OUT/event-bank"
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_same_support_variance.py --output "$OUT/event-bank-seed183"
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_goal_contract_gaps.py --output "$OUT"
CUDA_VISIBLE_DEVICES=0 $PY $TOOL/audit_large_mesh_contact.py --output "$OUT" --phase predict --shard 1
CUDA_VISIBLE_DEVICES=3 $PY $TOOL/audit_large_mesh_contact.py --output "$OUT" --phase predict --shard 0
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_large_mesh_contact.py --output "$OUT" --phase measure
CUDA_VISIBLE_DEVICES=0 $PY $TOOL/audit_goal_chain.py --output "$OUT" --phase reference
CUDA_VISIBLE_DEVICES='' $PY $TOOL/summarize_goal_chain.py --output "$OUT"
```

GPU entries require a fresh idle check and the same0–3 authorization. Commands above describe
reproduction/resume; launches used detached setsid via `Popen(start_new_session=True,stdin=DEVNULL)`
and narrow public runtime environment only, TMPDIR inside project. Actual argv/PIDs/start/end
are in `launch-*.json`, `complete-*.json`, `mesh-controller*.json`. GPU3 was claimed during
smoke and laterGPU2 acquired concurrent jobs; only owned primary PID2629688 was interrupted,
then exact completed batches resumed on3 (PID2639649). The GPU0 worker (shard1) continued uninterrupted (PID2629690).
All owned GPU tasks completed; no active diagnostic workers remain. No other jobs were controlled.

Pinned baseline/FМ identities and exact normalizer/finite checks are in the parent
`identifiability-audit-20261022-r1/checkpoint-audit.json`. Additional receipts are local
run manifests, not strict new content-hash infrastructure. Shared-workspace unrelated
modifications were preserved. Root verify retains6 pre-existing trajectory-policy card
schema failures; no audit-card/seed failure. Targeted tests9PASS1.97s.

## 2026-10-22 负结果 CPU 审计：结论范围修订

保留completion绑定主推理c971bca6的离线goal语义差异、条件敏感性、DEV泛化差与几何诊断。此处Planner是有oracle录制参考的路径子模块，E_next仍固定录制标签，缺完整抓稳/释放/支撑反馈；不是完整部署闭环。healthy与FM训练步骤/目标不同，K8是同模型抽样而非8个独立训练seed。禁止由本卡推出v6控制失败、FM架构因果劣势或τ不可辨识；离线相对距离/皮肤运输proxy也不等于真实手无穿透。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
