# Corrected causal calibration does not identify pretrained utility

P-20261003-surface-calibration completed as r2 audit recovery: UNPROMISING.
r1 at d47e1fe completed all four matched fits, then failed in the independent
SDK comparison due to NumPy advanced-index axis order. r2 at7709263 inherited
those exact fits/data and only ran the corrected CPU audit. No training,
checkpoint selection, tolerance or prospective gate changed; r1 FAILED and
traceback remain preserved. Actual optimizer updates total4800, all in r1;
r2 adds0. No new native physics or policy calls.

## Decision-changing result

Same corrected655 episode split as execution qualification:384train and384
disjoint held whole episodes,6144windows each. Held episodes have previously
been examined; this is a same-seed diagnostic screen, not fresh validation.
Retained train-only causal actuator, fixed10135hand samples/64object queries,
22features. Every arm freezes11200encoder parameters and trains the SAME
NEW8451parameter persistence-residual head,1200updates,commonhead4101 and
batchschedule4102. Normal and shuffled pretraining originally share7168
windows/1500updates/init. Scratch uses original random encoder, also frozen.
Hand-flow removal zeros six local/global predicted-motion columns in both
train/eval, retaining current geometry/state and the same persistence anchor.

| Condition | Held episode-mean object EPE/mm |
| --- | ---: |
|Normal MANO pretrained encoder|0.665444|
|Matched shuffled-target pretrained encoder|0.618116|
|Frozen original random encoder|0.633220|
|Normal pretrained, predicted hand flow removed|0.790197|
|Explicit persistence|0.629373|
|Zero object flow|0.660945|

Normal pretrained error is5.731%higher than persistence,5.089%higher than
scratch and7.657%higher than shuffled. It improves15.788%over hand-flow
removal, but that isolated gate cannot demonstrate pretrained knowledge
utility. Four fixed gates: persistence=false,scratch=false,shuffled=false,
hand_flow=true. Overall UNPROMISING exactly follows the prospective rule.
Calibration fixes most of the frozen decoder's large numerical/domain gap
(previous causal MANO7.346mm), without establishing useful pretrained features.
That cross-experiment contrast is descriptive, not a matched head comparison.

Predeclared diagnostics, independently reaggregated:

| Group | Windows/episodes | Normal/mm | Shuffled/mm | Scratch/mm | Removed/mm | Persistence/mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|Motion0|2048/128|0.318510|0.175292|0.204546|0.384021|0.116748|
|Motion1|2048/128|0.854958|0.858990|0.848028|1.058051|0.982376|
|Motion2|2048/128|0.822863|0.820066|0.847087|0.928518|0.788995|
|Near hand|2124/256|2.807138|2.790970|2.814123|2.867952|2.823930|
|Far from hand|4020/384|0.430890|0.300552|0.310921|0.681923|0.519538|

All actor-arm diagnostics retained in raw results. Near means current64query
unsigned minimum distance<2cm, not measured contact. Each subgroup separately
weights available episodes equally; no window-weighted reconstruction or
subgroup override of the all-window primary. Near normal gain0.595%over
persistence is small and shuffled remains better. Motion1 improvement is
shared with controls, not an identified pretraining advantage.

## Audited implementation and cost

Both source row banks independently rebuild12288raw rows/splits/timestamps
exactly; all6144new train predicted joint states rebuild from inherited
actuator coefficients.384predeclared first train windows receive independent
SciPy URDF FK/current-neighbor/coordinate/label reconstruction. Other train
geometry rows are not independently reconstructed. Held geometric audit
inherits the earlier completed run by hash. Five native body origins:
max5.307644e-5m (0.053mm). Feature max1.203615e-5 and target max1.185246e-5
normalized units; nearest-distance max1.587150e-7m, fixed tolerances pass.

ALL four held prediction banks independently recompute with NumPy encoder/
head, max5.047768e-7normalized flow. Parent metrics max1.997460e-9mm;
all per-parent/subgroup/baseline/gate checks pass. Common seed4101 head is
independently rebuilt, encoders exactly unchanged, no encoder gradient,
same schedule and actual1200optimizer-step states in each arm.
Independent NumPy gradients/AdamW replay first3steps of eacharm (12total),
parameter max2.796058e-8,loss max9.825269e-9; remaining4788steps not replayed.
Complete optimizer/schedule/loss/head/encoder states retained.

The existing authorized read-only supervisor independently confirms the
audit-only repair, strict recovery hash contract, all group metrics and
UNPROMISING classification; maxmetric discrepancy<2.9e-7mm.1464protected
inputs unchanged. This review initially missed the NumPy indexing bug;
successful audit recovery, not a claim of bug-free preflight, admits results.

r1 wall63.193289s (smoke3.918s, GPUgeometry28.520s, GPUfit22.340s,
failedCPUaudit2.421s); r2CPUaudit recovery wall14.405061s. Combined77.598349s,
74346217unique bytes across both runs, well below900s/1GiB. Freshly idleGPU6
used main preparation/fitting, no CPU main-model fallback. Parent592204/
595029 and children592379/592562/593265/593748/595088 all verified absent.

## Interpretation and next decision

Close this exact fixed-budget frozen-feature calibration and the explored
surface-feature transfer family, with the stated representation/treatment
scope. Do not increase its calibration budget, add seeds, tune gates, choose
a favorable subgroup or repeat granularity/scale to rescue the failure.
This does not reject arbitrary Cm, future trainable encoders or all
hand-domain priors. Current hand flow remains a useful input contrast here;
its predicted presence is distinct from pretraining knowledge and policy
benefit. No formal Validation or learned-policy utility follows.

Next representation review concerns physically constrained hand-to-object
rigid transport/coupling, with an explicit inertia baseline. Before any new
pretraining or policy fit, a cheap representation-capacity screen must ask
whether causal hand-transport candidates can even explain additional measured
object motion beyond matched state-only candidates. Oracle outcome-fitted
weights would be a capacity bound only, not deployable Cm or causal evidence.
[Decision](../decisions/D-20261003-after-surface-calibration.md).

Evidence: contact_response/output/P-20261003-surface-calibration-r1 (all
source-derived geometry,4800updates/checkpoints/predictions,failedaudit log)
and r2 (immutable inheritance contract,successful independentaudit/results).
[Prospective card](../experiments/probes/P-20261003-surface-calibration.md).
Mission unchanged; goal ACTIVE; journal NOT READY.
