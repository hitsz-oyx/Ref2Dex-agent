---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-scene-thaw
experiment_id: P-20261012-goal-point-wam-ref2-scene-thaw
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: d56cfa31
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 23
seed_pool: probe
seeds: [297, 298, 292]
decision_changed_if_positive: admit one frozen unassisted closure after strict scene-patch loading and nominal retention audit
decision_changed_if_negative: distinguish remaining fit or recovery coverage limits before expanding thaw depth
status: UNCLEAR
run_id: wam-ref2-head-scene-thaw-20261012-r1
---

# Partial PTv3 output-stage adaptation on audited recovery PD

## Decision Note

User suggests progressively unfreezing more than the prediction head. Fixed
head-only approach correction reduced train XYZ p95 136.61->44.28mm but
validation404.15->384.46mm; nominal PD worsened25.56->82.83mm. All frozen
hash/gradient/label audits passed. Independent per-state tau optimization
through the same frozen R reached train/val0.61/0.87mm in100 steps/2.40s,
with no zero-gradient rows or global bound exclusions. This demonstrates
local numerical PD reachability, not a valid trajectory or an actor: tau
point changes p95=197/247mm, unconstrained and privileged.

Root chooses one small continuation updating the scene's final PTv3 decoder
stage plus current trainable head. Architecture and input contract stay the
same. It tests practical partial adaptation, not a causal unfreezing effect
versus matched head-only continuation. If it repairs live fit while retaining
nominal control/trajectory metrics, consider one unassisted closure; otherwise
inspect missing state coverage/observability and do not blindly thaw all PTv3.
One GPU2, smoke1 update/no checkpoint plus100 fixed updates and optional
closure<=600s, all ref2 outputs/private scratch<=100MB. No new branch, real
robot or external modification. No further permission boundary is needed.

## Fixed fit

Warm-start exact approach-head3b879329acef50f24fc7eb07338492f06462e01b2ecb9377480b9abffde51aaf.
Same audited early query training128/validation32 and nominal teacher pools
as experiment22; do not use late failed placement labels or mixed-policy tau.
Unfreeze ONLY `scene.backbone.dec.dec0` plus trajectory/hand_query. Other
scene weights/buffers, goal/action/world/R remain fixed. Keep eval mode and
deterministic PTv3 serialization; gradients enabled on selected parameters.
Compute full original scene forward for each microbatch, cache raw inputs
only. No detached scene-token cache or mutable sparse/unpool Point reuse.

Seed297/model,298/draw;100 fixed updates/final only, AdamW wd.01/clip1.
Learning rates dec0=2e-5 and head=1e-4. Each effective batch128 comprises
64 nominal rows (16 startup+48 later) and64 live rows. Four nominal/live
microbatch16 pairs accumulate gradients, each group's loss divided by4.
Same nominal tau/.05 + nominal encodedPD/Rscale + live encodedPD/Rscale
Huber loss. Verify live-PD gradients into dec0, trajectory AND hand_query;
all frozen parameter/buffer hashes/gradients unchanged. Future expert query
observation remains label privilege, not WAM input.

Predeclared signal: validation PD XYZ p95 improves>=50% versus the parent,
nominal nearest8 EPE<=21.522mm (original startup17.217*1.25), and nominal
true-PD XYZ p95<=38.335mm (original startup25.556*1.5). The retention baseline
is the original startup, not the degraded parent. Preserve all raw outcomes;
no stage/checkpoint selection by physical success.

If all-pass, strict optional `scene_patch` contains exactly dec0 full-state
keys and authenticates schema/prefix/shape/dtype/finite/base/parent identity.
Save head+patch<10MB, no optimizer/base. Bridge must actually apply and
authenticate the patch before any closure. Default three roles/seed292/
542 controls; no teacher intervention in WAM. Calibration teacher/GT each
held>=45 plus controlled placement. WAM held>=45/clipping<1% PROMISING for
this initialization, zero held UNPROMISING, otherwise UNCLEAR; full task
readiness requires controlled placement separately.

## Cost and limitations

Entry `tools/run/fit_wam_visited_pd_thaw.py` under trajectory-policy. After
two updates project cost and reserve180s for optional closure; stop as an
ENGINEERING_BUDGET_STOP if projected total>600s, not a scientific negative.
Check actual GPU use and foreign PIDs; no source changes during execution.
Frozen world head does not preserve world outputs when shared scene changes.
Report descriptive scene/world drift on16 nominal robot inputs with zero/
invalid executed-action context; human/video/world quality remains unvalidated.
Single correlated rollout and partial stage do not establish generalized
robot adaptation, world preservation or Cm causal benefit.

## Results

Result: 100 updates improve train PD44.24->33.32mm but validation384.76->398.67mm; nominal PD46.82mm misses retention; UNCLEAR, no closure.
Decision: inspect unsupported pure-WAM recovery states and supervision/replay conflict before expanding thaw depth; do not thaw the full backbone on this evidence.

Engineering smoke completed1 update/17.95s; all three live-PD gradients
positive, frozen scene remainder and goal/action/world/R identities unchanged.
Actual fixed100 continuation completed109.46s on GPU2, peak757.88MiB
allocated. Runtime HEAD wasd56cfa31 (concurrent video documentation commit);
unchanged fit/helper code was introduced inc5df7848. No source drift occurred.

| Metric | Initial warm head, full forward | Final |
| --- | --- | --- |
| Live training PD XYZ p95, mm | 44.237 | 33.315 |
| Live validation PD XYZ p95, mm | 384.760 | 398.673 |
| Nominal true-PD XYZ p95, mm | 82.689 | 46.820 |
| Nominal nearest8 trajectory EPE, mm | 19.525 | 17.451 |

Query-halving FAIL; nearest8 retention PASS; nominal PD retention FAIL
(original startup limit38.335mm). Local conclusion UNCLEAR; no unassisted
closure admitted or run. The warm-head initial measurements use full-scene
microbatch forward, while the parent used a BF16 token cache and different
batch sizes; small numerical differences are retained in both reports.
This continuation does not isolate the causal effect of thawing: learning
rate, extra updates and draw order also differ from the prior fit.

Root independent checkpoint/array audit reproduced reported XYZ/rotation/
finger/nearest8 metrics and expert PD labels with max difference0; confirmed
128/32 disjoint live rows, absent live tau labels, all transitive input hashes,
exact44-key dec0 whitelist and final scene hash reconstructed from original
base plus saved patch. Goal/action/world hashes independently match the base;
the live PD gradient reaches dec0, trajectory and hand_query. Head8.968MB,
SHA8732c5876a5f240c457067a514bf2ad62cdcf44d839a20ad690e92da0e087c62.
Audit uses CPU for pure arrays and checkpoint statistics, no model inference.

The large validation error is concentrated in pure-WAM role2 ticks32:48:
XYZ p95433.06->435.20mm; recovered role3 ticks64:80 is31.93->33.77mm.
Training role2 ticks0:32 also fails to improve,45.94->50.23mm; training
role3 improves39.35->24.29mm. Aggregate training improvement therefore does
not establish recovery of autonomous-WAM states. These blocks are correlated
within one collection; this is not a generalized recovery estimate or proof
that additional capacity cannot help. Retain the fixed holdout unchanged.

Descriptive scene-token RMS change=.02237, world translation max change
.146mm/rotation-matrix max1.953e-4 on16 nominal robot inputs. Human/video/
world prediction quality was not validated. Strict optional-patch bridge
application/authentication was implemented separately in3b0d11f9 and17
CPU tests passed; this is engineering readiness, not physical execution.

[Result](../../../../../../outputs/trajectory-policy/wam-ref2-head-scene-thaw-20261012-r1/result.json),
[independent audit](../../../../../../outputs/trajectory-policy/wam-ref2-head-scene-thaw-20261012-r1/root_fit_audit.json).
Reproduce audit with trajectory-policy/tools/audit/audit_wam_scene_thaw_fit.py.
Smoke+fit127.42s; no closure consumed. GPU2 released. All ref2 artifacts
remain below100MB and shared monitor finished below300GB.
