---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-scene-thaw
experiment_id: P-20261012-goal-point-wam-ref2-scene-thaw
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
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

Not started; CPU patch admission tests passed, pending fixed commit and GPU smoke.
