---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-approach-pd
experiment_id: P-20261012-goal-point-wam-ref2-approach-pd
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: b002588c
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 22
seed_pool: probe
seeds: [295, 296, 292]
decision_changed_if_positive: evaluate one fixed unassisted head or a separately admitted partial-thaw continuation
decision_changed_if_negative: inspect training-state fit capacity and actor observability before another roll-in
status: UNCLEAR
run_id: wam-ref2-head-approach-pd-20261012-r1
---

# Early recovery PD supervision with nominal trajectory replay

## Decision Note

The original full-recovery gate remains unmet: the expert takeover held435
frames but failed placement. The original full-phase correction experiment21
is not run. However, true recovery from the precontact state produced stable
grasp by state99, with zero expert clipping, and independently audited live
queries. Root chooses a NEW, explicitly local approach hypothesis: early
visited-state PD labels may repair the currently failed approach and grasp
initialization, without asserting the expert can recover the complete task.
No late failed-placement labels enter this fit.

One idle GPU2, fixed300 updates and at most one closure, combined<=600s,
previous/new ref2 outputs/scratch<=100MB. Current ref2 artifacts42.13MB and
shared storage below298GB proactive/300GB hard bound. No new identity,
branch, external changes or real robot. Positive offline fit permits one
unassisted closure; poor training-state fit motivates a separately recorded
small partial-thaw Probe, following the user's staged-unfreezing suggestion.
Stop on finite/source/label/frozen-gradient/foreign-GPU/budget failures.

## Fixed protocol

Use `fit_wam_visited_pd.py --approach-only`, with the authenticated startup
head as warm start and audited recovery-r2/native source. Require full
teacher/GT calibration, takeover held>=45 with first stable tick<=128 and
clipping<1%, plus independent raw query/PD/prefix audit. This LOCAL admission
does not change the failed full-recovery gate.

Live pool is role2 ticks0:48 and recovered role3 ticks16:128. Reserve role2
ticks32:48 and role3 ticks64:80 as32 validation rows; remaining128 train.
Remove duplicated command-prefix row3 ticks0:16 and all failed late-phase
states. Correlated temporal holdout is diagnostic only. Real expert PD is
encoded against the same live q; live rows have no tau labels whatsoever.
Keep original qualified nominal teacher train env0:16/ticks0:519 and held
64-window report env24:28; test env28:32 remains untouched.

Each batch:64 nominal rows (16 startup +48 later) and64 live rows, all sampled
with replacement. Seed295/model,296/draw; these seeds were reserved but
unused by experiment21. Freeze scene/goal/action/world/R; train trajectory
and hand_query only. Same architecture, normalization, BF16 actor/FP32 R,
AdamW lr3e-4/wd.01/clip1. Fixed final300 only, no optimizer/base copies.
Loss is nominal Huber tau/.05 plus nominal Huber encodedPD/Rscale plus
live Huber encodedPD/Rscale. Verify live PD gradient through frozen R to
actor, both trainable modules, frozen hashes and cached/direct equivalence.

Report initial/final live TRAIN and validation PD XYZ/rotation/finger p95,
clipping and nominal nearest8 trajectory EPE, saving raw arrays. Preliminary
signal is validation PD XYZ reduced>=50% and nominal nearest8 EPE worsens
<=25%. If training XYZ p95 remains>20mm, it exposes a practical fit concern
and motivates checking partial thaw; it does not by itself prove backbone
capacity is the cause. No nominal-repeat control or causal DAgger claim.

Any admitted closure uses the default three-role frozen bridge, seed292,
542 controls and no intervention in WAM. Teacher/GT each require held>=45
plus controlled place. WAM held>=45/clipping<1% is PROMISING for the exact
initialization, zero held UNPROMISING, other outcomes UNCLEAR; full-task
readiness additionally requires controlled placement. A separately admitted
partial thaw is not automatically authorized by its mention here.

## Results

Engineering smoke passed1 update/13.45s with live-PD actor gradient425.43,
both trainable modules nonzero, exact frozen hashes and cache/direct0.
Fixed300 fit completed77.01s, GPU2 peak2476.56MiB, headSHA
3b879329acef50f24fc7eb07338492f06462e01b2ecb9377480b9abffde51aaf.
Live train PD XYZ p95=136.61->44.28mm, validation404.15->384.46mm.
Nominal nearest8 EPE17.22->19.51mm passes25% retention, but nominal PD
XYZ p95 worsens25.56->82.83mm. Query-halving gate fails; local conclusion
UNCLEAR. No head closure is admitted or run. Live gradient268.09; all five
frozen hashes unchanged/cache0. Saved raw arrays include training/validation
rows, targets, predictions and intended native actions.

Root Decision: before interpreting remaining error as backbone capacity,
screen the fixed R's PD range and per-state tau inverse. Its final Tanh->Linear
global interval contains all admitted expert targets (pure checkpoint/array
statistics on CPU; no network inference), so no global bound exclusion is
identified. Next cheapest GPU check optimizes independent tau variables at
the recorded train/validation states100steps, with the R and states fixed.
This is a privileged unconstrained diagnostic, never an actor or physical
trajectory evidence. Report actual PD and tau displacement; local optimizer
failure is not proof of global impossibility. Budget<=60s/<1MB, still inside
the shared600s/100MB allowance. Entry audit_wam_pd_reachability.py.
Only if this reveals useful local R sensitivity does partial scene thaw
remain a reasonable next check; its protocol is separate and not yet run.
