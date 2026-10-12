---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-visited-pd
experiment_id: P-20261012-goal-point-wam-ref2-visited-pd
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 21
seed_pool: probe
seeds: [295, 296, 292]
decision_changed_if_positive: retain the fixed correction only if separate teacher-free simulation establishes practical control progress
decision_changed_if_negative: diagnose expert PD fit versus actor input or frozen R range before another collection recipe
status: UNCLEAR
run_id: wam-ref2-head-visited-pd-20261012-r1
---

# Conditional live-state PD correction through the frozen retargeter

Decision Probe for the autonomous WAM trajectory-policy baseline. This is
conditional on successful tick16 expert recovery and independently audited
query admission in P-20261012-goal-point-wam-ref2-recovery. No fit is permitted
before both. Expert-assisted behavior cannot establish autonomous success.

Question: can true live-state expert PD targets reduce the measured compound
policy's errors while preserving nominal trajectories? If so, one teacher-free
closure decides whether to retain this adaptation; if not, inspect input
observability/frozen R representability. This is the cheapest bounded check
before expanding roll-ins, training human models longer or retraining R.

## Fixed fit

Warm-start only the authenticated600-update robot head. Preserve original
human10000 base implementation, scene/goal/world/action and physical R;
train trajectory plus hand_query only. Actor still consumes measured H4 and
planned object endpoint, without teacher reference hand/body/contact inputs.
Native actual PD targets are encoded with the SAME live current q, wrist6
relative and fingers absolute. R remains eval/requires_gradFalse but its
forward retains gradients to tau/actor. Independently check the live-PD
branch gradient, frozen hashes and cached/direct equivalence.

Nominal pool is qualified teacher env0:16,ticks0:519. Fixed held diagnostic
panel is the existing16 ticks x env24:28 (64); env28:32 remains untouched.
Live rows: unassisted WAM role2,ticks0:64; recovered role3,ticks16:519.
Remove role3's duplicated0:16 prefix. Reserve role2 ticks48:64 and role3
blocks64:80,256:272,448:464 as64 validation rows; all remaining503 rows train.
Splitting temporal blocks in one episode is a correlated diagnostic, not
independent unseen-episode validation. No physical/report model selection.

Each batch has64 nominal rows (16 uniform tick<16,48 uniform tick>=16) and
64 uniformly sampled live training rows, with replacement. Seed295/model,
296/draw, AdamW lr3e-4 wd.01, clip1, BF16 actor/FP32 R. Fixed300 updates,
final checkpoint only. Loss is nominal Huber(tau/.05) plus nominal
Huber(encodedPD/Rscale) plus live Huber(encodedPD/Rscale); live rows have
NO tau label or trajectory loss. Neither mixed-policy measured futures nor
source nominal futures are allowed as live trajectory labels. No nominal
future reaches actor inputs. This is correction usefulness, not a matched
causal comparison of DAgger versus equal-compute nominal continuation.

Predeclared offline signal: live validation PD XYZ p95 reduced>=50% and
nominal held-panel nearest8 EPE increases<=25%. All-pass permits at most
one ordinary542-control three-role teacher/GT-R/WAM closure, with no teacher
intervention in WAM. Calibration each held>=45 and controlled place required;
WAM held>=45 and clipping<1% is PROMISING for initialization, zero held
UNPROMISING for the exact checkpoint, otherwise UNCLEAR. Controlled place
is separately reported and required before claiming full nominal-task use.

## Resources and implementation

Entry `src/task/trajectory-policy/tools/run/fit_wam_visited_pd.py`.
GPU2 only; engineering smoke<=1 update/no checkpoint; fixed fit plus optional
closure<=600s; total previous/new ref2 artifacts/scratch<=100MB. Cache only
RAM/GPU, head-only checkpoint<10MB, no optimizer/base copy/video. Stop on
nonfinite, source/native/controller drift, foreign GPU process or label/gradient
audit failure. Default bridge three-role behavior remains the prior protocol.
Shared storage proactive298GB/hard300GB; current296.51GB. No new branch,
remote training or external source modification.

## Results

Not started; waiting for expert recovery and independent query admission.
See [design/source review](../../../../trajectory-policy/docs/research/20261012-wam-visited-pd-design.md).
