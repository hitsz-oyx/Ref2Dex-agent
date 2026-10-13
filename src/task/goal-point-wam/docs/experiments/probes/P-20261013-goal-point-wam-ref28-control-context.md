---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref28-control-context
experiment_id: P-20261013-goal-point-wam-ref28-control-context
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 8034be01
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 39
seed_pool: probe
seeds: [292, 295, 296]
decision_changed_if_positive: retain robot conditioner and qualify its closed-loop grasp before further scaling
decision_changed_if_negative: retain C and diagnose contact recovery without rerunning large geometry pretraining
status: UNCLEAR
run_id: wam-control-context-20261013-r1
---

# Robot control context and current C-state recovery

Result: IN PROGRESS; no observation sufficiency claim established.
Decision: Follow user ref2_8 with a small robot adapter, actual current C takeovers,
and bounded finetuning. Preserve geometry pretraining, raw-hand interface and Chunk-R.

## Motivation and fixed design

Decision Probe for C3 manipulation: does robot-only proprioception and temporal
object reference conditioning improve the contact transition of the existing C
actor? Chunk-R already receives q/dq; this intervention adds context to WAM.
No attribution between the two new inputs or Cm-on/off causal claim is attempted.
The cheapest useful evidence is current C own-state expert recovery at32/40/48,
adapter-only contact panels, then a matched-reset native rollout.

48D input is measured q18, dq18, previous actually applied ACTIVE12 command;
at reset previous command is zero. 27D goal is inv(current own object pose)
times planned reference pose at+8/+16/+24, each translation3 plus rotation6
(first two rotation matrix columns, column-major). MLP48/27 ->128 ->128,
two scalar gates initialized exactly0; residual added to existing decoder
queries after raw-hand conditioning. Human conditioner is masked to exactly0.
No actual future object pose, expert future hand, task1442D or contact force
is admitted as an actor observation. World remains goal-blind.

First collect each C-final takeover with128 controls, four native roles and
seed292. Only row3 actual applied expert futures are admitted after takeover;
audit prefix equivalence and require each admitted24-step window unclipped.
Do not substitute another row's future for failed late recovery.

Stage1 freezes all existing modules including decoder and R, trains adapter
only for2000 updates at1e-4, robot draws nominal/recovery3:5. Stage2 only if
each recovery contact panel (first16 post-takeover queries) has first8 and
first-PD XYZ p95 <=90% baseline and nominal PD/clipping do not regress beyond
10%+.1mm /1 percentage point. Stage2 adapter1e-4/decoder1e-5, exact
20% human/30% nominal/50% new recovery, initially5000 updates; extension up
to20000 only if contact panels continue improving. Evaluate every500 updates.
Supervision retains C trajectory + qualified Chunk-R control-aware loss;
World/action/R frozen. Frozen-context caching is allowed, labels remain own.
Initial equivalence is checked with identical dtype/context and zero gates.

One idle GPU for each sequential collection/training, max6 globally; collection
<=600s each, training <=2h, new artifacts <=2GB, total outputs+tmp <=400GB.
Stop on nonfinite output, incorrect frame/previous-action semantics, frozen
state drift, foreign GPU process, budget, or two consecutive gross contact
regressions. Same main branch; no new branch or remote push.

Closed-loop success uses unchanged >=45 consecutive held frames (gap<=10mm,
height>=30mm and unsupported), teacher/GT calibration and actual clipping;
full placement reported separately. A single reset is only a Probe.

## Inputs and outputs

- User design: ../../user/ref/ref2_8.md
- C-final: outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt
- Warm policy: outputs/trajectory-policy/wam-ref2-joint-multitakeover-20261012-r1
- Qualified Chunk-R: outputs/trajectory-policy/wam-v2-control-fit-20261012-r1/chunk.pt
- New recovery: outputs/trajectory-policy/wam-ref2-execution-C-t{32,40,48}-20261013-r1/native
- Adapter training: outputs/goal-point-wam/wam-control-context-20261013-r1

## Limitations and future evidence

New recovery windows are TRAIN diagnostics, not independent generalization.
Single seed/reset and joint proprio/reference intervention cannot prove
observation insufficiency or Cm causal benefit. Matched input ablations and
multi-reset validation are deferred until this Probe changes the route.

## 2026-10-13 session continuation: evidence review

The prior session completed the C-final t32 and t40 takeover collections;
t48 and adapter training have not completed. Both 128-control collections
retain teacher/GT-Chunk/unaided-C held counts73/75/0. Actual expert takeover
holds75 frames at t32 and74 at t40, with zero executed clipping. These are
local recoverability signals, not autonomous actor success or full placement.
The existing t32 audit passes; continuation ran `audit_wam_recovery.py` on
t40, also passing2316 checks with numerical max difference0 and physically
comparable replay prefixes. The rows are not exact hidden-state forks.

Read-only inspection finds73/65 complete unclipped24-step expert windows
for t32/t40 (query ticks32..104 /40..104); these overlapping windows are
not independent episodes. Their futures belong to the actual takeover row.
The same-state expert query in the t40 trace gives first-PD XYZ disagreement
with unaided C of51.64/119.96/120.46/353.30mm at ticks32/36/40/48.
Corresponding sampled hand-object gaps at32/36/40 are864.10/654.77/298.48mm:
control disagreement is already present before close contact. The first
executed actor clipping is tick92, so clipping does not explain its onset.
Expert disagreement diagnoses imitation/control mismatch, not a unique
optimal-action requirement or proof that missing proprioception is causal.

Route recommendation: preserve C and the qualified R, complete current-state
recovery coverage and test the bounded robot adapter before another scale-up.
Once testing decoder adaptation, compare the same new data/update budget
with and without the extra context to distinguish data from input benefits.
Score a fresh closed loop after the short fit; do not admit long training
solely from recovery TRAIN error or frozen-world gap. The adapter jointly
adds proprioception and planned reference, so a positive result alone cannot
attribute benefit to either input. No extra collection or model training was
performed during this evidence review; original protocol remains unchanged.

- [t32 result](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-t32-20261013-r1/native/result.json)
- [t40 result](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-t40-20261013-r1/native/result.json)
- [t40 audit](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-t40-20261013-r1/native/root_recovery_audit.json)
