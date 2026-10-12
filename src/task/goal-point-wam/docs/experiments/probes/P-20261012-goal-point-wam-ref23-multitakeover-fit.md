---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref23-multitakeover-fit
experiment_id: P-20261012-goal-point-wam-ref23-multitakeover-fit
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 32
seed_pool: probe
seeds: [295, 296, 292]
decision_changed_if_positive: retain actual multi-takeover adaptation and expand successful student-state coverage
decision_changed_if_negative: reconsider observation and goal sufficiency after checking this fit implemented the intended supervision
status: UNCLEAR
run_id: wam-ref2-joint-multitakeover-20261012-r1
---

# ref2_3: anchored multi-takeover continuation

## Decision Note

User ref2_3 explicitly requests fitting the643 audited rows from Probe31 and
one new student rollout; no horizon, architecture, loss-coefficient search or
new diagnostic experiment. This Decision Probe serves self-trained manipulation:
can actual later student-state expert support reduce t24/t32 drift while keeping
early control? Probe30 improved early closed loop but actor0held persists.
Coverage is the hypothesis being tested, not an established exclusive cause.

Cheapest test: one fixed1000-update warm continuation of the exact Probe30
WAM/R pair, batch64/seeds295+296/GPU2 and original learning rates, optimizer,
clip and K24. Startup16+later16 nominal teacher samples; actual future16 and
pure current-PD16, each source-balanced5/5/6 across8/16/32 (extra source drawn).
All259 actual futures and384 current-PD rows are TRAIN. No borrowed or failed
pure future geometry. WAM receives valid geometry and all predicted PD loss
through R with fixed parameters. R receives ONLY valid GT tau->PD loss.
Equal three loss weights. Scene/goal/trajectory/hand_query/R.net train;
action/world/statistical buffers remain frozen.

Original nominal/startup screens unchanged (13 gates, including nominal
predicted<=40mm, GT-R<=30mm and startup0<=20mm). If passed, one new128-control
four-role t16 takeover group using the newly learned exact pair; pure role2
is the autonomous student being assessed. Expert takeover row is a training-time
control and does not establish student success. Original qualifications do not
transfer to the new R. Record t8/12/16/24/32/48 actual current-PD error and
same-trace teacher wrist deviation, own h1 prediction error, clipping and held.
Descriptive comparator is Probe30 actual128/t8 trace (same motion/reset,
independent live launch); pure policy role2 is unaffected by takeover time.

Predeclared promising closed-loop signal: preserve early t8/12/16 PD errors
within2x Probe30 and halve BOTH t24 PD265.83mm and t32 wrist354.10mm;
report any held>0 separately. These are exploration decisions, not causal or
full placement validation. If screens fail, don't deploy; if fresh trajectory
still diverges, do not repeat tuning under this card. Review implemented
supervision before attributing a negative result to observation/architecture.

Resources: GPU2, fit<=1800s, fresh simulation<=600s, total<=40GPUmin,
new outputs<=250MB, cumulative ref2 outputs/caches<=850MB, global400GB.
Current cumulative576.17MB; expected addition~220MB. Engineering1 writes no
checkpoint. Final-only pair, no overwrite, main/no new branch or external
writes. Stop on nonfinite, unexpected device process, budget or actual input
change. Runtime compatibility proof for calibrated legacy K24 is retained;
no expanded hashing audit beyond executable contracts needed for this run.

## Entry points and inputs

- `src/task/trajectory-policy/tools/run/fit_wam_multitakeover_anchored.py`
- `src/task/trajectory-policy/tools/audit/audit_wam_multitakeover_fit.py`
- `src/task/trajectory-policy/tools/run/launch_wam_native_bridge.py`
- `src/task/trajectory-policy/tools/audit/compare_wam_short_rollouts.py`

Warm pair: `outputs/trajectory-policy/wam-ref2-joint-anchored-20261012-r1/`.
Audited aggregation: `outputs/trajectory-policy/wam-ref2-multitakeover-20261012-r1/`.
Original nominal teacher/base scene/canonical object/reference goal unchanged.
Input sample features consume the aggregation's actual state87/tau/current-q/PD;
own measured H4 and known object goal use each corresponding native row.
Nominal/startup reports keep their existing development environments and ticks.
Old recovery/pure panels are retained only as descriptive development reports;
all training recovery/current draws come from the new643 rows.

## Limitations / future evidence

Single seed, one motion/reset, overlapping teacher windows and privilege in
TRAIN labels. No held-out aggregate rows claimed. More successful expert
recoveries cannot establish autonomous policy improvement. Training loss is
not the decision metric; new actual rollout is required. Full placement,
independent motion/reset generalization and final matched Cm-on/off remain
deferred evidence. No world/evaluator/RL begun.

## Results

Pending engineering smoke, fixed fit and one admitted fresh rollout.
