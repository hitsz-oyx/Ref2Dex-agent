---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref23-multitakeover-fit
experiment_id: P-20261012-goal-point-wam-ref23-multitakeover-fit
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: bdc6abad
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

Engineering smoke completed42.12s/GPU2, peak2116MiB, one update/no checkpoint.
Predicted-action gradient into R0, GT-anchor into R1020.01; all four WAM
action-only gradients nonzero. Frozen weights/buffers unchanged.21 admission,
sampling and gradient-ownership contracts pass. Independent actual-source
label/256metric audit passes (PD0/tau1.19e-7m/metric0). Auditor resolves the
recovered t8 symlink to its original completed trajectory; no data changed.

### Completed fixed continuation

`wam-ref2-joint-multitakeover-20261012-r1` completed1000 updates471.29s,
peak2149MiB (device process ~2753MiB). Short utilization sample mean22.4%/
max43%; throughput ~2.3 updates/s remained steady, consistent with the prior
full policy-chain recipe. No performance or research variable changed.
13/13 original offline screens pass; initial hashes exactly equal Probe30
final policy/R/frozen/normalization hashes. All643 aggregation rows actually
appear in source-balanced batches. R predicted-action gradient0; GT gradient
567.70, four WAM action gradients nonzero.256 independently reconstructed
metrics/PD labels max0, tau max1.19e-7m; buffers/action/world unchanged.

| Offline development/TRAIN panel | Warm Probe30 | Final |
| --- | ---: | ---: |
| Nominal predicted PD XYZp95 mm | 22.67 | 31.21 |
| Nominal GT-tau R PD XYZp95 mm | 13.35 | 11.58 |
| Startup frame0 predicted PD XYZp95 mm | 5.82 | 6.91 |
| Startup all predicted PD XYZp95 mm | 8.31 | 13.56 |
| t8 actual-future predicted PD XYZp95 mm | 42.16 | 32.30 |
| t16 actual-future predicted PD XYZp95 mm | 63.49 | 34.34 |
| t32 actual-future predicted PD XYZp95 mm | 422.17 | 254.79 |
| t32 actual-future first8 EPE mm | 60.72 | 22.79 |
| t32 actual-future GT-tau R PD XYZp95 mm | 403.14 | 257.80 |
| Pure current-PD XYZp95 mm (each source) | 658.80 | 435.11 |

Late TRAIN geometry/action improve but large action errors remain even for
GT tau on t32 states. Nominal/startup predicted PD worsens while still passing
original limits; retaining physical mapping does not mean no retention cost.
All these aggregation panels are TRAIN, never held-out evidence.

### One new actual student rollout

`wam-ref2-execution-multi-20261012-r1` completed128 controls/129 states,
97.67s/GPU2. Same exact newly learned WAM/R pair in both pinned runtimes;
independent native audit2316 checks/max0 PASS. Teacher/GT-R held73/75 with
zero clipping; the NEW R earns only this local short-grasp calibration.
Pure actor held0 with clipping0% (Probe30 5.47%). t16 expert takeover held75
but expert commands107/108 clipped (2/128 total); its strict local recovery
flag is false. Do not call that row unclipped or grant all its futures training
admission. No second fresh rollout or training sweep was performed.

| Tick | Probe30 actual PD error mm | New PD error mm | Probe30 wrist deviation mm | New wrist deviation mm |
| --- | ---: | ---: | ---: | ---: |
| 8 | 12.93 | 23.90 | 6.43 | 11.39 |
| 12 | 21.98 | 20.19 | 9.33 | 15.82 |
| 16 | 13.90 | 28.28 | 10.03 | 18.95 |
| 24 | 265.83 | 108.10 | 65.15 | 49.71 |
| 32 | 257.46 | 74.90 | 354.10 | 63.17 |
| 48 | 530.24 | 328.07 | 522.77 | 105.09 |

Local late-drift magnitude reduction is **PROMISING** (t24 PD -59.3%,
t32 wrist -82.2%); autonomous usefulness remains **UNCLEAR**, held0.
Predeclared composite4/5: t8/t12 early retention and both late-halving checks
pass; t16 PD28.28mm exceeds its unchanged2x limit27.80mm. Thus the full
predeclared promising gate FAILS, despite the late signal. Do not move the
threshold or describe the policy as successfully retained. Drift persists
at t24 and later; onset delay or successful distribution coverage is not proved.

Decision: retain this candidate and evidence, preserve the Probe30 baseline,
and stop this fixed experiment. The changed late behavior does not justify
an immediate architecture/goal change or a loss sweep. Next experiment should
address remaining later visited-state supervision and early retention using
this newly visited distribution, with its own objective/budget; no additional
experiment started here. Independent training-source/gradient/native checks
confirm the intended intervention, without proving observation sufficiency.

Raw reports: fit `result.json`, `predictions.npz`, `root_joint_audit.json`;
native `short_rollout_comparison.json`, `closed_loop_decision.json` and
`root_recovery_audit.json`. New stage outputs/caches~228MB, cumulative ref2
804.52MB<850MB; outputs+tmp292.80GB<400GB. GPU2 released. Scientific
conclusion remains scoped to one-motion/reset development, no formal validation.
