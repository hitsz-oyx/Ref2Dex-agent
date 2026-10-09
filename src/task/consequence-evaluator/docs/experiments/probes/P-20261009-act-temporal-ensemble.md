---
schema: ref2dex.probe.v2
probe_id: P-20261009-act-temporal-ensemble
experiment_id: P-20261009-act-temporal-ensemble
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 6751843
claim_id: C3
hypothesis_family: HF-consequence-act-proposal
probe_index_in_family: 4
seed_pool: probe
seeds: [282]
decision_changed_if_positive: retain causal temporal aggregation as an ACT behavior route before any retraining
decision_changed_if_negative: retain the implementation but diagnose deployment coverage rather than claiming hard switching is the sole cause
status: UNPROMISING
run_id: act-temporal-ensemble-20261009-r1
---

# Does inference-only temporal aggregation recover ACT grasp behavior?

Result: UNPROMISING aggregation-only remedy: held478 open_loop24; held0 receding8/overlap8/temporal1, with passing native execution checks.
Decision: Keep tested inference modes; prioritize deployment-history diagnosis without further aggregation-only launches or retraining in this Probe.

## Decision and scope

User ref6 authorizes this bounded inference diagnosis, replacing the previous
pause on ACT behavior screens. It serves the self-trained manipulation baseline
subgoal; no Cm utility, Gate1, or full task success claim follows. No model is
trained, and Y, actor, native PD controller, reference bank and checkpoint are
frozen. Root chooses the cheapest discriminating probe: existing-data overlap
and horizon audits followed by four native behavior modes.

Ranked falsifiable hypotheses: (1) hard chunk switching causes control
discontinuities and aggregation restores held45; (2) first-eight horizon errors
dominate and horizon mixing improves the executed control; (3) deployment
coverage is insufficient and aggregation reduces discontinuities without
recovering grasp. Negative behavior does not by itself prove insufficient
capacity or close the broader ACT route.

## Fixed contract and budget

- Freeze the exact historical r1 checkpoint with SHA256
  `ed26abd5a02b9c706750b94ea28100f028564ee4bfd9054fabf3eb6a54c3d707`,
  including its stored standardizer. It predates corrected native RMS clip
  metadata; do not silently change normalization or retrain.
- Offline: original 17 train / 4 held-out clean episodes, stride1 predictions,
  horizon0--23 error and same-absolute-time overlap8 disagreements. Existing
  r2/r3 reactive teacher streams separately measure deployment error. No
  train/validation selection or new label is generated.
- Behavior: seed282, four environments with64 actor copies each, fresh native
  GPU PhysX/GPU tensors, start at tick0, full542 controls, original environment
  spacing. Same role layout for all modes: teacher / ACT / repeated teacher /
  repeated ACT. ACT repeat receives the same aggregated controls as ACT.
- `open_loop24` and `receding8` keep legacy hard dispatch; `overlap8` changes
  aggregation only at the same8-step query schedule; `temporal1` additionally
  queries every step. Weights follow [official ACT source](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py):
  oldest covering prediction first, `exp(-0.01 * rank)`, normalized. Timestamp
  membership preserves legitimate zero-valued controls; no future query.
- One idle GPU2; no more than four542-step launches, each bounded by150s after
  worker setup; offline audit180s. Total probe target below15minutes GPU time,
  outputs below1GB. Preserve failed outputs; stop on input/shape drift,
  nonfinite controls, early termination, or resource conflict. CPU is used only
  for array statistics and plotting; model inference uses GPU.

## Interpretation and checks

Primary symptom: native ACT reaches at least45 consecutive geometric held
frames; report maximum lift and full task outcome separately. A teacher failing
held45 invalidates behavior attribution for that launch. These are fresh
processes with uncontrolled hidden PhysX/contact-cache state; neither cross-arm
nor repeated-role differences are exact same-state causal evidence. The
historical open_loop24 packet has sustained lift, but **task success is false**
because terminal settle15 was absent.

Record every query tick/raw input and predicted chunk, independently reconstruct
the absolute-tick weighted control, compare requested vs native captured actions,
verify causal history, active count, same ACT/repeat stream and fixed checkpoint.
Offline same-time disagreement is not itself an executed control jump; compare
it to natural consecutive teacher-control change and inspect early contact ticks.
All artifacts remain engineering-only and non-trainable.

## Reproduction and artifacts

`tools/audit/audit_action_chunk_execution.py --behavior-packet ...
--require-held45` is the fast recorded native-dispatch symptom check. The old
r3 trace reproduces hard dispatch exactly and fails held45. A changed inference
policy requires a fresh native behavior screen; replaying stored failure does
not simulate a remedy.

Tools:

- [Offline/dispatch audit](../../../tools/audit/audit_action_chunk_execution.py)
- [Native behavior worker](../../../tools/run/run_gate1_gt_progress.py)
- [Executor tests](../../../tests/test_action_chunk_execution.py)

Outputs: `outputs/consequence-evaluator/act-temporal-ensemble-20261009-r1/`
for frozen screen manifest; separate run directories
`act-temporal-{open_loop24,receding8,overlap8,temporal1}-20261009-r1/`
preserve native packets and logs.

## Results

The native execution source was frozen at `6751843`; its hashes and exact
commands are retained in the screen manifest. Audit improvements are committed
through `c29ea82`. Four full542-step workers completed in about328s total wall
time, on GPU2 (roughly18.2GB memory, observed19--21% utilization). No actor or
proposal training occurred. GPU2 was released after the screens.

### Offline overlap and horizon checks

Retained audit/figure:
`outputs/consequence-evaluator/act-temporal-offline-20261009-r4/`.
All11,937 stride1 windows and the separately marked stride8 query grid were
evaluated. On four held-out clean episodes:

| query grid | windows | whole-chunk MSE | horizon0--7 MSE | horizon8--23 MSE |
| --- | ---: | ---: | ---: | ---: |
| every step | 2076 | 2.02935e-4 | 2.12462e-4 | 1.98171e-4 |
| every8, actual receding8 grid | 260 | 1.60673e-4 | 1.59940e-4 | 1.61040e-4 |

The stride8 total reproduces the original held-out MSE. The first-eight
average is approximately equal to the remaining horizons on the actual query
grid; a roughly7% difference on every-step inputs must not be used to explain
receding8. Temporal1 also queries observations outside the original training
stride8 grid, which is an additional limitation of that arm.

At stride8 query boundaries, the held-out mean same-absolute-time wrist
translation disagreement is15.40mm (17.86mm over the first120 ticks), versus
15.48mm natural consecutive teacher-control change at those boundaries. Wrist
rotation disagreement is0.0383rad; active-finger normalized L2 is0.02148,
versus teacher change0.01688. These are control/prediction differences, not
measured hand displacement or a stand-alone root-cause verdict. Stride1
overlaps yield18.92mm and are kept separately in the audit.

The first offline r1 attempt completed numeric inference but stopped on a
missing matplotlib dependency in the archived runtime. It is preserved as an
engineering failure. r2--r4 used the existing Torch2.4 GPU environment with the
same weights/normalizer; original stride8 metrics reproduce to numerical
precision. No dependency was installed or historical artifact replaced.

### Native behavior and execution checks

Independent native-dispatch audit:
`outputs/consequence-evaluator/act-temporal-final-audit-20261009-r1/audit.json`.

| mode | queries | max covering chunks | ACT max lift m | ACT held frames | teacher held frames | early120 wrist change at8-step boundaries mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| open_loop24 | 23 | 1 | 0.59524 | 478 | 484 | 30.10 |
| receding8 | 68 | 1 | 0.00000 | 0 | 297 | 33.56 |
| overlap8 | 68 | 3 | 0.00000 | 0 | 484 | 19.23 |
| temporal1 | 542 | 24 | 0.00000 | 0 | 484 | 8.58 |

Every arm completed542 controls with no early done. Requested versus captured
native control difference is exactly0 in all four launches. Independent NumPy
reconstruction of the covering-chunk weighted ACT action differs by at most
`2.24e-8`; raw query histories match the recorded pre-action frame exactly.
ACT/repeated-ACT streams are identical; repeated ACT also held478/0/0/0.
All four teacher arms pass held45. Initial ACT raw history, first8 controls,
and hand geometry through tick8 are exactly equal between open_loop24,
receding8 and overlap8; the checkpoint/backend/controller remain fixed. This
does not establish equality of unexposed PhysX solver state.

All ACT **full-task outcomes are false**, including open_loop24, which lifts
and holds but lacks terminal settle15. Held478 is not full task success. The
recorded symptom assertion goes green for open_loop24 and remains red for
receding8/overlap8; native history/weight/action checks pass independently of
that behavior failure.

## Decision after the screen

Temporal aggregation was missing from the previous inference implementation;
the new timestamp-based executor fixes that omission and demonstrably reduces
executed boundary changes. **Aggregation alone does not recover grasp for this
frozen checkpoint.** The bounded remedy is `UNPROMISING`; the underlying cause
of receding grasp failure remains `UNCLEAR`. Do not describe the result as a
general ACT negative, proof of inadequate capacity, or proof of deployment
distribution as the sole cause.

Keep the tested inference modes and audit tools. Preserve open_loop24 as the
behavior baseline; do not promote either ensemble arm or repeat inference-only
launches without a new discriminating hypothesis. The next priority is a
deployment-history diagnostic and explicitly bounded deployment-conditioned
proposal work, using the already observed serial-deployment coverage gap as
motivation. No retraining or candidate/Cm/Gate1 work was started in this Probe.

Example retained offline command (use a new output run ID to reproduce):

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/consequence-evaluator/tools/audit/audit_action_chunk_execution.py \
  --checkpoint outputs/consequence-evaluator/act-native-chunk-engineering-20261009-r1/action_chunk.pt \
  --teacher-packet outputs/consequence-evaluator/act-native-chunk-engineering-20261009-r2/act.pkl \
  --teacher-packet outputs/consequence-evaluator/act-native-chunk-engineering-20261009-r3/act.pkl \
  --gpu 2 --output outputs/consequence-evaluator/act-temporal-offline-20261009-r4
```

## Limitations / future evidence

Formal launch-level repeatability, corrected-normalization checkpoint behavior,
deployment-data fitting, CVAE, candidate ranking and Cm utility are deferred.
This diagnostic changes the near-term choice of inference versus retraining,
and cannot supply strict same-state Gate1 evidence.
