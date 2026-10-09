---
schema: ref2dex.probe.v2
probe_id: P-20261009-act-temporal-ensemble
experiment_id: P-20261009-act-temporal-ensemble
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: f98ce40
claim_id: C3
hypothesis_family: HF-consequence-act-proposal
probe_index_in_family: 4
seed_pool: probe
seeds: [282]
decision_changed_if_positive: retain causal temporal aggregation as an ACT behavior route before any retraining
decision_changed_if_negative: retain the implementation but diagnose deployment coverage rather than claiming hard switching is the sole cause
status: UNCLEAR
run_id: act-temporal-ensemble-20261009-r1
---

# Does inference-only temporal aggregation recover ACT grasp behavior?

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
for offline audit and frozen run manifest; separate child run directories
`act-temporal-{open_loop24,receding8,overlap8,temporal1}-20261009-r1/`
preserve native packets and logs.

## Limitations / future evidence

Formal launch-level repeatability, corrected-normalization checkpoint behavior,
deployment-data fitting, CVAE, candidate ranking and Cm utility are deferred.
This diagnostic changes the near-term choice of inference versus retraining,
and cannot supply strict same-state Gate1 evidence.
