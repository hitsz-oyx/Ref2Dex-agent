---
schema: ref2dex.probe.v2
probe_id: P-20261010-hand-action-contact-context
experiment_id: P-20261010-hand-action-contact-context
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 19a646d2046e94f80230cfb4620ad4abce737358
claim_id: C3
hypothesis_family: HF-hand-action-retarget
probe_index_in_family: 2
seed_pool: probe
seeds: [406]
decision_changed_if_positive: retain a query-time contact-conditioned R and consider a separately authorized native upper-bound replay
decision_changed_if_negative: stop contact-proxy R sweeps and keep H-to-hand/native execution/PW/evaluator routes frozen
status: UNPROMISING
run_id: ref7_2-contact-context-fit-20261010-r1
---

# Can current-state contact proxies improve full-action retargeting?

Result: `UNPROMISING`. A matched v2 contextual baseline and v3 contact arm
used the same structured rollout windows and shared initialization. Adding ten
query-time contact proxies improved held-out finger MAE by only `1.19%`, while
contact-onset finger MAE slightly worsened. The arm used the extra fields, but
the gain was far below the run-time threshold and does not justify native
execution.

## Decision note and contract

The preceding ref7_2 Probe fit commands offline but failed native GT-hand
execution, leaving contact/preload observability as the next cheapest question.
This Spike tests only whether current-state proxies add predictive information:

```
R(future_hand[t+1:t+24], q_t, dq_t, hand_object_context_t,
  pair_t, surface_gap_t, support_gap_t, object_velocity_t, table_footprint_t)
  -> action[t:t+24]
```

The ten fields are ordered as `pair`, two gaps, six target velocity channels,
and `table_footprint`. They are measured at state `t` after command `t-1` and
before command `t`; no future contact/state field is exposed. `pair` is the
configured five-body/object net-force threshold proxy, not an exact collision
pair. Object velocity is world-frame linear/angular velocity. Reset frame 0 is
excluded because its contact cache is invalid. Raw values are retained and
standardized with train-only mean/std; no clipping or robust transform is
silently applied.

The v2 baseline remains unchanged. The v3 model adds a zero-initialized
10-dimensional branch to the shared v2 encoder, so its initial shared weights
and zero-contact output are bitwise matched. Teacher-anchor packets are not
mixed into this first contact arm because their pair semantics are not the same
as the structured five-body proxy.

## Matched protocol

The run used the three existing structured splits (96/64/64 episodes), stride
2, reset-excluded windows, batch 256, seed 406, 3000 updates, and GPU2. The
window counts were 24,864/16,576/16,576 (train/val/test). Both arms used the
same batches, optimizer, validation selection and episode split. Output:
`outputs/consequence-evaluator/ref7_2-contact-context-fit-20261010-r1/`.

The fit completed in 84.7 seconds with finite losses. The original run
manifest records the exact input hashes, trainer hash
`a8decac04109674c2aa39b515815032240fe8caed663cfcf0f1a08d7afbff6da`, and
`git_commit=19a646d` (the trainer was not committed at run time). The post-run
commit `221af42` only adds fail-closed dtype/reset checks, synthetic alignment
tests, a loader hash in future manifests, and stricter reporting metadata. The
structured source fields already satisfy those checks, so the numerical result
is retained without a second fit.

## Results

| Test stratum | Matched v2 finger MAE | v3 contact finger MAE | Relative change |
| --- | ---: | ---: | ---: |
| all | .0190543 | .0188270 | −1.19% |
| pair=true | .0181175 | .0178832 | −1.29% |
| pair=false | .0200920 | .0198724 | −1.09% |
| near gap ≤1 cm | .0181307 | .0178874 | −1.34% |
| contact-onset | .0170863 | .0171286 | +0.25% |
| hold | .0210830 | .0206553 | −2.03% |

All-action L1 was `.0158264` vs `.0156897`; horizon-23 L1 was `.0193641`
vs `.0192388`. Replacing standardized contact inputs by their train mean gave
finger MAE `.0193423`, and shuffling them gave `.0202583`; the branch therefore
uses information, but not enough to meet the task-relevant improvement gate.

The run-time Probe gate required at least 10% overall held-out finger
improvement, no contact-onset or hold regression, and no more than 5%
horizon-23 regression. The later committed contract also checks pair-positive
and near-gap strata; recomputing those two checks from the stored metrics gives
no regression, but does not change the outcome. The run failed the 10%
criterion and contact-onset criterion (`UNPROMISING`).

## Attribution and stop rule

This is one offline seed and one structured rollout campaign. The native pair
field is a force proxy; the model has no exact collision pair, impulse, normal
force or preload measurement. The mean/shuffle ablations show that the branch
is not dead, so the negative result is not an unused-input wiring failure, but
it does not identify why preload is unavailable. Do not add more epochs, change
the threshold, or run native R execution from this arm. The current route stays
offline-only and does not unlock H-to-hand, PointWorld, evaluator, selector,
MPC, or a Cm claim.

## Artifacts and verification

- `outputs/consequence-evaluator/ref7_2-contact-context-fit-20261010-r1/`
- `src/task/consequence-evaluator/tools/run/train_matched_contact_context.py`
- post-fix contract tests: `216` Task tests passed
- `tools/verify.py --changed` and experiment-index checks passed
