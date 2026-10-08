---
schema: ref2dex.probe.v2
probe_id: P-20261009-act-deployment-conditioned
experiment_id: P-20261009-act-deployment-conditioned
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-consequence-act-proposal
probe_index_in_family: 2
seed_pool: probe
seeds: [20261009, 20261010]
decision_changed_if_positive: collect more native reactive-deployment episodes before any ACT behavior screen
decision_changed_if_negative: close the direct ACT proposal route until a new execution contract exists
status: PROMISING
run_id: act-native-chunk-deployment-20261009-r2
---

# Does deployment-conditioned history improve the native 24-step proposal?

## Decision Note

The clean action-chunk proposal had a deployment-distribution gap when its
checkpoint was evaluated on the reactive teacher histories from the two fresh
native GPU serial-cluster launches. The pending decision was whether to spend
more effort collecting deployment episodes and retraining an ACT-like proposal,
or close that engineering route. The cheapest discriminating Probe is a
leave-one-launch-out fit using only those two already-recorded teacher arms.

This is an action-space and deployment-coverage Probe. It does not restore
PhysX hidden state, run a simulator, alter Y or the physical reference bank,
train the evaluator, or authorize any Gate1--Gate5 claim.

## Contract

- Inputs are the `reactive_teacher` history/action arrays from
  `gate1-gpu-serial-cluster-20261009-r33` and `r34`. The other serial arms are
  excluded.
- Each 72-step launch supplies 13 non-crossing windows at stride 4, with
  `history[t]` predicting the native executed `action[t:t+24]`.
- Each fold trains the same 128-wide, two-layer decoder from scratch on one
  launch and evaluates on the other. The history standardizer is fitted only
  on the training launch. The fixed clean and route-s3 checkpoints are
  evaluated on the same held-out launch as references.
- The output is explicitly `engineering_only` and `training_allowed=false`.
  No simulator process is started.

## Results

The bounded GPU2 fit completed in about 21 seconds per two-fold run without
foreign GPU interference. The corrected-provenance run is
`outputs/consequence-evaluator/act-native-chunk-deployment-20261009-r2/`.

| train launch | held-out launch | fitted MSE | fitted first-action MAE | clean checkpoint MSE | route-s3 checkpoint MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| r33 | r34 | `1.776e-4` | `0.00582` | `2.017e-3` | `3.963e-4` |
| r34 | r33 | `1.726e-4` | `0.00609` | `1.932e-3` | `3.532e-4` |

The fitted proposal beats both fixed references and the train-launch mean
baseline on both held-out launches. The result is `PROMISING` for
deployment-conditioned action coverage only. It is not evidence of behavior
preservation, candidate utility, same-state replay, or Gate1 readiness; the
two launch-level episodes and overlapping windows are too small for a formal
generalization claim.

## Decision

Retain this proposal route as a bounded engineering direction and, if more
budget becomes available, collect additional native reactive-deployment
episodes before fitting a behavior proposal. Do not start a new ACT candidate
or serial-noise simulation from this result alone: the native GPU hidden
contact/cache state remains unavailable, and the strict same-state Gate1
contract is still closed. The next valid progress requires a verifiable
execution contract or an explicitly approved change of claim.

## Reproduction

```bash
python3 src/task/consequence-evaluator/tools/audit/fit_deployment_action_chunk.py \
  --packet outputs/consequence-evaluator/gate1-gpu-serial-cluster-20261009-r33/serial-cluster.pkl \
  --packet outputs/consequence-evaluator/gate1-gpu-serial-cluster-20261009-r34/serial-cluster.pkl \
  --output outputs/consequence-evaluator/act-native-chunk-deployment-20261009-r2 \
  --gpu 2 --steps 1200 \
  --reference outputs/consequence-evaluator/act-native-chunk-clean-airplane-base-20261009-r1/action_chunk.pt \
  --reference outputs/consequence-evaluator/act-native-chunk-route-s3-engineering-20261009-r1/action_chunk.pt
```
