---
schema: ref2dex.probe.v2
probe_id: P-20261017-wam-v4-interaction-head
experiment_id: P-20261017-wam-v4-interaction-head
date: 2026-10-17
task: ptv3-shared-wam-v4
branch: main
git_commit: UNCOMMITTED
claim_id: C1
hypothesis_family: HF-wam-interaction-head
probe_index_in_family: 1
seed_pool: probe
seeds: [170]
decision_changed_if_positive: keep the dynamic graph interface and run a matched tau-only versus tau-plus-C executor probe
decision_changed_if_negative: remove or redesign C before any long pretraining budget
status: ENGINEERING
run_id: not-run
---

# WAM-v4 dynamic interaction graph

This is an architecture probe, not a robot-benefit result. It keeps the v3
shared encoder and hand/scene heads and adds a dynamic object-centric graph:

```text
G_task -> WAM -> (tau_hand[24], tau_scene[24], C_HO[24]) -> executor
```

`C_HO` has shape `B,O,T,2,11,P,9` (`T=24`, `P=32` in the current data
contract). Each edge contains contact confidence, object-relative surface
position, radial normal, normal approach velocity and tangential slip speed.
The hand-point and object-point axes stay explicit; the graph is not reduced to
a global mean-pooled WAM feature. Labels use the existing 32 query points and
radial-normal approximation, so this probe does not claim dense-surface
supervision.

The trainer supports a model-only v3 warm start. Compatible trunk parameters
transfer and the graph relation modules initialize fresh. The cheapest
qualification is a forward/backward smoke and a masked-label audit. Only after
those pass should a matched `state + tau` versus `state + tau + C_HO` executor
probe decide whether sustained pretraining is justified.
