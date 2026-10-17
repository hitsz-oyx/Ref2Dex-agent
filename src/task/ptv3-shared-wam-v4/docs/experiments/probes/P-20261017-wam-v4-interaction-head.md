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
decision_changed_if_positive: keep the C interface and run a matched tau-only versus tau-plus-C executor probe
decision_changed_if_negative: remove or redesign C before any long pretraining budget
status: ENGINEERING
run_id: not-run
---

# WAM-v4 interaction head architecture

This is a Decision probe design, not a robot-benefit result. It implements the
new interface from the current architecture discussion:

```text
G_task -> local 24-step goal -> WAM -> (tau, C) -> executor
```

The v3 shared encoder and hand/scene heads remain fixed. The new C head reads
current hand/object relation features and predicts `B,O,3,11,9`: contact
confidence, approximate object-surface position, approximate normal, normal
approach velocity and tangential slip speed at WAM steps 8/16/24. Labels are
derived from the existing current-anchor hand/object future labels. This probe
uses the existing 32 object query points and radial-normal approximation;
dense CAD surface/normals are a later data upgrade.

The v4 trainer supports a model-only v3 warm-start. All compatible v3
parameters transfer, while the structured C relation modules are initialized
fresh; optimizer state is intentionally not reused.

The cheapest qualification is a CPU/GPU forward and backward smoke, followed
by a matched executor comparison using the same WAM checkpoint and seed:
`state + tau` versus `state + tau + C`. This result would decide whether C is
worth a sustained pretraining or executor budget. It does not yet justify
claiming improved grasping or Cm.
