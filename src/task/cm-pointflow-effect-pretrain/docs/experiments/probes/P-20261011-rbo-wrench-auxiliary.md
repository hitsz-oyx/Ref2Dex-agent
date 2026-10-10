---
schema: ref2dex.probe.v2
probe_id: P-20261011-rbo-wrench-auxiliary
experiment_id: P-20261011-rbo-wrench-auxiliary
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: TBD
claim_id: C1
hypothesis_family: rbo-measured-wrench-temporal-auxiliary
probe_index_in_family: 1
seed_pool: probe
seeds: [231]
decision_changed_if_positive: retain the calibrated RBO RGB-D plus measured-wrench route for a matched point-flow/wrench auxiliary arm
decision_changed_if_negative: stop this one-sequence wrench-forecast recipe while retaining the adapter contract and keeping T-Rex and simulator CHORD routes separate
status: RUNNING
run_id: rbo-wrench-auxiliary-20261011-r1
---

# RBO measured-wrench auxiliary Probe

Class: Decision. The preceding RBO adapter Probe established a finite,
timestamp-aligned RGB-D, object-pose, joint, and measured-FT contract. This
Probe asks the cheaper question of whether the two safe wrench layers carry
short-horizon temporal signal beyond persistence before wiring them into the
native point-flow model.

## Decision note

**Question:** can a short history of RBO geometry/state and past measured wrench
predict the next eight wrench samples on a chronological held tail?

**Evidence:** the adapter retained 295 RGB anchors, finite 300-point world
geometry, and finite raw/geometric six-axis wrenches. The geometric layer keeps
`payload_compensation=false`; it is not an object-COM net wrench or CHORD label.

**Cheapest test:** use one already acquired interaction, fixed `H=4,K=8`, the
first 220 windows for fit and the final chronological windows for held-out
evaluation, a small random-initialized MLP, and compare both wrench layers with
last-observation persistence. No new data, checkpoint, or external process is
needed.

## Fixed protocol

- Input: `rbo_aligned.npz` from the pinned `cardboardbox26_o` archive;
  source and adapter hashes are recorded in the result.
- Features: per-frame world-point centroid/spread, object-pose translation and
  6D rotation, joint position, and the past four samples of the target wrench.
- Targets: `wrench_sensor_myframe` and `wrench_object_geometric`, trained as
  separate heads with fit-only normalization.
- Split: chronological first 220 valid windows for fit; all final windows held
  out. No random frame split and no cross-window leakage in normalization.
- Baseline: repeat the last observed wrench through `K=8`.
- Device: one GPU when available, 1,000 updates per target, seed 231 (target
  heads use deterministic offsets 231 and 232), output cap 3 GiB.
- Boundary: force/torque units are preserved as supplied by the adapter;
  `payload_compensation=false`; no CHORD support or object-COM claim.

## Decision rule

At H8, a target is retained if model vector-L2 is at least 10% below its
persistence baseline. The overall Probe is `PROMISING` if both target layers
pass, `PROMISING_TARGET_SPECIFIC` if one passes, `UNPROMISING` if both are
worse than persistence, and `UNCLEAR` otherwise. Any result is only a
one-sequence learnability signal and cannot establish downstream Cm benefit.

## Results

Filled after the bounded run.

## Limitations / future evidence

This is one interaction with a chronological tail, not an object- or task-held
generalization test. The measured FT stream still lacks verified payload/COM
compensation, so the geometric target remains a frame-transformed sensor
wrench. A positive result justifies a matched RBO auxiliary arm; a negative
result cannot reject T-Rex fingertip F6, CHORD simulator support, or the adapter
itself.
