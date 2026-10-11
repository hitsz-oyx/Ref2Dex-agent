---
schema: ref2dex.probe.v2
probe_id: P-20261011-chord-action-ranking
experiment_id: P-20261011-chord-action-ranking
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: d9d147b
claim_id: C1
hypothesis_family: simulator-chord-wrench-action-ranking
probe_index_in_family: 1
seed_pool: probe
seeds: [17]
decision_changed_if_positive: retain simulator contact-wrench/CHORD support as a candidate action-ranking context for the next Cm-on/off integration Probe
decision_changed_if_negative: defer this simplified CHORD action-ranking context and keep calibrated sensor/geometry routes separate
status: RUNNING
run_id: chord-action-ranking-20261011-r1
---

# Simulator CHORD action-ranking Probe

Class: Decision. The preceding DExplore smoke established that the external
simulator exposes finite contact forces and that the local CHORD geometry
adapter produces finite wrench rays and support coordinates. This Probe asks
whether that context changes candidate action ranking in a controlled simulator
loop, which is the missing bridge between a wiring smoke and a downstream
policy decision.

## Decision note

**Question:** does current contact-force plus CHORD support context improve
ranking of small candidate actions over an effect-only state/action model?

**Evidence:** DExplore has a reproducible Inspire trajectory and contact tensor,
while no public H5/EgoTouch label provides an object-COM CHORD wrench. The
current simulator route is therefore the cheapest place to test the ranking
contract without importing official video weights or inventing force labels.

**Cheapest test:** use one read-only DExplore checkout and one materialized
`s1_airplane_lift` motion, 12 parallel candidate environments, four fit and
four held reference frames, horizon eight, and fixed small action deltas. At
each frame compare a ridge effect model (`E only`) against the same model with
body/object force, CHORD support, and action-support interactions (`E+C`).

**Stop/gate:** stop on simulator import failure, nonfinite tensors, missing
motion hash, GPU contention, or output growth beyond the bounded run directory.
A positive result requires E+C to reduce held mean regret by at least 20% and
not reduce top-1 hit rate. A negative result closes only this simplified
simulator ranking recipe; it cannot reject calibrated T-Rex/RBO wrench
supervision, native CHORD reward training, or a future per-contact normal
contract.

## Fixed protocol

- External checkout: `/home2/wyy/oyx_ws/Ref2Dex/third_party/DExplore`, read-only;
  the exact commit is recorded in the run result.
- Motion: `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/s1_airplane_lift`, read-only; the `interaction_hand_inspire.pt` hash is recorded.
- Runtime: existing Isaac Gym DExplore v117 environment, one visible GPU,
  `num_envs=12`, no actor/Cm checkpoint, no official model weights.
- Fits: frames 48, 56, 64, 72; held frames 80, 88, 96, 104; horizon eight.
- Candidates: reference action plus twelve fixed perturbations around base,
  including bounded wrist and selected finger deltas.
- Target: negative future object-position error relative to the stored reference
  at frame plus horizon. Candidate groups are evaluated independently.
- E-only features: current DOF position/velocity, object state, candidate action.
  E+C adds current selected-body net force, object contact force, 12-axis CHORD
  support, and candidate-action/support interactions.
- Normal source: current simulator net-force direction, used only as an
  approximate contact normal; this is not a per-contact normal or measured
  object-COM wrench label.

## Results

The run result and decision will be filled after the fixed simulator execution.

## Limitations / future evidence

This is a single motion, single seed, one-step candidate intervention with a
short follow-up horizon. The net-force normal approximation and simplified
support basis do not establish physical CHORD supervision. Even a positive
ranking signal would only justify a matched downstream Cm-on/Cm-off Probe; it
would not establish policy success or generalization to official video data.
