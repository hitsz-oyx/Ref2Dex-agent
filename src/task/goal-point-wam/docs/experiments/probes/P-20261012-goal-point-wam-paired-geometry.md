---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-paired-geometry
experiment_id: P-20261012-goal-point-wam-paired-geometry
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 8
seed_pool: probe
seeds: [247, 252]
decision_changed_if_positive: test a fresh real-data paired-geometry model against the matched endpoint input control
decision_changed_if_negative: preserve the scale observation but stop analytic tuning and reassess the real-data learning decision
status: UNCLEAR
run_id: paired-geometry-20261012-r1
---

# Paired goal geometry engineering comparison

Class: Decision, engineering-only. Same trajectory representation subgoal and
analytic teacher as controlled-goal Probe. Scaling improved conditional learning
but both full accuracy gates failed. Specific conjecture: explicit per-point
start/delta features avoid forcing the set transformer to infer small paired
geometric changes from two endpoint sets. This is the cheapest discriminator
before real-data scale-up; not a physical counterfactual training proposal.

Result: Pending.
Decision: Compare only endpoint versus paired start/delta representation.

## Protocol fixed before execution

- Same parent, six train scenes, cached eval scene, random offsets seed252,
  initialization seed247, loss,600 updates and held analytic queries as the
  preceding card. Strict load retains identical parameter keys and initial hash.
- Both arms use .05m units. Control input: start and endpoint XYZ. Treatment:
  start XYZ and paired (endpoint-start) XYZ; canonical point correspondences
  are the existing input contract. This is invertible input reparameterization;
  no extra parameters, residual output rule or hidden hand goal.
- All masks applied before subtraction; null token and start/end type embedding
  unchanged. Scene/action/world frozen, world loss OFF, only goal/trajectory train.
- Same full source gate: analytic EPE <=5mm, endpoint response error <=5mm,
  median gain [.75,1.25], error/blind error <=.25 on >=2 sources. Treatment must
  pass and reduce macro analytic EPE >=20% versus control to be PROMISING as
  a learning-contract screen. Otherwise UNCLEAR; keep component metrics.
- Re-run control within the same cached scene rather than selecting favorable
  previous snapshots. World cached isolation <=1e-5; exact parameter hashes.
- <=900s engineering total across these two cards, one GPU2, existing total
  <=20GPU-minutes /30minutes wall and <=5GiB new artifacts. Stop on nonfinite,
  OOM, competition or deadline. Save model-only artificial diagnostic checkpoint;
  never feed analytic-trained weights into any real-data run.

## Decision Note

Root chooses an input-only geometric comparison because .05m units brought
response near1 but absolute errors remained above the fixed gate. Passing
justifies a separately registered fresh PW-initialized real-data comparison;
failure stops further analytic architecture tuning within this task turn.
No core question/claim, branch, permissions or resource boundary changes.

## Limitations / future evidence

Only six familiar scenes and artificial translation labels. No inference about
real goal accuracy, hand feasibility, robots, world-loss utility or RL. Default
endpoint mode remains backward compatible; new mode is exploratory only.

## Artifacts

- [Input manifest](../../../../../../outputs/goal-point-wam/paired-geometry-20261012-r1/input_manifest.json)
- [Matched results](../../../../../../outputs/goal-point-wam/paired-geometry-20261012-r1/result.json)
- [Implementation](../../../tools/audit/probe_controlled_goal.py)
