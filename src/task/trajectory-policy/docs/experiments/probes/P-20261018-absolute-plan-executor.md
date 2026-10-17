---
schema: ref2dex.probe.v2
experiment_id: P-20261018-absolute-plan-executor
task: trajectory-policy
branch: main
git_commit: pending
date: 2026-10-18
classification: Decision
status: UNCLEAR
hypothesis: Adding the reference hand's current object-frame anchor will make direct tau execution identifiable after learner drift; a control-space PD objective will expose physical errors that raw absolute-PD normalization hides.
question: Does an absolute reference-hand anchor plus ordered future tau improve offline teacher-PD prediction on the existing DAgger queries sufficiently to justify a new recovery-basin roll-in?
decision_changed_if_positive: run one bounded-intervention DAgger collection with beta 0.25 using the repaired absolute-plan contract.
decision_changed_if_negative: do not spend another autonomous DAgger round; treat the anchor-plus-loss probe as insufficient and inspect interaction context or richer plan geometry.
protocol: Reuse the existing direct DAgger query packet. Add A_ref_t=R_ref_t(h_ref_t-o_ref_t), a 33-D current reference hand anchor in the reference object frame, to the live state87, ordered tau8x33, and previous active action12 inputs. Warm-start from the selected r3 checkpoint. Train with equal original/query batches using wrist control targets qdes-q and finger control targets qdes, normalized by train control-space dispersion, plus 0.1 action Huber. Query environments 0-4 are fit and 5-7 are held out. Report xyz mm, rotation rad, finger rad, held-out query metrics, and query_clipped rates.
query_source: outputs/trajectory-policy/direct-tau-dagger-rollin-20261018-r2/dagger_queries.npz
rollin_source: outputs/trajectory-policy/direct-tau-dagger-rollin-20261018-r2/trajectory.npz
base_checkpoint: outputs/trajectory-policy/direct-tau-bc-r3-20261018-r2/best.pt
runtime: /home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train, Torch 2.0.1+cu118, GPU 0
---

## Motivation and decision note

The prior direct executor supplied only the reference future displacement
`tau_ref`. On the teacher manifold this implicitly identifies the current
reference location, but after learner drift it leaves the recovery target
underdetermined. This probe adds only the missing current reference anchor; it
does not add oracle interaction context or retrain the WAM planner.

The same query packet is reused so the probe separates input-contract and loss
changes from a new physics distribution. The offline gate is deliberately
physical: normalized loss is reported for optimization, while xyz, rotation,
and finger errors are reported in their native control units. `query_clipped` is
audited rather than silently filtered.

## Results

The first offline fit reused the old autonomous query packet. The absolute
anchor and control-space loss were wired correctly, and `query_clipped` was 0,
but 99.30% of rows were catastrophic. Held-out query error after fitting was
152 mm wrist xyz RMSE, 0.482 rad rotation RMSE, and normalized Huber 0.329.
This is insufficient to select a physical checkpoint, but it is confounded by
the collection distribution.

A bounded-intervention roll-in with beta 0.25 was then collected. Its actual
intervention fraction was 0.25, but the state audit remained 0.46% near,
1.86% moderate, and 97.68% catastrophic; the direct arm achieved 0/8 hold45.
This confirmed that beta 0.25 is too weak when the starting executor is already
outside the recovery basin.

For a recovery-basin collection, teacher actions were forced for the first 120
controls (through the stable grasp phase), then beta 0.25 was used. The direct
arm held in all 8 environments during this collection, but placement was not
evaluated as autonomous behavior. The resulting query distribution was 22.19%
near, 10.61% moderate, and 67.20% catastrophic; actual intervention fraction
was 0.461 because of the warm-up. Query clipping remained 0.

Fitting the same absolute-plan policy and control-space PD objective on this
recovery-basin packet reduced held-out control RMSE from 0.345 to 0.273 and
normalized Huber from 0.784 to 0.493. Native errors were still 211 mm xyz,
0.433 rad rotation, and 0.182 rad finger RMSE, so the offline gate did not
pass. The source-sequence diagnostic was much better (41.7 mm xyz, 0.078 rad
rotation, 0.050 rad finger), showing that the remaining problem is the
learner-state distribution rather than checkpoint loading or a nonfinite
policy.

Artifacts:

- absolute contract fit on the old autonomous packet:
  `outputs/trajectory-policy/absolute-plan-pd-fit-20261018-r1/`
- beta 0.25 roll-in (insufficient recovery basin):
  `outputs/trajectory-policy/absolute-plan-dagger-rollin-20261018-r1/`
- beta 0.25 with 120-control teacher warm-up:
  `outputs/trajectory-policy/absolute-plan-dagger-rollin-20261018-r2/`
- recovery-basin absolute-plan fit:
  `outputs/trajectory-policy/absolute-plan-pd-fit-20261018-r3/`

The probe is `UNCLEAR`: the anchor contract and bounded collection repair the
data coverage, but the held-out live-state PD error remains too large to justify
an autonomous physical replay. Do not interpret the 8/8 hold during the warm-up
roll-in as executor success; teacher actions were part of that trajectory.

## Next decision

Stop adding DAgger rounds to this contract for now. The next informative branch
is an executor with live replanning or explicit interaction context `C`, tested
on a small matched probe; a longer PPO run is not justified by the current
offline gate.
