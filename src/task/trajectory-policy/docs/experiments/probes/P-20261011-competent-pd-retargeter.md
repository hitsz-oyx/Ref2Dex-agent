---
schema: ref2dex.probe.v2
probe_id: P-20261011-competent-pd-retargeter
experiment_id: P-20261011-competent-pd-retargeter
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 8af2e7a
claim_id: C3
hypothesis_family: HF-trajectory-policy-competent-pd
probe_index_in_family: 1
seed_pool: probe
seeds: [180, 181, 182]
decision_changed_if_positive: retain the real-PD retargeter and design joint measured-history/goal trajectory BC
decision_changed_if_negative: diagnose the specific target/conditioning/execution gap before high-level BC or PointWorld
status: PLANNED
run_id: competent-pd-teacher-20261011-r1
---

# Competent official rollout to measured-state/trajectory-conditioned PD targets

## Intent and Decision Note

The user confirmed ref3: official DExplore actor as demonstration generator,
actual targets entering the native PD controller as labels, object target pose
as G, and a learned R(current measured state, predicted hand trajectory).
Official weights never initialize the independent student. Mission A/B remain:
an independently trained policy for the target reference trajectory, followed by
matched Cm-on/off training (using the user's updated Mission Objective A).

Classification: Decision. First distinguish whether competent actual hand motion
and current measured state support sufficiently accurate, physically useful PD
control. Cheapest route: one fresh teacher wave, one matched offline R fit, then
one frozen source-PD/GT-hand-R/live-teacher wave if the precision screen passes.
No high-level policy, PPO, PointWorld or parameter/seed sweep in this card.

The official full reference includes deliberate table placement. This mechanism
Probe retains that complete task, separately reports sustained geometric holding
and controlled settling, and does not equate an intentionally placed endpoint
with a dropped object. It does not establish the independent high-level baseline.

## Data and model contract

Teacher: read-only `/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth`,
SHA256 `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`.
Use the archived Torch2.0.1/rl_games1.1.4 runtime and its verified single-RMS
compatibility. Exactly 32 frame0 episodes, seed180, 542 controls, GPU PhysX/GPU
pipeline. Actor calls are padded to 64 rows to avoid the documented legacy
single-row inference problem. No old engineering-only packet is relabeled.
Fresh raw manifest declares `training_allowed=true`, official origin and scope.

Record q/dq, 11 actual hand link points, actual object pose/velocity, requested
and applied native action, all 18 actual PD targets, sampled surface gap/table
support/footprint, and a known terminal object-pose goal from the source reference.
State[t] and PD[t] precede state[t+1]. Keep every episode; no outcome selection.
Teacher readiness: at least24/32 hold45 and controlled-placement success episodes.
Split whole envs 0:16 train, 16:24 selection, 24:32 test before creating windows.
No formal independent-seed or motion-generalization claim.

R uses measured state87 plus actual hand[t+1:t+24] expressed as displacement
from the live hand in the query object frame. No future robot q, future object,
teacher obs/action, force, phase or clock enters R. It predicts the first
control's 12 independent PD coordinates, with wrist targets represented relative
to measured q_t; finger targets remain absolute joint positions. This is an
invertible target representation, not replacing labels by actual future q or IK.
Native conversion uses live q every tick and restores the six coupled joints.
Native clamp is logged along with requested target and achieved actual PD target.

Matched 256-wide two-hidden-layer MLPs, identical initialization/sampled batches,
tau versus zero-normalized-tau state-only control. Train-only normalization,
3,000 updates, batch512, Adam3e-4, seed181. Standardized PD-coordinate MSE;
selection only by fixed validation checkpoints. Preserve first-control XYZ,
rotation and finger p95, startup/contact/hold breakdown, same-tick cross-env tau
shuffle and tau gradients. A null shuffle is not evidence of tau irrelevance
when donor futures have insufficient independent variation.

## Gates and physical interpretation

Offline precision screen on selection rows (test reserved for reporting): XYZ
p95 <=5mm, wrist rotation <=0.05rad, fingers <=0.05rad, native clipping <1%.
Require tau model lower validation loss than state-only and finite nonzero tau
gradient before physical evaluation. Report test once; no test model selection.
Failed offline precision is UNCLEAR for physics; diagnose without automatically
adding epochs. Only a passing screen permits one frozen 24-env full wave, seed182:
8 live official teachers, 8 source-PD replay controls, 8 R with frozen actual
future hand from source test envs24:32. Recompute R state every native tick.
GT future hand is an explicit oracle, not a deployed high-level policy.
These separately evolving rows are not exact hidden-state forks.

Physical PROMISING requires >=6/8 hold45 and controlled-placement successes in
each calibration arm and R, R median holding >=80% of source-PD control, and R
clipping <1%. Calibration failure is UNCLEAR, not R negative evidence. Passing
calibration with failed R is UNPROMISING for this fixed model/data/contract.
Audit independently reconstructs PD conversion/coupling, feature and label time
alignment, source qualification/split, model outputs, clipping and outcomes.
No scientific interpretation until that audit passes.

## Resources and stopping

One idle GPU0 (recheck before each stage), total <=20 GPU-min /1GiB new artifacts:
capture<=300s, fit<=300s, eligible physical<=300s, audits<=180s, one engineering
smoke<=120s. Existing global <=4 simultaneous GPUs/300GiB remains binding.
Neural training/inference and geometry/PhysX on GPU; CPU file/statistical work.
Record GPU utilization/memory and ETA at bounded intervals. Stop on input drift,
nonfinite data, unexpected reset, resource conflict, timeout or missing PD capture.
Retain failures; repair actual implementation blockers within this total cap.
Stay main, no branch/push or external environment modification.

## Results

Pending execution. Artifacts go under `outputs/trajectory-policy/competent-pd-*`.

## Limitations / future evidence

One reference with fixed G does not establish goal conditioning; collection
origin is official and must be reported. Same-wave env holdout is a Probe only.
Desired versus actual tau shift, independent teacher rollouts, full-reference
tracking with controlled placement, joint H/G->tau->R supervision and matched Cm utility
remain future evidence. Proceed to each only if it changes the next decision.
