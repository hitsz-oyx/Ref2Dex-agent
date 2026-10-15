---
schema: ref2dex.probe.v2
probe_id: P-20261015-R-current-information
experiment_id: P-20261015-R-current-information
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: 3132bee1
claim_id: C1
hypothesis_family: HF-R-current-information
probe_index_in_family: 1
seed_pool: probe
seeds: [160, 161, 162]
decision_changed_if_positive: carry the useful current-input group into a separately specified WAM-R adaptation Probe
decision_changed_if_negative: inspect current contact/body-state availability and fitting before expanding the production input contract
status: RUNNING
run_id: R-current-information-20261015-r3
---

# Can current reference and hand-object geometry improve q_PD prediction?

Result: RUNNING; first3000-update panel is UNCLEAR (mean unchanged, both-input
tail p95 improves14.5%); a bounded12000-update matched follow-up runs in parallel.
Decision Probe for Mission Objective A / the robot action
decoder. The user requests extra DExplore information at the current frame,
explicitly permits same-tick reference goals/errors, and excludes the16-frame
reference branch. This experiment adds no t+1 or t+16 reference at all.

## Motivation and input audit

DExplore actor observations concatenate delta1 and delta16 reference branches
(`third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py:382-431`).
Even the first half contains future reference, so neither half is copied.
Actual current body rotations/linear/angular velocities and five thresholded
contact bits are absent from R's state87. Exact measurements of those fields
were not retained in RL660; they require new same-execution collection.
Current object pose/velocity and hand positions are already present in R.
Current hand-to-CAD nearest displacement is available from the recorded hand,
object pose and canonical CAD. It matches the nature/sign of `compute_sdf`
(hand minus nearest object point), not force, signed distance or contact.

The cheapest decision-changing test uses existing fixed RL660 data and an
R-only continuation; no physics collection or WAM inference is needed.
The active four-GPU WAM run continues unchanged. If a current input group
helps, prioritize that group for a later joint recipe; if it does not, retain
uncertainty about missing contact/dynamics and optimization rather than
assert observation sufficiency or an irreducible error floor.

## Fixed matched protocol

- Four arms: baseline; current reference only; current geometry only; both.
- Every arm has the identical existing128D/two-layer/8-query/15-output R plus
  one57→128→128 extra encoder. All parameters and initial functions match;
  its final layer starts zero. Missing input groups are masked AFTER TRAIN
  normalization, so baseline has no hidden access to their values.
- Current reference24D: geometric q_ref[t] relative to measured q[t], encoded
  in the same branch-preserving15D geometry/lift representation; current
  reference object pose relative to measured object pose (translation3,
  first two rotation columns6). This is explicitly privileged reference
  information. Neither output labels nor future q_ref/object reference enter.
- Current geometry33D: the11 actual current hand points, in current object
  coordinates, minus their nearest canonical512-point CAD neighbours.
- Hold measured GT future-hand τ24 fixed in all arms. This is an oracle
  decoder-information diagnostic, not the deployed predicted-τ chain.
- All arms start the same R from the active run's step500; a complete R-only
  snapshot is retained as initial_R.pt before the parent can be pruned.
  No preliminary warmup or change to the active main training is performed.
- Same frozen TRAIN/dev sequence split and observed-fixed-grid.v1 membership
  as the active run (110469/11908 windows). Extra normalization uses all and
  only retained TRAIN rows. No per-source stats or dev normalization.
- Common sequence-uniform/half-startup samples, batch128, AdamW3e-4,
  weight decay.01, clip1, all8 standardized L1,3000 updates per arm; model
  extra initialization160, identical training draw seed161, TRAIN report162.
- Fixed TRAIN512 report and the active run's exact256-window dev panel.
  Evaluate every500; select by dev first-control XYZ p95, then mean, including
  step0. Report initial/selected/final, clipping, angle/finger errors,
  startup/later and per-object counts. This dev has been used for exploration,
  not an independent holdout. Save all selected immutable checkpoints/curves.
- An added-input arm is locally PROMISING only if its selected dev mean and
  p95 both improve at least15% over matched baseline and clipping<1%.
  Otherwise remain UNCLEAR for this budget. All comparisons are Probe-level.

## Resources and implementation boundary

Initially one idle GPU0; following the user's parallelization instruction,
remaining arms run concurrently on GPU0/1, within the six-project-GPU cap.
Each worker has <=900s including packing/evaluation; aggregate new artifacts
<=150MB, >=20GiB filesystem headroom. The existing four-GPU run uses
GPU4/5/6/7. Stop on nonfinite values, timeout,
foreign GPU processes, admission mismatch or resource breach. Do not edit
the active run's production src/config; the diagnostic lives under this
Task's tools/run and tests. No new branch or external writes.

Output: `outputs/ptv3-shared-wam-v2/R-current-information-20261015-r2/`.
Execution: r1 completed baseline/current_reference at commit884ec545. Its
partial_result.json and immutable selected weights are retained. The owned
worker was interrupted on user steering while current_geometry was incomplete;
the original FAILED/KeyboardInterrupt receipt is preserved, not negative
method evidence. At commit3132bee1, --arms only partitions work: remaining
current_geometry/both start from the identical retained initial_R.pt and RNG
seeds on GPU0/1. r2/schedule.json links each arm to its actual run/code; no
completed arm is retrained or chosen from multiple restarts. Partial geometry
from r1 is excluded. Panels and step0 predictions must agree before merging.
Tools: `tools/run/current_information.py`, `tools/run/probe_current_information.py`.
Known-answer tests verify parent-function equivalence, reference/geometry
values, absent-input masks and invariance to future states/references/labels.
Independent review must check same-tick semantics and matched controls before
interpreting any improvement.

## Limitations / future evidence

No contact or16-body dynamics have been added. Current reference is privileged
planning information and not an actual-state observation. A frozen same-step
R continuation is not a fresh-training or warmup comparison. GT τ is an oracle;
any improvement must later survive predicted τ and native execution before it
can support grasping, generalization or Cm-benefit claims. The retained corpus
still includes imperfect demonstrations; workspace admission is not success
qualification. No inference about body/force information absent from this
experiment follows from a weak result.

## First3000-update result and fitting-budget decision

All selected checkpoints are step3000. Fixed dev256 first applied PD wrist
XYZ errors after control decoding/clipping:

| Arm | Mean mm | P95 mm | Clipping |
| --- | ---: | ---: | ---: |
| baseline | 59.43 | 219.86 | 0 |
| current_reference | 60.85 | 199.81 | 0 |
| current_geometry | 58.65 | 209.09 | 0 |
| both | 59.67 | 187.94 | 0 |

The predeclared local threshold is not met: no mean improvement, although
both-input tail improves14.5%, wrist-angle/finger p95 improve9.4%/8.8%.
TRAIN sampled mean errors are69–70mm; these TRAIN/dev panels differ in
startup prevalence, so their absolute values are not an overfit estimate.
All four curves are still falling, so this is UNCLEAR rather than evidence
that current information is unnecessary. r2/result.json and its PNG contain
the exact merged results; identical panels/TRAIN stats/initial metrics and
original checkpoint selection were checked. Total artifacts75.9MB.

Decision Note: distinguish a transient short-fit effect from persistent
information benefit. Cheapest useful follow-up is the same four arms from
the same step500 initial R for12000 updates each, not changing architecture,
targets, initial state, seeds, panels, optimizer or selection metric. Run two
workers concurrently (GPU0 baseline/current_geometry; GPU1 current_reference/
both), each<=900s including packing. r3 aggregate artifact budget300MB,
>=20GiB headroom and project six-GPU cap remain enforced. This is one bounded
budget extension, not a seed sweep; stop after12000 even if UNCLEAR. If clear
benefit appears, qualify it on predicted τ before altering the main recipe;
otherwise prioritize unmeasured body/contact state availability over adding
these two groups directly to the production run. No production src edits.

Follow-up output: `outputs/ptv3-shared-wam-v2/R-current-information-20261015-r3/`.
