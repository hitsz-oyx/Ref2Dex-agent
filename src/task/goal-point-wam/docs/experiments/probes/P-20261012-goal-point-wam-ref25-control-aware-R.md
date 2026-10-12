---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref25-control-aware-R
experiment_id: P-20261012-goal-point-wam-ref25-control-aware-R
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 34
seed_pool: probe
seeds: [295, 296, 292]
decision_changed_if_positive: qualify control-aware Chunk-R before conditional hand anchoring and frozen predicted-tau world consistency
decision_changed_if_negative: retain the implementation but stop this fixed recipe and prioritize GT first-control execution
status: UNCLEAR
run_id: wam-v2-control-fit-20261012-r1
---

# ref2_5 A2: first-action and PD-aware Chunk-R

## Decision Note / motivation

User ref2_5 requests keeping the Chunk Transformer and correcting its objective
for first-only receding execution. Probe33 correctly applied GT tau but newR
held0 vs calibrated oldR75/teacher73; report first-PD XYZ p95 was31.09mm and
actual recovery TRAIN94.80mm. Test whether first-control emphasis and physical
scales improve that execution prerequisite. This is a **Decision Probe** for
the self-trained manipulation goal, not a WAM or Cm utility demonstration.

Cheapest discriminating test: one matched fresh3000-update fit, same model
initialization295/draw stream296, teacher/recovery sources, batch256, Adam3e-4,
gradient clip1, normalizers and partitions as Probe33. Compare its existing
stored report; no duplicate baseline training or sweep. New raw action loss:

`L = 4 * mean(abs((a0-y0)/sigma0))`
`  + mean(abs((a1:8-y1:8)/sigma1:8)) + L_PD0`.

`L_PD0` is the equally weighted mean of independent wrist XYZ error/.01m,
wrist angle error/.1rad and finger target error/.1rad. These are fixed units
chosen before the run to make control scales explicit, not tuned acceptance
thresholds. All coefficients are4/1/1. Native wrist PD differences equal
action differences times controller scale; finger differences use half scale.
Both targets use the same current measured q and native offset, which cancel.
Coupled passive joints are deterministic and not independently supervised.
Use raw outputs in training so clipping cannot suppress corrective gradients;
report and execute native clamped outputs, recording clipping separately.

Checkpoint selection every250 updates (also first/final), lexicographic
development **first applied PD XYZ p95**, then all8 standardized L1. Save one
selected final packet. Report episodes do not influence fitting or selection;
same motion/frame0 means these are development Probes, not generalization.
Control-aware loss and selection change together; this is not an ablation
isolating either component. Legacy `--control-aware`-absent behavior remains.

After finite model/labels/metrics checks, run one128-control actual native
comparison, same seed292/GPU2: teacher, qualified original oldPD-R, newChunk-R.
OldPD-R uses the corrected calibrated batch2 call from Probe33. Both R roles
query their own current state and same source-env24 GT future. Every tick
predict8/executefirst. Calibration: teacher/oldR>=45held, clip<1%; newR gate
>=max(45,.8*oldRheld), clip<1% (60frames if oldR75). This gate dominates
offline improvements. Failure stops B/C for this run; success allows the
already authorized ref2_4 staged work, beginning with offline B comparison.
No WAM/goal/layer-size/world/RL changes in A2. No new branch or external write.

Resources: GPU2 idle preflight, fit600s and physics600s deadlines; total20GPUmin.
All cumulative wam-v2 outputs/scratch remain below stage80MB (previous17.70MB)
and global outputs+tmp400GB. Preserve checkpoints and failed runs. Stop on
nonfinite, incorrect labels/coordinates/dispatch, foreign GPU process, reset,
deadline or output limit. Only repeat affected engineering checks if needed.

## Implementation / checks

- `src/task/trajectory-policy/src/trajectory_policy/chunk_objective.py`
- `src/task/trajectory-policy/tools/run/fit_chunk_retargeter.py --control-aware`
- `src/task/trajectory-policy/tools/run/execute_chunk_retargeter.py`
- `src/task/trajectory-policy/tools/audit/audit_chunk_retargeter.py`

Check physical PD delta against native conversion including half-scaled fingers;
first-control errors matter more and unclamped predictions retain gradients;
checkpoint selection favors first control when chunk mean disagrees. Reconstruct
labels/physical metrics/objective components/selection independently in the
root audit before interpreting physics.

## Results

Pending engineering smoke, matched fixed fit and independent GT comparison.

## Limitations / future evidence

One fixed-seed short fit cannot establish an architecture failure or cause of
failure. Directly emphasizing first control may trade later chunk precision
for immediate control; report both. Positive privileged-GT execution only
qualifies R locally, not autonomous WAM, full placement or Cm-on/off utility.
Motion/disturbance holdouts and component ablations are deferred until they
can change a next decision. B/C remain gated by new actual GT execution.
