---
schema: ref2dex.probe.v2
probe_id: P-20261015-R-native-current
experiment_id: P-20261015-R-native-current
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: 6e08ed66
claim_id: C1
hypothesis_family: HF-R-native-current
probe_index_in_family: 1
seed_pool: probe
seeds: [160, 161, 162]
decision_changed_if_positive: qualify native current inputs under predicted tau before changing the joint WAM recipe
decision_changed_if_negative: retain uncertainty about data quality and fitting rather than expanding production inputs immediately
status: RUNNING
run_id: R-native-current-20261015-r1
---

# Native DExplore current information for q_PD

Decision Probe for Mission Objective A: determine whether the real captured
DExplore delta0 observation improves the existing trajectory-conditioned R's
PD prediction. User requested parallel experiments after the full rollout.
Cheapest discriminating test is matched R-only fitting, avoiding WAM inference
and further simulation. The Viser data review proceeds independently.

## Matched protocol

Two simultaneous arms: baseline and native_current. Same retained step500 R
from `R-current-information-20261015-r1/initial_R.pt`; using a fixed early R
keeps this an extension of the previous information diagnostic, independent
of the ongoing R5 optimization state. Both receive identical state87 and
measured GT future-hand tau24x33. This is explicitly an oracle-tau offline
information diagnostic, not predicted-tau joint fine-tuning or grasp success.

Both arms have identical extra721→128→128 adapter capacity, with zero final
layer initialization preserving the parent's initial function. baseline masks
all721 values AFTER normalization; native_current receives all721 captured
values at t. Current reference targets/errors are explicitly permitted.
Only delta0 is loaded: no delta1 or delta16 teacher observation branch is
added. Actual five contact bits, complete native body/object/reference/IG
blocks are included, unlike the prior reconstructible24D/33D comparison.
Capture and independent raw audit are recorded in
[P-20261015-dexplore-current-rollout](P-20261015-dexplore-current-rollout.md).

Data: audited `dexplore-current-rollout-20261015-r1`, 660 complete sequences,
original inherited TRAIN/dev membership, newly captured same-execution q_PD.
Use existing tracking and observed-fixed-grid.v1 admission identically in
both arms. Native extra mean/std are fitted only on retained TRAIN windows,
scale floor1e-3. Existing R state/tau/action normalizer buffers remain exactly
the shared parent buffers; this isolates the added observation information.
Read/decompress/pack numeric arrays on CPU (file processing); all neural
training and inference on GPU. All8 geometry-lifted q_PD targets and first
control native PD decoding remain unchanged.

AdamW3e-4, batch128, weight decay.01, clip1, standardized all8 L1, 12000 updates
per arm, one seed and one budget only. Reuse seeds160(extra init and dropout),
161(matched sequence-uniform/half-startup draws),162(fixed TRAIN512/dev512
panels) for continuity with prior matched controls, not independent replication.
The new execution has different admitted windows; old numeric dev indices
must not be reused. Generate new fixed panels from new membership and retain
the exact sequence/tick identities. Evaluate every500; select dev first-PD
XYZ p95 then mean including step0. Report selected/final/startup/later/object
errors, clipping, normalized all8 loss, and immutable selected predictions.
Locally PROMISING requires both selected mean and p95 improve≥15% with
clipping<1%; otherwise UNCLEAR, without claiming information insufficiency.

## Resources and stop conditions

GPU0/2 in parallel, after checking both idle. GPU1/3 have unknown consumers
and are not touched. Existing main WAM GPU4/5/6/7 continues; project total6,
normal cap after the seven-card collection exception has ended. Each worker
≤1200s, ≤150MB artifacts; aggregate≤300MB, filesystem≥20GiB free. Stop on
nonfinite tensors, admission mismatch, new foreign assigned-GPU consumer,
timeout or disk limit. Do not extend automatically. Isolated tools only;
no production src/config or active detached training snapshot modifications.

Output: `outputs/ptv3-shared-wam-v2/R-native-current-20261015-r1/`.
Entry: `tools/run/probe_native_current.py`; helper:
`tools/run/native_current_information.py`. Exact commands, PIDs, code commit,
GPU observations and elapsed/ETA belong in the output launch receipts.

## Limitations / future evidence

Single-seed exploratory dev repeatedly used for research; no independent
validation or grasp claim. Dataset screening is permissive and may include
failed/partial grasps. Human review annotations do not alter this run's
frozen membership. Additional information may affect optimization as well
as identifiability; a null result does not refute DExplore information.
Positive signals must survive predicted tau and native execution before
changing the production joint recipe or claiming physical benefit.

## Startup

Code6e08ed66. Two known-answer tests passed. Real new-corpus engineering
smoke ran both arms one update each, initial functions matched and all metrics
finite. Independent read-only review found no semantic blocker; its suggestion
to fail closed on the native schema/delta0/future-invariance/full raw-audit
receipt was implemented before the scientific launch. Full matched workers
start concurrently on GPU0/2 under the output-owned supervisor. The baseline
includes the same encoder biases/capacity but no native value access.
