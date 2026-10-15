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
status: UNCLEAR
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

## Final12000-update result

Both arms finished12000 updates and exited0; supervisor and automatic matched
merge completed in247.46s. No budget extension. Same fixed new dev512, with
256 startup and256 later frames; select by the predeclared p95-then-mean rule:

| Arm | Selected update | XYZ mean mm | XYZ p95 mm | Clipping |
| --- | ---: | ---: | ---: | ---: |
| baseline | 9500 | 64.08 | 222.52 | 0% |
| native_current | 11500 | 57.23 | 192.58 | 0.1953% |

Native current inputs reduce selected mean10.69% and p9513.45%. This is an
exploratory favorable trend, but neither reaches the predeclared simultaneous
15%/15% gate: conclusion UNCLEAR. Do not change the production input contract
or claim better physical grasping from this GT-tau offline diagnostic. The
final12000-step mean/p95 are63.38/232.50 versus58.29/197.18mm; the direction
also holds at the common final update. Selected rotation/finger p95 are
1.1981/0.3510rad versus0.9963/0.3031rad. Results do not attribute any gain
specifically to contact because the intervention adds the full721D block.

TRAIN110250/dev12049 retained windows are shared exactly. Matched checks pass:
parent, TRAIN-only native stats, panel indices and sequence/tick identities,
initial functions, hyperparameters, original selection. Independent raw
q[t]/q_PD[t] plus saved predictions and native controller XYZ reconstruction
reproduce both selected mean/p95 exactly. Two known-answer tests, real two-arm
one-update wiring smoke, independent semantic review and repository path
verification passed. Source contract rejects future actor branches or missing
full raw-audit admission. No production model or active R5 snapshot was edited.

Resources: GPU0/2 simultaneous,30 stable samples average utilization16.7%/15.8%,
physical memory1229MiB each, approximately0.017s/update; small R computation is
light, with Python/optimizer dispatch limiting utilization. All neural updates
and inference ran on GPU; file unpacking and independent numeric audit on CPU.
Aggregate artifacts82,674,903B before this documentation, below300MB; no GPU
process remains for either worker. Original four-card WAM fine-tune continues.

Evidence: run `result.json`, `native-current.png`, `launch.json`,
`startup_gpu_samples.json`, `startup_observation.json`, per-arm manifests,
curves, selected predictions and immutable selected checkpoints. Worker code
6e08ed66; independent final merge/raw XYZ audit code5a9240d9. Decision: stop
this fitting budget, retain the modest native-information signal as UNCLEAR,
and continue the separately authorized human data-quality review. A later
predicted-tau or native-execution qualification requires a separately scoped
experiment; none is started by this result.
