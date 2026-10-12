---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-egodex-auxiliary
experiment_id: P-20261012-goal-point-wam-egodex-auxiliary
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 0fd399e3
claim_id: C3
hypothesis_family: HF-goal-point-wam-egodex-auxiliary
probe_index_in_family: 1
seed_pool: probe
seeds: [283, 284, 285, 286]
decision_changed_if_positive: admit native EgoDex hand auxiliary and start one remote four-rank mixed run
decision_changed_if_negative: repair data or masked auxiliary execution before remote training
status: UNCLEAR
run_id: egodex-auxiliary-20261012-local-r1
---

# Native EgoDex hands as auxiliary training data

Class: Decision. User asks to add EgoDex remotely after a small local test.
This serves the Task's learned hand representation, upstream of the Mission's
object/Cm/RL claim. Test whether native measured hand trajectories can train
the existing WAM actor path without fabricated object goals/world labels or
future-observation leakage. The cheapest test reuses existing official-TRAIN
HDF5 records and one free local GPU. No new branch, data download, multiseed
or matched control; preserve current local and remote scientific outputs.

Result: UNCLEAR overall; local native admission, four-step smoke and200-step
learning Probe passed. Remote repaired full-batch four-rank preflight passed; one mixed scientific model is RUNNING.
Decision: local positive learning signal supports remote same-recipe admission
and one four-rank warm-started auxiliary run, with original runs preserved.

## Decision Note and design

Official native EgoDex has named hand/camera transforms, confidences and RGB,
but no native object pose/scene point-track supervision matching this WAM.
Choose explicit actor-only auxiliary learning: three original full WAM updates
then one EgoDex actor update, one shared model. EgoDex uses masked null object
goal, the shared scene encoder, observed-hand adapter and trajectory decoder;
world/action forward is bypassed, not multiplied by zero after an all-masked
decoder. This expands a hand-motion prior; it does not teach object-goal or
world supervision on EgoDex. Monitor real/static object-goal sensitivity on
main-three and EgoDex held-session actor error for goal neglect/forgetting.
No global question/claim changes and no formal method superiority conclusion.

Reuse existing local six and remote26 official-TRAIN raw records, not the48
offline full-video teacher clips flagged training_allowed=false. Original
manifests remain untouched; new admission permits only native actor labels.
The roughly327GB full archive has FAILED download status and is not admitted.

Native HDF5 bytes must match completed acquisition SHA/size. Native video
metadata must show30Hz and matching annotation frame count. Use right,left;
wrist, five Knuckles, five tips (11 each), matching semantic main points
approximately, not pretending exact MANO anatomy. Confidence>=.5 is a declared
recipe, not official guidance. Reject records lacking native confidence; never
impute it. Require>=6 persistent qualified points in one hand over28 frames,
and keep remaining missing points individually masked. Native H4/K24 only,
no interpolation, synthetic timestamps only express the verified30Hz clock.
One CURRENT camera inverse transforms all history/future positions; no future
camera, full-clip teacher or future hand label enters observed inputs.

Split by session_name with frozen seed283 before model evaluation:20% of
qualified sessions for dev. Subject independence is unknown. Main-three
data/statistics and frozen1536-window source panel remain unchanged; use
fixed64/source for engineering/monitoring evaluation, not a new benchmark.
Data index and sequence hashes are checked. Null-goal windows are auxiliary,
not additional object-goal validation examples.

Local: one free GPU6, batch32,200 updates, lr1e-4, warm-start the latest saved
local main model (actual parent step/hash in input identity), reset optimizer/schedule.<=15min wall/15GPUmin; no local
optimizer checkpoint to preserve disk. Debug4 updates precede this main Probe.
Code is committed before either model run. Required engineering gates: finite
loss/gradient, real masked targets, no auxiliary world loss, no actor future
inputs, source integrity and final parameter identity. Local results report
fit/dev changes and main goal response; they cannot establish causal gains.

After local passes, save/stop only the owned remote main at its synchronized
boundary, preserve its final optimizer/RNG/checkpoint, then warm-start one new
four-rank auxiliary run from those model weights (optimizer reset). Keep old
checkpoint identity distinct. Remote microbatch256/global1024,10000 updates,
same3:1 schedule, lr1e-4, <=6h/1440GPUmin; no extra remote models/controls.
Four-rank full-batch main/aux smoke first, no smoke checkpoint. Only rolling
latest and hardlinked final in new scientific run (no best duplicate).
New code/data/outputs remain within the already authorized isolated NAS scope;
existing remote projects/raw archives untouched. Combined artifacts<=300GB,
298GB proactive stop and remote entire NAS scope<=5.1GB; check projected atomic
save peaks before handoff. Stop on source/input drift, foreign GPU conflicts,
nonfinite/OOM, disk cap or deadline. No silent batch reduction.

## Admission observations

Local native preprocessing preserves original acquisition flags and admits4
records;2 basic_fold records lack confidences and are excluded. Session-disjoint
train1488/dev66 windows; low-confidence fingertips masked, never fabricated.
First preprocessing attempts exposed absent confidences and excessive all11
joint admission; both failed before publishing any dataset. Require6 persistent
points plus per-point masks, reflecting the model's existing missing-point
contract. This change was made from raw coverage, without viewing model results.

## Artifacts and limitations

Local admitted data: `outputs/goal-point-wam/egodex-actor-admission-20261012-local-r1/`.
Runtime outputs: `outputs/goal-point-wam/egodex-auxiliary-20261012-local-r1/`.
The small local dev cohort contains one session and cannot establish broad
generalization. Remote26 raw records independently admitted24 after confidence checks. Adjacent windows overlap; sampled occurrences are not
unique data. No tactile signals or EPIC are introduced. Native hand tracking
has estimation noise; camera-centered hand-only inputs omit object context.
Additional optimization and new supervision lack a matched continuation
control, so before/after trends cannot isolate EgoDex causality. Final robot,
physical forward-model and Cm-on/off claims remain deferred.

[Primary-source contract](../../research/2026-10-12-egodex-training-contract.md).


## Local execution evidence

Codeaaf64037; native core/source/stat bytes unchanged. GPU6 was occupied by a
foreign process: startup guard stopped before any model worker was launched.
Use freeGPU7 instead; foreign jobs untouched. Four-step smoke at e92aac1e
completed in6.82s, finite losses/gradients, one actual auxiliary step.
After adding explicit same-model config checks, local200 steps completed at
aaf64037 in63.89s, batch32/6400 sampled occurrences (150main,50aux updates),
from the native main step6000 checkpoint with optimizer reset. No local model
checkpoint generated; logs, indices, parent/final parameter SHA are retained.
All50 auxiliary updates have world loss exactly0; all losses/gradient norms
finite. Actual GPU7 sample about6GiB, no occupancy on4/5. Three meaningful
CPU tests cover future hand/camera input isolation, fixed current transform,
missing-point masks and world/action bypass; independent read-only review
confirmed core semantics and identified source-guard omissions, repaired
in e92aac1e, and config-only semantic checks, repaired in aaf64037.

| Metric (mm) | Initial warm model | After200 mixed updates |
| --- | ---: | ---: |
| EgoDex fit macro (fixed128 windows) | 30.478 | 25.612 |
| EgoDex held-session EPE (64 windows) | 108.168 | 99.366 |
| EgoDex held-session static reference | 124.353 | 124.353 |
| Main-three actor macro (32/source) | 43.244 | 42.808 |

EgoDex dev improves8.14% versus the same initial warm model and20.09% versus
static, but this dev set is one small session. Main-three macro changes about
1%; ARCTIC alone worsens about1.9%. True goal remains better than static goal
on all three sources in the cached-scene diagnostic. These are single-run
local signals, not proof of causal EgoDex benefit, downstream execution or
full-corpus performance. Remote must use original remote main checkpoint,
not this local auxiliary-trained state: independently splitting larger remote
raw cohort could otherwise place local-fit sessions in remote dev.

[Local result](../../../../../../outputs/goal-point-wam/egodex-auxiliary-20261012-local-r1/train/result.json),
[local input identity](../../../../../../outputs/goal-point-wam/egodex-auxiliary-20261012-local-r1/train/input_manifest.json),
[local learning curve](../../../../../../outputs/goal-point-wam/egodex-auxiliary-20261012-local-r1/train/validation.jsonl).


## Remote handoff budget Decision Note

Local finite/masked execution and positive held-session learning signal permit
remote handoff. Remote admission accepts24 of26 raw records, excludes2 missing
confidence,6484 native frames; train4476/dev1305 windows in disjoint sessions.
The full corpus is still a subset, and repeated occurrences are not new data.
Initial handoff check blocked before any stop signal: the duplicate checkout
made estimated atomic checkpoint peak5.289GB exceed the self-imposed5GB NAS
run envelope. Sparse checkout removes unrelated reproducible checkout files;
old code/data/checkpoints are untouched. Remaining conservative peak5.040GB
includes old final, new rolling+atomic replacement, int32 draw indices and50MB
reserve. Root increases this run's NAS envelope to5.1GB within the unchanged
Campaign300GB aggregate bound; combined298GB proactive guard remains active.
This revises a root-created per-run allocation, not the user's global storage
limit. No new GPU, external data write or identity is needed. Success: preserve
old remote final, full-batch four-rank main/aux smoke, start one mixed model.
Stop on cap/drift/conflict/nonfinite/deadline; no duplicate best checkpoints.


## Four-rank preflight failure and repair

Original remote main saved cleanly at2100; do not overwrite it. First native
auxiliary preflight ran all4 batch256/global1024 updates, but failed during
final collective operations: rank0 entered ALLGATHER while ranks1--3 entered
ALLREDUCE at the same NCCL sequence53, timing out after180s. All original
logs remain under remote egodex-auxiliary-20261012-remote-r1/preflight.

Implementation defect: metric observations exist only on rank0. Testing
`not observations` to decide final evaluation makes other ranks enter an extra
evaluation/barrier while rank0 advances to parameter gathering. Fix0fd399e3
tracks last_evaluation_step identically on every rank; only metric output is
rank0-only. This is engineering failure, not negative evidence on EgoDex or
the model. Local world1 results are unaffected by the distributed-only defect.
New remote r2 code is pinned0fd399e3 and retries the same full-batch4-update
main/aux preflight with no checkpoint, before automatic scientific launch.
CPU chain189914 uses separate status/log/output paths; failed r1 is preserved.
The regression signal is actual four-rank completion plus identical final
parameter hashes, since world1 tests cannot catch this collective mismatch.

[Restart receipt](../../../../../../tmp/egodex-training-20261012-r1/remote_chain_receipt-r2.json).


## Remote mixed scientific training confirmed

Code0fd399e3, run_id=egodex-auxiliary-20261012-remote-r2. Actual four-rank
batch256/global1024 preflight completed4 updates in14.86s after the gate fix;
all four parameter hashes equal
`8b6777e9002e2875eca39a307a30c01d1cf8348aa36943283460d421a604aa6f`.
All losses/gradients finite, auxiliary world term0, no preflight checkpoint.
This actual distributed completion locks down the observed final collective
mismatch. Existing three semantic CPU tests also pass (1.47s); those alone
would not establish distributed correctness.

Scientific launcher190557 has started one4-rank model, native main final2100
model-only warm start, optimizer reset, lr1e-4,10000 mixed updates. At step10
all four synchronized gradient norms are0.7494766116; training logs reach18
with finite main and EgoDex losses. Steps12/16 are real actor-only auxiliary
updates, world loss exactly0. Initial engineering snapshot main steps about
1.413s and native auxiliary steps.352s, implying about3.2h compute remaining
plus validation/atomic checkpoint overhead; this is an initial ETA, not a
sustained performance measurement. All four GPUs sampled100% active, memory
35211--45569MiB/49140MiB devices; no extra jobs for utilization.
NAS scope3.775GB before first scientific checkpoint. Shared watcher4108465
covers local main r3 and remote chain; recent combined294.46GB after removal
of only reproducible local engineering smoke checkpoint files. Independent
remote5.1GB/source/GPU/deadline guards stay active on tunnel disconnect.

Remote processed data is24 native official-TRAIN records/6484 frames,
4476 training windows and1305 held-session windows (five held sessions).
Original main corpus still supplies4,378,478 training windows;2.56M EgoDex
sampled occurrences at10000 updates repeat this small native subset. This is
large-batch mixed training, not full-EgoDex-corpus training or proof of added
unique-data scale. No local auxiliary-fitted weights enter remote dev.
Future evidence: held-session/static comparison, main cached-goal retention,
tracking confidence/anatomy noise and broader native sessions. No causal or
robot-policy benefit is claimed from this launch.

Remote artifacts: isolated scope/egodex-repo/outputs/goal-point-wam/
`egodex-auxiliary-20261012-remote-r2/{preflight,training}/`;
chain status at scope/egodex-chain-status-r2.json.
[Confirmed runtime snapshot](../../../../../../tmp/egodex-training-20261012-r1/remote_observation-r2.json),
[shared budget watcher](../../../../../../tmp/egodex-training-20261012-r1/aggregate_budget_status-r2.json).


## Interim remote2000 validation and monitor recovery

Actual training continues beyond2600/10000, original scientific launcher190557
alive, code/config/model unchanged. Native held-session macro EPE at0/1000/2000
=63.280/47.646/48.924mm; static59.815mm. Thus2000 is22.69% below warm start
and18.21% below static, but slightly worse than1000; no monotonic improvement
or causal EgoDex claim. Main64/source actor macro46.028→45.149mm, with true
goal still better than static on all3 sources. Main updates median1.419s,
auxiliary.356s; initial remaining ETA~2.4h plus evaluations/saves.

Phase-chain189914 died on a transient `du -sb` return1 while NAS output changed.
Scientific child launcher/model never stopped: chain failure is not training
failure. Preserve chain log/status; do not restart active training or mutate
its guarded files. New CPU monitor198540 adopts the existing launcher and
tolerates isolated stat errors (three consecutive errors request owned save/stop);
cross-host watcher4161832 now reads actual training group_status.json and can
signal only the identified scientific launcher. Shared artifacts296.40GB,
remote4.406GB; thresholds unchanged. Full remote training result remains pending.
[Interim snapshot](../../../../../../tmp/egodex-training-20261012-r1/status_update-20261012-1430.json),
[monitor receipt](../../../../../../tmp/egodex-training-20261012-r1/remote_adopt_receipt.json),
[actual-launcher budget watcher](../../../../../../tmp/egodex-training-20261012-r1/aggregate_budget_status-r3.json).
