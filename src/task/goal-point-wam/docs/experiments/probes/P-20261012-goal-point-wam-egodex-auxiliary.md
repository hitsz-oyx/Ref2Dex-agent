---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-egodex-auxiliary
experiment_id: P-20261012-goal-point-wam-egodex-auxiliary
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: c3295651
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

Result: UNCLEAR; native admission is prepared, model training not yet verified.
Decision: complete the local engineering/learning Probe before remote handoff.

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
298GB proactive stop and remote entire NAS scope<=5GB; check projected atomic
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
generalization. Remote26 raw records need independent admission and may shrink
after confidence checks. Adjacent windows overlap; sampled occurrences are not
unique data. No tactile signals or EPIC are introduced. Native hand tracking
has estimation noise; camera-centered hand-only inputs omit object context.
Additional optimization and new supervision lack a matched continuation
control, so before/after trends cannot isolate EgoDex causality. Final robot,
physical forward-model and Cm-on/off claims remain deferred.

[Primary-source contract](../../research/2026-10-12-egodex-training-contract.md).
