---
schema: ref2dex.probe.v2
probe_id: P-20261005-oracle-y-utility
experiment_id: P-20261005-oracle-y-utility
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 82d210146c0346f45d4854df615409b02092d6e7
claim_id: C3
hypothesis_family: HF-oracle-y-utility
probe_index_in_family: 1
seed_pool: probe
seeds: [263, 265, 266]
decision_changed_if_positive: run synthetic Y noise tolerance before any paired representation fitting
decision_changed_if_negative: stop the fixed short-Y selector and distinguish proxy mismatch from absent candidate opportunity
status: UNCLEAR
run_id: oracle-y-utility-s263
---

# Ref13 Oracle-Y Utility Gate

Result: pending, engineering pairing first; no scientific result yet.
Decision: Reverse necessity testing, starting with perfect-Y candidate selection.

## Decision Note and purpose

Ref13 explicitly reauthorizes paired candidate acquisition after ref12's
user-directed deferral. Same original branch, no new predictor/PPO/execution
forecaster. MissionC3 needs task-meaningful physical prediction: ref12's actual
future flow→Y prognosis does not establish prospective candidate selection.
Question: on identical early-hold prefixes, can perfect short continuation Y
select an intervention that improves a distinct longer stable-grasp Z?
Cheapest decision experiment is a small fixed actual-execution candidate panel.

First 24-environment engineering reference/repeat, then96 environments with
three canonical airplane motions, simulator seed263. One fresh cold simulator
process per branch, frozen self-trained source_e260 checkpointSHA
16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f.
No official-policy or trained-Cm comparison claim. Native .5 hybrid reset,
full initial task/controller/RNG/property verification, identical saved action
prefix replay; PhysX warm solver caches are rebuilt by replay, not restored.
Shadow frozen actor is evaluated during prefix and compared against reference.
After intervention frozen actor continues its own feedback; native targets,
actual q, measured tips/base and physical outcomes recorded. No individual
reset; first native episode termination ends admissible outcomes for that row.

## Frozen candidate/current-state and outcome contracts

Seven candidates: baseline, thumb-yaw±, middle±, grip+, wrist-z+. Independent
finger residual±.2 action =10% driver range (native coupled PD); wrist+.01m.
K8 control steps (30Hz); no selected desired-flow/execution mapping claim.
Eligibility is BEFORE any candidate outcome:10-step history,≥6 consecutive
steps height≥3cm above reference rest plus force-pair proxy, sampled surface
proximity<6cm, all seven current action headrooms, >91 native horizon steps.
One first eligible anchor per environment, no future-active outcome filtering.
Any assigned incomplete90-step window fails implementation; no dropping it.
Candidate clipping after the first eligible tick is retained and reported.

Y8 unchanged ref8/ref12: contact/held/combinedfailure16/32 and lateheightfailure/
heightheldfraction32, ALL beginstep9 after8-step intervention. Fixed utility
U(Y)=Y7 +.25Y3 −Y6 (held fraction +contact bonus −heightfailure), exact ties
baseline first then candidate order. No coefficient/definition/test subset tuning.
Z is DISTINCT: qualify45 consecutive physical held steps (≥3cm and forcepair)
by postdecisionstep60, then no height<2cm or6-step contact loss throughstep90.
Thus1.5s held plus≥1s no-drop follow-up, within3s postdecision window. Forcepair
is aggregate-force proxy, not certified hand-object friction/contact. Z is an
early-hold target, not full-episode/task success. Restheight comes from each
motion's reference, not possibly already-lifted hybrid reset object height.

## Pairing noise screen before alternative arms

Baseline repeat uses its own feedback after trigger, shared teacher-forced
prefix beforehand. Cold initial/property/model/RMS identity and all raw prefix
trace values locked. Per-anchor max raw differences ≤1e-4 for physical72,
q/dq36, handroot13, actorobs/history and≤1e-5 action; pretrigger done exact.
These conservative mixed-unit maxima imply e.g. position≤.1mm and force≤.1mN.
Repeat shortY max difference≤.05 and EXACT same Z. Freeze accepted indices and
hashes in screen.json BEFORE collecting alternatives. Require≥80% assigned
anchors accepted; report ALL rejected counts, Y/Z noise and Wilson upper95.
This empirical Probe noise screen is not a formal population noise bound.
All alternatives must pass prefix screen for EVERY frozen accepted anchor;
fail whole panel on treatment-specific mismatch, never outcome-select rows.

## Decision gates and resources

GateA: report baseline Z, selected GT-Y Z, GT-Z candidate upper bound, per-arm
Z/Y, selection counts, realized step8 tip movement and clipping. Paired anchor
bootstrap2000seed265, adequate≥30 accepted anchors and≥2 motions.
PROMISING if selected gain≥5percentage points and lower95>0; otherwise fixed
selector UNPROMISING with adequate support, elseUNCLEAR. Absent candidate Z
opportunity cannot refute Y globally or prove a need to redefine it.

Only GateA positive→GateB: GT rawY +independent Gaussian noise sigma
0/.025/.05/.1/.2/.4/.8,1000repeats266, common draws acrosssigma,no clipping.
Report active pairwise ordering (abs utilitygap≥.02), GT utilitytop1regret,
selectedZ/gain/CI and retained oracle gain. Synthetic finite-panel threshold,
not learned-model generalization. GateC representation comparison needs
qualifiedA/B plus a separately frozen paired train/test protocol; GateD only
later. Neither is automatically launched from an unqualified A result.

GPU6 only after idle check;≤1 simultaneousGPU. Engineering2×240s shell caps,
main8×240s shell caps (native perbranch180s); total≤2400s wall,≤1.5GiB new
outputs, no fits. Abort on hash/initial/physics/controller drift, nonfinite,
incomplete outcomes, pairing failure, GPU ownership conflict or storage cap.
CPU only label/statistical analysis (no neural model calculation). Failed runs
retained. Output prefixes `oracle-y-smoke-s263-*` and `oracle-y-utility-s263-*`.
Code at `tools/run/collect_oracle_y_candidates.py`, wrapper
`tools/run/run_oracle_y_candidate.sh`, audit `tools/audit/audit_oracle_y_panel.py`.

## Limits / future evidence

One frozen policy, seed, object, local seven-arm panel and current-state-selected
early-hold anchors; oracle selector sees post-treatment Y and is unattainable.
Positive supports proxy utility upper bound here, not deployable planning or
final trained-policy Cm-on/off Mission claim. Multi-seed/newobjects/prospective
point-flow execution and formal Validation are deferred. E/I auxiliary/parallel
until it shows unique ranking/generalization/calibration/constraint value;
no compulsory F→E/I→Y information bottleneck. Negative stops only this fixed
panel/utility/target contract after implementation attribution review.

## Engineering failure record

`oracle-y-smoke-s263-reference` at82d2101 failed before first simulate: new
collector used snapshot36 instead of the existing snapshot55 in139-wide history.
No intervention data or scientific result. Restore existing55+18+13+53 layout;
current model H still consumes only first36q/dq history as inherited. Failed
manifest/initial retained. Retry uses unique `oracle-y-smoke-s263-r2-*` outputs.

R2 reference succeeded:24env,10anchors (6/2/2motions),652ticks,55.30s.
R2 repeat failed at its first selected trigger because GPU indices were used
on the saved CPU actor-history tensor. Correct explicit CPU indexing before
device transfer; no outcome/label/threshold change. Preserve failed manifest.
R3 smoke will repeat both branches at the fixed collector commit, with expanded
source/config/URDF/three canonical motion hashes checked before/after each run.

R3 input-provenance preflight found canonical motion links are directories,
not files;0simulate. Hash each linked `interaction_hand_inspire.pt` instead.
R4 smoke is the fixed retry. Engineering accumulated native wall budget stays
within480s; failed input preflight retained as empty directory plus log.

## R4 pairing blocker / minimal diagnosis

R4 cold/property fingerprint matched exactly, but9anchors/0accepted: prefix
physical115.06, q/dq9.23, shadowaction.1063; Y max1; Z2discordant. Thus no
GateA scientific result and no alternative arm collected. Add first-divergence
replay diagnostic (≤64ticks, stop at first rawprefix1e-4 exceedance) to separate
cold hidden-state/trace defects from numerical PhysX divergence. No threshold
relaxation, candidate or target change. Overall≤2400s budget remains binding.

## Decision Note: deterministic-physics pairing fallback

R5 minimal replay first differs attick1 solely object-force-z (.012724N);
q/dq/root/object/bodyposes exact. Separate R2/R4 fresh references first differ
objectforce tick1 (.01950N), policyaction tick12, q/dq tick13. Consistent with
contact-force numerical variability but not a proof of GPU PhysX root cause.
Cheapest falsifier: single-thread CPU PhysX fresh reference/repeat, same assets,
policy,30Hz/candidate/Y/Z/screen; actor inference remains GPU6. CPU geometry
and task buffers follow physics backend to avoid cross-device simulator state.
New CPU cold-reset RNG distribution and backend are disclosed; never mix GPU
and CPU branches. Runprefix `oracle-y-cpu-smoke-s263-*`, then main CPU panel
only ifrepeat screen passes. This is an engineering pairing repair, no label/
threshold adjustment. Existing≤2400s totalwall/≤1.5GiB cap still applies.

CPU smoke1 failed before simulate due native loader hardcoded CUDA motion
indices/tables. Task-local explicit immutable reference-table device adapter
fixes this, leaves native engine-owned root/DOF/contact views and actor untouched,
refuses unexpected device-mixed buffers. No external/native source alteration.
CPU retry uses unique `oracle-y-cpu-smoke-s263-r2-*`.

CPU retry2 preflight additionally identified immutable `object_points` loaded
onCUDA; add that reference-surface table to the explicit adapter whitelist.
Still0simulate. Retry3 uses unique outputs. All initial preflight errors remain
engineering-only; no GateA result.

CPU retry3 reaches reset but strict cold engine root hash rejects before first
simulate. Next instrument root/DOF error axes to establish whether it is native
normalization or substantive state change; no initial/prefix tolerance changed.

CPU retry4 cold-restore difference is≤1.788e-7 (quaternion), position≤5.96e-8m,
DOF/velocity EXACT. Native CPU pose setters round float32 bits. Bounded adapter
allows CPU cold root xyz/quaternion≤2.5e-7 ONLY, with EXACT velocity/DOF and
fresh-frame0, original task/property/RNG/cache fingerprints otherwise unchanged.
Default paired helper behavior stays bit-exact for all prior callers/GPU.
This documented initial-engine tolerance does not relax prefix/Y/Z acceptance.
Regression rejects velocity differences,largepose/nonfinite. R5 CPU smoke now
checks whether numerical setter allowance actually restores repeatedprefixes.

CPU R5 repeat passed EXACTLY, including identical panel SHA, butonly1/24
early-hold anchor (s3 only), so engineering proof alone gives insufficient
scientific coverage. One minimal GPU PhysX +CPU tensorpipeline/single-thread
smoke distinguishes GPU pipeline force-read variability while preserving GPU
physics substrate. Same immutable table adapter and cold numerical allowance
forCPU tensor views. Only ifpairing passes and adequate anchors exist expand
main. Backend now records actual physx.use_gpu separately from tensor_device
and actor_device; no pretending CPU tensor views mean CPU physics.
