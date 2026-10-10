---
schema: ref2dex.probe.v2
probe_id: P-20261010-lowrank-trajectory-decoder
experiment_id: P-20261010-lowrank-trajectory-decoder
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: 772190e
claim_id: C3
hypothesis_family: HF-trajectory-policy-action-space
probe_index_in_family: 3
seed_pool: probe
seeds: [297]
decision_changed_if_positive: initialize measured-history actor in the fixed learned time basis
decision_changed_if_negative: inspect measured-state anchoring and reconstruction before RL
status: UNPROMISING
run_id: lowrank-trajectory-decoder-20261010-r1
---

# Can a fixed learned time basis preserve executable trajectories?

## Motivation / Decision Note

Mission self-trained policy and eventual matched Cm benefit still require an
executable standalone actor action space. Previous goal turn excluded copied
four-node initialization; current analytic fitting gives32.20mm best-prefix
XYZ RMS and30.67mm FF RMS in the worst window even without bounded c. This
changes the cheapest next choice: learning the temporal shape rather than
re-running linear-node fitting. These are local exclusions, not three probes
with no decision progress and not abandonment of a core Mission hypothesis.

Choose fixed PCA D48 over fitting a large neural decoder now: small deterministic
fit, independent of learning H->c, preserves arbitrary dense temporal modes and
SO3/coupling/bounds. Alternative cubic splines still impose temporal geometry;
neural autoencoder is more costly before verifying representation utility.
No change to Mission claim, no oldTask experiments resumed, no new permission
boundary/branch. Data/basis remains initialization cost, not policy success.

## Protocol

One motion's verified hand-derived geometry q,543query windows,24future frames
clamped at end, measured/hand-derived current q and initial object frame only.
No recorded future q/object/commands. Each future has object-frame XYZ offset,
relative SO3 rotation vector via pi radial map and six sigmoid finger controls.
Translation tanh bound1m, relative rotation pi, native finger bounds/coupling.
Convert to288unconstrained coordinates with fixed scales[.1m-coordinate,.5rot,
4finger-logit]. Fit mean+48PCA components using one GPU SVD; latent standardizes
component scores by sqrt variance floored1e-4. No dimension/scale/seed search.
PCA D maps actor c plus current state to the whole trajectory, no per-query base
policy or online reference; mean is a fixed decoder parameter learned offline.

Offline report all68deployment windows (in-sample representation coverage,
NOT held-out/generalization). Fixed screen first8 hand3D RMS<=5mm, maxpalm<=10mm
and wrist FF target XYZ worst-window3D RMS<=10mm, finite bounded/coupled outputs.
Only if screen passes, one frozen-R native wave16env/542steps/seed297, same
original GT,dense FK,lowrank48,lowrank48 repeat roles4each; replan8, own live q
and object frame, oracle hand-derived c labels. Compare actual achieved grasp,
not GT reference label errors alone. No H->c learned actor yet.

Physical screen as previous card: GT and dense each>=3/4 >=433held+terminal;
each lowrank arm>=3/4 longheld+terminal and clipping<1%. Positive local
PROMISING, both fail with calibration UNPROMISING, otherwise UNCLEAR. No
threshold/node/parameter tuning or repeat simulation after observing results.
Audit saved basis identity, encode/decode, actual tau/FF/features/native PD and
outcomes. Separate live rows are not same-state causal efficacy or Cm benefit.

## Resources / stop

One idleGPU2, <=8GPUmin/512MiB, basis fit+coverage<=120s, conditional native wave
<=300s. No policy training, WM or extra data/seed. CPU handles coordinate
transforms/file audits; GPU SVD/FK/controller/native PhysX. Stop on nonfinite,
input drift, unsafe occupancy or unexpected reset. Do not overwrite outputs.
Stay main/no push. Long-term independent actor and matched policy learning
are required after coverage; this is a Decision Probe, not Validation.

## Results

Fixed code772190e, basis fit+coverage completed1.09s onGPU2, peakTorch7.54MiB.
All5current tiny contract tests passed. In-sample68windows relative dense FK:
prefix8 full-hand3D RMS5.61mm (previous four-node13.97mm), palm maximum23.62mm,
worst-window wrist FF target3D RMS18.50mm, full24 hand3D RMS5.33mm.
Misses unchanged5/10/10mm screen. Local **UNPROMISING** for this coordinate-PCA
initializer's close reconstruction; physical efficacy remains **UNCLEAR** because
conditional native wave was not started. Do not describe it as another grasp
failure or a learned H->c result. No neural/high-level policy training or WM.

Post-hoc decomposition of saved arrays (no tuning or new evaluation): prefix8
wrist XYZ RMS3.58mm/max23.62mm, largest at querytick0/frame1; rotation RMS.0339rad/
max.1147rad; independent finger maxerror.1092rad. Global reconstruction average
improves but the first commanded pose and derivatives are not retained closely.
PCA minimizes scaled unconstrained coordinate variance, not physical-point/FF
error; the next fit should use a physical execution-sensitive objective. This
is a design reason, not an isolated causal explanation of the observed residual.

Artifacts: `outputs/trajectory-policy/lowrank-trajectory-decoder-20261010-r1/{manifest.json,result.json,basis.npz,coverage.npz,decomposition.json}`.
Prefix-fit artifacts retained under `outputs/trajectory-policy/decoder-prefix-fitting-20261010-r1/`.
GPU2 released, no own live process. Native/audit lowrank branches are prepared
but have not been exercised by a simulation; unit roundtrip is not proof of
their native behavior. Do not promote this basis to actor initialization yet.

Follow-up Decision: retain independent trajectory actor, fixed learned D/R and
c48 as current research direction, but target executed-prefix hand geometry
and nominal wrist velocity in the next representation fit. No post-hoc change
to this card's scale/dimension or offline screen; a future method receives a
separate bounded protocol. More expressive temporal D is now justified over
repeated node-copy initialization, while policy/Cm utility remains unproven.

## Limitations / future evidence

Single-motion in-sample PCA coverage; oracle c encoder and frozen warmstarted R.
Projection may miss control-optimal c; no theorem of latent policy feasibility.
Query q is hand-derived offline, live robot q at execution; audit this anchoring
gap. Need measured-history H->c, actor learning and initialization coverage,
native-policy comparison and matched Cm-on/off training. No true future state
or phase/clock/tactile policy inputs are permitted.
