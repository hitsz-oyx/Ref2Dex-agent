---
schema: ref2dex.probe.v2
probe_id: P-20261005-contact-innovation
experiment_id: P-20261005-contact-innovation
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: pending
claim_id: C3
hypothesis_family: HF-contact-innovation
probe_index_in_family: 1
seed_pool: probe
seeds: [253]
decision_changed_if_positive: qualify bounded contact-relative geometry for strict nested consequence OOF and downstream comparison
decision_changed_if_negative: stop this surface-proxy and nuisance factorial without neural epoch or geometry sweeps
status: UNCLEAR
run_id: contact-innovation-s253
---

# Does bounded surface-relative motion add information beyond joint/finger flow?

Result: UNCLEAR: protocol frozen, run pending.
Decision: Ten closed-form source fits separate contact geometry from state-nuisance choice.

## Root Decision Note / Decision experiment

MissionB/ref10 needs physical action-conditioned E/I that adds task/policy
information. Finger mean/RMS and centered MLP failed; candidate-centering
locks common predictions to a potentially biased state nuisance, while the
mean/RMS summary omits object-relative motion direction. Continued fitting would not distinguish
those mechanisms. Root chooses a bounded physical-contact action basis and
two matched nuisance contracts, with closed-form ridge to remove finite
optimizer budget as a confounder. This is a new input/estimator question,
not another epoch/seed retry or radius search. Existing data only.

GPU6 if idle, main≤120s, smoke≤120s, replay≤120s, each main/smoke<40MiB,
combined new artifacts≤90MiB. No new simulation, checkpoint overwrite or
policy fitting. Stop on source/input/hash drift, nonfinite, nuisance replay
>1e−5, normal-equation residual>1e−8 or resource conflict. Success permits
strict nested consequence OOF/task-information; negative stops this fixed
surface-proxy family. No core claim, permissions or long-term identity change.

## Immutable data, causal geometry and physical meaning

Reuse all854windows, source688/test166 from spatial-s245; environment125/31,
same E12/I14 step8 and source scales. DatasetSHA138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149.
Execution fingerq uses repaired source OOF/testfull forecast, never actualq8,
future root/object/force. Preserve continuous wrist angles. Shared candidate
wrist is CURRENT nativeq6; FK predicted finger endpoints in current object
frame, current measured actor root. Arm0 is recipient predicted baselinefinger
endpoint with that same currentwrist. This explicitly tests local finger
geometry; future shared wrist/object evolution is not claimed predicted.

Use the same64frozen object surface samples and normals,120handpoints ordered
index/middle/pinky/ring/thumb,24each. No radius cutoff: every point contributes;
soft weight exp(−(nearest sampled distance/20mm)^2). Nearest sample/normal
is a proximity proxy, not an exact SDF, certified contact or load/friction law.
Candidate-minus-arm0 action scalars per finger:

1. Weighted signed movement along the arm0 nearest object normal /20mm.
2. Weighted approaching-normal movement /20mm.
3. Weighted tangential movement magnitude /20mm.
4. Change in mean soft surface proximity.
5. Change in mean positive(1−distance/20mm).
6. Change in mean distance/20mm capped5.

Average24points (do not divide by tiny contact mass), cap scalars±5, pad30→32.
Arm0 is exactlyzero. Positive signed motion follows outward surface normal;
approach uses negative part. These are frame-invariant local physical proxies.

Common inputs shared by ALL models: previous H156, plus current and predicted
arm0 surface summaries (six scalars/finger each: mean/min cappedgap, soft
proximity, signedplane gap,2cmfraction,RMSgap); source-only standardization,
then cap common features±8. Joint32 and hand-base mean/RMS flow32 are exactly
previous causal forecast inputs. Action norms use all15known candidates in
source688 only, stdfloor.1, normalized features cap±5. No unrestricted
state×action products, action PCA, unseen-dose claims or test preprocessing.

## Two nuisances × five matched action fits

Physics baseline: constant world linear/angular velocity for E, current I.
OOFState baseline: immutable three-environment-fold State prediction for train,
full-source State for test. Replay its four weights before fitting. Both train
targets (z−baseline)/sourceScale. Both allow a learned common-state correction;
NO candidate-centering convention. This can repair common nuisance errors.

For EACH nuisance fit State(actionblank), Joint, Finger, Contact and separately
trained ContactShuffled. Same common216+action32 dimensions, centeredfloat64
ridgeα32 and unpenalized intercept, source factual rows only. Closed-form
solution audited by normal equations. Shuffle factual arm within partition/
wave/motion/phase seed253, selecting recipient candidates, never donorgeometry.
Frozen test-only shuffle distinguishes trained contact sensitivity. All15same-
state candidates stored, arm-minus-zero compared to existing adjusted marginal
GT14arm estimates (exposed/shared dataset, exploratory).

Primary geometry branch fixed Physics_Contact. PROMISING only if I improvement
≥5% over frozenState with paired-env-bootstrap lower95>0, ≥3% vs ALL matched
Physics State/Joint/Finger/trainedShuffle with lower95>0, frozen testshuffle
penalty≥3% and intervalpositive, raw Icontrast corr≥.5/sign≥.65/zeroMSEgain≥.1,
and arm-centered gain>0. Otherwise fixed contract UNPROMISING. All E/I metrics,
31-env bootstrap2000 and alternate OOFState mode reported without winner
selection. Nuisance sensitivity is descriptive, not proof of unique cause.
Positive does not authorize policy/selector from this full-test-only screen.

## Deferred evidence and outputs

No nested consequence OOF, independent new contrast/seed, paired same-state
counterfactual, unseen dose/object or matched trained-policy test here. These
remain future decision/evidence requirements; never claim final Cm utility.
No contact-normal/kernel/regularization sweep. Negative cannot refute all
contact geometry, nonlinear state-dependent physical laws or Cm hypothesis.

Output `outputs/cm-interaction-oracle/contact-innovation-s253/`: manifest,
fit_records, diagnostics (features/norms/tenridgeweights/predictions/candidates),
result; replay, signed vectors and plot added without fitting. Source code,
dataset, original/corrected forecast/source diagnostics/manifests/assets bind
provenance. Preserve engineering smoke as UNCLEAR and any failed runs.
