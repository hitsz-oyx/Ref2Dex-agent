---
schema: ref2dex.probe.v2
probe_id: P-20261005-contact-innovation
experiment_id: P-20261005-contact-innovation
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 82ae2b699a77006fd8ca020d0efeb487872e006f
claim_id: C3
hypothesis_family: HF-contact-innovation
probe_index_in_family: 1
seed_pool: probe
seeds: [253]
decision_changed_if_positive: qualify bounded contact-relative geometry for strict nested consequence OOF and downstream comparison
decision_changed_if_negative: stop this surface-proxy and nuisance factorial without neural epoch or geometry sweeps
status: UNPROMISING
run_id: contact-innovation-s253
---

# Does bounded surface-relative motion add information beyond joint/finger flow?

Result: UNPROMISING for the fixed bounded basis / nuisance / ridge contract.
Decision: Close and preserve this result; user ref11 redirects work to oracle hand-flow planning.

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

## Engineering pre-fit failure

First GPU6 smoke `contact-innovation-smoke-s253`, source8a04df7, fails before
any ridge fit: GPU chunk index applied to CPU collector input. Repair transfers
only whitelisted current history/root/before tensors to the candidate device
before batching. Retain failed manifest/log, no scientific evidence. Targeted
CPU-input/GPU-q regression checks the actual device boundary; no research
variable/gate/budget change. Replacement smoke uses a new unique folder.

## Completed results and independent review

Main `contact-innovation-s253`, code82ae2b6, GPU6:3.185s model/run stage,
3.337s complete manifest, peak318.26MiB. Ten closed-form fits; no neural,
simulation, task or policy fitting. Replacement smoke3.10s is engineering-only
UNCLEAR, not scientific evidence; original pre-fit failure remains intact.
Main+smoke+failed artifacts about28MiB, below90MiB cap.

| Nuisance / action | E test MSE | I test MSE |
| --- | ---: | ---: |
| Physics State | 1.382006 | .988822 |
| Physics Joint | 1.391943 | .987559 |
| Physics Finger | 1.405803 | .997014 |
| Physics Contact | 1.390506 | .988571 |
| Physics ContactShuffled | 1.385880 | .994332 |
| OOFState State | .583821 | 1.034987 |
| OOFState Joint | .583996 | 1.029396 |
| OOFState Finger | .591395 | 1.046253 |
| OOFState Contact | .568375 | 1.038838 |
| OOFState ContactShuffled | .592439 | 1.044219 |
| Prior frozen State | .501971 | .883696 |

Primary PhysicsContact I gain vs matchedState.025%(95%CI−3.136..+2.651%),
vsJoint−.103%(−2.747..+2.280%), vsFinger+.847%(−1.292..+2.847%),
vs trainedshuffle+.579%(−2.593..+3.392%). Test-only shuffle penalty2.512%
(+.279..+4.959%) is a measurable local action sensitivity but below the
registered3% threshold. PhysicsContact is worse than priorState by11.868%.

Primary raw Icontrast corr.4047/sign62.86%/zeroMSEgain9.946%/amplitude.1871;
arm-centered gain14.407%, plus-minus corr.5369/gain25.067%/sign56.96%.
These exploratory subset scores do not rescue B or the failed factual-MSE
gate. Four main gates fail. Existing GT contrasts are current-state-adjusted
estimates on the already exposed166test windows, not paired same-state outcomes.

PhysicsContact versus OOFStateContact I gain4.84%(CI+.045..+8.64%) is a
nuisance/stacking/estimator sensitivity, not a uniquely identified failure cause.
Both allow common correction; imposing candidate-centering is not responsible
for this fixed screen. Post-result I families: PhysicsContact handforce1.3804,
objectforce1.5545,proximity.3516,proxy.5162; priorState1.1450/1.5245/.3088/.5289.
Do not select proxy or another family to rescue the overall gate.

Contact actions are nonzero. Predicted baseline soft-proximity means for five
fingers are train.164/.210/.059/.149/.067, test.171/.218/.052/.150/.070.
All30 action feature source candidate stds.00556.. .09039 fall below fixed.1
floor: equalα32 does not mean equal effective action shrinkage. This deliberate
basis/scale/regularizer restriction limits the negative; no floor/α search.

Root full input/norm/FK/nuisance/weight/candidate/shuffle/metric replay errors0,
normal-equation relative residual≤3.15e−15,3.38s including figure. Independent
CPU saved-matrix/statistical review.54s: predictions0, normal equations≤4.61e−15,
source-only stats and group/partition shuffle correct, metrics/contrasts0 and
bootstrap≤1e−6. No implementation defect affecting the negative was found.
Tiny matrix/statistics audit CPU avoids GPU startup/repeating root FK.
66Task tests including explicit CPU-input/GPU-q regression pass.

Reviewmd/json plus root `review_provenance.json`, signed vectors and standalone
`contact_innovation.png` bind evidence to the immutable main diagnostic.

## User-directed closeout / next route

User now explicitly stops the execution-predictor branch and directs ref11:
separate planning from control. No planned contrast-stability diagnostic was
implemented or run. No additional basis, scale, nuisance or epoch fitting.
Preserve all code/results; no task-owned GPU process remains.

Next route tests State versus actual measured/FK endpoint hand-flow versus
two trajectory chunks (0→4,4→8), with matched oracle-flow shuffle controls,
E/I only. This is a post-treatment oracle planning-representation Probe; it
does not deliver prospective joint control, paired candidate ground truth,
causal plan validity or final trained-policy Cm utility. Native execution
prediction is explicitly deferred. Global Mission claim remains unchanged.
