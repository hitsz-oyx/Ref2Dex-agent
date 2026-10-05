---
schema: ref2dex.probe.v2
probe_id: P-20261005-oracle-hand-flow
experiment_id: P-20261005-oracle-hand-flow
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 452d7e5dc7d8ad39a1769e3aed13724a4c398aad
claim_id: C3
hypothesis_family: HF-oracle-hand-flow
probe_index_in_family: 1
seed_pool: probe
seeds: [255, 256, 257]
decision_changed_if_positive: retain oracle flow representation and test whether its E/I support controlled flow-space planning
decision_changed_if_negative: distinguish oracle raw-flow model and temporal representation before execution mapping resumes
status: UNPROMISING
run_id: oracle-hand-flow-s255
---

# Ref11: oracle endpoint and temporal hand flow → E/I

Result: UNPROMISING for the predeclared joint E/I gate at this fixed budget; Chunk E passes, I remains uncertain.
Decision: Retain the oracle flow E signal and I uncertainty; paired work is deferred and ref12 tests task-information transfer on the original branch.

## Root Decision Note / user-directed Decision

User pauses the old execution route and directs Task ref11 on the original
`agent/cm-interaction-oracle` branch; no new branch is needed. The final
Mission remains matched Cm-on/off trained-policy utility; this stage asks only
whether actual hand motion predicts E/I. Prior RealizedSurface endpoint used
V13 bottleneck, so it did not separate temporal information from representation.
Root chooses State, raw corresponding endpoint points and two temporal chunks,
with equal-width direct heads and two trained/frozen flow shuffle controls.
No execution model, PD/native action, arm or taskY input; no V13/token pooling.
Source history uses only q/dq physical prefix, never previous controller action.

Same existing854windows and688/166environment split; GPU6 ifidle, main≤120s,
smoke≤120s, replay≤120s, combined new outputs≤60MiB, eachfitrun≤25MiB.
Stop on hash/input drift, FK/live error≥.1mm, nonfinite, resource conflict
or storage cap. Positive oracle prediction permits targeted planning-contract
work; negative stops this fixed raw-flow head, not all OI-Cm. No new simulation
or control/policy fit. No external authorization or global-claim change.

## Oracle and current-state contracts

DatasetSHA138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149.
Read split only from prior GT-s231-r2 diagnostic; no prior model weights.
Use actual nativeq atsteps4 and8 ONLY to reconstruct oracle hand surface,
same120corresponding points (24perfinger), measured CURRENT actor root. Do not
clip continuous wrist angles or predict execution. Compare FK wristbase and
five real fingertip positions at0/4/8, error<.1mm. All world endpoint positions
are expressed in the CURRENT object frame; no future object pose, force or
E/I field enters action. Future measured hand poses only validate FK.

Endpoint F08=P8−P0,360values padded to720. Chunk=[P4−P0,P8−P4],720values;
both use fixed20mm scale, no action PCA, local filtering, token or learned
latent bottleneck. Check F04+F48=F08. State receives720zeros.

H156 shared across all five: source-standardized currentphysical72 (hand body
positions centered at current object, object absolute positionzero), explicit
currentq/dq36, source-only physicalq/dq historyPCA32 and currenthand-point/
normal geometryPCA16. No currentbase action, PD, candidate arm, actorobs or
reference future sequence. World object orientation/velocity/force remain for
world-axis E12/I14 targets. Object canonical shape fixedairplane. Geometry PCA
is current-state compression only; action flow keeps all720slots.

Predict same step8E12 and I14 (lognetforce/proximity/proxy; no certifiedcontact).
Labels source-mean/source-scale standardized (floor.001), absolute target,
no learned nuisance or action-centering. All five876→64tanh→32tanh→26 heads
share init255,minibatch64/source-window schedule256,300AdamW.001/decay.01,
clip2. No epoch/checkpoint/test selection. Trained EndpointShuffled and
ChunkShuffled use donor flow tensors within partition/wave/motion/phase257;
recipient H/labels retained. Frozen testshuffle same flow permutations.
Transplanted fields need not be physically executable; they are negative
controls, never supervised same-state candidate ground truth.

## Prediction gates and contrast evidence boundary

Primary temporal branch Chunk. For EACH E and I: gain≥5% vsState,≥3% vs
trainmean and separately trainedmatching shuffle; all paired-environment
bootstrap2000 lower95>0. Frozen testshuffle penalty≥3% and intervalpositive.
PROMISING prediction only if both channels pass; otherwise fixedfit UNPROMISING.
Endpoint uses same declared per-link gates as a secondary mechanism screen;
no best-head selection. Chunk-vsEndpoint quantifies temporal increment without
requiring it for basic oracle flow information. Persistence also reported.

Actual trajectory is post-treatment and can reflect object/controller response.
Predictive oracle information does NOT establish an exogenous plan→outcome
law, attainable candidate flows, inverse control or prospective planning benefit.
No policy utility claim, Y scorer or selector here.

Audit exact duplicate CURRENT observed snapshots (physical before, history,
root, measured tips/base) ontest, without rounding or future filtering. Only
their factual flow/outcome differences can be reported as observed matched-
state contrast; ≥30pairs/≥10environments required for a supported Probe score
(Icorr≥.5,sign≥.65,zeroMSEgain≥.1). Equal observations still do not guarantee
restored hidden simulator/contact state. If absent, contrast is UNCLEAR;
never substitute donor E/I into a recipient state and label it counterfactual.
Same-state causal candidate validation remains a separate later experiment.

## Artifacts and deferred evidence

`outputs/cm-interaction-oracle/oracle-hand-flow-s255/` stores hashes, current
norms/PCA, raw720actions, fiveweights/predictions, sourcecurves, recipient
permutation, observedpair support and result. No prior/checkpoint duplication.
Smoke oneupdate is engineering-only UNCLEAR; preserve anyfailed runs.
GPU saved-input/FK/weight replay and independent anomaly/closing review before
interpreting negative. Multi-seed, newstates/objects, prospective prescribed
flows and matched trained-policy benefit remain deferred; not waived.


## Completed Probe: matched prediction information

Main `oracle-hand-flow-s255` completed on the original branch at commit
`452d7e5dc7d8ad39a1769e3aed13724a4c398aad`. Source688/test166 windows,
125/31 environments. Five heads each59066parameters, identical initial hash
and300updates; no fit/checkpoint/seed selected using test. All target MSEs
below use source per-coordinate scale and test-window mean.

| Input | E MSE | I MSE |
| --- | ---: | ---: |
| State | 0.528949 | 0.604857 |
| Endpoint | 0.505453 | 0.558270 |
| Chunk (primary) | 0.489677 | 0.556029 |
| EndpointShuffled (trained) | 0.545051 | 0.622444 |
| ChunkShuffled (trained) | 0.565084 | 0.616904 |
| TrainMean | 0.753523 | 0.890693 |
| Persistence | 0.882805 | 1.045766 |

| Comparison: relative error reduction | E gain [95% env CI] | I gain [95% env CI] |
| --- | ---: | ---: |
| Endpoint vs State | 4.44% [-2.03,10.78] | 7.70% [-0.17,15.21] |
| Chunk vs State | 7.42% [0.46,14.04] | 8.07% [-1.25,17.62] |
| Chunk vs trained ChunkShuffled | 13.34% [8.47,18.10] | 9.87% [1.66,19.00] |
| Chunk vs Endpoint | 3.12% [-0.81,7.04] | 0.40% [-5.82,6.13] |

Frozen Endpoint test shuffle increases E/I error27.67%/16.85%; frozen Chunk
shuffle increases37.81% [26.07,50.45] /27.53% [14.19,43.97]. Chunk vsTrainMean
gains35.01%/37.57% with positive intervals. Chunk E passes ALL declared gates;
Chunk I fails only the positive lower95 vsState condition. Endpoint E fails
minimum5% vsState and its CI, Endpoint I fails CI vsState. Thus the joint E/I
primary prediction link is UNPROMISING under its fixed protocol, while the
E channel supplies a PROMISING exploratory signal. No formal scientific
support/refutation, convergence, stable I gain, or temporal superiority claim.

The new common H exposes clean current physical/q/dq and current geometry;
its State I0.604857 differs from the old State0.883696 input/target/nuisance
contract. Cross-run MSE improvement cannot be credited to flow. Within this
Probe, every action model has the SAME new H, target, preprocessing and budget.
Model capacity is equal but endpoint has360nonzero action slots and chunks720;
the result measures the combined temporal representation, not an isolated
architectural effect. Source all-channel MSE State0.503609/Endpoint0.483439/
Chunk0.459397; finite300updates do not establish optimization sufficiency.

## Geometry, contrast support and engineering evidence

FK vs measured tips at0/4/8 max errors0.02989/0.02956/0.02803mm. Base matrix
max errors7.61e-6/7.55e-6/7.57e-6; translational entries are metres, rotational
entries dimensionless. Sum of scaled chunks matches endpoint within1.91e-6
(normalized units). Continuous wrist and current object/root are preserved.
No future object pose or force enters H/action; measured future hand motion
is intentionally an oracle post-treatment action.

Test exact observed-state duplicates:0groups/0pairs. The action-contrast score
is UNCLEAR and was not computed. Shuffling a donor's whole flow retains the
recipient H/label, serving solely as a negative control. This does not provide
counterfactual outcomes, achievable plans or same-state candidate ranking.

Oneupdate smoke is engineering-only UNCLEAR. Its historical manifest records
the briefly created branch; the code was fast-forwarded into the original
branch before the main run and that extra branch was removed per user direction.
No output/manifest was rewritten. Main result5.62s, manifest5.75s onphysical
GPU6 (logicalcuda:0); main+smoke approximately24.7MiB before tiny reviews.
Full Task verification69tests passed, including GPU FK/index regression.
GPU audit regenerated all inputs, source normalizers, raw flows, five saved
heads, two frozen test shuffles, labels and bootstrap summaries: ALL differences
0. No new fits. Replay1.76s; plot/export total2.19s. Artifacts:
`engineering_replay.json`, `oracle_hand_flow.png`, original `diagnostic.pt`,
`fit_records.json`, `result.json`, `manifest.json` in the main output directory.
Independent review is archived as `independent_review.md/json` with hash-bound
`review_provenance.json`. No implementation defect affecting the mixed result
was found. Independent CPU source/current-state and five saved-head algebraic
replay normalized error≤1.34e-6; metrics,2000env bootstrap and gates agree.
CPU0.17s, no new fits/simulation; CPU was used for small algebra/statistics while
root performed full GPU FK/input reconstruction. This review does not prove
convergence or upgrade oracle prediction into a prospective planning claim.

## Root next Decision / limitations and future evidence

Keep the old execution predictor route paused. Preserve the E prediction signal
and I uncertainty; do not relax thresholds, repeat seeds or extend this exposed
300update fit. Neither Chunk nor Endpoint superiority is established, so do not
choose a more expensive temporal/spatial architecture from their point estimates.

A read-only prerequisite inspection is complete: `CmResidual/paired_evaluation.py`
explicitly restricts capture/restore to never-stepped simulators because PhysX
solver caches are not serialized. Thus copying warm root/DOF tensors is NOT
full-state restoration. `CmResidual/parallel_sim_pair.py` already uses a
base/repeat/alternate design to bound hidden solver differences, but currently
covers only one step and one wrist alternative. The ref7 collector stores
observations and history, not complete physics/controller/RNG snapshots.

User now explicitly defers paired data and its related engineering check.
The previously considered parallel matched-history/base-repeat-alternate8step
approach remains future evidence only; do not implement it, run its smoke or
collect data under this current stage. Read-only feasibility inspection already
performed is preserved above. No new simulator runs or controller work were
executed. Complete the existing prediction Probe's review/docs/local commit;
keep native execution prediction paused. Missing same-state candidate evidence
is a scope limitation, not a blocker for reporting this matched prediction
Probe or a refutation of oracle flow.

Multi-seed prediction Validation, unseen objects/states, denser within-chunk
trajectory, target-coordinate/model design, source-only optimization sufficiency,
paired contrasts, prescribed attainable flows, E/I→taskY and matched trained
Cm-on/off utility remain future evidence. Current geometry samples and I proxies
are not certified contact, force law or load-bearing utility. No core research
question/claim, permissions or global budget was changed.
