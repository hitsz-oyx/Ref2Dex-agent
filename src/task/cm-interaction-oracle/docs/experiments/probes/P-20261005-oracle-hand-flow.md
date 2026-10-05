---
schema: ref2dex.probe.v2
probe_id: P-20261005-oracle-hand-flow
experiment_id: P-20261005-oracle-hand-flow
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-oracle-flow-planning
git_commit: pending
claim_id: C3
hypothesis_family: HF-oracle-hand-flow
probe_index_in_family: 1
seed_pool: probe
seeds: [255, 256, 257]
decision_changed_if_positive: retain oracle flow representation and test whether its E/I support controlled flow-space planning
decision_changed_if_negative: distinguish oracle raw-flow model and temporal representation before execution mapping resumes
status: UNCLEAR
run_id: oracle-hand-flow-s255
---

# Ref11: oracle endpoint and temporal hand flow → E/I

Result: UNCLEAR: protocol frozen, run pending.
Decision: Separate oracle planning input from the stopped native execution route.

## Root Decision Note / user-directed Decision

User explicitly stops the execution branch and directs Task ref11. The final
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
