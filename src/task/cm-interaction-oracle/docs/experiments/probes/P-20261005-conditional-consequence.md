---
schema: ref2dex.probe.v2
probe_id: P-20261005-conditional-consequence
experiment_id: P-20261005-conditional-consequence
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: pending
claim_id: C3
hypothesis_family: HF-conditional-consequence
probe_index_in_family: 1
seed_pool: probe
seeds: [237, 238]
decision_changed_if_positive: prioritize bounded same-state candidate mechanism before matched trained-policy utility
decision_changed_if_negative: identify conditional prediction versus task-transfer bottleneck without treating reconstruction as Cm utility
status: PLANNED
run_id: conditional-consequence-s237
---

# Can predicted action consequences retain oracle task information?

Result: Pending fixed conditional-predictability and OOF task-transfer Probe.
Decision: Test action contrasts and value against direct Ha before candidate/policy integration.

## Motivation and root Decision Note

Ref9 explicitly requests H vs Ha vs shuffled-a consequence prediction on ref7,
then task value against direct Ha. Ref8 GT prognosis PROMISING and noisy arm
bridge positive, full sufficiency UNCLEAR. This Decision serves GT→Cm→trained
policy utility; it does not retry the ref7 negative own-I gate or redefine
Mission. Cheapest discriminator: frozen existing854 windows and ref8 split,
one declared fit budget per model, no simulation or seed/width/horizon sweep.

Root keeps E12/I14 and14D intended arm-onehot EXACTLY ref8. Counterfactual
predictions use decision-time intended action only; future actions/PD/feedback
correction not inputs. New held categorical arm cannot be meaningfully inferred
from an unseen onehot dimension; do not silently swap action representation
just to report held-arm generalization. Instead PREDECLARE two-direction
cross-half extrapolation, holding both outer-test environments and opposite
five-wave half out of fitting. Held-arm extrapolation deferred to a future
physically continuous action representation if this Decision warrants it.
Signed18 extension is deferred sensitivity, not fitted to rescue main results.

Resource: idleGPU6 only, main timeout600s, synthetic engineering smoke≤60s,
combined cap660s, outputs≤100MiB. CPU label/hash/OLS/bootstrap/review only.
Stop on dataset/source drift, nonfinite, OOF/environment leakage or resource
conflict. Keep failed runs. Positive leads to a separately frozen candidate
mechanism Probe, not immediate PPO; negative diagnoses which link failed,
with independent review before closing this local method. User requires a
review agent for anomalous results; reviewer already checking design, and
will explicitly review the delivered anomalies. Boundaries unchanged.

## Frozen data and fitting contract

DatasetSHA `138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149`,
collectiona81b36d; prior ref8 `gt-consequence-s231-r2`, source08975e1.
Use its EXACT train688/test166 indices,125/31 environments and normalizers/PCA
for full outer-train fits. H=PCA32 historical/actor/context/base+physical72,
14action slots, MLP64/32tanh. ALL854 risk windows retained including193 early
failures. Mediatorstep8 E12/I14, Y eight old continuation/physical heads at9..32.
Do not change labels, model widths, horizon or physical action dose.

Training: FIXED1500 fullbatch epochs, AdamW.002/weightdecay.001, clip2,
seed237 initialization; no test-selected checkpoint or extra epochs. This
addresses ref8's small200epoch warning by a7.5× larger predeclared budget,
not a convergence assumption. Record every trainloss and last200 fractional
change; a model still improving>5% is flagged as fitting-limited and cannot
by itself justify closing the general predictor/representation hypothesis.
Same architecture and init within H/Ha/shuffle predictor pair and all new
scorers, same1500updates. Fit budget/time/params reported independently of RL
sample efficiency. Neural model computation usesGPU.

Three full predictors: H→E/I, Ha→E/I, H+shuffled-a→E/I. Permute assignment
within wave×motion×phase-quarter, separately for fit/hold/test, seed238.
Also evaluate the FROZEN Ha predictor with permuted TEST actions; report
permutation changed rates. Targets and states stay fixed in all shuffles.
E/I target standardization train only. Report E12/I14/joint prediction MSE,
per-axis rawunit MAE, train/test gaps and physical/prognostic transfer.

Three motion-stratified environment OOF folds seed237 inside outertrain;
each H/Ha/shuffled predictor fitted ONLYother-fold environments. Each fold's
H PCA+all feature/target normalizers fit ONLYfold-fit indices; save provenance.
Convert raw predictions to outertrain-standardized E/I for downstream. All
outertrain scorer inputs use OOFPREDICTIONS, outertest uses full-train predictor.
Test Y never enters predictor, preprocessing or early stopping.

Downstream primary: fit ONCE then freeze H,Ha,GT_HEI,P_H,P_Ha,P_Shuffled,
Ha_P_Ha using same162D slots (signed18 zero), init237/1500epochs, allY8 heads.
OOF predictions for training P arms. Frozen Ha/permutation evaluation changes
prediction E/I only, never re-trains downstream. Separately reuse ref8's
frozen200epoch GT_HEI scorer as plug-in diagnostic:replace GT with full/OOF
predicted E/I in OLDnormalization; no downstream training or selection there.
This distinguishes oracle-to-predicted input shift from OOF-trained task value.
Old H/Ha/GT scores remain reference, not a matched1500epoch comparison.
Re-estimate same-budget GT oracle for the NEW R denominator; report old-oracle
plug-in R separately. Compare old/new budgets transparently, do not use old
weak baseline to advertise a stronger newly fitted mediated model.

## Pre-outcome metrics and gates

Main primaryY axes3/6/7 (late contact, physicalheight failure, height-held
fraction), failureAUC and supported-stratum factual prognosis ranking.
Paired2000 held-out environment bootstrapsseed238; training-seed uncertainty
not included. Support inherits ref8: test166/31env/3motion counts100/45/21,
allarms≥3 and physicalfailure90/nonfailure76. OOFfolds≥20hold environments,
full outertest contrast design rank14. Crosshalf contrast rank is audited
separately; missing14-arm identification makes ONLY that descriptive contrast
UNCLEAR, not the adequately supported main comparison.

Gate A conditional prediction: Ha vs H I14 and joint normalized MSE gains≥5%,
both95% lower>0; Ha vs trainedshuffled I14 gain≥5%, and frozenHa testshuffle
I14 penalty≥5%. E separately reported, not required to exceed5% because ref8
showed I dominates. Shuffled labels must change≥60% of fit/test assignments.

Gate B action contrasts: on OUTERTEST estimate current-only adjusted factual
GT and factual P coefficients; separately compare SAMEH mean model candidate
arm-minus-zero vectors to adjusted GT. Report14×E12/I14 signed vectors, units,
sign agreement (GTnormalized axis magnitude≥.1), correlation, gain vs zero,
amplitude ratio/error. Main predicted Ha candidate I contrast correlation≥.50,
sign agreement≥.65 and MSE gain vs zero≥.10. Candidate-H contrasts must be
exactlyzero (no action slot). Descriptive cross-half predictors trained on
outertrain first5/last5waves each forecast outertest oppositehalf: report
E/I MSE and candidate-I vs marginalGT contrasts, do not replace main gate
with whichever half works. Different-state randomized contrast estimates
are not true individual counterfactual labels; shared-zero uncertainty remains.

Gate C task value: P_Ha OR Ha_P_Ha must improve directHa primary normalized
MSE≥5% with paired95% lower>0, physicalheight failure cannot worsen>2%,
P_Ha must improve P_H primary≥3%, and frozenactionshuffle must worsenP_Ha
primary≥2%. R=(L(H)−L(P_Ha))/(L(H)−L(GT_HEI)) point≥.25, denominatorGT gain
positive with bootstrap95% lower>0 and ratio valid≥95% draws. This compares
representation value with direct intended-action value, not reconstruction.
Report allcomparisons/ranking; alternative Ha_P_Ha is prespecified additional
consequence value, cannot ignoreaction-insensitive P_Ha to rescue the chain.

OverallPROMISING only adequate support+A+B+C, elseUNPROMISING for the fixed
contract if adequate implementation/support and observedgatesfail. If key
fit remains improving>5% in last200updates or support/implementation fails,
overallUNCLEAR with per-link failures preserved. No posttest epoch/signed/seed
rescue. EvenPROMISING is not identifiedmediation, same-state selection or
matched trainedpolicy Cm utility.

## Deferred evidence

Singlefit/sourceactorcohort; initialHcompression, sourcepolicy feedback and
noise maylimit predictability. ContactI usesaggregate netforce/proximity,
not pairedslip/friction. Onehotdoesnot permit arbitraryunseenactiongeneralization.
Finite sample bootstrap doesnot coverfitseedvariation. Future conditional
predictability improvements requirea new Decision and fixed protocol, not
an unbounded continuation. Validation and policyintegration remain deferred.


## Pre-main engineering support record

46 Task tests pass. Frozen ref8 split audited before ANY fit: OOFhold env
42/42/41, eachOOFsource arm≥17. Main outertest treatmentrank14. Each
opposite-half outertest has83rows; one lacksindexplus, adjusted14arm ranks
13/9 because nuisance support is too sparse. Retain eachhalf E/I prediction
MSE and sameH model candidate vectors, but set adjustedGT contrast metrics
NULL/UNCLEAR when rank<14; pseudo-inverse coefficients are not identified
individual effects. Do not add data/change split or silently use those
coefficients as evidence. Main gates rely on full outertest; crosshalf
contrast limitation is pre-run, not post-outcome rescue.
Pipeline additionally uses physical supervision and predictor compute; matched
162D downstream capacity does not match total pipeline compute/capacity.
