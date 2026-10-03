# State-anchored, proximity-gated direct-flow transport

Decision Probe. Previous goal turn is progress: coefficient/confidence learned
gate fails despite positive rigid capacity; saved-coefficient oracle selection
also loses to matched learnedstate. Change the objective/decoder instead of
rescuing old epochs/width/density/score regression. Mission unchanged.

Question: can direct realized point-flow supervision of a state-anchored hand
correction recover matched information advantage? Cheapest test reuses exact
corrected655384TRAIN/384HELDepisodes,16ticks each,6144windows pergroup,
64canonicalairplane queries and disjoint-train actuator, no newphysics.
Previously inspected heldpanel: diagnostic only, not formalValidation.

Freeze the previous r1 learnedstate-only23874parameter checkpoint/normalizer,
including its coefficients ANDconfidence selector. Compute its TRAINandHELD
predictions without outcomes, called B. Same inherited TRAIN-only120column
normalizer used for allnewheads. No baseline optimizer updates.

Correction head120->128ReLU->64ReLU->1,23808trainable parameters, commoninit
4301; lastweights zero, finalsharedbias omitted because softmax is invariant
to its additive shift (caught in prefit tiny-gradient smoke).14tokens:
Btoken plus13correctiontokens. Btoken
copies originalzero-endpoint rawfeatures with its finalkindonehot set000;
relativepose/twist remainzero,99commonfeatures identical. Full/shuffle13hand
tokens use originalcausal features/endpoints IDs2:15. State-onlycorrection
uses originalzero/inertia IDs[0,1]*6+[0],13calls. All99commonjoint/state/
geometry/command/aggregatehandmotion info SAME; control is not action-blind.

Softmax rawlogits plus fixedlogpriors: B=log99, fullhands=-log13 each,
statecopies=-log(2*count[id]) (7zero/6inertia). Totalinitial Bmass.99 and
correctionmass.01, balancedstate kindmass.005each. Decoder
P=B+sum_{j=1..13}p_j*(E_j-B). At inference use same differentiable mixture,
not hardwinner. This EXPANDS previous single-segment family to convex
transport means; need not be rigid pose or contact-feasible. Positive result
would not isolate loss from representation/anchoring/proximity changes.

Fixed gate: current64query min unsigned distance to10135fixedsampled hand
surface<.02m. Beyond gate P=B EXACTLY, all effectivecorrectionweights0.
Radius inherited from earlierdiagnostic, no radius tuning. Training gate from
auditedcalibrationr1TRAINfeatures column15 (r2completed independentrecovery);
HELDgate from execution stationaryfeatures. This column uses currentgeometry
only, not predictedfuturehandflow; independent768row KDTree reconstruction.
Near points indicate proximity, not attributed physical contact. All models
use samegate. This guarantees far preservation, not learned benefit.

Threearms full,state_only,shuffled; shuffle4303permutes TRAINworldpointflow
targets only, leaves fields/currentgate/B fixed. Shared frozenstate knowledge
remains in shuffledcontrol, so it tests incremental correction information.
Sharedschedule4302:1500x32uniformTRAINwindow draws, AdamW3e-4/weightdecay1e-4,
gradnormclip10.4500newupdates total,48000windows/672000headtoken calls perarm,
correlatedderived rows not independentdata. TRAINloss mean pointEuclidean
error inmm, sqrt(sum_xyz(residual_mm^2)+1e-12); differentiates through
softmax and actual mixture, no oraclecoef/score labels. Farrows zeroheadgrad
by fixedgate; sameuniformbatchschedule allarms, no success/contactfilter.
No earlystop/checkpointselection/additional loss/data/seed/steps scan.
Save allcheckpoints/optimizers/commoninit/schedules/losses/first3states,
frozenbaseTRAIN/HELD predictions and allheldlogits/effectiveweights/flows.

Fixed gates on equal-episode EPE/mm ALL6144heldwindows:
full<=.9frozenB; full<=.9learnedstate_onlycorrection;
full<=.95shuffled; full<=.9rawpersistence;
nearfull<=.95nearlearnedstate_onlycorrection.
PROMISING all5; UNCLEAR iff frozenB+statecontrol gates pass but notall;
otherwise UNPROMISING. Outsidegate equality is engineering invariant, not a
scientific benefitgate. Report fixedmotion/arm/near/far without subgrouprescue.

CPU4row3update engineering smoke: independentNumPysoftmax/decoder/manual
gradients+AdamW, zero-fargate gradient, baselinepreservation and outcome
inputisolation. Main fitting/baseline batchinference on oneidleGPU. CPUaudit:
inheritedgeometry/sourcehash contracts, ALL12288base outputs independently
NumPy reconstruct; ALL gates/normalizer/shared99/order;768firstwindow per
train/heldepisode currentfullsurfaceKDTree distance/gate checks. ALLheldnew
neural outputs/mixture/metrics;3updates each independentNumPy replay9total,
remaining4491not replayed. Forwardtol2e-4, fieldtol2e-6m, metric1e-9mm,
earlyparam5e-5/loss2e-5. Exactfrozencheckpoint and schedule/init verified.

OneidleGPU(mainprefer6),900s/512MiBnewoutput,300GBownedtotal/deadline3Oct23:59
Beijing. CPU onlytinyengineering/independentstatistical audit. Uniqueoutput,
committedcode/card/ancestor+newsourceSHA; stopownedchild only on drift/budget/
contention/nonfinite/auditfailure. Preserve anyfailure/successfulfits.
Positivelearnedgate permits separatefresh randomizedcorrected-native
qualification beforeactortraining; failure returns to higher-level
representation/utility, no endless local transport/refinement. No final
scientific Cm utility/novelty or journalclaim. GoalACTIVE, NOTREADY.
