# Budgeted short physical information for actual policy learning

Decision Probe frozen before collection/training. Same ownP0checkpoint15ed218f,
threeairplane syntheticreferences/105task/nativephysics/originalspacing, no
official/sourceactor calls, oldfittedCm/V/Q/actors/labels neverused for learning.
Allthree motions remainin evaluation, including unsolvedmotion0.

PRELAUNCHamendment D-20261003-budget-layout-prelaunch replaces the initial384-
environment extra panel before any scientific collection. ALLpanels remain768,
same sceneorigins/layout as the evaluator. FRESH651/652short101controlticks,
653commonlabeled202ticks,654extra-budgetlabeled202ticks. Balanced4arms/motion,
zeroP0+duplicate andGaussian+antithetic, same raw12/nativeprojection/scales/
PRElift-8 heldcontroluntilstop+30. Independent placements/privateassignments,
no exactpairing. Bothshortpanels have no complete105label; never zero-filled.
Allcurrent/future at8ticks beforefirstcriterion; no model/actor atcollection.

Commonphysicalpool651/652/653=2304pairs; commonrewardpool653768; budgetrewardpool
653/6541536. FITSDK80mean/std from current/futureCOMMON653ONLY, floor.001/
clamp10, P0normfixed,current152. Normalizer correction is pre-fit; budgetQ
receives no short-panel input statistics. Target-only statistics follow below. Targetdelta=future[DYNAMIC]-current[DYNAMIC],
131unknown coordinates, targetmean/std physicalpoolONLYfloor.001. Knownfuture21
analytic/excluded fromphysicaltargets. Extra654neverdefines normalizers/auxdata
or feedsCm/off/coldQ. No oldsourcefit/tasklabels or pretrainedCm/Qmodels reused.

Four commoninitialized critics seed3651:encoder164->64ReLU->64ReLU, sigmoid
scalar taskhead, linear131physicalhead. Allweights common; physicalheadzeroinit.
Taskinput=current152+tanhactualraw12BOTHon/off andQcontrols. Physpass input
actualtanhraw forCm, zeros12foroff. Cold/budget same physicalpass shape butloss
weight0. Jointloss measuredterminal105BCE+.05normalizedphysicalMSE(Cm/off),
BCEonlycold/budget; NOgenerated successor, Vcomposition, fictitiousreward or
model-error intrinsicreward. ALLweights trainable forjointfit,1500Adam3e-4/
weight_decay1e-4/gradclip10each. CommonsameprivateCPU3654rewardbatch256(on/off/cold);
budgetsamealgorithm/private3654with1536pool. SharedphysCPU3652batch256acrossfour.
Eachcritic updates onceperstep:6000actualoptimizersteps TOTAL;3000aux-bearing
updates are a SUBSETof6000, never doublecounted asadditionalphysicsupdates.
No modelselection or intermediateforecastgate.

Freezecritics beforeactors. Fourcurrent152->64ReLU->64ReLU->12tanhactors atEXACT
common3555init,1000Adam3e-4/clip10steps EACH(4000actual), objective -taskQ+
.05mean(raw^2). Commonprivate3556batch256on/off/cold from653; budgetsamealgorithm/
3556from653+654. Policyfit nevercallsphysicaldecoder/predictedfuture. Finalactors
only deployed, exactnativecouplings/PD andsameheldoption timings. Retain actual
critic1500/actor500/1000weights, optimizer/privategeneratorstates, no overwrites.

CostperCm/off:1536*101physical+768*202full=310272env-controlticks. BudgetQ:
1536*202=310272exact. ColdQhas sameavailable shortdata but noauxobjective;
report ignoreddata/lesscompute, not equalpretrainingusage. SourceP0costseparate,
oldhistory notfreepretraining. Actualaggregate trainingcollection=465408ticks,
evaluation=620544ticks, total1085952. Nativecontrolfrequency/simulationdt/substeps
recorded; scalarenv-controlticks not mislabeled as solversteps or walltime saving.

Fresh655/656evalP0/Cm/off/coldQ;657/658evalP0/Cm/off/budgetQ.768env202ticks each,
64perarm/motion/seed. Cm/off/P0pooled768perarm; cold/budget384each inblocks.
PROMISINGrequires: Cm>=5ppoffandP0pooled; CmblockA>=5ppcoldQ; CmblockB>=5pp
budgetQ; eachseedCmnoninferiorALLpresentcontrols; motion1Cmno worseP0by>5pp
pooled. All mandatory, no subset/checkpoint/gate rescue. Allsuccesscounts by
motion/seed/arm and exactfull105meshthresholds retained. Oneoptimizationseed,
no formalValidation/generalization/hardware or distinctive-methodclaim.

Fullnative/P0/PD/mesh/privateassignment/antithetic/rawtiming audits EVERYpanel,
shortfullobservedprefix audit with explicitlyNOterminaltasklabels; inputSDK/PRE
current/future/knownplan/normalization/labels independently reconstructed; all
finaltaskandphysicalheads/actors compared againstNumPy<=2e-5; commoninit/actual
finitegradients/changedweights/counts/optimizercheckpoints retained, noindependent
optimizerreplayclaimed. GPUtraining/native, CPUfile/geometry/audits. Whole2400s/
3GiB, onefreshidleGPUphaseguard, source675/newcodeSHAguard, ownedprocessesonly.
No seed, width, coefficient, responsehorizon, truncationfraction or labelbudget
scan afterfailure. GoalACTIVE/journalstandardunchanged.

Pre-fit normalizer correction: D-20261003-budget-normalizer-correction. First
short651collector retained; r1stopped under code guard, r2inherits exactly651
under SHA and independently audits it. No valid scientific collection repeated.

Outcome: r2 COMPLETED/UNPROMISING. Cm383/off399 per768, blockA Cm200/coldQ189
per384, blockB Cm183/equal-budgetQ220 per384. Four of six frozen gates fail.
All eight native and complete training audits pass. r1protected-code stop
and inherited valid651panel are preserved; no repeated collection or fit.
[Full result](../../archive/2026-10-04-root-research/research/20261003-budgeted-physical-critic-results.md).
