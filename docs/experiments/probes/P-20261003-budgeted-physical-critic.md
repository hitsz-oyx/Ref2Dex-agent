# Budgeted short physical information for actual policy learning

Decision Probe frozen before collection/training. Same ownP0checkpoint15ed218f,
threeairplane syntheticreferences/105task/nativephysics/originalspacing, no
official/sourceactor calls, oldfittedCm/V/Q/actors/labels neverused for learning.
Allthree motions remainin evaluation, including unsolvedmotion0.

FRESH651short768env101nativecontrolticks,652commonlabeled768env202ticks,
653extra-budgetlabeled384env202ticks. Balanced4arms/motion, zeroP0+duplicate and
Gaussian+antithetic, same raw12/nativeprojection/scales/latePRElift-8 heldcontrol
untilstop+30. IndependentXYplacements/privateassignments, no exactpairing assumed.
384layout differs in origins because nativeenvironment count changes; same
spacing/materials/physics, explicit, not claimed exactcoldstates. Short651ends
before complete105judgment, labels unavailable rather than zero. Allobserved
current/future at8ticks beforefirstcriterion. No model/actor at collection.

Commonphysicalpool=651+652(1536current/future pairs), commonrewardpool652(768),
budgetrewardpool652+653(1152). FITSDK80mean/std from current/futurephysicalpool
ONLY, floor.001/clamp10, P0normfixed, current152. Physicaltargets=future[DYNAMIC]
minuscurrent[DYNAMIC],131unknown coordinates; normalize targetdelta mean/std
physicalpoolONLYfloor.001. Knownfuture21analytic excluded fromphysicaltargets.
Extra653neverdefines normalizers/auxdata or feeds Cm/off/coldQ.

Four commoninitialized critics seed3651:encoder164->64ReLU->64ReLU, sigmoid
scalar taskhead, linear131physicalhead. Allweights common; physicalheadzeroinit.
Taskinput=current152+tanhactualraw12BOTHon/off andQcontrols. Physpass input
actualtanhraw forCm, zeros12foroff. Cold/budget same physicalpass shape butloss
weight0. Jointloss measuredterminal105BCE+.05normalizedphysicalMSE(Cm/off),
BCEonlycold/budget; NOgenerated successor, Vcomposition, fictitiousreward or
model-error intrinsicreward. ALLweights trainable forjointfit,1500Adam3e-4/
weight_decay1e-4/gradclip10each. CommonsameprivateCPU3654rewardbatch256(on/off/cold);
budgetsamealgorithm/private3654with1152pool. SharedphysCPU3652batch256acrossfour.
Eachcritic updates onceperstep:6000actualoptimizersteps TOTAL;3000aux-bearing
updates are a SUBSETof6000, never doublecounted asadditionalphysicsupdates.
No modelselection or intermediateforecastgate.

Freezecritics beforeactors. Fourcurrent152->64ReLU->64ReLU->12tanhactors atEXACT
common3555init,1000Adam3e-4/clip10steps EACH(4000actual), objective -taskQ+
.05mean(raw^2). Commonprivate3556batch256on/off/cold from652; budgetsamealgorithm/
3556from652+653. Policyfit nevercallsphysicaldecoder/predictedfuture. Finalactors
only deployed, exactnativecouplings/PD andsameheldoption timings. Retain actual
critic1500/actor500/1000weights, optimizer/privategeneratorstates, no overwrites.

CostperCm/off:768*101physical+768*202full=232704env-controlticks. BudgetQ:
1152*202=232704exact. ColdQhas sameavailable shortdata but noauxobjective;
report its ignoreddata/lesscompute, not an equalpretrainingusage claim. SourceP0
pretraining cost separate, oldmodelhistory nofreepretraining. Nativecontrolfreq/
substeps and episodecounts recorded; scalarenv-controlticks not mislabeled as
solverintegrationsteps or a walltime saving. Report complete data/NN/wallcosts.

Fresh654/655evalP0/Cm/off/coldQ;656/657evalP0/Cm/off/budgetQ.768env202ticks each,
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
