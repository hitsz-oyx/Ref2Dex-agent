# Rotation contribution to actual full-mesh clearance decisions

Decision Probe. Previous goalturn is progress: matchedstate-anchored transport
4/5gatespass but8.955%fails10%, archived; localheadfamilyclosed. Read-only
codefacts: old6Daux target translation+linearvelocity, no futureorientation;
current64point-flowALREADYincludesrotation. Do not claim omission caused any
learnerfailure. This test decides whether explicitrotationalresponse deserves
a separatelydesigned representation screen or should be dropped.

Reuse corrected655 fixed768episodes/all202native transitions,155136total,
384train/384held inheritedseed4001 wholeepisode split within3motions×4arms.
Use ALL202steps ratherthan previoussparse16ticks because taskclearance can
fail on any of105hold/drop ticks. No newphysics/models/optimizer/intervention.
Fixedairplane25002meshvertices and actualthintableupperplane/rotation.
GPUfloat64fullgeometry over203poses×768; CPUindependentSciPyquaternion/NumPy
fullmesh audit ofALL203poses, noNN. Infiniteupperplane is same conservative
metric as task, not actualcollision or forceclosure.

For each actual PRE/POST objectpose, compute Ccur/Cnext. Translationoracle
withpersistedcurrentorientation: Ct=Ccur+worlddz. Rotationonly hybrid:
Cr=Cnext-worlddz. Actualtableupverifiedworldz. Therefore deltaC=dz+dr where
dr=Cnext-Ccur-dz, exactalgebra for this fixedupperplane; unlike nonlinear
causalmechanisms this separablemetric has no translation/rotationinteraction.
TinyCPUmesh smoke must also explicitlyconstruct hybridposes to verify this
identity/tablethinaxis handling (assetthinaxis1, not assumedlocalz).
Bothactualtranslation and rotation use futureoutcomes: diagnosticoracle,
not an identified counterfactualphysics transition or deployment prediction.

Primary eligible windows use CURRENTonly: rootrise>=.03m relativeinitial,
currentfullmeshclearance in[.015,.025]m inclusive. No futurecontact/success/
outcome eligibility, no selection of phases/arms/motions. Binary decisions
Cnext>=.02 vs Ct>=.02. Count a resolvedflip only if decisions differ AND
both values more than1e-6m fromthreshold. Unresolvedrows remain in denominator
and are reported separately (no favourable exclusion). Equalepisode mean
fliprate over eligibleepisodes. Rotationmagnitude weighted95thpercentile of
abs(dr), every eligibleepisode totalweight1, inverseCDFdefinition.

Fixed HELDgates: >=128eligiblewindows AND>=32eligibleepisodes;
equalepisode resolvedfliprate>=.10; weighted95thabsrotation>=.002m.
PROMISING ifall3; UNCLEAR ifeligibilityminimumfails; elseUNPROMISING.
TRAIN/ALL/per-motion/arm/phase summaries diagnostic only, no subgrouprescue.
Also report entireallwindowdelta magnitudes/geodesicorientation changes,
full105actualtask and persteptranslationoracle success discordance using
identical actualfuture rootrise>=.03 plus75plateau+30drop masks. The oracle
uses truecurrentorientation at EACHstep, not a physicallyintegrated no-
rotationtrajectory. These ancillary taskcounts do not override primarygates.

Verify ALL155136rows PREhistory/POSTtrace/timestamps/noresets, split, old6D
target and contexttranslation/velocity; reconstructtarget inoriginalfloat32
beforeCPUfloat64diagnostic. Fullgeometry matches retainedtraceclearance
within2e-6m. IndependentCPUfull155904pose clearance max1e-10m againstGPU;
ALLalgebra/derivedfields/masks/equalepisode/weightedquantiles/taskcounts/gates
recomputed, sourcehashes fixed. No contactlabel reinterpretation.

OnefreshidleGPU preferably6, CPUtinyengineering/audit,<=300s/64MiBnewoutput,
300GBtotal/deadline3Oct23:59Beijing. Immutable code/card/source/raw/mesh SHA,
uniqueoutput, stopownedchild only on drift/budget/contention/auditfailure.
No newexternalauthorization or coremission/claimchange. Success permits
separatelyspecified SE3response learnability, not policyfit; failure drops
this candidate and returns to action-decision-level route review. Any
learner still needs matched controls/freshqualification/policyutility and
novelty research. GoalACTIVE,journalNOTREADY.
