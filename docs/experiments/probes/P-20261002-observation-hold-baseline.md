# P-20261002-observation-hold-baseline

Decision/Blocker. One fixed scratch BC initializer, collection510/evaluation511/512,
trainingseed734. Follow D-20261002-observation-hold-baseline exactly. Existing self-
trained sourceactor286 is loaded ONLY to construct the native player; no actorcall
or source network/RMS weight enters the BC policy. Only teacher trajectory data
fromnew510 determine normalization/training; no test or outcome selection.

96env/seed,32per each of3motions,202nativephysicalsteps, initialframe0. Teacher
runs fixed native reference goals and.30rad ramp from d15ef0f, clipped in time
at firstplateau_stop. It holds finaltarget thereafter; no release test in this
Probe. Policy independently maps the explicit70current/planned features to18
normalizedactions, six exact hardware-nullcommands0. Fixedreference timing is
provided to both; no futureactualstate, geometricoracle actionselection or fallback.

Retain fullcontexts/actions/root/q/forces/progress/clearance/targets andinitial
states, nativeactor/gravity/massmetadata. Teacheronly saves target mapping errors;
policy records actualPDtargets and model/statistics fingerprints. All inputs,
collection/context/checkpoint tensors finite. Policyweights andscales unchanged
duringtest. No earlytermination/kappa, no objectstate assignments afterreset.

GPUfit2000Adamupdates,512batch,.001rate,gradclip10,512/256/128ReLU/Tanh,scratchseed734.
FIT-onlymean/std(floor.001),normalizedclip10; actionlossfixedcoordinate scales as
Decision Memo. Finalonly, no selectedseed/checkpoint or heldloss criterion.

Primaryphysical105 and gate knownmotion1>=50%pooled,>=25%eachseed as Memo. All
three motions retained; forceproxy75 secondary without success relabeling. CPU
independent world-up fullmesh/savedlabel audit, no modelcompute. BC success cannot
replace matched Cm-on/off policytraining utility or formalValidation. Stop exact
recipe onfailedgate,drift/nonfinite/coverage/PD/schema/checkpoint/budget failure.
One freshly admittedfreeGPU per phase,<=1800s/1GiB. Preserve completed phases and
resourceadmission failures; never interrupt unknown GPUprocesses.
