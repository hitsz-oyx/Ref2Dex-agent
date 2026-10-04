# State-anchored direct-flow transport improves prediction, misses matched gate

P-20261003-state-anchored-transport-r2 at171af9c COMPLETED/UNPROMISING.
The changed objective/convexdecoder/proximity prior/frozenstate package
improves the previously failed learned transport, but misses one of five
prospective gates. Keep the positive numerical observation and failed gate.

| Predictor | Equal-episode EPE, mm | Near EPE, mm | Far EPE, mm |
| --- | ---: | ---: | ---: |
| Full hand-correction Cm |0.395164593|2.381678154|0.093955756|
| Matched learned state-correction |0.434032117|2.628625966|0.093955756|
| TRAIN-target-shuffled hand-correction |0.493532713|3.040401073|0.093955756|
| Shared frozenstate B |0.448095692|2.722162720|0.093955756|
| Previous-point-motion persistence |0.629367707|2.823930168|0.519530074|

Full gains:11.812%overfrozenB,8.954988%overmatchedstatecorrection,
19.931%overshuffled and37.212%overpersistence. Neargain9.394559%passes5%.
Required whole-panel10%statecontrol gain FAILS: threshold0.390628906mm,
actual0.395164593,0.004535687mmabove. Fourgatespass, labelUNPROMISING by
fixedrule; no rounded-up success, gatechanges or subgrouprescue.

Samecorrected655 previouslyviewed panel,384TRAIN/384HELDepisodes,16fixedticks
each/6144windows each; TRAINnear2296/HELDnear2124. HELDnear256episodes/far384;
subsetmean availableepisodes equally, not windowweighted to primary.
Motion0allnewmodels0.0178355mmidenticalB (noeligiblecorrection), motion1
full0.647965vsstate0.643671, motion2full0.519693vsstate0.640590. This is not
tasksuccess, freshseedgeneralization or measuredpolicy improvement.

All3newheads23808parameters/commoninit4301/shared4302schedule,1500AdamW
updates each/4500newtotal,48000windowdraws/672000correlatedheadtoken rows
perarm. Shared inherited frozenstate23874parameters/1500oldupdates reported
separately. Statecontrol receives SAME99commoncausalgeometry/state/command/
aggregatehandmotion information; balanced13zero/inertia correctioncopies,
same14headcalls. TRAINshuffle4303onlyworldflowoutcomes, sharedfrozenB retains
its previous correctlytrained stateknowledge. Noheldlabels/fine-tuning/
earlystop/checkpointselection or newnativephysics/actortraining.

Outsideunsignedcurrent2cm proximity, effectivecorrectionweights zero and
outputequalsBbyteforbyte. This is an engineeredproximity gate, not a learned
far improvement or truecontact classification. Nearconvexmixtures combine
B+13causalhand endpointflows; they EXPAND previous one-segmentfamily and
need not define a rigidpose or contact-feasible motion. Multiple designparts
changed together, so improvement cannot be uniquely attributed to directloss.

## Actual implementation and complete retained histories

PrefitCPU all3arm4row3update independentmanualgradient/AdamWsmoke passes;
sharedunidentifiable softmaxoutputbias removed BEFOREscientificfit instead
of relaxing parameter tolerance. Prefitreadonlyreview fixes NumPy body/
translation indexaxis order.8rowcurrentKDTree smoke verifies causalproximity.

r1f6ec05f FAILED before anynewheadupdates/weights: newbaseline normalized
onCPU whereas originalfrozenstate usesGPU, causing5.794512e-9m drift despite
same6144winners. Strict1e-12mcheck stops before constructing/trainingheads.
r1wall21.673869s/20722482bytes, allrawbase/log/manifest preserved.171af9c
restores EXACToriginalGPU arithmetic, not different statistics or model,
and r2passes frozenbaseequality; no successfulfit repeated or gate changed.

r2wall80.614550s/55035294bytes: CPUsmoke5.591267s, GPU6fit28.915713s,
CPUaudit33.862356s. Combined102.288419s/75757776newbytes,4500actualnewupdates,
0newnative ticks. Parent656434/children656557/656836/657655 absent;
r1ownedPIDsalso absent. All1500protectedsource inputs unchanged.

Independent audit: all12288frozenbase predictions/coefficients/scores and
training-onlynormalizer; inheritedraw/transport geometry audits bySHA;
768firstwindow perepisode TRAINandHELD current10135surface/64queryKDTree
checks (newfulltransport geometry not rebuilt); ALLheldnewneural logits/
weights/decodedflows/parent/per-episode/motion/arm/near/far/baselines/gates.
BaseNumPymax7.15256e-7, distancemax4.02785e-7m, headmax2.62260e-6,
metric0mm. First3updates perarm NumPymanualgradient+AdamW replay9total,
parametermax2.38675e-8/loss7.14134e-8; remaining4491updates not replayed.
Frozenstate/init/schedule/optimizer accounting allmatch. Independentreadonly
review reaggregates alloutputs exactly, verifies allsourceSHA andbyteidentical
frozenB/oldstateoutputs/faroutputs; no concrete implementation blocker.

## Research consequence

Close this bounded local head/transport correction family without additional
seed/steps/width/radius/mixture/component sweeps. A positive8.95%observation
does not erase the failed matched qualification or authorize policytraining.
Return to the higher-level physicaltransition representation rather than
another localscorer/loss variant. Read-onlysource check notes that the older
continuouscritic6Dauxiliary target encodes translation and linearvelocity,
not rotation/angularvelocity
(`src/task/CmResidual/continuous_critic_cm.py:109`); current64point-flow
DOESinclude rotation.
Savedcontact2flags threshold SDKnetforces at0.1N
(`src/task/CmResidual/physical_value_live.py:9`); they do not supply contactpair,
normal/frictioncone or torque labels. These are inspected codefacts, not
proven causes. A prospective rotation/clearance label-adequacy audit can
decide whether a rigidpose/rotationalresponse direction merits investment,
without repeating older measuredgeometry/force-aware barrier recipes.
Goal ACTIVE, Cm policy-training utility NOTDEMONSTRATED, journal NOTREADY.
