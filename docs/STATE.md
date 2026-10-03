# Independent contact-response research state

Updated 3 October 2026. User instruction: continue the previous research route.
Goal ACTIVE; journal readiness NOT READY. State-anchored transport and full-mesh rotation/clearance completed UNPROMISING; contrast-acquisition completed UNPROMISING; manuscriptv12 sources prepared.

## Workspace and task

Writable standalone clone `/tmp/Ref2Dex-contact-response-continuation`, branch
`agent/cm-active-acquisition` (retains rigid-coupling, surface-prior and response-actor branches/history). Original user tree and old own contact-response tree
are read-only. No remote push, broker or unknown process control. User explicitly
authorized one read-only implementation-review subagent for this review only.
Native Python3.8 graspenv / IsaacGym / PyTorch2.4.1. Idle GPU admission for
model fitting/native physics, CPU independent file/geometry audits. One-GPU
ordinary Probe, 300GB total owned storage; deadline3October23:59Beijing.

Same airplane, three synthetic references. Success requires root rise>=30mm
and full25002vertex mesh tabletop clearance>=20mm on ALL105controlticks
(75synthetic plateau+30drop checking).202ticks/reference stops[162,140,152],
lift starts[63,44,55]. Control1/30s, simulation1/60s, controlFrequencyInv2,
substeps4. SDK body origins/net forces are not attributed contact/force closure.
Privileged current simulator state and public reference plan remain explicit.
No user-track six-expert/twelve-motion evidence is transferred.

## North-star

| Requirement | Current independent status |
| --- | --- |
| Own self-trained manipulation | PARTIAL: P0 motion1; task-Q actors motion2 up to117/128; motion0 remains unsolved |
| Short physical information | Actual8tick successor oracle diagnostic PROMISING; oracle unavailable at deployment |
| Matched Cm policy-training utility | NOT DEMONSTRATED; fixed actual learning recipes fail complete gates |
| Distinctive method | Unestablished; generic auxiliary predictive features are prior art |
| Validation/generalization/hardware | Absent; manuscriptv11 is a historical draft |

## Latest actual result

P-20261003-contrast-acquisition-r1 COMPLETED/UNPROMISING: same512initial+
512extra labels,4800newupdates, contrastrelativecausalriskvsuniform+.362847mm²,
vsabsolute-1.923996,vszero-2.988855;3/6gatespass, no nativefollowupfromfailure.
AllnewauditsPASS;22.104057s/9.92MB,0native/actor, inputstable/ownedPIDsgone.
[Result](research/20261003-contrast-acquisition-results.md).

P-20261003-rotational-clearance-adequacy-r1 COMPLETED/UNPROMISING at13cbd0b.
Held144eligiblewindows/113episodes: equalepisode flips1.9174%fails10%;
weighted95throtation7.0397mm passes2mm, coveragepasses. Full105actual and
persteptrue-translation diagnostics agree on ALL768episode labels,325successes
(held154/384). Not a deployable/causal no-rotation trajectory; currentorientation
is refreshed at EACH step and futuretranslation is privileged. Motion0 has
noeligiblewindows. Current64pointflow alreadyhasrotation; do not attribute
learnerfailure to old6Dtarget omission. ALL155904meshposes/155136transitions
independently audited, geometry1.11e-16m/native5.61e-8m/metric2.09e-17max,
source chronology/context/labels/split/full105task/gates PASS.37.967316s,
19509015bytes,0NN/nativeupdates, GPU6geometry/CPUaudit,inputs unchanged,
ownedPIDs absent. No subagent audit claimed for thisrun. Close rotation-only
remedy on this panel; next action-decision-level route review before newfit.
[Result](research/20261003-rotational-clearance-results.md).


P-20261003-state-anchored-transport-r2 COMPLETED/UNPROMISING at171af9c:
full0.395165mm vs matchedstatecorrection0.434032,shuffle0.493533,frozenstate
0.448096,persistence0.629368.4/5gatespass; state10%gain fails at8.954988%,
near9.395%passes5%. FaroutputbyteidenticalBbyhardcurrent2cmproximity gate,
notlearnedfarbenefit.3x1500newupdates/23808params/sharedinit+schedule,4500new
and1500sharedstateinherited separately;0newnative/actor. r1FAILEDbefore0new
updates fromCPUinstead oforiginalGPUbaseline normalization;171af9c restores
originalarithmetic, samegates/1e-12tol, frozenBbyteidenticaloldoutputs. Both
historiesretained, combined102.288419s/75757776bytes/ownedPIDsabsent.
All12288base/768KDTreecurrentgeometry/ALLheldnewneural+metrics/9manualAdamW
updates auditpass;remaining4491notreplayed. Independentreview repeats all
metrics/1500sourceSHA/commoninit/schedules/frozenBexactness withnoblocker.
Smallpositiveobservation preserved withoutpromotingfailedqualification.
[Result](research/20261003-state-anchored-transport-results.md).

P-20261003-rigid-coupling-learnability-r1 COMPLETED/UNPROMISING at348302a:
full0.558129mm vs matched learnedstate0.448096,shuffle0.731395,persistence
0.629368. Persistence/shuffle gates pass; state10% and near5% gates fail
(near gain4.614%). Same99common causalstate/geometry/command info in state
control, not action-blind.3x1500updates/23874params/commoninit+schedule,
4500actualtotal,244.564910s/286423899bytes,0newnativephysics. All12288rawrows,
768geometry,184320TRAINcertificates/ALLheldNumPy+metrics/9earlyAdamWsteps
pass; remaining4491not replayed. Independent review confirms all metrics,
labels, schedules and causal isolation. Protectedinputs unchanged/PIDs absent.
Separate c84f7bd saved-array diagnosis COMPLETED/UNPROMISING,3.441455sCPU,
0updates: even true-error selector with fixedlearnedcoeff0.486083 loses to
learnedstate0.448096. Coefficient and selection gaps are path-dependent
oracle diagnostics, not independent causal contributions. Close bounded
coefficient/confidence recipe; retain positive geometrycapacity/stateanchor.
[Result](research/20261003-rigid-coupling-learnability-results.md).

P-20261003-rigid-transport-capacity-r1 COMPLETED/PROMISING at69cc802 for
oraclecapacity only. Causaltransport0.294084mm vs equally outcome-fitted
state-only0.368403 (20.173%gain), near1.879200vs2.363974 (20.507%),measured
hand0.294313; all3gatespass.15endpoint/scalar segments,not fullcontacthull;
oraclecoefficients/winners use actualfutureoutcome,not deployable/contact/
policy evidence.6144raw/statefields/joints+384geometry+196608convexcertificates
and12288SciPysolves/allmetrics audit pass,protectedinputs unchanged,PIDs
absent.56.656s/353030828bytes,0newNNupdates/physics. SubsequentSAMEfamily
causal coefficient learning fails; state-anchored expansion has smallgain
but also misses matchedgate. Noactor/native until a newusefullearnedgate.
[Result](research/20261003-rigid-transport-capacity-results.md).

P-20261003-surface-calibration-r2 COMPLETED/UNPROMISING at7709263; r1d47e1fe
completed four fits then FAILED on independent audit NumPy index axis order.
r2onlyCPUaudit inherits all data/checkpoints bySHA,0newupdates;4800actual
updates total. Normal pretrained0.665444mm vs persistence0.629373,
scratch0.633220,shuffled0.618116,hand-flow-removed0.790197. Onlyhandflow gate
passes; all raw12288rows/train6144joints/384newgeometry/fullheldNumPy+metrics
audit pass, encoders unchanged and12earlyoptimizersteps replayed.
Combined77.598s/74346217unique bytes, protected inputs unchanged,PIDs absent.
Close this bounded frozen-feature transfer family; next rigid transport/
coupling representation-capacity screen BEFORE newfit/collection.
Same-seed previously viewedholdout, no policy utility or genericCm rejection.
[Result](research/20261003-surface-calibration-results.md).

P-20261003-surface-execution-input-r1 COMPLETED/UNCLEAR at169db6c.
Reused corrected655data,384train/384held wholeepisodes,6144windows each.
Causal54coefficient actuator handEPE0.899870mm vs velocity2.600907/stationary
9.710036, execution gate passes. FrozenMANO/Inspire7168oracle objectEPE
7.301355/6.725405mm vs persistence0.629373; causal7.346440/6.753861, all four
prior gates fail. Near-hand oracle3.222090/3.160013 vs persistence2.823930.
No subgroup rescue or solephysics attribution. Preserve actuator, close
THISfrozen directtransfer combination; corrected-data feature/output
calibration is next. All12288rawrows/fulljoint+NNaudits and384geometry checks
pass, SDKbody originmax0.066mm;195.424s/249839074bytes, inputs unchanged,
own PIDs absent; zero new physics/optimizer steps. Same-seed reusedepisode
holdout and measuredq FK labels only, no policy utility.
[Result](research/20261003-surface-execution-input-results.md).

P-20261003-cm-granularity-r1 COMPLETED/UNPROMISING at3c19e61. Eight matched
64/256context x4neighbor mean/detail fits; eachhand same2048windows/50objects,
same23107params/commoninit/1500updates and64supervised target points. Own-hand
EPE mean64/mean256/detail64/detail256: MANO3.456930/3.457202/3.465854/3.464157mm;
Inspire4.392716/4.390056/4.399568/4.397399mm. All six10%gates fail; changes<0.3%.
All6982raw rows and independent geometry/neural/metric/schedule audits pass;
12000updates,156.272s/371854792bytes, protected inputs unchanged, own PIDs absent.
Stop exact matrix; no useful fine-context/neighbor signal at fixed budget.
Still offline future-hand input/globalmean Cm, not all fine representations
or policy utility. [Result](research/20261003-cm-granularity-results.md).

P-20261003-cm-scale-cross-hand-r1 COMPLETED/UNPROMISING. Shared22feature
surface-motion prior,19651parameters, nested512/2048/7168windows perhand,
50train objects perhand. MANOheld EPE3.869/3.601/3.559mm; Inspireheld
4.743/4.417/4.432mm. Gains8.015%/6.560%fail10%gates. Fixed256window
Inspireadaptation: MANOprior4.727 vs scratch4.830/shuffled4.840mm, gains
2.143%/2.330%fail10%/5%. Frozen MANOzero-shot4.412mm, adaptation worsens
7.141%. All complete gates fail; no scale/main-cause or pure hand-effect claim.
Realized future hand movement is offline input; legacy-source physics, no
new corrected-native policy evidence. All18161source rows and geometry/neural/
schedule/gate audits pass;15600updates,458.194s/273459235bytes, inputs unchanged,
owned PIDs absent. [Result](research/20261003-cm-scale-cross-hand-results.md).

Current-geometry granularity audit completed at927da81: MANO2048/Inspire10135
raw hand points, but this prior uses64object queries with4averaged hand
neighbors each. Full18161window audit finds zero selected near-hand queries
in0.849%MANOeval/1.584%Inspireeval; all groups below prospective10%gate,
UNCLEAR. Unsigned2cm proximity is not physical contact or complete local
patch coverage. Subsequent fixed64/256context and mean/detail comparison failed allgates;
no new model fit and no change to the active causal-input blocker.
[Result](research/20261003-surface-granularity-results.md).

Independent implementation review confirmed a native collision-filter bug:
body names were indexed by shape number despite25bodies/13shapes. Five hand
shape filters were wrong. SDK ownership-based correction passes the actual
method regression,5mismatches before and0after, same URDF. No confirmed neural
label/frame/gradient bug found in inspected paths. Historical counts below
remain legacy-physics observations; previous audits did not cover filters.
One fixed655panel completed: P0/Cm/off/coldQ before[65,96,101,95]/192,
after[61,81,92,91]/192; Cm -7.813pp passes the absolute sensitivity gate.
All motion0remain0. PROMISING is physics sensitivity, not improvement or Cm
benefit. All768filters and independent actor/PD/fullmesh audits pass;70.962s,
250472932bytes, inputs unchanged and own PIDs absent. One seed, no exact native
pairing or retraining. [Review](research/20261003-implementation-review.md),
[result](research/20261003-inspire-filter-impact-results.md).

P-20261003-budgeted-physical-critic-r2 COMPLETED/UNPROMISING. Frozen design22e47f5,
scene-size amendment42e2cbd, implementation48f8d2b, pre-fit normalization
correction09e76ac. All panels768; fresh651/652short101ticks,653common202,
654extra202,655--658actual trained actor evaluation. All four methods use
only common653input normalization; extra654is exclusive to budgetQ. Short
physical data have no terminal task labels.6000joint critic+4000actor actual
optimizer updates;3000aux-bearing steps are a subset of6000.

| Evaluation | P0 | Cm | Physical-action-removed | Task-only Q |
| --- | ---: | ---: | ---: | ---: |
| All four seeds, common methods /768 |244|383|399|Separate controls below|
| ColdQ block655/656 /384 |129|200|198|189|
| Equal-budgetQ block657/658 /384 |115|183|201|220|

Four of six frozen gates fail. Cm-minus-off -2.083pp pooled; Cm-minus-coldQ
+2.865pp below +5pp; Cm-minus-budgetQ -9.635pp. Each-seed noninferiority fails;
overP0 and motion1 safeguard pass. All motion0counts zero. BudgetQmotion2
115/128 remains partial own learning evidence, not Cm utility.

All eight native/fullmesh/P0/PD and complete training/input/final neural audits
pass. Current deployment observations reconstruct exactly; final network
NumPy max1.795e-6, mesh clearance max3.397e-7m. Optimizer/generator/checkpoints
retained, no independent optimizer-trajectory replay claimed. Same310272env-
controlticks per allocation. Aggregate1085952; r2new1008384+inherited77568.
r1valid651native collector completed then stopped under protected-code drift
when a short-normalizer accounting error was corrected BEFOREcomplete task
data/fitting/evaluation. r2inherits that panel underSHA and audits it; no valid
collection or fit repeated. Both histories retained. r2wall1059.607s/1757385080
bytes; combined recorded wall1142.990s, within2400s/3GiB. All own PIDs absent.
[Result](research/20261003-budgeted-physical-critic-results.md).

## Confirmed earlier evidence

- Self-trained P0 checkpoint15ed218f: motion1 56/64 on517/518; other motions
  thenzero. Scratch reference-relative controller, no source actor weights.
- Original full continuous auxiliary learner: Cm129/state129/noaux147/P0137
  per384, UNPROMISING.9120updates;22native/60first-batch optimizer checks,
  remaining9060updates not independently replayed.
- Deterministic successor actor: Cm172/off174/directQ222/P0129 per384;
  observed-successor law: Cm188/off188/directQ222/P0126;
  physical encoder -> measured-returnQ: Cm219/off218/directQ225/P0143.
  All complete utility gates fail; directQmotion2117/128 is partial competence.
- Full-parameter derivative control variate fails variance conditions despite
  actual gradients/reverse/physics audits. No actor fit from that failed gate.
- Actual8tick future oracle Brier.084409 versus Q.122460 and meanCm.120472:
  four diagnostic gates pass; no deployable utility or identified Jensen claim.
- Corrected late support: motion0no full or transient105joint-lift successes
  in256zero+256Gaussian trajectories. Original cohort-label error preserved,
  corrected r2verifies actual0P0/1zero duplicate/2Gaussian/3antithetic.
- Finite early24tick preparation also UNPROMISING: motion0random0/256 on two
  seeds. No universal infeasibility or sole-cause claim from sampled absence.
- Identical-reset paired environments diverge before intervention; original
  native layout retained. No exact same-state counterfactual oracle inferred.

Detailed negative families, numerical corrections and run provenance remain
in [research index](research/README.md), experiment cards and Git.

## Active question and next step

Causal execution, negative surfacecalibration, positive rigidtransport capacity
and negative learnedcoefficient/confidence screen are complete. Close this
exact inference recipe; score-only repair cannot recover overallgain even
with perfect future-outcome selection at fixedlearnedcoefficients. Do not
rescan density/data/epochs/seeds/width or use near subgroup to rescue failure.

State-anchored direct-flow/convexcorrection has now completed,4/5gatespass
but overallmatchedstate advantage8.955%fails10%; boundedlocalheadfamily
closed. No additionalsteps/seed/width/radius/objective/mixing rescans or
subgroup rescue; no nativequalification/actortraining from this failedgate.

Full-mesh rotational target-adequacy screen is complete and fails its decision
flip gate. Stop this candidate; no SE3 model or actor launched. Old6Dtarget
omitsfutureorientation, current64pointflow includesit; no causal attribution
of model failure to that omission. Netforce flags are not attributedcontact.
[Card](experiments/probes/P-20261003-rotational-clearance-adequacy.md),
[decision](decisions/D-20261003-after-rotational-clearance.md).

Training-label acquisition Probe now COMPLETED/UNPROMISING at4baf9a7:
contrast-minus-uniform+.362847mm²/upper2.202990fails; minusabsolute-1.923996/
upper-.126221passes; minuszero-2.988855pointpasses.3/6gatespass, bothTESTseed
uniformdifferences positive. Close exactallocator, no nativecollection/actor
fit or seed/budget rescans. All6144rawresponses/newforward/ranks/selections/
commoninit/scales/schedules/pairedrisk/bootstrap PASS;18manualAdamWupdates,
4782notreplayed; contextgeometry inheritedbyauditedidentity.4800newupdates,
22.104057s/9923571bytes,0physics/actor, protectedinputs unchanged/PIDsabsent.
All3072FITpool+3072TESTnativewindowsalreadypaid, not1024onlineinteractions;
legacyphysics/behavior and previouslyviewedtest, no corrected/nativepolicygain.
[Result](research/20261003-contrast-acquisition-results.md),
[decision](decisions/D-20261003-after-contrast-acquisition.md).

Current work: integrate actuallaterpolicy/filter/data/hand/granularity/transfer/
rigidcapacity-learnability/rotation/acquisition evidence into manuscriptv12.
Eightnewgeneratedtables/twoscientificfigures; fulloldv11/assets preserved.
FirstexportfinalizationmissingrootPDF assumption retained separately; corrected
native-v11PDF path, no trainingrepeated. Nativecompile/independentpaper audit
pending. Workingrevision only; not a changedclaim or journalcompletion.


User-authorized fixed-data granularity comparison completed with all six
gates failed. Close this exact matrix; do not sweep density, neighbor counts,
seeds, capacity or steps to rescue it. The subsequent causal execution-input
qualification is now complete. [Card](experiments/probes/P-20261003-cm-granularity.md),
[closeout](activities/20261003-granularity-closeout.md).

Core mission remains whether action-conditioned short physical transfers help
train a self-trained manipulation policy. Existing experiments do not establish
that claim. Close the exact joint auxiliary/data-allocation recipe, without
coefficient/width/steps/seed/horizon/truncation/label-budget rescue. No formal
Validation launched from a failed Probe.

The causal hand-input blocker is provisionally qualified on current reused
corrected episodes, not universally solved. Broad source prediction info does
not directly transfer even with oracle handflow; completed calibration also
fails. Preserve filterfix/all old evidence; newrigidendpointcapacity is
promising; completed learnedgates fail despite smallstate-anchored gain. No replay of closed
recipes or newfit without a separate prospective design.

## Preservation and debt

Immutable `/home2/wyy/tmp/ref2dex-contact-response-20261002-v11` and supplements
r1--r10 retain prior raw/code/model/audit/failure evidence. r10through22e47f5,
573885579unique bytes, all copies and prior manifests SHAverified.
[Latest receipt](activities/20261003-prelift-contact-delivery.md).
Current joint-budget r1/r2 preserved in supplement-r11 through a3563d9,
1960071139 unique bytes; all copies and prior manifests SHA verified.
[Receipt](activities/20261003-budgeted-physical-critic-delivery.md).

Supplement-r12through1530872 preserves filter red/green, fixed corrected655
and complete scale/hand matrix;603791182unique bytes, copies/prior manifests
SHAverified and Git bundle verified. [Receipt](activities/20261003-shape-filter-scale-delivery.md).

Supplement-r13throughc479cb3 preserves the coverage audit and complete eight-
model granularity Probe;456016084unique bytes, all copies/prior manifest SHA
and Git bundle verified. [Receipt](activities/20261003-granularity-delivery.md).

Supplement-r14 through3475ca1 preserves completed execution qualification,
raw/geometry/coefficients/features/predictions/audits and decision records;
334336017 unique bytes, all copied-file/prior manifest hashes and Git bundle
verified. [Receipt](activities/20261003-surface-execution-delivery.md).

Supplement-r15 throughd65a4e0 preserves calibration r1FAILEDaudit+completed
fits and r2verifiedaudit recovery;159222731unique bytes,60copiedfileSHAchecks
andGitbundle verified,priorv11/r1--r14manifests unchanged.
[Receipt](activities/20261003-surface-calibration-delivery.md).

Supplement-r16 through7dabad3 preserves completed rigidtransport fields,
196608oraclecoefficient/error/winner predictions and all audits/docs;
438819510unique bytes,44fileSHA+Gitbundle verified,base/prior manifests
unchanged. [Receipt](activities/20261003-rigid-transport-delivery.md).

Supplement-r17 throughe2cd561 preserves all3causal coupling fits/data/labels/
outputs/audits and separate saved-array diagnosis,372862110unique bytes,
allcopiedfileSHA/Gitbundle verified,base/prior manifests unchanged.
[Receipt](activities/20261003-rigid-coupling-delivery.md).

Supplement-r18 through31d8705 preserves failedprefitnormalization r1 plus
completed3state-anchored heads/r2/data/base/outputs/audits,163257183unique
bytes, allcopiedSHA/Gitbundle verified,base/prior manifests unchanged.
[Receipt](activities/20261003-state-anchored-transport-delivery.md).

Supplement-r19 throughc12d883 preserves completed fullmeshrotation Probe,
all155904poses/155136transitions/audits/logs/source/card/results/decision;
106913163unique bytes,46fileSHA/Gitbundle verified,base/r1--r18manifests
unchanged. [Receipt](activities/20261003-rotational-clearance-delivery.md).

Manuscriptv11 is an audited historical21page draft, not updated for recent
experiments or journal-ready. Formal multi-training-seed utility Validation,
broader tasks, specific novelty assessment and paper integration remain
[Research Debt](RESEARCH_DEBT.md); no negative observation is upgraded to a
formal scientific refutation. Goal ACTIVE.
