# Physical information under an actual task-data budget

Question: can short physically labeled episodes improve joint action-conditioned
Qlearning when their cost is included, compared to action-removed physical aux
and using the same ticks for additional complete task episodes?

Evidence: P0and strong direct-Q support motion1/2, but imagined future values,
physical encoder warm-start, and state-only Vcritic auxiliary PPOfail Cm gates.
Moving random preparations earlier gives no complete/transientmotion0support.
Do not keep adjusting those recipes or optimize the no-Cm baseline forever.

Choose joint action-conditioned Qencoder, real-return task head and measured
8tickphysical-response auxiliary head. Physical-on/off differ ONLYin whether
that auxpass receives the actualoption or zeros; BOTHtaskpasses seeactualactions.
All sharedencoder parameters are trained jointly on actual physical supervision
and measured task returns; there is no pretrained encoder replacement or predicted
future passed to V. Retain no-auxcoldQ and equalcost extra-labelQ strong controls.

Prelaunch amendment D-20261003-budget-layout-prelaunch keeps ALLpanels at768
nativeenvs to avoid layout/origin change. FRESH651/652each768episodes truncated
at101ticks without terminal105labels;653768full202labels;654768extra-full202
forbudgetQONLY. Physics+commonlabels cost1536*101+768*202=310272ticks. Additional-
labelQuses1536*202=310272exactly. Short101=half202is for exactaccounting, not a
response-horizon scan. Physicalpool651/652/6532304pairs; common653768labels;
budget653/6541536labels. Extra654excluded from normalizers/auxpool/commonlearners.
Old sourcefits/labels excluded, only ownP0foundation and unused construction
metadata retained. No data or model has run under initial384extra design.

Same model/actor initialization, jointcritics1500steps and actors1000steps;
constantaux.05/groupnormalizedphysical131, no coefficient/epoch/labelbudgetscan.
Fresh655/656evalP0/Cm/off/coldQand657/658P0/Cm/off/budgetQ, fulltask105unchanged.
Cm must gain>=5ppover bothlearned alternatives in their respective blocks, off
andP0pooled, allseed noninferiority and motion1safeguard. Everygate mandatory.
No labels generated for truncated episodes or claimed free data/compute.

One idleGPU, whole<=2400s/3GiB, original/source/newcodeSHAguard and allnative/
input/SDK/NN/PD/fullmesh/label audits. Positive triggers formalValidation and
novelty development; failure closes this allocation/jointQrecipe without
coefficient/steps/physicalwidth/labelbudget or seed scans. Coremission/claim and
externalresource permissions unchanged; genericauxQalreadyexists in priorart.

Input-normalizer accounting correction before complete task data/fitting:
[D-20261003-budget-normalizer-correction](D-20261003-budget-normalizer-correction.md).
All methods use only common653current/future SDKstats, never short651/652stats.
