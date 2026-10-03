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

Collect FRESH:651768episodes stopped after101controlticks with no terminal105
labels;652768full202episodes with labels;653384additionalfull202episodes for
budgetQonly. Physics+commonlabels cost768*101+768*202=232704env-controlticks.
Additional-labelQuses(768+384)*202=232704exactly. Short101=half202is selected for
that exact accounting, not a task/transition-horizon scan. Model-responseh8fixed.
All sourcephysical/Qfits/labels beyond P0foundation are excluded from training.
Old sourceP0and unused construction heads/checkpoint provenance stay explicit.

Same model/actor initialization, jointcritics1500steps and actors1000steps;
constantaux.05/groupnormalizedphysical131, no coefficient/epoch/labelbudgetscan.
Fresh654/655evalP0/Cm/off/coldQand656/657P0/Cm/off/budgetQ, fulltask105unchanged.
Cm must gain>=5ppover bothlearned alternatives in their respective blocks, off
andP0pooled, allseed noninferiority and motion1safeguard. Everygate mandatory.
No labels generated for truncated episodes or claimed free data/compute.

One idleGPU, whole<=2400s/3GiB, original/source/newcodeSHAguard and allnative/
input/SDK/NN/PD/fullmesh/label audits. Positive triggers formalValidation and
novelty development; failure closes this allocation/jointQrecipe without
coefficient/steps/physicalwidth/labelbudget or seed scans. Coremission/claim and
externalresource permissions unchanged; genericauxQalreadyexists in priorart.
