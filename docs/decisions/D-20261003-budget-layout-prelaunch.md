# Pre-collection amendment: preserve the native scene layout

Initial22e47f5design uses384extra environments to equal768*101ticks. Reading the
native reset/SDK pipeline confirms environment count changes scene origins/layout.
A budgetcontrol should not pay an avoidable scene-distribution penalty relative
to768-environment evaluators. No new budget experiment data, model outputs or
utility results have been generated. This is a prelaunch design correction.

Use TWO768*101unlabeled physical panels and ONEextra768*202labeled panel:
1536*101+768*202=1536*202=310272env-controlticks EXACT. Alltraining/evaluation
panels768environments, same origins/layout/physics. Fresh651/652short,
653commonfull,654extrafull;655/656coldQeval and657/658budgetQeval.
Physicalpool651/652/653=2304pairs; commonlabels653768; budgetlabels653/6541536.
Allnormalizers exclude654. Critics/actors/auxweight/updates/h8/finaltask/gates
unchanged. Whole2400s/3GiB unchanged, oneadditionalnativephase withinbounds.

Replaces initial768short/384extra counts BEFOREcollection, without selecting
results, scanning taskhorizons or dropping available labels. Original design
history retained. Currentcard is authoritative for scientific launch.
