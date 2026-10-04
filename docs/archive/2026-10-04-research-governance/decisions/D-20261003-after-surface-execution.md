# After causal execution qualification

Question: should the failed frozen transfer trigger more actuator/data/density
tuning, or test whether corrected-native output calibration can retain useful
pretrained features?

Evidence: execution gate passes (0.900mm vs2.601velocity/9.710stationary).
Frozen priors oracle6.725/7.301mm vs persistence0.629mm; causal replacing
oracle adds only0.42%/0.62%. Near-hand oracle also loses to persistence.
Independent raw/neural/geometry/gate audits and read-only review pass.
This excludes the simple actuator-error remedy for THISfixed transfer,
not all priors; sampling/state/physics/source factors are not isolated.

Action: preserve the actuator, close exact frozen direct transfer. Choose ONE
bounded matched corrected-data causal calibration comparison next. Use an
explicit persistence anchor and compare useful pretrained encoder, matched
shuffled-target pretrained encoder and scratch features under the SAME new
prediction head/targets/data/update budget. Freeze or otherwise predefine
encoder treatment and all gates BEFOREfitting; no post-hoc model selection.
This asks whether information survived pretraining after output/domain
calibration, not whether an unavailable futurehand input predicts labels.

Cost: targetoneGPU/<15min/<1GiB using retained correctedtrain episodes;
no new collection for the first screen. Success leads to fresh randomized
corrected-physics qualification before policy fitting. Failure closes this
bounded surface-feature transfer family and triggers representation/target
review rather than larger calibration banks/epochs/coefficients/seeds.
Same MISSION/claim/resources; no new external authorization needed. Formal
Validation, novel-method claims and journal readiness remain unestablished.
