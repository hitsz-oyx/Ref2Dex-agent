# Fixed randomized-request object-response screen

Classification: Decision, reused training ONLY. Execution-headroom screen passes
100%; projection repair is not the principal route. Now distinguish whether the
current one-step object target misses a usable delayed closed-loop response.

Use all20training547--566panels, all64motion2no-auxiliary episodes/panel. Never
use final568/569or select another arm/axis/lag/window. Eligible current states:
the original105reference ticks, current PRE-action root rise>=30mm and full-mesh
table clearance>=20mm, with all required future samples available. These are
airborne-geometry states, not certified mechanical grasps. Require>=1024eligible
transitions and>=32eligible episodes, otherwise UNCLEAR and stop.

Treatment is saved STANDARD-GAUSSIAN innovation in request coordinate2, the
third translational coordinate. Outcome at fixed horizon h=1or8 is actual world
object z at t+h minus current PRE-action z minus h*dt*current z velocity, in mm.
Recover dt from the saved cfg_env controlFrequencyInv2 and native source
SIM_TIMESTEP1/60 (no dt override): one action tick1/30s. Hash both sources.
The score moment `mean(epsilon_t*Y_h)` estimates a
one-standard-deviation Gaussian mean sensitivity under the realized closed-loop
policy; it is not the effect of intervening on an endogenous executed target,
not an open-loop model forecast, and not a counterfactual physical rollout.
Future adaptive actions are part of the closed-loop response. No future hand,
policy action or measurement enters a learned input; no model is fitted.

Negative control: innovation at t+8 multiplied by Y_8, whose last outcome is
post-step t+7. The control innovation is sampled strictly AFTER the outcome.
Available t<=193for202steps. Eligibility uses PRE-action states and deterministic
reference times only; never future success/contact filters. Independently check
one-step outcome against stored physical_transition z*.005m and saved standardized
noise against the behavior checkpoint's Gaussian request identity. Absolute
audit limits: one-step position reconstruction1e-5mm, noise reconstruction2e-5.

Exploratory uncertainty:2000episode-cluster bootstrap samples, stratified within
each of20panels, fixed bootstrap seed947. These condition on one realized learner
sequence; not independent optimization runs or formal Validation. Bootstrap each
panel's ALL64episodes (including zero eligible counts), retaining denominator
weighting. Compute95%percentile intervals for moments at1and8, paired8-minus1,
and the future-innovation control. No multiple lag/axis/threshold selection.

Fixed decision gate: h8moment>=0.1mm AND lower95%interval of paired8-minus1>0.
Negative-control95%interval must lie within[-0.1,+0.1]mm; otherwise UNCLEAR and
stop. If all gates pass, PROMISING only for this delayed-response target and
design a representation that conveys such response directly to policy training.
If the delay gate fails, UNPROMISING; stop this fixed lag hypothesis without
scanning lags/axes/doses, and choose support/retention-relevant physical features
or a different policy integration. Neither outcome establishes novelty, Cm
policy utility, mechanical support or universal absence of response.

CPU file processing/moment/bootstrap statistics, zero model inference/fits,
zero fresh physics/optimizer updates,<=120s/5MiB. Preserve all source hashes,
eligibility counts and original outcomes; no change to any continuous recipe.
