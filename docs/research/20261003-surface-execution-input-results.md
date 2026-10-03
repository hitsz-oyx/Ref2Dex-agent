# Causal execution qualifies; frozen surface-prior transfer does not

P-20261003-surface-execution-input-r1 COMPLETED/UNCLEAR at code169db6c.
The execution gate passes, but both frozen priors fail their complete oracle
and causal gates. Preserve this distinction; the overall prospective label is
UNCLEAR, not a positive Cm-utility result or a universal prior rejection.

## Evidence that changes the next decision

Corrected655data were reused, with384whole train episodes and384whole held
episodes balanced by3motions x4fixed own-policy arms;16fixed ticks each,
6144windows per split. No new native physics, actor calls or optimizer updates.
Only54closed-form actuator coefficients percondition fit from currentq,
currentdq/30, senttarget-minus-currentq and intercept. Labels are measured
nextq; held futures enter only labels/oracle. Same native18joints, including
actual mimic-joint deviations, rather than enforcing ideal realized coupling.

| Hand-motion condition | Held hand-surface EPE/mm |
| --- | ---: |
|Stationary hand|9.710036|
|Instantaneous PD target|27.447348|
|Velocity-only calibrated|2.600907|
|Action+velocity calibrated|**0.899870**|
|Measured-nextq oracle|0|

Action+velocity improves65.402%over velocity-only and90.733%over stationary;
all three fixed execution conditions pass (<=75%velocity,<=50%stationary,
<=5mm). This is promising execution-input information on this distribution,
not an interventional or hardware accuracy guarantee. The oracle hand error
is zero by construction because measuredq FK supplies the label.

Object64point flow EPE/mm, equal weight perheld episode:

| Hand input | Frozen MANO7168prior | Frozen Inspire7168prior |
| --- | ---: | ---: |
|Stationary hand|4.772859|4.454851|
|Instantaneous PD target|20.951281|17.365145|
|Velocity-only|7.530724|6.926857|
|Action+velocity|7.346440|6.753861|
|Measured-nextq oracle|7.301355|6.725405|

Zero-object-motion baseline0.660945mm; previous-object-motion persistence
0.629373mm. Oracle errors are11.601x/10.686x persistence. Causal versus oracle
errors differ only+0.6175%/+0.4231%. Therefore inaccurate hand execution alone
cannot explain the large gap of THISfrozen transfer combination. This is not
an attribution of failure solely to physics or an assertion of formal equivalence.

Predeclared current64query minimum distance<2cm diagnostic:

| Subset | Windows | Available episodes | MANO oracle/mm | Inspire oracle/mm | Persistence/mm |
| --- | ---: | ---: | ---: | ---: | ---: |
|Near hand|2124|256|3.222090|3.160013|2.823930|
|Far from hand|4020|384|14.441342|13.083480|0.519538|

Near oracle still trails persistence14.10%/11.90%; neither subset rescues a
gate. Each subset separately gives equal weight to available episodes, so
window-weighted averaging does not reconstruct the all-window primary.
Near means unsigned geometric proximity, not measured contact/support.
The source prior train selected full4096pool proximity candidates, whereas
this all-phase native primary includes far states. That support difference
is explicit, not silently filtered after seeing results. All motion/arm
diagnostics are retained; no favorable subgroup changes the classification.

## Engineering evidence and cost

Wall195.423775s,249839074bytes, within900s/1GiB. CPU tiny smoke4.110s;
freshly idleGPU6qualification69.571s; CPU independent audit117.823s.
No new physics/optimizer steps. Existing collection cost is inherited,
not erased by reuse. Parent571927 and children571994/572145/573730 absent;
protected source, asset, checkpoint and code hashes unchanged.

All12288raw rows/split/timestamps rebuild exactly. Independent closed-ridge
coefficients max2.213507e-15; all6144joint predictions reconstruct. Geometry
is proper10135globalarea hand surface samples, new fixed correspondence
not claimed byte-identical to the external cache. URDF/mesh/seed/triangle/
barycentric/link ownership independently rebuild. Five retained SDK body
origins checked at every held nextstate: FKmax6.598234e-5m (0.066mm), rotation
entrymax2.056360e-6. This does not validate the entire native mesh surface.

Independent geometry/nearest-neighbor/features reconstructed at the FIRST
fixed window of every held episode (384rows, not all6144geometry rows):
featuremax1.212095e-5, targetmax1.194817e-5 normalizedflow;
nearest distance max1.042432e-7m, hand-error max3.055662e-5mm.
All ten full held network prediction sets match independent NumPyforward,
max6.675720e-6normalizedflow; parent metrics max1.694712e-7mm.
Baselines/gates independently recompute, as do all initial/split/geometry
contracts. Non-oracle predictions unchanged by future-label perturbation.
The existing read-only supervisor independently reaggregated all conditions,
motion/arm/near/far reports; maxdifference<3.9e-6mm, no implementation blocker.

Additional CPU geometry consistency check: all64canonical prior object
queries lie within the native25002vertex airplane convex hull, maximum signed
outside2.065019e-8m, with matching extent. It supports coordinate/scale
consistency, not a full triangle/collision or contact-validity proof.

## Scope and action

Retain the causal actuator for the next bounded corrected-data comparison.
Close this frozen direct-transfer combination; do not tune gates, samples,
seeds, density or the actuator to rescue its oracle failure. Existing9/24
actuator evidence used different legacy substrate and is not upgraded by this
single reused seed. Whole-episode holdout is not fresh policy/seed validation.
Hand labels are FKfrom measuredq; reused actors, mixedsource training, new
sampling, state support and original prior capacity remain possible factors.

Next distinguish output calibration from unusable pretrained features using
ONEprospective matched corrected-native causal prediction comparison. That
decision must precede any new policy training, large pretraining collection or
formal claim. A predictive gain would still require fresh randomized native
qualification and matched learned-policy benefit. Mission unchanged, journal
NOTREADY, goal ACTIVE. [Decision](../decisions/D-20261003-after-surface-execution.md).

Evidence: `src/task/CmResidual/research/contact_response/output/`
`P-20261003-surface-execution-input-r1`: manifest/logs/audit, selected raw
train/held rows, geometry and correspondence, closed coefficients, joint
predictions, five input feature banks, all ten frozen prior prediction sets,
per-episode errors and diagnostics. [Prospective card](../experiments/probes/P-20261003-surface-execution-input.md).
