---
schema: ref2dex.probe.v2
probe_id: P-20261005-amplitude-authority
experiment_id: P-20261005-amplitude-authority
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 37be6b1
claim_id: C3
hypothesis_family: HF-amplitude-authority
probe_index_in_family: 1
seed_pool: probe
seeds: [221, 222, 223, 224, 225, 226]
decision_changed_if_positive: independently compare directions at the lowest demonstrated interaction authority amplitude before any Cm fitting
decision_changed_if_negative: distinguish inadequate delivered amplitude from weak interaction measurement; stop this amplitude contract without a larger sweep or network retry
status: COMPLETED
run_id: amplitude-authority-s221
---

# Is there a task-related interaction threshold within larger feasible actions?

Result: Task-aligned authority UNPROMISING; localized thumb interaction response PROMISING (Probe only).
Decision: Preserve local thumb control evidence; no registered task-aligned threshold, so conditional Stage2 and neural fitting are not activated.

## Research decision and root reasoning

User-selected ref6 proposes a threshold/stable-basin explanation for weak
duration response. This is a hypothesis, not an established property: the
previous early-hold cohort already has roughly47% failure, so it is not a
uniformly stable grasp population. Thresholds may depend on state/direction.
Decision Probe asks whether larger feasible residuals expose task-relevant
interaction authority, rather than whether every sufficiently destructive
command can eventually make an object fall. Root retains separate mechanical,
interaction and useful-control judgments. Mere larger motion or universal
destruction does not establish a useful Cm candidate-ranking mechanism.

Cheapest discriminator: fixedK8 and joint alpha×direction randomization in
one cohort, with existing measured force vectors plus explicit geometry
projections. A pre-run file-only check of previous1,006 base actions finds
ALL support all seven candidates at alpha4 before assignment (also alpha1/2),
so common maximum-amplitude headroom is feasible without a smaller-arm cohort.
No other architecture, duration, seed, reward or RTG/RECAP fit is varied.

## Decision Note and budget

- Question: were previous residuals below a retention-relevant authority boundary?
- Evidence: hand movement increases with duration, but contact/I14 response
  does not pass; directional force was compressed to norms/absolute values.
- Root choice: alpha1/2/4 atK8, unchanged native feedback operator/physics/source,
  full signed force recording and nearest-surface-normal projection PROXIES.
  Do not equate normal/tangential projections of total force with paired
  contact forces, true friction margin or slip. No claim change.
- Cost: idleGPU6;84-episode engineering smoke≤300s, main168×10waves=1,680 full
  episodes≤1,000s. Only a positive task-aligned interaction gate permits a
  fresh direction cohort168×6≤700s at one selected alpha. Cumulative GPU
  cap2,000s, artifacts≤1GB, within global4GPU/300GB limits. CPU statistics
  and plotting only; frozen actor inference and simulation GPU.
- Stops: nonfinite data, bad native action/PD mapping, window/reset violation,
  input drift or resource conflict. Retain all assigned trials, including
  clipped and early-failed cases. Require≥20/nonzero alpha×arm cell,≥60
  pooled zero,≥8/cell in each five-wave half, full/half treatment rank18.
  Each cell's delivered signed dose ratio≥.90; otherwise authority status
  UNCLEAR, not proof of action ineffectiveness. No extra budget/seed to rescue.

## Fixed operator and measurements

Same frozen self-trained source_e260 and three airplane references. Same
10-state history, lift≥3cm/contact-proxy≥6 current observations, surface
proximity<6cm, >33 steps remaining in BOTH horizons. Common BEFORE-assignment
headroom covers all seven directions at alpha4. Uniform21cell assignment,
then clip(pi(actual_obs)+alpha*delta) for8steps, then baseline. Wrist base
delta±.01m is per-step PD offset relative to currentq; alpha4 gives±.04m,
not a40mm absolute target. Active finger±.1 native action units becomes±.4
at alpha4 (=±20% physical joint range, plus native coupling). Zero arms at
all alpha are identical physical operators and pooled; nominal tags saved.

Old72D physical trajectory retained, plus per-body nearest sampled object
surface normals at BEFORE state and each post-step. Signed body force15D
and object force3D already exist; preserve sign with asinh(force/1N).
At step8 report oldI14, raw force vectors, distance and geometry-projected
normal/tangential netforce: signed normal F dot n, tangent magnitude
norm(F-(F dot n)n), and tangent/(abs(normal)+1N). The1N denominator is a
numerical regularizer, not a friction coefficient. Nearest body-center
surface normals are approximate; netforces may include object/table/self
contacts. True contact identity, slip and mu are NOT inferred.

Primary short-contact fraction1..8, late retention9..32, independent physical
height failure9..32 and1..32, combined proxy-or-height failure with lost-run
starting atstep1. No post-step8 survivors selected. Time-varying vector,
normal/tangent projection, distance/contact/drop curves are plotted for all
amplitudes/directions, with state-adjusted point estimates and raw summaries.

## Stage1: nonlinear authority gate (before outcomes)

Use reviewed current-only nuisance and wave×motion×phase-quarter blocks.
18 nonzero alpha/direction indicators versus pooledzero allow arbitrary
threshold shapes; no linear dose assumption.1,999 within-block label perms
seed223, within-family max-tail. Record all contrasts and two fixed halves.
Families: short-contact, late-contact, late-height failure, combined failure,
all32height, I8, signed18force, surface-projection10, projection-ratio5.
No cross-family formal significance claim or feature search.

Task-aligned interaction authority is PROMISING when one SAME direction at
alpha2 or4 shows short-contact arm-minus-zero magnitude≥.15 with tail≤.10,
grows by≥.10 relative to alpha1, and has aligned late retention≥.05 or
opposite late physical height-failure≥.10 with corresponding tail≤.10.
Both five-wave halves must have same-direction short effect≥.03 and aligned
late effect. Alpha2/4 need not be smoothly monotonic: a threshold response is
allowed. Alternatively an existing I8 surface-distance axis family tail≤.05
with alpha-dependent change≥3mm and same-arm aligned late task contrast
passes only with the same repeated-half direction. Force direction/projection
alone is descriptive; more force is not assumed better grasp stability.

Select the LOWEST alpha≥2 passing this fixed gate, not the biggest destructive
arm or best posthoc score. Valid failure UNPROMISING; support/dose/rank failure
UNCLEAR. Also report mechanical/height authority separately: these cannot
rescue absent interaction/task alignment. No Cm/S fitted in either stage.

## Stage2: conditional independent direction comparison

Only if Stage1 passes, collect a FRESH independent cohort at the selected
single alpha with uniform seven directions, zero operator matched. Seeds
224/225/226, six fixed waves. This repeats discovered effects while asking
whether any direction improves retention and height safety relative to zero,
instead of merely producing harmful amplitude effects. Require≥40/arm,
≥15/arm in each three-wave half, delivered dose≥.90, valid native mapping.
Use the same physics labels and reviewed current nuisance (six indicators),
conditional permutation families and fixed independent repeats.

A useful-control signal requires late retention improvement≥.10 AND late
height-failure reduction≥.10 with both family tails≤.10, same-direction signs
in both halves and an interaction/force-direction response family tail≤.10.
Report harmful directions, current-state heterogeneity and any static arm
benefit separately. No useful direction: local useful-control UNPROMISING,
even if Stage1 destructive authority passed. Positive signal only justifies
a later state-conditioned consequence/value Decision; a fixed best arm does
not demonstrate need for Cm or trained-policy utility.

## Evidence, artifacts and scope

Task-local tools; new packets explicit v3, old K4/amplitude1 default preserved.
Unique `outputs/cm-interaction-oracle/<run_id>/`, project tmp logs/cache, no
videos/checkpoint replacement. Capture actual motion/checkpoint/config/code
hashes before launch. If an engineering flaw is found, preserve failed output
and repair only the affected still-decision-relevant part.
Multi-seed Validation, paired friction/slip sensing, counterfactual candidates
and trained-policy Cm-on/off utility remain deferred. This small exploratory
authority comparison cannot establish a universal stability threshold or
abandon the core Cm hypothesis.

## Pre-outcome engineering review

Smoke `amplitude-smoke-s13`:46/84 complete32 windows; amplitude1/2/4,
fixedK8 action mapping, nativePD reconstruction max error1.01e-7, finite
unit world normals all PASS. Smoke does not supply scientific gate evidence.
Collector execution commit `37be6b1` saves actual local surface sample,
seed42/stride8 and SHA, bridge/SurfaceGeometry/URDF/object-mesh source hashes.

Independent read-only review fixed two analysis guards before outcomes:
Stage2 zero counts must also pass both half support; stored normals must
match an actual nearest sampled point allowing numeric ties. Stage2, if
activated, will retain alpha4 common headroom instead of narrowing the
eligibility screen to alpha2. New threshold/distance/useful-control contract
tests and all Task tests pass30/30. No formal effects used for these fixes.

Partial geometric audit exposed float32 `cdist` MM cancellation near zero:
max saved-versus-direct distance0.2255mm, q99 about0.0098mm; reviewer independently
confirmed float64/direct difference only0.00013mm. This is not pose-frame drift.
Audit uses a squared-distance roundoff bound
`16*float32_eps*(||body||^2+max||surface_point||^2)` and records actual direct
error against the unchanged3mm gate. Every stored normal must still match a
real near-neighbor sample. Preserve original data and targets. Final complete
packet is audited again; projections remain aggregate geometry proxies.

## Executed result and evidence

Main `amplitude-authority-s221`, collector commit `37be6b1`:1,680 full
episodes,863 assigned trials,863 complete32-step windows. Minimum nonzero
cell29, minimum five-wave-half cell14, pooledzero125; treatment rank18 in
full and both halves. Minimum delivered signed dose0.99999988; nativePD
error1.12e-7. No trial removed after assignment. Current risk863, early
failure185, all32 height failure423, combined failure425, late combined423;
two early-failed recoveries retained. These are proxy/height labels, not
certified object-hand separation or slip. All13 input hashes unchanged.

Analysis `amplitude-authority-response-s223-r2`, commit `17d1af5`, seed223:

| Registered family | Within-family max-tail | Largest partial explained fraction |
| --- | --- | --- |
| Contact1..8 |0.8895|0.01525|
| Contact9..32 |0.5540|0.02310|
| Late physical height failure |0.1965|0.03136|
| Late combined failure |0.1705|0.03226|
| All32 height failure |0.2045|0.03087|
| I8 |0.0005|0.10163|
| Signed18 force |0.0780|0.05096|
| Surface projection10 |0.0005|0.09078|
| Projection ratio5 |0.0005|0.08010|

Primary task-aligned authority gate **UNPROMISING**:alpha4 maximum absolute
short-contact contrast3.68pp, below15pp. No contact or distance candidate has
registered late alignment and repetition. `selected_alpha=null`; no Stage2
cohort collected. No late task family passes its0.10 gate. These family tails
are exploratory, within-family conditional permutation summaries; they are
not a cross-family formal Validation claim.

Mechanical authority is real: at the SAME step8, measured wristx+ body-centroid
dx grows12.95→23.53→47.61mm for alpha1/2/4; wristz+ dz13.74→23.13→49.35mm.

Localized interaction authority is **PROMISING**:strongest I8 axes are
`thumb_distal` distance and force norm (contactBodies order checked against
actual hashed source config). At alpha4:

| Direction | Thumb distance minus zero | Two half distance effects | log1p thumb-force norm contrast | Two half force effects |
| --- | --- | --- | --- | --- |
| finger+ |−8.44mm|−8.06/−7.33mm|+0.234|+0.367/+0.192|
| finger− |+22.95mm|+14.09/+32.57mm|−0.578|−0.421/−0.756|

Thumb also dominates the tangent projection/ratio response; signed normal
projection alone is weak. Force values above are transformed contrasts,
not a0.578N force difference. They demonstrate controllable local aggregate
force/proximity, not friction margin or a force vector predictive advantage.
Original norm-based I already captures this signal; loss of direction cannot
by itself explain the missing task contrast. Global contact proxy is an OR
across hand bodies AND object netforce, so it can hide changes in one thumb;
independent physical height outcomes also remain weakly differentiated.

## Final engineering audit and root interpretation

Independent reviewer and root reconstruct targets/labels exactly and joint
OLS to≤5.1e-14 including both halves, with the same empty candidate set. Root also reconstructs
the global OR contact-proxy exactly from saved body/object force vectors.
Complete packet geometry audit:direct distance max error0.2255mm/q99
0.00966mm;29 normal ties across142,395 body observations match real sampled
near-neighbors. Large normal-component replay discrepancy1.936 is attributable to
tied nearest surfaces, not arbitrary normals. Independent direct-distance
sensitivity changes the thumb coefficients by at most0.000518mm (all bodies
0.002895mm); this cannot explain22.95mm response or the3mm gate. No affected
scientific target was changed and no simulation was rerun.

One analysis CLI attempt failed BEFORE statistical computation because a
relative `__file__` was passed to source identity construction. Original
empty output directory, failure.json and log retained; `17d1af5` fixes it and
r2 uses the identical dataset/seed/protocol. This was not a scientific retry.

Root Decision Note:the useful information is a split result, not another
blanket failure of action→I. Alpha4 exposes local thumb force/geometry control,
but no task-aligned threshold or useful interval has been identified. Thus
neither the stable-basin explanation nor a universal absence of thresholds is
established. Root follows the registered stop for Stage2; does not select
alpha4 just because it makes the largest local change. Within this operator,
next meaningful question is whether a controllable contact component is
actually load-bearing/state-dependent for physical retention. More amplitude,
seeds or predictor capacity to repeat motion authority are not the next
Decision. A new operator/measurement question would need a new minimal
contract, preserving this negative task gate. Global Cm policy utility stays
OPEN; no core hypothesis or final claim is changed.

Artifacts (all beneath `outputs/cm-interaction-oracle/`):
- `amplitude-authority-s221/`:packet, actual geometry sample/provenance,
  source-hash audit, collection manifest/result and full episode summary.
- `amplitude-authority-response-s223-r2/`:full registered results, diagnostics,
  engineering audit/review, mechanical/force response curves; standalone
  `amplitude_adjusted.png`, `amplitude_raw.png`, `time_response_adjusted.png`,
  `signed_force_response.png` and legible `direction_facets.png`.
- Facets are descriptive thumb/global-outcome views, not new tests or gates;
  separate `direction_plot_manifest.json` hashes plotting inputs/code.

GPU6 player collection671.48s total (smoke56.00+main615.48; startup overhead
in manifests:66.21s smoke+625.72s main including player initialization),
CPU registered statistics29.92s; bounded artifacts<40MiB.
All GPU collection ended. Conditional seeds224/225/226 unused. Deferred:
matched multi-seed Validation, paired/contact load identity, same-state
candidate comparison and trained-policy matched Cm-on/off utility.
