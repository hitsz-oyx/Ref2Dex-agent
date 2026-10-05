---
schema: ref2dex.probe.v2
probe_id: P-20261005-amplitude-authority
experiment_id: P-20261005-amplitude-authority
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: actual execution commit in manifest.json
claim_id: C3
hypothesis_family: HF-amplitude-authority
probe_index_in_family: 1
seed_pool: probe
seeds: [221, 222, 223, 224, 225, 226]
decision_changed_if_positive: independently compare directions at the lowest demonstrated interaction authority amplitude before any Cm fitting
decision_changed_if_negative: distinguish inadequate delivered amplitude from weak interaction measurement; stop this amplitude contract without a larger sweep or network retry
status: PLANNED
run_id: amplitude-authority-s221
---

# Is there a task-related interaction threshold within larger feasible actions?

Result: Pending fixed-K8 amplitude authority and conditional direction probes.
Decision: Compare alpha1/2/4, preserving signed forces; train no neural model.

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
