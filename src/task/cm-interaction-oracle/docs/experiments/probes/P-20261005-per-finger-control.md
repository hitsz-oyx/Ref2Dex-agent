---
schema: ref2dex.probe.v2
probe_id: P-20261005-per-finger-control
experiment_id: P-20261005-per-finger-control
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: actual execution commit in collection manifest
claim_id: C3
hypothesis_family: HF-per-finger-control
probe_index_in_family: 1
seed_pool: probe
seeds: [227, 228, 229]
decision_changed_if_positive: prioritize an isolated load-relevant control dimension before state-conditioned GT or Cm usefulness testing
decision_changed_if_negative: preserve local response and close this single-DOF range contract without more amplitudes or network fitting
status: RUNNING
run_id: per-finger-control-s227
---

# Does separating the finger synergy expose task-related interaction control?

Result: Smoke43 full windows audited; main isolated six-DOF comparison collecting.
Decision: Run one fixed physical-range fraction with matched synergy controls.

## Research decision and root interpretation

Task aims at GT E/I usefulness, then predicted Cm and matched trained-policy
utility. Ref6 found strong local thumb response but weak task outcomes;
ref7 proposes cancellation or dominant redundant thumb in the composite arm.
These are hypotheses, not known load identities. **Decision** asks whether an
individual independent finger DOF has own interaction and late-task response
masked by the composite. Positive result changes which control variable merits
GT/value work; negative stops this contract, not the core Cm hypothesis.

Cheapest discriminator: six isolated DOFs, both signs, zero, and original
five-driver synergy both signs in ONE randomized cohort (15arms). Includes
previously omitted thumb-yaw. Fixed K8, fixed20% independent driver's range,
no amplitude sweep. Actual native mapping has slope range/2: delta_action0.1
ALREADY means5% of each driver's range; range normalization alone is not a
new operator. Splitting the drivers is the material change.20% matches ref6
alpha4 so smaller dose cannot confound the comparison. Pre-run file-only
screen finds all863 previous decision base actions support ALL new arms,
including yaw; this does not imply future support or load-bearing identity.
Equal fractions imply neither equal radians nor equal fingertip millimetres.

## Decision Note and resources

- Evidence: ref6 thumb distance/force responses repeat, task-aligned gate fails.
- Root choice: isolate native indices6/8/10/12/14/15 with unchanged native
  mimic coupling; retain composite controls in same cohort. Default API5%
  available, this fixed Probe20%. No Jacobian5mm calibration: it adds a
  state-dependent operator and is unnecessary to isolate synergy effects.
- Cost: idleGPU6;84episode smoke≤180s;main168×10=1,680episodes≤850s;
  total GPU cap1,200s including initialization, outputs≤1GiB. CPU file/OLS
  statistics only; no neural training, simulation/model inference GPU.
- Stop on input drift, nonfinite values, reset/window/action/native mapping
  error or resource conflict. All assigned trials retained, including early
  failures and future clipping. No support-rescue cohort or seed retry.
- Within existing Mission/Campaign/user authorization; no external boundary.

## Frozen collection and physical audit

Same source_e260, three airplane motions, current-only early-hold/horizon
screen, full episode reset and32post-step window as ref6. Headroom checks ALL
15 arms BEFORE uniform assignment. Feedback baseline plus delta for8steps,
then baseline. Wrist is not directly perturbed. Per-finger nominal target
amplitude is±20% driver URDF range; follower targets use actual native ratios,
NOT20% of follower ranges. Delta action±.4; no direct residual on mimic DOFs.
Yaw has no mimic followers; pitch still moves16/17 by0.6/0.8. Composite excludes
yaw exactly as before; unrelated dimensions have zero DIRECT target difference.

Explicit v4 payload adds all32 measured native joint positions (q0:3 metres; q3:18 radians),
five actual `*_tip` world positions and measured hand-base poses. Existing
contact-body positions remain mostly intermediate bodies, thumb_distal; do
not label these old points fingertip motion. Save actual geometry sample,
config/native code/URDF/mesh hashes and contact/tip names. Audit action doses,
PD target radians/degrees/range fractions and mimic followers per arm. Record
min/mean/max delivered doses and clipping rates. Record measured joint changes
and per-finger tip motion atstep8 versus BEFORE plus same-cohort zero-adjusted
contrasts, with units. Hand-base-local tip motion separates wrist drift;
world/object-relative descriptions may differ. Neither raw motion nor target
difference is an isolated realized causal joint trajectory.

Each independent-DOF magnitude and mimic mapping plus ALL arms' five-finger
motion and six-driver responses will be appended to THIS card. Ref6's card
also receives its per-alpha target audit; legacy packet has no post-step q or
true tips, so missing quantities are marked unavailable, never fabricated.

## Pre-outcome decision gate

Reuse current-only nuisance and wave×motion×phase-quarter blocks,14nonzero
indicators versus zero,1,999 within-block permutations seed229. Families and
time labels unchanged from ref6: shortcontact1..8, latecontact9..32, late
physical height failure, late combined failure, all32 physical height failure,
I8, signed18force, projection10 and ratio5. Two fixed five-wave halves.
Require≥30 ALL arm cells includingzero,≥12 ALL cells per half, full/half rank14,
and each nonzero cell signed action dose≥.90. Otherwise UNCLEAR. No post-step
survivor filtering. Report all contrasts, not only discovered candidates.

For any isolated arm1..12, require OWN-body I8 force-norm contrast magnitude
≥0.15 log1p(N) or distance≥3mm, corresponding within-I8-family tail≤.10 and
same sign in BOTH halves. Index/middle/pinky/ring map contactbody0/1/2/3;
yaw/pitch both map thumbbody4. Then require same arm latecontact magnitude
≥.08 OR opposite-oriented late physical height failure magnitude≥.10, with
corresponding tail≤.10 and repeated same sign in both halves. This establishes
exploratory task-linked control (beneficial OR harmful); contact and height
when both qualify must align. No assumed relation between more force and
more stable grasp. No candidates with valid support/dose: UNPROMISING.

Separately useful-direction signal requires latecontact+.10 AND late physical
height failure−.10, both tails≤.10 and both halves same signs, plus the own-I
criterion. A harmful-only result is not policy usefulness. The composite
controls are reported, not allowed to rescue absent single-DOF candidates.
Single-versus-synergy effects are marginal contrasts, not same-state additive
cancellation identification or certified contact load attribution.

## Limitations and deferred evidence

Joint-range fractions do not standardize Jacobians or actual tip displacement;
measured asymmetry is part of the audit, not hidden. Total force/proximity and
geometric projections do not certify paired friction/slip or true load-bearing
contacts. Source early-hold proxy does not certify an established stable grasp.
Single cohort is a Probe; independent repeats, 5mm FK/Jacobian calibration,
true paired contact identity and trained-policy matched Cm-on/off Validation
remain deferred unless the decision signal justifies them.

## Pre-main engineering record

Collector commit741e18d engineering smoke completed; fixed source/15arm
encoding/native PD/coupling/true tip-body names reviewed independently.
Nominal driver20%range:four fingers±0.320rad (±18.335deg), followers±0.336rad
(±19.251deg); yaw±0.230rad (±13.178deg); pitch±0.110rad (±6.303deg), followers
±0.066/0.088rad (±3.782/5.042deg). Composite13/14 exactly matches ref6 alpha4
finger± in native command space. No future capture added to actor input.

Before main, fix units metadata:smoke incorrectly labelled whole18D q as
radians; numerical tensors/driver audit are unaffected, original smoke kept.
Main explicitly annotates q0:3 metres and q3:18 radians. Measured finger
angles and true tip millimetres still require delivered-data audit.

Smoke `per-finger-smoke-s15`:43/84 complete32windows,15arm table/K8;
expected action and nativePD exact (max error0), finite real q/tip arrays;
geometry/sample replay PASS. Source commit for main `a81b36d`, seed227/228.
Pre-main independent review confirms single-driver/coupling/post-state capture;
pre-outcome audit/gate review and36 Task contracts including clean CLI entry
and rigid-hand motion removal pass. Standalone audit root import repaired
before main statistics; initial failure occurred before reading any dataset.
