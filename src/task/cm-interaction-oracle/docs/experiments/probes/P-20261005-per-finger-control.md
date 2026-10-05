---
schema: ref2dex.probe.v2
probe_id: P-20261005-per-finger-control
experiment_id: P-20261005-per-finger-control
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: a81b36d
claim_id: C3
hypothesis_family: HF-per-finger-control
probe_index_in_family: 1
seed_pool: probe
seeds: [227, 228, 229]
decision_changed_if_positive: prioritize an isolated load-relevant control dimension before state-conditioned GT or Cm usefulness testing
decision_changed_if_negative: preserve local response and close this single-DOF range contract without more amplitudes or network fitting
status: COMPLETED
run_id: per-finger-control-s227
---

# Does separating the finger synergy expose task-related interaction control?

Result: 854 full windows; registered linked/useful gates UNPROMISING, with late-contact action response and yaw-minus/middle-minus clues retained.
Decision: Preserve physical magnitude tables and targeted task clues; no amplitude/seed sweep or Cm fitting from this completed contract.

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

## Completed evidence and root decision

Main `per-finger-control-s227`, commit `a81b36d`, seed227/228:854 complete
32-step windows from1,680 full episodes. All15 arms have42–78 trials; both
five-wave halves minimum17, full/half treatment rank14/14/14. All nonzero
signed action dose ratios1.0; active driver coordinates clipping0%; nativePD
replay exact.17 input hashes unchanged. Packet SHA:
`138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149`.

Statistics/audit `per-finger-response-s229`, commit `30a0980`, seed229:

| Family | Within-family max-tail | Largest partial explained fraction |
| --- | --- | --- |
| Short contact1..8 |0.1725|0.02538|
| Late contact9..32 |0.0005|0.06370|
| Late physical height failure |0.1535|0.02682|
| Late combined failure |0.1185|0.02808|
| All32 physical height failure |0.1440|0.02708|
| I8 |0.0005|0.09776|
| Signed18 force |0.0010|0.06593|
| Surface projection10 |0.0005|0.07735|
| Projection ratio5 |0.0005|0.07693|

Registered own-I+late-task gate **UNPROMISING**, useful-direction gate
**UNPROMISING**:no qualifying single-arm candidate. This does NOT mean no
single-finger task response. Late contact response is present in this Probe;
its global netforce proxy is not certified grasp stability. Height safety
family does not meet0.10; no changed gate, new seed or predictor fitted.

Two informative, nonqualifying directions are preserved:

| Isolated direction | Late contact contrast | Two half contact contrasts | Late height failure contrast | Two half height contrasts | Gate limitation |
| --- | --- | --- | --- | --- | --- |
| middle− |−10.94pp|−8.46/−13.76pp|+6.23pp|+8.33/+4.17pp|own force−0.281 log1p(N)/distance+11.25mm repeat, but I-family tails0.131/0.324 fail |
| thumb-yaw− |+10.76pp|+7.27/+9.08pp|−11.34pp|−5.02/−9.67pp|height family tail0.1535; own force−0.079/distance−1.15mm too small and half signs disagree |

Composite− late contact−12.60pp repeats but cannot rescue a single-arm gate.
These are exploratory point estimates, not validated individual contrasts.
All854 are initially at risk;193 early failures,466 all32 combined failures,
465 all32 physical height failures,462 late physical height failures and240
proxy-loss failures retained. No survivor-only subset is used.

Root interpretation: splitting synergy exposes real task-proxy differences,
including the previously omitted yaw direction; marginal comparisons do not
identify additive cancellation or prove any finger redundant/load-bearing.
Equal20% commanded range produced markedly unequal realized motion: isolated
index tips21.69/24.38mm (+/−), middle25.71/26.85, pinky15.19/17.56,
ring27.62/29.06, thumb-yaw9.33/9.87 and thumb-pitch10.15/13.94mm. These are
NORMS OF ADJUSTED VECTOR CONTRASTS in the measured hand-base frame, not every
trial's travel or mean-travel differences. Thus thumb was not the largest
moving tip; ref6's thumb-dominant force response cannot be explained solely
by assuming greater thumb displacement. Joint response also varies relative
to commandedPD; detailed tables below make that explicit. Nonperturbed
fingers can move through contact/closed-loop feedback despite zero DIRECT PD
offset; e.g index− has a6.85mm thumb contrast, middle+ changes ring4.12mm.

Root Decision Note: retain yaw− as a possible beneficial direction and middle−
as a harmful direction clue; keep the original gate negative. The next useful
Decision is a narrowly targeted independent
yaw−/zero comparison of physical height safety and legitimate retention,
not another blanket amplitude/seed sweep or5mm calibration just for symmetry.
The I8 norm/distance gate and weak height evidence do not yet establish a
reliable interaction→height-safety chain or justify Cm training. A later
new targeted protocol must distinguish a reproducible physical benefit from
netforce-proxy improvement; it cannot retroactively rescue this gate.
Global trained-policy Cm utility stays OPEN. No core claim changed.

## Delivered-data engineering audit

Independent reviewer and root reconstruct targets exactly and joint OLS to
≤4.2e-14 including halves; registered candidate sets agree. Reviewer independently
reconstructs range/2+mimic PD differences to2.09e-7, and actual q/tip tables
using a separate NumPy quaternion transform to8.42e-6mm. Root's separate
inverse-quaternion formula differs by only5.35e-6mm. Root also reconstructs
all global contact-proxy flags exactly from recorded netforce vectors.

Geometry replay:saved-versus-direct distance max0.1343mm/q99 0.00972mm;
33 numerical normal ties match real nearest-surface samples. No code-frame,
reset, dose or timing error explains the signals or unequal motion. Measurement
limits remain:intermediate contact bodies differ from true tip points, total
body force is not certified paired contact, and vector-contrast norm is a
point estimate. Independent review cannot itself upgrade this Probe to
Validation. No raw packet or prior scientific result changed.

GPU6 smoke/main manifests68.05+623.40=691.45s including initialization,
CPU analysis/audit/plot6.68s, new artifacts<40MiB. Collection ended;36 Task
tests and scoped repository checks pass. No videos/checkpoint overwrite.

Outputs:`outputs/cm-interaction-oracle/per-finger-control-s227/` for raw packet,
source/geometry provenance and full episode summary; `per-finger-response-s229/`
for immutable registered result/diagnostic, per-finger JSON/Markdown tables,
response curves, standalone `per_finger_response.png`, engineering audit and
`independent_engineering_review.json`. JSON includes all12 native finger joints
6..17 with actual target min/mean/max, degrees, physical fractions, clipping,
raw realized q/travel min/mean/max and adjusted driver/tip vectors.

## Per-finger perturbation amplitude audit

Units: native action dimensionless; PD targets/actual q radians; tip displacement mm.
PD target changes are commanded, not actual executed joint angles. Driver table order: index, middle, pinky, ring, thumb-yaw, thumb-pitch. JSON PD/action arrays use all12 native joints6..17, including followers.

| Driver | Native index | Physical range (rad) | Native mimic followers |
| --- | --- | --- | --- |
| index | 6 | 1.6000 | 7: ×1.05 |
| middle | 8 | 1.6000 | 9: ×1.05 |
| pinky | 10 | 1.6000 | 11: ×1.05 |
| ring | 12 | 1.6000 | 13: ×1.05 |
| thumb_yaw | 14 | 1.1500 | none |
| thumb_pitch | 15 | 0.5500 | 16: ×0.6;17: ×0.8 |

### Delivered PD target magnitude per arm

Signed offsets per active step, not cumulative K-times angles. Finger targets are absolute native PD targets with a per-step baseline offset. Min/mean/max per joint, degrees and range fractions are preserved in `finger_amplitudes.json`. Composite arms exclude thumb-yaw. Clip % counts active commanded finger coordinates.

| alpha | Arm | n | index rad | middle rad | pinky rad | ring rad | yaw rad | pitch rad | clip % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | zero | 56 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | index_plus | 49 | +0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | index_minus | 56 | -0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | middle_plus | 51 | +0.00000 | +0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | middle_minus | 63 | +0.00000 | -0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | pinky_plus | 72 | +0.00000 | +0.00000 | +0.32000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | pinky_minus | 46 | +0.00000 | +0.00000 | -0.32000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | ring_plus | 60 | +0.00000 | +0.00000 | +0.00000 | +0.32000 | +0.00000 | +0.00000 | 0.00 |
| 1 | ring_minus | 55 | +0.00000 | +0.00000 | +0.00000 | -0.32000 | +0.00000 | +0.00000 | 0.00 |
| 1 | thumb_yaw_plus | 49 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.23000 | +0.00000 | 0.00 |
| 1 | thumb_yaw_minus | 54 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | -0.23000 | +0.00000 | 0.00 |
| 1 | thumb_pitch_plus | 78 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.11000 | 0.00 |
| 1 | thumb_pitch_minus | 42 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | -0.11000 | 0.00 |
| 1 | synergy_plus | 62 | +0.32000 | +0.32000 | +0.32000 | +0.32000 | +0.00000 | +0.11000 | 0.00 |
| 1 | synergy_minus | 61 | -0.32000 | -0.32000 | -0.32000 | -0.32000 | +0.00000 | -0.11000 | 0.00 |

For composite ± at a20% driver range dose, intermediate followers6→7 etc receive ±0.336rad (10.70% of their3.14rad range); thumb-pitch followers receive ±0.066/0.088rad (2.10/2.80% of their ranges). At5% driver dose these are divided by4. Yaw has no target coupling.

### Measured native driver response at step8

Current-state-adjusted arm-minus-zero contrasts of actual q(step8)−q(before), radians. Not PD targets.

| Arm | index | middle | pinky | ring | yaw | pitch |
| --- | --- | --- | --- | --- | --- | --- |
| index_plus | +0.12875 | +0.03508 | +0.01204 | +0.00308 | -0.00562 | -0.00613 |
| index_minus | -0.15809 | -0.02648 | -0.00266 | +0.00306 | +0.00053 | -0.02202 |
| middle_plus | +0.05977 | +0.16934 | +0.02540 | +0.06805 | -0.00172 | -0.01241 |
| middle_minus | -0.03351 | -0.14832 | +0.00320 | -0.03896 | +0.00427 | -0.00623 |
| pinky_plus | +0.01215 | +0.02435 | +0.26734 | -0.00121 | +0.00015 | -0.00239 |
| pinky_minus | -0.00205 | +0.01421 | -0.26966 | -0.03733 | +0.00081 | +0.00415 |
| ring_plus | +0.00987 | +0.04957 | +0.01291 | +0.23303 | -0.00184 | -0.00111 |
| ring_minus | -0.00517 | -0.00878 | +0.00370 | -0.22309 | +0.00526 | -0.00427 |
| thumb_yaw_plus | -0.02100 | -0.01908 | -0.00647 | -0.01102 | +0.21367 | -0.00504 |
| thumb_yaw_minus | +0.00048 | +0.03201 | +0.00474 | +0.00392 | -0.20989 | -0.00512 |
| thumb_pitch_plus | -0.02185 | +0.00034 | +0.00140 | -0.00016 | +0.00352 | +0.08702 |
| thumb_pitch_minus | +0.01170 | +0.02278 | +0.00628 | -0.00391 | +0.00023 | -0.10148 |
| synergy_plus | +0.20111 | +0.25448 | +0.29047 | +0.25572 | -0.01000 | +0.08441 |
| synergy_minus | -0.21090 | -0.23692 | -0.30172 | -0.29002 | +0.00434 | -0.12145 |

### Measured five-finger tip response at step8

True tip positions in the measured hand-base frame. Values are norms of adjusted displacement vector contrasts (mm), not differences of average travel. Raw within-arm travel means are in JSON and include ordinary baseline evolution.

| Arm | index mm | middle mm | pinky mm | ring mm | thumb mm |
| --- | --- | --- | --- | --- | --- |
| index_plus | 21.69 | 2.43 | 0.28 | 0.12 | 1.05 |
| index_minus | 24.38 | 2.07 | 0.14 | 0.11 | 6.85 |
| middle_plus | 4.03 | 25.71 | 0.80 | 4.12 | 2.96 |
| middle_minus | 2.42 | 26.85 | 0.13 | 2.70 | 1.34 |
| pinky_plus | 0.80 | 1.61 | 15.19 | 0.05 | 0.48 |
| pinky_minus | 0.11 | 0.58 | 17.56 | 2.66 | 1.68 |
| ring_plus | 0.74 | 3.27 | 0.45 | 27.62 | 0.96 |
| ring_minus | 0.45 | 1.05 | 0.37 | 29.06 | 0.35 |
| thumb_yaw_plus | 1.55 | 1.74 | 0.21 | 0.92 | 9.33 |
| thumb_yaw_minus | 0.09 | 1.65 | 0.19 | 0.21 | 9.87 |
| thumb_pitch_plus | 1.48 | 0.26 | 0.27 | 0.42 | 10.15 |
| thumb_pitch_minus | 0.73 | 1.44 | 0.26 | 0.18 | 13.94 |
| synergy_plus | 25.86 | 30.97 | 15.86 | 29.79 | 8.58 |
| synergy_minus | 27.79 | 32.82 | 18.92 | 33.61 | 19.81 |
