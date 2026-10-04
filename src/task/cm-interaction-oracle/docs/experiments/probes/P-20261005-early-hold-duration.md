---
schema: ref2dex.probe.v2
probe_id: P-20261005-early-hold-duration
experiment_id: P-20261005-early-hold-duration
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: actual execution commit in manifest.json
claim_id: C3
hypothesis_family: HF-hold-duration-response
probe_index_in_family: 1
seed_pool: probe
seeds: [217, 218, 219]
decision_changed_if_positive: retain a demonstrated duration-sensitive retention operator for later action-conditioned Cm work
decision_changed_if_negative: stop duration extension without network fitting or additional seeds; distinguish clipping, insufficient support and valid local negative
status: PLANNED
run_id: early-hold-duration-s217
---

# Does longer feedback-residual execution change grasp-retention information?

Result: Pending randomized K4/K8/K16 physical response; no learned model planned.
Decision: Test duration as the next missing action-control link before any Cm training.

## Motivation and cheapest discriminator

Decision Probe under user-selected Task ref5 and Mission C3. Prior ref4 gives
strong GT I prognosis but no registered four-step action→retention-I response.
Question: does extending the SAME seven residuals make interaction/retention
respond systematically to duration? Positive evidence changes the available
action/operator contract for later Cm; no response stops investing in this
duration extension. It does not establish global action ineffectiveness.

Initial read-only factual check of the existing494-trial packet confirms PD
offset ends after step4. Live baseline wrist offset commands change by centimetres
over the next steps, including in zero arms. Raw drift across heterogeneous
states does not establish compensatory feedback. Duration is selected over
fixed absolute targets because it changes one factor and preserves reference
tracking, feedback and action units. This experiment can establish duration
responsiveness; it cannot uniquely identify feedback cancellation as the cause.
Fixed-target execution and larger amplitudes are not included in a sweep.

## Decision Note, budget and stops

- Question: is the four-step control window too short for a retention response?
- Evidence: prior494 effective early-hold interventions, A UNPROMISING;
  GT I prognostic information present, C never fitted.
- Root action: one randomized factorial cohort with unchanged source_e260,
  seven arms, amplitudes, eligibility, physics and32-step factual window.
  K4/8/16 at30Hz corresponds to0.133/0.267/0.533s. Joint assignment occurs
  only after common eligibility; no models, PPO or selector.
- Cost: one idleGPU6,84-episode smoke once, main168env×12waves=2,016 full
  episodes, ≤1,200s main collection plus300s smoke, ≤1,800s total GPU work,
  ≤1GB new artifacts. Existing project storage125GB tmp plus other research
  outputs remains below global300GB; no external permission or claim change.
- Stop on code/input drift, nonfinite tensors, native reset/dose error,
  incomplete assigned windows or resource conflict. After main, require
  ≥30 trials per nonzero arm/duration and≥90 pooled zero trials;≥12 per
  nonzero cell in each predeclared wave half. Low coverage/clipping is UNCLEAR.
  No extra waves/seed/threshold retry after a valid failed gate.

## Common decision region and execution

Same pinned self-trained source_e260 SHA16fd261b… and three canonical airplane
references as ref4. Early-hold requires10 history observations, lift≥3cm and
contact proxy for6 consecutive current states, surface proximity<6cm,
reference AND rollout remaining>33steps, ALL seven candidates current
headroom. All assignment screens precede treatment; no step8/16 survivor
selection. Full-batch resets only between waves; terminal trials retained.

Draw uniformly among21 duration×arm cells after eligibility. Zero residuals
at all three nominal durations are the identical physical operator, so their
data are pooled for effect estimation; duration tags remain in raw packets.
Nonzero residuals are wristx/z±.01m and active finger synergy±.1 native action
units, corresponding to±5% of physical joint range (native coupling preserved).
Wrist residual is a per-step PD offset relative to current q, not an absolute
world-position target. For its assigned K, each trial uses
clipped pi(actual_current_obs)+delta, then baseline feedback. Save all32 actual
and baseline actions, PDtargets and measured72D post-step trajectories.
Clipping during treatment is audited, not filtered after assignment.
All actual assigned windows/early failures enter the analysis.

## Fixed physics analysis before any formal outcomes

All measurements use common elapsed times across treatments. Primary short
contact fraction is steps1..16; continuation retention is17..32. Independently
report physical height loss (lift<2cm) in17..32 and1..32, combined six-step
proxy-loss failure, with contact-loss runs crossing the step16 boundary.
GT I14 readouts at4/8/16/24/32 retain the old force/proximity representation;
I16 is the secondary control readout. No aggregate RTG or advantage label.
Net-force contact is not certified hand-object pairing, friction or slip.

Use current-state nuisance adjustment from the reviewed ref4 audit: current
object pose, q, body positions, force norms, hand root, phase and tick;
blocks are wave×motion×reference-phase-quarter. No future states or action
feedback enters nuisance. Fit18 nonzero duration/arm indicators against
pooled zero, with rank-aware residualization. Within-block joint-cell label
permutation1,999 times atseed219 reports partial explained variance and
family-max tails. Current nuisance remains an exploratory precision device,
not proof of exact state matching or formal causal mediation.

Predeclared retention-control signal requires ALL of:

1. Primary short-contact omnibus permutation tail≤.10, a K16 arm-minus-zero
   magnitude≥.10, and same-arm magnitude growth K16 versusK4≥.05.
2. Its signed K4/K8/K16 contrasts are approximately monotonic: K4≥−.02,
   K8≥0 and K8≥K4−.02, K16≥K8−.02 after multiplying by the K16 sign.
3. Same arm has aligned continuation contact change≥.05 (its family tail≤.10)
   OR opposite physical height-failure change≥.10 (its family tail≤.10).
4. The predeclared first6/last6 wave halves each have short-contact change
   ≥.03 in that same direction, with the selected continuation quantity in
   the aligned direction. No choosing wave subsets after outcomes.
5. Support and all execution checks pass. Each nonzero cell's signed achieved
   action dose ratio≥.90, zero/post-K action and PD offset exactly zero.

Signal can be beneficial or harmful: harmful duration response proves a
local action-sensitive interaction pathway but is not a useful selector or
policy improvement. More force alone cannot pass. Report all arm/duration
coefficients, including failures; no raw best-arm promotion. Secondary I14
family effect without retention alignment is descriptive, not gate rescue.
If this gate passes: PROMISING for this local physical operator, retain its
direction/time contract for a later Cm Decision. Valid failure: UNPROMISING;
support or execution failure: UNCLEAR. No neural training follows automatically.

## Response and compensation diagnostics

For every duration/arm save common-time measured hand displacement, object
motion, I14 and retention curves. Baseline action drift along assigned
residual channels is a POST-TREATMENT measurement: compare its adjusted
randomized contrasts against zero and inspect return after release. Negative
projected drift is consistent with compensation but not sufficient to prove
baseline feedback causes lack of task response. K16 may still be too brief;
remaining directions, fixed-target operators and true slip remain untested.
No event-aligned survivor subset or additional fitting is permitted.

## Artifacts and deferred evidence

Task collection tools preserve old default K4. New per-trial duration packets
use schema v2 and explicit duration metadata; old fitting/audit tools are not
used on this packet. Outputs `outputs/cm-interaction-oracle/<run_id>/`, bounded
logs/cache under project tmp, no videos/checkpoint overwrite. Before launch
capture checkpoint, all three actual motion files, configs and code hashes.
Simulation/frozen actor run GPU; file/label/dose/permutation analysis CPU.
Code commit and actual launch manifests remain the source of provenance.

Formal multi-seed validation, exact-state candidate comparison, certified
paired contacts/slip and trained-policy Cm-on/off utility remain deferred.
One randomized cohort is a mechanism Probe. No formal conclusion or Mission
completion follows from dose-response, predictive information or engineering tests.

## Pre-main engineering smoke

`early-hold-duration-smoke-s11`, code33303d9,84 complete episodes/40 assigned
full windows,56.0s simulation. All three durations present; factual action
equals clipped live baseline+assigned residual for exactly K steps, then
baseline, and post-K PD offset is zero. All represented nonzero cells have
signed dose1.0. This small debug sample is not used for scientific effect
assessment or protocol tuning. Independent reviewer confirms assignment,
native units/coupling, timing, eligibility, horizon and reset contract.
