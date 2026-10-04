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
status: UNPROMISING
run_id: early-hold-duration-s217
---

# Does longer feedback-residual execution change grasp-retention information?

Result: 1006 full randomized windows; dose and support pass, but no duration-sensitive retention-I gate; hand displacement responds while I16/contact do not pass.
Decision: Close K4/8/16 feedback-residual extension after independent review; no Cm fitting, extra duration/seed or selector.

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
Independent native PD reconstruction maximum error1.10e-7; no offset- or
coupling-unit discrepancy. Smoke does not contain every cell; unit tests
exercise full joint assignment and exact K boundaries, main support gate
must still pass for all18 nonzero cells.

## Formal collection and analysis entry

ref5 guidance SHA256 `0e019f4ce34b0a837b14bef3e5306cb612119db4344c5ad2e39681ce3bf25475`;
user-owned guidance files are not modified or staged. Collection commit422e081:

```bash
bash src/task/cm-interaction-oracle/tools/run/run_intervention_collection.sh early-hold-duration-s217 168 12 217 218 1200 early-hold "4 8 16"
```

Physics/dose analysis is `tools/audit/probe_duration_response.py`, with a new
`early-hold-duration-response-s219` output directory. It audits all assigned
actions and reconstructs PD differences from URDF limits/coupling before
reporting coverage or randomized effects. Saved `diagnostic.pt`,
`response_curves.json`, manifest and immutable results permit read-only
recomputation. No supervised model fit or repeated learned inference.
Analysis tests check crossing step16, preservation of early failures,
both directions of duration response, late alignment and both wave halves.

## Completed execution and provenance

`early-hold-duration-s217` completed at code422e081, all2,016 native episodes
and1,006 assigned32-step windows; simulation731.39s. Collection manifest
captures8 pre-run hashes (collector/contract, checkpoint, two configs, three
actual linked motion files). Dataset SHA256:
`3bd64f6c9dccb8c6b7c87cf291e341b6b15b2fc35ea2b0d9e326773048e7e135`.
All input hashes and code hashes independently match current unchanged files.
New guidance files remain user-owned and unmodified.

Cell counts, columns zero/x+/x−/z+/z−/finger+/finger−:

| Duration | Counts |
| --- | --- |
| K4 | 59 / 40 / 38 / 50 / 51 / 51 / 57 |
| K8 | 46 / 47 / 38 / 49 / 43 / 52 / 54 |
| K16 | 43 / 59 / 32 / 51 / 54 / 50 / 42 |

Minimum nonzero cell32, pooled zero148, minimum cell in either predeclared
wave half13: coverage passes without extra collection. Motion counts598/309/99
describe the eligibility-selected cohort, not motion-wide grasp performance.
Current native risk and pre-hold screens pass for all assignments. Actual
clipped action reconstructed exactly, post-K/zero action and PD offset zero,
all nonzero dose ratios1.0 within float error1.2e-7; native PD mapping max
error1.12e-7. No clipping explanation or missing-window exclusion is needed.

Physics analysis command (code305e635, CPU statistics only):

```bash
python src/task/cm-interaction-oracle/tools/audit/probe_duration_response.py --dataset outputs/cm-interaction-oracle/early-hold-duration-s217/interventions.pt --run-dir outputs/cm-interaction-oracle/early-hold-duration-response-s219 --seed 219
```

Completed in2.81s; no learned model, critic, policy or selector executed.
Outputs under the two run IDs contain immutable manifests/results, factual
packet, complete episode summaries, diagnostic, engineering/statistical audit,
all response curves and copied logs. Compact reviewed evidence is retained in
[results](P-20261005-early-hold-duration-results.json) beside this card.

Failure variation is real:478 any1..32 combined failures,475 with physical
height loss,225 with six-step proxy loss (overlap).383 fail in1..16;
7 early-only failures recover before the late window. All7 remain in the
packet with all32failure=1 and late failure=0, correctly distinguishing
"no late failure" from "never failed". Late combined failure471, height468.

## Registered response result: UNPROMISING

Current-state nuisance rank152,102 wave×motion×phase blocks, treatment rank18
for full data and each wave half.1,999 conditional joint-cell permutations.
Registered retained-response candidate list is empty.

| Readout family | Max partial explained variance | Exploratory family-max tail |
| --- | ---: | ---: |
| Contact fraction1..16 (primary) | 2.825% | 0.1645 |
| Contact fraction17..32 | 1.899% | 0.5810 |
| Physical height failure17..32 | 2.578% | 0.2440 |
| Combined failure17..32 | 2.660% | 0.2190 |
| Physical height failure1..32 (secondary) | 3.125% | 0.0850 |
| I14 at16 (secondary) | 3.098% | 0.5720 |

Maximum K16 short-contact magnitude6.624pp, below10pp; primary tail also
fails≤.10. Adjusted short-contact effects against pooled zero, in percentage
points (all directions shown, not selected best arms):

| Arm | K4 | K8 | K16 |
| --- | ---: | ---: | ---: |
| wrist x+ | −4.10 | −1.90 | −0.74 |
| wrist x− | −3.40 | +1.58 | +1.88 |
| wrist z+ | −5.78 | −0.27 | −6.62 |
| wrist z− | −2.23 | +2.21 | −0.26 |
| finger+ | −5.76 | +1.76 | −4.90 |
| finger− | +1.05 | −2.03 | +0.08 |

Neither a consistent duration-growth pattern nor aligned late retention/height
effect passes. The secondary all32 height family has weak tail0.085,
including K16 wristz− adjusted+15.24pp versus zero (K4+8.01pp, K8+0.72pp).
This secondary harmful response is not ignored, but is nonmonotonic and does
not establish the required interaction mediator. Tails are within-family,
not corrected across six families; no formal significance or best-arm policy.

## Physical response and feedback interpretation

Measured hand body-centroid x displacement at the SAME step16, adjusted
against zero, confirms motion changed with the duration of wristx+:

| Duration | wristx+ vs zero | wristx+ minus wristx− |
| --- | ---: | ---: |
| K4 | +4.51mm | 9.35mm |
| K8 | +10.47mm | 24.46mm |
| K16 | +21.92mm | 43.56mm |

These are measured body positions, not inferred PD motion. K16 wristx
plus/minus displacement remains16.96mm atstep32, after release, rather than
being completely erased. Thus "nothing was physically executed" is excluded,
and a longer/larger hand response alone did not create a registered retention-I
response for this candidate set.

Some compensation is plausible: K16 wristz+ atstep16 has only+2.93mm handz
effect against zero; its adjusted baseline action drift is−1.61 times the
assigned+.01m residual at that time. The analogous K4/K8 end-K ratios are
−0.52/−0.66. This post-treatment command response is consistent with the
baseline opposing perturbation, and z+ displacement plateaus. However,
duration changes cumulative dose and time since release, native wrist targets
are relative to current q, and hand/object state feeds back into pi. These
descriptions cannot uniquely establish cancellation as the cause of failed I
control or rule out a fixed-target operator. No event-selected subset is used.

## Independent review and root judgment

Read-only `ref5_engineering_review` confirms actual21cell/half support, action
and PD doses, all pre-run/code hashes, independent numpy labels and whole/half
joint OLS, all response curves and baseline projection. Coefficients match
to≤3.4e-14; curve differences only float32 construction≤4.4e-8. No engineering
defect invalidating the local negative was found. Root independently
reconstructs labels/I16 (I difference≤2.39e-7) and direct joint OLS (≤1.49e-14),
checks all hashes and confirms seven recovered early failures are retained.

Root attribution: extending the existing feedback-residual directions to16
steps does increase a real hand response, but does not expose a stable
retention-sensitive I14 pathway. Feedback opposition remains a plausible
contributor, particularly for z+, not a resolved causal explanation. This
shrinks the viable local contract; it does not show interaction information
is inherently uncontrollable or that a Cm predictor failed. No predictor was
trained. GT I's prior prognostic value remains, with its original Probe limits.

### Closing Decision Note

- Question: does extending this feedback-residual duration justify Cm fitting?
- Evidence:1,006 windows, all execution/support checks pass; hand displacement
  responds to K, but primary contact/I16 and aligned continuation gates fail;
  independent engineering reconstruction passes.
- Root action: close this K≤16 feedback-residual extension as UNPROMISING.
  No additional seed, duration/amplitude sweep, network fit or selector.
  Preserve the measured co-response/compensation traces. I14 remains useful
  for prognosis; its role as a control mediator under these directions is
  unestablished. Do not promote it into world-model action selection merely
  because GT prediction was strong.
- Cost/next/stops: GPU6 main731.4s plus56.0s smoke; CPU analysis2.81s;
  total new run artifacts<37MiB. All processes finished, GPU6 idle. Any next
  control Probe needs a genuinely different, justified operator or retention
  direction that separates competing mechanisms; another duration fit or
  supplement to old B coverage does not change this decision. Fixed absolute
  targets are untested and would require their own matched zero-operator and
  safety/dose contract, not a posthoc extension of this card.
- Boundary: Mission/claim/Campaign unchanged, no external authorization needed
  for closure. No core Cm refutation or trained-policy utility conclusion.

## Verification and remaining limits

All25 Task tests and `tools/verify.py --changed` (base previous delivery
d7a72df) pass. Old K4/default tools remain compatible; mixed-duration packets
are explicit v2. Full-run provenance and independent audit agree with saved
statistics; only card/result/index/state records are added after execution.
Exploratory permutation conditions on many coarse blocks and current linear
nuisance; repeated-environment dependence, uncertain true contact/slip and
absence of exact counterfactual branching limit inference. A failed gate is
not a confidence-bound equivalence test proving no smaller effect. Single
cohort/wave halves are not independent-seed Validation.

## ref5 completion audit

The selected guidance is checked against current artifacts after delivery,
not just the intended plan. The preceding goal turn made research progress:
it implemented and executed the requested duration comparison and obtained
a reviewed negative that stops its conditional Cm stage.

| Guidance requirement | Authoritative completion evidence |
| --- | --- |
| Physics response before any new Cm/S/network/PPO fit | Analysis command/manifest and source305e635; result `neural_models_executed=false`; only frozen source actor used in GPU collection. |
| Same early-hold region; K4/8/16 comparison or changed execution operator | Actual v2 packet contains all21 jointly randomized arm/duration cells; all1,006 decisions satisfy lift≥3cm/contact proxy/pre-hold≥6; all windows complete. The duration alternative was executed; fixed absolute targets remain untested. |
| Directly measure interaction and retention/loss/drop | Packet-derived I16, common1..16/17..32 contact, independently separated physical height/proxy failures, all32 outcomes and all saved response curves. Independent labels and native PD reconstruction agree. |
| Check dose growth, monotonic response and repetition | Registered same-direction K contrasts and first6/last6 wave coefficients, full/half ranks18, minimum cell32/13 and pooled zero148; no qualifying candidate, short-contact tail0.1645/I16 tail0.572. |
| Enter Cm only after action-sensitive interaction response | Registered gate is valid UNPROMISING; the positive-only prerequisite is false and no model/selector stage was executed. More seeds, B support or network capacity cannot substitute for it. |
| Interpret control versus prediction, with engineering review | Actual-dose/hand-motion evidence, independent full-data review and root jointOLS reconstruction; retain prior GT prognosis and distinguish possible compensation from an identified cancellation mechanism. No core Cm or policy-utility conclusion. |

Collection and analysis manifests are COMPLETED, executed hashes/dataset SHA
match, no Python process for either run ID remains and GPU6 is idle.
ref5's requested physics Decision has a verified negative outcome; learning
and policy work remain conditional future research, and Mission utility stays OPEN.
