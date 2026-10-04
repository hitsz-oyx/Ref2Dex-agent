---
schema: ref2dex.probe.v2
probe_id: P-20261004-early-hold-intervention
experiment_id: P-20261004-early-hold-intervention
date: 2026-10-04
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: actual execution commit in manifest.json
claim_id: C3
hypothesis_family: HF-action-intervention
probe_index_in_family: 2
seed_pool: probe
seeds: [213, 214, 215, 216]
decision_changed_if_positive: train an action-conditioned retention I predictor only after randomized contrast and GT retention information gates
decision_changed_if_negative: stop this early-hold short-residual contract after review, with no seed or encoder retry
status: RUNNING
run_id: early-hold-intervention-s213
---

# Can actions manipulate interaction information useful for keeping a grasp?

Result: Pending early-hold randomized intervention and retention information gates.
Decision: Address the previously untested hold/drop regime before any further consequence predictor training.

## Motivation, hypothesis and cheapest discriminator

Decision Probe for Mission C3 under Task ref4 (explicit user selection after
the initial ref3 continuation request). The completed pre-lift contract
had no at-risk drop trials; localized GT retention information and weak object
rotation responses do not establish a controllable task-relevant I. New
hypothesis: once the object is lifted/contacted, randomized short actions can
change the interaction variables which predict subsequent retention/drop.
This is a different decision region, not a re-run of the failed pre-lift fit.
The user-owned ref4 requests this A/B/C sequence; the guidance file itself
is not edited or included in root's selective commits.

Cheapest sequence: smoke42 environments once, then168 environments×6 waves
(1,008 complete episodes maximum) to obtain the early-hold cohort. Check
randomized I/task contrast and fixed-capacity H-versus-GT I prognostic value.
Only A+B positive permits action-conditioned I training/direct/Cm comparison;
only useful predicted ranking permits an online candidate Probe. No online
GT oracle, state copying, RECAP target, extra encoder or I feature search.

## Decision Note and resource boundary

- Question: was the pre-lift gate testing a phase outside grasp retention control?
- Evidence: all320 previous decisions below8.74mm lift, zero drop-risk; later
  complete episodes nevertheless contained10545-step holds and76 later drops.
- Root choice: early-hold randomized residuals with unchanged actor/dose/physics
  and fixed E/I representation. Contact and drop become primary; Δz is an
  explanatory measurement, not mixed into an aggregate selection scalar.
- Cost/stopping: idleGPU6, ≤30min cumulative simulation/model work, ≤4GB new
  artifacts, ≤1,008 main episodes plus42 smoke. Stop on invalid dose/reset,
  nonfinite data, truncation, resource conflict, low coverage or time cap.
  No extra cohort/seed/threshold search after a failed valid gate. Need ≥196
  assigned trials, ≥20/arm and meaningful retention/drop variation before
  model comparison. Coverage failure is UNCLEAR, not action ineffectiveness.
- Mission/claim and Campaign boundaries unchanged; no new external permission.

## Random assignment and eligibility

Same frozen self-trained source_e260 SHA16fd261b…, three canonical airplane
references, deterministic feedback actor, no episode noise. Same seven arms:
zero, wristx±.01, wristz±.01, finger(6/8/10/12/15)±.1; four-step feedback
operator, then baseline actor. Native units/coupling and actual PDtarget/dose
audits preserved. Each arm is independently uniform at first eligible state.

Eligibility BEFORE assignment:10 observation history, >33 steps remaining in
BOTH reference and native rollout limits, reference-rest lift≥3cm AND net-force pair proxy for6 consecutive
current observations, body-to-object surface proximity<6cm, and current
headroom for ALL candidates. Record pre_hold_steps/current pose/history; no
future success, survival or loss conditioning. At-risk status is now present
by design, not a post-treatment subset. Full-batch resets only between waves;
assigned terminal/truncated trials retained, no outcome-based filtering.
Pair contact is net-force/proximity proxy, not certified hand-object identity,
friction or slip. Early-hold labels must not be generalized to true slip sensing.

## Outcomes and physical timing

Save the existing measured72D post-step trajectory32, planned/actual actions,
PDtargets, native observation/history/context and full episode summaries.
E12 and I14 at8 use the identical prior representation. Primary short I
contrast also reports measured net-force contact fraction over steps1..8,
derived from existing signals, not another feature encoder.

Task outcome endpoints16/32 evaluate continuation AFTER the consequence window:
steps9..16 and9..32 retention fraction, lifted/contacted held fraction and
drop (height<rest+2cm OR six successive lost-contact observations). Thus
teacher I8 does not share the first8 observations in the continuation target.
All assigned trials remain at risk from time0. The loss-contact run initializes
from observed steps1..8 and continues into9..32; prior drop is also reported
separately. This cannot make GT realization into a deployable control oracle.
Full45-step held success/later-drop retained as later descriptive outcome.
Combined failure is a height-or-contact-proxy event; physical height-loss and
proxy-loss components, first failure step and any1..32 failure are saved
separately. Already failed step8 trials are NEVER excluded; GT can predict
failure persistence, which is post-intervention prognosis, not an unobserved
pre-action causal advantage.

## Gates A/B/C

Gate A: randomized-arm control signal in I/contact. Fixed current-state nuisance
adjustment and conditional-block permutation reuse the reviewed audit method,
with seed215/1999 permutations and within-family max-tail. Report all I14 axes,
short1..8 contact fraction and late continuation outcomes separately. A useful
Probe signal requires short contact fraction adjusted arm-minus-zero range≥.10
and permutation tail≤.10; alternatively an existing body-surface-distance I
axis family tail≤.05 accompanied by aligned later retention/drop contrast
(task-family tail≤.10). Alignment means the same arm reduces distance vs zero
AND increases continuation retention by≥.05 or reduces failure by≥.10.
Force axes remain descriptive: more net force is not necessarily a better
grasp. No raw best-arm policy claim or formal statistical significance.

Gate B: grouped physical-environment train/test20%, stratified motion, seed215;
train-only PCA32 and normalization unchanged, MLP64/32/tanh/AdamW.002/decay.001,
100epochs. Matched H vs H+GT I14 (common26-slot conditional channel, zero forH),
same capacity/init/training. Six targets (retention/held/drop16/32) normalized
train only. Primary32-step retention and drop macro pair rank with fixed
within-motion/reference-phase pairs from DIFFERENT arms, plus error calibration.
GT must improve primary rank≥3pp and normalized task error≥5% over H, with
test continuation-failure positives AND negatives≥10 each and ≥3 supported
pair strata (≥10 pairs per stratum per primary head); otherwise B UNCLEAR for inadequate support or UNPROMISING for
a valid failed fit. Report per-head and cohort effects, no posthoc target swap.

Gate C only if A+B pass: H→I vs H+planned action→I, same protocol, assigned
actions only (no realized future feedback input). Require I prediction improve
≥5% and shuffled intent worsen≥5%; scorer uses environment-grouped OOF
predictor outputs, matched direct critic, and test predicted mediation must
improve primary rank≥3pp vs H and≥2pp vs direct, with intent permutation
drop≥2pp. This remains a transition MLP, not the historical frozen point-flow
model. No Cm/direct training if A or B fails. No seed/capacity/dose sweep.

## Artifacts and limitations

Task-local collection tools retain old contact-region default; early-hold is
explicit CLI. Unique outputs `outputs/cm-interaction-oracle/<run_id>/` resolve
to the existing authorized baseline output symlink; no old artifacts overwritten.
All three actual motion files (followed directory links), checkpoint/config/code
hashes captured BEFORE launch. Logs/cache in project tmp; no videos or /tmp.
CPU hashes/statistics/unit tests, GPU simulation/actor/neural fits. Reused
environment clusters stay in one split/fold; randomization diagnostic remains
exploratory given carryover/coarse nuisance adjustment and multiple families.
Final learned-policy Cm-on/off utility, multi-seed Validation and true paired
contact/slip evidence remain future stages. Local negative is not core Cm refutation.

## Engineering smoke and final pre-main contract

`early-hold-smoke-s9`, code dfd4eaf:42 full episodes,18 early-hold interventions,
18 complete32-step windows. Unique early-hold/geometry/headroom coverage all18;
no saturated-action cohort loss in this debug sample. Simulator55s. No formal
arm-effect or GT gate assessed on the debug sample. Review found a general
native rollout-limit reset in addition to reference-limit reset; added its
pre-assignment remaining-time guard before main. The pinned rollout2000
was not binding in this smoke, so no assigned window was truncated.
Before main, after user selected ref4, primary support is≥3 observed
motion/phase strata, matching three fixed references without demanding extra
phase spread. No support threshold changed using formal data or fitted test
outcomes. All formal gates remain pending.
