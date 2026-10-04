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
status: UNCLEAR
run_id: early-hold-intervention-s213
---

# Can actions manipulate interaction information useful for keeping a grasp?

Result: 494 complete early-hold trials; A UNPROMISING, B UNCLEAR for insufficient strata despite 49.6% GT I error improvement; C not executed.
Decision: Close this four-step residual contract; retain GT I prognosis, without more seeds, fitting or online selection.

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

Before any C fitting, planned residual input normalization is explicitly
train-only mean/std per native channel (inactive channels have scale floor.001).
Wrist metres and finger normalized ranges therefore do not receive one common
.1 divisor. This label-free unit normalization is fixed before formal model
results; it does not change the actual simulator dose or create another arm.
Scorers retain26 conditional slots and six outcome heads; GT/predicted I use
first14 slots, direct action18 slots, H all zero. Stored unclamped regression
scores define ranking/AUC; failure scores are not claimed calibrated probabilities.

Implementation `tools/run/probe_early_hold.py` captures dataset/collection
manifest and all executed code hashes. The A decision uses full randomized
cohort; B/C generalization uses disjoint physical environments. Primary rank
averages continuation-retention32 and height-or-proxy-loss-failure32, each
macro over ALL nonempty observed pair strata. The separate coverage gate
requires three strata with at least10 pairs per primary head. Small strata
can inflate the macro gain; they are not silently dropped from the registered
metric. No new labels used for H preprocessing.

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
outcomes. Formal execution and results follow below.

## Execution and provenance

User-selected ref4 SHA256:
`f6ae8751ffea0e73f7cbfa41787e15ddeae70ded348ffd444710190b0b6c1906`.
Collection code `eb16b2f`, gate code `d51ee60`; full SHAs and executed file
hashes are in the immutable manifests. Collection command:

```bash
bash src/task/cm-interaction-oracle/tools/run/run_intervention_collection.sh early-hold-intervention-s213 168 6 213 214 1200 early-hold
```

Gate command used GPU6/cuda:0, seed215, project TMPDIR and a600s cap:

```bash
python src/task/cm-interaction-oracle/tools/run/probe_early_hold.py --dataset outputs/cm-interaction-oracle/early-hold-intervention-s213/interventions.pt --run-dir outputs/cm-interaction-oracle/early-hold-gates-s215 --seed 215
```

One1,008-episode cohort yielded494 assignments, arm counts
59/59/79/75/78/76/68; all494 at risk with six pre-hold observations and complete
32-step windows. Every wave's region/geometry/headroom unique counts agree:
83/71/86/83/88/83. Thus the extra geometry/headroom screens did not remove
environments reaching this early-hold decision region in this cohort.
Each nonzero arm's executed dose ratio is1.0 (float rounding≤1.2e-7);
zero-arm and post-chunk action error0. No assigned survivors discarded.

Dataset SHA256 `a068bedbe2a858f1b42ec97910e86ec2cf3eab4499061c8dabd3d6edde249357`.
Outputs: `outputs/cm-interaction-oracle/early-hold-intervention-s213/`
(packet, manifest, result, complete episode summaries, run.log),
`outputs/cm-interaction-oracle/early-hold-gates-s215/`
(manifest, immutable result, randomized_arm_result, diagnostic, statistical_audit,
run.log). Compact reviewed results are also retained beside this card in
[results](P-20261004-early-hold-intervention-results.json).

The cohort has214/494 any1..32 failures:212 physical height losses and99
six-step proxy-contact losses, with overlap.98 fail within1..8, and all98
also satisfy failure in9..32; there are zero early-only recovered failures.
The target is therefore not driven only by a net-force label restatement.
It still includes predicting persistence of an already-started failure.

## Registered A/B results

**A UNPROMISING:** short-contact adjusted arm range5.684pp versus required10pp,
within-family permutation tail0.1335 versus required≤.10. I14 family max-tail
0.5005, maximum partial explained variance2.736%; no distance-axis/later-task
aligned arm passes. Continuation family tail0.0845 (held16), all32 failure
tail0.2505; neither establishes the required action→retention-I control chain.
This is not an equivalence test proving zero physical effect.

**B UNCLEAR:** disjoint387 train/107 test trials from111/28 physical environments.
Train motion counts238/124/25; test64/36/7. Test late failure41, nonfailure66,
but each primary head has only TWO sufficiently supported strata instead of
the required three. Coverage blocks the gate even though fitted gains are large.
Same6054 parameters and initialization for both scorers:

| Metric | H | H + GT I8 |
| --- | ---: | ---: |
| Normalized six-head test MSE | 0.355304 | 0.179191 |
| Continuation-retention32 MAE | 0.138674 | 0.077525 |
| Continuation-failure32 MAE | 0.220038 | 0.168772 |
| Continuation-failure32 AUC | 0.899113 | 0.949372 |
| Registered primary macro pair accuracy | 58.46% | 77.33% |

Normalized error improves49.57%; registered rank improves18.86pp, but tiny
strata with1–2 pairs count equally in this macro. Independent reviewer and
root recompute the existing ≥10-pair strata descriptively:87.76%→96.32%,
8.55pp. This does NOT replace the original denominator or repair the gate.
Pairs compare different randomized arms at different factual states; this is
prognostic ordering, not same-state candidate action ranking.

Posthoc saved-prediction sensitivity (no refit):92 test trials have not failed
in1..8,26 of them subsequently fail. Their normalized error0.303940→0.185413
(39.0% gain), failure AUC0.86538→0.92016. GT signal therefore is not confined
to already-failed cases. This subset conditions on post-treatment survival;
it is descriptive prognosis, not a replacement causal cohort or registered gate.

**C not executed.** Neither an I predictor, direct critic, mediated scorer,
candidate selector nor policy training was launched. Overall Probe UNCLEAR
reflects B's limited coverage; A's valid local negative remains UNPROMISING.

## Independent engineering review and root attribution

Read-only reviewer `ref5_engineering_review` checks actual packets, source and
saved predictions. It confirms I/continuation/details reconstruction, contact
loss across step8, zero environment split overlap, train-only preprocessing,
matched6054 parameters/init, current-only nuisance, correct I distance8..12
and continuation target18/20 indices, executed code hashes and C gating.
No result-invalidating engineering defect was identified. It independently
flags the tiny-stratum macro issue; root reproduces that sensitivity and does
not alter registered results. Audit tool `tools/audit/audit_early_hold.py`
independently loops factual labels, verifies dose/timing/split/hash and saved
metrics; CPU is used for these file/statistical checks without inference.

Root judgment: changing the decision phase exposes real retention/failure
variation and preserves GT I prognostic information. The missing link is
action controllability under this four-step feedback residual operator and
I8 readout, not evidence that interaction information is inherently useless.
Current data cannot distinguish cancellation by frozen feedback, response
timescale, heterogeneous effects or insufficient measurement/dose. Fitting a
larger Cm would bypass this unresolved prerequisite rather than resolve it.

### Closing Decision Note

- Question: does this early-hold action/I/continuation contract justify C?
- Evidence: effective randomized dose and494 full windows; A fails both
  registered control alternatives, B has strong error/prognostic signals but
  only2 supported strata; independent engineering review passes.
- Root action: stop this local contract and preserve the cohort. No extra
  seed, arm/dose sweep, PCA/MLP expansion, feature search or best-arm selector.
  Do not infer core Cm failure. The next substantive question must distinguish
  a retention-relevant action/operator timescale from feedback cancellation;
  another predictive fit or more B support alone cannot fix failed A.
- Cost/stopping: oneGPU6, main simulation362.6s + smoke54.8s + models2.31s;
  all new run artifacts<18MiB, neural peak37.1MB (simulation memory separate).
  All processes completed, GPU6 idle. No additional cohort authorized by this
  card after failed gates; reuse factual trajectories for a new decision only
  if it separates a concrete control hypothesis at lower cost than simulation.
- Boundary: Mission/Campaign/claim unchanged; no external authorization needed
  for this closure. Trained-policy matched Cm-on/off utility remains OPEN.

## Deferred evidence and completion

Three-reference coverage, multi-seed validation, true hand-object contact/slip,
same-state candidate value and trained-policy causal utility remain deferred.
They are not supplied by this single-cohort prognosis. More support alone is
Evidence for B while A fails, so no extra collection merely to pass B.
Formal conclusions await Validation; this card only records Probe judgments.

All21 Task tests pass, including early-hold eligibility, lost-contact boundary,
early-failure preservation, environment isolation and primary rank semantics.
`tools/verify.py --changed` PASS for both the branch and staged result/audit
changes; generated index current and owned whitespace checks pass.
Completed source/gate runs are pinned to their
original commits; adding this audit/result record does not rewrite them.
