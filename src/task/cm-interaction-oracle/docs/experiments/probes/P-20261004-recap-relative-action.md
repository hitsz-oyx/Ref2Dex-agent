---
schema: ref2dex.probe.v2
probe_id: P-20261004-recap-relative-action
experiment_id: P-20261004-recap-relative-action
date: 2026-10-04
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: actual execution commit in each manifest.json
claim_id: C3
hypothesis_family: HF-relative-action
probe_index_in_family: 1
seed_pool: probe
seeds: [203, 204, 205, 206]
decision_changed_if_positive: add predicted consequence controls only after reliable labels and deployable K1 action information
decision_changed_if_negative: stop critic and Cm expansion on this label/data contract; diagnose coverage versus implementation
status: UNCLEAR
run_id: recap-relative-action-s203-s204
---

# Can relative task outcome supervise a deployable action critic?

Result: G0 label stability failed (train34.7%, test39.1%); counts/support and V-versus-zero checks pass; G1 not executed.
Decision: retain the relative-outcome direction but stop critic/Cm expansion on this unstable discrete label contract; no action-information or policy-utility conclusion.

## Decision question and cheapest method

Following Task ref1, distinguish whether existing noisy episodes can support
reliable local physical advantage labels, then whether current action helps
predict them beyond history. This is a Decision Probe toward the root Mission's
Cm-on/off policy utility. Absolute RTG prediction did not answer this question.
Reuse all four complete source_e260 pre-step shards; no new interaction, Cm
fitting or broad network search. Source order is e/f/g/h, matching the audited
historical split. Original episode summaries contain two successes, both later
drop; do not label whole successful/failed episodes uniformly.

The primary RECAP paper learns MC state value and n-step advantage; our binary
action critic is an adaptation, not replication of its distributional value
model or advantage-conditioned policy. Primary source:
[π*0.6 / RECAP](https://arxiv.org/html/2511.14759v2).

## Fixed label protocol

- Task reward `r = held/10 + lift_progress/5 + stable`: existing reward components
  only, normalized by existing bounds; stable already includes drop events.
  Base/approach shaping is excluded. Stable reward uses a 30-step tracker,
  while accepted episode success uses 45 steps; report rather than equate them.
- H: ten factual pre-step observations, state55 + previous action18 + context435
  + raw progress1 + known episode noise1. All controls receive noise regime;
  it is fixed per episode and otherwise a confound for action norm/outcome.
- Train V against complete same-gamma MC task return (raw units, standardized
  MSE). NOT a truncated 32-step target. State GRU48, head64, AdamW2e-3,
  weight decay1e-5, 16 epochs, batch512, clip2.0; 128 evenly spaced fit rows
  per episode, no episode length weighting. No fit/seed/epoch retry on failure.
- Outer split: historical identifier20261004, sorted `(namespace0,run,episode)`
  then shuffle: 90 train/22 test episodes. Three inner motion/noise-stratified
  episode folds. Each OOF label's current and future V use the same model
  excluding that entire episode. Test uses the mean of the three train-only
  fold models. Independent fits use base seeds203/204 and fold offsets0/1/2
  (actual initialization seeds203–206); all are Probe seeds.
- Query rows: 128 evenly spaced per episode, full history and four valid
  recorded actions. Horizon32 task rewards plus same V endpoint bootstrap,
  discount from source gamma. Include last reward, stop at terminal, zero
  terminal bootstrap, never cross reset. Preserve factual previous action.
- Train-only 30/70% quantiles per motion × 8 reference-phase bins × noise;
  require32 train rows, no sparse-bin fallback. Negative threshold≤-0.05,
  positive≥0.05 in normalized task return units. This amplitude floor prevents
  zero/tiny advantage quantiles from manufacturing supervision. Mid labels0
  are excluded from binary critic training. Primary labels average two V fits;
  individual fit labels separately assess stability using the same recipe.

## Predeclared gates

G0 requires all of: individual fit-label agreement ≥75% on their union of
nonneutral labels, separately train/test; mean held-out V MC MSE lower than
zero-V MSE; ≥500 train and ≥100 test rows of EACH class; EACH test class
supported by ≥12/22 episodes; no episode >15% of either test class. Report
real 32-step nonzero task-reward coverage, labeled coverage, MC scale and V
fit quality separately. Agreement includes neutral/extreme disagreements.
Insufficient labels or unreliable V yields UNCLEAR, not proof that RECAP fails.
Do not lower thresholds, cherry-pick a V seed or proceed to critic if G0 fails.

G1 only after G0: identical ActionCritic architecture, initialization203,
optimizer/16 epochs, train rows and episode/class-balanced loss. Arms H with
zero standardized action, HaK1 (current action right-aligned in four slots),
HaK4_realized (recorded feedback actions; privileged diagnostic). No later
action is an input to K1. All action normalization uses common training rows.
Report BA/AUC, individual episodes, within-motion/phase/noise held-out action
permutation, and episode bootstrap (2,000 draws, descriptive only).

Deployable K1 passes only if BA and AUC each exceed H by≥0.03, action permutation
reduces BA by≥0.02, paired episode BA-delta CI95 lower bound>0, and average
episode BA gain remains positive excluding preidentified largest-|MC mean|
held-out episode. K4 alone cannot authorize G2: realized future feedback is
not an open-loop candidate. Passing K1 is PROMISING conditional predictive
information, not a causal action-treatment claim. A failing G1 is UNPROMISING
for this fit/data contract, not formal refutation of action-quality learning.

G2 predicted E/I contribution and G3 real closed-loop utility are deferred
unless earlier gates pass. They require new explicit same-input/matched
controls, not information claimed from deterministic consequence features.

## Resources and stopping

One idle physical GPU6 (`CUDA_VISIBLE_DEVICES=6`, process cuda:0), two CPU
threads, ≤30 minutes cumulative model execution including smoke, ≤4GB new
artifacts. CPU is used only for file/label statistics and unit tests; model
fitting/inference uses GPU. Unique outputs, no checkpoint overwrite or new
simulator. Timeout main job at1,740s. Stop on nonfinite values, invalid episode
joins, OOM, resource conflict or deadline. Tiny smoke one epoch/eight rows
per episode is engineering only and cannot advance gates.

Implementation: `src/task/cm-interaction-oracle/tools/run/probe_relative_action_critic.py`
and `src/task/cm-interaction-oracle/src/relative_action.py`. Artifacts:
`outputs/cm-interaction-oracle/<run_id>/{manifest.json,result.json,labels.pt,value_models.pt,run.log}`.
The existing outputs symlink resolves to the declared external artifact root
`/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/cm-interaction-oracle/`,
inside authorized ai_ws. Preserve the symlink and all existing artifacts.
Manifest records actual commit, script/module/source hashes, split, parameters;
labels preserve source row/step keys and current/endpoint values for audit.

## Decision Note

Question: can sparse continuous hold/drop evidence support relative action
supervision before spending on I prediction or G architecture?
Evidence: absolute G gains depend on one episode; only two success/drop episodes;
known episode noise can confound actions; labels need cross-fit calibration.
Root choice: fixed task-reward G0 then conditional K1 G1, cheapest existing-data
gate. Cost: one GPU≤30min, zero interaction; failure stops critic/Cm expansion,
positive supports only a next matched predicted-consequence probe.
Mission/claim/resource boundaries remain unchanged; no new authorization needed.

## Limitations / future evidence

Only four source runs and112 episodes, few local successful holds. Other policy
observation missing from H may make action informative without causal treatment.
Episode-fixed noise is conditioned/stratified, not same-state randomized action
counterfactuals. Cross-episode score classification does not prove within-state
candidate ranking. MC calibration and the 0.05 amplitude guard are exploratory
choices. Distributional V, alternate horizons/rewards, formal multi-seed
generalization, candidate ranking and Cm-on/off trained policy comparison are
future evidence, not an automatic sweep after negative results.

## Completed results and engineering audit

Execution commit `9dd503d6374b430da029967f8f1fc657721cd424`, run
`recap-relative-action-s203-s204`, gamma0.99; six V models from two base fits
and three folds. 59,903 eligible ten-step histories, 11,520 train/2,816 test
queries, 90/22 whole episodes, all96 motion/phase/noise threshold strata.
Duration14.77s, peak allocated GPU330,649,088 bytes. Smoke at `58a4830` used
one epoch/eight queries per episode (4.51s); it is engineering only. No new
simulation, critic training, consequence fitting or threshold/seed retry.

| G0 diagnostic | Train | Test | Predeclared requirement |
| --- | ---: | ---: | --- |
| V-fit extreme-union label agreement | 34.72% | 39.15% | ≥75% each |
| Positive labels | 3,417 | 840 | ≥500 / ≥100 |
| Negative labels | 3,453 | 710 | ≥500 / ≥100 |
| Positive-label episode support | 90 | 22 | ≥12 test |
| Negative-label episode support | 89 | 22 | ≥12 test |
| Largest episode share of positive labels | 3.72% | 9.64% | ≤15% test |
| Largest episode share of negative labels | 2.58% | 6.90% | ≤15% test |
| Queries with nonzero real32-step task reward | 13.90% | 16.30% | Reported, not selection gate |

Held-out MC MSE229.58 versus zero V324.80 passes the coarse value check; it
does not establish accurate advantage derivatives or calibrated uncertainties.
Primary average-fit test positives/negatives have real32-step reward coverage
35.60%/11.41%; thus most supervised labels would come from bootstrap/V
differences in intervals without observed local task reward.

Root's read-only statistical audit distinguishes the failed discrete contract
from a blanket negative conclusion. Continuous A Pearson is0.904 train/0.978
test; excluding largest-|MC mean| test episode it remains0.952. Test
agreement is82.39% on the intersection where BOTH fits label an extreme,
but the registered union agreement is39.15%; 52.49% of that union disagrees
between neutral/extreme, and8.37% has opposite signs. No changing the
denominator or selecting only agreed labels to claim G0 passed.

On test queries with real local reward, extreme-union agreement is78.02%;
on zero-local-reward queries it is29.56%. Zero-reward A Pearson0.755 and
median absolute inter-fit difference0.255 exceed the0.05 amplitude floor.
This is not merely tiny numerical threshold jitter. The active subset is a
post-hoc mechanism diagnostic using FUTURE outcome, not a deployable filter
and not permission to train only favorable examples or redesign the gate.

Independent read-only reviewer `ref5_engineering_review` found one clear
engineering bug before the formal run: binary AUC initially included neutral
rows in ranking. Fixed at `9dd503d`, with an explicit neutral regression test;
this bug affected only unexecuted G1 metrics, never G0. Reviewer then checked
all source/module hashes, all112 episode MC/local32 targets, source/current/
future row joins, same-fold current/endpoint V, train-only thresholds and
outer-test exclusion: exact matches, reconstruction error0. All817 terminal
queries have zero discount/bootstrap. No further fatal implementation problem
found in these checks. Root independently reproduced the advantage/fold/hash
and stability diagnostics; audit tool is
`src/task/cm-interaction-oracle/tools/audit/audit_relative_action.py`.

Evidence: `outputs/cm-interaction-oracle/recap-relative-action-s203-s204/`
contains immutable execution manifest/result, query label packet, six V
checkpoints, run.log and `statistical_audit.json`. Post-run audit records the
source gamma omitted from original manifest; source hashes and run.log also
pin0.99. No rerun was needed for this metadata omission.

### Closing Decision Note

- Question: advance to action critic/Cm despite unstable advantage labels?
- Evidence: only fit-label agreement fails G0; sparse local reward and
  sizeable V-residual variation account for much disagreement; continuous
  trends correlate even without the largest test episode; engineering audit
  confirms targets and split rather than discovering a labeling join bug.
- Root choice: stop this discrete-label pipeline before G1. Preserve
  relative-outcome as a candidate; label/value reliability is the next decision
  prerequisite, not more I/G capacity. This does not close the core Cm
  hypothesis or prove absent action information. Any revised label/value
  protocol must have a new explicit hypothesis/gate, rather than relaxing this
  failed gate, selecting future-active test rows or replaying seeds.
- Cost/stopping: <20s model execution including smoke, one GPU, <4GB artifact
  limit, zero interaction; all processes finished. No evidence justifies G2/G3.
- Boundary: no new authorization required; root Mission's causal trained-policy
  Cm-on/off result remains OPEN. Formal claims and Validation are deferred.

## Verification and deferred implementation details

Task tests plus recursive index/governance tests:34 passed. Scoped
`verify_changed` over this round's code/card/README/STATE/index:PASS. Full
`tools/verify.py --changed` remains FAIL on three pre-existing example links
in `.agents/skills/domain-modeling/GLOSSARY-FORMAT.md` (ordering/billing/
fulfillment GLOSSARY paths). Unrelated skill templates were not changed.
Formal run3.8MB and smoke2.2MB; GPU6 is idle after completion. Changes are
local commits only; existing user changes/checkpoints are preserved.

The G1 branch was not executed. Before an actual G1 run, report within-stratum
metrics and the number of episodes containing BOTH classes for paired
bootstrap; per-class≥12 support alone does not imply12 usable paired episodes.
These are deferred G1 readiness details, not a reinterpretation of failed G0.
