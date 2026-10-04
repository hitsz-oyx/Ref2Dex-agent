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
status: RUNNING
run_id: recap-relative-action-s203-s204
---

# Can relative task outcome supervise a deployable action critic?

Result: pending bounded G0/G1 gates.
Decision: execute G0 first; G1 and predicted Cm are conditional, not automatic experiments.

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
