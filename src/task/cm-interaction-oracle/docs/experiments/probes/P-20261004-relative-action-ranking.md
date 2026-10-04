---
schema: ref2dex.probe.v2
probe_id: P-20261004-relative-action-ranking
experiment_id: P-20261004-relative-action-ranking
date: 2026-10-04
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: actual execution commit in manifest.json
claim_id: C3
hypothesis_family: HF-relative-action
probe_index_in_family: 2
seed_pool: probe
seeds: [207, 208]
decision_changed_if_positive: test matched predicted physical consequences and a small physical candidate-ranking probe
decision_changed_if_negative: stop this current-action ranking fit/data route after implementation review; no threshold or V retry
status: RUNNING
run_id: relative-action-ranking-s207
---

# Does current action improve ranking of frozen relative task advantage?

Result: pending matched continuous/pairwise G1.
Decision: run H versus HaK1 directly; evaluate frozen target ranking stability without repeating discrete G0.

## Motivation, hypothesis and cheapest discriminator

Task ref2 explicitly requests a new ranking question, preserving the two
cross-fitted advantage fits from ref1. The previous three-class contract
remains UNCLEAR; its gate/result is not modified. Hypothesis: current action
contains predictive ordering information about relative local task outcome
after controlling history and episode-fixed noise. A positive result supports
testing Cm representations/candidate decisions toward the global Mission's
causal trained-policy utility; a negative result stops this local pipeline
before extra consequence training or simulation. Classification instability
alone never established that action information was absent.

Cheapest method: reuse immutable `recap-relative-action-s203-s204/labels.pt`,
the same112 complete source episodes and90/22 split. Do NOT refit V, change
reward/horizon/quantile/threshold/seed, filter future-reward-active queries,
or select only agreed V labels. No new policy or simulator in this first gate.
Pearson0.978 does not establish rank stability: this probe actually computes
Spearman and within-stratum pair-order agreement between the two frozen fits.
These are reported diagnostics, not a new pre-training G0 veto that bypasses
the requested direct matched action-critic experiment.

## Frozen inputs and pair contract

All14,336 queries from ref1; ten factual pre-step observations H510 (state,
previous action, context, raw progress, known noise). H never contains future
outcome, future observations, V or A. Frozen target is arithmetic mean of
the two stored32-step advantages. Reward remains held/10 + lift_progress/5
+ stable; gamma0.99 and whole-episode cross-fit unchanged. Source file hashes,
every current/endpoint/query/split key, local reward and discount must match
the stored packet before training. Exact advantage reconstruction is checked.

Four sampled partners per anchor, same motion × eight reference-phase bins ×
episode noise, DIFFERENT episodes. Train and test pairs are formed separately
using only observed group keys, with seeds207/208; partner choice never sees
A, V-fit agreement or future reward. Unsupported single-episode strata yield
no pairs and are reported. Replacement can duplicate partners; pair count is
not independent sample count. Phase-bin proximity is not the same physical
state and does not certify kinematic near-neighbors or counterfactuals.

Pair loss `softplus(-(C_i-C_j)*sign(A_i-A_j))`; exact equal-target pairs are
skipped, with no magnitude/quantile threshold. Each anchor has equal pair
weight; the128 evenly spaced rows per episode keep episode row counts equal.
No nonlinear target transform, clipping or MSE objective. Targets use future
information for supervision; inference inputs do not.

## Matched arms and metrics

- H: common action GRU receives four standardized zero vectors.
- HaK1: only current action placed in the LAST slot, other slots zero.
- HaK4_realized: the recorded four feedback actions; privileged diagnostic,
  never treated as a deployable candidate. It distinguishes missing current
  action contrast from contrast appearing only in later realized actions.

Same ActionCritic: H/action GRU48, fused96→64→1; same parameter count AND
initial-state hash, seed207, fixed16epochs, AdamW2e-3/decay1e-5, clip2,
batch512 pairs, identical pair/minibatch order. H normalization uses train
query rows only; action channel statistics use train CURRENT actions for all
arms. All three models receive the same known noise observation.

Primary metric: macro motion/phase/noise-stratum pair accuracy on held-out
episodes. Correct score ties get0.5; exact target ties are excluded/reported.
Report conditional macro Spearman, global Spearman (secondary), row/pair/
stratum coverage, anchor-episode accuracy and train metrics. Report target
rank stability separately; high continuous Pearson cannot substitute for it.
All scored queries and group support remain fixed across matched arms.

Held-out action permutation stays inside motion/phase/noise (seed208), with
no training data changes. It is an input-sensitivity diagnostic; mismatched
H/action combinations are not physical counterfactuals. H scores must remain
identical under its constant-action permutation.

PROMISING only if HaK1 macro pair gain≥0.03 AND macro Spearman gain≥0.03
over H; action permutation reduces its macro pair accuracy by≥0.02; gains
remain positive against EACH of the two frozen V targets; ≥12/22 anchor
episodes improve; and pair gain remains positive after removing BOTH members
of any pair involving the preidentified largest-|MC mean| test episode.
Report paired usable episode N and2,000-draw descriptive bootstrap CI; CI is
not a hard Validation gate. K4 alone cannot authorize a deployed ranking route.

Failed gates yield UNPROMISING for this fixed fit/data/proxy-target route after
engineering review, not absent physical action information or refuted Cm.
Insufficient pair coverage/nonfinite implementation yields UNCLEAR. If K1
passes, continue to a separately specified matched predicted E/I control and
a small genuine physical candidate/outcome dataset before claiming selection
utility. Same-state/near-state candidate ordering and Cm-on/off trained-policy
utility remain required future stages, not demonstrated by these pairs.

## Decision Note and resources

Question: is the discrete boundary hiding usable current-action ordering
information? Evidence: ref1 V fits correlate continuously but disagree on
three-class labels; no critic was trained. Root choice: freeze all V/data
contracts and directly fit matched ranking, rather than re-tune V/thresholds.
Success invests in predicted consequence/physical candidate gates; failure
stops this fit contract with independent implementation review. Global Mission
and claim unchanged; no new external authorization needed.

One currently idle physical GPU6/logical cuda:0, two CPU threads, ≤20min
cumulative model execution including one-epoch smoke, ≤4GB new artifacts,
zero environment interactions at this gate. Main process timeout1,140s.
Stop on source drift, wrong joins, nonfinite values, OOM, resource conflict
or deadline; no width/seed/epoch/target search. Smoke uses debug seed7,
run_id `relative-action-ranking-smoke-s7`; it is engineering only.
GPU model fitting/inference; CPU file/rank-statistics/unit tests.

Implementation: `src/task/cm-interaction-oracle/src/action_ranking.py` and
`src/task/cm-interaction-oracle/tools/run/probe_action_ranking.py`.
Artifacts `outputs/cm-interaction-oracle/<run_id>/` contain execution manifest,
result, fixed pairs/targets, three checkpoints, normalization, raw/permuted
scores and run.log. Manifest pins label/source/code hashes and actual commit.
Existing outputs symlink resolves to declared external artifact root
`/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/cm-interaction-oracle/`,
within authorized ai_ws; only unique new runs, no overwrite or symlink changes.

## Limitations / future evidence

One critic seed, four runs,112 episodes, rare successful holds and substantial
bootstrap residual supervision. Predicting both V-fit targets can still learn
shared V bias, not verified physical quality. Cross-episode pairs can exploit
state variation within bins, and actor observations missing from H may make
action a proxy. Conditional permutation can produce unsupported combinations.
Anchor episode bootstrap neglects partner-induced dependence and is descriptive.
No ranking-evaluation selection by future outcome or V agreement. Multi-seed
Validation, full actor-observation controls, genuine same-state candidate
ranking, calibrated physical outcomes and matched Cm-on/off policy training
are deferred until a decision-relevant positive signal warrants their cost.
