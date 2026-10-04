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
status: UNPROMISING
run_id: relative-action-ranking-s207
---

# Does current action improve ranking of frozen relative task advantage?

Result: H macro pair59.71%, HaK1 59.62%, HaK4 realized60.52%; current-action ranking gate fails after independent implementation review.
Decision: stop critic/Cm expansion on this frozen target/data/fit contract; proxy ordering also lacks strong V-fit stability, so do not refute physical action information or Cm.

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

## Completed result

Formal execution commit `1a64eabe7a073b60e60fa4e76cc3358408955ddf`, run
`relative-action-ranking-s207`; ranking code introduced at `72e76d7`, runner
path-hash fix at `3cc4add`. The intervening commit only changed shared
verification/skill-link handling; formal manifest pins unchanged model/runner
hashes. Frozen V/labels still come from `9dd503d`, with exact target joins.
All14,336 queries,46,080 train pairs,9,728 test pairs. Three models each have
96,705 parameters and identical initial-state hash
`65ae0cb9328c9c72293acc08e6bb19275e312b111b7ff400db028a596e6d8bff`.
All normalization tensors and training protocols match. Formal duration37.30s,
peak allocated GPU554,187,776 bytes;3.4MB formal artifacts, zero interactions.

| Arm | Train macro pair accuracy | Test macro pair accuracy | Test macro Spearman | Test pair accuracy after action permutation |
| --- | ---: | ---: | ---: | ---: |
| H | 92.93% | 59.712% | 0.2339 | 59.712% |
| HaK1 current | 93.29% | 59.617% | 0.2360 | 59.484% |
| HaK4 realized | 94.83% | 60.520% | 0.2329 | 60.706% |

Pair metrics cover64 test strata,2,432/2,816 query anchors and19/22 episodes.
Unsupported episode indices73,94,105 have no other test episode in their
motion/noise strata; pair exclusion is defined by observed support, not
advantage/outcome. Registered macro Spearman covers88 strata including24
single-episode strata without cross-episode pairs. These two denominators
must not be described as the same coverage. Root's post-run common64-strata
Spearman check is H0.2760 / HaK1 0.2723 / K4 0.2936, also not a deployable K1
gain; this diagnostic does not replace the registered metric/gates.

K1 macro pair gain **−0.095 percentage points**, macro Spearman gain0.00208;
action permutation drops pair accuracy only0.133pp. Both individual frozen
V targets give negative gains (−0.231/−0.153pp). Eight of19 paired episodes
improve, versus the fixed≥12 requirement. Episode mean-delta descriptive
CI95 `[−1.984pp, +1.162pp]`; no independent-pair significance claim.
Removing every pair touching the largest-MC test episode leaves8,682 pairs
and gives K1 +1.333pp. Retain this weak sensitivity observation, but primary
gain/action-sensitivity/target-consistency gates still fail. K4's +0.808pp
cannot rescue current-action ranking and its permutation slightly improves
accuracy. No predicted E/I, candidate collector, teacher or policy training
was launched after this failure.

### Frozen ordering diagnostics and root interpretation

The premise that Pearson0.978 establishes stable ordering is not supported.
Actual between-fit Spearman is **0.363 train / 0.540 test**; macro conditional
pair-order agreement is62.39%/66.83%. This uses all pairs without discrete
thresholds or selecting only fit agreement/reward-active rows. Removing the
old class boundaries did not create strong target ordering stability.

Large train/test ranking gaps (93%–95% versus≈60%) show weak generalization
of these proxy labels under the fixed fitting protocol; they do not isolate
capacity, sparse physical outcomes, shared V bias or data contrast as the sole
cause. Inputs DO reach the model: K1 score permutation RMS0.991 versus held-out
score std4.051; K4 RMS2.071 versus std4.355. Thus nearly unchanged accuracy is
not a zero-input wiring failure. It is absence of useful held-out ordering
gain under this contract. Shared V artifacts and cross-state pair comparisons
remain plausible limitations; absent physical action information is unproven.

### Engineering corrections and independent review

First smoke `relative-action-ranking-smoke-s7` failed before loading data or
fitting any model: Python3.8 invocation left `__file__` relative while code-hash
recording required an absolute path. Corrected with SCRIPT.resolve at `3cc4add`.
Failed directory/log retained. Valid `relative-action-ranking-smoke2-s7`
completed one epoch at debug seed7 in10.70s; engineering evidence only.
No model/label/threshold change or formal research retry was needed.

Independent read-only reviewer `ref5_engineering_review` verified exact frozen
target/source/code hashes, all pair split/stratum/cross-episode boundaries,
current-only K1 inputs, train-current-action normalization, matched parameters/
initialization and correct pairwise loss. Independent SciPy average-rank
Spearman and raw pair-sign metrics match report; H permutation is identical.
Episode coverage and descriptive bootstrap are consistent. No further fatal
implementation defect found in this chain. Root reproduced score/metric/hash/
pair/normalization/CI checks with
`src/task/cm-interaction-oracle/tools/audit/audit_action_ranking.py`; the
immutable formal result is supplemented by `statistical_audit.json`.

### Closing Decision Note

- Question: extend to predicted Cm/candidate selection after ranking G1?
- Evidence: current-action gain−0.095pp, permutation effect0.133pp, negative
  gains against both V fits,8/19 improving episodes; continuous Pearson hid
  much weaker rank stability; formal inputs/targets/metrics audit correctly.
- Root choice: stop this frozen proxy-label/data/fit ranking expansion, as
  ref2's negative branch specifies. Preserve code/audits and weak +1.333pp
  outlier-removal diagnostic. Do not close the core relative-action/Cm idea.
  Before another critic architecture or predicted E/I investment, establish
  a decision-relevant physically grounded action/outcome contrast contract;
  repeating V seeds, width or thresholds on this dataset would not resolve it.
- Cost/stopping: one GPU, <49s cumulative valid model execution incl smoke,
  <8MB new artifacts, zero new environment interaction; all jobs finished.
  G2/Cm and true candidate counterfactual stages are conditional on useful
  action-ranking evidence and were not justified by this failed gate.
- Boundary: no changed Mission/claim/resource authorization; this is Probe
  UNPROMISING for one contract, not formal refutation. Cm policy utility OPEN.

## Verification and completion audit

39 Task/index/governance tests passed; scoped verification PASS before smoke.
Full `tools/verify.py --changed` now PASS, including experiment schema, seeds,
links and selected historical-route tests; shared `1a64eab` resolved the prior
skill-template link issue. New independent audit ran successfully on the actual
formal scores/targets, not synthetic examples. Index check and owned-file
whitespace check pass. GPU6 is idle; no experiment process remains.

Ref2 requirements: preserve existing V/data/advantage verified by exact hashes
and joins; actual H/Ha continuous pairwise fitting executed; Spearman/conditional
pair accuracy/action sensitivity measured; negative branch independently
reviewed and stopped. The positive-only predicted E/I and small physical
candidate stages were not activated. No whole-core research completion or
physics/candidate causal claim is made. All changes are local; unrelated user
edits and existing checkpoints preserved.
