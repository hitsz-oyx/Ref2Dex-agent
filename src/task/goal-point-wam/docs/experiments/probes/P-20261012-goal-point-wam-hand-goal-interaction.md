---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-hand-goal-interaction
experiment_id: P-20261012-goal-point-wam-hand-goal-interaction
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: fbb33460
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 13
seed_pool: probe
seeds: [247, 248, 249]
decision_changed_if_positive: keep observed-hand query substrate and select the supported goal/world-loss combination for validation
decision_changed_if_negative: stop these local query/loss changes and diagnose sample budget or target information before larger scale
status: PROMISING
run_id: hand-goal-interaction-20261012-r1
---

# Observed hand access, goal route and world auxiliary loss

Class: Decision. Continues user ref1_2 and the compact observed-input positive,
serving real hand-trajectory representation upstream of the final RL/Cm claim.
Question: does access to observed per-hand geometry make goal useful, and does
world auxiliary loss prevent learning that substrate? Rather than explore
unrelated hyperparameters, five extra matched arms isolate these two questions.

Result: PROMISING for memory observed-hand access and memory/query goal utility on the specified panel; world-loss-off remains UNCLEAR. Original raw isolation failures are retained separately.
Decision: Retain observed-hand + memory goal + world_weight1 as the next candidate; stop query-only/world-off tuning. Validate the positive components on independent evidence before any formal claim or L_goal.

## Decision Note: explicit temporary full-machine authorization

On2026-10-12 the user explicitly says today's tests may use more GPUs, including
all idle cards and parallel experiments. This temporarily supersedes Campaign's
4-GPU concurrency cap for today's owned tests only, up to all8 available3090s.
No foreign processes may be disturbed; authorization does not expand external
project permissions or the global300GB artifact limit. Original goal-injection
arms occupy GPU2/3/4. New arms use idle GPU0/1/5/6/7; preflight each before launch.
Aggregate current-turn cap<=365GPUmin/50min wall from larger launch,20GiB new
artifacts including all old/new runs, each trainer<=2700s. Estimated total
outputs+tmp after all checkpoints<285GB. Stop on OOM/nonfinite/identity drift/
foreign conflict/deadline. Do not extend training adaptively or silently change
batch. Temporary8-GPU permission expires after today's tests.

## Protocol fixed before execution

- Same fresh PW50000 scene transfer, seeds247/248/249, batch64/1200updates,
 76,800-window train stream,384 balanced validation windows, lr.001/100warmup/
  cosine tenth/decay.01/clip1/BF16, workers2. No artificial labels or diagnostic
  checkpoint initialization. Object goal32 paired points in one current frame,
  .05m endpoint encoder. Human future hand is label/executed world action only.
- Five arms: memory+hand/world1 (GPU0), query+hand/world1 (GPU1),
  none+hand/world1 (GPU5), memory+hand/world0 (GPU6), query+hand/world0 (GPU7).
  All five allocate exactly the same modules/parameters and initialization;
  route and world loss coefficient are the only config differences. World0
  still runs the executed-action world branch but its loss coefficient is zero.
- Observed hand query: per hand/keypoint concatenate current XYZ/.05m and9D
  observed velocity/acceleration/three-step displacement normalized by existing
  scene stats3:12; shared12->128->GELU->128->LN adapter, mask missing hands
  before and after MLP, broadcast over24 time queries. No future hand, future
  action, source identity or goal is used by this adapter. It is added as an
  independent residual alongside the existing positional/optional goal query.
- The original three no-hand arms were already running. Append hand adapter
  after ALL original modules and use fork_rng(devices=[]) during allocation:
  core parameter values and training RNG state stay exactly unchanged. Fresh
  run asserts original-core parameter SHA equals old arms' full initial SHA
  f507174fb29792c199c2a290f938e4eb68b89edc2063a4fc82877c9733b0397c.
  Full SHA of five new arms must match. Old arms have no dormant hand module;
  its absence changes neither computation nor the preserved core/RNG state.
- Keep old code in detached a90b57eb checkout under project tmp for its strict
  checkpoint audit. New runs use committed current code, separate output roots.
  Match actual train/val index hashes and validation window masks across all8.
  Compare FINAL1200 only, never select by best validation checkpoint.
- Hand-access positive: hand-enabled/world1 EPE >=5% below corresponding
  no-hand/world1 route and zero displacement on >=2 sources. Goal positive:
  within hand/world1, goal route beats hand/no-goal and its own static-goal
  ablation >=5% on >=2 sources. Report both separately; hand benefit is not goal
  utility. World-off positive: EPE >=5% below corresponding hand/world1 on >=2
  sources; this concerns auxiliary-loss interference, not world usefulness or
  physical reliability. No-goal response must be<=1e-7m. Goal/world isolation
  <=1e-5m/rad. Mechanism response is descriptive, not overall accuracy evidence.
- PROMISING only for named passing components; otherwise UNCLEAR. Any positive
  stays local single-seed evidence. Hardware differences and pooling residuals
  do not establish universal architecture superiority. No RL/Cm conclusion.

## Limitations / future evidence

Train windows may overlap; reused exploration seeds/panels are not untouched
Validation. Larger batch is not proof of convergence or data sufficiency for
50M-parameter full WAM. No multi-seed validation, action-world reliability,
counterfactual physical ground truth, robot feasibility, reference tracking or
Cm-on/off learning-efficiency evidence is produced. L_goal remains deferred
until forward-action prediction is reliable; frozen W is still exploitable.

## Artifacts

Planned outputs: `outputs/goal-point-wam/hand-goal-interaction-20261012-r1/`,
five arm directories with frozen indices/manifests, final checkpoints, validation
metrics and cached-scene goal ablations. Comparator will bind actual final
checkpoint hashes, update budgets and code/data/stats identities before using
results. CPU component tests verify common initialization/RNG and observed-only
feature semantics; full training/evaluation use GPU.

## Completed result and attribution

All five arms completed the fixed1200-update batch64 budget from fbb33460.
Each sees76,800 train windows; all eight old/new train/validation/data/stats
hashes and original-core initialization match. Full initialization of five new
arms is71df275c5feaa2b3345ea6ae982670dbc3bfeff03719dc44b2ba81121fa68d29.
Final-checkpoint audits match trainer EPE within1e-5m; no best-checkpoint choice.

| Hand-enabled arm | OakInk2 EPE mm | GRAB EPE mm | ARCTIC EPE mm |
| --- | ---: | ---: | ---: |
| Memory / world1 | 24.056 | 73.958 | 56.594 |
| Query / world1 | 24.660 | 80.334 | 59.239 |
| No goal / world1 | 24.755 | 86.558 | 66.203 |
| Memory / world0 | 23.739 | 73.076 | 56.410 |
| Query / world0 | 24.302 | 78.308 | 58.038 |

Memory observed-hand access improves its no-hand counterpart by6.60%/7.07%/
8.52%, passing all3 sources. Query/no-goal hand access only passes OakInk2,
so their multivariate gates remain UNCLEAR. This identifies a useful information
path, not an isolated proof that PTv3 cannot represent hands.

With observed hand and world1 fixed, memory goal beats no-goal14.56%/14.51%
and its own static-goal ablation15.88%/12.34% on GRAB/ARCTIC. Query goal also
passes those2 sources (no-goal7.19%/10.52%, own-static10.30%/9.52%), but memory
has lower true-goal EPE in all3. OakInk2 goal improvements remain below5%.
No-goal has exactly0 intervention response. Goal utility and hand access are
separate named PROMISING accuracy components, each still exploratory.

World-off gains are only1.32%/1.19%/.33% for memory and1.45%/2.52%/2.03% for
query; no source passes5%. Current evidence does not identify world auxiliary
loss as the main trajectory-learning blocker. World0 world heads are untrained,
so their forward-world scores are not physical-model comparisons.

### Isolation measurement review

Six world1 arms fail the original whole-forward1e-5 isolation threshold;
world0 arms pass. Same-input repetition produces comparable variation, including
valid-only maxima above threshold. Independent sparse-scene forwards and padded
world predictions make the raw statistic unsuitable for attributing goal leak.
All eight bound final-checkpoint supplementary audits use identical cached
scene/executed-action memory: translation/rotation changes exactly0, and no
world-to-goal autograd edge exists. Comparator retains each original failed
statistic and distinguishes supplementary engineering gates from accuracy
component labels. It does not label the original isolation protocol passed,
rerun training, or establish world accuracy. See the original route card for
independent reviewer evidence and no-change repeat numbers.

### Resource record and stopping decision

Five trainer times: memory-world1 786.67s, query-world1 799.54s,
none-world1 781.82s, memory-world0 800.61s, query-world0 772.22s.
All eight larger trainers total105.74GPUmin; training samples about10.1GiB
per GPU with initial utilization mostly84--99%, declining around validation
and I/O. Final model audits and engineering diagnostics completed without
extending training. Global outputs+tmp measured279.873GB, below300GB; owned
GPU processes ended. Foreign work later appeared on GPU2/3 and was avoided by
supplementary audits. Temporary full-machine authorization is no longer used.

Next Decision: preserve the memory+observed-hand candidate; a matched multi-seed
Validation with independently frozen evaluation is warranted before scientific
support. Do not add L_goal until forward-action reliability is discriminated.
Query-only and world-off variants receive no further budget in this screen.

[Bound comparison](../../../../../../outputs/goal-point-wam/hand-goal-interaction-20261012-r1/comparison.json),
[Memory goal audit](../../../../../../outputs/goal-point-wam/hand-goal-interaction-20261012-r1/memory-world1/goal_audit.json),
[Memory isolation diagnostic](../../../../../../outputs/goal-point-wam/hand-goal-interaction-20261012-r1/memory-world1/world_isolation.json).
