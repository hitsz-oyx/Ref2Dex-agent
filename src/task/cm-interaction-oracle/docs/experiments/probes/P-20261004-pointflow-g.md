---
schema: ref2dex.probe.v2
probe_id: P-20261004-pointflow-g
experiment_id: P-20261004-pointflow-g
date: 2026-10-04
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: execution commit recorded in run result.json
claim_id: C3
hypothesis_family: HF-pointflow-G
probe_index_in_family: 1
seed_pool: probe
seeds: [201]
decision_changed_if_positive: audit action sensitivity before a bounded action-improvement teacher probe
decision_changed_if_negative: inspect E error and train/deployment mismatch; no online policy training or capacity sweep
status: UNCLEAR
run_id: pointflow-g-k1-rootfix2-s201
---

# Does real point-flow predicted E preserve E-to-G information?

## Decision question and cheapest method

Ref5 asks whether geometry + action-derived hand flow → predicted physical E
can preserve the directional GT E → G gain. Generic raw (H,a) prediction did
not answer that question. This is a Decision Probe, not a policy-utility test.

Reuse the corrected-history `tmp/e260_all4_h16_histfix.pt` and the existing
Inspire-adapted K1 spatial encoder from
`tmp/P-20261004-cmv2-unified-ei-k1-v7.best1.pt`. Only its strict-loaded `base`
is used: no I head, no context branch. The GRAB/MANO K4 checkpoint cannot
directly populate the old Inspire 13D K8 E contract; choose permitted K1.

Fix E as six-dimensional decision-object-frame translation and axis-angle.
Inputs to the spatial predictor are current object/hand geometry and nominal
hand flow generated from current native q and current action by exact target
mapping/FK. No realized future action, hand flow, object state or reward is an
input. Source `physical_value_live.snapshot` fixes q at state[0:18] and object
root at state[36:49]. The native trace confirms q equality, but its identity
hand root does NOT transfer to the four G sources (see engineering correction).

Use the same episode split key and split identifier 20261004 as ref4, with up
to 64 evenly spaced rows per episode. Split/geometry identifier is a pinned
historical sampling contract, not a new training seed. Training seed 201 is
in the Probe pool. H uses corrected ten-step history of state, factual previous
action, context and progress; target is stored exact MC RTG. K1 pose, corrected
history and subsampling mean raw MAEs cannot be compared to old K8 results.

## Matched arms and predeclared decision

- H: HE-shaped bridge with constant E (identical architecture to other E arms).
- H+GT E: oracle information upper reference.
- H+CM E swap: same GT-trained bridge, replace only held-out E.
- H+CM E fit: train/evaluate on frozen CM predictions.
- H+action: control for new current-action information, rather than E accuracy.
- H+action+current hand root (HAR): control for the additional current root
  observation required by valid point-flow geometry.

All arms use the same 30D padded auxiliary branch (E6+24zero, action18+12zero,
action18+root12, or 30zero), and identical parameter counts. Root12 is the
observed 3x3 rotation and XYZ translation, standardized from training rows.
All newly fit bridges use seed 201, the same train rows, minibatch order,
optimizer and 16 epochs. Train-only H/E and G normalization. Primary metric is
episode-balanced held-out MAE. Bootstrap over 22 held-out episodes is descriptive
only; four source runs are not independent large-sample validation.

PROMISING requires GT MAE reduction ≥5%, CM-fit reduction ≥3%, retention ≥25%
of GT absolute gain, and CM-fit beating HAR control. If GT does not reach
5%, the probe is UNCLEAR about predictability-to-value. Otherwise failed CM
conditions are UNPROMISING for this frozen checkpoint, not refutation of Cm.
No online policy training follows an unresolved/negative result.

## Resources and stopping

One currently idle GPU6, 2 CPU threads, ≤20 minutes wall time, ≤2 GB new
artifacts. No environment interaction and no checkpoint overwrite. Wiring
smoke uses separate run_id `pointflow-g-k1-smoke-s201`; it is not evidence for
scientific conclusions. Stop on nonfinite values, broken schema/geometry, OOM,
resource conflict or timeout. Run through a 1200-second process timeout.

Implementation: `src/task/cm-interaction-oracle/tools/run/probe_pointflow_g.py`.
Artifacts: `outputs/cm-interaction-oracle/<run_id>/result.json`, `bridge.pt`
and `run.log`. Result records input/checkpoint/mesh/URDF/script hashes, actual
execution commit, split, normalization, metrics and resource usage.

## Engineering correction before valid Probe

`pointflow-g-k1-s201` completed, but is invalid for research interpretation:
it assumed identity actor root based on a different native trace. Cross-source
FK audit found up to 2.23cm actor-base translation in the actual G sources.
This run and its smoke remain evidence of the rejected assumption and do not
count as an effective Probe. Its calculated `UNCLEAR` field is superseded here.

For the valid retry, reconstruct the *current* actor root from current measured
body pose and FK, then verify all five bodies. The cache audit found max residual
5.35e-6m, state/object alignment error 1.19e-7 and reconstructed E discrepancy
2.38e-7. Apply current root to both current and nominal-action FK, holding root
fixed over that nominal step. Never load future roots into the predictor.
An explicit current-root/action control is required because that observation
was absent from original H. Do not attribute its information to physical E.

Cache: `outputs/cm-interaction-oracle/surface_i_gt_rootcache_s202/geometry_cache.pt`;
its paired audit.json hashes source and cache. The run requires `--geometry-cache`
and checks source hash, cache hash and exact absolute source-row correspondence.
Same total 20-minute/2GB budget; this technical retry does not reset it.

`pointflow-g-k1-rootfix-s201` then failed before producing metrics because a
helper insertion displaced the inference no-grad decorator. Restore it and
require detached finite predictions; retain failed log. Valid retry is
`pointflow-g-k1-rootfix2-s201`, with identical design and budget.

## Predeclared bounded follow-up: GT E contract

After valid K1 GT gain fails 5%, distinguish short-horizon/pose-only E from the
original K8 full13 GT effect signal before adapting a predictor. This Decision
diagnostic changes whether K4 point-flow deserves investment. Cheapest method:
no geometry/model inference or new data, only five short same-capacity G fits.

Run `gt-effect-contract-s201` uses identical subset/split/H/G/seed/optimizer/16
epochs. Every arm has an 8-step 30D auxiliary branch: H constants; pose6 K1;
pose6 K4; pose6 K8; full13 K8. Missing channels/horizons are standardized
training-mean constants. Common train-only per-channel normalization and same
parameter count and minibatch order; no new hyperparameter sweep.

A contract is PROMISING only if MAE gain ≥5%, at least 12/22 episodes improve,
and gain remains positive excluding the preidentified largest-|G| episode.
This sensitivity requirement is predeclared for this follow-up, not a revised
primary metric for the completed K1 probe. Choose smallest promising pose
horizon; full13-only gain requires a twist-aware E contract before prediction.
If none passes, do not fit a new E predictor into this G target. Remain within
the original total resource budget. Multiple contracts are exploratory choices,
not independent confirmations or formal validation.

Before closing this follow-up, independent review found that left padding leaves
K1/K4 information followed by seven/four zero GRU steps before the final-state
readout. Thus horizon comparisons also depend on memory decay through padding.
Predeclare one controlled `gt-effect-contract-right-s201` repeat: only right-align
K1/K4 to the final slots, retaining chronological order, all other arms/method/
seed/capacity/gates unchanged. This resolves a specific engineering design
confound before choosing whether to retain short-horizon E. It is not a new
predictor search or budget reset. The original left-aligned results remain.

## Limitations / future evidence

No new policy, causal Cm utility or formal validation claim. More seeds and
source-run generalization are deferred until a decision-relevant positive signal.
GT E is a future oracle. The predictor uses geometry rather than direct current
object velocity, so rigid-motion persistence is a necessary diagnostic if errors
are large. If only CM-fit works, audit geometric/action information and a matched
feature control before using it as a physical consequence teacher.

## Completed results and root decision

Result: UNCLEAR for the valid K1 point-flow-to-G bridge; UNPROMISING for expanding a pose-only K4 predictor into this G regression from the fixed GT contract diagnostic.
Decision: keep E as the physical main route, freeze I head/K4 expansion and this frozen K1 G teacher; audit task-relevant hold/drop value evidence before new predictor fitting or online training.

Valid K1 run code: `0852c64cedc45c315a25f0ac9b9478bf4cbcfb50`, seed201,
5,760 train / 1,408 test rows; 90 / 22 episode groups from four source runs.
All six arms have 112,753 parameters. Duration 25.14 seconds, peak CUDA
allocated 143,500,800 bytes. Actual source/checkpoint/code/geometry hashes
are in the [run result](../../../../../../outputs/cm-interaction-oracle/pointflow-g-k1-rootfix2-s201/result.json).

| G input | Episode-balanced MAE | Relative improvement vs H |
| --- | ---: | ---: |
| H + constant E | 23.3652 | — |
| H + GT E | 23.2837 | 0.35% |
| H + CM E, GT-trained bridge swap | 23.6477 | -1.21% |
| H + CM E, trained on predictions | 22.8961 | 2.01% |
| H + current action | 24.2527 | -3.80% |
| H + current action/root | 24.9192 | -6.65% |

CM-fit delta CI95 is `[-0.713, 2.457]`, descriptive only. The biggest-|G|
episode contributes 70.54% of H error and 185.45% of total CM-fit improvement;
excluding it, the remaining 21 episodes are 5.82% worse than H. Only 8/22
episodes improve. HAR being worse than H does not make CM physically valid.
See [statistical audit](../../../../../../outputs/cm-interaction-oracle/pointflow-g-k1-rootfix2-s201/statistical_audit.json).

CM held-out translation RMSE is 19.33mm, versus zero flow 11.49mm and current
twist persistence 4.92mm. Axis-angle RMSE is 0.1391rad versus persistence
0.0804rad. These units are reported separately, not fused into a physical
error claim. The nominal root is held fixed; current inertia is not a direct
input to this frozen spatial base. Even prediction-fit G benefit would not
identify accurate E as its cause.

GT contract follow-up code: `ae4ee11a7f36f165cc27764e85b136d8b5422f67`, same
rows/seed/split; every arm is an 8-step 30D padded branch with 112,753
parameters. Duration 14.58 seconds. Its newly trained H MAE is 23.5476;
do not compare raw MAEs directly with the K1 bridge.

| Oracle contract | Relative gain | Improving episodes | Gain excluding largest absolute-G episode |
| --- | ---: | ---: | ---: |
| Pose6 K1 | -0.03% | 12/22 | 0.78% |
| Pose6 K4 | 0.29% | 14/22 | 0.74% |
| Pose6 K8 | 2.75% | 17/22 | 3.01% |
| Full13 K8 | 7.42% | 11/22 | -1.66% |

The table above uses left-aligned short sequences; GRU padding can attenuate
K1/K4, so it does not by itself establish the effect of horizon. Pose K8 has
a weak directional signal (MAE delta CI95 `[0.126,1.567]`), but
does not reach the predeclared 5% practical gate. Full13's CI crosses zero and
its gain does not survive the predeclared sensitivity check. None passes.
Keep the K8 pose observation as deferred evidence, not permission for a broad
predictor sweep. [GT contract result](../../../../../../outputs/cm-interaction-oracle/gt-effect-contract-s201/result.json).

Independent read-only reviewer `ref5_engineering_review` verified all 58,111
assembled action/previous-action joins against pre-step shards (max error0),
source/cache hashes, row-key alignment, strict spatial-base load, current-only
root input and matched parameters/initialization/training order. No further
fatal engineering defect found after the documented no-grad fix. Reviewer and
root agree that these runs do not justify policy teacher or scientific utility
claims. The existing supervised E/I checkpoint was jointly fitted to E/I;
this probe freezes its spatial base and does not claim optimal E-only training.
Full13 versus pose6 also changes quaternion versus axis-angle representation,
so their difference does not isolate the contribution of future twist.

Reviewer read-only episode coverage: 112 episodes contain just 2 successes,
2 drops-after-success and 32 nonzero holds. The largest-|G| episode has an
8.87-second maximum hold but also a later drop. Do not immediately train a
success/drop G on this sparse sample, or interpret high RTG as stable grasp.
Audit continuous hold/drop timing and label semantics before deciding a new fit.

### Decision Note

- Question: is there enough evidence to invest in K4 point-flow/G teacher or I prediction?
- Evidence: valid K1 GT gain0.35%; pose K4 GT gain0.29%; K8 pose only2.75%; I and full13 RTG gains dominated by one episode; CM prediction loses to inertia persistence.
- Root choice: retain physical E route and corrected geometry, stop these local expansions. Next inexpensive decision should audit continuous hold/drop targets and action contrast, rather than improve average RTG regression unconditionally.
- Cost/stopping: this round used two isolated GPUs, zero new interactions and <10MB outputs; all jobs finished within the fixed bound. Future audit is pure existing-data statistics first; a new fit needs its own predeclared task-relevant decision, no automatic budget reset.
- Authorization: none newly required within current campaign; Mission/claim remains unchanged. Formal multi-seed causal Cm-on/off policy validation is still outstanding.
