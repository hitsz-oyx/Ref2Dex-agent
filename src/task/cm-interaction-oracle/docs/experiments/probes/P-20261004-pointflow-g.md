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
status: PLANNED
run_id: pointflow-g-k1-rootfix-s201
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

## Limitations / future evidence

No new policy, causal Cm utility or formal validation claim. More seeds and
source-run generalization are deferred until a decision-relevant positive signal.
GT E is a future oracle. The predictor uses geometry rather than direct current
object velocity, so rigid-motion persistence is a necessary diagnostic if errors
are large. If only CM-fit works, audit geometric/action information and a matched
feature control before using it as a physical consequence teacher.
