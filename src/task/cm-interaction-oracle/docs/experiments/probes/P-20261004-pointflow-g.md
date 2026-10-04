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
run_id: pointflow-g-k1-s201
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
root at state[36:49]. The native trace confirms q equality and identity hand
root; FK/body consistency is audited before interpreting the result.

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

All newly fit bridges use seed 201, the same train rows, minibatch order,
optimizer and 16 epochs. Train-only H/E and G normalization. Primary metric is
episode-balanced held-out MAE. Bootstrap over 22 held-out episodes is descriptive
only; four source runs are not independent large-sample validation.

PROMISING requires GT MAE reduction ≥5%, CM-fit reduction ≥3%, retention ≥25%
of GT absolute gain, and CM-fit beating action control. If GT does not reach
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

## Limitations / future evidence

No new policy, causal Cm utility or formal validation claim. More seeds and
source-run generalization are deferred until a decision-relevant positive signal.
GT E is a future oracle. The predictor uses geometry rather than direct current
object velocity, so rigid-motion persistence is a necessary diagnostic if errors
are large. If only CM-fit works, audit geometric/action information and a matched
feature control before using it as a physical consequence teacher.
