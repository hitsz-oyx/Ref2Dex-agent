---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-point-wam-controlled-goal
experiment_id: P-20261011-goal-point-wam-controlled-goal
date: 2026-10-11
task: goal-point-wam
branch: main
git_commit: ce269baa851e020fcd540ffd9bd71b438e7b1ff2
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 7
seed_pool: probe
seeds: [247, 252]
decision_changed_if_positive: return to bounded real-data object-goal training using the successful input units
decision_changed_if_negative: inspect spatial conditioning before any physical training scale-up
status: UNCLEAR
run_id: controlled-goal-20261011-r1
---

# Controlled multi-goal engineering diagnosis

Class: Decision, engineering-only. This serves the trajectory representation
subgoal supporting the eventual RL/Cm claim. The previous six-window real-data
fit can memorize H without using g. Distinguish inability to learn conditional
motion with current wiring from the shortcut permitted by that dataset. This
analytic teacher does not test physical feasibility, real-data accuracy or RL.

Result: UNCLEAR under the full engineering gate; input scaling strongly improves conditional response but not absolute teacher accuracy.
Decision: Defer real-data scale-up; inspect an explicit paired displacement encoding in a separate matched engineering Probe.

## Protocol fixed before execution

- Reuse exactly six saved real train scenes, real object start/end and real
  hand trajectories from object-goal-response-20261011-r1; no val/test loader.
- Parent is that diagnostic real-data 300-update checkpoint. Strict parameter
  load, checked stats, recorded input/source hashes. Cache scene once in eval;
  freeze scene, action and world. Train only goal encoder and trajectory decoder.
- Per sample and step draw independent continuous d in [-.1,.1]^3m. Goal end
  becomes g+d; analytic hand target becomes real trajectory+(t/24)*d on valid
  hand points. World inputs/targets remain untouched and world loss is OFF.
  These labels deliberately test conditional capacity, not feasible hand motion.
- First raw arm:600 updates, BF16, AdamW lr.001/decay.01,20-update warmup then
  constant lr, gradient cap1, seed247; offsets seed252. If raw gate fails, second
  arm uses identical initial parameters, cached scenes, offsets and optimizer
  settings, with only goal input coordinates divided by .05m rather than1m.
  Exact initial parameter hashes must match. Do not tune thresholds afterward.
- Four-update engineering smoke first. Held analytic queries are +/- .025 and
  .075m on all XYZ axes plus [.025,-.025,.025] and [-.075,.025,-.025]. These are
  new offsets within the same scenes, not physical held-out examples.
- Source gate: analytic trajectory EPE <=5mm, endpoint response error <=5mm,
  median directional endpoint gain [.75,1.25], and error <=25% of the fixed
  zero-offset prediction on the same altered labels. At least2/3 sources pass.
- Record every source and snapshots each100 updates. Cached-scene world
  differences <=1e-5 required. Stop on nonfinite, OOM, resource conflict or900s.
- Save analytic-diagnosis checkpoint kind with artificial_labels/diagnosis_only.
  Never initialize real-data training from these artificial-teacher weights.

## Decision Note / bounded follow-up

Current real-data fit is strong but frozen goal response is weak; replay finds
no missing inputs or disconnected gradients. Root chooses this cheap controlled
screen rather than inferring a hard architectural failure. If raw passes, run
one fresh PW-initialized600-update real object-goal Probe. If only scaled passes,
run matched raw/scaled600-update real Probes with identical initial weights and
samples. If neither passes, inspect conditioning and defer real scale-up. Any
real Probe gets its own preregistered protocol; no synthetic weights carry over.

One idle GPU2, engineering <=900s, total conditional follow-up <=20GPU-minutes
and30minutes wall, <=5GiB new artifacts; current artifacts about258GB against
300GB cap. No new branch, downloads, robot/RL, videos or external writes. Mission
and claim stay fixed; no extra authorization required. This Probe excludes a
learning-contract hypothesis; it cannot itself advance physical North-star metrics.

## Limitations / future evidence

Analytic translation labels and six familiar scenes can only establish an
engineering learning capability under this particular teacher. Failure within
600 updates is inconclusive about larger budgets or other representations.
Passing does not establish feasible real hand responses. Real conditional
generalization, explicit hand spatial anchoring, world-loss ablation and RL/Cm
utility remain future evidence, chosen only when they change the next decision.

## Artifacts

- [Run inputs](../../../../../../outputs/goal-point-wam/controlled-goal-20261011-r1/input_manifest.json)
- [Run results](../../../../../../outputs/goal-point-wam/controlled-goal-20261011-r1/result.json)
- [Diagnostic implementation](../../../tools/audit/probe_controlled_goal.py)

## Completed result

Code ce269baa, four-update smoke completed. Both600-update arms completed,
raw21.93s/scaled21.70s training and audits; same initial parameter hash, fixed
scene cache and offsets. Frozen world goal difference exactly0 in both arms.
Current tests:9 passed/1 skipped. Training samples GPU2 utilization4--20%,
about807MiB memory. Both source gates and the overall full gate fail.

| Source | Raw / scaled trajectory EPE (mm) | Raw / scaled endpoint response error (mm) | Raw / scaled gain |
| --- | ---: | ---: | ---: |
| OakInk2 | 29.43 / 8.64 | 45.68 / 5.58 | .113 / .982 |
| GRAB | 39.07 / 11.85 | 46.81 / 6.77 | .087 / 1.044 |
| ARCTIC | 31.71 / 8.80 | 45.59 / 6.07 | .116 / 1.012 |

The scale intervention shows much lower analytic errors and near-unit response
within the same budget. It narrows the raw-conditioning optimization concern,
but it does not satisfy the preregistered <=5mm accuracy gate. Do not call it a
passed Probe, or infer physical/real-data usefulness. Per-source error ratios
against zero-offset prediction remain .304/.400/.306 in scaled arm, above.25.
No real-data training launched based on this run.

Decision Note: retain thresholds and original arms; check whether feeding
paired goal displacement reduces the remaining inference burden, using the
same teacher and a separately registered matched comparison. This is within
the original900s engineering and5GiB artifact budget, without new authorization.
