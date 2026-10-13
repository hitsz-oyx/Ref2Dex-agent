---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref3-curation-audit
experiment_id: P-20261013-goal-point-wam-ref3-curation-audit
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 6ac24d99
claim_id: C3
hypothesis_family: HF-goal-point-wam-curation
probe_index_in_family: 1
seed_pool: probe
seeds: [299]
decision_changed_if_positive: test actor-specific soft weighting on problematic target-window strata while retaining interaction transitions and world supervision
decision_changed_if_negative: retain current sampler and prioritize conditioning or execution coverage rather than static-window deletion
status: UNCLEAR
run_id: main3-curation-audit-20261013-r1
---

# ref3: is actor-specific curation justified on the local native three domains?

Result: UNCLEAR for a causal benefit of cleaning. Target-specific actor weighting
is a PROMISING next Probe; uniform static-window removal is not justified.
Completed all 4,378,478 TRAIN windows, frozen 3,072 TRAIN / 1,536 VAL inference,
cross-sequence neighbors, and VAL per-hand diagnostics. No source dataset,
sampler, model weights or pre-existing training job changed.
Decision: prioritize OakInk2 target/hand association and native pre-contact
coverage; preserve static contact and departure windows. Do not adopt ref3's
.05/.1 weights without a matched short comparison.

## Motivation and fixed scope

User ref3 requests native-three-domain evidence before cleaning. This Decision
Probe serves goal-conditioned hand prediction upstream of manipulation. It
distinguishes a sizeable low-progress/far-hand supervision burden from useful
static-object approach/contact/holding, and tests whether the former coincides
with poor frozen prediction. Static goals alone cannot establish high
conditional entropy: the actor also sees hand history and scene geometry.

Full TRAIN census uses the frozen4,378,478 source indices. Exact original
source/category probabilities define expected optimizer exposure separately
from raw population fractions. No TEST data. Geometry is pure NumPy/SciPy
CPU work (up to4 workers,20min); one idle GPU (15min) may evaluate the original
local final10000 checkpoint, unchanged1536-window validation panel and a
fixed train-only neighbor sample (up to1024/source, seed299). No optimization,
new branch, model selection or deletion. Outputs <=1GB under the run directory;
global artifacts remain <=400GB. Stop on input/semantic errors, nonfinite
valid data, resource conflicts, limits or user request.

Use the same32 canonical point IDs as actor object goals. g is mean endpoint
point displacement, in mm; mo is maximum over24 future steps of mean object
point displacement, preserving rotations and out-and-back motion. mh is the
maximum across horizon and valid hands of mean11-point hand displacement.
Distances are unsigned nearest distances from valid11-point hands to512
canonical surface samples under actual object poses, not mesh contact/force.
Measure target distance and any recorded object distance separately. Future
labels inform audit strata only, never actor inputs.

Report g<1/2/5mm. Exploratory strata: object motion>=5mm; otherwise current
or future target proximity<=50mm, or approach>=20mm ending within100mm;
otherwise hand motion<5mm idle; otherwise target stays>100mm far-moving;
remaining rows ambiguous/middle-distance. Preserve contact/approach before
idle classification. Threshold sensitivity and out-and-back exceptions are
reported; these thresholds do not authorize filtering.

Frozen model neighbors exclude the same sequence and match source and hand
presence. Their future dispersion is a descriptive proxy, not a conditional
entropy estimate or proof of inconsistent labels; representation aliasing and
unobserved intent remain possible. Validation remains report-only, never
defines cleaning thresholds or chooses training samples.

## Limitations / future evidence

No causal benefit of cleaning can follow from this audit. A matched short
actor-weighting experiment with unchanged full validation, same initialization,
draws and world supervision is needed before adopting weights. Static hold,
pre-contact, release and other-object interactions must not be discarded just
because endpoint motion is small. Literature SCIZOR/DemInf/ReMix motivates
curation generally; it does not validate ref3's proposed .05/.1 weights here.

## Completed evidence (2026-10-13)

The actual source mixture is OakInk2/GRAB/ARCTIC = 5/9, 2/9, 2/9.
OakInk2 category sampling is .6/.2/.2; native sources have two admitted
categories, so their probabilities renormalize to .75/.25. Percentages below
are within each source. These are exact expectations under the sampler, not
the realized draw counts of a historical training run. Windows overlap and
must not be treated as millions of independent demonstrations.

| Source | TRAIN windows | Raw g<2mm | Sampled g<1 / <2 / <5mm | Sampled static-goal far-moving |
|---|---:|---:|---:|---:|
| OakInk2 | 3,860,393 | 81.63% | 53.81 / 59.33 / 65.21% | 26.47% |
| GRAB | 207,179 | 6.39% | 23.32 / 26.40 / 28.76% | 0% |
| ARCTIC | 310,906 | 10.43% | 20.06 / 26.27 / 30.47% | 0% |

Across all expected mixed training draws, g<2mm is 44.67%; g<1/5mm is
39.53/49.39%. Within ALL draws, static-goal target proximity is 26.06%,
idle-away only 0.65%, and static-goal far-moving 14.70%. Of the latter,
12.07 percentage points are near another recorded object, while 1.96 points
stay >100mm from every recorded object. These are mainly OakInk2 background
anchors: a good hand interaction with object B can be paired with stationary
goal A. This is a candidate target-association problem for actor supervision,
not evidence that the whole trajectory is garbage. "Other object" refers to
all recorded objects; not all must appear in the model's truncated local scene.

Small endpoint goals with >=5mm intermediate object excursion account for
0.15% of mixed exposure. Motion metrics include rotations, so these windows
are preserved by the manipulation stratum rather than classified as idle.

### Coverage and interaction phase

The native preprocessing function `eligible_rows` in
`src/task/cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion/native_data.py`
admits anchors only from `near[t] & pose_valid[t]`. Census confirms **zero**
GRAB/ARCTIC TRAIN windows with both hands >50mm from the target at the current
frame. The native contribution therefore excludes the far-to-near starting
phase, although approach inside 50mm and approaches by a second hand can
remain. Deleting static windows cannot recover this missing coverage.

Define departure-like geometry as g<2mm, minimum hand-target distance <=50mm
now and >100mm at step24. This is 3.09/67.97/22.49% of sampled static-goal
windows for OakInk2/GRAB/ARCTIC. It is a geometric proxy, not verified semantic
release annotation; GRAB static goals especially cannot be called idle. On
the frozen VAL panel, 114 of 128 GRAB static-goal windows are departure-like.
The panel's phase composition differs from TRAIN, limiting simple train/val
error attribution to overfitting.

### Frozen actor: hard examples remain learnable

Original checkpoint `goal-memory-ddp-20261012-r3/train/final.pt` step10000,
unchanged balanced VAL panel (512/source), full24-step per-window macro EPE:

| VAL subset | n | Actual goal EPE mm | Static goal EPE mm | Zero-displacement EPE mm |
|---|---:|---:|---:|---:|
| OakInk2 static-goal far-moving | 140 | 22.44 | 22.41 | 40.75 |
| OakInk2 manipulation | 169 | 11.14 | 15.35 | 24.03 |
| GRAB static-goal near | 128 | 164.32 | 164.32 | 337.35 |
| GRAB manipulation | 371 | 45.62 | 85.77 | 90.87 |
| ARCTIC static-goal near | 148 | 77.42 | 77.97 | 96.41 |
| ARCTIC manipulation | 347 | 45.69 | 71.04 | 75.22 |

Replacing an already static goal with current points is a small perturbation;
its small effect is not by itself evidence of a defective goal interface.
On manipulation, actual goals clearly help this frozen model. Even the
suspect OakInk2 far-moving subset beats zero motion substantially, so it is
not demonstrated to be unpredictable or harmful to joint training.

GRAB static-goal all24 EPE164.32mm contrasts with first8 30.55mm and h1
4.29mm: long-horizon departure prediction contributes materially. The static
subset TRAIN EPE50.19mm vs VAL164.32mm is a concern, but phase composition and
motion amplitude differ; this audit cannot isolate memorization.

Supplemental per-hand inference tests the hypothesis that only the uninvolved
hand causes static-goal errors. GRAB near-hand EPE is188.38mm (141 hands),
far-hand132.53mm (114 hands); near hands account for63.14% of total hand error.
ARCTIC near-hand75.51mm (259 hands), far-hand91.39mm (17 hands); near hands
account for85.34%. Thus "discard the free hand and the problem disappears"
is not supported by this panel. Near means <=50mm at some observed/future
frame, so it includes a contacting hand that subsequently departs.

### Condition-neighbor diagnostic

Sample 1,024 TRAIN windows/source using the existing category sampler and
seed299. Concatenate pooled scene, pooled goal and flattened observed-hand
features, standardize per feature on TRAIN (scale floor1e-3), normalize each
block by sqrt(dimension); retrieve k5 within source and matching hand presence,
excluding the same sequence. Compare GT future pairwise EPE with five random
eligible other-sequence neighbors:

| TRAIN subset | Condition-neighbor EPE mm | Random-neighbor EPE mm |
|---|---:|---:|
| OakInk2 static-goal far-moving | 52.39 | 56.87 |
| OakInk2 manipulation | 30.69 | 50.54 |
| GRAB static-goal near | 272.11 | 315.92 |
| GRAB manipulation | 120.22 | 197.16 |
| ARCTIC static-goal near | 152.09 | 176.61 |
| ARCTIC manipulation | 79.74 | 111.23 |

This supports investigating weakly determined futures, but is **not** an
estimate of conditional entropy or proof of contradictory supervision.
Pooled embeddings can alias scenes, nearest conditions are not identical,
and motion amplitude/phase confounds cross-stratum comparisons. Do not use
VAL error to select samples or thresholds.

## Decision Note

- Question: is ref3-style native data curation worth a follow-up Probe?
- Evidence: sizeable OakInk2 target-far/other-object-near exposure; little
  pure idle-away exposure; native far-to-contact exclusion; static goals
  include useful departure/contact, and remain partly predictable.
- Action: retain the current data and record targeted actor weighting as
  PROMISING to test. First isolate OakInk2 target-mismatched windows with
  reduced actor loss, preserving world supervision and interaction phases.
  Reanchoring is a separate future intervention, not silently mixed into A/B.
  Native far-to-contact coverage should be investigated as a separate arm;
  do not assume one cleaning rule fits three sources.
- Cost/stop: the requested audit is complete, with no new optimization.
  A future short matched A/B needs its own fixed budget and decision gates;
  retain unchanged full/phase-stratified VAL and goal intervention metrics,
  and judge downstream relevance using the existing robot/rollout probes.
  If actor weighting does not improve goal-conditioned interaction without
  damaging contact/departure, do not scale it to a long run.
- Authorization: no new boundary crossing or core-claim change. No conclusion
  that data cleaning caused or will solve the existing autonomous grasp failure.

## Reproduction, checks and artifacts

Implementation commits: census/tests `c835b448`, frozen model `f669c51f`,
per-hand supplemental inference `6ac24d99`. Base repository was `7e25cb30`.
Tools: [census](../../../tools/audit/audit_native_curation.py),
[frozen model](../../../tools/audit/audit_curation_model.py),
[summary](../../../tools/audit/summarize_native_curation.py).

Run directory `outputs/goal-point-wam/main3-curation-audit-20261013-r1/`:
`result.json`, three source `.npz` censuses, `production_contract_check.json`,
`model-r1/{train,val}.npz`, `model-r1/result.json`,
`model-hand-r1/{val.npz,result.json}`, `per_hand_summary.json`,
`transition_summary.json`, and `curation_summary.png`.

Full census53.67s on4 CPU workers; original frozen model audit53.76s on GPU0;
supplemental audit25.67s (TRAIN inference reused, VAL14.70s). GPU monitoring
showed roughly1.9GB process memory at the original inference check; jobs
completed and released the GPU. No training launched and no TEST accessed.

Three focused geometry/stratum tests pass. All admitted geometry is finite;
historical near-mask disagreements beyond rounding tolerance:0. Independent
production GoalWindows transforms plus brute512-point distances on24 windows
agree within0.000351mm. Repeated frozen VAL inference differs by mean absolute
window EPE0.0294mm, max0.964mm, so GPU inference is not claimed bitwise
deterministic; the small rerun differences do not change the qualitative
comparisons. The main error table uses the original inference run.

Inputs: [user ref3](../../user/ref/ref3.md), original data-main-worktree and
local goal-memory-ddp-20261012-r3/train/final.pt. Output:
`outputs/goal-point-wam/main3-curation-audit-20261013-r1/`.
