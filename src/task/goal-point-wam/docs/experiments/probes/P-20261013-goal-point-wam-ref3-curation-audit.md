---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref3-curation-audit
experiment_id: P-20261013-goal-point-wam-ref3-curation-audit
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 7e25cb30d81ba08fc3d17cbbaee85a2209bb4a38
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

Result: IN PROGRESS. No dataset, sampler, model weights or running job changed.
Decision: first measure all TRAIN windows and actual sampling exposure; then
inspect frozen-checkpoint error and cross-sequence condition-neighbor spread.

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

Inputs: [user ref3](../../user/ref/ref3.md), original data-main-worktree and
local goal-memory-ddp-20261012-r3/train/final.pt. Output:
`outputs/goal-point-wam/main3-curation-audit-20261013-r1/`.
