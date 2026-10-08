---
schema: ref2dex.probe.v2
probe_id: P-20261008-tcc-phase-alignment
experiment_id: P-20261008-tcc-phase-alignment
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: ef9a770
claim_id: C3
hypothesis_family: HF-consequence-tcc-phase
probe_index_in_family: 1
seed_pool: probe
seeds: [230, 261, 281]
decision_changed_if_positive: freeze the temporal matcher and prepare direct GT-value Gate1 without evaluator training
decision_changed_if_negative: retain alignment uncertainty and do not train an evaluator or claim Gate1 failure
status: UNCLEAR
run_id: tcc-phase-encoder-20261008-r2
---

# Can a learned TCC geometry embedding recover useful reference progress?

Result: Learned alignment recovers all four clean final phases, but local
regressions exceed the frozen development gate. No evaluator fit or Gate1.
Decision: Preserve the fixed checkpoint and labels; inspect repeated-lift and
placing phase ambiguity before using local Y to select candidates.

## Decision Note

Purpose is to support user ref4_1's causal24step delta-progress labels and the
[full chain](../../../../../../docs/user/完整链路.md) Gate1, which directly uses GT futures for
same-state candidate selection before an evaluator exists. The core Mission
claim and original reference remain fixed. User confirmed that final stable
grasp/normal placing counts as success even after recovery; intermediate loss
is separately reported. This episode metric is independent of local Y.

The geometric8clip label Probe passes self/static/prefix invariance but four
actual clean episodes endP0.262..0.437with total backwards motion0.774..0.871.
The prior's irreversible pruning defect was fixed; temperature/history length,
step bound and limited prior forgetting do not recover trustworthy phase
ordering. Independent source/object/FK review found no object parsing or
future leakage defect. The official physical trajectory remains elevated in
some original-reference return segments, has orientation-motion deviations,
and finishes with hand/object positions differing from literal reference.
No more unprincipled prior sweeps or artificial clock-driven progress.

Hypothesis: cycle consistency on successful physical train views can learn
which common3D trajectory features carry stable temporal phase, without
requiring exact reproduction of every reference-coordinate component. Positive
freezes a phase matcher for label checks and then same-state GT-value Gate1;
negative records insufficient phase alignment, not failed evaluator learning
or a refutation of the reference-progress concept. This is one short Probe,
not an obligation to train or collect indefinitely.

## Frozen implementation, data and budget

Original543frame R is the sole progress anchor, unchanged. It retains original
retargeted18DOF geometry; raw q has differences from native reset clamping and
coupling, which remain explicit rather than silently projected. Object features
match native reference object pose within0.4micrometres. Actual features always
use measured11hand points; no ideal-FK replacement for perturbed states.

Use [upstream Google deterministic TCC](../../research/REFERENCE_PROGRESS_SOURCES.md)
at commit a1e7371c5e006f4e8b314bd23d99220d2fe44c51. Copy the loss module with
Apache attribution/license; the only helper change creates one-hot eye on the
label device. Cycle-back regression_mse with normalized sampled frame indices
is used **only in representation training**. No actual clock/index, actions,
old S/P/M or native future reference clock enters the encoder. This is a 3D
MLP adaptation, not a reproduction of XIRL visual performance.

Eight trailing standardized90D frames→MLP128/GELU/128/GELU/64unit embedding.
Reference-only frozen geometry scales. Each update pairs original R with one
of eight physical source230nominal train views; select first eight passing the
pre-existing full-reference weak-success audit. Old weak success only qualifies
training views, never generates new Y. Their sources/hashes/IDs are recorded.
No source261episode or val/test view enters fitting/checkpoint selection.

Fixed seed281, AdamW3e-4/weight_decay1e-4,1000updates,256sampled frames per
view, gradient norm cap1.0. One idle GPU1,<=300s encoder fit,<=180s eight-episode
label replay,<=0.5GiB artifacts. Save one unique final encoder checkpoint;
do not pick a best checkpoint by source261 results or append epochs. Freeze
source hashes and code commit. Stop on occupancy, drift, nonfinite or deadline.
Loss decrease proves fitting only; label quality decides the research outcome.

Deployment uses only each causal8frame embedding to compare with fixed R
embeddings. No reverse cycle over an actual episode at label inference. Retain
the corrected log-domain historical filter and four-reference-frame posterior
mean-step bound, no monotonic clamp. Unit embedding similarity temperature0.1
matches upstream train loss. Y remains exactlyP[t+24]-P[t], epsilon0.01.

## Development transfer gate and limitations

Replay the same fixed four clean/four contact source261episodes, which have
already been exposed during geometric method development. They are independent
of encoder fitting, **not untouched test data**. Check actual prefix truncation,
reference self end>=0.95/indexMAE<=0.05, static initial end<=0.05. Require all
four clean finals>=0.85and backwards progress total<=0.2before declaring this
phase matcher promising. Inspect contact/failure event values separately:
positive progress during a physically failed transition or confident OOD phase
mapping can still block Gate1 despite nominal coverage. No value-head fitting.

## Observed result at ef9a770

The first encoder attempt `tcc-phase-encoder-20261008-r1` stopped in preflight,
before training: the nominal source manifest froze three Python files at its
collection commit9ced94d, while the live checkout had later legitimate code
changes. The correction verifies each recorded old source against its exact
Git blob and separately freezes current execution hashes. All254source entries
were checked; only those three historical Python files differed. Data, original
motion, assets and weights matched their frozen hashes. This exception does
not accept changed data or unverified source drift; four regression tests
cover the distinction. The failed log is retained.

Encoder r2 completed its fixed1000updates on GPU1 in10.840s, peak allocated
41,701,376bytes. Final100update mean TCC loss0.0001506223 establishes fitting
only. Unique checkpoint `encoder_step_1000.pt` SHA256
`d9dbf5857293e09c3754f9f111983ac44c442ac0c33f8c5572ec214943759902`.
No checkpoint selection or appended epochs used source261.

Frozen transfer run `reference-progress-tcc-labels-20261008-r1` completed
8episodes/496windows in9.694s on GPU1, peak allocated14,541,312bytes. Reference
self finalP0.993479, indexMAE0.001734; stationary finalP0.004652; actual prefix
distribution error0; posterior step bound4/542holds. Fixed64row padded encoder
batches preserve prefix numerics without introducing cross-frame information.

| Clean episode | Geometric r2 finalP | Learned finalP | Learned total backwards progress |
| --- | ---: | ---: | ---: |
| e0 | 0.3188 | 0.9937 | 0.3675 |
| e3 | 0.4168 | 0.9932 | 0.2729 |
| e6 | 0.2621 | 0.9435 | 0.2494 |
| e7 | 0.4374 | 0.9316 | 0.2814 |

All four finalP values exceed0.85; **none satisfies backwards total<=0.2**.
This sum is accumulated negative per-frame phase increments, not a failure
rate or the magnitude of one regression. Clean episodes have5/6/6/8negative
K24windows at epsilon0.01. A bounded CPU audit of existing train packets found
all four at tick176 have Y−0.0233..−0.0251, while measured object height rises
from approximately6.1–6.3cm to29.8–30.4cm. Some final placing windows also have
negativeY. These observations require local reference-phase/event review;
height alone does not establish full task progress or prove a labeling defect.
Removing regression permission or adding an actual clock would violate the
intended signed, causal target and was not attempted.

Contact e1 drops below the tabletop, remains physically failed, and endsP0.0258
with interventionY−0.01232. Other contact finals0.7200/0.9552/0.8788 show that
not every intervention is a failure. The inherited diagnostic distance cutoff
`max_cost=1` marks all four intervention windows invalid; this cutoff is **not
calibrated learned-embedding confidence**, so it cannot certify OOD rejection
or an operational abstention rule. No same-state preference pairs were generated.

Artifacts under `outputs/consequence-evaluator/`: encoder r2 manifest/checkpoint/
loss log; learned-label r1 manifest/labels/phase distributions/
`train-progress-examples.png`; `reference-progress-tcc-audit-20261008-r1/negative-window-audit.json`.
Preserve original reference and all previous geometric runs. GPU1 released.
The label manifest remains `training_allowed=false`; completion is engineering
completion, not semantic approval. Probe result **UNCLEAR** for local signed
utility. This neither establishes nor fails Gate1, which has not run.

Next decision is whether remaining local regressions are phase-matching
ambiguities or physically meaningful departures from R. Use the existing
event traces first; do not relax the frozen gate post hoc or launch an evaluator
to hide label uncertainty. Any later encoder revision gets its own bounded
protocol and untouched evidence before a formal claim.

Original32anchor cold-replay scripts remain but their old outputs directory is
absent. New Gate1 must use fresh simulator complete-prefix replay; mid-state
PhysX restore and switching between prerecorded candidate suffixes are invalid.
Current native-twin r4 can support engineering replay checks but its one state,
two residual arms and no baseline/end outcomes are not Gate1 utility evidence.
Freeze success/control/candidate protocol and conduct a small paired Probe only
after labels pass. Prior23/32→27/32 uses oldY and is not evidence for newY.

Weakly qualified train demonstrations and repeated reference cycles can permit
TCC permutations or spurious phase cues; a low cycle loss cannot establish
task value. Learned embeddings can be unreliable off-reference. One task,
eight training views and one exposed development source do not establish
generalization, success causality or final Cm utility. Matched multi-seed
policy Validation and PointWorld/Execution bridge remain later Gates.
