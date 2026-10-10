---
schema: ref2dex.probe.v2
probe_id: P-20261010-history-tau-proposal-diagnosis
experiment_id: P-20261010-history-tau-proposal-diagnosis
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 0860e0b
claim_id: C3
hypothesis_family: HF-history-tau-proposal
probe_index_in_family: 1
seed_pool: probe
seeds: [296]
decision_changed_if_positive: repair history representation before investing in multimodal tau proposal
decision_changed_if_negative: prioritize train-only candidate coverage and multimodal proposal over more deterministic regression
status: PROMISING
run_id: measured-history-tau-proposal-20261010-r1
---

# Diagnose the history-to-candidate-tau blocker

## Motivation / Decision Note

User explicitly redirects research through [ref8](../../user/ref/ref8.md): focus
on usable H-to-candidate-tau, freeze PointWorld, and do not repeat GT E/I-to-Y
necessity experiments. C3/Mission are unchanged. Existing observed tau-only
ranking .7007 and shuffle .4599 are approximate-H evidence from seven informative
anchors; they do not pass a formal same-state ranking Validation. The previous
single-tau MLP improves train/val but held seed414 RMSE .30280m exceeds current
hand persistence .26214m. Historical split seeds412/413/414 have already been
explored and are reused as Probe data, not fresh Validation holdout.

Classify this as a Blocker/Decision audit: distinguish train-only input scaling,
frame/split drift, and one-to-many future ambiguity before paying for a larger
proposal model. Cheapest first step replays frozen weights/data. Fixed saved
prediction replay reproduces .30279982 versus .26213902m (6144/4096/4096 windows).

Ranked hypotheses: near-constant training channels amplify unseen variation;
coordinate or source-distribution mismatch; deterministic prediction averages
multiple possible structured-controller futures. Hypotheses are not conclusions.

## First fixed protocol

Read/hash the previous best checkpoint and its original three source manifests/
trajectories. Reuse exact collection, frame transform and train-only statistics.
GPU2 inference must reproduce saved test predictions. Report per-split normalized
input extrema, train-floor channels, target/current displacement, and per-tick
errors. Two fixed **diagnostic** interventions, no fitting or test-based choice:
zero train-floor normalized channels (training std<=1.01e-4), or clamp normalized
inputs to [-10,10]. Apply each consistently to every split. Neither changes the
frozen checkpoint or task target. Repaired deployable preprocessing would require
a separate matched fit; frozen-weight interventions are only diagnostic.

Stop on hash/source/schema/split/shape/replay mismatch or nonfinite. NN inference
uses an idle GPU, statistics/file transforms CPU. No oracle future object, future
q, force or U32 label enters proposal inputs; old H is retained verbatim and its
semantic content must be audited before calling it deployable sensor history.

If a specific preprocessing defect is found, permit one bounded matched repair
from the same initialization/1200steps, train-only statistics and val selection,
seed296. If it is not found, the cheapest next discriminator is train-only
nearest-neighbor candidate displacement coverage before CVAE/diffusion training.
Specify that follow-up before running. Best-of-K against GT is only coverage,
never a deployable selector or guaranteed execution value.

## Resources / boundaries

One idle GPU, total<=16GPUmin/1GiB; each replay<=180s, each fit<=600s. Fresh output
directories, no overwrites/new branch/push. Shared300GB/four-GPU global bounds
still apply. Monitor utilization/memory/ETA and preserve foreign processes.
No new simulation, τ→A retraining, E/I necessity fit, PointWorld/online selector,
or formal claim. Literature compares ACT/CVAE/DP/PointWAM/DexWM against actual
input/target contracts, not model names alone.

## Results

Frozen replay40f03cb, GPU2/339MiB,7.12s: exact saved prediction replay error0.
Test original .30279982m; zero train-floor channels .30280095m; clip10 .43596327m.
Train/val stay .04328/.06698m, so train-floor amplification is not supported as
the main explanation. Test normalized current-hand RMS2.415 versus train1.0
and val.916; raw current hand can be >6m from object in two failed trajectories.
All measured rotation matrices are orthogonal within1.08e-6. Per-episode model
RMSE median.04705m, but worst two are1.90673/1.21583m; whole-split failure must
not be removed by excluding these rows. Original replay artifact:
`outputs/consequence-evaluator/history-tau-proposal-diagnosis-20261010-r1/`.

Semantic defect: `base_dexplore_task.py:_compute_observations_iter` explicitly
reads reference at t+1/t+16 and includes reference/contact differences. The
old1442-D actor H was never pure measured history. Its old prediction metrics
remain valid under the declared actor-observation contract but cannot establish
a future-reference-free/tactile-free proposal. This is distinct from an invalid
coordinate transform or corruption of recorded labels.

User clarification: only current/past measured hand/object states, **no task
phase or clock**. Decision: one bounded matched preprocessing/target repair,
not a new architecture sweep. Four measured states t-3:t ->300-D hand geometry,
object relative SE3, finger q/dq and object velocity, using the current object
frame; exclude actor observation/ref/contact/action/phase/reward. Compare
absolute position target and **current-hand displacement target** with identical
width256MLP initial weights, episode-balanced batches, optimizer and1200steps.
Train-only history std floor.001 and clip10 applied identically in both arms;
each arm selects on val physical L1 only. Seed296. Historical held seed414 is
used only for final exploratory metrics, not preprocessing/model/K tuning.

In the same run, retrieve8 train-only nearest histories, one window per distinct
train episode; transplant their source-current-hand-relative displacements onto
the query current hand. No future target/utility participates in retrieval.
Report nearest top1 separately from best-of-8 **GT coverage upper bound**, sample
diversity, per-episode tails and all-window/H24/persistence screens unchanged.
Train retrieval excludes its own episode. GT-nearest candidate is not a selector
and the shifted segments are not guaranteed native/contact feasible. No CVAE,
DP, evaluator/PW/native rollout in this run. Stop after this minimal comparison;
use coverage versus top1 to choose the next question. Output:
`outputs/consequence-evaluator/measured-history-tau-proposal-20261010-r1/`.

Matched run0860e0b completes29.76s onGPU2 (~421MiB, sampled2–9%util during
the short fit). Both arms select1200steps by validation physical L1. Small MLP
computation finishes within seconds; data preparation/retrieval/compression
account for the rest. Total frozen replay+matched run is under1GPUmin/200MiB
new artifacts, well within16min/1GiB. GPU2 is idle after completion.

| Test proposal | Point RMSE (m) | H24 RMSE (m) | Old fixed screen |
| --- | ---: | ---: | --- |
| persistence | .26213902 | .40718773 | baseline |
| measured-H absolute position | .46329695 | .51887554 | false |
| measured-H displacement | .21266073 | .33152354 | true |
| train-only nearest candidate | .24278082 | .37549460 | false |
| train-only best-of-8 GT coverage | .17191198 | .26879540 | coverage only |

Displacement improves test point/H24 RMSE18.87%/18.59% over persistence while
retaining all64episodes/4096windows, including tails. Train/val point RMSE is
.10096547/.10757317m. Test episode median.10158417m; two worst episodes still
contribute61.23% of its squared error. Absolute-target test tails contribute
89.70%; neither median reporting nor future-based exclusion replaces full data.
The old1475-input model used different information/normalization: comparisons
with its .3028m are descriptive, not a matched causal ablation. The two new
arms share inputs/weights/batches/optimizer, but target statistics also change;
do not attribute their entire difference solely to anchoring.

KNN eight candidates come from eight distinct train episodes, with test mean
pairwise trajectory RMS.141337m; point-step p99.067672m/max.243046m per30Hz
control. Those statistics do not establish geometry/contact/dynamic feasibility.
Nearest-candidate top1 improves only7.38%, below the fixed10% screen, while
GT best-of-8 improves34.42%: coverage and usable selection remain different.
Query GT determines only the retrospective coverage index. Independent review
recomputes all RMSEs, reconstructs candidates/targets exactly, verifies source
hashes, and replays seven dispersed nearest-neighbor queries without GT sorting.

Decision: retain the **future-reference-free measured-H displacement baseline**
as a PROMISING offline prediction component and the train-only retrieval bank
construction as a coverage diagnostic. No current evidence requires a full
ACT/diffusion/visual world-model replacement. Stop this planned comparison;
next discriminate whether generated-tau scoring/selection preserves the observed
tau ranking signal and whether candidates are feasible, before expanding K or
model architecture. An eventual small CVAE remains a method option, not an
experiment already validated. PointWorld/online execution remain frozen.

Pinned displacement checkpoint:
`outputs/consequence-evaluator/measured-history-tau-proposal-20261010-r1/displacement-best.pt`,
SHA `de98077c51ee7c8de3ffe604d91b8fa02459c80ef40b276f34c946f7df813d3c`.
Saved `test-candidates.npz` contains absolute/displacement predictions,8retrieved
trajectories, source rows/distance, GT coverage index and labels for offline audit.
Reproduction protocol: `tools/run/probe_measured_history_tau.py`, the original
`history-preserving-candidate-bank-{train,val,test}-20261010-r1` directories,
`--gpu 2 --seed 296 --steps 1200 --seconds 600`, fresh task-owned output. Input/target mean/std and
input clip10 are saved in each checkpoint; using weights without those transforms
does not reproduce the proposal. Source code is pinned by manifest/commit.
Full input layout/frame/decoder/hashes and exact reproduction argv:
`outputs/consequence-evaluator/measured-history-tau-proposal-config-20261010-r1/proposal.json`.

## Limitations / future evidence

Single motion and structured actor plus random residual futures; action plans
are not supplied to H-only model. Split separation does not alone guarantee
matched distributions or actual sensor-only history. Strong candidate execution,
exact-state ranking, held-out-motion generalization and Cm-on/off remain deferred.
