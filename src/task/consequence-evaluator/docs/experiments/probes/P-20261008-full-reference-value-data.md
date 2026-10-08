---
schema: ref2dex.probe.v2
probe_id: P-20261008-full-reference-value-data
experiment_id: P-20261008-full-reference-value-data
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 3b31457
claim_id: C3
hypothesis_family: HF-consequence-value-data
probe_index_in_family: 1
seed_pool: probe
seeds: [230, 244]
decision_changed_if_positive: use the identified official actor for a subsequent phase-conditioned intervention dataset under the full-reference task
decision_changed_if_negative: repair generator or physical-label semantics before collecting intervention/value training data
status: RUNNING
run_id: official-value-nominal-20261008-r2
---

# Full-reference outcome supervision with controlled normal placing

Result: Pending geometry-audited nominal collection.
Decision: The user explicitly chose complete reference and normal placing as
task success. Retain that task; first audit nominal physical outcomes before
estimating targeted perturbations. No evaluator or policy training here.

## Purpose and Decision Note

The [generator screen](P-20261008-official-generator-screen.md) found64/64
official lifted-and-held trajectories, but0/64 under a full-end perpetual-hold
criterion. The reference ends at table height. The user resolved this ambiguity:
retain the full reference and count **controlled normal placing** as success.
This establishes the evaluator dataset's task, not a replacement for the
Mission's final matched self-trained policy benefit.

Question: Can we generate and automatically annotate complete task outcomes
without treating intentional placing as dropping, or counting a free-fall
landing as success? Positive permits a targeted intervention collector; negative
requires fixing physical semantics, not changing the label to native PPO reward.

The cheapest next experiment is a nominal collection:64train episodes,
32parallel environments/two waves, seed230, one pinned official controller,
one unchanged s3_airplane_lift reference, no intentional residuals/forks.
Source group230 is train-only; label-pair ordering seed244 is also Probe-only.
Success/failure class coverage and sampled examples are audited before expanding
to held-out seeds. This follows the user-provided rollout reference's nominal
failure-distribution-first recommendation; fixed amplitude noise is not claimed
to be a learned CARE/DART distribution.

## Fixed physical weak labels

`S`: Achieve at least45continuous measured geometric near-hand frames
(sampled surface gap<=1cm) elevated>=3cm. No premature dropping after that
completion and before the reference's final placing phase. That phase starts
immediately after the last reference state elevated>=3cm; it is defined from
the source alone. During placing, reject six consecutive unheld/unsupported
frames, or unheld downward velocity<-0.25m/s while more than2cm from support.
The final15frames must have object bottom sampled gap within2cm of the actual
tabletop plane, object center over the table footprint, linear speed<=0.05m/s
and angular speed<=0.3rad/s. Settling on the floor fails the support check.

`P`: Physical stages0..5: no interaction, persistent proximity grasp, lift,
45frame hold, reference-aligned placing, controlled settled placement. Progress
is stage/5; failed episodes can have positive intermediate progress. These are
automatic task proxies, not manually verified grasp/contact ground truth.

`M`: Median near-hand elevated height, held fraction and negative sampled
surface gap over the last8frames of each24frame future. Preferences prioritize
S, then mean future P (deadzone0.05), then M with5mm/0.125/2mm deadzones;
ties abstain. Training preferences may compare different H, only within the
same split/task/reference/controller and different source episodes. Old strict
H-matched preference data remain a separate unchanged headroom evaluation.

Every legal decision-known24plan window uses the **whole-task factual outcome**
under the fixed continuation. Stages completed before the decision remain
completed; a late placing window need not re-grasp/re-hold for45future frames.
All windows still need a full24frame future. Native reward, force contact and
actual noisy/clipped action traces are not value labels or model input. Model
input remains H/requested residual/E/I; reference/support flags stay sidecars.

## Identity, budget and stop conditions

Official SHA8f6823db… is kept separate from self-trained ancestry. The fresh
single-controller route pins the previous completed grasp-coverage screen;
it does not fake the old full-end readiness result or bypass the old six-expert
production schema. New raw schema is `ref2dex.consequence-value.episodes.v1`.
The full source reference, environment and actor hashes are frozen per run.

One idle GPU2, <=900s and2GiB collection; label/statistical CPU processing<=120s
and1GiB. CPU is used only for file/label/statistical operations. Stop on GPU
collision, input drift, unsupported table geometry, nonfinite/incomplete
episodes, missing diagnostics or fixed budget. No installed environment/source
changes, no overwritten checkpoint, no new branch/push. Collection exports
complete measured states and known zero plans; it is not evaluator training.

## Limitations and future evidence

R1 stopped before output/physics because the optional native router imports
joblib eagerly, absent from the archived venv. R2 adds user-site packages as
the **last** fallback, preserving archived Torch/NumPy/rl_games precedence,
and reuses the already compiled compatible gymtorch extension. The failed
log remains `tmp/consequence-official-generator/value-nominal-r1.log`.
R2 is capped at800s within the original900s collection budget.

Point-sampled proximity and tabletop bounding support are weak physical labels,
not exact collision pairs. One reference and one train seed cannot establish
generalization, optimal Q or policy utility. More than one planned mid-episode
placement would need a separately specified stage rule; this pilot uses the
fixed airplane reference's final placing phase. Purely nominal data may have
insufficient negative outcomes; that changes the next targeted collection,
not the thresholds after observing results. Any future class-balanced fit
requires independent episode/seed-group splits and separate source provenance.
