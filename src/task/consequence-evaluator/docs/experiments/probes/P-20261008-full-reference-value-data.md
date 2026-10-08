---
schema: ref2dex.probe.v2
probe_id: P-20261008-full-reference-value-data
experiment_id: P-20261008-full-reference-value-data
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 9ced94d
claim_id: C3
hypothesis_family: HF-consequence-value-data
probe_index_in_family: 1
seed_pool: probe
seeds: [230, 244]
decision_changed_if_positive: use the identified official actor for a subsequent phase-conditioned intervention dataset under the full-reference task
decision_changed_if_negative: repair generator or physical-label semantics before collecting intervention/value training data
status: PROMISING
run_id: official-value-nominal-20261008-r2
---

# Full-reference outcome supervision with controlled normal placing

Result: Completed64nominal episodes; geometry weak labels give62success/2failure,
4160legal24frame windows and256S/P/M preferences. Training readiness remains false.
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

## Completed nominal pilot

R2 ran at9ced94d on GPU2 for115.057s;64episodes each contain542controls and
543measured states. Requested residual plans were known zeros throughout;
no intervention or fork was used. Source/implementation hash guards passed,
the process exited normally and GPU2 was released. Observed GPU memory was
about18.4GB and sampled utilization about45%; this short geometry-export
collector is not a neural training throughput benchmark.

Raw: `outputs/consequence-evaluator/official-value-nominal-20261008-r2/`
(140.53MiB). Label preparation took9.280s on CPU for file/statistical work.
Windows: `outputs/consequence-evaluator/official-value-labeled-20261008-r1/`
(24.26MiB including the subsequent automatic geometry audit). Its manifest
has `training_allowed=false`: train-only seed230,62success/2failure episodes,
no validation/test groups, and fewer than4train failures. No fit was started.

All62positive episodes achieved45frame geometric holding and finished with
sampled bottom/table gap between-1.774mm and+0.349mm, over the table footprint,
at settled velocity; all62ended released from near-hand geometry. Thus normal
table placing is admitted even after hand release. Negative episodes
`s230_w0_e7_airplane` and `s230_w0_e24_airplane` also achieved45frame holding,
but lost geometric control/support during final placing and ended with bottom
gap-45.574mm/-692.715mm. The second still had1.357m/s linear speed. These are
numeric weak-label checks, not manually confirmed collision/video outcomes;
in particular, penetration/support proxy errors remain a possible limitation.
See labeled `geometry-audit.json` for every episode's measured checks.

Each episode contributed65windows (stride8, full24step future). Preferences:
24success,120progress,112margin, with at most one sampled comparison per
unordered episode pair and no H-match requirement. Most comparisons distinguish
stages/margins rather than binary outcome; this is not evidence that the future
oracle improves a trained evaluator or that residual actions cause a benefit.

Decision: PROMISING for the **data-generator and label pipeline** only. Nominal
failures are sparse and appear at placing. Next collect a bounded stage-targeted
intervention Probe retaining clean controls, then independent held-out seed
groups if labels remain plausible. Do not fit these train-only windows or
reinterpret nominal zero plans as action-effect coverage. Manual weak-label
spot checks remain future evidence before upgrading to a formal comparison.

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
