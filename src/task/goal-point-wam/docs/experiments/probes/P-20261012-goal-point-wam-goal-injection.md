---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-goal-injection
experiment_id: P-20261012-goal-point-wam-goal-injection
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: a90b57eb
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 12
seed_pool: probe
seeds: [247, 248, 249]
decision_changed_if_positive: retain query conditioning and consider larger matched learning before world consistency
decision_changed_if_negative: stop injection-only tuning and prioritize observed-hand representation and world action reliability
status: UNCLEAR
run_id: goal-injection-20261012-r2
---

# Matched object-goal injection routes

Class: Decision. Implements the first bounded experiment in user ref1_2,
serving goal-dependent hand representation upstream of the final RL/Cm claim.
Question: does a direct trajectory-query path overcome weak object-goal use,
and does its response improve real held-out prediction? Three short full-WAM
arms are the cheapest matched discriminator; synthetic response training would
not establish real conditional utility.

Result: UNCLEAR for direct-query superiority; all three larger arms and frozen audits complete. Memory goal has a local conditional-utility signal on GRAB/ARCTIC.
Decision: Stop query-only injection tuning; retain memory goal and test explicit observed-hand access. Continue deferring closed-loop world goal loss.

## Protocol fixed before execution

- Frozen measured OakInk2/GRAB/ARCTIC corpus/stats and strict PW50000 scene
  transfer; fresh WAM heads seed247, train stream seed248/1200 samples,
  balanced96 val seed249. No future hand goal, artificial teacher checkpoint,
  compact-model initialization, robot or video inputs. Same current scene,
  object start/goal32-point encoder, endpoint representation and .05m scale.
- Same full parameter set in all arms, including the query adapter. Actual
  initial hashes, train/val index hashes, source manifest and stats must match.
  A: existing goal memory. B: mask goal memory; pool the same encoded start/end
  blocks separately and use LN256->Linear128->GELU->Linear128->LN128 as a
  broadcast residual on all learned time/hand/keypoint queries. C: mask goal
  memory and disable query residual. The encoded null is zeroed in B/C because
  it can carry goal through encoder attention; same memory length in all arms.
- Query pooling/projection is our local injection implementation, NOT an exact
  ViGAR reproduction. All arms retain the same encoder; B also pools its output
  for query conditioning, so success is evidence for this route implementation,
  not an abstract theorem about query vs memory.
-600 updates/arm, microbatch2, BF16 GPU2, AdamW lr.001/decay.01,50 warmup,
  cosine to10%, grad cap1, world_weight1, same scene/world loss. Decisions use
  FINAL600 only, same96 previously inspected exploration windows; no test split.
- Frozen audits cache scene once per actual batch. Compare true goal, static
  goal and +/-10cm X translations. No-goal must have <=1e-7m output response.
  Record trajectory EPE, endpoint EPE, symmetric endpoint RMS and signed X gain.
  No physically labeled counterfactual or unit-gain requirement is asserted.
- Accuracy positive: query EPE >=5% below memory, no-goal, zero displacement
  AND its own static-goal ablation on >=2 sources; macro world EPE regression
  <=5%, goal/world isolation <=1e-5. Mechanism-only positive: translated-goal
  endpoint RMS >=1mm and >=5x memory on >=2 sources. Mechanism success alone
  cannot pass the overall PROMISING gate. Otherwise UNCLEAR; no route refutation.
- One GPU serial, <=25GPUmin/35min wall and6GiB new artifacts; each trainer
  <=600s wall and stops on nonfinite/OOM/input drift/resource conflict. Current
  outputs+tmp262.66GB, global300GB cap respected. CPU-only small component tests
  avoid sparse GPU startup; neural training and batch inference use GPU.

## Decision Note

Compact observed-input learning is promising but object goal utility remains
unclear; user now selects ref1_2's goal-injection direction. Root chooses the
three-arm matched route test before applying world-consistency gradients to an
actor. Forward WAM world is only weakly better than static on prior600-update
screens, so L_goal would be hard to interpret and could exploit model error.
No core mission/claim, branch, external project or resource boundary changes.
Positive keeps query route; mechanism-only/negative stops route-only tuning
and returns to observed-hand access or action-specific world reliability.

## User-directed budget revision before larger execution

The user explicitly notes idle GPUs and that the tiny screen may be unable to
distinguish efficacy, authorizing larger batch and concurrent execution. The
small memory run completed600 updates in305.16s, val EPE35.85/141.78/71.90mm,
isolation2.50e-7. Keep it intact; the two small comparison arms are not started.
It is not a completed three-arm result and will not enter the larger comparison.

Root now chooses run_id `goal-injection-20261012-r2`: fresh strict PW initialization
and the same complete parameter set in memory/query/none. Only route differs
within r2. Microbatch16,1200 updates =19,200 actual train windows per arm (16x
the small screen),100 warmup, same lr.001/cosine/decay/clip/loss. Workers2 for
all arms,384 balanced validation windows (128/source), validation/checkpoint
every400, final1200 only. This expanded exploration panel is not untouched
Validation. All accuracy/response gates above remain fixed. Same seeds247--249.

Three idle3090s (GPU2/3/4), one complete arm per GPU, concurrently; no DDP or
cross-arm gradient sharing. Benchmark six batch16 training updates first on
GPU3, engineering-only and no checkpoint, to measure actual memory/throughput.
Then launch the matched runs only if memory fits with headroom. Shared bound
<=140GPUmin/50min wall and8GiB new artifacts (including the retained small arm),
<=2700s trainer wall each, under the global4-GPU/300GB limits. Stop on OOM,
nonfinite, foreign resource conflict or deadline; no silent batch reduction.
If batch16 does not fit, choose and record one common smaller batch before
launching all three arms. This budget revision is user-directed and does not
change mission/claim or external permissions.

Batch16 engineering preflight completed: peak allocated2.57GB/reserved2.81GB,
steady synchronized compute0.204s/update. There is ample24GiB device headroom.
Before launching scientific r2 arms, root chooses a further common batch64
preflight (six updates, no checkpoint) to better use idle GPUs and distinguish
the user's undertraining concern. If it fits with headroom, r2 uses batch64
and1200 updates =76,800 train windows/arm (64x the original1200), all other
r2 settings and limits above unchanged. Both engineering reports are kept;
they are not conditioning/accuracy evidence. No training arm uses preflight
weights. Current large config files express the intended batch64 choice.

## Primary-source design input

[ViGAR Table2a](https://arxiv.org/html/2610.02368v1#S5.SS5) verifies the ref's
62.17/61.72/50.97/49.18 figures, but compares VAE geometry/generator vs semantic
reasoner, not this local query implementation. [SelfWAM](https://arxiv.org/html/2608.00725v1#S3)
supports teacher-forced clean executed-action information isolation; it does
not establish W accuracy or a physical guarantee from goal consistency.
Detailed notes reside in paper/vigar, paper/selfwam and paper/act2goal.

## Limitations / future evidence

Single initialization seed, overlapping 76,800 train windows and a reused
384-window exploration panel cannot establish architecture superiority.
Untouched evaluation and multi-seed Validation remain required. The retained
tiny run uses a different96-window panel; its numbers do not establish a matched
small/large-budget effect. World action sensitivity, held-out forward accuracy,
candidate ranking and bounded L_goal require separate evidence; current task
results cannot establish human-to-robot transfer, physics or Cm/RL utility.

## Artifacts

- Retained tiny output: `outputs/goal-point-wam/goal-injection-20261012-r1/memory/`
- Completed matched output: `outputs/goal-point-wam/goal-injection-20261012-r2/`
- Each arm saves `result.json`, `input_manifest.json`, indices and final checkpoint.
- Post-training frozen audits save `goal_audit.json`; the summary is `comparison.json`.
- [Comparison implementation](../../../tools/audit/compare_goal_injection.py)

## Completed larger comparison

Actual batch64 preflight: 7.76GB allocated/8.57GB reserved, synchronized steady
compute0.456s/update. All three fresh arms completed1200 updates with76,800
sample exposures each,384 balanced validation windows and fixed final checkpoints.
Actual code a90b57eb; initialization and train/val/data/stats hashes match.
Training wall per arm: memory802.41s, query793.43s, none807.55s.

Trajectory EPE (mm; average over valid hand points and24 future steps):

| Arm | OakInk2 | GRAB | ARCTIC |
| --- | ---: | ---: | ---: |
| Zero hand | 35.333 | 114.981 | 73.638 |
| Memory, real goal | 25.755 | 79.587 | 61.866 |
| Memory, static goal | 26.284 | 89.029 | 66.302 |
| Query, real goal | 26.455 | 82.327 | 62.262 |
| Query, static goal | 27.108 | 91.239 | 67.359 |
| No goal | 26.403 | 88.711 | 68.353 |

Memory beats no-goal by10.3%/9.5% on GRAB/ARCTIC and its own static-goal
ablation by10.6%/6.7%. Query responds more to translated goal (endpoint RMS
33.9/46.1/42.9mm versus memory24.5/35.6/35.6mm), but no source passes either
the registered accuracy or5x-response superiority gate. Query world macro EPE
is12.57% lower than memory; this does not make query trajectory superior.
No-goal intervention responses are exactly0. Frozen EPE matches trainer final
within1e-5m, using the same TF32 setting and cached scenes for all goal variants.

The original full-forward world-isolation threshold FAILS (maximum.001987).
Same-input repeated forwards also differ, so this statistic confounds goal
intervention with sparse scene numerical variation and includes padded objects.
Independent query audit found repeat.006293 versus changed-goal.004926;
valid-only repeat5.33e-5 also fails, so masking padding is not a threshold fix.
All three supplementary final-checkpoint audits hold scene/executed-action
memory identical: world translation/rotation differences exactly0 and gradient
to goal is absent. This supports architectural independence of goal, not world
physical reliability. Original failures and comparison.json remain unchanged;
no thresholds or accuracy gates are retroactively relaxed. Supplementary audit
is engineering-only, with final checkpoint/source/data/stream hashes bound.

[Original comparison](../../../../../../outputs/goal-point-wam/goal-injection-20261012-r2/comparison.json),
[Independent measurement review](../../../../../../outputs/goal-point-wam/goal-injection-20261012-r2/world-isolation-independent.json).
Per-arm `world_isolation.json` saves the supplementary diagnostics. The original
small run remains intact and is excluded from this comparison.
