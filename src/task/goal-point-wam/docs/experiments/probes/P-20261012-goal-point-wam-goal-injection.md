---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-goal-injection
experiment_id: P-20261012-goal-point-wam-goal-injection
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 00ca63ef
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 12
seed_pool: probe
seeds: [247, 248, 249]
decision_changed_if_positive: retain query conditioning and consider larger matched learning before world consistency
decision_changed_if_negative: stop injection-only tuning and prioritize observed-hand representation and world action reliability
status: UNCLEAR
run_id: goal-injection-20261012-r1
---

# Matched object-goal injection routes

Class: Decision. Implements the first bounded experiment in user ref1_2,
serving goal-dependent hand representation upstream of the final RL/Cm claim.
Question: does a direct trajectory-query path overcome weak object-goal use,
and does its response improve real held-out prediction? Three short full-WAM
arms are the cheapest matched discriminator; synthetic response training would
not establish real conditional utility.

Result: UNCLEAR; execution pending.
Decision: Run memory/query/no-goal at the existing600-update budget; defer closed-loop world goal loss.

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

## Primary-source design input

[ViGAR Table2a](https://arxiv.org/html/2610.02368v1#S5.SS5) verifies the ref's
62.17/61.72/50.97/49.18 figures, but compares VAE geometry/generator vs semantic
reasoner, not this local query implementation. [SelfWAM](https://arxiv.org/html/2608.00725v1#S3)
supports teacher-forced clean executed-action information isolation; it does
not establish W accuracy or a physical guarantee from goal consistency.
Detailed notes reside in paper/vigar, paper/selfwam and paper/act2goal.

## Limitations / future evidence

Single seed,1200 update samples and reused96-window val panel cannot reject
goal conditioning or establish architecture superiority. A larger matched
budget, untouched evaluation and multi-seed Validation are deferred until a
positive decision signal. World action sensitivity, held-out forward accuracy,
candidate ranking and bounded L_goal require separate evidence; current task
results cannot establish human-to-robot transfer, physics or Cm/RL utility.

## Artifacts

- Planned output: `outputs/goal-point-wam/goal-injection-20261012-r1/`
- Each arm saves `result.json`, `input_manifest.json`, indices and final checkpoint.
- Post-training frozen audits save `goal_audit.json`; the summary is `comparison.json`.
- [Comparison implementation](../../../tools/audit/compare_goal_injection.py)
