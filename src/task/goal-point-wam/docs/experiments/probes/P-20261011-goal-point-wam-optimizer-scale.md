---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-point-wam-optimizer-scale
experiment_id: P-20261011-goal-point-wam-optimizer-scale
date: 2026-10-11
task: goal-point-wam
branch: main
git_commit: b59e79b5462be1cbdc93622166b1955f4583d338
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 5
seed_pool: probe
seeds: [247]
decision_changed_if_positive: prepare matched world-loss-off trajectory learning comparison before robot integration
decision_changed_if_negative: stop optimizer-only scaling and diagnose batch/source or decoder representation
status: UNCLEAR
run_id: three-domain-lr1e3-20261011-r1
---

# Matched optimizer-scale Decision Probe

The train-only fixed-window diagnosis established that this WAM can fit six
examples and that lr1e-3 fits them substantially better than1e-4. This experiment
asks if that single change is useful under the actual stochastic mixed-source
training protocol. It serves the same WAM representation subgoal, not a robot
or RL claim. The cheapest test is the original600-update run with only learning
rate changed, followed by a small fixed validation comparison.

## Protocol fixed before execution

- Config `configs/three_domain_no_video_lr1e3_probe.json` differs from the
  prior600-update config only in `learning_rate`: .0001 -> .001.
- Same PW step50000 initialization, seed247, normalization, data, microbatch2,
  warmup50, cosine schedule, drop-path/serialization training mode, optimizer
  settings, clipping, losses,600updates and fixed96-window balanced validation.
- Check initial parameter hash and saved validation indices against the
  completed ref1 alignment run. No fixed-window fitted weights are reused.
- Best selection remains world macro h24 EPE + trajectory macro EPE, on the
  validation panel only. All diagnostic arms use these saved validation indices.
- Compare the higher-rate best checkpoint, the old-rate best checkpoint replay,
  and static world/hand plus a simple linear start-to-goal hand baseline.
  No test loader or test outcome is used to select a setting.
- PROMISING requires >=5% moving-world improvement over static on at least
  two validation sources, and no source's trajectory regresses >10% versus
  the old-rate checkpoint on the same windows. Otherwise UNCLEAR; a visibly
  worse matched result closes this optimizer-only recipe, not full WAM.
- One idle GPU2, <=15minutes including evaluation, <=2GiB new artifacts;
  stop on input drift, nonfinite/OOM, goal isolation >1e-5, resource conflict
  or deadline. Preserve earlier results; no new branch/download/simulation.

## Decision Note

Fixed-input fitting is PROMISING and eliminates a useful failure hypothesis,
so this is a new discriminating optimizer decision rather than more epochs
without a signal. Only learning rate changes; the new original-recipe run
separates that factor from fixed batch/eval-mode fitting. Validation results
are exploratory because this panel already informed previous model selection.
No final claim or resource/permission boundary changes.

## Result

Completed 600 updates on GPU2 in 287.2 seconds; best checkpoint is update500.
The initial parameter hash and all 96 validation indices match the old-rate
run exactly. The config differs only in learning rate. All measurements are
finite, final training world goal-isolation max is 2.30e-7; sampled peak memory
is 1513MiB and run artifacts total about1.8GiB. Evaluation uses only val windows,
32 per source, at evaluator commit `5cca22d3`.

Average 24-step trajectory EPE (mm):

| Source | Old lr1e-4 | New lr1e-3 | Static hand | Linear start-to-goal |
| --- | ---: | ---: | ---: | ---: |
| OakInk2 | 35.24 | 35.30 | 34.80 | 9.16 |
| GRAB | 141.70 | 141.41 | 141.32 | 67.14 |
| ARCTIC | 71.54 | 71.97 | 71.63 | 34.20 |

Moving-world h24 EPE (mm):

| Source | Old lr1e-4 | New lr1e-3 | Static world | New gain over static |
| --- | ---: | ---: | ---: | ---: |
| OakInk2 | 23.70 | 23.75 | 23.65 | -.429% |
| GRAB | 160.71 | 161.23 | 161.26 | +.017% |
| ARCTIC | 87.01 | 87.98 | 87.85 | -.157% |

No source reaches the predeclared5% moving-world improvement. Trajectory changes
versus old weights are +.185%/-.208%/+.596%, well below the10% regression limit,
but essentially comparable to static hand and worse than the linear-goal proxy.
The positive gate fails, so the Probe remains **UNCLEAR**. Strong fixed-batch
fitting did not transfer into a useful short mixed-source result.

Root stops optimizer-only scaling and extra epochs, and chooses a train-only
forward-mode/goal-response diagnosis recorded in the learnability card. This
closes this short optimizer recipe, not the overall WAM hypothesis, and does
not establish action-conditioned world modeling or final RL utility.

## Artifacts

- [Matched validation results and identities](../../../../../../outputs/goal-point-wam/three-domain-lr1e3-20261011-r1/matched_validation.json)
- [Higher-rate best parameters](../../../../../../outputs/goal-point-wam/three-domain-lr1e3-20261011-r1/best.pt)
- [Single-variable config](../../../configs/three_domain_no_video_lr1e3_probe.json)
- [Validation-only evaluator](../../../tools/audit/evaluate_validation.py)
- [Train-only follow-up card](P-20261011-goal-point-wam-learnability.md)

## Limitations / future evidence

Single seed and previously used validation windows cannot establish generalization
or formal utility. A positive world predictor does not establish world-supervision
benefit to trajectories: that requires a matched world-loss-off arm with fair
checkpoint selection. The trajectory endpoint proxy remains simpler than an
independent robot goal. Held-out Validation and actual RL benefit stay deferred.
