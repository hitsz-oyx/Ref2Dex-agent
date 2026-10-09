---
schema: ref2dex.probe.v2
probe_id: P-20261009-ref13-progress-recovery
experiment_id: P-20261009-ref13-progress-recovery
date: 2026-10-09
task: cm-interaction-oracle
branch: main
git_commit: 96e7770
claim_id: C3
hypothesis_family: HF-ref13-progress-recovery
probe_index_in_family: 1
seed_pool: probe
seeds: [263]
decision_changed_if_positive: compare old utility and frozen physical-reference delta-progress in actual rolling control under the recovered contract
decision_changed_if_negative: retain execution or baseline failure without treating it as negative evidence for either Y
status: UNCLEAR
run_id: ref13-progress-recovery-20261009-r3
---

# Recover ref13 whole-world cold replay before replacing Y

## Decision Note

User explicitly requested reuse of the old Oracle method with a new Y. This
Blocker/Decision Probe asks whether the historical Torch2.4.1, GPU PhysX,
CPU tensor pipeline, single-thread, 96-env whole-world cold replay can provide
both usable grasp opportunities and paired candidate states. It serves Mission
C3 by separating execution validity from GT value decision utility.

The recent host diagnostics used Torch2.0.1, forced frame0/hybrid probability1,
and across-env broadcast controls. They are not a complete reproduction of
ref13's Torch2.4.1, hybrid probability0.5 and per-env feedback with corresponding
envs compared across fresh whole-world simulations. Reopen only this concrete
contract; do not repeat the recent across-env group design.

Original actor SHA16fd261b and complete original panels are unavailable. The
available reconstructed self-trained e260 has SHA8882fabd and s3-only training
ancestry; all arms use this same frozen endpoint. Three canonical motion names
are recovered from currently preserved converted assets, with their new hashes
recorded. This is an experiment-method recovery, not an original-weight/motion
replication, and no old23/32 expectation is imposed on the new baseline.

## Frozen staged protocol

1. Use `graspenv` Torch2.4.1+cu121/rl_games1.6.5, GPU PhysX, CPU tensor
   pipeline, `num_threads=1`, 96envs, seed263, r7 environment/training configs.
   Keep native hybrid probability0.5 and each env's own reactive actor control.
   Decode a uniform torch.compile prefix only; keep model/RMS tensors frozen.
2. Fresh baseline and corresponding-env zero-repeat use the historical
   `collect_oracle_y_candidates.py`. Save complete initial inventory, RNG,
   measured geometry and actual controls. Compare old prefix/outcome contracts;
   never substitute different env instances for matched simulations.
3. If repeat passes and current-only early-hold support exists, freeze a small
   synchronous s3 group from baseline current6<=hold<45, history/proximity,
   all-seven-action headroom and complete horizon. s3 is the only current
   physical-reference bank motion; do not score s7/s9 with that bank.
4. Recover seven native candidates, each residual for8steps followed by native
   continuation. Old U uses original32-step labels; new Y uses exactly
   P[t+24]-P[t] from actual causal prefix and measured future. Freeze the existing
   eight-reference bank/TCC; do not use future beyond t+24 for new Y. Remove
   the old U<=1.25 shortcut for new-Y selection. Exact score ties use baseline
   then candidate order for both arms; report any different deadzone separately.
5. Only valid candidates can feed separate real mixed execution and subsequent
   eight-step replans. Initial bounded rolling screen has three replans on one
   group, with old Z90 unchanged across baseline/old-Y/new-Y. This is a Probe
   of Y substitution, not full-episode Gate1, formal Validation or Cm utility.

## Resources and stopping

Use one idle GPU2 initially, no policy/evaluator/world-model training. Engineering
has a480s native-worker cap; conditional candidate/rolling work gets2400s total
wall and4GiB artifacts. Each worker has180s native/240s shell limits; one worker
at a time, preserve failed runs and stop on changed inputs, replay mismatch,
missing complete window, GPU conflict, absent baseline opportunities or budget.
All outputs stay under `outputs/cm-interaction-oracle/`; no video/checkpoint
overwrite, external writes, new branch or remote push. Record GPU memory and
utilization during execution. Small metadata/hash/statistics use CPU; TCC
inference uses the idle GPU. Actual run commits are in each manifest.

## Limitations / future evidence

Reconstructed policy/motions prevent a pure historical before/after attribution.
Within this run, actor/motion/execution are matched across Y arms. One s3 group
does not satisfy the old>=30anchors/>=2motions screen; smaller results remain
UNCLEAR or engineering-only. Independent cohorts, broader successful reference
banks, full-episode outcome and formal Cm-on/off training remain later work.

## Engineering results

At run commit96e7770, baseline completed652ticks in118.43s shell/100.85s
native and collected45anchors (31s3/14s7/0s9). Repeat completed in91.30s shell:
45/45 accepted by the unchanged historical screen (prefix tolerance, Y max
difference<=.05 and matching Z; acceptance>=80%). Actual before/history,
actor observation, hand root, height, pair, first8controls and PD targets are
bitwise equal, Y maximum difference0, Z disagreement0 and Z90 both33/45.
The repeat path did not retain a full trace, so full-world geometry/hidden-state
identity is not claimed. Each rolling worker separately enforces its full-world
prefix contract and scoring checks measured geometry through the query.

Current-only synchronous support selects25s3envs at tick71, retaining>122future
steps. Reanchor-r2 has zero recorded prefix errors and baseline Z90=20/25.
GPU2 observed approximately7.45GiB and21--38% GPU utilization. No training.
All35 candidate panels and six actual mixed executions completed. All rolling
workers passed the full-world prefix contract, and measured object/11-hand-point
prefixes through each query were bitwise equal within each seven-candidate
comparison. Frozen causal progress starts matched to1e-12. Initial interventions
were realized to<=1.49e-8 residual error, with zero clipped steps and mean
step8 fingertip displacements9.69--28.05mm across alternative arms.

Startup failures are preserved: r1 hit Python3.8 Path.is_relative_to, r2 had a
relative worker path, and reanchor first attempted a rolling-only record flag.
These failed before simulation and do not count as valid scientific Probes.
Fix commitsb9f1c22/96e7770/ead0dc5; all actual workers record their exact commit.

Budget adaptation: observed native worker shell cost35--41s and frozen
reference scoring cost roughly10s/candidate. Four replans for both Y arms
would exceed the2400s total wall budget. Root reduces the matched screen to
three decisions at offsets0/8/16, retains actual90 continuation at the first
decision to report one-shot independently, and actual90 at the final decision
for rolling Z. This preserves the initial Y comparison and complete outcomes;
a fourth replan is deferred unless the smaller screen changes the decision.
No resource boundary, Y, actor, candidate or outcome threshold is changed.

## Completed Y comparison

| Actual Z90 | Baseline | Old Y | New delta-progress Y |
|---|---:|---:|---:|
| One-shot residual8 + native continuation |20/25|21/25|22/25|
| Three real decisions at offsets0/8/16 |20/25|23/25|22/25|

The first decision differed on21/25rows: old Y selected22baseline/3alternative,
new Y selected3baseline/22alternative. Old rolling rescued3 and harmed0;
new one-shot and rolling both rescued2 and harmed0 relative to baseline, with
the same final success set. New rolling versus old rolling lost one rescue
(0rescues/1harm in that paired comparison). The extra successes all belong
to frame0 initialization: baseline12/17, old one-shot13/17, new one-shot14/17,
old rolling15/17, new rolling14/17; all8hybrid-nonzero anchors succeeded in
every arm. Thus the observed gains are not from favorable mid-motion starts.

Exploratory env bootstrap rolling gains: old +.12 [0,.28], new +.08 [0,.20],
new versus old -.04 [-.12,0]. Only one seed/motion and25anchors; no interval
or raw count upgrades this to Validation, episode-wide success or Gate1 pass.
The saved `result-summary.json` keeps statusUNCLEAR and full_episode_gate1=false.

Exact argmax was kept for both arms to isolate the Y replacement. As a separate
descriptive diagnostic only, the preexisting progress-selector .01 deadzone
would select7 initial alternatives rather than22; median new-Y advantage over
baseline was.00264. No deadzone execution was run or tuned after seeing Z.

Conditional work finished in approximately2231s (cap2400), with44 completed
native workers across engineering/reanchor/candidate/actual stages and1
pre-simulation startup failure in r3. Artifacts approximately1404MiB (cap4GiB),
oneGPU2; GPU released. Code contracts passed17targeted tests. Run manifests
pin each worker commit and immutable actor/RMS/config/motion/source inputs.

Decision: recover the old whole-world method as a viable engineering route.
New Y has a small positive one-shot signal against baseline in this cohort;
three replans add no gain and do not outperform old Y rolling. Do not close
the new-Y hypothesis, enlarge this run, fit evaluator/PointWorld, or claim
Gate1. Next decision experiment should use the recovered execution contract
for a full-episode matched new-Y/baseline screen, with its selector/tie rule
specified before execution; broader references/cohorts remain future evidence.

Artifacts: `outputs/cm-interaction-oracle/ref13-progress-recovery-20261009-r3/`
contains `engineering-audit.json`, `initial-scores.json`, per-decision scores,
`initial-intervention-audit.json`, `one-shot-summary.json`, `rolling-status.json`,
`result-summary.json`, `resource-summary.json` and pinned native manifests/logs.
