---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-point-wam-ref1-alignment
experiment_id: P-20261011-goal-point-wam-ref1-alignment
date: 2026-10-11
task: goal-point-wam
branch: main
git_commit: 412c6aef
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 3
seed_pool: probe
seeds: [247]
decision_changed_if_positive: prepare a matched world-loss-off trajectory control before robot integration
decision_changed_if_negative: diagnose action conditioning and representation before longer training
status: UNCLEAR
run_id: ref1-alignment-20261011-r1
---

# Ref1 alignment: pretrained scene and full executed-action memory

Decision Probe serving the WAM subgoal of a usable action-conditioned interaction
representation, with eventual matched RL utility still required by the Mission.
Question: after fixing the two known deviations, can a bounded run learn object
motion beyond a static baseline while learning the teacher-goal trajectory?
The cheapest decision experiment is four engineering updates followed by 600
updates using the existing seed 247, data draw, optimizer and validation recipe.
Seed 247 is reused within the existing family to make this an implementation
follow-up, not an independent repeat.

## Protocol and predeclared decision

- Initialize PTv3 and scene projection from repaired sharedstats temporal PW
  `training-r1/train-action/final.pt`, step 50000; new WAM heads/object projection.
- Strict config, normalization buffer/hash, vendor hash and key/shape checks;
  parent SHA and transferred tensor identity are recorded in each run manifest.
- World receives all 528 time/hand/keypoint action tokens and masks, after scene
  encoding. No goal or future action enters PTv3; world remains goal-blind.
- Original learning recipe: 600 updates, microbatch 2, AdamW, seed 247, fixed
  balanced 96-window validation. Best = world macro EPE + trajectory macro EPE.
- Fixed diagnostic panel: 96 windows/source, seed 250, OakInk2/GRAB test and
  ARCTIC val (the frozen corpus has no ARCTIC test). Compare exact same windows
  against identity/no-motion world output. Report counts and moving/static EPE.
- PROMISING requires at least 5% moving-world EPE improvement over static on
  two sources, with finite trajectory metrics and no more than 10% trajectory
  regression against the previous diagnostic run on any source. Otherwise
  UNCLEAR unless convincing contradictory mechanism evidence emerges.
- Engineering gate: finite loss/gradients, both losses reach shared scene
  projection, goal/world isolation <=1e-5 and checkpoint replay <=1e-5.
- One idle physical GPU2, at most 20 GPU minutes total, 5 GiB new artifacts.
  Deadline-bounded runs; stop on nonfinite/OOM/contract drift/resource conflict.
  Preserve original runs. No new branch, downloads, robot/video/PD work.

## Result

Implementation fixed at `412c6aef`. Contract tests: 5 passed / 1 skipped.
Engineering run `ref1-alignment-smoke-20261011-r1` precedes the registered
600-update run; both use physical GPU2 and the same input checkpoint/stats.
The 4-update engineering gate completed: finite training, both branch gradients
reach the shared projection (trajectory `1.2402e-4`, world `8.4805e-5`),
checkpoint replay max `4.1972e-8`, and goal/world isolation max `2.1101e-10`.
Strict transfer retained 454 tensors / 50,436,224 parameters from PW step 50000;
the total WAM has 52,258,730 parameters. This is engineering evidence only.
The registered run completed all 600 updates in 273.4 seconds on physical GPU2.
Best checkpoint selection used only the unchanged 96-window validation panel
and selected update 400. All logged losses/gradients were finite; the final
checkpoint goal/world isolation max was `1.391e-6`, below `1e-5`.
Sampled GPU utilization was 3--26%, peak memory 1513 MiB, and median update
0.2495 seconds. The small fixed microbatch was retained for comparability;
the run completed faster than the historical 327-second run. These samples
are not a profiler or evidence of an architecture speedup. Smoke plus training
produced about 3.6 GiB, within the 5 GiB artifact limit.

The source panel compares the same 96 windows/source across all three arms.
The earlier archived diagnostic used different windows; the previous model is
therefore replayed here, and its archived scalar values are not subtracted.
Trajectory EPE is averaged across 24 future steps; world EPE is at h24.
Errors below are millimetres:

| Source / split | Trajectory aligned / previous | World aligned / previous / static | Moving world aligned / static | Moving gain |
| --- | ---: | ---: | ---: | ---: |
| OakInk2 / test | 33.78 / 33.71 | 9.10 / 9.87 / 8.28 | 30.30 / 30.65 | 1.14% |
| GRAB / test | 152.95 / 153.62 | 148.42 / 149.29 / 149.18 | 199.96 / 201.57 | 0.80% |
| ARCTIC / val diagnostic | 72.10 / 72.29 | 68.80 / 69.28 / 68.93 | 93.10 / 93.61 | 0.55% |

There were 159/597, 71/96 and 141/192 moving/total objects, respectively;
all arms have identical counts. No source met the predeclared >=5% moving
improvement gate. Trajectory changed by +0.20%, -0.43%, -0.26%, respectively,
so it passed the no-regression gate but did not establish a trajectory benefit.
Conclusion: **UNCLEAR** for this bounded pretrained/full-action WAM Probe.
The two structural deviations are corrected, yet this run does not demonstrate
useful world prediction or WAM utility. It does not refute the full architecture
or the final Cm/RL claim.

### Execution correction and decision

The initial evaluation invocation failed before measuring any panel because
`MixedWindows.source_names` is not part of the loader contract. Commit
`f1eb97b7` obtains names from `meta['sources']`; the complete rerun succeeded,
with original training/checkpoints preserved. Both evaluator logs are retained.
The training code remains fixed at `412c6aef`; the panel records its actual
corrected evaluator SHA.

Root stops extra epochs and robot/video integration for this experiment. The
next cheapest Decision Probe is a bounded train-only fixed-batch learning
check, distinguishing inability to learn the present target from insufficient
short-run/source coverage. Only after a positive learning signal should a
matched world-loss-off trajectory comparison be prioritized. No new fitting,
seed, download or native simulation was launched after this result.

## Limitations and future evidence

Two implementation corrections change together; this cannot attribute any gain
specifically to pretraining versus full action tokens. Temporal PW weights are
reused under WAM's scene-only spatial forward, so PW accuracy is not inherited.
Endpoint hand goals and rigid outputs remain the approved phase proxies.
A matched world-loss-off control is necessary to establish trajectory benefit;
multi-seed validation and actual self-trained RL utility remain deferred.
CPU is used only for small contract tests and checkpoint weight/hash transfer
checks; neural training and batch evaluation use GPU.

## Artifacts

- [Engineering result](../../../../../../outputs/goal-point-wam/ref1-alignment-smoke-20261011-r1/result.json)
- [Engineering input identity](../../../../../../outputs/goal-point-wam/ref1-alignment-smoke-20261011-r1/input_manifest.json)
- [Previous simplified Probe](P-20261011-goal-point-wam-three-domain-probe.md)
- [Completed training result](../../../../../../outputs/goal-point-wam/ref1-alignment-20261011-r1/result.json)
- [Training input identity](../../../../../../outputs/goal-point-wam/ref1-alignment-20261011-r1/input_manifest.json)
- [Same-window diagnostic and saved indices](../../../../../../outputs/goal-point-wam/ref1-alignment-20261011-r1/matched_diagnostic_panel.json)
- [Gate and resource summary](../../../../../../outputs/goal-point-wam/ref1-alignment-20261011-r1/decision_summary.json)
- [Best checkpoint, update 400](../../../../../../outputs/goal-point-wam/ref1-alignment-20261011-r1/best.pt)
- [Failed evaluation log](../../../../../../outputs/goal-point-wam/ref1-alignment-20261011-r1/evaluator-first-attempt.log)
- [Completed evaluation log](../../../../../../outputs/goal-point-wam/ref1-alignment-20261011-r1/evaluator-completed.log)
