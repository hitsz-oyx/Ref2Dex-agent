---
schema: ref2dex.probe.v2
probe_id: P-20260930-cm-inference-bottleneck
date: 2026-09-30
branch: agent/cm-scratch-mlp-policy-probe
git_commit: see parent run_manifest.json
claim_id: C3
hypothesis_family: HF07
probe_index_in_family: 1
seed_pool: probe
status: COMPLETED
---

# Real-policy Probe: explicit physical prediction input

Decision and frozen mechanism are in
[memo](../../decisions/D-20260930-cm-inference-bottleneck.md).

Distinguish useful physical prediction input from no extra information,
generic untrained projection, and extra teacher-action information. Same
source_e260 substrate, three canonical airplane motions, BC architecture,
weight initialization, fit rows, updates and evaluation seeds. Physical
predictor was fit on randomized support; BC fit/holdout use fresh simulator
seeds 279/280. All first completed episodes are retained and later episodes
excluded from BC. Four actors each fit seed278, 2000 steps, batch512.

Evaluate seeds281/282, 64 environments each, disabled early termination.
Primary metric: held-lift, evaluator definition dz >=0.03 m and five
consecutive hand/object-contact steps. Validate pairing on environment,
motion, start frame and episode length before interpreting differences.
Upgrade requires on-minus-each-control >=5pp overall and both seed-level
on-minus-off nonnegative. No tuning based on BC holdout or policy evaluation.

Budget: 1 idle GPU, all phases <=60 minutes, <=5 GiB, no video. Stop on
input drift, nonfinite tensor, incomplete first episodes, GPU conflict,
invalid pairing or budget. Parent manifest and each native manifest retain
commands, input hashes, execution commit and execution status.

Scope: one BC training seed, fixed airplane subset, expert remains required
at inference. Only a Probe label is permitted. A positive result motivates
multi-training-seed matched Validation; a negative result closes HF07.

Result: `UNPROMISING`; execution `COMPLETED` on code
`fadcd72d7478a9c6d6093154a9c7c0a8d6c497a5` (run `r2`).

| Evaluation seed | Cm-on | Cm-off | Random features | Teacher-action features |
| --- | --- | --- | --- | --- |
| 281 | 10/64 | 7/64 | 12/64 | 5/64 |
| 282 | 8/64 | 11/64 | 9/64 | 9/64 |
| Total | 18/128 | 18/128 | 21/128 | 14/128 |

On-minus-off 0pp, on-minus-random -2.34375pp, on-minus-action +3.125pp;
none reaches the fixed +5pp gate, and seed282 is negative versus off.
All eight native arms completed on the same execution commit. Unique env ids,
motion/start-frame/episode-length pairing and matched actor initial weights
passed. Fit retained 34,608 rows; holdout 34,706 rows, first-episode-only.
An independent audit matched fit row counts to native episode lengths and
verified source supervision equalled executed actions. Cumulative wall time
including technical retry was 1,188.65 seconds, one GPU; all child processes
exited and GPU memory returned to idle. Output bytes and hashes are in the
[result index](P-20260930-cm-inference-bottleneck-results.json).

The first run `r1` failed the pairing gate: random feature construction used
global `torch.manual_seed` inside a CPU-only RNG fork, resetting CUDA start
frame sampling in 62/64 environments. Its physical results are invalid and
excluded. A private CPU generator fixed the cause; the CUDA RNG regression
test failed before and passed after. The completed source collections were
unaffected and explicitly reused with hashes; all four students and all
eight physical arms were rerun under the fixed commit, not selected by score.

Decision: close HF07; no Validation and no local width/seed/step/target sweep.
This rejects this fixed BC inference-input implementation, not the core Cm
hypothesis or an RL-trained bottleneck. Cm policy utility remains OPEN.

Artifacts: `outputs/P-20260930-cm-inference-bottleneck/r2/`; the explicit
source collections and invalid original run are preserved under `r1/`.
