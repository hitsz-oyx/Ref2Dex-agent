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
status: PLANNED
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

Result: pending.

Artifacts: `outputs/P-20260930-cm-inference-bottleneck/r1/`.
