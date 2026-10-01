# Current-policy value diagnostic R2 — FAILED

- **Task:** `T-20261001-current-policy-value-diagnostic-collection-r2`
- **Experiment/run:** `P-20261001-current-policy-value-diagnostic` / `r2`
- **Terminal status:** `FAILED` (engineering wrapper exception before GPU ownership validation)
- **Scientific status:** `NOT_ASSESSED/AWAITING_CPU_AUDIT`; no value, utility, bias, convergence, or gain claim.

## What ran

CPU preflight ran from the corrected cwd `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/third_party/DExplore` using `/home2/wyy/miniconda3/envs/graspenv/bin/python`. It passed with `status=READY_FOR_ROOT_RUNTIME_TASK`, `isaac_imported=false`, `gpu_started=false`, and `collection_started=false`; the meaningful fixture validated 4 rows, 2 episodes, and reset-boundary preservation. The preflight artifact is `outputs/P-20261001-current-policy-value-diagnostic-r2-support/preflight.json`.

The bounded two-stage wrapper then exited before GPU ownership validation. At line 29, `set -euo pipefail` treated `du -sb "$OUTPUT_ROOT" "$SUPPORT_ROOT" | awk ...` as a failure because the deliberately new output root did not exist. This is a wrapper engineering failure. The output-root guard had passed, and no `collect_s286`, `collect_s287`, Isaac, GPU, or simulator process started. The registered orchestrator PID/PGID was `359122`; terminal evidence is in `process_terminal_r2.json`. No retry or method change was made.

## Inputs and frozen implementation

Accepted implementation commit: `ff48f6d61fb4e63d2c3525896237971512bfb576`; worker HEAD at dispatch: `39d0615db896ee66befa41504f605e87a6f9d136`; source origin pin: `f3ae0dfd6072f0d3cde8112f71574828b6a5a4bc`. Accepted executable SHA-256 values were checked in preflight: `collect_current_policy_value.py` `4c5b4687491d9e784ca2ea3f271162114834965990f8c58c4a691aebbc0216de`, and `run_current_policy_value_environment.py` `ffe1c061dbd9accb947e248a82cf7c01c2e6afa9229e2949080b9cfdeee474a7`. The preflight JSON pins the environment, training, runner, checkpoint, manifest, and three motion hashes.

## Resource and artifact evidence

- Whole R2 budget: 1100 seconds; no completed runtime duration, and the wrapper terminated before the GPU phase (post-check observed within 174 seconds of the recorded start).
- Combined artifact budget remaining after R1: 2,143,928,431 bytes; R2 support artifacts total 12,714 file bytes (25,002 bytes as `du -sb` directory usage) at handoff; output root is absent (0 bytes).
- GPU4 UUID `GPU-0606f00a-d9d0-3a00-5b49-c9e747b77307` had no compute owners in the post-check; other observed compute processes were not touched.
- Rows/episodes: s286 `0/0`, s287 `0/0`; manifests/results/shards and Episodes validation are `NOT_RUN_NO_OUTPUT`.
- Exact per-file SHA-256, mode, and byte sizes are in the JSON handoff.

## Next step

Do not retry this frozen invocation. Repair or replace the bounded wrapper in a separate engineering task, then dispatch a new diagnostic run. This R2 failure is not a scientific negative.
