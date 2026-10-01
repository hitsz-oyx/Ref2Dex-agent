# P-20261001 current-policy value diagnostic

## Decision questions

1. **What is being distinguished?** This diagnostic separates insufficient fit of the frozen physical-value model to the online frozen-critic lambda target from a difference between that target and complete first-episode Monte Carlo return distributions. The collector preserves the actor's PPO critic value, reward components, terminal mask, and episode boundaries so those errors can be compared without training the value model.
2. **What decision can the result change?** A lambda-target-specific error points to the next target or bootstrap check; a broad Monte Carlo distribution mismatch points to adaptation or representation checks. A result that does not separate them leaves the current Cm utility route unchanged and only identifies the next diagnostic.
3. **What is the cheapest discriminating method?** Reuse the two frozen s286/s287 e420 policies for one complete frame-0 first episode per environment: 96 environments per checkpoint, three motions with 32 environments each, common seed 290. This is a retrospective Decision diagnostic, not a new training sweep.

## Fixed scope and boundaries

- The HF08 utility slot remains closed at 1/1. This probe does not tune, retrain, or claim utility, convergence, bias, or a scientific gain.
- The actor is sampled Gaussian with the accepted PPO preprocessing and clip, `enable_eps_greedy=false`, no extra noise, teacher, or parameter update. Frame 0 is a fixed diagnostic sub-distribution, not the full training reset distribution.
- Both e420 checkpoints are collected sequentially under one 1200 second, one GPU, 2 GiB, 192000-row workflow budget (96000 rows per checkpoint; MAX_STEPS=700). The output and support roots are new direct worker outputs and are never reused.
- Each row retains the same-forward PPO critic value, `next_value`, shared vector environment `global_tick`, phase modulo 32, previous action, reward components, terminal-before-reset next state, and int64 checkpoint-disjoint episode ID. Native stable/drop/hold diagnostics remain per episode.

## Provenance and reproducibility

- Worker HEAD at dispatch: `39d0615db896ee66befa41504f605e87a6f9d136`.
- Root accepted implementation commit: `ff48f6d61fb4e63d2c3525896237971512bfb576`.
- Executable SHA-256 is checked before and after runtime: `scripts/collect_current_policy_value.py` = `4c5b4687491d9e784ca2ea3f271162114834965990f8c58c4a691aebbc0216de`; `scripts/run_current_policy_value_environment.py` = `ffe1c061dbd9accb947e248a82cf7c01c2e6afa9229e2949080b9cfdeee474a7`.
- Runtime uses `/home2/wyy/miniconda3/envs/graspenv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, and an LD library path beginning with `/home2/wyy/miniconda3/envs/graspenv/lib`. Any torch extension cache is confined to the new support root.

Collection status and scientific analysis are recorded separately in the terminal handoff. Collection completion only establishes the data contract; `agent_cm` owns subsequent value analysis.

## R2 cwd-correction run plan

R2 is the same experiment and retrospective diagnostic with a new run id after the R1 launch-cwd engineering failure. The frozen tools, checkpoints, motion panel, frame 0, seed 290, Gaussian policy, reward, schema, and 96-environment first-episode contract remain byte-identical. The only launch correction is executing the accepted absolute scripts from `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/third_party/DExplore`, where the source-relative `dexplore/data/assets` path resolves; no source or tool file is edited.

The R2 whole-panel deadline is 1100 seconds and the remaining combined artifact budget is 2,143,928,431 bytes after R1's 3,555,217 bytes. A two-stage process group will run s286 then s287, with `MAX_JOBS=2`, torch/OMP/MKL threads set to 2, and support-local cache/log directories. GPU 4 is selected only after exact UUID and compute-PID ownership checks. Any hash drift, ownership conflict, incomplete episode, nonfinite value, budget violation, or runtime exception terminates R2 without retry. Scientific status remains `NOT_ASSESSED/AWAITING_CPU_AUDIT` until `agent_cm` validates the collected panels.

## R2 terminal outcome (2026-10-01)

The cwd-correction attempt is `FAILED` before GPU ownership validation or simulator startup. CPU preflight passed from `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/third_party/DExplore` with `isaac_imported=false`, `gpu_started=false`, and `collection_started=false`; its 4-row/2-episode boundary fixture passed. The new two-stage wrapper then exited at its first `check_budget()` call because line 29 ran `du -sb "$OUTPUT_ROOT" "$SUPPORT_ROOT"` while the deliberately new output root did not yet exist, under `set -euo pipefail`. This is an engineering wrapper exception, not a scientific result. No `collect_s286`, `collect_s287`, Isaac, GPU, or output-root process started; the registered PID/PGID 359122 is terminal. The GPU post-check records GPU4 UUID ownership empty. No retry or method change was made. R2 science remains `NOT_ASSESSED/AWAITING_CPU_AUDIT`; a separate engineering task must repair the wrapper before a new run.
