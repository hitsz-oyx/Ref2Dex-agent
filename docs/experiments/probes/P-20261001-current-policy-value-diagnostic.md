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

## R3 terminal outcome (2026-10-01)

R3 is `FAILED` after CPU admission and GPU initialization, during s286 before the first transition export. The wrapper CPU correction was validated: a missing output root counts as zero bytes, existing support is counted, synthetic over-budget and past-deadline checks stop, and only an owned sleep process group is terminated. Both Isaac-free mock panels were Episodes-compatible (3,168 rows each). The first `--collect` invocation exposed and recorded a wrapper bookkeeping error (`cpu_mock_s286` already existed); it stopped before GPU. Under CONTROL 258/260, the wrapper was corrected without changing frozen collector code, and the absolute deadline was conservatively clamped to epoch `1790836597`.

The corrected invocation admitted GPU4 UUID `GPU-0606f00a-d9d0-3a00-5b49-c9e747b77307` with no compute owner and entered Isaac/PhysX. s286 failed before any complete episode because the frozen runtime's `PhysicalPlayer` config contains an rl_games `DefaultRewardsShaper` object, while `run_current_policy_value_environment.py` calls `.get` on it: `AttributeError: 'DefaultRewardsShaper' object has no attribute 'get'`. The runtime manifest is FAILED; no transitions/results/shards were exported, s287 was not started, and all owned PIDs are terminal. This is an engineering runtime failure, not a scientific negative. The accepted scripts and all scientific inputs remain byte-identical. Science remains `NOT_ASSESSED/AWAITING_CPU_AUDIT`; do not retry this frozen run without a separate runtime compatibility repair.

## R3b terminal outcome under CONTROL 263

CONTROL 263 authorized only the API-compatible `DefaultRewardsShaper` guard, with new r3b output/support roots and the same conservative epoch `1790836597` deadline. The modified runtime accepts a mapping or installed `DefaultRewardsShaper(scale_value=1)` and rejects non-unit scale; the CPU object test passed. The modified runtime SHA is recorded in the R3 handoff, while the collector script and all scientific inputs remain unchanged.

The r3b s286 launch reached Isaac/PhysX and passed the guard, then stopped at the first actor forward because the checkpoint produced a non-positive Gaussian standard deviation: `RuntimeError: normal expects all elements of std >= 0.0`. No row, complete episode, result, or s287 panel was exported. GPU4 was released and all owned PIDs are terminal. This is an engineering/runtime failure; no additional retry or sampling modification is allowed. Science remains `NOT_ASSESSED/AWAITING_CPU_AUDIT`.
