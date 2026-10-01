# Current-policy value diagnostic R3 — FAILED

- **Task:** `T-20261001-current-policy-value-diagnostic-collection-r3`
- **Experiment/run:** `P-20261001-current-policy-value-diagnostic` / `r3`
- **Status:** `FAILED` after CPU admission and s286 simulator startup
- **Science:** `NOT_ASSESSED/AWAITING_CPU_AUDIT`; no V, utility, bias, convergence, or gain claim

## CPU admission and wrapper correction

The new wrapper was tested without Isaac: a missing output root counted as zero, existing support counted, synthetic over-budget and deadline guards passed, and only an owned process group was terminated. The accepted CPU preflight passed (`READY_FOR_ROOT_RUNTIME_TASK`, `isaac_imported=false`, `gpu_started=false`, `collection_started=false`). Both Isaac-free mocks exercised the shared loop and loaded through Episodes (`3,168` rows each).

The first collect invocation exposed a reversible wrapper bookkeeping error: it reran the already completed mock stage and hit `FileExistsError` on `cpu_mock_s286`; it stopped before GPU. The correction reuses immutable CPU evidence and was rechecked. Controls 258 and 260 were applied. The earlier `workflow_start.json` was overwritten by that first collect attempt; this is preserved explicitly. The conservative absolute deadline clamp is epoch `1790836597`, with no budget extension.

## Runtime failure

The corrected launch admitted GPU4 exactly as requested: UUID `GPU-0606f00a-d9d0-3a00-5b49-c9e747b77307`, memory `2 MiB`, and no compute owner. s286 entered Isaac Gym/PhysX from `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/third_party/DExplore`; the native asset path resolved. Before the first episode could be exported, the frozen runtime raised:

```text
AttributeError: 'DefaultRewardsShaper' object has no attribute 'get'
```

The error occurs in `run_current_policy_value_environment.py` while checking `self.config['reward_shaper']`; rl_games supplies a `DefaultRewardsShaper` object. The s286 runtime manifest is `FAILED`. No transitions, `results.json`, shards, complete Episodes, or s287 panel were produced. All owned wrapper/collector/Isaac PIDs are terminal, and the GPU post-check shows no owner on GPU4. This is an engineering runtime failure, not a scientific negative; the frozen collector scripts and scientific inputs were not modified.

## Provenance, cost, and artifacts

Accepted implementation commit: `ff48f6d61fb4e63d2c3525896237971512bfb576`; worker HEAD: `39d0615db896ee66befa41504f605e87a6f9d136`; source origin/current pins are recorded in the JSON. Accepted script hashes remain `collect_current_policy_value.py` `4c5b4687491d9e784ca2ea3f271162114834965990f8c58c4a691aebbc0216de` and `run_current_policy_value_environment.py` `ffe1c061dbd9accb947e248a82cf7c01c2e6afa9229e2949080b9cfdeee474a7`. The s286 simulator interval was about `32.661 s` (manifest start to FAILED completion); the wrapper process ended at epoch `1790836114.885`, before the absolute clamp. Output and support file hashes, modes, bytes, GPU/PID evidence, and all input hashes are listed in the JSON handoff.

## Next step

Do not retry this frozen run. Repair the rl_games `DefaultRewardsShaper` compatibility in a separate CPU/source engineering task, then dispatch a new collection.

## CONTROL 263 extension: R3b

CONTROL 263 authorized only the runtime API guard for rl_games `DefaultRewardsShaper`, with new r3b roots and the unchanged absolute deadline epoch `1790836597`. CPU validation passed for the installed object at unit scale and rejected a scaled object. The modified runtime SHA is `d537eb8a88ba46179ac2d43e4e79b766877419dfa0a05684a54a3234203be97c`; the accepted collector script and all scientific inputs remained unchanged.

R3b s286 entered Isaac/PhysX and passed the guard, then failed at actor forward with `RuntimeError: normal expects all elements of std >= 0.0`. No rows or complete episodes were exported and s287 was not started. GPU4 and all owned PIDs are terminal. This remains an engineering failure, not a scientific negative.
