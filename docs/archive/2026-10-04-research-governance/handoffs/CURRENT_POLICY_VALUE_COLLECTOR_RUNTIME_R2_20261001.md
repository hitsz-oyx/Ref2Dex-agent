# Current policy value collector runtime R2 — 2026-10-01

## Result

`T-20261001-current-policy-value-collector-runtime` is **COMPLETED** as an engineering implementation task. The real `--collect` path is implemented but was not executed: no GPU, Isaac Gym, simulator, collection, training, PPO, or online process was started.

## Implementation

- `scripts/collect_current_policy_value.py` is now an Isaac-free thin entry. `--collect` delegates to the independent runtime wrapper; `--mock` exercises the same runtime loop without Isaac.
- `scripts/run_current_policy_value_environment.py` imports Isaac Gym before torch only in the real runtime process. It does not call the old source `main()` collect guard, does not modify the source runner, and validates the accepted e420 checkpoint hashes before runtime.
- The runtime loop records actor-forward `value_at_state`, no-random `eval_critic` `next_value` with terminal zero, shared simulator `global_tick`, `horizon_phase`, true int64 episode IDs, pre-reset terminal next state, previous action, reward components, done reasons, and per-episode stable/drop/hold/lift/contact diagnostics.
- It enforces the shared two-checkpoint workflow wall, rows, output, path, provenance, and no-overwrite guards. A second split is admitted only under the same root provenance and a fresh checkpoint child directory.

The source scientific inputs are pinned by origin ancestor `f3ae0dfd6072f0d3cde8112f71574828b6a5a4bc`, current source HEAD observed at preflight `20f6937202fe7d36b045396e2649d3a67ef5ba7c`, runner SHA `1fc914f75b5579078eea533660a57f2c820414995da18cd7304858054ab26a57`, and the checkpoint/config/motion/native-manifest hashes recorded in the JSON handoff.

## CPU evidence

```text
python3 -m pytest -q tests/test_current_policy_value_collection_contract.py
6 passed

python3 -m py_compile scripts/collect_current_policy_value.py scripts/run_current_policy_value_environment.py tests/test_current_policy_value_collection_contract.py
PASS

git diff --check
PASS
```

The mock `--collect` path emitted 3,168 rows from 96 complete first episodes, with int64 disjoint episode IDs, `global_tick` 0..32, `horizon_phase` 0..31, terminal `next_value=0`, original reward components, and the source `Episodes` reader accepted the shard. Source preflight returned `READY_FOR_ROOT_RUNTIME_TASK` with `isaac_imported=false`, `gpu_started=false`, and a 4-row/2-episode reset-boundary fixture. Existing preflight MD/JSON were preserved byte-for-byte (SHA-256 in JSON).

Exact changed-file SHA-256, mode, and byte sizes are in `CURRENT_POLICY_VALUE_COLLECTOR_RUNTIME_R2_20261001.json`. No Git commit was made because linked Git metadata is not writable; unrelated worker modifications were preserved.

## Reproducible CPU commands

```bash
python3 scripts/collect_current_policy_value.py --preflight --run-root /tmp/current-policy-value-preflight-r2-20261001b
python3 scripts/collect_current_policy_value.py --collect --mock --checkpoint-seed 286 --run-root /tmp/current-policy-value-runtime-mock-loop-20261001
python3 -m pytest -q tests/test_current_policy_value_collection_contract.py
```

Only a later authorized Broker run task may invoke real `--collect`; it must select an idle physical GPU by UUID and compute-PID ownership and use a new direct child output root.
