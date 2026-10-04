# Native player contract repair — CPU READY

- **Task:** `T-20261001-current-policy-native-player-contract`
- **Status:** `READY_FOR_COLLECTION` after CPU-only validation
- **Scientific status:** `NOT_ASSESSED`; no simulator, GPU, transitions, V calibration, utility, or convergence claim

## Repair

The exact pre-task runtime was `d537eb8a88ba46179ac2d43e4e79b766877419dfa0a05684a54a3234203be97c`. Its actor path called the raw `a2c_network` directly, skipped the model's running-mean/std normalization, and treated raw `logstd` as `sigma`; its critic path also bypassed native normalization. That runtime SHA came from the preceding CONTROL 263 `DefaultRewardsShaper` guard repair and is preserved as the old baseline.

The new runtime is `bd708a0edbaa74692928d8e8dec68354717465da6e9bdd4c72aae94fdbcf8654` (also recorded in JSON). `_native_actor_forward` now calls the installed `self.model(is_train=False, obs=_preproc_obs(...), rnn_states=...)` exactly once, leaving native `norm_obs`, `exp(logstd)`, Gaussian sampling, and state update in control. `_native_critic_value` uses the same native normalization and a critic-only path with no random draw. Outer `{obs: tensor}` and tensor observations work; actions follow native clamp/rescale semantics. Config validation accepts Mapping and installed `DefaultRewardsShaper.scale_value`, requires `normalize_value=false` and scale 1, and rejects unknown types. The shared CPU mock now constructs the installed `ModelA2CContinuousLogStd.Network`, with raw negative logstd, instead of a fake std-returning namespace.

## Evidence

Focused command: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_NUM_THREADS=2 python3 -m pytest -q tests/test_current_policy_value_collection_contract.py` — **9 passed**, 3 environment warnings. It uses nonidentity running statistics and negative logstd, restores RNG for native equivalence, checks one actor draw, unchanged RNG for critic-only, state propagation, clip/rescale range, dict/tensor handling, installed `DefaultRewardsShaper`, and the Episodes-compatible shared mock loop. Measured wall `7.94 s`, user `7.04 s`, sys `1.21 s`, max RSS `529060 KiB`; no GPU or Isaac process was started.

The current final research card is `c26337b3f87459ea04151a38f343b40aad1116a9fd81b59e05fc079d9ab1841a` (7672 bytes, mode 0664). An archived R3 handoff contains a stale pre-R3b card hash; this delivery records and uses the current hash. Exact per-file SHA-256, mode, and byte sizes are in the JSON handoff.

## Next step

Root should audit the CPU change and separately dispatch any bounded recovery collection. This task does not start the simulator or reopen HF08.
