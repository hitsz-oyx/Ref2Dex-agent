# Current-policy value diagnostic contract — engineering smoke

## Scope

This deliverable implements the factual diagnostic contract for a later
current-policy trajectory audit. It does **not** run a scientific audit on the
HF08 pool, collect transitions, train a model, run PPO, or make a policy
utility/counterfactual/convergence claim. The generated JSON is explicitly a
synthetic engineering smoke.

## Implemented contract

- Required collector fields: complete `physical_value.v1` Episodes plus
  `value_at_state` (same-action frozen PPO critic output), `global_tick`,
  `done`, `terminate`, `timeout`, int64 `episode_id`/`step`/`checkpoint_seed`.
- Complete MC target: reverse episode return with `gamma=.99`; terminal and
  timeout rows have zero bootstrap.
- Frozen-critic GAE(lambda): `tau=.95`, `horizon=32`; one-step next value comes
  from the next step of the same episode/checkpoint. The lambda carry is reset
  when `global_tick % 32 == 31`, at terminals, and at episode/checkpoint
  boundaries. Nonterminal rows require `global_tick[next] == global_tick + 1`.
  Optional `value_at_next_state` is cross-checked; terminal next values must be
  zero. Missing values stop with an explicit `DiagnosticError`.
- Saved V path: the e420 `physical_value.value` weights are loaded from the
  online checkpoint while `Features` statistics/context dimension come from
  the frozen maximum-tier pretraining checkpoint. V outputs remain raw
  reward-return units; `return_scale` is never applied at inference. The
  synthetic smoke exercised this path with s286 e420 (epoch 420, frame 327680,
  295,681 finite V parameters, optimizer step 7680).
- `Episodes` loading rejects nonzero `excluded_rows` and cross-checks native
  per-episode 45-step primary/drop labels when collection results provide them.
  Primary labels use terminal-inclusive `next_state` and integer
  `HoldTracker.run_steps >= 45`.
- Reports separate PPO-critic and saved-PV-V MC/lambda errors, episode counts,
  motion/phase strata, and primary-success/drop strata. Slices with fewer than
  five independent episodes are marked `UNDETERMINED_RARE_SLICE`.

## Smoke and tests

Synthetic fixture checks cover hand-computable MC values, timeout masking,
rollout-boundary lambda truncation, episode separation, explicit missing-value
failure, collector field retention through `Episodes`, and actual saved-V
checkpoint inference. The fixture has no scientific interpretation.

```text
python3 -m unittest tests/test_current_policy_value_diagnostics.py -v
Ran 5 tests in 0.448s — OK
python3 scripts/audit_current_policy_value.py --synthetic \
  --output docs/handoffs/CURRENT_POLICY_VALUE_DIAGNOSTIC_CONTRACT_20261001.json
COMPLETED, CPU-only smoke elapsed 0.442 s
```

## Limits

No current-policy collector output was available or consumed here. A later
factual run must supply a checkpoint-seed mapping such as
`--online-checkpoint 286=/path/GRAB_00000420.pth`, verify the native manifest
hashes, and stop on missing/nonfinite fields or excluded rows. The diagnostic
target is produced from a frozen critic and is not a reconstruction of PPO
training labels.

## Files and reproduction

- [diagnostic script](../../scripts/audit_current_policy_value.py)
- [CPU tests](../../tests/test_current_policy_value_diagnostics.py)
- [synthetic JSON](CURRENT_POLICY_VALUE_DIAGNOSTIC_CONTRACT_20261001.json)

All file SHA256, mode, size, CPU limits, and the synthetic checkpoint hashes
are recorded in the JSON and canonical Broker handoff.
