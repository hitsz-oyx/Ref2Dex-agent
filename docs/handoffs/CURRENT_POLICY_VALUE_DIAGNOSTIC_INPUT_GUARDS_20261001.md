# Current-policy value diagnostic input guards (2026-10-01)

## Scope

CPU-only engineering repair for `T-20261001-current-policy-value-diagnostic-input-guards`. No new data, scientific analysis, training, collection, Isaac Gym, GPU, PPO, utility, counterfactual, convergence, or calibration claim. The prior contract report remains unchanged.

## Guards

- `next_value` is now accepted alongside legacy `value_at_next_state`. Every present field is finite and length checked, terminal rows must be zero, and nonterminal rows must equal the next critic value from the same checkpoint and int64 episode. If both fields exist, both checks force agreement; mismatches raise explicit `DiagnosticError`.
- Saved `physical_value.value` inference now requires factual `previous_action` with exact shape `[N, 18]` and finite values. Missing or malformed history raises `DiagnosticError`; zeros are never fabricated.
- MC, frozen-critic GAE, gamma `.99`, tau `.95`, horizon `32`, history boundaries, terminal masking, primary-label logic, and raw V units are unchanged.

## Verification

`python3 -m unittest tests/test_current_policy_value_diagnostics.py -v`

**PASS: 11 tests, 0 failures** (suite-reported 0.859 s; command wall 2.65 s). Tests cover new/legacy bootstrap compatibility, successor mismatch, terminal nonzero, missing and malformed `previous_action`, plus the original MC/GAE, timeout, episode split, Episodes loader, and saved-V smoke checks.

## Exact artifacts

See the JSON report for exact SHA256, mode, and size.

- Source baseline accepted by root: `ad82d5ac73b4d769fb19c6dc92c685c9939347ed93d247d0be3b1f982c256b4e`
- Branch: `agent/cm`; HEAD: `a0ecee73db7bc7bf9ddb6b9016d028ba16943237`
- CPU only, two threads; no GPU or external writes.

## Limits

This is an input-contract repair only. Real current-policy collection and factual analysis remain a separate root-dispatched step.
