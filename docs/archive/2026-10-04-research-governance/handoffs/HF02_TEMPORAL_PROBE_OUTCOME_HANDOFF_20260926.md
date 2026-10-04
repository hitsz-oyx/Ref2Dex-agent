# HF02 temporal Probe outcome handoff

This addendum records the completed canonical temporal Probe for the baseline/provenance owner. It does not create a second HF02 card, change the route manifest, or authorize an online Cm run.

The temporal owner’s latest card state is commit `f8a60499b6cc3a2095cbddb9b0b12200192c11f3` on `agent/cm-temporal`. The card remains `P-20260926-temporal-expert-credit`, status `UNPROMISING`, and its route SHA256 is the frozen `afedfa54c8573096c4d2104d3328efba32b5daf11445323792f45eca19c04d16`. The tracked result index is:

```text
docs/experiments/probes/P-20260926-temporal-expert-credit-results.json
blob at f8a6049: c0bf301240d9ee852f835708947feb0605ab7b85
workspace SHA256: 54112b28307fd7a4500534d1f500b2f7163b185b2b56d7b3bbdab9c0a4176358
```

The exact collection route stayed fixed: `simulator_object_id`, object `airplane`, base expert `source_e260`, six candidate experts, three canonical airplane motions, 10 history steps, 10 option steps, 20 future steps, first episode only, and known propensity `1/6`. The evaluator content used by the successful collection has blob SHA1 `22f0bf8d1470969e14a1030bfcd708a4f1d618ad` and SHA256 `8297c5a09c4bec5f3fa0987d9bb57a552d5a5e43e2dc162a3324ae8ce5b2e0ba`. The successful run manifests identify collection commit `5bc9f8820e4c80553b4b77c0a844d1c612fafc0c`; the later `f8a6049` changes only card/result provenance text.

## Evidence

| split | run | valid rows | arm counts | records SHA256 |
| --- | --- | ---: | --- | --- |
| fit, simulator seed 254 / assignment seed 20260926254 | `fit_s254_r5` | 187 | `31/31/32/31/31/31` | `38f311850f31549f6d3a050cff12df7ebccd00bd9a5ef65ed3a9b12a250f2506` |
| holdout, simulator seed 255 / assignment seed 20260926255 | `holdout_s255_r1` | 186 | `30/32/31/32/31/30` | `cb851a96229df87501d890732cbdf0b90b2c2952e07d018edb6d83d47d641fe8` |

Artifacts are under `outputs/CmResidual/agent_temporal_expert_credit_20260926/`. The contract accepted both payloads, including route/checkpoint/motion hashes, first-episode boundaries, action equality, finite tensors, start frames, and propensity. The CPU report is `cpu_fit_report.json`, SHA256 `2f440f6c55fe625ec24140c981ac40086b8021dd981253af1686281ca940598d`; the fitted model artifact is `cpu_fit_models.pt`, SHA256 `5c3c35ea2eb8ca8209ae4e92da8af11463b8c3af14bd3b93efcd3bce0c6b7048`.

The seed-255 IPW policy values were:

| model | held-lift | supported lift |
| --- | ---: | ---: |
| `history_only` | 25.806% | 2.856 mm |
| `history_plus_expert_id` | 51.613% | 4.346 mm |
| `state_only` | 32.258% | 1.987 mm |
| `temporal_cm` | 38.710% | 3.811 mm |
| `action_shuffled` | 48.387% | 2.488 mm |

`temporal_cm` is `+12.903` percentage points above `history_only`, but `-9.677` points below the action-shuffled placebo. Supported lift is non-regressive by `+0.955` mm versus history-only and `+1.322` mm versus the placebo. The row minimum, finite-value, and supported-lift checks pass; the held-lift margin and ranking-direction gates fail. This is a valid `UNPROMISING` Probe result, not an implementation invalidation.

## Handoff boundary

Freeze HF02 slot 2 at this result. Do not rescan seeds, horizons, metrics, or representations; do not start PPO continuation, a matched online Cm-on/off Probe, a duplicate baseline collection, or a second HF02 card. The fixed-route Cm-off preflight in [HF02_FIXED_ROUTE_CM_OFF_PREFLIGHT_20260926.md](HF02_FIXED_ROUTE_CM_OFF_PREFLIGHT_20260926.md) remains a reproducible baseline entry, while this temporal outcome closes the current offline decision gate.

The engineering retry records remain excluded as documented by the temporal card. CPU checks remain `9 passed`, evaluator dry-run reports `isaacgym_imported=false`, and no new GPU work was started for this outcome handoff.

## Independent provenance replay

After the outcome handoff was written, the current temporal worktree was
replayed with `validate_frozen_contract(..., verify_artifacts=True)` and
`validate_record_payload(...)` against both saved payloads. The CPU-only replay
returned:

```text
frozen_contract: PASS
fit payload: PASS       rows=187  arms=31/31/32/31/31/31
holdout payload: PASS   rows=186  arms=30/32/31/32/31/30
independent invariants: PASS for both splits
```

The independent checks also confirmed that every row is `airplane`, every
`route_expert` is `source_e260`, all assignment propensities are finite and
equal to `1/6`, and the recorded checkpoint/motion hashes still match the
canonical route. No Isaac Gym import or GPU process was used for this replay.
