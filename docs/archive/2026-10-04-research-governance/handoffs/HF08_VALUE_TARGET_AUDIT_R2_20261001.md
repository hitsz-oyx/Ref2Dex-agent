# HF08 value-target audit R2 — primary-label and online-V repair

## Result

**Contract PASS.** The HoldTracker reconstruction matches the native terminal records exactly: **15/1,920 primary successes** and **12/1,920 primary drop-after-success episodes**. The former R1 25 count was a one-second `advance_events.ever` event and is retained only under the name `one_second_held_event`.

The conclusion remains `UNCLEAR / TRAINING_SUFFICIENCY_OR_DISTRIBUTION_UNKNOWN`. These checks establish label/provenance correctness; they do not establish current-policy V calibration, policy utility, counterfactual ranking, or convergence.

## Primary labels

For each complete episode, the audit uses pre-action `state[0].z` as `initial_height`, then replays every terminal-inclusive `next_state` row:

```text
held = (next_z - initial_height >= .03) and both_contacts
previous_success = stable
run_steps = run_steps + 1 if held else 0
stable |= run_steps >= 45             # 1.5 s at 30 Hz
falling = (next_z - initial_height < .02) or (next_lost >= 1 - 1e-6)
drop_after_success |= previous_success and falling
```

This is the integer `HoldTracker.step` rule. It includes the final `next_state` before reset and does not use the 30-step `ever` marker as primary success. Native per-episode records from both collection results were matched row-for-row; any mismatch would have stopped the audit.

| split | episodes | primary successes | drop after primary |
|---|---:|---:|---:|
| fit | 1,536 | 14 | 11 |
| holdout | 384 | 1 | 1 |
| total | 1,920 | 15 | 12 |

| motion | episodes | primary successes | drop after primary |
|---|---:|---:|---:|
| 0 | 660 | 10 | 8 |
| 1 | 546 | 4 | 3 |
| 2 | 714 | 1 | 1 |

The fixed 2,048-row development sample contains **6 primary-success rows from 1 episode**; this slice is marked `UNDETERMINED_RARE_SLICE`. Primary failure contains 2,042 rows from 380 episodes. The full holdout has only one primary-success episode, so no success-stratified calibration claim is possible.

## Reward contribution cross-check

The reward decomposition is unchanged and was recomputed from the same 1,041,599 finite rows / 1,920 complete episodes (`gamma=.99`). Episode-level mean discounted contributions are base/source `5.2299`, approach `-0.3377`, held `21.5046`, progress `2.8374`, stable `0.1626`. The held term is the largest observed contribution. Motion and phase tables are in the JSON; phase `held` remains a one-second diagnostic phase and is not the primary label.

## Actual saved online V inspection

Both `cm_value` `GRAB_00000420.pth` files were loaded directly on CPU. The native component-audit artifact checkpoint hashes match the files:

| seed | checkpoint SHA256 | epoch/frame | V params | optimizer state | changed vs max-tier V |
|---|---|---|---:|---|---|
| s286 | `0fe81f67f7b95356253edbdc755ccbc475ea77fb8a472ee99bb565dd6f327cf0` | 420 / 327680 | 295,681 finite | 20 entries, every step 7,680 | 10/10 tensors; L2 44.0759 |
| s287 | `afdb2cdf56d3350e18862d782d777e9aeb8a8133bf6690c440b43b9c8d5698bf` | 420 / 327680 | 295,681 finite | 20 entries, every step 7,680 | 10/10 tensors; L2 44.3009 |

Both checkpoints have the same 10-key V state as the frozen maximum-tier pretraining V (`tier_1000000.pt` SHA `80879e69e160b7fca512d537a6f39429722eeccbd5deca5488b314390717df72`), and all inspected tensors/optimizer states/running statistics are finite. No teacher labels or return targets are serialized in these checkpoints. Parameter changes and optimizer steps prove that online V updates occurred; they do not prove current-policy calibration.

Confirmed target/training facts: r7 sets `normalize_value=false`; online PPO V uses GAE(lambda) returns with `gamma=.99`, `tau=.95` and 7,680 optimizer updates at e420; offline maximum tier is 1,000 updates of batch 128, i.e. 128,000 sampled rows or `0.1539` of the 831,811 fit-row count per nominal pass. These facts do not justify a loss-based sufficiency claim.

## Resource, hashes, and reproduction

- CPU-only, 2 threads; no GPU, Isaac Gym, PPO execution, training, collector, or new transitions.
- Audit computation: 63.11 s; no failed R2 attempts; output 36,796 bytes. Claim-to-handoff workflow wall time was 1,200.89 s (the 600 s task lease/limit is recorded explicitly in JSON); this includes transport/tool time, while the CPU computation stayed within the 10-minute computation ceiling.
- R1 files were byte-checked before and after; their hashes are recorded in `r1_reference_hashes` and were not overwritten.
- Reproduce: `python3 scripts/audit_hf08_value_targets_r2.py --output docs/handoffs/HF08_VALUE_TARGET_AUDIT_R2_20261001.json`.

Final SHA256: R2 script `c5b6c1ac5285a711f73a26ddbbdf50c29ecce5c64f2329fb9039ee8a30f59771`; R2 JSON `e4fe36068eb3bbe2b8747ed32f96ddb6b96a0f1ba1aab676ef45d0d14d9b27a0`; R2 Markdown is recorded in the Broker handoff.

Files:

- [R2 report](HF08_VALUE_TARGET_AUDIT_R2_20261001.json)
- [R2 script](../../scripts/audit_hf08_value_targets_r2.py)
