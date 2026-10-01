# HF08 value target audit — 2026-10-01

## Decision

**Probe result: `UNCLEAR` — training sufficiency and policy-distribution suitability remain unknown.**
The existing data do not isolate an implementation error or establish a reward-target mismatch.  Do not repair or retrain V, or run further Cm policy training, from this audit.  The next discriminating step is a bounded collection of complete **current-policy** trajectories with terminal provenance, followed by V evaluation against those returns.  The existing source-policy Monte Carlo return is not a current-policy truth target.  This report makes no policy-utility or counterfactual-ranking claim.

## Contract and provenance

- Native schema: `ref2dex.physical_value.v1`; `gamma=0.99`.
- Offline target: complete-episode Monte Carlo return, computed backward from stored `reward`; episode split is `episode_id % 10 < 2` for holdout (384 episodes / 209,788 rows), with no episode crossing the split.  Fit is 1,536 episodes / 831,811 rows from 1,920 complete episodes / 1,041,599 rows.
- Online target code fact: `common_agent.py::discount_values` uses `delta = r + gamma * next_value - value` and recursive `lastgaelam = delta + gamma*tau*(1-done)*lastgaelam`; r7 config is `gamma=.99`, `tau=.95`, `horizon_length=32`. `next_value` is zeroed only when `info['terminate']`; the physical-value agent sets `info['terminate']=done.clone()`. The resulting PPO GAE(lambda) return is the target passed to the online V/Q losses. This is separate from the offline complete return.
- Stored reward columns are `(base_source, approach, held, progress, stable)`. `base_source` is the source `original_reward` (there is no independent tracking-only stored column); source code applies approach `2*potential_approach_reward`, held `10*held_lift_reward`, progress `5*contact_lift_progress_reward`, and tracker stable/drop reward. All rows passed schema, finite-value, reward-sum, terminal-reason, quaternion, and native-shard checks.

All 32 transition shards were rehashed against `full_collection_audit.json`; collection result hashes and the selected `tier_1000000.pt` hash were also checked. The machine-readable report contains the complete shard hash map.

| artifact | SHA256 |
|---|---|
| `full_collection_audit.json` | `5297841d9af0ff3f3b22ea9f2b9d6725623912e4f33c9037bc982557e92dfb88` |
| `models/run_manifest.json` | `3239ba83a2264261f5712e51fada2403df1535f94c7649c438da0f5aba88889` |
| `models/results.json` | `30349766ce442bc0c44bf02b3da3bae36b0d068f7db55e3b9f4de4f7b9e4db6f` |
| selected `tier_1000000.pt` | `80879e69e160b7fca512d537a6f39429722eeccbd5deca5488b314390717df72` |

## Discounted reward contribution and rare events

Episode-level discounted means (`gamma=.99`, so adjacent rows are never treated as independent events) over all 1,920 episodes are:

| component | mean discounted contribution | signed share of total |
|---|---:|---:|
| base/source | 5.2299 | 17.79% |
| approach | -0.3377 | -1.15% |
| held | 21.5046 | 73.15% |
| progress | 2.8374 | 9.65% |
| stable | 0.1626 | 0.55% |

The held term dominates the observed source return.  Motion-level means are included in the JSON (`motion_0`: base 5.4804 / held 25.9260; `motion_1`: 5.9897 / 32.7112; `motion_2`: 4.4173 / 8.8478), as are contact/held/pre-contact/drop phase aggregates. The phase table reports episodes touched and rows; it does not promote rows to independent successes.

Stored tracker events give 25/1,920 episodes reaching the 1-second stable threshold and 22/1,920 episodes with a drop-after-success proxy (17/15 on motion 0, 7/6 on motion 1, 1/1 on motion 2 for stable/drop). These are rare source-collection events, not policy success estimates.

## Frozen largest-tier V on development holdout

The CPU audit loaded the frozen maximum-tier checkpoint (`fit_rows=831,811`) and evaluated the same deterministic 2,048-row holdout sample used by the existing diagnostics (381 episodes). Fit-only baselines were a global fit return mean and a motion-phase fit mean with explicit fallback.

| slice | V RMSE | motion-phase baseline RMSE | constant-fit RMSE | rows / episodes |
|---|---:|---:|---:|---:|
| all sampled holdout | 24.4276 | 34.2319 | 45.4802 | 2,048 / 381 |
| failure proxy | 20.6982 | 24.0391 | 24.4131 | 2,041 / 379 |
| success proxy | 222.8588 | 417.5516 | 656.8055 | 7 / 2 |
| pre-contact phase | 16.7526 | 19.4511 | 20.7941 | 1,862 / 378 |
| contact phase | 50.3511 | 89.7432 | 89.5677 | 180 / 121 |
| held phase | 246.5438 | 247.8303 | 704.7773 | 4 / 1 |

The success and held slices are too sparse for calibration claims. The aggregate improvement over constants is a development-distribution diagnostic and does not show current-policy calibration or ranking utility. Native model diagnostics also report maximum-tier value-weighted MSE 151.162 and action-blind 27.2934; these are factual source-development metrics only.

## Saved online e420 V inspection

The existing s286/s287 e420 artifacts were read and hashed, not replayed or recalibrated. Both are epoch 420 / frame 327,680 `cm_value` checkpoints trained/evaluated on fixed noisy source-policy development states. Their saved physical-V state has 295,681 finite parameters (10 tensors), the PPO critic has 4,102,145 finite parameters (10 tensors), and the running observation statistics count is 5,160,961 for each seed. No teacher labels or return targets are saved in either checkpoint. Their reward RMSE diagnostics are 0.9746 (s286) and 0.9833 (s287); checkpoint hashes and initial-model hash are in the JSON. The artifact scope explicitly permits distribution mismatch and excludes current-policy value calibration. Source MC labels therefore cannot answer the current-policy V question.

## Resource and reproduction evidence

- CPU only, `torch.set_num_threads(2)`, no CUDA/GPU/Isaac Gym, no PPO execution, no online/Cm training, no collector, no new transitions.
- Audit elapsed 41.87 s; JSON output is 33,722 bytes. Worktree HEAD: `a0ecee73db7bc7bf9ddb6b9016d028ba16943237`.
- Reproduce from this branch: `python3 scripts/audit_hf08_value_targets.py --output docs/handoffs/HF08_VALUE_TARGET_AUDIT_20261001.json`.
- Script SHA256: `af7bc56c93f167545a1cfc9ee380b35ed5533e18cded72d020e32400bd03ee3e`.

The JSON report is the canonical metric/hash record: [HF08_VALUE_TARGET_AUDIT_20261001.json](HF08_VALUE_TARGET_AUDIT_20261001.json).
