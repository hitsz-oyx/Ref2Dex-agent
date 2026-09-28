# Cm scratch CPU calibration r2 axis handoff

## Task and execution boundary

- Broker task: `T-20260928-cm-scratch-teacher-arbitration-cpu-calibration-r2-axis`
- Worker: `agent_cm`
- Branch/HEAD: `agent/cm` / `a0ecee73db7bc7bf9ddb6b9016d028ba16943237`
- Device: CPU only; GPU count used: `0`
- No Isaac Gym, collector, PPO, online/Cm training, student distillation, new transition, fake label, or existing-output overwrite was used.
- No `docs/ref.md`, `docs/STATE.md`, queue, or decision file was modified.

## Read-only input evidence

| input | SHA256 | mode | bytes |
|---|---|---:|---:|
| `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-rl/outputs/P-20260928-six-expert-support-collection-r7-axis/fit/run_manifest.json` | `936f43ff85cd4d6691bfdee52d56f3b75b338e78a9706b38673606b48d8d7088` | `0664` | 9377 |
| `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-rl/outputs/P-20260928-six-expert-support-collection-r7-axis/fit/records.pt` | `aa60b82b35d95ea6278e8cf0c386d00b5fcf6ed5fc3ec31bc7a858c0f31dbf93` | `0664` | 2098361 |
| `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-rl/outputs/P-20260928-six-expert-support-collection-r7-axis-holdout/holdout/run_manifest.json` | `43802d0a5fe7cb7cf96cd608e3420313e1adf37ccfc7d0e4ad970ee7b8cefe97` | `0664` | 9431 |
| `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-rl/outputs/P-20260928-six-expert-support-collection-r7-axis-holdout/holdout/records.pt` | `b49e30045ba2370546cc0c6ea8949660fbcdc7984a584fe4d67113f59ddeef80` | `0664` | 2076857 |
| `src/task/CmResidual/scratch_teacher_arbitration_contract.py` | `40b88bd892481db04475c970b5a4f7247da8ca5e093bb4d830ee3098714e6497` | `0664` | 9369 |
| `src/task/CmResidual/configs/cm_scratch_teacher_arbitration_v1.json` | `c1f345a3acf798ca72eda8167795fab693d8eaa3aed525c230c23f4edc832fa3` | `0664` | 4875 |

The canonical r7 contract passed for both payloads (`188` fit rows and `186` holdout rows). The canonical collector SHA is `74abd06318684c574d5f26f758f2a5d075965b1dabfd36f2ef2b0ff112b8717e`; canonical route SHA is `afedfa54c8573096c4d2104d3328efba32b5daf11445323792f45eca19c04d16`.

## Support and provenance audit

- Six-arm counts (order `balanced_e360, cup_e340, duck_e340, mixed12_e300, source_e260, train5_e320`): fit `[31,32,32,30,31,32]`; holdout `[32,30,30,31,31,32]`.
- Stored propensity is float32 `0.1666666716337204` in every row and passes the canonical `1/6` tolerance. Scratch-row validation used exact `1.0/6.0` after this checked representation normalization.
- Fit and holdout episode IDs are unique within each split and disjoint across splits. Episode-ID sequence SHA256: fit `9796be2a7d33437babca93854a637e5ba2696b5808c3ea638ef15853768d43b`; holdout `ca62dae655f026a0c101e6bd0dfda2efc9de3e3b0d9d598287efd992a4def3de`.
- `object_lift_axis` is finite and unit: norm range fit and holdout `[0.9999999404, 1.0000001192]`, maximum absolute unit error `1.1920928955e-7`.
- Axis provenance is `world +Z` inverse-rotated by trigger-time object quaternion XYZW, frame `object_local_at_trigger_t`, with conversion `local_translation_target_inverse_quaternion_xyzw`.
- One-step target delta equals `object_pose_t_plus_1_object_local_frame - object_pose_t_object_local_frame` exactly (`max_abs_error=0.0`) in both splits. Five-step contact target equals the first five future-contact entries exactly.
- Executed action equals the assigned candidate action exactly. All router teacher records use `c1_observation_router`; router model SHA is `1fa84c94891d63824be0100b5eb78f4aa1e37825c6d71fff65e517c0c7f2fc14`.
- Calibration inputs were only `pre_action_observation`, `candidate_action`, and `candidate_id_one_hot`. Future, next, final, held-lift, terminal, and teacher-label fields were excluded. No static `source_e260` fallback was used; Cm-off and no-pass Cm-on fallback use the recorded C1 router label.

## Restricted CPU calibration

The fit split only was used to fit a centered ridge one-step delta model and contact mean model. Contact q10/q50/q90 are additive empirical residual quantiles; delta q10/q90 residual spans provide the deterministic uncertainty radius `0.0012312361843404428`. No checkpoint or model artifact was written.

| metric | fit | holdout |
|---|---:|---:|
| delta RMSE | 0.0004798565 | 0.0032903468 |
| delta MAE | 0.0002905361 | 0.0018525833 |
| contact mean MAE | 0.0188931 | 0.2044826 |
| contact q10 lower-bound coverage | 0.9414894 | 0.7688172 |
| delta q10/q90 coordinate coverage | 0.7978723 | 0.2634409 |
| delta q10/q90 all-coordinate row coverage | 0.6063830 | 0.0268817 |

The holdout q10 lower-bound coverage misses the nominal 0.90 calibration target and the transition interval coverage is poor, so this calibration is not promoted to distillation or online work.

## Cm-on/off labels and stability

The machine-readable output contains the per-row labels, episode order, label hashes, predictions, and metrics: `docs/handoffs/CM_SCRATCH_CPU_CALIBRATION_R2_AXIS_20260928.json`.

| split | Cm-on labels (six-arm order) | Cm-off router | fallback to router | Cm-on changed from Cm-off |
|---|---|---|---:|---:|
| fit | `[17,12,105,1,24,29]` | `[0,0,0,0,188,0]` | 24/188 (0.1276596) | 164/188 (0.8723404) |
| holdout | `[21,6,101,1,32,25]` | `[0,0,0,0,186,0]` | 32/186 (0.1720430) | 154/186 (0.8279570) |

Only `34/188` fit and `26/186` holdout Cm-on labels equal the randomized assigned candidate, so physical outcomes for selected unassigned candidates are unavailable. Predicted certified progress is therefore not a held-lift or policy-utility measurement.

Threshold sensitivity around the declared contact LCB threshold `0.60` was deterministic: holdout agreement with 0.60 was `0.9784946` at threshold 0.55 and `0.9677419` at 0.65; fallback rates were `0.1559140`, `0.1720430`, and `0.2043011` at 0.55, 0.60, and 0.65.

## Decision

**`UNPROMISING` for this CPU calibration artifact.** Support and provenance are valid, but holdout calibration coverage fails the declared q10/interval expectations. This is not a Cm policy-utility claim; no causal, held-lift, or student-distillation conclusion is made. Stop before any follow-up distillation/online path and require a separate decision for any recalibration design.

## Output evidence

- JSON output: `docs/handoffs/CM_SCRATCH_CPU_CALIBRATION_R2_AXIS_20260928.json`
- JSON SHA256: `1664796e708b07fdb1e1d6dda34029a2f5a2a7b62efa1c62327cdf4642c33e05` (mode `0664`, 53869 bytes)
- Label SHA256: fit `22ceb2087156d9f6905d8c740984668fd40de5e7de71e1871a100acd2803f99b`; holdout `51eda918c1f2a041c6a8a35ef2e423c1af29715f5626b5351a736af3a05f0889`.
