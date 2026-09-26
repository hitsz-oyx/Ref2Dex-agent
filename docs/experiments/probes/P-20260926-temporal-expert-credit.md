---
schema: ref2dex.probe.v2
probe_id: P-20260926-temporal-expert-credit
date: 2026-09-26
branch: agent/cm-temporal
git_commit: 33e828859185edb2b4590cbb9eea292d5283be71
baseline_handoff_commit: 546f6f8
canonical_route_manifest: src/task/CmResidual/configs/hf02_temporal_canonical_route.json
canonical_route_sha256: afedfa54c8573096c4d2104d3328efba32b5daf11445323792f45eca19c04d16
tracked_temporal_evaluator: third_party/DExplore/dexplore/evaluate_temporal_expert_option.py
tracked_temporal_evaluator_commit: 33e828859185edb2b4590cbb9eea292d5283be71
tracked_temporal_evaluator_git_blob_sha1: 388184b00f9cde55ec2ff508f0581af1b531e91a
tracked_temporal_evaluator_sha256: 0de5f9743af69e8b4373fd411c1f0ecd2cb3de0ca2089405ac6f68470111ca76
tracked_collector_config: src/task/CmResidual/configs/airplane_temporal_expert_probe.json
tracked_collector_config_sha256: 47162342929f2b2889197d2d80b6a9069f6ee513d6a83d33cf99c25cba681536
tracked_contract_module: src/task/CmResidual/temporal_option_contract.py
tracked_contract_module_sha256: 4cbe646bebe395c3d3065919da874eb0346c51852670eac1001355d8213be697
tracked_cpu_tests: src/task/CmResidual/tests/test_temporal_option_contract.py
tracked_cpu_tests_sha256: 42886719b2dd897e00dd9fd2e8874a38a74e0077d5a7113570ffcccb2c1eb728
claim_id: C3
hypothesis_family: HF02
decision_changed_if_positive: Freeze the history-conditioned option-value representation and propose one independent matched Cm-on/off online confirmation on the same route.
decision_changed_if_negative: Mark HF02 UNCLEAR or UNPROMISING, freeze this option family, and switch to a higher-level Cm representation or credit-allocation hypothesis.
probe_index_in_family: 2
seed_pool: probe
status: PLANNED
---

# Probe: temporal expert-option credit

## Decision question

On one frozen self-trained specialist substrate, does a ten-step interaction
history contain information that predicts which frozen expert option will
produce future contact-supported lift and held-lift? This is an offline
decision Probe on the airplane-only diagnostic route. It does not test
cross-object transfer and it does not launch PPO or an online Probe before the
offline gate is evaluated.

## Baseline handoff and canonical choice

The baseline handoff is
`docs/handoffs/HF02_TEMPORAL_SUBSTRATE_HANDOFF_20260926.md`, introduced by
`8af61c6` and digest-pinned by `546f6f8`. It freezes one small diagnostic
substrate; the machine-readable manifest is
`src/task/CmResidual/configs/hf02_temporal_canonical_route.json` with SHA256
`afedfa54c8573096c4d2104d3328efba32b5daf11445323792f45eca19c04d16`.
That manifest is the only canonical route for this HF02 slot:

* route mode is `simulator_object_id`, with `airplane -> source_e260` as the
  default/base route expert;
* the object set is exactly `airplane`;
* the motion root is `outputs/CmResidual/agent_contact_option_airplane_motions`
  with exactly `s3_airplane_lift`, `s7_airplane_lift_Retake`, and
  `s9_airplane_lift`;
* candidate experts are fixed, in manifest order, to the six self-trained
  checkpoints below:

  | expert | checkpoint | SHA256 |
  | --- | --- | --- |
  | `balanced_e360` | `outputs/Dexplore/agent_crossobject_train5_balanced_s179_e360/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000360.pth` | `a41fd8281dcf4639579a9cd71baa104f007969cf0642c506d3d358507025f03f` |
  | `cup_e340` | `outputs/Dexplore/agent_cup_specialist_s70_e340/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000340.pth` | `c7367b92248a01795abafe1761e2e96c86615f70fd406dbc9759c1dd3fc368fc` |
  | `duck_e340` | `outputs/Dexplore/agent_duck_specialist_s70_e340/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000340.pth` | `9bcac13e814cc0de03deb9dcf9fdc8ee8bd9af4e6bb1c71e795c37f8a97e4a7a` |
  | `mixed12_e300` | `outputs/Dexplore/agent_multitrajectory12_s70_e300/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000300.pth` | `93626a58cb8448eb8c56b86d0cbdef7a1a66a11c5807a1490319d97eb3974ad3` |
  | `source_e260` | `outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth` | `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f` |
  | `train5_e320` | `outputs/Dexplore/agent_crossobject_train5_s179_e320/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000320.pth` | `6907c12f8ee4ffa9af22ccae8ffe7599d524e401ff2d8f21b8804185fd5d4961` |

There is no learned observation router in this substrate; `simulator_object_id`
is privileged routing and every result must be labelled that way. The
59-motion route, any ten-expert route, the old three-expert airplane
configuration, and the old `3/64` Cm-off result are explicitly excluded and
must not be mixed into this card. The earlier uncommitted five-step
randomized-wrist checkpoint is also excluded.

The handoff's fixed-route Cm-off entry is
`third_party/DExplore/dexplore/evaluate_object_router.py` at blob
`84e086d502a1b838e4607f3d51f2f0d77073f114`. The exact six-expert temporal
option evaluator is now tracked at
`third_party/DExplore/dexplore/evaluate_temporal_expert_option.py` in commit
`33e828859185edb2b4590cbb9eea292d5283be71`, with Git blob SHA1
`388184b00f9cde55ec2ff508f0581af1b531e91a` and file SHA256
`0de5f9743af69e8b4373fd411c1f0ecd2cb3de0ca2089405ac6f68470111ca76`.
It is contract-first and imports Isaac Gym only after the CPU preflight passes.
The companion collector config is tracked at
`src/task/CmResidual/configs/airplane_temporal_expert_probe.json` with SHA256
`47162342929f2b2889197d2d80b6a9069f6ee513d6a83d33cf99c25cba681536`.
The shared CPU contract module is tracked at
`src/task/CmResidual/temporal_option_contract.py` with SHA256
`4cbe646bebe395c3d3065919da874eb0346c51852670eac1001355d8213be697`, and the
contract tests are tracked at
`src/task/CmResidual/tests/test_temporal_option_contract.py` with SHA256
`42886719b2dd897e00dd9fd2e8874a38a74e0077d5a7113570ffcccb2c1eb728`.
The baseline handoff's uncommitted-evaluator note is therefore resolved by this
implementation commit. This engineering task does not authorize collection or
an online Probe.

## Frozen data collection

For each of two owned simulator seeds, a future collection may use the tracked
evaluator only under a separate approved execution step, using exactly the
route manifest and three-motion root above:

* fit seed `254`, assignment seed `20260926254`;
* holdout seed `255`, assignment seed `20260926255`.
* the simulator route remains `simulator_object_id` with object ID `airplane`
  and default route expert `source_e260`; no object or route substitution is
  permitted;
* use DExplore's seeded `start_times` sampler, do not provide a manual frame
  list, and record each realized `per_episode.start_frame`.

At the first valid hand-object contact after ten valid history steps, assign
one of the six experts with a deterministic balanced assignment. Every arm
has known propensity `p(a|x)=1/6`; the assignment, propensity, candidate
actions, and executed action are saved. The assigned candidate controls the
next ten steps, then the frozen object-route expert resumes. The future window
is the next twenty simulator steps. No Cm is used in collection (`Cm-off`).

Each row must contain (the tracked evaluator enforces these shapes):

* `env_id`, `motion_id`, `object_name`, `simulator_object_id`, `route_expert`,
  `assignment`, and `assignment_propensity`;
* `trigger_step`, `start_frame`, current `state[49]`, `base_action[18]`, and
  `candidate_actions[6,18]`;
* `history_state[10,49]`, `history_action[10,18]`, and
  `history_contact[10]`;
* ten-step `option_candidate_action[10,18]` and
  `option_executed_action[10,18]`, with equality checked so an assigned arm
  cannot silently fall back to the source continuation;
* twenty-step `future_contact_mask[20]` and
  `future_contact_supported_lift_m[20]`, plus the aggregate
  `followup_contact_fraction` and `followup_max_contact_lift_m`;
* complete first-episode `final_lift_success`, `final_max_contact_lift_m`,
  `final_contact_fraction`, and `final_episode_steps`.

Rows are valid only when the trigger and all twenty future steps belong to the
first episode and the recorded route/config/motion hashes match the manifest.
The fit and holdout collections use the same six arms and input schema; only
the simulator/assignment seed changes. Do not substitute the 59-motion,
ten-expert, or old `3/64` route/evidence for either split.

## Prespecified offline comparison

Fit each arm on seed 254 and evaluate once on seed 255, with one fixed feature
normalization learned from the fit rows. Estimate policy value by the
inverse-propensity estimator on the observed assigned arm. Compare exactly:

1. `state_only`: current state and candidate action;
2. `history_only`: ten-step history with no candidate identity or action;
3. `history_plus_expert_id`: history plus a six-way expert one-hot, with no
   candidate action;
4. `temporal_cm`: ten-step history plus the candidate expert action;
5. `action_shuffled`: the same temporal input shape after a fixed, recorded
   candidate-action permutation placebo.

The primary outcome is held-lift (`final_lift_success`); the secondary physical
outcome is the mean contact-supported lift in the twenty-step window. The
offline continuation gate is numeric and must hold on the seed-255 estimate:

* `temporal_cm` policy value is at least **+5 percentage points** above
  `history_only` and at least **+5 percentage points** above
  `action_shuffled`;
* temporal contact-supported lift is not lower than either comparator (point
  estimate, in millimetres);
* the same direction holds for both the held-lift and supported-lift ranking
  summaries, with no nonfinite estimate;
* the fit split has at least 30 valid observed rows per arm and the holdout
  split has at least 20 per arm.

If any necessary condition fails, stop this slot. Do not change seed, horizon,
metric, threshold, or representation and do not start PPO. If all conditions
pass, freeze the fitted temporal head and write a separate confirmation memo
before one same-route matched Cm-on/off online Probe is considered.

## Resource and stop contract

One idle GPU, one collection per seed, at most 30 minutes per collection,
200 MB total transition/report output, and CPU fitting under 20 minutes. Stop
immediately on checkpoint/route/motion hash drift, GPU occupancy above 1 GiB,
missing first-episode history or future labels, assignment/propensity/action
misalignment, fewer than the row minimums, nonfinite tensors, or incomplete
manifests. This card cannot be upgraded to a formal Cm causal claim.

## Status and artifacts

Status: `PLANNED` (tracked evaluator/config ready; no collection run executed).
The one permitted cwd-corrected engineering smoke is
recorded separately and does not consume this Probe's offline gate:

### Invalid run record

The previously launched run
`agent_temporal_cm_online_probe_20260926_on_s254_e280` is retained only as an
invalid engineering record:

* manifest: `outputs/Dexplore/agent_temporal_cm_online_probe_20260926_on_s254_e280/run_manifest.json`;
* final status: `STOPPED/INVALID_IMPLEMENTATION`;
* stop reason: `KeyboardInterrupt` after the user-stopped process group crossed
  the global goal, with the last completed epoch `276/280`;
* `train.log` is preserved unchanged (SHA256
  `b3f88dc19791e668fa5e9a5d2c1e040b43def29a04f5c02bbe145260d791e9b3`);
* it is excluded from all scientific conclusions and does not consume the
  HF02 slot. It used an older training/motion setup and is not the canonical
  offline Probe route.

* run: `agent_temporal_cm_smoke_20260926_r3`;
* commit: `2d5d0b5`, cwd `third_party/DExplore`, one GPU, 8 environments,
  one epoch from source checkpoint 260;
* result: `COMPLETED`, with the temporal module loaded, one reward log emitted,
  and a checkpoint saved;
* inputs were the older five-step history checkpoint and three airplane
  motions, so this smoke is not evidence for the canonical six-expert route
  above and does not change HF02 status.

No option collection, offline fit, matched online Probe, or PPO continuation
is part of this card yet. In particular, the baseline handoff's fixed-route
Cm-off command is an entry point only; it has not been run here. The tracked
implementation's pure-CPU checks are:

* `9 passed` in
  `src/task/CmResidual/tests/test_temporal_option_contract.py`;
* `py_compile` passes for evaluator, contract module, and tests;
* evaluator `--dry-run` verifies all six checkpoint hashes and three motion
  hashes, reports `isaacgym_imported: false`, and gives `32` assignments per
  arm for both seed-owned splits.

Planned collection artifacts after a separately approved execution step:

`outputs/CmResidual/agent_temporal_expert_credit_20260926/`

The card will be updated with fit/holdout reports, exact run manifests,
checkpoint/model hashes, and the resulting HF02 decision after the single
offline Probe.
