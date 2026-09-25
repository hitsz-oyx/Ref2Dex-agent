---
schema: ref2dex.probe.v2
probe_id: P-20260926-temporal-expert-credit
date: 2026-09-26
branch: agent/cm-temporal
git_commit: 8f4cf3845f3f48749aa6171a9265a6efd6b5d289
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
decision Probe. It does not test cross-object transfer and it does not launch
PPO before the offline gate is evaluated.

## Baseline handoff and canonical choice

The baseline handoff is `docs/HANDOFF_20260925_1208.md`. It identifies the
curated 12-motion/10-object self-trained route and the six-expert observation
route as the current usable substrate. That route is therefore canonical here:

* Route config: `src/task/CmResidual/configs/multitrajectory_object_router_with_cup_probe.json`, SHA256
  `6e3205a8b5fe2be3ab46308678e673d10e62745a7eda8196a7fe0e0d365bb7a8`.
* Motion root: `outputs/Dexplore/agent_multitrajectory12_s70_e300/motions`.
  It is frozen to the twelve directories `s3_airplane_lift`,
  `s7_airplane_lift_Retake`, `s9_airplane_lift`, `s7_apple_lift`,
  `s1_alarmclock_lift`, `s1_cubesmall_lift`, `s1_cup_lift`,
  `s1_duck_lift`, `s1_mug_lift`, `s1_phone_lift`, `s1_toothpaste_lift`,
  and `s1_waterbottle_lift`.
* Objects are exactly airplane, apple, alarmclock, cubesmall, cup, duck, mug,
  phone, toothpaste, and waterbottle. The object list is a fixed substrate
  definition, not a generalization claim.
* The six frozen self-trained experts and checkpoint hashes are:

  | expert | checkpoint | SHA256 |
  | --- | --- | --- |
  | `source_e260` | `outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth` | `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f` |
  | `mixed12_e300` | `outputs/Dexplore/agent_multitrajectory12_s70_e300/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000300.pth` | `93626a58cb8448eb8c56b86d0cbdef7a1a66a11c5807a1490319d97eb3974ad3` |
  | `train5_e320` | `outputs/Dexplore/agent_crossobject_train5_s179_e320/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000320.pth` | `6907c12f8ee4ffa9af22ccae8ffe7599d524e401ff2d8f21b8804185fd5d4961` |
  | `balanced_e360` | `outputs/Dexplore/agent_crossobject_train5_balanced_s179_e360/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000360.pth` | `a41fd8281dcf4639579a9cd71baa104f007969cf0642c506d3d358507025f03f` |
  | `duck_e340` | `outputs/Dexplore/agent_duck_specialist_s70_e340/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000340.pth` | `9bcac13e814cc0de03deb9dcf9fdc8ee8bd9af4e6bb1c71e795c37f8a97e4a7a` |
  | `cup_e340` | `outputs/Dexplore/agent_cup_specialist_s70_e340/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000340.pth` | `c7367b92248a01795abafe1761e2e96c86615f70fd406dbc9759c1dd3fc368fc` |

The earlier uncommitted three-expert airplane configuration and the older
five-step randomized-wrist history checkpoint conflict with this substrate.
They are not used: the handoff's six-expert route is the only canonical
portfolio for this HF02 slot, and choosing it keeps provenance aligned with
the measured self-trained base.

## Frozen data collection

For each of two owned simulator seeds, collect first-episode rows with the
same route and motion root:

* fit seed `254`, assignment seed `20260926254`;
* holdout seed `255`, assignment seed `20260926255`.

At the first valid hand-object contact after ten valid history steps, assign
one of the six experts with a deterministic balanced assignment. Every arm
has known propensity `p(a|x)=1/6`; the assignment, propensity, candidate
actions, and executed action are saved. The assigned candidate controls the
next ten steps, then the frozen object-route expert resumes. The future window
is the next twenty simulator steps. No Cm is used in collection (`Cm-off`).

Each row must contain:

* `env_id`, `motion_id`, `object_name`, `route_expert`, `assignment`, and
  `assignment_propensity`;
* `trigger_step`, `start_frame`, current `state[49]`, `base_action[18]`, and
  `candidate_actions[6,18]`;
* `history_state[10,49]`, `history_action[10,18]`, and
  `history_contact[10]`;
* twenty-step `future_contact_mask[20]` and
  `future_contact_supported_lift_m[20]`, plus the aggregate
  `followup_contact_fraction` and `followup_max_contact_lift_m`;
* complete first-episode `final_lift_success`, `final_max_contact_lift_m`,
  `final_contact_fraction`, and `final_episode_steps`.

Rows are valid only when the trigger and all twenty future steps belong to the
first episode. The fit and holdout collections use the same six arms and input
schema; only the simulator/assignment seed changes.

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

Status: `PLANNED`. Planned artifacts:

`outputs/CmResidual/agent_temporal_expert_credit_20260926/`

The card will be updated with fit/holdout reports, exact run manifests,
checkpoint/model hashes, and the resulting HF02 decision after the single
offline Probe.
