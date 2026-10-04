# HF02 canonical fixed-route Cm-off preflight

This is the baseline-owner handoff for the temporal owner. It freezes the matched-off entry for the canonical six-expert, three-airplane substrate without running Isaac Gym or consuming HF02 slot 2.

The audited temporal state is commit `bf3421d808f261d777d0720c5c5bf9e1a3d82287`, whose Probe card remains `PLANNED`. Its route is `src/task/CmResidual/configs/hf02_temporal_canonical_route.json`, SHA256 `afedfa54c8573096c4d2104d3328efba32b5daf11445323792f45eca19c04d16`. The route is `simulator_object_id`, contains only `airplane`, uses `source_e260` as the fixed object route, and declares exactly these six experts:

```text
balanced_e360, cup_e340, duck_e340, mixed12_e300, source_e260, train5_e320
```

The three and only three motion inputs are `s3_airplane_lift`, `s7_airplane_lift_Retake`, and `s9_airplane_lift` under `outputs/CmResidual/agent_contact_option_airplane_motions`. The complete checkpoint and motion hash map is in the companion JSON manifest.

## Fixed-route entry

The reusable fixed-route evaluator is `third_party/DExplore/dexplore/evaluate_object_router.py`, Git blob SHA1 `84e086d502a1b838e4607f3d51f2f0d77073f114` and source SHA256 `4f4feddb3bbad7d5b0dbd67848f4a9a88fa1923bdfcdeb316d5916640c6b6320`. The exact fit-seed command is:

```bash
cd /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent
CUDA_VISIBLE_DEVICES=<one-idle-approved-gpu> \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
third_party/DExplore/dexplore/evaluate_object_router.py \
  --route-config src/task/CmResidual/configs/hf02_temporal_canonical_route.json \
  --task Dexplore_Inspire \
  --cfg_env dexplore/data/cfg/inspire_object_balanced.yaml \
  --cfg_train dexplore/data/cfg/train/rlg/inspire.yaml \
  --motion_file outputs/CmResidual/agent_contact_option_airplane_motions \
  --checkpoint outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth \
  --disable-early-termination --headless \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --num_envs 64 --cm-mode off --cm-candidate-mode experts \
  --cm-contact-gate instant --seed 254 \
  --output outputs/CmResidual/agent_hf02_temporal_canonical_off_s254/results.json
```

This is a command for a later approved GPU run, not a command run during this handoff. Use seed `255` and a new output directory for the holdout counterpart. The evaluator requires a new output directory and records a `run_manifest.json` beside the result.

## What the blob does and does not guarantee

| Requirement | Audit result | Boundary |
| --- | --- | --- |
| `simulator_object_id` route | Expressible | `task.object_id` is mapped through `task.object_name` and `object_route`; the result does not emit an explicit `simulator_object_id` column. |
| Six experts | Expressible with this route | It loads every entry in `CONFIG["experts"]`, but does not assert that the count is six. The canonical route and temporal contract provide that assertion. |
| Three motions | External pin required | `--motion_file` is passed through; the evaluator does not validate the three names or their hashes. The manifest pins both. |
| Seed and `start_frame` | Expressible, not self-validating | `--seed` is forwarded to DExplore and `evaluate.py` records `task.start_times` as `per_episode.start_frame`. There is no temporal assignment seed in this evaluator. |
| Cm-off | Expressible | `--cm-mode off` leaves the Cm model unset. |
| Output | Fixed-route metrics | `summary` and `per_episode` include lift/contact/survival metrics and `start_frame`; they do not include temporal assignment, propensity, ten-step history, option actions, or twenty-step future arrays. |

The fixed result therefore is reusable as a matched baseline preflight entry, but it is not the temporal offline collector. The exact temporal collector at `bf3421d` is the only source for the six-arm option records. No old 59-motion, ten-expert, or 3/64 result is included in this package.

The frozen route JSON contains an older `matched_cm_off.note` saying that the temporal evaluator was not committed. That note predates `bf3421d`; the route bytes and SHA are deliberately unchanged. This supplement supersedes that status note and points to the tracked temporal evaluator hashes above.

The fixed evaluator imports Isaac Gym at module import, so it was not imported or executed here. The runtime YAMLs and output artifacts are read-only workspace inputs and remain untracked; their hashes are recorded in the JSON manifest. This is the only execution blocker for this CPU handoff.

## CPU audit evidence

All checks below were read-only and CPU-only on the temporal worktree at `bf3421d`:

```text
pytest -q src/task/CmResidual/tests/test_temporal_option_contract.py  -> 9 passed
py_compile evaluator + contract + test                              -> passed
evaluate_temporal_expert_option.py --dry-run --skip-artifact-hashes -> passed
  isaacgym_imported=false; fit/holdout arm counts=32 per arm
validate_frozen_contract(..., verify_artifacts=True)                 -> passed
six checkpoint hashes + three motion hashes                          -> passed
fixed evaluator static AST audit                                     -> passed
```

The fixed-route preflight is **reusable for the canonical Cm-off baseline entry**. It remains intentionally unexecuted until the temporal owner schedules the approved simulator run; no new route, GPU job, Probe card, or temporal status was created or changed.
