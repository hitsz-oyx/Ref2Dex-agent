# HF02 temporal-cm substrate handoff

This handoff is owned by `agent/grab-full-baseline`. It freezes the substrate
for the single existing HF02 Probe 2 card
`docs/experiments/probes/P-20260926-temporal-expert-credit.md`.
It does not create or consume another HF02 card.

## Canonical route

The canonical substrate is the **six-expert, three-airplane-motion** small
substrate stated by the temporal Probe card:

- route mode: simulator object-ID diagnostic (`airplane -> source_e260`);
- objects: `airplane` only;
- motions: `s3_airplane_lift`, `s7_airplane_lift_Retake`, and
  `s9_airplane_lift` from `outputs/CmResidual/agent_contact_option_airplane_motions`;
- candidate experts, in fixed order:
  `balanced_e360`, `cup_e340`, `duck_e340`, `mixed12_e300`, `source_e260`,
  `train5_e320`;
- base expert and default route expert: `source_e260`;
- no official actor checkpoint and no observation router.

The machine-readable route is
`src/task/CmResidual/configs/hf02_temporal_canonical_route.json`. Its six
checkpoint paths and SHA256 values are copied from
`outputs/CmResidual/grab59_six_expert_route.json`, whose SHA256 is
`637d0e7cfef1ea2b6578bbff468afefc461e98e82f20bdfdf56bc9f892319e9a`.
The three motion tensor SHA256 values are pinned in the canonical route file.

This is intentionally not the 59-motion route, the ten-expert group route, or
the 660-motion pool. Those routes must not be substituted while retaining the
HF02 Probe 2 name.

## Code and router provenance

The frozen baseline substrate is anchored at commit
`8625cd3e4bf958c1af29e0ccf9c69fbd4a402ebb` (`agent/grab-full-baseline`). The
historical six-expert route was evaluated with route/evaluator code at commit
`3e9e7a53acfea134f9715bd6d7165ee47f527cdf`.

There is no learned observation router in this substrate. The router is the
simulator object-ID lookup, so any result must be labelled privileged routing
and must not be reported as a single observation-driven GRAB actor.

## Start-frame contract

The canonical evaluator uses DExplore's seeded `start_times` sampler. There is
no manually supplied start-frame list. The evaluation seed controls the draw,
and every first episode must record its realized `start_frame` in
`per_episode`. The temporal card's intended offline seeds are 254 (fit) and
255 (held-out); these seeds and the realized per-episode frames must remain in
the run manifests. Existing seed234/235 airplane runs are only provenance
references and are not canonical six-expert results.

The three canonical motions are confirmed by the existing corrected motion
directory. The prior two-expert contact-option runs sampled all three motions
with 22/21/21 episodes at seed234 and seed235; this confirms the motion order
and first-episode evaluator behavior, but those runs are not the HF02 matched
control because they exposed only `source_e260` and `balanced_e360`.

## Matched Cm-off entry

For the frozen route itself, the reproducible Cm-off evaluator is the tracked
`third_party/DExplore/dexplore/evaluate_object_router.py` at blob
`84e086d502a1b838e4607f3d51f2f0d77073f114`, with:

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

The command is an entry point, not an instruction to run it under this
handoff. No GPU or Cm training/evaluation was started by the baseline owner.
The output directory must be new; the evaluator records the route/config
hashes, seed, GPU, start frames, and strict first-episode held-lift metric.

## Existing matched-off evidence and boundary

There is no exact completed matched-off result for the canonical combination
of six experts plus these three airplane motions:

| Artifact | What it fixes | Why it is not the canonical result |
| --- | --- | --- |
| `outputs/CmResidual/agent_grab59_six_expert_route_s244_r2/` | all six experts, Cm-off, 64 first episodes; 3/64 held-lift | uses the 59-motion route, not the three-motion temporal substrate |
| `outputs/CmResidual/agent_contact_expert_option_s234_r2/` and `s235/` | the three airplane motions and first-episode start-frame recording | exposes only source/balanced two-expert option, not six experts |

The temporal branch has since committed `8f4cf3845f3f48749aa6171a9265a6efd6b5d289`
(`run_temporal_online_probe.py`). That launcher is provenance-verifiable, but
it starts a source-e260 temporal-reward continuation; it does not load the six
candidate experts, perform first-contact option assignment, or save the
10-step/20-step option records required by the Probe card. The separate
`evaluate_temporal_expert_option.py` collector remains an uncommitted worktree
file. Therefore the exact six-expert temporal-option Cm-off collector is still
not provenance-verifiable. The temporal owner must commit that evaluator (or
provide an equivalent tracked evaluator) before collecting Probe 2; until
then, the baseline owner freezes this handoff and performs no further work.

## Handoff state

- substrate: **FROZEN**;
- canonical route manifest: committed with this handoff;
- fixed-route Cm-off entry: **AVAILABLE**;
- exact temporal-option matched-off evidence: **BLOCKED** by uncommitted
  evaluator;
- new GPU/Cm training: **NONE**;
- HF02 Probe 2 consumption: **NONE**.
