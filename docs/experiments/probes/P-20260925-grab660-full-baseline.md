# P-20260925-grab660-full-baseline

- Classification: Decision Probe.
- Cm: off.
- Coverage: all 660 currently compatible single-right-hand GRAB
  interactions, 50 objects. This is not the raw 1335-sequence dataset.

## Question and decision

Can the current DExplore PPO route become a self-trained baseline when the
whole filtered single-right-hand pool is retained, rather than truncating at
64 motions? The engineering change fixes one object mesh per environment and
cycles every trajectory for that object at reset. A positive result would
justify extending the uniform full-pool actor and then adding Cm to the same
actor. A zero-contact result would move the baseline substrate to the existing
specialist/hierarchical route.

The least expensive test was a 20-epoch Cm-off continuation from the pinned
self-trained e260 checkpoint, followed by a matched 64-environment first
episode. Because that actor was specialized to a small pool, a second Probe
trained a random-initialized policy on the same 660 inputs to e300.

## Inputs and implementation

- Frozen input inventory: `outputs/CmResidual/agent_grab660_input_20260925/`.
- 660 finite `[T,598]` tensors, 50 objects, 166,337 total frames; 59 are
  lift-like by filename and reference object rise.
- Config: `src/task/CmResidual/configs/grab_full_object_cycle.yaml`.
- Training uses one GPU, 64 environments, contact-frame resets, no lift-frame
  reset branch for non-lift interactions, and `--cm-distill-coef 0`.
- The loader now fails loudly instead of silently truncating motions; the
  sampler records per-trajectory reset coverage.

## Results

The continuation completed in `outputs/Dexplore/agent_grab660_s70_e280_probe_r2/`.
All 660 motions were visited; the e280 checkpoint SHA256 is
`7e321f7cde46a6004bb9eafbf5e00042d0b1cf070810e9e6bc6af8fa07b45d5e`.
On new seed221, source e260 and e280 used the same full-pool protocol:

| actor | held-lift | mean hand-object contact | mean max contact lift |
| --- | ---: | ---: | ---: |
| source e260 | 0/64 | 0.787% | 0.000 m |
| full-pool e280 | 0/64 | 1.714% | 0.000 m |

The random-initialized full-pool run completed in
`outputs/Dexplore/agent_grab660_s71_e300_scratch/`, visited all 660 motions,
and produced e300 SHA256
`eaf23dddfaffce0c06d6fedbd0a90beb2f95796e102542a235538e50567d4eab`.
On new seed240 it achieved 0/64 held-lifts, 0.905% mean hand-object contact,
and 0.000 m mean max contact lift.

Two additional random-initialized repetitions used the same frozen input
manifest, config, 300-epoch budget and 64-environment first-episode protocol:

| run | checkpoint SHA256 | eval seed | held-lift | mean hand-object contact |
| --- | --- | ---: | ---: | ---: |
| `s72_e300_scratch_r2` | `5a99095a7320ec26586cd855b1afd1779fe0a2583a7c27ee7a939d7f6ef51fed` | 243 | 0/64 | 0.061% |
| `s73_e300_scratch_r2` | `fb609a2218b857fce56d58e3ffc359511d8e5b27635c7571f25dc377d8b3404d` | 243 | 0/64 | 0.100% |

Both runs visited all 660 motions (minimum 44 and maximum 801 resets per
motion). The evaluation summaries are stored under each run's
`eval_s243_e300_full/results.json`; both had zero mean maximum contact lift.

## Decision

`UNPROMISING` for a uniform full-pool actor under this training budget and
both tested initializations. The result is a baseline route failure, not
evidence that all self-trained full-GRAB policies are impossible. Do not add
more epochs to this exact route. Use the existing observation-driven
specialist/hierarchical substrate for the next Cm Probe, and require a matched
Cm-off arm before any claim of Cm utility.

### Current disposition (2026-09-26)

The recommendation above was the local decision boundary for this baseline
Probe. The later HF02 temporal Probe is `UNPROMISING`, and
`docs/archive/2026-10-04-research-governance/decisions/D-20260926-after-hf02-temporal.md` now freezes the Cm
policy-utility campaign. Therefore this card does not authorize a new Cm Probe;
the specialist/hierarchical substrate remains provenance only until a separate
HF03-style goal is created.

## Limits

- Evaluation is one first episode per 64 parallel environments and is a Probe.
- The 660 pool excludes left-hand-contact sequences and doorknob names; raw
  GRAB still contains 1335 `.npz` sequences.
- `success_rate` in the evaluator means episode completion; the manipulation
  metric is the held-lift definition above.

## CPU engineering re-verification

On 2026-09-26, the committed coverage implementation was rechecked without
starting a simulator or GPU process:

```text
pytest -q dexplore_approach, dexplore_v120_motion_input, object_disjoint_split,
          object_motion_sampler, stage_dexplore_object_mesh, v140_router -> 23 passed
py_compile sampler, filtered-motion loader, baseline launcher, and task      -> passed
git diff --check against the route commit                                 -> passed
```

These checks verify approach shaping, route compatibility, object-mesh identity,
asynchronous reset coverage, native contact preservation, translation
provenance, object-disjoint splitting, and explicit full-pool sampler wiring.
They do not change the Probe's `UNPROMISING` decision or authorize another
full-pool training run.

The frozen inventory was independently replayed on 2026-09-26 from
`outputs/CmResidual/agent_grab660_input_20260925/input_inventory.json`:

```text
inventory SHA256: 0ac5b4cec689f70d35d5896823d9a431ef9fecc5933a09435c006bc426c5bc5b
motions: 660 | objects: 50 | frames: 166337 | lift-like: 59
tensor files missing: 0 | tensor hash drift: 0
source manifest SHA256: 87f9a27b675133f1e1f122e0ed7f6318c06b62a9977f36bd3b890220fc90d240
```

This audit only reads the existing inventory and tensor files; it does not
recreate, relabel, or regenerate the full-pool inputs.

The four run manifests were also cross-checked on the same CPU audit:
`agent_grab660_s70_e280_probe_r2`, `agent_grab660_s71_e300_scratch`,
`agent_grab660_s72_e300_scratch_r2`, and `agent_grab660_s73_e300_scratch_r2`
are all `COMPLETED`, declare `motion_count=660`, and pin the same input
manifest SHA256 `eae14c02a078a430f6cc8c6754ca53267d647793c2a3347a2bf9a06787a40214`
and environment-config SHA256
`e1e41c5e4050f022690f8fef86774259afec1106216325d0ef035430e2867731`.
The `eval_s244_e300_full_lift59r2` files retained under the s72/s73 output
directories are explicitly excluded 59-motion subset records; they are not
part of this 660-motion baseline or its metrics.
