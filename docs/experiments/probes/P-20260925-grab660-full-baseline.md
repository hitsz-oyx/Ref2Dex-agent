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
pytest -q src/task/CmResidual/tests/test_object_motion_sampler.py -> 3 passed
py_compile sampler, filtered-motion loader, baseline launcher, and task      -> passed
git diff --check against the route commit                                 -> passed
```

These checks verify object-mesh identity, asynchronous reset coverage, and the
explicit full-pool sampler wiring. They do not change the Probe's `UNPROMISING`
decision or authorize another full-pool training run.
