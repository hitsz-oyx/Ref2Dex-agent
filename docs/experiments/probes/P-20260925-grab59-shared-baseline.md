# P-20260925-grab59-shared-baseline

- Classification: Decision Probe, conditional on the 59-motion input smoke.
- Cm: off. Actor initialized from self-trained airplane e260 checkpoint.
- Coverage: all 59 currently filtered single-right-hand GRAB lift-like
  trajectories across 29 objects; not all 1335 raw GRAB sequences.

## Question and decision

Can a single PPO actor gain useful held-lift coverage when continued on
the whole available right-hand lift pool rather than the earlier 12-motion
subset? Train seed70, 64 environments, e260→e300 with the same optimizer,
reward and sampling setup as the 12-motion Probe. Evaluate e300 and its
source e260 on identical new seeds221/222, 64 first full episodes each,
disabled early termination. Report pooled and per-object held-lift and
contact. A meaningful positive signal requires e300 to exceed e260 by
at least 10/128 held-lifts, cover at least eight object identities with
one or more held-lifts, and not reduce the three previously strong
airplane/duck/toothpaste identities to zero. If it passes, consider the
shared actor as a Cm substrate; if not, retain the observation-driven
specialist baseline and investigate curriculum/object coverage.

Use the frozen `filtered_geometric_dexplore` motion spec from the input
gate. One idle GPU, <=60 minutes training plus short matched evaluation,
<5 GB outputs. Stop on asset/input drift, nonfinite training, incomplete
evaluation, occupied GPU or missing e300 checkpoint. This is a Probe,
not Validation of full-GRAB success.

## Results

The input smoke passed. One-GPU continuation of source e260 to e300
completed on seed70 with all 59 filtered motions; the e300 checkpoint
was produced in `outputs/Dexplore/agent_grab59_s70_e300/`. Source and
e300 were then evaluated on identical new seeds221/222, 64 first full
episodes each, without early termination. All four evaluation manifests
report `COMPLETED`.

| Actor | Held-lift | Positive object identities | Mean contact fraction |
| --- | ---: | ---: | ---: |
| Source e260 | 2/128 | 2/29 | 0.1546 |
| Shared59 e300 | 4/128 | 4/29 | 0.1952 |

The e300 successes were airplane 1/10, phone 1/6, toothpaste 1/8 and
wineglass 1/8. Source e260 succeeded on airplane 1/10 and duck 1/2.
Thus the 40-epoch continuation increased contact by about 4.1 percentage
points but gained only 2/128 held-lifts and lost duck's isolated success.
The prespecified +10/128, eight-object coverage and no-collapse joint
gate **failed**. Result: `UNPROMISING` for this short, uniform shared
continuation as a broad grasp baseline. This does not establish that
longer training, better sampling or other architectures cannot work.

The current DExplore loader truncates a motion directory to `num_envs`
before setup, so this 59-motion experiment reaches the 64-environment
pool limit. Scaling to the remaining filtered right-hand interactions
requires a loader/sampling design that preserves each environment's
object mesh. The raw GRAB dataset also includes left-hand or other task
types outside this experiment.

Matched source evaluations are under
`outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/eval_s221_e260_full_grab59probe/`
and `..._s222_...`; e300 evaluations are under the new run's
`eval_s221_e300_full/` and `eval_s222_e300_full/`.
