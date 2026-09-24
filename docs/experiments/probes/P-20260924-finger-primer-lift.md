# P-20260924-finger-primer-lift

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe.

## Question

The self-trained multi-object actor reaches hand-object contact but rarely
sustains a >=30 mm lift on the unseen apple. A single wrist action has a
strong immediate effect yet unstable closed-loop value. Does a short,
grasp-preserving action sequence create a more task-aligned, physically
learnable distinction across training objects?

## Protocol and decision

At pre-contact states under the pinned **self-trained** e260 actor, randomly
choose closing versus opening the five independent finger flexion commands
for one step (`+/-0.2`), then apply the same wrist-z lift command (`+0.1`)
in both arms on the next step. Follow ten actual simulator steps from the
first action. Use train-object split airplane/mug/toothpaste only,
64 environments, steps 50..200 stride 10, seed 189. Do not train on or
inspect apple for route selection. Record both exact executed doses,
ten-step contact fraction, object-z displacement and survival.

Primary outcome: survival-adjusted ten-step object-z displacement times
ten-step hand-object contact fraction. Continue to an independent train
seed only if the close-minus-open point effect is >=5 mm pooled,
positive on at least two of three objects, with >=50 treated rows/object;
also report contact-fraction effect. This is a Probe gate, not evidence
that Cm itself matters. If failed, stop small two-step finger/wrist action
adjustments and revisit higher-level supervision or state coverage.

One idle GPU, <=30 minutes, <=100 MB output. Stop on source hash drift,
clipping/dose mismatch, GPU conflict, non-finite states, incomplete
followup or budget overrun.

## Result

Pending.
