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

Pending input smoke.
