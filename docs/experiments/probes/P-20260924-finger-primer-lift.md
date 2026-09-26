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

Status: `UNPROMISING` for the two-step macro's lift target. The first
`s189` run failed before any intervention because the new runtime flag
was omitted from the collector's in-memory configuration; no scientific
data were accepted. The same seed was rerun after the wiring fix as
`agent_crossobject_finger_primer_train_s189_h10_retry`. It completed,
and all first-step `+/-0.2` five-finger doses and common second-step
`+0.1` wrist-z doses were verified. Treated counts: airplane 222,
mug 274, toothpaste 226.

Close-minus-open ten-step contact-supported object-z effects were
+1.08, +0.33 and +3.78 mm respectively, pooled +1.64 mm with
environment-cluster 95% interval [-0.48,+3.56] mm. This missed the
predeclared >=5 mm gate; do not replicate or train Cm for this exact
two-step effect. Contact fraction did increase by +3.46 pp pooled,
interval [+0.36,+6.47] pp. Thus the finger primer changed grasp contact
but did not produce enough subsequent lift under this actor/horizon.
The physical finding suggests studying a *sustained* grasp-and-lift
option, rather than another one-step finger coefficient tweak.
