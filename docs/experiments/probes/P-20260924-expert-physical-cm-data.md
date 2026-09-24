# P-20260924-expert-physical-cm-data

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe.

## Question

Two randomized action families around the current self-trained airplane
actor produced weak task-aligned effects on most other objects, despite
the objects being physically graspable. Is the data distribution, rather
than merely the Cm architecture, withholding productive grasp transitions?

## Minimal protocol and boundary

Use the read-only official Inspire actor checkpoint **only as a simulator
data collector**. It may generate successful physical contact states and
randomized action transitions for Cm training/diagnosis. Its weights must
not initialize the final self-trained actor or appear in the final
Cm-on/off policy comparison. Any Cm trained from this data has explicit
expert-origin provenance; policy utility must still be demonstrated on a
self-trained actor.

Fix the same object-disjoint Cm split: airplane/mug/toothpaste for model
training, apple entirely held out from Cm fitting, normalization and model
selection. At contact states from the official actor, randomize wrist-z
`+0.1` vs `-0.1`, follow five executed simulation steps, 64 envs,
steps50..200 stride10, seeds186 train/187 heldout. The collector must pin
the official checkpoint SHA and motion split SHA, record actor role, exact
executed dose, object ID and five-step contact-supported object z.

Train-side continue gate: >=50 treated samples per object and pooled
contact-supported z contrast >=5 mm, with at least two of three objects
having positive point effects. If not, do not train another Cm on this
fixed action family; reconsider action distribution/target. If yes,
train a *shared* object-conditioned geometric/action model on train
objects only and evaluate apple against action-blind/state-only and raw
state+action controls. No offline accuracy result alone can prove policy
utility. A held-out run is only needed once the train-side gate passes.

One idle GPU per data run, <=30 min, <=100 MB output. Stop on source hash
drift, non-finite states, incomplete followup, dose mismatch, GPU conflict
or budget overrun.

## Result

Pending.
