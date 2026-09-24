# P-20260924-crossobject-cm-action-info

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe.

## Question

Does the same *pre-action* wrist-z perturbation have learnable, task-relevant
physical consequences across object identities? This is a prerequisite for
training an object-conditioned Cm on actual simulator transitions. It does
not itself prove that Cm improves the actor.

## Fixed minimal design and decision

Use the frozen self-trained s3-airplane e260 actor (not the official actor or
the degraded multi-object adaptation). The object split is fixed: train
airplane/mug/toothpaste, held-out apple. For each partition, run 64 simulator
environments and randomize eligible contact states at global steps 50..200
stride 10 equally to wrist-z action `+0.1` or `-0.1`, following each action
for five physical steps under the actor. Save pre-action q, q velocity,
object state, executed action, object identity, five-step object state and
five-step contact count. Use new simulator seeds 182 (train) and 183
(held-out), one GPU at a time. The randomized collector must validate actor,
split and motion hashes, and reject clipped actions.

Primary physical label: signed five-step object z displacement multiplied by
five-step hand-object contact fraction, in mm. Estimate the ± treatment
difference within intervention step and object, with environment-cluster
uncertainty. Need at least 50 treated rows per training object; if fewer,
classify this action/data route `UNCLEAR` and change sampling before training
Cm. If the pooled training-object point effect is <5 mm or two of three
objects have opposite signed effects, do not train a Cm for this fixed action
family; move to contact-maintaining finger/wrist actions. If the gate passes,
train a shared geometric/action-conditioned Cm using each object's own mesh,
then test only on apple against no-action/state-only and raw-action controls.
No apple transitions may enter model fitting, calibration, normalization or
model selection.

Budget: one idle GPU, <=30 min per partition, <=100 MB each. Stop on input
drift, GPU conflict, incomplete followup, non-finite rows or budget overrun.

## Result

Pending. This Probe can only justify an information/model step, never a Cm
policy-utility claim.
