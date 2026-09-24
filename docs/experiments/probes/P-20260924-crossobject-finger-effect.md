# P-20260924-crossobject-finger-effect

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe.

## Question and decision

The same wrist-z command moves the hand in four objects but only produces
material five-step contact-supported object motion on airplane. Does a
coordinated finger-closing action instead create a task-relevant *contact
persistence* effect across object identities, worth modeling with Cm?

Randomize a single pre-contact-state action step to +0.2 or -0.2 on the
independent native Inspire flexion commands `(6,8,10,12,15)` (index,
middle, pinky, ring, thumb pitch); all other actor actions remain unchanged.
These joints have positive flexion limits in the pinned URDF. Follow the
frozen self-trained e260 actor for ten steps. Fixed split: train
airplane/mug/toothpaste, held-out apple. Seeds184/185, 64 environments each,
steps50..200 stride10. Require exact unclipped executed doses and at least
50 treated rows per train object. The reference actor checkpoint is not used.

Primary outcome: ten-step hand-object contact fraction difference between
finger-closing and opening. Score resets before complete followup as zero
contact for the interrupted remainder; report reset differences. Continue
to an object-conditioned Cm only if pooled train effect >=+5 percentage
points and at least two training objects are nonnegative. If absent, stop
this fixed synergy and reconsider the action family or supervision target.
Held-out apple is a locked generalization test, not used to pick joints,
delta or thresholds. Secondary: contact-supported object z motion.

One idle GPU, <=30 min and <=100 MB per partition. Stop on source hash drift,
GPU conflict, incomplete step blocks, dose clipping, non-finite transitions
or budget overrun. Physical effect is not a per-state counterfactual or Cm
policy utility.

## Result

Pending.
