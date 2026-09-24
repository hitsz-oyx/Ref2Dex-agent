# P-20260924-waterbottle-start-anneal

- Classification: Decision Probe.
- Cm: off.
- Source: failed but physically contacting waterbottle e340 checkpoint,
  SHA256 `e2a1611bb07e721c84029edea1dcb893c5bf5953317595f0693162e9d06d62e0`.

## Question and decision

Waterbottle e340 had near-zero contact from ordinary starts despite
substantial training contact under near-contact/lift resets. Can a linear
anneal of those special resets from their current 75% share to zero over
epochs340–380, followed by 20 ordinary-start epochs, restore approach
without losing the learned near-contact behavior? If unseen seed209 yields
>=16/64 held-lifts and >=0.30 contact, retain reverse curriculum as a
candidate for other hard objects. If not, stop this local waterbottle
continuation and investigate representation/reward or object-specific
physical control instead of increasing epochs.

Use the same corrected waterbottle motion, reward, seed70 and 64
environments. Continue e340→e400 with `--curriculum-anneal-start 340` and
`--curriculum-anneal-end 380`. One idle GPU, <=60 minutes, <5 GB outputs.
Stop on checkpoint/data drift, GPU conflict, nonfinite training or missing
e400 checkpoint. First full episode on seed209 is the fixed gate; this
remains a Probe.

## Results

Pending.
