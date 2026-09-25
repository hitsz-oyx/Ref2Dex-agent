# P-20260925-grab59-extended-baseline

- Classification: Decision Probe.
- Cm: off. Continue the exact 59-motion shared PPO implementation.
- Source: self-trained shared59 e300 checkpoint SHA256
  `5f2686103ba3fbd9070b95d658a7ad08aff11fe22e3de56fbed16531160c037d`.

## Question and decision

The first 40-epoch e260→e300 continuation increased contact but only
reached 4/128 held-lifts on new seeds. Was that simply too short for
59 motions? Continue the same actor, optimizer, reward, curriculum
and fixed 59-motion input from e300 to e380 on training seed70.
Evaluate frozen e300 and new e380 on identical, new seeds238/239,
64 first full episodes each, disabled early termination. Count
held-lifts and successful object identities over the entire 29-object
pool, and check per-object regressions.

The extended shared route is promising only if e380 exceeds e300 by
>=10/128 held-lifts, reaches >=8 object identities with at least one
held-lift, and does not reduce airplane, duck or toothpaste to zero.
If passed, consider this actor as a better Cm substrate and decide
whether more training is worth a Validation. If failed, stop uniform
shared continuation on this 59-motion pool and focus on specialist/
hierarchical training or representation change. One idle GPU,
<=60 minutes training and <=20 minutes evaluation, <5 GB new output.
Stop on input/checkpoint drift, nonfinite training, GPU conflict or
incomplete endpoint/evaluations. Probe only; 59 filtered right-hand
lift motions do not cover all raw GRAB sequences.

## Results

Pending.
