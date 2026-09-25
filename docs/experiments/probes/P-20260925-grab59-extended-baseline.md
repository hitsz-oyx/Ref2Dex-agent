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

The e300→e380 continuation completed on GPU7 in 491 s at commit
`ecfc93e`. The checkpoint SHA256 is
`0d58fe9b41fec3ce2c33977022cf44b8b5db9e79007b2c1577f9d9260613ecb6`.
All four new-seed evaluations completed and each e300/e380 pair had
identical motion IDs and start frames in all 64 environments.

| Actor | Seed238 | Seed239 | Pooled held-lift | Successful identities | Mean contact |
| --- | ---: | ---: | ---: | ---: | ---: |
| e300 | 4/64 | 2/64 | **6/128** | 5/29 | 15.12% |
| e380 | 3/64 | 6/64 | **9/128** | 6/29 | 24.08% |

e300 succeeded on airplane 2/10 and one each of binoculars,
gamecontroller, mug and stamp. e380 succeeded on airplane 3/10,
toothpaste 2/8, and one each of apple, flashlight, mug and wineglass.
Mean maximum contact-supported lift rose from 1.82 to 2.67 cm, but
the strict five-consecutive-step held-lift gain was only +3/128.
The predeclared +10/128 and eight-identity gates failed. Status:
`UNPROMISING` for simply extending this same uniform shared actor.
Do not add more epochs to this local route. Favor a different
specialist/hierarchical or training representation for broad coverage.
This result covers the 59 compatible right-hand lift motions only.

Training manifest and checkpoint:
`outputs/Dexplore/agent_grab59_s70_e380/`. Evaluation results are
in its `eval_e{300,380}_s{238,239}_full/` subdirectories.
