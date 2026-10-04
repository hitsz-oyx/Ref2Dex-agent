# Complete continuous policy-learning Probe: UNPROMISING

Fixed experiment `P-20261002-continuous-critic-policy`, scientific code
`e37eca3`, completed in unique execution `...-resume-r3`; runtime continuation
code `99a2521`. The original interrupted parent, native-memory failure,
engineering reproduction and update-17 audit failure remain unchanged.

All 20 native training panels547--566, 20 updates and final-only deterministic
panels568/569 complete:15360training and1536evaluation trajectories, 202native
ticks each, 3840training episodes/learned variant and9120total actual optimizer
steps. Fresh independent closeout rebuilds every cohort count and all seven
gates, checks675protected paths and unchanged scientific files. Combined recorded
execution including conservative prior reserves3311.593s, retained5,303,126,219
bytes, below3600s/6GiB. No valid panel/update is repeated or selected.

| Cohort | Unchanged | Cm | State-only | No auxiliary |
| --- | ---: | ---: | ---: | ---: |
| Pooled, each /384 |137|129|129|147|
| Seed568, each /192 |68|59|65|71|
| Seed569, each /192 |69|70|64|76|
| Motion0, each /128 |0|0|0|0|
| Motion1, each /128 |114|109|108|118|
| Motion2, each /128 |23|20|21|29|

Cm33.59375%, state-only33.59375%, no-auxiliary38.28125%, unchanged35.67708%.
Cm gains0pp over state-only and-4.6875pp over no-auxiliary, against the frozen
requirement>=5pp over BOTH. Cm also fails pooled noninferiority to unchanged,
both seed checks against no-auxiliary and seed568against state-only. Six of seven
conditions fail: **UNPROMISING**. The exact one-step auxiliary-critic recipe stops
without coefficient, learning-rate, checkpoint, seed or duration searches. This
does not establish statistically significant harm or universal Cm ineffectiveness.

The three learned variants share initial actor/critic parameters and actual
interaction/optimization budgets. Value is state-only; the Cm physical branch
consumes current state plus executable target-minus-current-q12 and predicts real
next object position/velocity. Separate actor/critic encoders prevent a direct
auxiliary path to the actor; joint norm clipping still couples actor scaling to
critic gradients. Generic auxiliary PPO is established, not a new method.

All22native trajectory/geometry/likelihood/target/PD audits complete. Independent
NumPy checks60predetermined first minibatches, every actor/critic gradient and
first Adam update; the remaining9060steps are retained/schedule checked but NOT
independently replayed. Original u17float64 audit fails at one near-zero ReLU
branch. Same-GPU saved-batch replay reproduces ALL saved gradients/forward/losses
exactly with zero updates; a separate witnessed-branch NumPy correction passes
the SAME scalar tolerances (gradient1.2223e-6) after checking the FP32 rounding
enclosure. It is explicitly retained separately, not a retroactive original pass.
[Branch decision](../archive/2026-10-04-research-governance/decisions/D-20261002-continuous-gradient-branch-correction.md).

GPU1native allocator failure leads to an explicitly recorded same-model GPU4
migration. Initial packets agree; no cloned solver/trajectory claim follows.
All controls share their device within a panel. First failed incomplete550and
its engineering-only reproduction never enter training or evaluation.

This is one optimization seed, one low-mass airplane, three synthetic-reference
motions, privileged current object/hand state and a bounded202-tick task.
Physical105means root rise>=30mm and complete-mesh table clearance>=20mm on ALL
105specified ticks. It does not prove force closure/mechanical support in every
successful learned-policy rollout. The earlier support-removal witness concerns
a different base cohort. No formal Validation, generalization, hardware, novelty
or journal readiness is established.

Evidence in isolated continuation:

- `src/task/CmResidual/research/contact_response/output/P-20261002-continuous-critic-policy-resume-r3/`
- `.../P-20261002-continuous-critic-policy-closeout-r3/results.json`
- `.../P-20261002-continuous-gradient-diagnosis-r1/`
- `.../P-20261002-continuous-gradient-correction-r1/`
- `paper/native-v11/manuscript-v11.pdf`: native LaTeX,21pages,21tables/4figures;
  all231hashed inputs and previous revisions6--10unchanged. Result table rebuilt
  from all1536rows; pages16/17visually inspected. Minor TeX overfull warnings
  remain within visible margins; no text clipping detected.

The journal objective remains unmet. The next decision must concern how
action-conditioned physical information reaches task-relevant policy learning;
another one-step critic weight/seed scan is not warranted.
