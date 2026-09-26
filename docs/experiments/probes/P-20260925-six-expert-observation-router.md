# P-20260925-six-expert-observation-router

- Classification: Decision Probe.
- Cm: off. Six self-trained PPO checkpoints.

## Question and decision

Can the initial 1442-D actor observation recover the newly frozen
six-expert object route, including cup, on unseen seed232? The prior
training feature exports from seeds214/215 are independent of which
expert was run; relabel them using the new route. Export initial
features on seed232 with the new route. Train the previously selected
balanced SVC with the same hyperparameters, without tuning on seed232.

Pass if held-out accuracy is >=90% and all airplane, cup, duck, mug
and toothpaste examples choose the specified expert. If passed,
evaluate online observation routing on fresh seed233 and compare
expert choices and held-lifts with the fixed six-expert route. Online
pass requires >=60/64 initial choices matching the fixed route,
all cup examples choosing the cup expert, and held-lifts within 6/64
of fixed route while cup held-lifts are >=4/6. If failed, inspect
only the confusion matrix before choosing a different representation.
One GPU for <=10 minutes and <100 MB outputs; stop
on incomplete episodes or config/checkpoint drift. This is a Probe,
not policy-utility Validation.

## Results

Initial observation classifier passed: 62/64 correct on held-out
seed232 with the unchanged balanced SVC. All airplane, cup, duck,
mug and toothpaste examples are correct. The only errors are one
cubesmall and one waterbottle example, both sent to cup. Model SHA256:
`1fa84c94891d63824be0100b5eb78f4aa1e37825c6d71fff65e517c0c7f2fc14`.
Feature and config hashes, confusion matrix and per-object counts are
in `outputs/CmResidual/agent_six_expert_router_model_20260925/heldout_s232.json`.

Online seed233 passed the predeclared gate. All 64 environments had
matched motion IDs and start frames across runs. Initial expert
choices matched the fixed route in **60/64** environments; all six
cup examples selected `cup_e340`. The four errors sent three
waterbottle and one cubesmall examples to `cup_e340`.
Held-lifts were **27/64** for observation routing and **21/64** for
fixed six-expert routing; cup was **6/6** in both. Results and manifests
are under `outputs/CmResidual/agent_six_router_{fixed,obs}_s233/`.
Because the same expert may have different outcomes in repeated
simulator runs, the 27 versus 21 difference is not evidence that the
observation classifier improves grasping; the useful finding is
that the nonprivileged route preserved the new cup coverage in this
Probe. The classifier still needs wider seed and object validation
before a formal claim.
