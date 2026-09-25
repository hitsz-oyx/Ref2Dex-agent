# P-20260925-online-observation-router

- Classification: Decision Probe.
- Cm: off; the same five self-trained PPO experts in both arms.
- Question: Does a router that sees only the initial actor observation
  preserve the grasp coverage of the fixed simulator-object-ID route?

## Protocol and decision

Use the balanced observation SVC trained on seeds214/215, selected after
the prior offline diagnostic and checked on seed217. Freeze its model SHA
`9986fc3a5cae2ecfab95071d5a38d91d19813db7113ab525b3eaf55cc57c581c`
before running. On fresh seed218, evaluate the fixed object-ID router and
the online observation router for the same 12 converted motions, 64 first
full episodes, disabled early termination and matched checkpoints. The
observation router assigns an expert from the first 1442-D policy input
and keeps that expert until the environment resets. Reset metadata may
mark episode boundaries but must not supply the expert choice. Save
per-environment initial predictions and per-episode grasp results.

Pass the online fidelity gate if at least 90% of initial expert choices
match the fixed route and held-lift is no more than 2/64 below it on
seed218. If passed, this is a usable Cm-off, observation-driven routing
baseline for these known motions. If not, inspect classification and
execution mismatches before Cm experiments. One idle GPU sequentially,
<=15 minutes and <200 MB output. Stop on model/checkpoint drift,
incomplete evaluation, nonfinite action, or GPU conflict.

## Results

Pending.
