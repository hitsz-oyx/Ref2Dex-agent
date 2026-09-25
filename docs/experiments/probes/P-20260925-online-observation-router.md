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

Both sequential one-GPU evaluations on unseen seed218 completed. The
observation router's saved initial expert choices matched the fixed
simulator-object-ID route in **64/64** environments. The fixed route had
**16/64** held-lifts and mean contact fraction 0.3233; the observation
route had **19/64** held-lifts and mean contact fraction 0.3670. The
predeclared fidelity gate **passed**. Initial motion IDs, start frames
and episode lengths were identical for every environment, while 11/64
held-lift outcomes differed. This shows independent simulator executions
are not identical even with the same seed and expert choices; the +3/64
must not be read as an efficacy gain from observation routing. The
result is `PROMISING` as a usable observation-driven Cm-off baseline on
these same twelve known trajectories.

The router takes only the actor's initial 1442-D observation to choose
an expert and holds that choice through the episode. It uses reset IDs
only to know when to classify again. Its expert ID is saved in
`outputs/CmResidual/agent_router_observation_s218/initial_routes.json`.
Full paired results and manifests are in
`outputs/CmResidual/agent_router_fixed_s218/` and
`outputs/CmResidual/agent_router_observation_s218/`. The model comes from
`outputs/CmResidual/agent_observation_router_model_20260925/router.joblib`.

Next decision: the baseline is ready for a Cm-on/off option test with the
same expert set. Cm must alter a meaningful policy choice, and the
comparison needs an action-aware and an action-shuffled control; offline
classification accuracy alone does not establish Cm value.
