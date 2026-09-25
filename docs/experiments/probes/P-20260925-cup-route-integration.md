# P-20260925-cup-route-integration

- Classification: Decision Probe.
- Cm: off. All six experts are self-trained PPO checkpoints.
- Route remains a simulator-object-ID diagnostic, not deployable
  observation routing.

## Question and decision

Does the cup e340 specialist's 180/192 single-trajectory held-lift
signal survive in the mixed 12-motion simulator and raise pooled
coverage? Freeze two routes: previous five-expert route and the same
route with only `cup` changed to cup e340. Evaluate both on entirely
new seeds229–231, 64 first full episodes per seed, disabled early
termination. Use the same motion pool, checkpoints, environment config
and strict held-lift criterion; record per-object denominators.

Pass if new route improves pooled held-lift by >=9/192, cup contributes
>=12/18 held-lifts, and non-cup successes fall by no more than 6/174.
If passed, update the observation-driven router for the expanded expert
portfolio and then test it online. If not, diagnose whether the cup
specialist is sensitive to mixed-environment conditions. One idle GPU,
<=30 minutes, <200 MB outputs. Stop on checkpoint/config drift,
GPU conflict or incomplete episodes. This is a Probe, not formal
matched multi-seed training Validation.

## Results

Pending.
