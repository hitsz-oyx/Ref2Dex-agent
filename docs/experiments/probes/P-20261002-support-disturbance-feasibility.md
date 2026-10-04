# P-20261002-support-disturbance-feasibility

Classification: Decision / physical task calibration. Frozen specification in
[decision memo](../../archive/2026-10-04-research-governance/decisions/D-20261002-postlift-disturbance-feasibility.md).
The main journal objective remains unmet. All old30/105forecast and utility
gates remain unchanged; this NEW recovery90 task allows a bounded recovery phase.

Run id `P-20261002-support-disturbance-feasibility-r1`. Native seeds527/528,
768environments each, three motions,32balancedcells/motion. One explicit object
COM worldX impulse step, four fixed loads, two directions, four mechanical
recovery arms. Current state observed after the impulse; no source actor acts,
no object state reset after the first counted tick. Record all202ticks.

Primary motion1 eligible-load gates are frozen before launch, with lowest
eligible load selected. Single GPU, <=900s/512MiB. No training; GPU for native
simulation and frozen actor inference, CPU for independent NumPy geometry/control
audit. PROMISING permits a NEW prospective Cm information/utility design only.
No eligible dose stops this exact design without additional force/axis scans.
Terminal results will be recorded separately without modifying the frozen card.
