# P-20261003-latent-load-feasibility

Classification: Decision Probe. Prospective design; results initially absent.
Separates negligible task effect versus useful, nondegenerate unknown-load
headroom. Cheapest useful screen: frozen own actors, not training adaptation.
Decision memo: D-20261003-latent-load-feasibility.

Fixed panel: evaluation seed701;768 independent placements;3 synthetic motions;
4 frozen own arms P0/Cm/action-off/coldQ;32 environments per motion×arm×load.
Private arm seed16701, placement11701, load31701. Load split is random within
each motion/arm, before any simulation. Density20 or1000kg/m³; mass/inertia
factor1 or50; COM, shapes, corrected collision filters and reset packet unchanged.
Geometry25002vertices; dt1/30;202ticks; success ALL105 root-rise>=30mm AND
tabletop-clearance>=20mm. No early termination, omitted motion, or subset gate.
All policy force channels use nominal mass2.593613g even for the heavy object;
actual mass is recorded honestly in native physical metadata and excluded as
an explicit input/normalization. Current SDK forces/velocities remain observed,
so this is not a force-unobservability experiment. Inputs remain privileged
simulator states; no tactile/hardware claim.

Primary coldQ pooled over3motions has96 environments/load. All gates required:
1. Nominal success >=25%.
2. Heavy success between10% and80%, inclusive.
3. Nominal-minus-heavy success >=15 percentage points.

PROMISING requires all three, otherwise UNPROMISING. If implementation invalid,
run FAILED and no scientific label. Additional arms/motion counts descriptive,
not gate alternatives. Exact solver-state pairing or counterfactual effect is
not assumed. No policy/Cm update, no matched utility claim, no independent
training seeds, no sample-efficiency inference.

Engineering prerequisite: same airplane on GPU, zero gravity/no contacts;
single1/60s step with.001N worldX force and1e-6Nm worldZ torque. Explicit
pre-prepare mass/inertia update with recomputeInertia=False; linear response
Fdt/m and angular response inverse(I)τdt, and factor50 response ratios must
all match within2%. This validates this setter pathway only. The engineering
smoke changes damping to0 to isolate integration; task damping stays.01.

Execution: native Python3.8/IsaacGym then Torch, one freshly admitted idle GPU,
max600s/512MiB, owned process group guard, unique output, fixed code commit
and protected-input SHA. Independent full native/PD/full-mesh label replay,
all768 assignments/masses/inertias/nominal-feature weights, all causal152
observations and trained-actor NumPy forwards. Any failure stops the run.
