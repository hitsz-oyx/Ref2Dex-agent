# Persistent hidden density: fixed-actor screen

P-20261003-latent-load-feasibility-r1 **COMPLETED / UNPROMISING**.
Code `2bd386b6983c86c5418ca34702e8797d1df4daed`, branch
`agent/cm-latent-load`. One fresh seed701,768 native trajectories,155136
controlticks,0 optimizer updates. Wall85.621018s;251273612bytes before the
terminal closeout file. Same corrected collision ownership, full25002vertex
geometry, and ALL105tick root-rise30mm/clearance20mm success protocol.

The airplane's actual nominal mass is2.593612997g at source density20kg/m³.
The single prospective heavy level uses density1000kg/m³, mass129.68065g and
all inertia entries×50, unchanged geometry/COM. Changes occur in body creation
before prepare_sim. Within each motion/actor cell,32 nominal and32 heavy
assignments are randomly selected before simulation. All force channels in
the deployed152-input actors use **nominal** weight for both loads; actual mass
is honestly recorded separately and is not an explicit policy feature. Current
SDK forces/velocities remain observed: this does not show that history is
necessary to identify load.

| Frozen own actor | Nominal successes /96 | Heavy successes /96 |
| --- | ---: | ---: |
| P0 |36|0|
| Cm-trained |36|2|
| Action-off-trained |47|0|
| Cold directQ (primary) |32|0|

| Motion | P0 nominal/heavy /32 | Cm nominal/heavy /32 | Off nominal/heavy /32 | ColdQ nominal/heavy /32 |
| --- | ---: | ---: | ---: | ---: |
|0|0/0|0/0|0/0|0/0|
|1|31/0|16/0|20/0|13/0|
|2|5/0|20/2|27/0|19/0|

Primary nominal33.333% passes>=25%; drop33.333percentage points passes>=15pp;
heavy0% fails the prespecified10..80% band.2/3gatespass, so the complete
decision is UNPROMISING. Two Cm successes do not override the primary gate
and do not establish training utility. Actors were trained earlier at nominal
physics and frozen here; this is a difficulty screen, not training across loads,
formal robustness validation, or a matched adaptation study. The single-seed
groups are not exact solver-state counterfactual pairs.

The same airplane in a zero-gravity/contact-free GPU sim responds to equal
.001N X force and1e-6Nm Z torque with light/heavy linear ratio50.0 and angular
ratio50.0000038. Relative Fdt/m error4.731788e-8; inverse(I)τdt error1.727873e-6,
both below2%. This verifies that the setter pathway changes GPU dynamics,
not just the SDK getter. Smoke uses zero damping; task damping stays.01.

Independent native audit recomputes all768 load draws, actual masses, positive
inertias/scales, creation/reset motion/arm agreement, nominal force denominators,
causal actor observations(maximum error0), all trained-actor NumPy forwards
(2.304496e-7), P0 forward(1.941453e-7), executed targets/native PD, all complete
105tick labels and full-mesh clearance(maximum5.974489e-8m). All pass.
Terminal independent closeout also reconstructs all768 raw counts/all3gates
and free-body equations without the aggregate helper. Protected inputs
unchanged; parent838110 and children838223/838876/840696 absent. No external
project writes, new subagent review, unknown PID control, or remote push.

Artifacts: `src/task/CmResidual/research/contact_response/output/`
`P-20261003-latent-load-feasibility-r1`: manifest, engineering.json,
engineering/native/audit logs, s701/initial.pt/trace.pt/option_decisions.pt,
physical_metadata/results/rows/panel_audit JSON, results and closeout_audit.
No new actor/Cm checkpoint; frozen input actors retained by identity.

Decision: close this **exact fixed density contrast**. No weight/seed/checkpoint
rescue or history adaptation fit from the failed screen. It rules out immediate
investment in this particular frozen-actor benchmark, not all latent physical
representations, not heavy-airplane trainability after retraining, and not the
Mission. Generic history adaptation remains prior art; a distinctive method
and matched policy-training gain remain unestablished.
