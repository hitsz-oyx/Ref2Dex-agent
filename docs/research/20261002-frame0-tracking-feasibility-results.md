# Fixed approach/lift controller: strict primary fails, geometry and force disagree

Implementation d15ef0f, with initial card/identifier preflight typo fixed before
any native launch. Run P-20261002-frame0-tracking-feasibility-r1 completes s506 on
admitted GPU5, then its parent FAILS on protected GPU5 admission for s507 after
other workloads occupy it. The completed panel and original FAILED manifest are
immutable. Continuation3c8c78f verifies original parent exited, all frozen inputs
and s506 outputs, and runs ONLY missing s507 on newly admitted free GPU1. It then
uses the unchanged frozen analyzer through local data symlinks. Continuation
COMPLETED66.09s, original53.14s:119.23s total active parent time, no repeated physics.
All198 continuation-protected hashes remain unchanged; all owned processes exit.

192 trajectories start at the original native frame0. Each has its full90-row
phase; simulation162steps globally, with finished-motion target clamped at its
own plateau end. This is not complete native reference episode retention.
Matched fixed reference controller0 versus.30rad ramped curl, fresh506/507,
96per arm,32per motion per arm. No actor calls, clipping, state teleport after
reset or external force. Actor/RMS unchanged; nativePDtarget error<=1.193e-7.
Independent reference/ramp/coupling/action/progress/force/clearance checks pass;
fullmesh world-up clearance agrees within3.804e-8m.

| Arm | Strict75 /96 | Motion0 /32 | Motion1 /32 | Motion2 /32 | Geometry-only75 /96 |
|---|---|---|---|---|---|
| Native label targets |0|0|0|0|0|
| +.30rad closure |5|0|5|0|32|

Pooled10% and every-motion5% gates fail; gain>=5pp passes. UNPROMISING for this
fixed controller. No larger dose, changed ramp timing, extra seed or success
relabeling is used to rescue the primary. A static suspended state is not evidence
that the same state can be reached along a reference approach. Here the positive
geometric pickup signal is motion1, whereas prior static strict-positive cases
were motion0/2; they are distinct controllers/starts, not contradictory comparisons.

Native metadata for all192 targets: one rigid object body, mass.002593612997kg,
flags0 (gravity enabled), gravity[0,0,-9.81000042], three actors per environment.
Weight.02544334N; historical independent force threshold.1N is3.9303times weight.
Net force above an arbitrary threshold is not pairwise grasp force. This mismatch
alone does not prove false negatives or causally explain any learner failure.
The prospective geometry-only SECONDARY reports32/32 motion1 trajectories above
frame0+3cm and fulltable clearance20mm for75ticks; only5/32 pass uninterrupted force.
All original primary labels remain UNPROMISING.

A separate POSTHOC measurement-component audit on prior suspended-start data:
geometry75 counts11/9/30/32 versus strict0/0/6/13 out of48 by dose. At.30rad all
16motion0 and16motion2 satisfy geometry75;19fail strictforce75, and none of these
geometric75 cases has a six-tick loss of force-proxy conjunction. This is an
important protocol uncertainty, not a replacement successful primary. Native mass
is measured in the separate tracking runtime and attributed accordingly.

Next Decision/Blocker: prospective support-removal witness from this fixed motion1
controller, randomized unchanged holding versus opening/retracting the hand after
the original phase. Check actual root/gravity/fullclearance and measured retreat,
without object-state writes. It distinguishes sustained hand support from another
support or measurement artifact before changing a scientific retention criterion.
No learned policy or Cm training gain follows from this witness.

Artifacts: original run, continuation_r1/results.json, physical_metadata.json in
both native panels; prior preload retention_component_audit_r1. Paper revision5
predates this tracking result; this document contains the newer evidence.
