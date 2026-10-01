# Post-lift disturbance calibration: no eligible task

Frozen721b44d, `P-20261002-support-disturbance-feasibility-r1` COMPLETED115.440s,
372,347,518bytes at terminal closeout. All248protected inputs independently match.
GPU1 independently admitted for each native panel; CPU geometric/control audit.
Seeds527/528 retain all1536trajectories and202physical ticks each. No Cm or actor
training. No initial XY placement perturbation in this explicit NEW task variant.
One-step force is applied only to the object COM, with no torque, all other
bodies/ticks zero. SDK actor and rigid-body indices, randomized32cells/motion,
native target/PD execution and causal current-state inputs independently agree.
Full-mesh geometry through validated convex hull agrees within8.43e-7m.

Primary motion1: prior geometry10 is128/128 for EVERY load. Recovery90 counts:

| Issued force N | Unchanged /32 | Follow /32 | Oppose /32 | Curl /32 |
| --- | --- | --- | --- | --- |
| 0 | 32 | 32 | 32 | 29 |
| .05 | 32 | 32 | 32 | 26 |
| .15 | 32 | 32 | 32 | 27 |
| .45 | 30 | 31 | 31 | 25 |

No nonzero load reaches the required unchanged10--80% recovery range or the
required15pp /10pp contrasts. **UNPROMISING**, no eligible load and no selected
force. End this exact one-step worldX design without force/axis/window scanning.
The opposite-direction arm's31/32 at .45N also prevents interpreting following
as a direction-specific recovery mechanism. Curl is weaker descriptively; these
finite counts do not prove that tightening universally harms grasp.

All other motions reported: motion0 recovery90 is0for every cell; motion2 has
only1/32 in opposing at .05N, all other arm/load groups0. These do not replace
the primary or establish a new learned baseline. Both outcome90 and the recovery
window belong to this task only; old failed30/105criteria stay unchanged.

A separately retained descriptive response audit uses actual pre/post-impulse
object velocities. Primary signed mean delta-vx by load0/.05/.15/.45N is
.001313/-.001536/-.001789/.026192m/s. Contact reactions remain active; requested
force/dt/mass is not an observed free-flight acceleration. Exact issued-force
tensors plus SDK execution and observed responses are documented, not replaced
by a prescribed object velocity reset. This is not a new success gate or formal
force-response causal conclusion.

This experiment excludes this finite perturbation recipe as a useful recovery
curriculum. It does not refute Cm, disturbance recovery generally or larger/other
loads. Broad modeling and recovery are established in prior papers; distinctive
methodology, fresh matched Cm-on/off trained-policy benefit and journal readiness
remain unmet. Next work must address those requirements rather than expanding
these failed calibration families.
