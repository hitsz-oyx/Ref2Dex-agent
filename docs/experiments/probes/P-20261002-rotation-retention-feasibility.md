# Independent rotation-only retention feasibility

Decision Probe, two fresh seeds571/572,768env each,256per motion,
64per motion/arm BEFOREphysics using private eval+15000. Source scratch
P0policy15ed218f..., original synthetic90plateau/physical105criterion,
initialXY±10mm, native gravity/object/geometry unchanged. All trials count.

Arms0unchanged,1duplicate unchanged,2fixed early rotation hold at progress
lift_start-8,3event rotation hold after FIVEconsecutive currentPRE-action
ticks with rootrise>=.03m/fullmeshclearance>=.02m and progress>=lift_start.
Latch current native q3..5 once; no velocity trigger, future state or outcome
selection. Preserve scratch actor XYZ and finger targets computed from each
arm's own current state. Desired heldrotation is clipped each tick to actual
current q3..5±abs nativePDscale3..5, recording the exact feasible target;
this is a declared actuator projection, not post-hoc clipping. Do not freezeXYZ.

Primary event arm3must improve pooled physical105rate by>=.05over EVERYarm0/1/2,
and be>=eachcontrol in EACHseed. Original105window stop-74..stop+30, rootrise
>=30mm/full-mesh clearance>=20mm onALL105ticks. No window/seed/controller
selection. Early arm2is a fixed strong comparator, no post-hoc primary switch.
All motions/seeds/counts and current-event ticks retained. Report event use and
goal changes descriptively, not event-conditioned causal estimates.

Native initialreset, dofnames/coupling, current70feature reconstruction, private
assignment/placement, all before/after inputs, fullmesh hull containment and PD
inverse verified independently. Verify targetXYZ/fingers exactly equal saved
base SAME-state target; hold changesonly3..5. All raw roots/q/forces/goals/actions
and latches retained202ticks. Sourceactor never called, ownactor unchanged.

Single fresh idleGPU4,<=900s/512MiB, no fits or policyupdates; CPU independent
geometry/controller/PD/statistics only. Unique committed run, pin all inputSHA;
abort only ownedchild on resource contention/drift/budget. PROMISING onlyifall
primary gates pass, elseUNPROMISING. No novelty/Cm/Validation/generalization or
mechanical support claim transfers from the originalworktree.
