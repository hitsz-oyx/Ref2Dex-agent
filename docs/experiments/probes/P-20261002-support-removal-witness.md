# P-20261002-support-removal-witness

Decision/Blocker Probe. Design frozen before new physics; implementation pending.
Follow D-20261002-support-removal-witness exactly. Source fixed frame0 controller
and.30rad ramp from d15ef0f; preliminary reason for choosing motion1 comes from
P-20261002-frame0-tracking-feasibility-r1/continuation_r1, explicitly candidate data.

Two fresh seeds508/509,96environments each;32per motion,16unchanged/16release per
motion/seed with private CPU generator seed eval+9000. Start original nativeframe0.
Use original reference goals and.30rad ramp through native plateau_stop. Beginning
next physical tick, release arm opens ALL12finger targets to0 and raises constant
absolute wrist-z target20cm, preserving orientation; control holds final goal.
Both remain native PD actions; no target-object reset, state writes or external
forces after initialization. Source model loaded for player construction only;
actor invocation forbidden and model/RMS unchanged. No clipping; PD error<=1e-5.

202steps globally; fullphase90steps for all motions; earlytermination/kappaoff.
Record full roots/nativeq/hand andobjectforce/actions/targets/progress, actual
initial states, arm/RNG, reference/phase/ramp, allphysical native metadata. Source
assets/checkpoints immutable. Correct thin-Y tabletop normal, actual fullmesh
support plane and independent world-up reconstruction. Native gravity on and
three actors perenvironment checked. Allmotion1 rows count in primary; no
selection on test pre-retention, force, geometry, outcome or release success.

Primary gates and fixed eleven-tick late window as in Decision Memo. All motion
and arm scores reported; force75 secondary and original failed results retained.
PROMISING only if EVERY gate passes; elseUNPROMISING or implementationUNCLEAR if
physical/inverse/input contract invalid. No policy-learning, sample-efficiency,
force-closure, multi-object generalization or formal Validation claim.

One free admitted GPU,<=600s/128MiB. Stop drift, nonfinite, clipping, PD mismatch,
wrong geometry, missing phase/window, native end or budget failure. Never stop
unknown GPU jobs. Preserve completed native phases if admission later rejects.

Prelaunch horizon amendment:202 ticks includes motion0 stop162+40; primary
motion1 window and every gate unchanged. No physics existed when amended.
