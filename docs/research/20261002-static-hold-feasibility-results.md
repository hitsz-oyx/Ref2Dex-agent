# Static native reference-pose feasibility and corrected geometry

Run P-20261001-static-hold-feasibility-r1, code1e38cf6, COMPLETED74.72s,
8.29MiB, two fresh seeds502/503,192 elevated zero-velocity trajectories of90
steps. No learned actor call; model/RMS unchanged. Nominal native PD goal error
<=2.981e-8. All186 protected input hashes remain unchanged.

The fixed unmodified pose does not pass the necessary height/proxy criterion:
0/192 achieve75 consecutive steps; each motion0/64. This necessary failure is
independent of any table-clearance calculation. It stops this exact controller,
without proving intrinsic impossibility of these grasps or all control methods.

The original clearance implementation used local Z as tabletop normal. Actual
mesh extents show local Y is the thin axis; the native table quaternion maps
local -Y to world +Z. Therefore the original full-clearance gate has status
INVALID_TABLETOP_AXIS; its recorded UNPROMISING counts and original files remain
intact, but those geometric distances are not interpreted physically.

Separate offline correction68bf178 reuses full saved poses, without new physics.
Full25002 object vertices and21304 table vertices define the upper support plane.
Correct initial clearances are307.03/130.93/153.56mm by motion. Independent NumPy
world-up support and corrected table-local computation agree within1.172e-7m.
Corrected75-step count remains0/192. POSTHOC_GEOMETRY_CORRECTION is a diagnostic,
not prospective Validation or replacement of the invalid primary measurement.

Saved-data joint audit (scripts/audit_static_joint_tracking.py) checks URDF joint
semantics directly. Wrist joints3:6 are CONTINUOUS, with no lower/upper position
bounds. The geometry helper's +/-pi fallback cannot establish an actual violation.
Motion2 joint5 goal-4.59205rad is tracked in[-4.59234,-4.59181]; all bounded finger/
translation goals satisfy their stated bounds. No wrist wrap correction is warranted.
Maximum realized wrist-position deviations26.903/.351/.132mm and finger deviations
.0661/.1940/.0693rad document actuator behavior, without identifying the failure cause.

Next decision is whether bounded extra finger closure supplies sufficient gripping
in these exact suspended starts. That is a different mechanical controller Probe,
not more epochs or larger doses selected after failed tests. A positive result
would justify a separate approach/lift teacher experiment, not policy utility.

Artifacts: src/task/CmResidual/research/contact_response/output/
P-20261001-static-hold-feasibility-r1, subfolders tabletop_axis_correction_r1 and
joint_tracking_audit_r1. Original input hashes and trajectories remain protected.
