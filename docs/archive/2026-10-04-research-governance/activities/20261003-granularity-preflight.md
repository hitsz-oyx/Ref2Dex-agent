# Granularity Probe preflight

User authorized the fixed-data finer-granularity comparison. Existing
`agent/cm-surface-prior` branch; all original/external trees remain read-only.
Eight fixed models, same2048windows perhand (MANO254parents/50objects,
Inspire247parents/50objects), original held63/30parents.64identical supervised
targets,64/256context crossed with4neighbor means/ordered detail; all23107
parameters/common initialization/1500x32 schedule. Density increases compute,
not supervised data. Main model calculation uses fresh idle GPU admission.

The existing user-authorized implementation supervisor reviewed the new
model/extraction/fitting/runner/audit read-only and found no launch blocker.
Cached neighbor rank was independently checked on both original2048window
banks, with no descending distances. Its two coverage suggestions were applied:
all selected dense rows check rank order; the tiny SE3/feature/isolation/update
smoke is a retained guarded phase before extraction.

First ad hoc CPU smoke stopped at an assertion after applying a synthetic
world transform in float32. Repeating the coordinate-invariance test with a
float64 synthetic transform (avoiding new world-position rounding before small
displacement subtraction) passes the ORIGINAL2e-5/2e-6bounds. Production code
and tolerances unchanged. The retained tiny smoke passes:2windows/3updates,
SE3max1.192093e-7, NumPyforwardmax4.656613e-10;49mean slots collapse to the
old22features, mean is neighbor-permutation invariant and detail distinguishes
the same rearrangement. This is engineering evidence, not accuracy evidence.

Prospective [card](../experiments/probes/P-20261003-cm-granularity.md) fixes
1800s/2GiB, perhand10%gain gates, full matrix reporting and a stop after it.
Hand movement remains realized future input; legacy physics/eval reuse and
single initialization explicitly limit inference. No policy benefit claim.
