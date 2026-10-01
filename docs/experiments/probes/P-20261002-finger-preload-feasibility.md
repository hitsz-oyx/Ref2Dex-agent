# P-20261002-finger-preload-feasibility

Decision/Blocker Probe; no actor calls or neural fitting. Compare predetermined
absolute finger targets generated from the same unmodified elevated native reset.
Independent curl offsets0/.05/.15/.30rad at joints6,8,10,12,15, with coupling
7/9/11/13=1.05parent,16=.6*15,17=.8*15. Clamp parent by all native parent/dependent
bounds. Wrist0:6 and thumb yaw14 unchanged. No action clipping. Native PD inversion
error<=1e-5 each step. Unwrapped continuous wrist angles are valid and preserved.

Fresh504/505,96env each,32eachmotion; randomly permute8 copies of each dose within
motion with private CPU seed eval_seed+7000 before first action. Each trajectory
90physical ticks with complete saved root/q/force/action/target/progress trace.
Zero initial velocities, disableearlytermination/kappa, no native episode or pickup
claim. Actor286 source loaded only for construction; its model/RMS stay unchanged.

Primary:75 consecutive within90 requiring root>=original frame0+3cm AND>=elevated
initial root-1cm, both force proxies, and ALL object collision vertices>=20mm above
actual thin tabletop upper support plane. Determine local thin axis from full mesh,
its sign from actual xyzw rotation and world-up alignment. Save full table pose.
Independent CPU NumPy world-up support reconstructs clearance/force/longest labels.

For EACH of three positive doses, prospective candidate gate: pooled75>=50%, EACH
motion75>=25%, and pooled75 minus zero>=25percentage points. PROMISING iff at least
one dose passes all gates. Report all arms; candidate selection is not Validation
and no causal policy-learning gain or force-closure claim follows. If none passes,
UNPROMISING and no dose/seed/epoch expansion of this exact family. <=600s/128MiB,
one free admitted GPU. Stop input drift, nonfinite, clipping, limits/coupling mismatch,
wrongtableorientation, incompletehorizon, source changes or budget overrun.
