# P-20261001-static-hold-feasibility

Decision/Blocker Probe. Is there an unmodified native reference pose that a
fixed PD controller can retain physically, or should pose/grasp construction
be reassessed before any further learner? This tests this exact static controller,
not intrinsic feasibility of all actions or a learned/novel method.

Use the same three synthetic references, initialize at first stationary row
73/51/63 through the actual native reset path. All object/joint velocities zero.
Hold the ACTUAL clamped/coupled initial native q as an absolute PD target for
90 physical ticks. Invert the actual native offset/scale and dependent action
mapping; canonicalize six null commands to0. Verify nominal PD target error
<=1e-5 at every tick and no action clipping. Model/actor is loaded only for
native player construction and is NEVER invoked. Actor/RMS remain frozen.

Fresh seed502/503,96env per panel,32per motion, one admitted GPU. Disable early
termination/adaptive kappa and curriculum. Each90-step mechanical trajectory
must complete; no native reference-horizon task or frame0 pickup is claimed.

Require75 consecutive physical ticks in90 with object root>=original reference
frame0+3cm AND>=its elevated initial native height-1cm AND both force proxies.
Additionally require ALL object collision-mesh vertices>=20mm above the highest
table-mesh point in table-local z. Compute from full native xyzw poses and the
actual scale1/origin0 meshes, not sparse object samples. This is a conservative
plane clearance test, avoiding supported tilting; it still is not a force-closure
certificate. Save object root/native q/force vectors/progress/clearance for audit.

PROMISING iff>=50%pooled AND>=25%in EACH motion. OtherwiseUNPROMISING for the
unmodified static controller. Report height/proxy-only labels separately and do
not replace the primary clearance gate. Positive permits a separate frame0 PD
tracking/teacher feasibility design; negative stops this exact static controller
and returns to grasp pose refinement/feasibility, not more epochs of old PPO.

One GPU, two sequential native panels, <=600s/128MiB total. No neural training
or policy inference. CPU saved-label audit allowed. Stop drift, nonzero initial
velocities, nonfinite trajectory, wrong mesh/pose conventions, clipping, mismatch,
incomplete90-step coverage or budget overrun. Source checkpoints/data read-only.
