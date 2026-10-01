# Independent effective-finger feasibility: UNPROMISING

RunP-20261002-selective-finger-feasibility-r1, code9ef3050. COMPLETED112.349s;
1536fresh545/546native trajectories,202ticks each,279,708,481bytes. All255protected
inputs rechecked; independent NumPy full-mesh, current state, requested/executed
target offset, PD, private assignment and SDK-versus-URDF mapping pass. Clearance
error<=2.222e-7m. Three coupling/limit/mapping tests pass. Owned87539terminal0;
no source actor calls, external force, physics change or post-reset state writes.

| Arm | Adjustment | Motion0 /64 | Motion1 /64 | Primary motion2 /64 | Pooled /192 |
| --- | --- | --- | --- | --- | --- |
| 0 | Unchanged | 0 | 54 | 12 | 66 |
| 1 | Common curl | 0 | 39 | 21 | 60 |
| 2 | Index curl6 | 0 | 39 | 10 | 49 |
| 3 | Middle curl8 | 0 | 45 | 6 | 51 |
| 4 | Pinky curl10 | 0 | 55 | 10 | 65 |
| 5 | Ring curl12 | 0 | 56 | 6 | 62 |
| 6 | Thumb curl15 | 0 | 54 | 8 | 62 |
| 7 | Thumb yaw14 | 0 | 53 | 4 | 57 |

Fixed+0.15radatlift_start-8, persistent relative to evolving scratch base target,
with dependent-joint/native limits. None of six selective arms improves primary
motion2by20ppover BOTHunchanged and common curl or is no worse in BOTHseeds.
UNPROMISING. All3motions and8arms retained; the slight motion1pinky/ring point
differences never replace the predefined primary or imply learned-policy benefit.
Candidate selection is explicit mechanical feasibility, not formal Validation.

Actual SDK order confirms index6,middle8,pinky10,ring12,thumb yaw14/pitch15.
At the first intervention on motion1, thumb yaw targets are saturated/null64/64;
thumb curl47/64are null and all64partial/full saturated. Primary motion2curl
coordinates have NOtarget saturation/nulls at first intervention; thumb yaw is
partially saturated64/64but not null. These are executed TARGET offsets at the
decision, not future measured finger motion or proof of whole-trajectory policy
equivalence. Requested doses must not be mistaken for physical displacement.

The figure packetP-20261002-selective-finger-figure-r1reports every assigned
cohort's actual physical105counts, with source/output SHA and no trained Cm label.
Stop this exact positive-direction finger basis without sign/dose/timing scans.

Static pre-lift corrections and fixed slip reflexes are now poor candidates for
Cmutility. Next is a different actual policy-learning interface: continuously
train bounded independent target residuals from real returns, with a causal
action-conditioned short-physical auxiliary critic compared with identical
state-only auxiliary and no-aux critics. It must not change reward into model
predictions or use future realized motion as model input. Generic auxiliary PPO
is established; success would still need a distinctive method and Validation.
