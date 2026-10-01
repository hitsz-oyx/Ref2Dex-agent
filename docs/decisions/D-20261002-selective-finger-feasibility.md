# Frozen selective effective-finger feasibility

Question: does replacing the common curl synergy with independently actuated
finger directions create mechanically useful long-retention corrections? Prior
natural feedback fails; common curl damages motion1and does little for motion2.
This is a DIFFERENT action representation, not a common-curl dose/trigger scan.
No new Cm fitting until feasible actions are established.

Fresh native545/546,768env each, all3motions,8arms32/motion/seed. Private
assignment seed+12000 and original XY seed+11000uniform+/-10mm. Original scratch
reference-target actor, synthetic90frame holding references,202ticks, unchanged
mass/materials/gravity and physical105 on every stop-74..stop+30 tick. No source
actor actions/weights, external force or state writes after combined reset.

Native independent coordinates [6,8,10,12,14,15]; dependent7/9/11/13are1.05of
their parents,16=.6*15,17=.8*15. SDK DOF names are checked against the actual
URDF: four distinct non-thumb proximal/intermediate pairs and named thumb
yaw/pitch/intermediate/distal. No finger identity guessed from XML ordering.

At lift_start-8, persist one fixed target adjustment relative to each evolving
base target:0unchanged;1common curl+0.15rad on6/8/10/12/15;2only6+0.15;
3only8+0.15;4only10+0.15;5only12+0.15;6only15+0.15;7only14+0.15.
Wrist targets are untouched. Parent limits incorporate dependent limits before
reconstructing children; thumb yaw clips to native limits. Save requested AND
realized six-coordinate target deltas at every actual tick. Saturation-induced
null commands must remain visible, not relabeled as distinct physical actions.

Primary is preselected weak motion2, with64perarm: at least ONE of selective
arms2--7improves physical105 by>=20percentage points over BOTH unchanged and
common curl, with no worse rate against either in EACH seed. This is an explicit
six-candidate mechanical feasibility screen, not Validation or a learned policy.
Motion2is fixed before new physics because previous controls are weak there;
all motions, pooled outcomes and all arms remain mandatory reports. No new
primary can replace the former failed all-motion policy-training gates.

Positive permits a separate conditional-action information/learning design with
privileged per-motion/action mean controls; selecting a better static finger arm
alone cannot establish Cm utility. Failure stops this exact positive-direction
finger basis without signs/doses/timing scans. Return to acquisition/contact
representation rather than more prediction fitting. Negative labels retained.

One freshly admitted GPU, whole run<=900s/512MiB, all1536trajectories retained;
CPU independent raw context/full-mesh labels/private assignment/native DOF
mapping/target deltas/PD reconstruction. Three small CPU unit checks validate
native coupling/limit invariants, not neural model computation. Commit fixed
code before launch, hash all protected inputs before/after. No new external
authorization boundary; method novelty and actual trained Cm utility still absent.
