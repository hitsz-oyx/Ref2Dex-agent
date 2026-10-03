# Oracle capacity of an inertial/rigid-transport coupling representation

Decision Probe; Mission and final claim unchanged. Previous goal turn was
progress: matched calibration rejected normal frozen-feature transfer and
preserved4800actual updates/audit-only recovery. Do not repeat that family.

Question: can causal hand-imposed rigid transports explain object flow beyond
an equally outcome-fitted state-only inertia family? Insufficient capacity
closes this EXACT scalar/endpoint family before training; useful capacity
permits learning the coupling from deployment-available inputs. Cheapest
method is an offline oracle-capacity screen using existing corrected traces,
not new neural pretraining, native collection or policy optimization.

## Fixed representation and scope

Reuse the384whole held episodes/6144windows (16fixedticks) in completed
P-20261003-surface-execution-input-r1. Already examined same seed/own fixed
policy panel, no fresh Validation. No training fit or held-data model selection:
EACHwindow's coefficients use its own actual nextobject outcome deliberately
as an ORACLE to characterize representational capacity. The whole-episode
split is inherited for identity, not protection against this intentional
oracle access. Retained causal actuator was fit on disjoint train episodes.

Keep original64canonical object points, native URDF/18realizedjoint order and
fixed handactor base. Float64world poses/points from recorded root quaternions.
Anchor A is previous-point-motion persistence in meters. Two state-only
endpoints: zero motion and rigid extrapolation using nextroot translation
2*t_current-t_previous, nextrotation
R_current*R_previous^T*R_current (constant world rotation increment).

Thirteen hand endpoints correspond to ALLvisual-owned linkIDs
0,1,2,3,4,6,7,9,10,12,13,15,16, checked against inherited sampler and URDF.
For each link, DeltaH=H_predicted_next*inverse(H_current) transports ALL64
current object points rigidly. The causal nextq comes from the unchanged
train-only command/currentq/dq actuator. A second diagnostic family uses
measurednextq, with all other choices identical; no causal-mode futureq input.
The current handbase is fixed and never uses a future handbase pose.

State-only oracle minimizes point EPE over A+w*(E-A), w in[0,1], with either
state endpoint. Causal/measured-hand oracle each includes those same two state
endpoints plus the13hand endpoints, choosing one endpoint and one scalar
perwindow. This is a union of15segments, NOT the whole multi-contact convex
hull. Single hand endpoints are rigid; blended point flows are mixture MEANS
and need not form a rigid pose. Unrestricted endpoint choice is a permissive
capacity screen, not physical contact feasibility or identified intervention.
Neither the winner nor outcome-fitted w is a deployable representation.

Each segment minimizes smoothed mean Euclidean point error, sqrt(r^2+eps^2),
eps=1e-9m, by52float64derivative bisections on[0,1] plus exact endpoint signs.
True EPE difference from smoothing<=1e-9m. Select segment by true EPE.
Total196608solvedsegments, no neural optimizer. Save all exact float64field
banks/targets/anchor, all scalar coefficients/errors/winners/predictions.

## Fixed primary and gates

Primary equal-weight episode-mean64-query EPE/mm on all6144held windows.
Compare full causal oracle to equally outcome-fitted STATE-only oracle,
not only uncalibrated persistence; this avoids crediting arbitrary inertia
attenuation as hand information.

1. Causal EPE<=90%state-only oracle EPE.
2. Same<=90%comparison in inherited current64query near subset (<2cm).
3. Causal EPE<=110%measured-hand oracle EPE (execution gap).

PROMISING iff allthree; UNCLEAR iff either action-capacity gate passes but
not all; otherwise UNPROMISING. Labels concern capacity ONLY. No subgroup
override. Near is unsigned geometric proximity, not measured contact.
Diagnostics: all motion/actor arms, near/far with equal-weight available
episodes, rawzero/persistence/rigidinertia baselines, winner counts/scalar
quantiles. No topology/endpoint/seed/density scan or favorable-subset rescue.

## Engineering/audit/resources fixed before execution

Four-source-state CPU smoke avoids GPU startup: independently SciPy
reconstruct hand endpoints, max<2e-6m; perturb future labels/nextq and verify
causal candidates invariant; analytic scalar cases0/.25/1/clamp1.5 pass.
Main batched FK/field evaluation/scalar optimization uses freshly idleGPU,
prefer6; no neural training and no new native ticks. CPU independent
raw/source/statistical/geometric/solver audit.

ALL6144raw rows/timing, retained causaljoint predictions and state-only
pose/point fields independently reconstruct. Independent SciPy URDF13link
fields for FIRSTfixedwindow of every384heldepisode, plus five SDKbody
origins; remaining5760hand geometry rows not independently reconstructed.
Tolerance2e-6m transport; SDK2e-4m; statefields1e-12m. Existing full native
source/geometry audits inherited by SHA, not claimed repeated.
All196608segments receive independent NumPy convex supporting-line gap
certificates (<1e-7m including smoothingbound), coefficient constraints,
prediction/error/winner reconstruction and superset-state check. Independent
SciPy bounded scalar re-solves ALL32segments in each384selected row (12288
solves), objective discrepancy<1e-7m. All parent/subgroup/baseline/gate metrics
recompute<1e-9mm; rawpersistence versus inheritedfloat32metric within1e-3mm.

Unique terminal run and fixed code commit/card/input/URDF/source SHA guards.
OneidleGPU,<=300s/512MiB newrun, within300GB overall/3Oct23:59Beijing.
Stop OWN child only on budget/input drift/contention/nonfinite/failedaudit.
Keep failed evidence and do not replay on an observation timeout. Archive
all completed field/coefficient/prediction/audit artifacts and code bundle.

After PROMISING: separately prospectively design learned causal coupling and
state-only/shuffled controls; eventual fresh randomized native qualification
and matched policy-training utility remain required. Other labels close or
review this family without coefficient/endpoint rescue. A negative result is
not a universal contact-law/physics/Cm refutation; no novelty/journal claim.
