# Corrected-physics causal execution input qualification

Decision Probe, same MISSION and isolated surface-prior branch. Previous goal
turn: PROGRESS, completed matched granularity matrix and excluded its cheap
remedy. Next blocker is replacing realized future hand input with information
available before acting. Historical9/24gain/velocity probes already existed
but use different policies, physics and old prior weights; this is not their
formal rerun or validation. Use the existing corrected655panel before any
new simulation or large fitting. This is the cheapest discriminating test.

## Question and decision

Distinguish inaccurate command->hand execution from frozen prior's corrected-
physics prediction mismatch. Fit a bounded causal actuator, then pass its hand
motion through the existing frozen MANO7168/Inspire7168priors. If causal hand
prediction and end-to-end information pass, qualify fresh randomized native
data before policy training. If oracle prior itself fails, close frozen direct
transfer; consider a separately designed corrected-physics representation fit.
If oracle passes but bridge fails, first repair the causal input or use direct
state/action. Do not blindly increase pretraining volume/granularity.

## Fixed protocol

Source P-20261003-inspire-filter-impact-r1/s655:768fixed-policy corrected-
physics trajectories/202ticks, previously audited all collision ownership,
states/actions/PD/fullmesh. Split WHOLEenvironment trajectories within each
3motion x4arm group:32of64train and32held, RNG4001,384each. Select16fixed
ticks linspace(1,200),6144windows each. No outcome/contact/success filtering,
neighbor frames never shared across fit and held episodes. Same seed/policies,
previously examined evaluation trajectories, not a fresh unseen-seed test.
This reuse is for execution/prediction qualification only, not a policy claim.

At selected tick t:preq/dq=posttrace[t-1], sent target=trace.target[t], label
nextq=posttrace[t]. Current object=trace[t-1], previous=initialobject if t=1
otherwisetrace[t-2], targetobject=trace[t]. All timestamps audited against
the retained causal context. Actor base fixed, full18native joints used
(realized mimic joint ratios not assumed exact). No nextq/object enters the
causal actuator fit inputs. Held future q is only a kinematic label/oracle.

Perjoint closed ridge predicts Δq from target-minus-currentq, currentdq/30,
and intercept; train-only RMS scales clamp1e-4, ridge1e-6. Full and action-
removed velocity-only use the same3slots/54coefficients, latter gap slot0.
Predicted finger joints projected to fixed URDF native limits; no adaptive
coefficient/model/horizon/seed scanning. Stationary and instantaneous target
are controls. No neural actuator, optimizer or new policy training.

Generate10135proper global area-uniform hand samples on the pinned13visual
meshes, fixedseed2024and correspondence across current/next/predictedposes;
sampling is NOTasserted byte-identical to the external cache. All18joint FK
and fixed actor base produce geometry. Actual hand label is reconstructed
from measured native nextq, not independently measured all-shape surface.
Cross-check all selected held nextstates against the five retained SDK body
origins (<0.2mm maxengineeringbound); report orientation discrepancy too.
Retain local triangles/barycentric/link IDs. This does not prove full surface
fidelity, attributed contact, penetration safety or hardware execution.

Object64canonical points/normals are the original prior's first Inspire-train
airplane record in its current object frame, no held outcome sampling. Actual
native root poses give before/current/next surface points. Current closest4
hand points are selected once perwindow. Predict at same64points using the
UNCHANGED22feature/19651parameter MANO7168andInspire7168checkpoints. Conditions:
stationary hand, instantaneousPDtarget, fittedvelocity-only, fittedaction+
velocity, realizednextq oracle. Geometry/previousobject flow identical, local
and globalhandflow vary. No fine-tuning, normalization or checkpoint selection.

## Prospective gates

Primary hand metric equal-weight episode surface EPE/mm. Execution bridge
passes only if action+velocity <=50%stationary, <=75%velocity-only, and <=5mm.
Perprior oracle information:oracle object EPE <=90%persistence and <=90%
stationary-hand input. Perprior causal information:action+velocity object EPE
<=110%oracle, <=90%persistence, <=90%velocity-only input. Object EPE equal-
weight episode64point flow EPE/mm. PROMISING only if execution gate and BOTH
oracle/causal gates of at least one prior pass. UNCLEAR if execution or any
oracle passes without full gate; otherwise UNPROMISING. Report all conditions,
all gates and per-episode/motion/arm diagnostics; no post-hoc subgroup rescue.
Single reused seed is exploratory, not statistical Validation or equivalence.

Prospective support diagnostic: report current64query minimum hand distance
<2cm near/far subsets (unsigned geometric proximity, not attributed contact).
Old prior train windows had a full4096candidate proximity filter; the native
all-phase primary has no such filtering. This diagnostic can reveal a support
mismatch, but does not rescue failed all-window gates, identify physics alone
or establish complete contact-patch coverage.

## Resource and integrity

900s/1GiB evidence, one freshly idle GPU for fullFK/batchinference/closed
actuatorfit; CPU mesh/file preparation and independent numeric audit. Tiny
engineering FK smoke can use CPU because only four states. Zero new native
ticks/actor calls/optimizer updates; existing trajectories explicitly reuse
previous collection costs. Fixed commit before unique run, source/URDF/mesh/
checkpoint/code SHA protection, owned child only termination on contention or
budget/drift. Retain source selections/rows, coefficients, geometry, hand errors,
features, joint/model predictions, perparent metrics and failures. Independent
audit must rebuild raw rows/split, closed coefficients (tol1e-6), causal-input
isolation, sampled barycentric points/normals, selected geometry/FK/nearest4,
features at the first selected window of EVERYheld episode (384geometry rows;
not a full6144row geometric reconstruction), all model predictions and
metrics/gates. All12288raw selected rows and full6144joint predictions are
checked. Numeric feature bound2e-5/
2e-6, NumPyforward2e-4normalizedflow, metric1e-4mm. No new native control
benefit, policy utility, formal hand morphology or universal failure claim.
