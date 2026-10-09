---
schema: ref2dex.probe.v2
probe_id: P-20261009-object-relative-gt-servo
experiment_id: P-20261009-object-relative-gt-servo
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 46b06cf
claim_id: C3
hypothesis_family: HF-object-relative-gt-servo
probe_index_in_family: 1
seed_pool: probe
seeds: [282]
decision_changed_if_positive: retain object-conditioned geometry targets and independently confirm the retarget/control gate
decision_changed_if_negative: distinguish contact preload and motion cancellation before learning a new retargeter
status: UNCLEAR
run_id: object-relative-gt-servo-20261009-r1
---

# Does transporting GT wrist targets with the live object preserve grasp?

Result: UNCLEAR overall: bounded source-command transport and post-grasp finger geometry are PROMISING; cold-start geometry control remains blocked.
Decision: Follow user ref7_1 for a two-hour autonomous investigation; retain paused ACT/G/H-to-V/neural-retargeter training.

## Motivation and Decision Note

User ref7_1 identifies that ref7's recorded-command servo is fixed-source
world replay, not a geometry inverse. Its hand RMSE4.35mm/held117 versus
teacher484 cannot refute geometry actions. This Decision Probe tests whether
live-object transport can repair contact-relative drift. Cheapest order:
existing tick140..200 packet audit, then analytic target transport in one
fresh four-role native launch. No new policy/model or fork panel.

Hypotheses: (1) source-world targets fail to follow live object drift;
(2) measured q replay loses commanded contact preload; (3) residual contact
sensitivity persists despite relative correction. Report negative signals
without closing the core geometry/Cm hypothesis.

Wall budget authorized by user: continue from2026-10-09 10:24:16UTC to at
least12:24:16UTC, respecting GPU <=4 and total outputs300GB. Default one idle
GPU2; initial <=8 bounded native launches <=150s/worker plus repairs;
offline GPU geometry audits <=60s each, new outputs <=2GB. Adapt the next
cheapest discriminating probe after results in this card; no budget spent
solely to fill time. Do not touch unrelated AGENTS.md user modifications.

## Initial frozen protocol

Source: held-out seed282 reactive_teacher in
`act-receding1-20261009-r1/act.pkl`, SHA
`f356a9fe79a8431b829d67e5529c187cb1005bad6bf89e81d0ecf58d4b0f2f53`.
Same frozen self-trained e260 actor,4env/64copies,542controls/native GPU
PhysX, unchanged episode outcome. This source has sustained hold, not full
placement success. Privileged source geometry/q/actions are oracle-only.

Four roles: live teacher; fixed-world recorded-command servo (matched
baseline); live-object-transported recorded-command servo; live-object-
transported measured next-q servo. Transport D(t)=Tobj_live(t) inv(Tobj_source(t))
and desired wrist'=D(t) wrist_source_target, fingers unchanged. Current-t
object anchor preserves the teacher's next-step commanded/measured motion.
Using source object at t+1 instead cancels that object motion; retain as a
separate diagnostic if the first test needs it. Rotation uses native
intrinsic XYZ and nearest Euler branches. Native PD adapter reads current q
each tick. Check actual native targets and clips; no neural model involved.

Primary screen: transported-command held >=90%live teacher (teacher>=45),
full542-step and no uncontrolled loss after stable grasp. Object-frame hand
coordinate RMSE<40mm is a secondary geometric screen, not proof of physical
contact. Measured-next-q arm discriminates loss of preload. Negative runs
trigger targeted drift/contact/timing diagnosis, not H-to-V training.

## Existing-packet audit

Compare source/live teacher/world-GT servo at140..200: world and object-local
11-point error, object pose/velocity, per-finger native net force, q/commanded
targets and sampled surface gap. Net body force is not a labeled hand-object
contact pair; sampled mesh gap is an approximate visual-surface proxy. If
finger-specific historical surface pose cannot be recovered reliably, state
that limit and add actual rigid-link recording for subsequent runs.

## Limitations / future evidence

One source and correlated role layouts, unexposed PhysX solver/cache state;
oracle feedback every tick is not a deployable24-step geometry policy.
No formal representation sufficiency/necessity claim from this Probe.
No complete placing success source. Frozen source-time replay can retain
oracle motion despite object correction; explicitly audit anchor/time usage.

## Initial results and adaptive decision

Historical audit `object-relative-contact-20261009-r1` matches ref7_1's
prediction: GT replay t160..176 world-hand RMSE7.60mm but object-local65.0mm,
object position difference52.7mm and angle difference1.49rad; after176 local
error461.8mm and aggregate finger net force nearly zero. The proxy cannot
name a unique hand-object pair loss: multiple fingers decline around168..177,
little-finger signal diverged earlier. This is drift correlation, not causality.

First intervention r1 at46b06cf, current-object-t anchor: full542/70.77s,
teacher481; matched world command484/6.06mmworld/33.20mmlocal; full SE(3)
relative command105 with one loss, unstable wrist/object motion (peak
object lift6.63m/tick222, firstclip167,782clipped coords); measured-nextq38.
All full-task success=false. World replay can hold in another role/launch;
the old held117 is not a repeatable fixed-source failure or matched causal
comparison with this run. Never promote fixed-world-frame error to the sole
explanation. Actual wrist q_xyz vs measured hand-base point maxerror5.96e-7m.

Decision at10:34UTC: before changing representation or learning, distinguish
recursive motion amplification from contact insufficiency. Run r2 with the
already-implemented source-object t+1 anchor, changing only source anchor
time. It cancels measured source object motion and preserves only commanded
preload in the command arm; measured-q arm has no global object progression.
If runaway is reduced but lift disappears, geometry action must specify
desired object motion plus relative hand shape, rather than silently erase
transport. If still unstable, next isolate object rotation feedback and
bounded/slower corrections. This is the second of <=8 initial launches;
same one GPU/resource boundaries. No training.

r2, future anchor, full542/70.08s: teacher14 (behavior control failed; invalid
for gate comparison), world command483; transported command104/one loss,
transported measured125/maxlift.155m. Relative next-q tracks local geometry
24.38mm while world geometry250.50mm; motion cancellation remains relevant.
Do not treat teacher failure or source-time change as a formal method negative.
First teacher drift from r1 occurs at force tick44, history45, q46; no initial
state/control difference or unrequested control intervention found.

Decision at10:40UTC: isolate full-rotation feedback and its unbounded
amplitude. r3 layout transport_ablation uses teacher/world GT command /
translation-only GT command / bounded SE(3) GT command. Keep current-t
anchor and source commanded finger preload. Bound the latter's wrist
correction to20mm and.15rad about the world nominal; preserve nominal future
transport motion. Translation-only remains unbounded as a diagnostic. This
tests whether recursive full-pose correction, rather than relative geometry
itself, explains the runaway. No parameter sweep or network training.

Independent r1 matrix/native audit PASS, max transform errors9.69e-7/5.98e-7;
source FK vs measured11 points max3.68e-7m. Reconstructed historical link
poses are therefore usable for sampled visual-surface gaps. Native finger
net force cannot name an object-contact loss: source pinky has positive net
force despite ~17mm average object visual-surface gap in140..200; self-contact
or mesh/collision differences can contribute. Retain this qualification.

r3 at8a76649, transport_ablation/current-t: teacher484/world484;
translation-only483/localRMSE36.17mm but peaklift2.244m versus source.827m;
boundedSE3 held484/local30.46mm/world10.82mm/peaklift.876m, zero clips, zero
intermediate loss. This bounded recorded-command control screen is PROMISING,
not proof that11 measured keypoints alone specify commands or that relative
correction improves teacher/world performance (both already near ceiling).

Decision at10:44UTC: test query-anchored24-step trajectory transport. At each
query q=0,24,..., use live object(q) and source object(q) to transform all
recorded targets q..q+23, freeze the transformed target chunk, then native
mechanical-q adapter each step. This preserves desired future object/wrist
motion and avoids per-frame phase-locking to source/live object rotation.
r4 roles teacher/world/query24 full SE3/query24 bounded SE3; current-t
anchor, same finger commanded preload, no model. Primary gate remains
held>=90%teacher, no intermediate loss; localRMSE reported separately.
Do not transition to a learned retargeter merely because a bounded oracle
that uses source commands holds. First isolate geometry inverse and preload.

r4 atca555c3: teacher243/one loss, world484, query24-full365/no counted loss,
query24-bounded271/one loss. Low-rate anchoring reduced runaway versusr1
but neither reaches>=90%source484. Old recorded gate uses only live teacher,
so query-full is marked passing its original relative floor219; this is not
accepted as the user-requested near484 upper bound. Preserve that raw gate.

Decision at10:48UTC: strengthen follow-up confirmation floor to90% of
max(live teacher,source teacher), without rewriting historical packet gates.
Teacher>=45 alone must not allow a weak live rollout to lower required
holding far below430. r5 repeats r3's bounded-every-step layout unchanged,
testing whether its positive signal recurs. Data-only gate/metadata edits
also correctly identify24-step target freezing and saved20mm/.15rad caps.
Then isolate measured finger shape vs PD preload; no model training yet.

r5 at225c33a confirms bounded per-step control: held483/local31.40mm,
zero clips/loss, maxlift.851m; world484. Teacher305 is a rare complete task
success under the unchanged recoverable-hold/controlled-place/settle metric,
so retain its full q/dq/geometry packet as a second, placement-capable source
for later oracle transfer checks; independently inspect before relying on it.
Two fresh bounded signals are PROMISING for the command-based controller,
not a multi-seed efficacy claim or proof of keypoint-only sufficiency.

Decision at10:56UTC: distinguish measured geometry from commanded effort
without training R. Frozen train3-only preload statistics use lifted near
object unsupported samples; no test-source calibration. Median active-finger
PD-target-minus-next-measured-q is [.22347,.05182,-.00051,-.00421,.00298,-.03126]
rad. Wrist residual slopes versus next measured velocity are.0898..1005s,
near native damping/stiffness .1s. Source-specific dynamics remain imperfect.
r6 finger_preload keeps the exact source wrist commands and compares source
measured next-finger q without/with fixed train-median preload; teacher/world
command controls unchanged. This isolates preload from wrist/object transport.
If preload is useful, r7 geometry_pd will test measured future wrist q plus
.1s forward-difference velocity compensation, with source commanded fingers
versus measured fingers+fixed preload. These are analytic privileged controls;
future q is a stronger oracle than11 hand points until inverse is checked.
No neural training or H-to-V transition follows solely from command-based hold.


r6 at9030005: teacher483/world484, measured fingers no preload36 and fixed
train-median preload0. World hand RMSE4.38/5.52mm remains small despite
failed grasp. Independent audit PASS; no/preload clipped coordinates3/0.
A constant lifted-stage preload applied throughout approach is not an
effective controller; do not infer contact force is intrinsically unknowable.

r7 atc95eda8: teacher484/world83 (4.19mm world hand error), measured future
wrist plus .1s forward-velocity compensation with source fingers0 and with
fixed preload0. Both candidate native targets are exact within1.2e-7 and
zero clips; this simple analytic dynamic inverse is UNPROMISING. The world
command control itself varies strongly across fresh contact simulations.
No formal sufficiency/necessity or same-state causal claim follows.

Decision at11:05UTC: before spending another native launch or training, use
a bounded GPU kinematic audit to distinguish inverse observability from
control effort. Compute the11-point Jacobian for full18 measured joints and
for12 coupled control coordinates; compare measured coupling deviations and
the best coupled pose fit. This cheapest offline diagnostic changes whether
to implement a geometry inverse or contact/dynamic state augmentation.
Allow <=120s GPU audit / <100MB fresh outputs. No additional neural training.


GPU geometry audit r1 at6bd097c took11.16s. FKmaxerror8.34e-7m; sampled
full18 Jacobian rank17 vs coupled12 rank12 at all9 inspected poses. Actual
dependent-joint coupling residualRMSE [.288,.124,.053,.093,.297,.058]rad;
truth-initialized/fixed-real-wrist coupled local fit reduces11-point
coordinateRMSE3.36 to1.08mm but thumb tip point-distanceRMSE5.07mm and
index3.01mm. This favorable local fit is not a globally optimal or deployable
inverse, and geometric observability does not determine contact effort.

Decision at11:16UTC: r8 finger_preload_late freezes original source commanded
targets through tick119, switches only finger targets at120 to measured
next-q versus next-q+fixed train-median preload, and retains source wrist
commands. This isolates sustained-grasp maintenance from premature closure
during approach in r6. Report each candidate's physical state at the switch;
without a stable incoming grasp, its maintenance result is not interpretable.
This is the eighth bounded launch; no model training or global claim change.

Relevant primary-source boundary: [adaptive feedforward/impedance grasping]
(https://arxiv.org/html/2107.08996v2) treats mapped joint geometry separately
from adaptive control using joint tracking errors (not tactile sensing in
that setup). [Contact transfer across dexterous hands]
(https://arxiv.org/html/2606.15516v1) uses calibrated torque/contact information
with pose and force-limited compliance. Neither paper proves effectiveness
for this IsaacGym/controller setup; native net-body-force norm is not the
calibrated pair contact/joint effort those methods require.


r8 at9ae83f3, full542/71.67s: teacher485/world484, late measured fingers
without/with median preload both483, zero clips/loss. At switch120 both
candidates are lifted .237/.241m with1.15mm surface gap and unsupported.
Thus r6's failure cannot justify a general claim that measured finger
geometry cannot maintain an established grasp. Its cold-start/approach
controller differs materially from grasp-maintenance control. Audit PASS.

Decision at11:22UTC: extend exploratory budget from8 to <=12 native launches
(<=150s each, one idle GPU, same2GB output/global bounds), within the user's
two-hour authorization. New positive maintenance signal changes the next
decision: r9 wrist_geometry_late retains exact source commands before120
and source commanded fingers throughout; after120 compare11-point analytic
wrist inverse without/with .1s velocity compensation. Calibrate fixed root
geometry only at the known reset, recover later wrist transforms with
Kabsch from palm/five root points and previous reconstructed Euler branch.
This tests if initial contact establishment, rather than steady tracking,
explains the wrist negative. If positive, combine post-grasp geometry; if
negative, keep wrist dynamics as blocker. No future joint-state read for
the reconstructed wrist, but privileged future geometry/source fingers remain.
Stop these local launches after12 or if no discriminating hypothesis remains.


r9 atba32296, full542/61.07s: teacher292/world483, post-grasp11-point
wrist inverse without/with velocity compensation both483, zero clips/loss.
World hand coordinateRMSE20.88/5.14mm, object-local46.79/32.54mm; compensation
improves tracking here without being necessary for holding. Independent
wrist reconstruction vs source q: maxposition5.96e-7m, matrix1.97e-6;
future wrist targets read11-point geometry only after reset calibration.

Decision at11:26UTC: combine post-grasp wrist and fingers, and distinguish
privileged measured finger q from a true11-point coupled inverse. Build
a GPU inverse artifact using only future geometry, static FK and reset q.
Two fixed reset/midrange starts with300 iterations, <=110s and<100MB; select
by geometry error only, never native outcome/future-q labels. r10 will
retain source commands before120, compare future measured finger q with
this geometry-only inverse; both use11-point wrist+.1s velocity feedforward
after120. If the combo holds, investigate cold contact establishment next;
if not, identify whether combination or coupled geometric projection breaks
the independently positive partial controllers. No old R training.


Geometry-only inverse artifact at e50762c takes12.31s onGPU2 and reaches
1.078mm coordinateRMSE (thumb tip5.071mm), matching the favorable truth-init
fit without any future q/action labels. The static coupled-pose mismatch
therefore persists beyond initialization. Artifact and runtime provenance
are source/hash/reset/geometry checked. r10 uses the frozen artifact; its
geometry arm reads no source finger actions/q after the known grasp prefix.


r10 at16e2c9d, full542/74.76s: teacher484/world484, combined measured
geometry124/one loss; true11-point inverse250. Both world hand errors
~5.1..5.3mm despite eventual loss. Independent partial maintenance positives
do not imply joint wrist/finger sufficiency; do not declare cold contact
establishment the sole blocker. Retain raw packet and inspect local drift.

Decision at11:36UTC: r11 geometry_inverse_relative_late compares the same
full11-point inverse/world nominal with bounded live-object wrist correction
(20mm/.15rad, fixed prior caps). Both retain the known command prefix before120;
only wrist object feedback differs thereafter, finger geometry is identical.
This tests the combined controller's relative-drift explanation and the
ref7_1 intervention without giving it source finger commands after120.
If positive, use r12 to check cold-start relative inverse; otherwise audit
load/geometry mismatch rather than train old R. Same GPU/time/output bounds.


r11 at0ab1100, full542/70.53s: teacher484/world484; world11-point inverse
483/local32.25mm, bounded object-relative inverse181/local226.21mm. Thus
relative feedback is not a universal remedy for geometry-only control. Same
world inverse held250 in r10 role3 but483 here role2; role-specific physical
state/contact dynamics complicate a direct causal explanation. Both are
post-grasp oracles with a120step recorded-command bootstrap, not cold policies.

Decision at11:40UTC: r12 geometry_inverse_repeat puts the identical world
11-point inverse into both roles2/3 after120, and checks their absolute PD
target streams bitwise. This discriminates observable role/contact
sensitivity from an unintended arm-implementation difference, with no
additional model or target variation. Record state/force divergence before
the switch and actual control conversion. These are replicas, not two seeds.
If both hold, geometry maintenance is PROMISING but cold contact remains
blocked; if divergent, retain UNCLEAR and localize the physical divergence.


r12 atecaa368, full542/71.40s: teacher353/world242; identical inverse
absolute target streams give held483 vs250, local32.65 vs220.28mm. Target
streams bitwise equal, q position diverges around45/native force44, before
switch120. Tiny velocity/object rounding differences start earlier. Role3
repeats r10 role3 metrics exactly. Physical grasp state, not two different
inverse implementations, explains the replica difference; strict same-state
causal comparison remains unavailable. Conditional geometry maintenance
has repeated positive signal in role2, overall robustness remains UNCLEAR.

Decision at11:46UTC: retain the role/contact sensitivity rather than expand
old neural R. Extend budget to<=16 native launches (<=150s/worker), within
remaining user2h and unchanged oneGPU/2GB bounds. r13 tests causal command
anchoring: at120 take own actual last PD target u119, subtract the geometry
equilibrium g120 (wrist includes .1s velocity), then add this fixed offset
to subsequent g(t+1). Roles compare absolute geometry inverse versus
geometry increments anchored to that already-applied command. This retains
contact preload/pose offset without reading future PD target or force labels.
It is a120step bootstrap oracle, not cold geometry sufficiency or a calibrated
force controller. If positive, test cold-start dynamics; if negative, inspect
which anchored target/clip breaks contact and stop this adaptation.


r13 atd999ca7, full542/70.19s: teacher484/world31; absolute inverse483 and
command-anchored inverse483, zero clips. The retained offset includes finger
index+.169/middle+.217/thumb pitch-.215rad and wrist millimetre offsets.
Independent before-switch targets match r12, but observed physical state
DOF/objects differs from tick53 (force44), so the improvement over old role3
250 is a PROMISING signal, not a matched causal proof of preload retention.

Decision at11:49UTC: cold-start geometry dynamics remains the immediate
blocker. r14 replaces equilibrium/forward-velocity heuristics with a small
identified one-step PD inverse u=q+a*(qdesired_next-q)+b*dq using live q/dq.
Fit2 coefficients per12 independent joints from the existing3train launches'
first40 approach frames only (small CPU statistics, no neural model).
Both arms use the11-point wrist inverse from tick0; role2 keeps source
commanded fingers to isolate wrist dynamics, role3 also uses geometry-only
fingers and the calibrated inverse. No source future q/dq or wrist commands
enter either candidate wrist; role3 has no recorded command bootstrap.
If wrist-only holds, target remaining finger/contact load; if both fail,
inspect contact-stage model error before adding feedback. <=150s/worker.


r14 atbcec32d, full542/71.34s: teacher17 (behavior gate invalid), world484;
cold geometry wrist+source fingers483/world2.91mm/local24.88mm, full cold
11-point geometry PD inverse485/world2.42mm/local48.96mm/no loss. The
full arm has161 clips: index4, thumb yaw8, thumb pitch149; it is explicitly
a saturated native-envelope controller, not exact equilibrium matching.
No source commands/bootstrap or future joint labels enter that full arm.
Retain raw gate=false because live teacher<45, do not retrospectively
change the screen. This cold-start signal is PROMISING pending confirmation.

Decision at11:58UTC: r15 repeats exactly the r14 cold dynamics protocol and
frozen coefficients/artifact. Recheck teacher/world behavior and positive
full-geometry holding; add a dependency test exercising the actual control
branch with poisoned future q and altered future command labels. The
controller must ignore these labels while remaining sensitive to geometry
and live mechanical velocity. If confirmed, use the last16th launch to
transfer the same controller to the independently retained placing source
from r5; no parameter/label changes or old-neural R training.


r15 at31906a7, full542/72.50s: teacher355/world484; cold wrist geometry
with source fingers483, full11-point inverse238/world4.01mm/local222.08mm,
308 finger clips. The full arm does not confirm r14; raw gate=false. Wrist
dynamics is PROMISING over two launches, full coupled geometry controller
robustness remains UNCLEAR. Actual-branch dependency test passes: future
q labels can be NaN and source command labels altered without changing the
full geometry arm; changing geometry/live velocity changes control.

Decision at12:08UTC: do not transfer to placement or train V/R yet, since
full geometry confirmation failed. Final r16 isolates finger geometric
projection from dynamic contact compensation. Both arms use the confirmed
11-point wrist+identified dynamics; replace fingers with privileged measured
source next-q, compare no load memory versus causal EMA of own previous
PDtarget-currentq-.1*dq. EMA alpha=dt/(D/K)=1/3, bounded by train-only lifted
p10/p90 residuals enlarged only to include reset zero. This uses future
joint labels and is explicitly NOT11-point geometry-only or deployable.
No fresh training/calibration or future force/PD command label is used.
If measured-q solves it, state-conditioned loaded inverse is the next
blocker; if only memory helps, causal effort retention matters; if neither
holds, defer new learning and redesign physical tracking/contact control.
This completes the <=16 launch budget; remaining time is for independent
execution/label audit, evidence synthesis and recording the next decision.
