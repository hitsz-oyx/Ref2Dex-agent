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

Result: UNCLEAR: diagnostic and execution pending.
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
