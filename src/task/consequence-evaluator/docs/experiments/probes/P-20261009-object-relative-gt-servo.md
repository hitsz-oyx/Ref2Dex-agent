---
schema: ref2dex.probe.v2
probe_id: P-20261009-object-relative-gt-servo
experiment_id: P-20261009-object-relative-gt-servo
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: pending
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
