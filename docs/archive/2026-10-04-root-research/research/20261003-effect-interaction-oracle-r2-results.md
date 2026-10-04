# Qualified fixed-background oracle r2: no incremental utility

P-20261003-effect-interaction-oracle-r2 COMPLETED / UNPROMISING. Collector
source443d844; independent geometry/protocol audit at a0e800b. Fresh seed763,
12designated subjects, each in its own12-instance world with11P0 background
controllers.84nonzero short68tick queries and26unique selected full202tick
executions. Frozen r1TRAIN-only Qs/normalizers;0new optimization,6000inherited.
No Cm, actor/PPO updates or official actor weights.

| Selector | Full105 successes /12 | Motion0/1/2 |
| --- | ---: | --- |
| P0 | 5 | 0/4/1 |
| state | 8 | 0/4/4 |
| effect | 8 | 0/4/4 |
| interaction | 8 | 0/4/4 |
| joint | 8 | 0/4/4 |

Joint-minus-state and joint-minus-effect both0pp, failing both fixed5pp
gates; joint>=P0 passes. This is a qualified negative for the fixed finite
options/frozen-selector/protocol recipe, not a universal Cm or contact-prior
refutation. All8subjects in motions1/2 are already solved by state, effect,
interaction and joint; headroom can only come from the4motion0subjects.

ALL26selected whole-world first68control state/force/PD arrays are exact;
ALL136physics frames (both updates/control tick) also have maxerror0. Selected
descriptor error0. Independent NumPy Q probability error<=7.16e-7 and all
original probability argmax choices agree.111native panels pass independent
fullmesh/PD/contact coverage/sign/unit checks. ALLactual full105episode labels
and aggregate counts agree with independent SciPy mesh evaluation. Independent
SciPy/raw-quaternion feature computation covers all84nonzero and all12zero
candidates: maxscaled error9.1553e-5<=1e-4. Initial audit normalized SDK
quaternions implicitly and failed at1.0681e-4; preserved diagnosis and
raw-quaternion algebra correction fix the audit without changing science
inputs, learned models or tolerances.

Separate numeric audit finds a real ranking issue: float32 sigmoid saturates
distinct logits to exactly1.0. I/joint each choose option0 for subjects1/7
while raw-logit argmax chooses3/4 respectively. Example I subject1 logits
17.1213vs21.9057 both become1.0. Three subjects have maximum-probability ties;
two choices change. This is ordering loss, not failed gradients or a contact
frame bug. Do not retroactively alter r2 choices/results. Fix ranking using
raw logits (mathematically monotone-equivalent), verify CUDA/NumPy ordering,
then execute only the2new choices; no refit or repeated queries. Because
both affected subjects already succeed, this bug cannot explain the missing
positive joint-vs-state gate in r2.

Next decision: finite-candidate capacity on the4unsolved subjects. Each has
actual failed executions for options3/6/7 and P0/zero; only1/2/4/5 remain.
Collect16full executions after selector choices are locked. Retrospective
best-candidate labels are a capacity diagnostic, not deployed oracle utility.
A zero headroom result closes this control bank before more Cm fitting; a
positive result licenses a matched oracle-information/learning diagnostic.

Limits: one scene seed/model seed, one object, three synthetic references,
rich compressed normal-contact moments rather than full friction/manifold.
Frozen training used96-instance global-option worlds;12-instance fixed
background is a shared scene/behavior shift. No mathematical full-information
upper bound, RL-training gain, formal Validation or journal-ready claim.

Collection wall 2861.250s; parent terminal bytes 2511405778 before final artifact audit.
CPU independent final audit 27.147s; owned run/audit PIDs are terminal.
Raw source: `src/task/CmResidual/research/contact_response/output/
P-20261003-effect-interaction-oracle-r2/`.
