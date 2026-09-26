# P-20260924-object-pool-audit

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe / read-only data audit.

## Question and decision

The present three-object train split is too narrow to judge whether a Cm
learns object-general dynamics. Do existing retargeted `s1_*_lift`
tensors provide at least eight additional *distinct object identities*
with finite data, reference lift >=30 mm, reference contact occupancy
>=5%, and median contact-frame wrist-object distance <=250 mm? If yes,
start a corrected-conversion and simulator-feasibility gate for an expanded
object-disjoint split before another Cm architecture experiment. If no,
prioritize data conversion/quality before model complexity. This scan is
not a physical grasp evaluation; penetration and coordinate alignment
remain unverified. Read-only CPU, <=5 minutes, <1 MB report. Preserve
apple as already-explored diagnostic, not a pristine validation object.

## Result

The first scan of local `inspire_rl_partial_filtered_20260905` found
41 finite S1-lift tensors with mesh and reference lift, but *none* passed
the 250 mm relative wrist/object gate: even its airplane tensor has a
349 mm median contact-frame distance. This is a source-coordinate mismatch,
not evidence that no object data exist; do not use that source directly.

The corresponding read-only, coordinate-aligned external
`inspire_geometric_dexplore` source has 26 S1-lift object identities.
All 26 pass the prespecified finite/contact/lift/wrist-distance gate,
including **22 additional objects** after excluding train3 and explored
apple. The data-availability gate passes. Reports:
`outputs/CmResidual/agent_lift_object_pool_audit/report.json` and
`report_geometric_reference.json`. This only authorizes a small corrected
conversion and simulator-feasibility screen; it does not authorize claims
of physical success, data cleanliness, or Cm generalization.

## First physical gate fixed before execution

Start with `s1_cubesmall_lift` as a new identity. Rebuild the same
raw-contact 30 Hz input and coordinate correction used for train3,
verify a finite [T,598] tensor and corrected contact-frame wrist/object
separation <=250 mm, then evaluate the read-only official actor on 64
first episodes, seed 174, with early termination disabled. This actor is
only a physical data-feasibility diagnostic and will never enter the
self-trained policy or Cm. If >=16/64 achieve held-lift and mean binary
contact >=0.30, mark this sequence suitable for an expanded-train-data
pilot; otherwise mark it difficult/unclear and screen another object.
One idle GPU, <=20 minutes, <=10 MB evaluation output. This gate does
not certify the reference is penetration-free or that our actor can
learn it.

### First physical-gate result

`s1_cubesmall_lift` rebuilt and converted successfully. The corrected
tensor is finite [799,598], contact occupancy 65.1%, reference lift
147 mm, and median contact-frame wrist/object separation 141 mm.
Its relative wrist/object vector matches the aligned geometric source
to floating-point precision (max absolute error <0.001 mm), although
absolute positions and robot DOFs differ because the converter is
reconstructed; no label of exact source reproduction is claimed.
The official diagnostic actor achieved **64/64 held-lift** and mean
binary contact fraction 0.844 on seed 174, clearing the prespecified
16/64 and 0.30 gate. This authorizes adding cubesmall to a future
expanded training-data pilot, not a Cm efficacy claim. Artifacts:
`outputs/CmResidual/agent_objectpool_cubesmall_gate/` and
`outputs/Dexplore/agent_objectpool_cubesmall_official_s174/`.

## Diversity gate fixed before execution

Screen two deliberately different shapes next: `s1_waterbottle_lift`
(elongated, prospective training identity) and `s1_torussmall_lift`
(holed, prospective *untouched* held-out identity). Use the same
conversion invariants and official-actor diagnostic thresholds as
cubesmall. If both clear the physical gate, make an object-disjoint
expanded pilot split with waterbottle and cubesmall in train and
torussmall held out, while keeping apple only as an explored diagnostic.
If torus fails, do not tune a model on it; screen another held-out
identity. If waterbottle fails, retain only cubesmall. At most two
short evaluations and <=100 MB new conversion data each.

### Diversity-gate result and fixed fallback

Both corrected tensors were finite and within the 250 mm wrist/object
gate: waterbottle 142 mm (607 frames, 76.3% reference contact),
torussmall 140 mm (489 frames, 61.1% reference contact). Official
diagnostic actor held-lift was **64/64** for waterbottle (contact 0.877)
but only **6/64** for torussmall (contact 0.668), so waterbottle enters
the prospective training pool while torussmall fails the held-out
physical gate. It remains a difficult stress case, not a Cm tuning or
final-validation object. Physical feasibility differs markedly even
when offline reference contact looks plausible.

Following the prespecified fallback, screen `s1_alarmclock_lift` as
the next held-out identity with the unchanged conversion checks and
official-actor 16/64 held-lift and 0.30 contact thresholds. If this
also fails, preserve an unassigned held-out slot rather than lowering
the gate or selecting on Cm performance.

### Fallback result

Alarmclock converted to a finite [510,598] tensor with corrected median
contact-frame wrist/object separation 152 mm, reference contact 74.1%,
and reference lift 431 mm. The official diagnostic actor achieved
**64/64 held-lift**, mean binary contact 0.895 on seed 174, so it clears
the fixed gate. It is assigned as an *untrained, unevaluated-by-Cm*
held-out identity for the expanded pilot. Train identities are the
existing airplane/mug/toothpaste plus newly screened cubesmall and
waterbottle. Formal cross-object validation will still require more
untouched identities and matched multi-seed controls.

## Held-out baseline difficulty check fixed before execution

Evaluate the existing self-trained airplane e260 actor (no Cm, no new
training) on alarmclock seed 174, 64 first episodes. This is not a
head-to-head Cm comparison. If held-lift is already >=48/64, alarmclock
is a saturated pilot challenge; retain it but screen a second untouched
held-out identity before any Cm utility decision. If below 48/64,
alarmclock can serve as a nontrivial exploratory held-out check.
One idle GPU, <=20 minutes, <=10 MB output.

### Baseline difficulty result

The self-trained airplane e260 checkpoint achieved **0/64 held-lift**
on alarmclock seed 174; mean maximum contact-supported lift was 6.06 mm.
Thus alarmclock is a nontrivial exploratory held-out identity and the
prespecified saturation contingency was not triggered. Official 64/64
versus self-trained 0/64 is an availability gap, not evidence that Cm
can close it. Result:
`outputs/Dexplore/agent_v139_s3_standard_s70_e260/eval_s174_e260_full_pilot_alarmclock/`.
