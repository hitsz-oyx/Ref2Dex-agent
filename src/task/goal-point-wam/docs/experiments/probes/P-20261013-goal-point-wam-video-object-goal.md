---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-video-object-goal
experiment_id: P-20261013-goal-point-wam-video-object-goal
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 3ac69607
claim_id: C3
hypothesis_family: HF-goal-point-wam-object-motion-goal
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: run a bounded matched object-motion-conditioned video actor probe on visually admitted object tracks
decision_changed_if_negative: repair target identity and tracking before training goal-conditioned video actors
status: UNPROMISING
run_id: video-object-goal-audit-20261013-r1
---

# Video object-motion goal admission

Research question: can the existing video point corpus supply a meaningful target-object
motion goal without using hand futures as input or granting desired motion to
the action-conditioned World? User explicitly chooses geometric object goals
instead of language and authorizes continued bounded exploration. This remains
within the goal-point-wam subproblem and the unchanged C3/Cm mission.

Cheapest first step is a CPU-only label/coordinate/visual audit, not a new
reconstruction or large fit. Existing303 TRAIN/86 held-window corpus is read-only;
remote training was user-stopped and must stay stopped. Existing download and
reconstruction processes are independent and untouched. NAS metadata/teacher
permission failures are recorded and not changed; use readable qualified arrays
and original camera/RGB to independently project current points where possible.

## Scope and fixed admission criteria

- Read at most24 windows from distinct raw recordings, chosen deterministically
  by recording and source clock before future-motion inspection. Keep original
  session train/held assignments. Download <=100MB of compact arrays/current
  RGB and provenance to this run's local directory; no full video/depth copy.
- Target selection uses current RGB only, as a declared manual annotation from
  a supplied demonstration. Record visible-surface polygon and object/part name.
  Do not choose the object or select points based on future displacement.
- Select up to32 spatially spread CURRENT target points, retaining identities
  and observed positions. Pack desired positions at+8/+16/+24, explicit times
  and per-point/per-horizon validity. Never fill missing desired points with
  invented motion; no future hand or robot command in the goal packet.
- Reprojection agreement and current-image surface purity must be inspected.
  A coarse polygon crossing the hand/background is not an object mask.
- Initial rigid-object admission: >=16 selected current points, >=80% valid
  pairs at each requested horizon, raw rigid-fit residual median<=5mm and
  p95<=15mm at each horizon. Rigid fit is a diagnostic; never overwrite raw
  targets with fitted ones. Deformable objects are deferred for this first probe.
  No selection of only future-visible points to make the current object mask.
- An audit pass only admits semantic/geometry input, not physical goal
  feasibility or policy success. Need >=3 TRAIN sessions and>=2 held sessions
  with visually confirmed targets before a scientific short-training probe;
  otherwise only engineering interface verification is justified.
- World input and labels remain unchanged and goal-blind. Goal fields belong
  only to the actor; test mutation isolation. Temporal goals must preserve
  time order and masks; a null goal is different from an explicitly static one.

## Resources and stop

CPU file arithmetic/projection/plots <=15min, local artifacts<=200MB, no remote
writes. No neural inference in this stage. Shared local outputs+tmp<400GB;
remote full-corpus storage has its separate existing authorization. Any later
model probe needs its own recorded fixed budget and idle GPU check. Stop
admission on target ambiguity, bad correspondence, insufficient held coverage,
missing provenance, or scope/resource violations; retain all excluded examples.

## Limitations / future evidence

Manual polygons on a small cohort do not solve automatic target perception.
Future video object tracks are hindsight supervised goals; deployment requires
user/reference goals in the same coordinate contract. The same initial scene
with multiple physically executable goals and actual robot success are later
Validation needs. Label filtering here is target-goal admission, not destructive
cleaning of the original World corpus. No new actor/R architecture is selected.


## Completed result — 2026-10-13

Result: 22 windows audited;12 current polygons;3 TRAIN/0 held numerical passes, all near-static supports.
Decision: Do not train; specify manipulated parts and acquire denser target tracks before matched actor probes.

Export code commit `1d8be444`; packet/audit code `3ac69607`. No RNG, no model
training, no GPU, no remote writes. Export took30.7s and24.53MB; all local audit
artifacts approximately54MB. File/statistical computation used CPU; no neural
inference was involved. Four focused packet contract tests passed. This stage
only implements the data packet/audit, **not a new actor temporal encoder**.
World model code and its input contract were not changed; packet construction
reads no hand and owns independent storage, checked by mutation tests.

The qualified source has22 distinct recordings (r02/r06 absent), not24. All22
clock-median windows were retrieved, without export failures. Current-only
inspection fixed12 conservative visible-surface polygons before looking at
future frames or displacement; six deformable and four ambiguous/occluded
assemblies were deferred. These are provisional assistant visual annotations,
not independently confirmed segmentation ground truth.

| Outcome | Count / details |
|---|---|
| Numerical admission | 3 TRAIN windows/sessions; **0 held** |
| Passed targets | r03 desk base24pts; r08 table base24pts; r09 table base20pts |
| Passed target +24 median displacement | 2.69 /3.37 /1.13mm respectively |
| Other annotated targets | All9 have only2–12 current points; several also lose visibility |
| Moving drawer r05 | 6 current points ->6/4/3 valid; +24 displacement34.36mm on surviving3 |
| Lamp bulb r07, held | 5 current ->3/2/3 valid |
| Yellow bottle r12 | 4 current ->2/1/0 valid despite visible bottle transport |

All12 overlays were inspected. The three numerical passes track their stated
base surface reasonably, but the hands manipulate independently moving legs.
They provide weak/near-static support constraints, not the intended moving-part
trajectory. r13's bowl is static context while the blue item is manipulated;
current intent was already marked ambiguous and is rejected. r22's conservative
polygon still touches fingers at its right edge and is rejected for purity.
Do not enlarge masks across background/hands or replace raw tracks with rigid
fits merely to pass. No threshold was relaxed after inspecting results.

**Local UNPROMISING:** directly extracting dense-enough object motion goals from
this existing sparse scene cohort with this recipe. This is not evidence against
object-conditioned policy learning, nor proof the underlying video is unusable.
The16-point/80% thresholds are conservative admission choices, not physical
minimum requirements; narrow polygons contribute to low counts. A target-specific
retracking or a different representation may recover useful supervision.

## Decision Note / next minimal probe

Do not launch goal-conditioned video training on this cohort: held admission
fails and the apparent numerical successes mainly describe static support.
Continue the user-approved non-language interface: **explicit selected object or
movable part + current points + desired points at times + validity**. For furniture,
name and select the manipulated part, not the whole articulated assembly.

Next decision experiment should fix current target-part masks first, reserve
queries on those surfaces and test denser target-specific tracking on a small
cohort with both TRAIN and held sessions. Preserve failed examples and near-static
support cases separately; do not retrospectively infer task identity from the
largest future displacement. Only after useful object-motion labels are available
compare null/static/correct/shuffled goal conditions on the same short actor fit.
Goal remains actor-side; the action-conditioned World cannot observe wishes.
No hand-goal input is required by this packet; current hand/history remain observations.
For a far-hand approach stage, a near-term static object goal may be legitimate
but underdetermines approach intent: provide the later object-motion waypoint or
a suitable longer-horizon plan, rather than falsely labeling it a no-op task.

## Artifacts / reproduction

Run root: `outputs/goal-point-wam/video-object-goal-audit-20261013-r1/`.
`inputs/manifest.json` retains source windows, sessions and camera/RGB provenance;
`annotations.json` fixes current-only polygons; `summary.json` contains numeric
results and `visual_review.json` is the completed separate visual decision.
`goals/*.npz` are **audit-only**, not admitted training data.
`current-only-sheet.jpg` and `overlays/*.jpg` preserve inspection evidence.

```bash
PYTHONPATH=src/task/goal-point-wam/src /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/goal-point-wam/tools/audit/audit_video_object_goals.py \
  --root outputs/goal-point-wam/video-object-goal-audit-20261013-r1
PYTHONPATH=src/task/goal-point-wam/src /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m pytest src/task/goal-point-wam/tests/test_object_motion_goal.py -q
```
