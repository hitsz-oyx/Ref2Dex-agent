---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-video-object-goal
experiment_id: P-20261013-goal-point-wam-video-object-goal
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-object-motion-goal
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: run a bounded matched object-motion-conditioned video actor probe on visually admitted object tracks
decision_changed_if_negative: repair target identity and tracking before training goal-conditioned video actors
status: RUNNING
run_id: video-object-goal-audit-20261013-r1
---

# Video object-motion goal admission

Decision: can the existing video point corpus supply a meaningful target-object
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
