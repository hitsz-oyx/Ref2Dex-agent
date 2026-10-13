# Object goals, hand intent and static-object windows

Date:2026-10-13. Scope: a design recommendation grounded in local native-three
data and a cached information Probe. No core-claim change or demonstrated closed-loop improvement.
The initial design analysis is followed below by bounded training evidence.
Evidence/protocol: [curation and goal-information experiment](../experiments/probes/P-20261013-goal-point-wam-ref3-curation-audit.md).

## What the present contract actually specifies

`GoalWindows._object_goal` selects the current anchor flag, uses32 paired
canonical object points, and transforms them with that object's actual
step24 effect. At30Hz this is a0.8s endpoint. The current anchor frame is
shared by start, goal, observed hands and future labels. The object-goal
adapter deliberately removes `goal_hand` and `goal_hand_valid`. The actor
still receives current hand/history through its existing observed-hand path;
hand observation and a desired hand state are different variables.

Thus G says where this selected object should be at t+0.8s. It does not say
which hand is active, whether the goal is approach/grasp/hold/release, where
on the object contact is intended, or when the transition should happen.
An unchanged object endpoint is a valid geometric constraint. It is not a
complete task instruction when the requested hand behavior distinguishes
several outcomes consistent with the same object position.

For example, reaching toward an untouched cup, holding it on the table,
releasing a cup already placed, and manipulating another object while leaving
the cup untouched can all satisfy the cup's unchanged endpoint. These are
not literally identical full conditions: observed hand state/history may
distinguish many cases. Nor does a moving-object goal uniquely determine
every finger action. Ambiguity is normal in inverse control; the issue is
whether the omitted intention matters to the selected task and loss.

## What the local evidence changes

The substantial OakInk2 background-anchor contribution makes target relevance
the first issue. A scene-forecasting anchor is not automatically the object
whose desired state explains a demonstration's hand motion. Adding the exact
future hand endpoint could make such windows easier to fit while leaving
the object-goal learning problem unresolved. Consider actor-only downweighting
or separately audited reanchoring; preserve the original data/world labels.
Reanchored copies must be deduplicated by their physical window and role so
that a scene with many background objects does not gain extra actor weight.

GRAB/ARCTIC only admit anchors already near a hand. Their static windows
contain departures and holds, and far-to-near start states are missing. The
same cleaning rule should not be applied to all three sources. Future-window
labels may define training curricula, but they must not be silently injected
as observed contact or runtime phase.

The cached neighbor test gives wrist endpoints a long-horizon information
signal, with weak/inconsistent immediate-motion improvement. All22 endpoints
are not materially better under that metric. This favors testing sparse
intent before a dense future-hand conditioning system, but does not prove
that sparse goals suffice for grasp precision. The coarse phase input itself
did not improve first8 consistently, so phase-conditioning remains a design
hypothesis, not a selected winner.

## Recommended goal semantics

Keep object outcomes as the primary task goal. Permit a masked interaction
subgoal when the task needs to distinguish hand behavior:

`G = (target object, object outcome, interaction subgoal, timing, validity)`

Possible interaction fields, introduced only as required by a Probe:

| Field | Meaning | Why not infer it from unchanged object displacement alone? |
|---|---|---|
| Active hand / role | left, right, both; manipulating or supporting | An object may remain still while different hands act |
| Interaction phase | approach, establish contact, maintain, release | Static object endpoints permit all of these |
| Sparse desired geometry | object-relative wrist/palm approach pose or contact regions | Distinguishes grasp side/direction without fixing every finger trajectory |
| Event or time target | reach a pregrasp/contact/placement event; bounded time-to-go | The same hand endpoint admits different timing and paths |
| Validity / confidence | supplied, absent, uncertain, or unconstrained component | An unspecified hand goal must not become "keep the hand still" |

This is a candidate interface, not a requirement to add every field at once.
For a first comparison, use one active-hand sparse target with a validity
mask; keep the original object-only arm and all other training conditions.
Evaluate object task completion as well as hand error, and include a goal
intervention to detect an actor ignoring object goals after hand conditioning.
Unspecified goals should be masked, not encoded as zero coordinates. Null
goal and an explicit desired unchanged state must remain distinguishable.

Use a shared coordinate frame and explicit semantics. A hand subgoal can be
represented relative to the selected object's current or desired pose, but
the chosen convention must be identical in training and execution. Human
finger geometry is not automatically feasible for a robot hand; an
object-relative approach/contact region can be more transferable than all
22 absolute human points, pending actual tests.

## Where a deployable hand goal would come from

1. A supplied reference demonstration or task plan can provide intended hand
   waypoints if the declared task permits them. This is a reference-conditioned
   policy, not autonomous inference from an object goal alone. Align geometry,
   active-hand role, timing and robot feasibility before use.
2. A high-level grasp/subgoal proposal can choose feasible contact/palm targets
   from current geometry and the desired object outcome. The low-level actor
   executes them. Predicted/candidate goals, rather than exact future labels,
   must be evaluated at deployment; include proposal failures in task scores.
3. Language or a task command can specify "approach", "hold" or "release"
   without exact hand coordinates. If intent is not in the available inputs,
   a phase predictor cannot recover the demonstrated choice by assumption;
   multiple feasible behaviors may need to be modeled or selected.

Actual expert/learner future hand coordinates are valid supervision and
oracle diagnostic inputs. Feeding them during testing without a legitimate
goal source changes the problem and does not show deployable improvement.
An auxiliary hand-goal prediction head does not add new external information;
it might improve decomposition/regularization, which needs its own test.

The fixed0.8s endpoint is also worth separating from task-goal time. A distant
hand may need more than0.8s just to approach, leaving every short-horizon
object goal unchanged. A later contact/lift event or a longer-term object
outcome can express intent while the actor still emits a short control chunk.
Training such a contract requires matching runtime goal horizon and phase;
simply using later GT object poses without that contract is not a solution.

## Relation to primary literature

The original [PointWAM paper](https://arxiv.org/html/2610.02840v1) conditions
on a language instruction plus scene/current hand state. It does not establish
that an isolated0.8s object endpoint supplies equivalent task information.
Replacing language with local object geometry is our method choice, so the
loss of intent should be evaluated directly.

[UniDexGrasp](https://arxiv.org/abs/2303.00938) separates diverse grasp-pose
proposal generation from goal-conditioned execution. This supports the
concept of supplying a planned hand grasp as a goal with an explicit source;
it does not demonstrate benefits for our WAM or human-three-domain data.
[Dexterous Functional Pre-Grasp Manipulation](https://arxiv.org/abs/2403.12421)
also treats relative hand/object pose and contact as task-relevant. Neither
paper validates the exact optional fields, weights or thresholds proposed here.

## Next decision and stopping boundary

First resolve actor target relevance and establish a legitimate sparse-goal
source for the chosen task. Then compare object-only with object-plus-sparse-
interaction goals using matched initialization/data/compute and explicit
oracle versus supplied/predicted-goal arms. Report first8/contact-phase errors,
object-goal sensitivity and actual execution, not only all24 endpoint error.
Do not run long pretraining based solely on the oracle retrieval gain.
The current native audit cannot explain away the latest robot rollout failure;
robot recovery, timing, proprioception and execution remain separate evidence.


## Follow-up: actual sparse-reference conditioning Probe

[Matched wrist-goal Probe](../experiments/probes/P-20261013-goal-point-wam-reference-wrist-goal.md)
now implements +8/+16/+24 planned human wrist poses, expressed in the measured
current-object frame and admitted only as learned conditioning. Same parent,
R, training pools, draw sequence, parameter capacity and1000-update budget:
A blank wrist vs B planned wrist. Exact three C-query TRAIN first8 mean
57.082->57.667mm; t32 contact-control guard also fails. This fixed recipe is
UNPROMISING, so the recommendation is not to extend it or claim that adding a
wrist solves intent ambiguity. The broader goal question remains unresolved.
Human masking is exact, the path has gradients, and input intervention changes
predictions slightly. Small learned gate, one fixed reference, only three old
C states, and observed batching sensitivity limit attribution. Priority shifts
to current actor visited-state corrections and explicit interaction/contact
coverage; richer goals remain a hypothesis, not an automatic next architecture.
