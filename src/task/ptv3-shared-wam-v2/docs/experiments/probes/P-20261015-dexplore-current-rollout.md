---
schema: ref2dex.probe.v2
probe_id: P-20261015-dexplore-current-rollout
experiment_id: P-20261015-dexplore-current-rollout
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: 46061aa7
claim_id: C1
hypothesis_family: HF-dexplore-current-observation
probe_index_in_family: 1
seed_pool: probe
seeds: [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161]
decision_changed_if_positive: enable a matched native-current-observation input comparison on the freshly paired corpus
decision_changed_if_negative: fix capture alignment before any input-benefit training
status: UNCLEAR
run_id: dexplore-current-rollout-20261015-r1
---

# Complete same-execution DExplore current-input rollout collection

Result: Engineering capture/audit COMPLETE and PASS; native-input benefit remains
UNCLEAR. Engineering/data Blocker for Objective A, not an input-benefit
result. User explicitly requests the complete660-motion recollection on
physical GPU0/1/2 concurrently, after correcting the previous incomplete
reference/geometry-only experiment. The source remains the frozen660 motions;
the official actor is only the data generator, not the learned policy base.
Decision: collect the actually missing native current observation/contact/body
state together with q_PD before testing whether that information helps.

## Native contract

Use the same teacher, reset patch, PD controller, source motions,30Hz and
batch16 as original RL660. Teacher still executes its trained delta1+delta16
1442D observation. Additional learner observation is the original native
`_compute_observations_iter(all_ids,0)`721D; no future reference branch is
saved as a learner input. Same-tick reference targets/errors are authorized.
Reference velocity fields are the native current reference state attributes.
Do not approximate contact with proximity or substitute an11-point IG.

Before each action[t], save measured q/dq,11 hand points, object pose/velocity,
native_current_obs721, all humanoid body states13D (position/quaternion/linear/
angular velocity), all humanoid contact-force vectors, object contact-force
vector, current native reference state, and integer frame_tick. Capture
action/applied/q_PD[t] from the actual controller on that execution. State
arrays have T rows, command arrays T-1. Full fields are retained for later
offline semantic audit; body/key/contact identities and layouts are recorded.

Live gates: exact tick alignment without reset; current native contact bits
and reference-contact errors match raw same-tick forces/reference; observation
shape721; original teacher obs_buf unchanged; changing all future reference
rows at t=0 leaves delta0 observation exactly unchanged; existing actual
command/q_PD roundtrip <=1e-6; all exported tensors finite. Small capture-on/off
native smokes check the path; live before/after equality checks establish
nonmutation of teacher obs and measured physics. Separate PhysX executions
are not assumed byte-identical.

Reuse original660 membership and original TRAIN/dev sequence assignment
from rl660-corpus-20261014-r2/manifest.json. New input arms must later use
the SAME newly captured execution and labels. Do not join new contacts onto
old trajectories. Per-batch RNG120+index reuses original collector seeds;
these are paired collection identities, not42 independent research repeats.

## Resource / Decision Note

GPU0/1/2, one native process per card,42 fixed16-motion batches (last4),
900s per batch and10800s overall,8GB new artifacts,20GiB minimum headroom;
no videos or policy training. Only owned child groups may be terminated.
Stop on schema/alignment/nonfinite failure, unknown GPU occupancy, deadline
or resource breach. Preserve failed attempts and allow receipt-based resume.
Default six-card boundary would conflict with four-card training; user
explicitly permits seven cards for this collection plus existing fine-tune.
CAMPAIGN records this run-scoped exception. Source and old corpus read-only.

At preflight, existing fine-tune had already failed at logged3340 after a
concurrent source change triggered its implementation guard. Preserve its
checkpoints/error. Restore it separately from the last complete checkpoint
under fixed code, preserving recipe/optimizer/world size, within the seven
card exception. Do not overwrite the concurrent source edits.

Output: `outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1/`.
Tools: existing ptv3-shared-wam/tools/run/collect_rl660{,_batch}.py with explicit
capture flag, inherited split and three-GPU scheduling. New root manifest is
not training-admitted until all660 receipts and payload gates complete.

## Limitations / future evidence

Collection success is not evidence of grasp success or input benefit. Native
reference flags are planning labels, measured contacts are separate. PhysX
rollouts may differ numerically from old exports, so old trajectories cannot
be treated as byte-identical labels. Original reference tracking windows are
recomputed on new execution, never copied blindly. Matched R/WAM training and
execution qualification are separate next steps; no scientific comparison is
claimed by this collection.

## Preflight and active execution

Capture code46061aa7; full supervisor1548597, GPU0/1/2 concurrently, seed120,
batch16,42 fixed batches; first96/660 complete at183s. Initial count-based ETA
about18min remaining; variable batch lengths make this approximate. Per-batch
metadata records the actual code revision (later documentation/audit-only
commits do not change the two capture scripts). Collection remains
training_allowed=false until complete raw audit.

Independent review found an inherited terminal-reference bug: setting all
episode lengths to the batch maximum let short sequences query zero-padded
teacher future references. New capture mode now fills process-local hoi_data
and hoi_refs tails with each last valid reference, exactly reproducing the
original per-motion reference clamp for t/t+1/t+16 while resets stay suppressed.
No source motion is edited. The known-answer mixed-length test checks this
equivalence; old exports remain unchanged and are not reused as new labels.

Final smoke r3: native delta0/current contact slices, same-tick q_PD conversion,
clock continuity, future-reference invariance and before/after physics-state
equality all pass. Native2-motion425-frame/423-command raw audit PASS,245 frames
with measured contact, independent PD maximum error0. Nine tests cover resume,
native-vs-legacy receipt rejection, terminal reference clamping, valid native
payload and deliberate contact/label/clock corruption. Artifacts are under
tmp/dexplore-current-rollout-20261015-r1 and dexplore-current-smoke-20261015-r3.

Paired on/off runs match at initialization/early steps but diverge after
contact; repeating the off run on the SAME GPU also diverges, with comparable
errors. Therefore do not claim bitwise rollout equivalence or join executions.
Live invariants verify the capture does not mutate teacher input or physics;
the two complete off receipts and differences remain as repeatability evidence.

CPU post-collection audit PID1553774 waits for all660, then checks every field,
all frame ticks, original split, raw contact/reference-contact slices, measured
hand/body identity, current root/reference error, reference object position,
applied-command equality/bounds and independent same-tick PD reconstruction.
Only a PASS admits the new corpus. Output current_observation_audit.json;
pre-audit manifest retained. Script audit_current_rollout.py commit39860e44,
log tmp/dexplore-current-rollout-20261015-r1/audit-supervisor.log. This is
engineering data qualification, not a benefit/generalization/grasp claim.

Independent first-wave review checks48 sequences/9622 frames,5252 frames
with measured contact, independent PD max error0. SciPy reconstruction of
native actual-body blocks15:252 from saved raw body_state matches within
6.37e-7 position,3.44e-6 rotation6D,8.59e-6 linear velocity,3.59e-5 angular
velocity; quaternion norm error3.58e-7. This is engineering evidence that the
new body observations are actually captured, not a benefit result.

## Completion / human review

Full run completed in1193.60s:660 sequences,166337 state frames,165677 command
frames;88783 frames contain measured native-hand contact. Full CPU raw audit
PASS with independently recomputed same-tick PD maximum error0; original
TRAIN/dev membership preserved. Native payload admission now explicitly
`training_allowed=true`, audit in `current_observation_audit.json`, original
pre-admission root manifest retained. Recomputed position tracking admits
122623 windows. This qualifies payload consistency, not grasp quality.

User requests visual supervision because the current<=100mm object-position
window gate does not enforce hand tracking, object orientation, penetration,
held/lift success or contact stability. Viser review entry is
`tools/audit/review_dexplore_rollouts.py` (Task-relative), using saved actual
25-body states and object poses, URDF meshes, current reference object/keypoints
and measured contact. Labels go into separate review output JSON files, never
rewriting this corpus or automatically changing the training split/windows.
No GPU or new rollout is required. User explicitly requests parallel native
input experiments while this review tool is developed; those are a separate
matched-learning Probe, with the default6-GPU cap restored after collection.

Descriptive full-corpus review prioritization is saved separately under
`outputs/ptv3-shared-wam-v2/dexplore-review-20261015-r1/quality-review-priority.json`.
An independently inspected example is `s8_spheresmall_inspect_1`, tick255:
actual q translation is[5,-5,1.5664]m, mean16-key-body reference error7.9286m,
but current/future24 object position errors peak at0.030024m and the old
position-only gate admits that window. This is a concrete screen failure,
not a general grasp-success statistic. The existing encoder workspace gate
is a separate compatibility restriction, not a human-quality filter.
Prioritization uses errors/contact only to choose review candidates; it
does not generate human verdicts or alter corpus/window membership.
