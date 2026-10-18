# P-20261018-unidex-live-oracle-tau

`task`: `wam-rl-policy`  
`status`: `PROMISING` (closed-loop tau+DAgger holds; state-only controls fail)  
`stage`: Probe  
`classification`: Decision  
`code_branch`: `main`  
`git_commit`: `f731196c`  
`run_id`: `unidex-serial-replay-r1`, `unidex-teacher-rollouts-r1`, `unidex-partial-obs-200tick-32seed-r1`  
`runtime`: `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train` (Python 3.8, Torch 2.0.1+cu118), Isaac Gym at `/home2/wyy/isaac-gym`, local RTX 3090 GPU 0.

## Decision

This probe asks whether an oracle future hand plan, generated from the same
live teacher state, is a usable interface for an executor that cannot see the
object pose/velocity or contact state. The matched comparison is:

\[
\pi(o_t)\rightarrow a_t \quad\text{vs}\quad \pi(o_t,\tau_t^{GT})\rightarrow a_t
\quad\text{vs}\quad \pi(o_t,\tau_{\rm shuffle})\rightarrow a_t .
\]

A positive result justifies a closed-loop receding-horizon test with the
validated serial fork contract. A null result would stop this route for the
11-point plan representation and redirect the next experiment to interaction
context `C`.

## Why the original full-state identification was not informative

The official UniDexGrasp++ `state_based_model.pt` actor is a feedforward
`300 -> 1024 -> 1024 -> 512 -> 512 -> 24` MLP on the 300-D `full_state`
observation (hand DOF position/velocity/force, fingertip state, hand pose,
previous action, object pose/velocity; `goal_cond=false`). Therefore
`a_t^teacher = f(s_t)` and `tau_t^GT` is also a deterministic function of
`s_t`. On full state, a state-only model has the same Bayes error as a
state-plus-tau model; a null result could not indict tau. The executor
observation is therefore made deliberately partial.

## Replay contract

The earlier restore-based fork wrote intermediate PhysX root/DOF tensors back
into the live simulator. It was exact before contact and diverged after a
contact-rich teacher rollout (8 env, tick 1, maximum observation error 13.26).
It was replaced by the narrow serial contract already validated in
consequence-evaluator ([P-20261010-serial-finger-pulse-prefix](../../../../consequence-evaluator/docs/experiments/probes/P-20261010-serial-finger-pulse-prefix.md)):

1. one environment;
2. a fresh process;
3. a fixed seed and a real reset (with one warm-up reset so the construction
   pose is never measured);
4. replay the recorded action prefix from reset; never restore an intermediate
   PhysX state.

Audit `unidex-serial-replay-r1`: four fresh processes, seed 421, 100 ticks,
fields `obs/q/dq/hand/object_pose/forces/dof_force/action/success` were
**bitwise identical** in all pairs; first divergence tick is null. The
single-environment rollout reaches contact, grasp and lift (object lift
0.205 m, official success 1 at tick 99).

Multi-environment is not a twin for this task. An 8-environment teacher
rollout with the same seed lifted only environment 0 (0.091 m); the other
seven stayed at zero. The 8-environment oracle-tau collection is therefore
retained only as engineering evidence and is not used for science. All
reported teacher data below is one environment per fresh process.

## Oracle tau from teacher states

A single-environment closed-loop teacher rollout was run for 200 ticks for
seeds 421--452 (bottle `core/bottle-109d55a137c042f5760315ac3bf2c13e`, scale
0.08). At every tick the oracle plan is the teacher's own future:

\[
\tau_t^{GT}=H^{teacher}_{t+1:t+8}-H_t,\qquad H\in\mathbb{R}^{11\times 3}.
\]

Because both the state and the future are taken from the *same* teacher
rollout, this is the live-state oracle plan for teacher states without any
fork. Ten seeds finish with official success; four additional seeds lift
0.08--0.16 m without final success. The dataset contains 6112 rows
(32 seeds x 191 usable ticks) and a success-only subset contains 1910 rows
(10 seeds). Successful seeds: 421, 422, 424, 425, 436, 438, 439, 445, 449,
452.

## Partial observations

The official 300-D observation is split as:

* `full` (300): the reference observation.
* `p1` (68): DOF position + DOF velocity + previous action.
* `p2` (139): `p1` plus fingertip rigid-body states and hand pose.

`p1` and `p2` contain no object pose/velocity, no DOF-force channel, no
fingertip force/torque and no visual feature. They are a deliberate
handicap, not the final WAM executor contract; they make tau carry the
object/future information that the state lacks.

## Matched identification

The same MLP (`2 x 256`, ELU, 800 full-batch Adam steps, weight decay 1e-4)
is trained for every input variant with three model seeds. Splits are grouped
by teacher rollout seed (4 folds for the 32-seed set, 5 folds for the
10-seed success-only set), so validation uses unseen object orientations and
trajectories. `tau_shuffle` permutes tau rows across the whole dataset before
splitting, preserving the marginal but destroying the state-plan pairing.
Reported errors are native action L1; `xyz/rot/fingers` are the active
wrist-translation, wrist-rotation and finger action groups.

## Results: 32 seeds, 6112 rows

| model | dim | train L1 | val L1 | xyz | rot | fingers | approach | contact | lift |
|---|---|---|---|---|---|---|---|---|---|
| full | 300 | 0.0779 | 0.1218 ± 0.0120 | 0.1327 | 0.2247 | 0.1029 | 0.0817 | 0.1463 | 0.1232 |
| p1 | 68 | 0.1140 | 0.1373 ± 0.0180 | 0.1761 | 0.2485 | 0.1123 | 0.1020 | 0.1756 | 0.1356 |
| p1_tau | 332 | 0.0799 | **0.1087 ± 0.0121** | 0.0867 | 0.2404 | 0.0904 | 0.0733 | 0.1341 | 0.1092 |
| p1_shuffle | 332 | 0.1102 | 0.1437 ± 0.0177 | 0.1862 | 0.2625 | 0.1168 | 0.1179 | 0.1828 | 0.1405 |
| p2 | 139 | 0.0944 | 0.1302 ± 0.0133 | 0.1548 | 0.2307 | 0.1093 | 0.0959 | 0.1543 | 0.1307 |
| p2_tau | 403 | 0.0676 | **0.1046 ± 0.0101** | 0.0731 | 0.2226 | 0.0901 | 0.0687 | 0.1236 | 0.1062 |
| p2_shuffle | 403 | 0.0865 | 0.1331 ± 0.0121 | 0.1614 | 0.2364 | 0.1112 | 0.1065 | 0.1593 | 0.1323 |

Paired differences (same fold and model seed):

| comparison | Delta all | Delta xyz | Delta rot | Delta fingers |
|---|---|---|---|---|
| p1_tau - p1 | -0.0286 ± 0.0063 | -0.0894 ± 0.0085 | -0.0081 ± 0.0055 | -0.0219 ± 0.0067 |
| p1_shuffle - p1 | +0.0064 ± 0.0011 | +0.0101 ± 0.0014 | +0.0140 ± 0.0041 | +0.0045 ± 0.0014 |
| p1_tau - p1_shuffle | -0.0350 ± 0.0060 | -0.0996 ± 0.0088 | -0.0221 ± 0.0062 | -0.0264 ± 0.0063 |
| p2_tau - p2 | -0.0256 ± 0.0039 | -0.0816 ± 0.0089 | -0.0081 ± 0.0093 | -0.0192 ± 0.0035 |
| p2_shuffle - p2 | +0.0029 ± 0.0024 | +0.0066 ± 0.0052 | +0.0057 ± 0.0059 | +0.0018 ± 0.0023 |
| p2_tau - p2_shuffle | -0.0285 ± 0.0030 | -0.0882 ± 0.0068 | -0.0138 ± 0.0059 | -0.0210 ± 0.0033 |
| p2_tau - full | -0.0173 ± 0.0027 | -0.0596 ± 0.0043 | -0.0021 ± 0.0147 | -0.0127 ± 0.0031 |

## Results: success-only, 10 seeds, 1910 rows

| model | dim | train L1 | val L1 | xyz | rot | fingers |
|---|---|---|---|---|---|---|
| full | 300 | 0.0291 | 0.1310 ± 0.0288 | 0.1375 | 0.2226 | 0.1147 |
| p1 | 68 | 0.0526 | 0.1058 ± 0.0253 | 0.1394 | 0.1676 | 0.0899 |
| p1_tau | 332 | 0.0348 | 0.1029 ± 0.0263 | 0.0652 | 0.1917 | 0.0944 |
| p1_shuffle | 332 | 0.0512 | 0.1145 ± 0.0260 | 0.1429 | 0.1829 | 0.0984 |
| p2 | 139 | 0.0424 | 0.1307 ± 0.0327 | 0.1411 | 0.1990 | 0.1176 |
| p2_tau | 403 | 0.0286 | 0.1131 ± 0.0236 | 0.0607 | 0.1909 | 0.1089 |
| p2_shuffle | 403 | 0.0370 | 0.1258 ± 0.0284 | 0.1365 | 0.1960 | 0.1123 |

Paired differences: `p2_tau - p2` is -0.0176 ± 0.0102 overall,
-0.0804 ± 0.0184 xyz, -0.0081 ± 0.0247 rot; `p1_tau - p1` is
-0.0029 ± 0.0053 overall, -0.0742 ± 0.0113 xyz, +0.0241 ± 0.0083 rot.
The xyz gain is consistent; the overall gain is diluted by rotation and
finger trade-offs in this smaller subset.

## Follow-up: orientation-aware tau and object pose

Two cheap additions were tested on the same 32 rollouts. `tau_rot` adds the
teacher's future 8-step hand-euler deltas (24-D); `p3` adds the current object
position and quaternion (7-D) to `p2` (146-D).

| model | dim | train L1 | val L1 | xyz | rot | fingers |
|---|---|---|---|---|---|---|
| p1_rot | 92 | 0.1057 | 0.1375 ± 0.0164 | 0.1638 | 0.2533 | 0.1138 |
| p1_tau_rot | 356 | 0.0772 | 0.1094 ± 0.0123 | 0.0854 | 0.2424 | 0.0912 |
| p2_rot | 163 | 0.0888 | 0.1308 ± 0.0111 | 0.1459 | 0.2302 | 0.1117 |
| p2_tau_rot | 427 | 0.0670 | 0.1070 ± 0.0097 | 0.0729 | 0.2261 | 0.0928 |
| p2_tau_rot_shuffle | 427 | 0.0673 | 0.1070 ± 0.0099 | 0.0741 | 0.2269 | 0.0925 |
| p3 | 146 | 0.0927 | 0.1330 ± 0.0127 | 0.1569 | 0.2327 | 0.1124 |
| p3_tau | 410 | 0.0675 | 0.1062 ± 0.0102 | 0.0755 | 0.2228 | 0.0919 |
| p3_shuffle | 410 | 0.0857 | 0.1358 ± 0.0123 | 0.1646 | 0.2404 | 0.1135 |

Paired: `p1_tau_rot - p1_tau` +0.0007 ± 0.0009 (rot +0.0019);
`p2_tau_rot - p2_tau` +0.0024 ± 0.0017 (rot +0.0035);
`p2_tau_rot_shuffle - p2_tau_rot` +0.0000 ± 0.0013;
`p3 - p2` +0.0028 ± 0.0026 (rot +0.0020);
`p3_tau - p2_tau` +0.0017 ± 0.0015 (rot +0.0002).

Neither the orientation-aware plan nor the object pose closes the rotation
gap. More importantly, the same holds for the full-state reference: full
train rotation 0.137 vs validation rotation 0.225; `p2_tau` 0.143 vs 0.223;
`p3_tau` 0.144 vs 0.223. The rotation gap is therefore not explained by the
11-point plan or by hiding the object pose; it is a generalization/data
coverage limit of this 32-orientation teacher set (or a rotation-action
parameterization limit), not evidence that the plan representation is the
cause.

## Closed-loop feasibility (seed 421)

Protocol: one persistent single-environment student simulator is advanced only
by the student's own actions.  For the tau arm, every tick writes the applied
action prefix and current 300-D observation, then a fresh
`oracle_tau_query.py` process resets with the same seed, replays the prefix,
records the replayed-observation error, and rolls the frozen teacher eight
steps to produce `tau_t`.  The state-only arm uses the same student
architecture without tau and without oracle queries.  Both students are BC on
the ten successful teacher rollouts (1910 rows, `2 x 256`, 3000 Adam steps;
train L1 0.0247 for `p2`, 0.0180 for `p2_tau`).  The initial seed is 421;
150 ticks are run.  The teacher calibration on the same seed lifts 0.205 m and
reaches official success at tick 99.

| arm | max lift above start | object outcome | success | fork replay error | hand-vs-teacher at 60 / 100 |
|---|---|---|---|---|---|
| p2 | 0.000 m | stays on the table | 0 | n/a | 0.114 / 0.303 m |
| p2_tau | 0.057 m at tick 71 | grasps, lifts to z=0.754, drops by tick ~90 | 0 | 0.0 all ticks | 0.017 / 0.301 m |

The `p2_tau` object height is 0.623 (tick 40), 0.666 (50), 0.714 (60), 0.754
(71), 0.700 (80), 0.651 (85), 0.621 (95).  The state-only arm never leaves
the table.  The tau arm tracks the teacher closely through grasp and initial
lift (`hand_vs_teacher` 0.017 m at tick 60), then diverges and drops the
object; the closed-loop action L1 against the teacher is 0.399 overall
(xyz 0.415, rot 0.755, fingers 0.337), with the wrist-rotation error reaching
1.5 around the drop.  The state-only action L1 is 0.373 (xyz 0.500, rot 0.750,
fingers 0.288), but its hand is 0.114 m from the teacher already at tick 60.

The online fork is valid: every query replayed the student's prefix with
**zero** observation error, so the tau fed to the student is the teacher
future from the student's real state.  The failure is the BC student, not the
oracle.  A second simulator cannot be created in the same process
(`Foundation object exists already. Only one instance per process can be
created.`), so the fresh-process query is the only supported online route.

This is a partial positive: conditioning on GT tau changes the closed-loop
outcome qualitatively (grasp and a 13 cm table lift versus no lift), but pure
BC does not hold the object.  The drop coincides with a rotation/finger
divergence, consistent with BC covariate shift and with the offline rotation
gap; it is not evidence that the live tau is wrong.

### DAgger iteration 1 (tau arm)

The first closed-loop run is also a DAgger packet: every oracle query already
replayed the student state and saved the teacher's eight-step action sequence,
so its first row is the teacher action at that student state.  The packet has
150 rows; the student action L1 against those labels is 0.234 (xyz 0.117,
rotation 0.388, fingers 0.228).

Fine-tuning from the BC checkpoint with the packet oversampled 10x (2000 steps,
lr 3e-4) reduces the BC L1 from 0.0180 to 0.0125 and the DAgger L1 from 0.266
to 0.0073.  The rerun with the same seed and 150 ticks reaches:

| metric | value |
|---|---|
| final official success | 1 |
| success frames | 62 |
| max lift above start | 0.219 m |
| object z at tick 100 / 149 | 0.914 / 0.915 m |
| hand-vs-teacher at ticks 60 / 100 / 149 | 0.014 / 0.015 / 0.016 m |
| action L1 vs teacher | 0.071 (xyz 0.079, rot 0.129, fingers 0.060) |
| fork replay error | 0.0 all ticks |

DAgger fixes the covariate shift cleanly: the tau student now grasps, lifts
0.219 m and holds through the end of the episode.  The matched state-only
controls use the same architecture, BC data, DAgger size, fine-tuning schedule
and seed; the only difference is the tau input.

| arm | DAgger packet | DAgger L1 before -> after | max lift | success frames | final success |
|---|---|---|---|---|---|
| p2_tau | own | 0.266 -> 0.0073 | 0.219 m | 62 | 1 |
| p2 | own | 0.285 -> 0.0114 | 0.019 m | 0 | 0 |
| p2 | tau arm's | 0.320 -> 0.0085 | 0.082 m | 0 | 0 |

Both state-only controls fit their teacher labels well but never reach
official success.  The stricter control, trained on the exact tau-arm packet
with tau removed, lifts to z=0.776 (0.082 m above start) around tick 108 and
then drops; the tau arm lifts to 0.917 and holds 62 frames.  On this fixed
seed, the live GT-tau conditioning is the differentiator, not DAgger alone.

### Stage 2: held-out initial states (bottle)

The current BC data covers successful seeds 421--452.  Seeds 453--484 were
collected afterwards as a held-out teacher panel; 10/32 are successful
(453, 463, 466, 468, 471, 472, 474, 478, 480, 482).  None of them is in the
student's BC or DAgger data.

Seed 453 (teacher lift 0.197 m, success first at tick 186) is the first
held-out closed-loop check with the fixed network and DAgger recipe:

| arm | final success | success frames (first) | max lift | final object x/y |
|---|---|---|---|---|
| tau+DAgger | 1 | 19 (131) | 0.181 m | (-0.031, 0.028) |
| state-only+DAgger (tau packet) | 0 | 0 | 0.180 m | (-0.088, -0.058) |
| teacher | 1 | 13 (186) | 0.197 m | (-0.010, -0.001) |

The state-only control lifts just as high but ends 6--8 cm off the goal in
x/y, outside the 5 cm success tolerance; the tau arm stays inside it and
holds through the end.  This is one held-out seed; seeds 471 and 482 are
running to estimate whether the effect is seed-specific.

## 2026-10-18 audit caveat: action prefix and success frames

The pinned `shadow_hand_grasp.py` rewrites `actions[:,0:3]` and `actions[:,3:6]`
in place inside `pre_physics_step`. The original runner appended the action to
`actions_history` **after** `sim_step`, so the saved prefix could contain the
world-frame wrist command rather than the canonical command that was issued.
The oracle then submitted that prefix as a canonical command, rotating the
wrist twice. The current code saves the action before the step and passes a
clone to the simulator; the manifest records
`action_prefix_matches_executed`.

The archive was checked directly. In all four retained closed-loop runs
(`run-p2`, `run-p2_tau`, `run-p2_tau-dagger1`, `run-p2-labels`),
`trajectory['action']` and `trajectory['action_prefix']` are exactly equal and
the maximum replay error is 0.0. Their wrist action channels are nonzero. The
runtime copy in the same output directories contains the in-place rewrite, so
the equality implies that the rewrite did not change the executed values under
the measured reset state (the effective yaw transform was the identity for
these runs). The historical zero replay error is therefore credible, but the
old code was relying on an accidental condition. With a nonzero yaw the same
code would have saved a double-rotated prefix. The fixed code removes that
dependence; future runs record `action_prefix_matches_executed`.

The same card's `success_frames` was computed by summing the upstream
`successes` tensor. That tensor is a cumulative "ever reached" flag, not a
contiguous hold count. The corrected runner records realtime
`goal_distance <= 0.05` frames, `longest_hold_frames`, `first_success_tick`,
`final_success`, `success_frames_ever` and drop events. The sentence
"holds 62 frames" should therefore be re-derived from the archived trajectory or
from a rerun; it is not supported by the old aggregate alone.

## Interpretation

1. **The oracle plan is usable for translation and fingers.** Adding tau
   halves wrist xyz error (0.155 -> 0.073 for `p2`; 0.176 -> 0.087 for
   `p1`) and improves the overall action L1 in the 6112-row set. The shuffle
   control is worse than the matched tau arm, so the gain is from the
   state-plan pairing, not merely from a larger input.
2. **Rotation is unresolved, but not attributable to the 11-point plan.**
   Rotation error is unchanged by tau (`p2_tau - p2`: -0.0081), by the
   orientation-aware plan (`p2_tau_rot - p2_tau`: +0.0035), and by adding
   object pose (`p3_tau - p2_tau`: +0.0002). The full-state reference shows
   the same train/validation gap (0.137 vs 0.225). The current probe cannot
   separate "plan lacks orientation" from "the 32-orientation data or the
   rotation parameterization does not generalize"; do not claim the former.
3. **Partial plus tau can beat the full-state state-only reference on unseen
   seeds** (`p2_tau - full`: -0.0173 ± 0.0027). This is not evidence that tau
   is better than full state in general: the full model overfits
   (train 0.078 vs val 0.122), while the lower-dimensional partial model
   generalizes better. It is evidence that tau is a useful compressed
   interface when object/contact state is unavailable.
4. **The serial replay route is alive, but multi-environment is not.** The
   four-process single-environment audit is bitwise exact. A faithful fork
   must therefore be built as a fresh single-environment prefix replay, not
   as a second GPU environment or a restored PhysX state.
5. **DAgger makes the closed-loop tau result real.** With the live oracle
   fork (zero replay error) and one DAgger iteration, the tau student reaches
   official success, lifts 0.219 m and holds 62 frames.  The matched
   state-only students, including one trained on the identical tau-arm packet
   with only the tau input removed, plateau at 0.019--0.082 m and never
   succeed.  For this fixed bottle/seed task, the live GT-tau plan is the
   operative ingredient.  This is one seed and one episode per arm; it is not
   yet a generalization claim.

## Resource and stop condition

Local GPU 0 only; no remote resources. Each 200-tick single-environment
teacher rollout takes about 7--14 s including startup. The 32 rollouts and
the identification training are bounded and complete. A tau-conditioned
closed-loop episode costs one fresh process per tick: 150 ticks took 1371 s
(about 9 s/query) in this run. The state-only arm took 6.3 s because it needs
no oracle. DAgger with the same query increases the per-episode cost only
linearly in ticks; the fresh-process reset is the dominant cost.

## Limitations

* One object (bottle, scale 0.08). The 32 seeds vary the object orientation,
  but object shape and position are fixed. Only 10/32 teacher rollouts reach
  final success; the all-seed set is the primary result and the success-only
  set is underpowered.
* Offline teacher-action identification is a sanity check, not closed-loop
  task success. It does not show that a closed-loop executor can hold or
  place. The oracle tau is taken on teacher states only, so recovery from
  student drift is not tested.
* `p1`/`p2` keep the previous action, which in this dataset is the teacher's
  action and can leak object/contact information. In closed loop it would be
  the student's own action, so this identification is conservative about
  tau's benefit rather than optimistic.
* The partial observation is a deliberate handicap. It is not the final WAM
  executor observation and its absolute L1 is not comparable to the native
  policy's closed-loop behavior.
* The full-state model is a reference, not a capacity-matched upper bound;
  its overfitting is expected with 6112 rows.

## Next decision

Stage 1 is complete on the fixed seed: the live GT-tau executor with one
DAgger iteration grasps, lifts and holds, while the matched state-only
controls do not.  Stage 2 should now use a small set of held-out successful
seeds/orientations that were not in the student's BC or DAgger data, with the
same two-arm protocol.  Only after that should the plan be replaced by
WAM-predicted tau or extended to more objects.  The current result is one
seed, one episode per arm, and one object; do not describe it as a general
closed-loop grasping claim yet.

## Artifacts

* Serial replay audit: `outputs/wam-rl-policy/P-20261018-unidex-serial-replay-r1/audit.json`
* Teacher rollouts: `outputs/wam-rl-policy/P-20261018-unidex-teacher-rollouts-r1/serial200-seed-421..452/`
* Identification data: `outputs/wam-rl-policy/P-20261018-unidex-partial-obs-200tick-32seed-r1/identification_data.npz`
* Training and summary: `outputs/wam-rl-policy/P-20261018-unidex-partial-obs-200tick-32seed-r1/{train-all,train-success,summary}/`
* Orientation and object-pose follow-up: `outputs/wam-rl-policy/P-20261018-unidex-partial-obs-200tick-32seed-p3-r1/`
* Closed-loop students and runs: `outputs/wam-rl-policy/P-20261018-unidex-closed-loop-r1/`
* DAgger packets and fine-tuned students: `outputs/wam-rl-policy/P-20261018-unidex-closed-loop-r1/{dagger-p2_tau,dagger-p2,student-p2_tau-dagger1,student-p2-dagger1,student-p2-dagger1-tau-packet}/`
* Online oracle query: [`oracle_tau_query.py`](../../../tools/run/oracle_tau_query.py)
* Closed-loop runner: [`closed_loop_unidex_oracle_tau.py`](../../../tools/run/closed_loop_unidex_oracle_tau.py)
* Student training: [`train_closed_loop_student.py`](../../../tools/run/train_closed_loop_student.py)
* Runners: [`probe_unidex_serial_replay.py`](../../../tools/run/probe_unidex_serial_replay.py),
  [`train_partial_obs_identification.py`](../../../tools/run/train_partial_obs_identification.py)
* Audits: [`audit_serial_replay.py`](../../../tools/audit/audit_serial_replay.py),
  [`build_partial_obs_dataset.py`](../../../tools/audit/build_partial_obs_dataset.py),
  [`summarize_partial_obs.py`](../../../tools/audit/summarize_partial_obs.py)
