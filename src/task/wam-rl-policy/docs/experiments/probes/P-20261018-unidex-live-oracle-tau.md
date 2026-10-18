# P-20261018-unidex-live-oracle-tau

`task`: `wam-rl-policy`  
`status`: `PROMISING` (translation/finger identification; rotation limit)  
`stage`: Probe  
`classification`: Decision  
`code_branch`: `main`  
`git_commit`: `a523c597`  
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

## Interpretation

1. **The oracle plan is usable for translation and fingers.** Adding tau
   halves wrist xyz error (0.155 -> 0.073 for `p2`; 0.176 -> 0.087 for
   `p1`) and improves the overall action L1 in the 6112-row set. The shuffle
   control is worse than the matched tau arm, so the gain is from the
   state-plan pairing, not merely from a larger input.
2. **The 11-point plan does not carry wrist orientation.** Rotation error is
   essentially unchanged (`p2_tau - p2`: -0.0081; `p1_tau - p1`: -0.0081 in
   the 32-seed set, with a positive rotation delta in the success-only set).
   The plan is a position-only object, so the executor still needs the object
   orientation or an orientation-aware plan to resolve the wrist rotation.
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

## Resource and stop condition

Local GPU 0 only; no remote resources. Each 200-tick single-environment
teacher rollout takes about 7--14 s including startup. The 32 rollouts and
the identification training are bounded and complete. Stop before closed-loop
rollout: a closed-loop executor requires one fresh single-environment prefix
replay per query, which is a separate cost/engineering decision.

## Limitations

* One object (bottle, scale 0.08). The 32 seeds vary the object orientation,
  but object shape and position are fixed. Only 10/32 teacher rollouts reach
  final success; the all-seed set is the primary result and the success-only
  set is underpowered.
* Offline teacher-action identification is a sanity check, not closed-loop
  task success. It does not show that a closed-loop executor can hold or
  place.
* The partial observation is a deliberate handicap. It is not the final WAM
  executor observation and its absolute L1 is not comparable to the native
  policy's closed-loop behavior.
* The full-state model is a reference, not a capacity-matched upper bound;
  its overfitting is expected with 6112 rows.

## Next decision

The positive translation/finger result justifies one closed-loop test with
the serial fork contract. Before spending that budget, decide the plan
representation: either keep the object orientation in the executor
observation, or extend tau with hand orientation. Without one of those, the
rotation limitation above will remain. If the closed-loop fork is too
expensive, the next cheap experiment is an orientation-augmented tau
identification on the same 32 rollouts.

## Artifacts

* Serial replay audit: `outputs/wam-rl-policy/P-20261018-unidex-serial-replay-r1/audit.json`
* Teacher rollouts: `outputs/wam-rl-policy/P-20261018-unidex-teacher-rollouts-r1/serial200-seed-421..452/`
* Identification data: `outputs/wam-rl-policy/P-20261018-unidex-partial-obs-200tick-32seed-r1/identification_data.npz`
* Training and summary: `outputs/wam-rl-policy/P-20261018-unidex-partial-obs-200tick-32seed-r1/{train-all,train-success,summary}/`
* Runners: [`probe_unidex_serial_replay.py`](../../../tools/run/probe_unidex_serial_replay.py),
  [`train_partial_obs_identification.py`](../../../tools/run/train_partial_obs_identification.py)
* Audits: [`audit_serial_replay.py`](../../../tools/audit/audit_serial_replay.py),
  [`build_partial_obs_dataset.py`](../../../tools/audit/build_partial_obs_dataset.py),
  [`summarize_partial_obs.py`](../../../tools/audit/summarize_partial_obs.py)
