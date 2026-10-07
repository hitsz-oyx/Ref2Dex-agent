---
schema: ref2dex.probe.v2
probe_id: P-20261007-consequence-baseline-rebuild
experiment_id: P-20261007-consequence-baseline-rebuild
date: 2026-10-07
task: consequence-evaluator
branch: consequence-evaluator
git_commit: 567c9091c642f3b6c21c36b830d0a171437231ca
claim_id: C3
hypothesis_family: HF-consequence-baseline-rebuild
probe_index_in_family: 1
seed_pool: probe
seeds: [289, 290]
decision_changed_if_positive: promote the newly trained parent to s3 transfer and the remaining expert recipe
decision_changed_if_negative: diagnose native geometry and learning before spending the six-expert budget
status: UNCLEAR
run_id: baseline-rebuild-original-20261007-r1
---

# Rebuild a self-trained rollout substrate after asset loss

## Motivation and Decision Note

The user explicitly authorizes retraining experts and regenerating continuous
rollouts when the old six checkpoints cannot be found. This supplies the robot
H/A/physical futures and reliable success/preference examples required by ref1.
It does not change Mission/C3 or reuse the missing baseline's validation claims.
Git recovery did not restore its outputs. Backups elsewhere remain possible;
the current bounded searches do not prove permanent loss.

The external read-only Ref2Dex project still has canonical filtered GRAB
geometric tensors and native Inspire/scene assets. Thirteen selected motions
(the twelve-task panel plus s1 airplane) pass shape/finite/left-contact checks.
The local DExplore data symlink pointed at the deleted baseline; its original
link is preserved, and assets/configs are copied into owned outputs. These are
imitation references, not robot evaluator data or evidence of grasp success.

Historical s3 direct scratch PPO failed. The historical expert was derived by
transferring a learned s1 actor, so first test a new s1 scratch parent with the
existing native Cm-off approach/held-lift/progress shaping and contact curriculum.
The old parent used a CmLite-trained initialization; this new Cm-off recipe is
an adaptation and its success is unknown. No official actor weights are loaded.

First native two-epoch engineering smoke: debug seed17, eight envs. If it fails,
do not start the parent fit. Then random-init seed289,64envs,horizon32,
minibatch256,lr1e-5,200epochs; reset contact/lift fractions.5/.25,
curriculum annealing40..80, save every20epochs. Use a fresh initialization after
smoke. One GPU, at most300s smoke/3600s parent fit/5GiB total outputs. Queue waits
at most3600s for the confirmed owned PointWorld launcher and requires COMPLETED
plus an empty GPU0 compute PID list before any Isaac/model initialization.
Other users' jobs are never stopped. New-stage runtime commit and all frozen
source/input hashes are recorded in the queue manifest; later docs-only commits
may follow the code revision above.

## Minimum evaluation and next decision

After training, separately evaluate the fixed endpoint under full native frame0
starts with early termination disabled, seed290,64first episodes. Primary task
qualification is >=3cm contact-proxy elevation sustained45frames with no later
drop (<2cm or >=6 lost-contact frames). Report counts and durations, not training
reward as success. Native net-force contact is a proxy, not pairwise collision
identity. If at least8/64 reliable successes are present, proceed to the new s3
transfer and six-expert construction; otherwise inspect geometry/control and
fit before expanding. This is an operational data-readiness gate, not a formal
validation claim or a replacement for phase/object coverage in evaluator data.

The six experts are planned as airplane_base/mixed12/train5/balanced5/duck/cup;
they receive new weight hashes and a new route config. The temporary s1 parent
is not counted as one of the six. Further expert fits and robot sampling remain
pending parent qualification. Never fill six route slots with the same parent
and claim the old substrate is restored.

## Artifacts and current status

- Inputs: `outputs/consequence-evaluator/baseline-inputs-20261007-r2/manifest.json`.
- Queue: `outputs/consequence-evaluator/baseline-rebuild-20261007-r2/run_manifest.json`.
- Detached queue PID870294 was confirmed live, initially WAITING for PointWorld.
  First shell-background attempt lost its process before any GPU stage;
  `baseline-rebuild-20261007/startup-check.json` records this execution failure.
- Preserve initial failed preparation in `baseline-inputs-20261007`; it copied
  assets before encountering the broken data symlink and never ran simulation.
- PointWorld completed normally. The r2 native smoke then failed before any
  PPO update: Isaac Gym imported torch_utils before the existing native NumPy
  alias compatibility shim, raising AttributeError for np.float. The wrapper
  now imports the Torch-free native bootstrap first; no installed library or
  external project is modified. Original smoke logs/FAILED manifest remain.
- Next attempt is `baseline-rebuild-20261007-r3`, on the now-empty GPU0 with the
  same staged inputs,2-epoch smoke then200-epoch parent and original fit/output
  caps. The queue also accepts its owned PointWorld COMPLETED/exit0 record;
  GPU occupancy is still checked before every stage. r2 produced no model
  updates and does not consume a scientific Probe or justify changing recipes.
- Native GPU smoke completion and learning evaluation are pending. No real robot
  examples, expert qualification or oracle headroom claim exists yet.

r3 passed the NumPy import but failed in the first environment reset, before
PPO updates: the CPU PhysX tensor pipeline was mixed with native CUDA reference
indices/tensors. The task-local wrapper now selects the GPU pipeline explicitly
to keep native simulation views, references and actor on CUDA. Original r3 logs
are retained. Next retry is baseline-rebuild-20261007-r4, GPU0only, the same
staged inputs and2/200epoch caps. User explicitly allocates GPUs1/2to separate
OakInk2 continuation; this expert run never shares those devices. No physics
success claim follows from fixing tensor placement.

## Resumed GPU0 execution

User resumes consequence-evaluator while the four-source PointWorld fit runs
on GPU1/2. GPU0 was confirmed empty; launch source commit7c85d5d, queue1004318.
The native GPU-pipeline smoke completes2PPO epochs in51.07s with endpoint
`smoke/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000002.pth`.
This resolves the r3 first-reset device failure in the actual integration;
it proves updates/checkpoint saving, not successful grasping. Fresh parent_s1
seed289/64env starts separately, child1007162, fixed200epochs/3600s. Sources
and staged inputs remain frozen; original failed r2/r3 logs are retained.
Artifacts: `outputs/consequence-evaluator/baseline-rebuild-20261007-r4/`.

Qualification entry is `tools/run/qualify_parent.py`: reuse the native player
with GPU tensor pipeline, force and verify all64first episodes start at frame0,
disable early termination, capture step-major physical transitions, and compute
the predeclared45-frame/no-later-drop gate independently of the player's weaker
5-frame lift indicator. First-done truncation excludes later resets, and pre/
post-state and progress clocks are checked. Seed290,<=900s/1GiB, fresh output.
Qualification transition actions can be native-mutated PD values; these are
diagnostic traces, explicitly not promoted to evaluator training examples.
Six synthetic gate tests cover45-frame hold, later drops/resets, missing done,
shifted clocks/poses and nonzero starts. The entry also accepts a separately
marked reference-controller replay, which can never qualify the learned actor.

Parent completes200epochs in1114.91s at training commit7c85d5d; endpointSHA256
69d7e32ccfa4a89cecbc2f0703ed42a8fa21528cd4b365cb2efa46d1ea6933ca.
The parent queue completes in1193.65s including smoke/supervision intervals.
Last training contact/held-lift statistics remain0, despite improving negative
reward; no grasp quality is inferred from PPO reward. Pure CPU reference FK
geometry at contact/lift frames57..82 has sampled gap0.49..1.62mm, vs1.43m at
frame0 (`reference-surface-audit.json`); this does not prove executable physics.

Actual qualification starts at codecommitc0b10fb, GPU0,seed290,PID1048746.
Output `baseline-rebuild-20261007-r4/qualification-s290/`, log adjacent
`qualification-s290.log`. The bounded owned wait/launch record is
`tmp/consequence-baseline-rebuild/qualification-queue-r4.json`. Native GPU
qualification completes all64first episodes,431steps each:0/64qualified,
no3cm lift, mean contact proxy fraction0.0001. Qualification saysUNCLEAR and
data_readiness_pass=false, below the operational8/64 gate. No six-expert
expansion or evaluator fit has been started. This is insufficient substrate
readiness, not evidence against the evaluator or world-model hypothesis.

Decision Note: distinguish weak new scratch learning from an unexecutable
reference/control/geometry setup before paying for six experts. Ranked
hypotheses: (1) missing historical CmLite initialization/insufficient PPO fit;
(2) native reset/control-frame integration mismatch; (3) canonical retargeted
references have geometric proximity but cannot hold the real object. Cheapest
probe is one same-seed290/64env/frame0 reference-controller replay with lead1,
same checkpoint/environment/45-frame audit, GPU0<=900s/1GiB and a fresh output.
Only action source changes; replay never qualifies the learned parent. If
replay holds, prioritize a bounded informed initialization/learning check; if
replay fails, inspect measured/reference q, object and contact before more PPO.
Stop on clock/state mismatch, input drift, timeout or output cap; no further
external authorization boundary is crossed.

## Reset implementation blocker (2026-10-07)

Both the learned-parent qualification and the reference-action replay have
64/64 first-step object displacements of1.4823..1.4841m. Objects start at the
requested reference position(-.02165,-1.18536,.91826), but the first measured
post-step state is near the native asset creation position(0,0,.025), with
near-zero velocities. This is an implementation blocker; these runs cannot
establish whether an executable parent recipe learns the intended task.
The original artifacts are retained. The historical CmLite initialization
chain must also be restored rather than equated with new Cm-off scratch PPO.

Decision Note: first isolate reset persistence in one real GPU physics step,
before spending another parent or six-expert training budget. Ranked causes:
setter-after-refresh state loss, repeated GPU root setters, and index lifetime.
NVIDIA's installed tensor documentation requires a single call per setter
between simulate calls and refresh before setters. The native reset violates
both timing constraints. The task-local `tools/audit/native_reset_probe.py`
uses eight envs/seed17/reference control on GPU0,180s/1MiB per fresh run.
First run the unchanged native path as a red regression; then test a process-
local batched reset without touching native/shared training code. A successful
first-step check alone does not certify initial hand FK, partial resets or
grasping; audit those before production training. Stop on input drift,
occupied GPU, deadline, nonfinite states or unphysical displacement. No new
external authorization is needed within the existing user-authorized rebuild.

The red command was `graspenv/bin/python -B tools/audit/native_reset_probe.py
--run-dir <r4>/parent_s1 --output <owned>/reset-native-20261007-r1 --gpu0`
(actual arguments are frozen in the run manifest). It completed35.31s with an
assertion: all8first-step displacements1.4824..1.4842m. Same inputs/seed/GPU
with batched roots and no immediate refresh complete14.86s, displacements
1.4346..1.4353mm. This fixes a real physics reset defect, not merely a mock.

Process-local implementation is in `src/consequence_evaluator/native_reset.py`.
It retains int32 actor IDs, submits all changed roots once, reconstructs the
initial hand bodies using actual native DOF/body names and URDF FK, resets old
contact caches, and preserves unrelated env roots during native subset resets.
It does not add a settling physics tick. FK velocities use native body COMs
because PhysX linear velocity is measured at COM. At commitc6775e1, GPU check
`reset-fk-20261007-r3` completes with max measured/FK position error3.16um,
quaternion error9.9e-7, linear-velocity error8.48e-5m/s and angular error
8.11e-4rad/s. Alternating subset resets[0,2,4,6]/[1,3,5,7]/[0,2,4,6]
preserve unrelated caches and pass real first-step persistence. The patch is
wired into training, qualification and continuous collection. Qualification
now rejects a frame0first-step displacement>10cm.51CPU contract/geometry
tests pass; these and short physics checks do not certify parent grasping.

Historical reconstruction audit also finds the original s1 input still exists:
`/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/dexplore_reconstructed_v120_coordfix_v4/converted_attempt1/s1_airplane_lift/interaction_hand_inspire.pt`,
SHA305dfd16d9bac0e93a95de5b6fa8b9e721d1edfee6946e49100fe231760a205b.
It is[432,598], but is not equivalent to the canonical tensor used in r4.
Original frame0object(.01824,-.08290,.91826) differs from canonical
(-.02165,-1.18536,.91826); q also differs by up to1.316m wrist translation,
1.779rad wrist rotation and3.142rad finger angle. Original right-contact
labels are-1, whereas canonical has1068active entries. Matching relative
right-wrist/object displacement alone therefore does not preserve the recipe.

Old and current training both use `/home2/wyy/miniconda3/envs/graspenv/bin/python`,
`Dexplore_Inspire` and the same installed `isaacgym/_bindings/.../gym_38.so`
path. Current Python3.8.20/Torch2.4.1+cu121/NumPy1.24.4/rl-games1.6.5 are
recorded, but historical binary/dependency hashes are not available. The
separate `dexplore_v120_data` interpreter was for conversion, not PPO training.
Current staged object-balanced env config differs from the old default.

The original chain is s1 CmLite scratch e140→s3 Cm-off e160→e180→backtrack
e260. The original s1 command is recoverable in09/22session01a0c738 line3122:
CmLitecoef5/predicted-contact/max-gap.1,64env/h32/mb256,200epochs,seed45,
approach2/held10/lift-progress5,contact±3/fractions.5/.25,anneal40→80,
save10; noLRoverride (historical records sayconstant2e-5/mini-epochs6).
CmLite checkpointSHA d5de89b895d272e85ee72ee5ffddd76e6af3a4a0734115426074ad817c55bc52
and its old transitions remain absent from the directly checked roots. Native
s3 corrected input also needs reconstruction. Exact recovery is not claimed.
Commit23074d3 starts bounded `reset-original-s1-20261007-r1` onGPU0 using
the original read-only s1 tensor, before selecting the next training recipe.
It completes15.78s: eight first-step displacements1.4347..1.4352mm,
maxFK position error2.46um/linear velocity error4.16e-5m/s/angular error
3.58e-4rad/s; all three alternating subset resets pass. This validates reset
compatibility with recovered original inputs, not a learned grasp policy.

Original-reference full-frame0 replay at commit1b96a4e also completes all64
episodes in40.18s, mean maxlift6.7mm/contactlift6.4mm,0qualified. This is
close to the historical6.5mm reference replay and is engineering evidence
that reference following alone is insufficient, not a learned-policy result.
Artifacts: `outputs/consequence-evaluator/reference-original-s1-20261007-r1/`.

Decision Note: recover useful self-trained data cheaply before rebuilding a
missing CmLite dependency. A corrected-input Cm-off parent is a cheaper first
probe than recreating four lost Cm-off/Cmv2 rollout sources and then CmLite.
Historical same-input Cm-off scratch had partial success; r4 supplied no valid
negative evidence because reset and reference inputs were wrong. Choose one
200epoch Cm-off fit using recovered s1SHA305dfd16, original single-motion
inspire.yaml, LR2e-5/constant/mini-epochs6,64env/h32/mb256, the old geometry
rewards2/10/5 and contact/lift curriculum. Anneal40→80,save20,seed289;
this is an explicit new recipe without the old learned CmLite reward, not exact
reproduction. Two-epoch8env/seed17 smoke<=300s, then fresh parent<=3600s;
oneGPU0/total5GiB. Stable frame0 evaluation uses seed290/45frame/no-later-drop
criterion and the same operational8/64 gate. Positive permits s3 transfer and
actual normalized-action rollout data for CmLite/evaluator; negative first
audits contact, learning, intended target and recovered input semantics before
choosing informed initialization or rebuilt CmLite. No checkpoint scan or new
six-expert budget is activated merely by declining PPO loss. Stop on drift,
nonfinite updates, device conflict, deadline or output cap.
Staging: `outputs/consequence-evaluator/baseline-original-inputs-20261007-r1/`;
run: `outputs/consequence-evaluator/baseline-rebuild-original-20261007-r1/`.

The new queue starts at commit567c909. Native two-epoch8env smoke completes
36.18s, endpointSHA2b8e2b8150ddd61754ec9343503f862a2200bd047e9b300908d420d404b20582;
real contact proxy19.1%/held-lift-positive9.4% at the first rollout, confirming
that the intervention reaches physical task dynamics. Fresh parent then
starts separately; earlyepoch9 totalFPS476, approximately14minutes remaining.
GPU0 uses18251MiB(17.82GiB), sampled utilization0/42/39%; native stepping/synchronization
leaves the GPU intermittent. Preserve frozen probe hyperparameters for this
bounded fit rather than mix a throughput experiment into its learning result.
GPU1/2 mixed pretraining is not interrupted. Telemetry is retained in the run.

Parallel blocker work recovers the original s3 corrected-reference pipeline:
`tools/run/recover_native_motion.py` reuses the unchanged original adapter and
converter, restores2170native-contact frames before nearest-index conversion
to543frames, then applies the right-wrist/object inverse-X90 translation
correction. No distance-contact mode, interpolation, extra retarget passes or
external writes. Bounded single-sequence CPU conversion<=900s/2GiB because
GPU0 runs the parent,1/2 run PointWorld and3-7 belong to other jobs. Source,
URDF/model/dependency hashes are recorded. Both[543,598] tensors must be
finite, right-wrist/object and object/table relative errors<=1e-5m. This is
reference data reconstruction, not robot rollout or grasp evidence.
Erratum: original s1 right-contact-1 labels come from geometric noncontact
classification; they are not proof of missing annotation. The native reward
only activates reference-contact supervision for values>0.01. The original
staging metadata's word `unknown` is retained as frozen historical metadata;
the numerical labels are copied unchanged and this interpretation supersedes it.

The s3 reconstruction completes23.17s onCPU at commite98d0e2. Corrected
tensorSHAf8ce89f6df8bdc24a38d8f2e41df65402d4deb78ffaebca974abddffb36bdbf4;
baselineSHA658c5d2ff591a4303ffe440d3e8a95125003c5234537e5383c84dbfb9b4a708d.
Wrist/object relative maxerror2.384e-7m, object/table error1.192e-7m,
contact values0/1,left-active0,right-active2911. All historical geometric
invariants pass. It is still a new conversion, not proof of historical byte
identity. Artifacts: `outputs/consequence-evaluator/s3-corrected-rebuild-20261007-r1/`.

The parent endpoint qualification is explicitly queued by
`tools/run/queue_parent_qualification.py`: wait only for this task's owned
successful parent supervisor<=3600s; then run the frozen endpoint once at
seed290/GPU0,<=900s/1GiB. Preserve qualified and failed episodes alike.
Queueing does not start six-expert expansion or select among checkpoints.

## Limitations / future evidence

New motion references differ in provenance from the deleted corrected outputs;
CPU shape/filter checks do not verify contact geometry or physics replay.
The first parent is a bounded attempt, not guaranteed recreation of an old
checkpoint. The six-expert route, continuous phase-targeted perturbations,
reliable absolute progress, local preferences and matched E0/Eoracle training
still need actual execution and auditing. Later WM/proposal work remains
conditional on useful oracle headroom.
