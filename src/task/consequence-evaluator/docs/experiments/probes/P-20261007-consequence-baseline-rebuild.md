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
status: PROMISING
run_id: baseline-rebuild-original-20261007-r1
---

# Rebuild a self-trained rollout substrate after asset loss

Result: Reset-repaired original-s1 Cm-off parent completes200epochs;50/64full-frame0 episodes meet the fixed45-frame/no-later-drop gate. The six experts and evaluator dataset are not yet complete.
Decision: Proceed to a bounded20epoch s3 transfer from this self-trained parent, then independently qualify its fixed endpoint before expanding other experts.

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

## Original-s1 parent qualification and bounded s3 decision

The original-input parent completes200epochs in1123.50s at training commit
567c9091c642f3b6c21c36b830d0a171437231ca. The queue (smoke plus parent) takes
1168.33s. EndpointSHA256:
50a028eba1b4aaaf66ea5d669e14c8b2e31847b8820bd23938d694fb2c26f330.
Qualification at commit053e8dc finishes64full-frame0 first episodes of431steps;
50/64(78.125%) satisfy >=3cm/contact-proxy hold for45consecutive frames with
no subsequent drop. Independently counted per-episode flags agree with50.
First-step displacement remains approximately1.43mm. The weaker native5-frame
metric is59/64; it is not the qualification criterion. These positive Probe
observations permit data preparation; they do not restore the old Validation
or demonstrate Cm utility. Qualification exports remain diagnostic only,
training_allowed=false because native PD actions can be mutated.

Evidence: `baseline-rebuild-original-20261007-r1/parent_s1/run_manifest.json`,
`qualification-s290/{qualification.json,run_manifest.json,native-results.json,transitions.pt}`
and `qualification-s290-queue.json`, all under `outputs/consequence-evaluator/`.
No missing CmLite weight is needed to proceed with this qualifying Cm-off
substrate; the historical exact recipe remains unreproduced.

Decision Note: Test whether the recovered s3 motion can inherit this useful
parent before paying for the remaining five experts. Cheapest useful fit is
8env/two resumed epochs (200→202, debug17), then a separate64env20epoch fit
(200→220, seed289), both loading the same frozen parent200, not smoke202.
Keep h32/mb256/mini6/rewards2/10/5 and LR1e-5; restore model/RMS/optimizer/
epoch state, explicitly override LR. Preserve the already completed anneal
40→80 (scale0 at epoch200), so no implicit180→220backtrack enters the recipe.
OneGPU0, <=300s smoke/900s fit/5GiB, then <=900s/1GiB fixed64frame0 qualification
with seed290 and the same8/64gate. GPU1/2 mixed pretraining remains independent.
Stop on nonfinite, source/input drift, GPU conflict, timeout or output cap.
Positive permits the next expert recipe; negative audits s3 control/contact
and transfer learning before further spending. No new claim or external write.

The queue verifies completed owned Cm-off checkpoint ancestry back to random
initialization and freezes every ancestral checkpoint/manifest. Qualification
uses the same ancestry check, so transferred self-trained actors are accepted
without accepting an official imported actor. Actual runtime identity is
recorded after implementation checks and commit. Inputs are the previously
audited `baseline-transfer-inputs-20261007-r1/`; output is the fresh
`baseline-transfer-s3-20261007-r1/`. Other six-expert fits remain pending.

## First s3 transfer result and bounded continuation

Run `baseline-transfer-s3-20261007-r1`, commitdf77ed0, queue1278541/GPU0,
8env2epoch smoke completes37.12s, then64env200→220 completes145.10s.
Both load parent200 independently. The actual native log confirms LR1e-5,
restored epoch200 and anneal40→80scale0 at epochs210/220. EndpointSHA256:
16ea1c2eb5b9f87d8434edbaf6ec31fac90b50ab69a974c112e52d7c752859cb.
Qualification seed290 completes64first-frame0 episodes,542steps each:
2/64reliably qualify;4ever reach45frames and2later drop. Weaker5frame
lift is16/64, mean maxlift76.73mm/contactfraction.1967/mean maxhold14.61frames.
Maximum first-step displacement1.717mm; no reset teleport. This endpoint fails
the8/64gate. Keep the positive parent result scoped to s1; the s3 result is
UNCLEAR and does not permit five more experts or evaluator collection.

Implementation check: ancestry/RMS/optimizer/absolute epoch/load hashes and
single s3 input are correct; both engineering and actual execution use the
specified LR and completed anneal. Training contact rises from~5% at early
updates to~26% by epoch213 and episodic reward rises4.74→55.5, but neither
proves stable success. Full-start qualification has actual lifts and holds,
so the immediate question is whether additional fitting develops retention.

Decision Note: one bounded40epoch continuation220→260, from the frozen new
s3 endpoint, same seed289/LR1e-5/input/config/rewards/anneal. Separate8env2epoch
smoke220→222 is discarded;64env fit reloads220. Then qualify fixed260 once,
seed290/same64frame0/45frame/no-later-drop/8success gate. <=300s smoke/900s fit/
5GiB plus<=900s/1GiB evaluation, oneGPU0. This is explicitly a continuation
of an unqualified actor, not authorization to expand experts; its exact motion,
environment and learning rate must match the failed run. Stop on drift,
nonfinite, GPU conflict or budget. Positive permits the next recipe; negative
inspect measured control/contact/drop timing and curriculum before further
epochs, rather than automatically appending more fitting. Artifacts are the
fresh `baseline-transfer-s3-20261007-r2/` under task outputs.

Independent CPU blocker preparation also recovers the cup corrected tensor
in27.83s at053b7ff:934frames, right-wrist/object relativeerror2.98e-7m,
table-relativeerror1.19e-7m, left-active0/rawcontact0/1. TensorSHA256:
94aab2e4c88f2f7bcafd95fda2a2b15ee7c3ea999b27120f39bd6a0a0bba85f1.
This is a reference input only; no cup policy or rollout is trained.

Remaining reference preparation is an independent Blocker, not expert fit:
ten original conversions, same source/contact/retarget/alignment contracts,
one sequentialCPU batch<=600s/4GiB and<=180s per sequence, no GPU/model fit.
The frozen helper and each child record raw/native/converter/dependency hashes;
stop the batch on any child failure, drift, deadline or output cap. Output:
`expert-reference-recovery-20261007-r1/`, entry `tools/run/recover_expert_references.py`.
Geometry checks and actual output counts must pass before any new expert uses
these references. Input preparation does not bypass the s3 readiness gate.

## Limitations / future evidence

New motion references differ in provenance from the deleted corrected outputs;
CPU shape/filter checks do not verify contact geometry or physics replay.
The first parent is a bounded attempt, not guaranteed recreation of an old
checkpoint. The six-expert route, continuous phase-targeted perturbations,
reliable absolute progress, local preferences and matched E0/Eoracle training
still need actual execution and auditing. Later WM/proposal work remains
conditional on useful oracle headroom.

## Fixed s3 endpoint260 and recovered references

Run baseline-transfer-s3-20261007-r2, commit507091e, completes8env220→222
smoke35.28s and independent64env220→260 fit254.911s (queue311.837s).
Qualification-s290 completes54.103s:36/64 (56.25%) retain the45frame/no-later-drop
operational criterion; mean maximum lift540.077mm and contact fraction.614132.
The fixed8/64gate passes: PROMISING for this s3 data substrate. EndpointSHA256
8882fabd2d83c56312ca90e3b27ea145f628dc1ddcedafbcc751303a27d52871.
These native net-force traces remain training_allowed=false and lack the new
ref2 sampled-proximity corroboration; they are not evaluator labels or formal
Validation. Parent50/64 and airplane_base36/64 do not constitute six experts.

The sequential remaining-reference batch at50ad2d2 completes270.591s: ten
references,481–1062frames, all left-contact0, right native-contact0/1, relative
geometry maximum error<=3.576e-7m. Manifest is
outputs/consequence-evaluator/expert-reference-recovery-20261007-r1/run_manifest.json.
Together with recovered s3 and cup, all12selected original-style inputs exist.
Five further experts remain untrained. GPU0 is currently occupied by another
user, and the user cannot coordinate its release; no foreign jobs are stopped.
While waiting, ref2 repairs plans/three arms/contact/pairing, ref3 separates
PointWorld data-source semantics. Next expert recipes need actual multi-object
approach geometry rather than an implicitly fixed airplane surface.

## GPU0 空闲后的下一专家 Decision Note

23:26 GPU0外部任务自然退出，用户已授权双管齐下，root继续基线重建。
当前关键证据为新parent50/64、s3endpoint260为36/64，corrected12输入
全部恢复。下一步先duck单motion20epoch（260→280），同source/RMS/
optimizer/LR1e-5、原anneal40→80；先8env2epoch丢弃smoke，再64env完整
正式端点并seed290固定64frame0资格。修复非airplane approach几何选择，
保留原采样/奖励公式，不把airplane mesh用于duck。mixed12/train5/
balanced5/cup输入和有界入口同时准备，未因输入就绪当作已训练。

这区分已有s3策略是否能低成本产生另一物体的稳定数据。单GPU0，
smoke<=300s/fit<=900s/5GiB、资格<=900s/1GiB；与PointWorld1/2合计
3GPU，仍受4GPU/300GB/free20GiB限制。GPU冲突、source/input漂移、
非有限、超时或cap立即只停止owned进程。若duck达到8/64，继续下一
角色独立有界20epoch；否则先核对该物体reset/控制/接触，不自动堆步数
或用单权重冒充六专家。五角色最多依次短fit，不立即领取新的长训练预算。
不改变Mission/claim、不改PointWorld活跃输入和源码、不停止他人进程。

## Duck r1 smoke asset failure and bounded retry

At325517b duck r1 failed before the first PPO update: native URDFs existed,
but the owned legacy assets contained only airplane/table object meshes.
Preserve failed smoke/queue logs; this is invalid input readiness, not a
duck learning result. Staging now copies complete owned assets and adds
9missing canonical GRAB object meshes byte-for-byte from recovered inputs.
All canonical/raw/staged hashes agree, all URDF visual/collision origins and
scales are identity, all native mesh dependencies resolve;147assets and
38recovery dependencies are frozen. Two readiness regression tests pass.
Runtime assets symlink is repaired reversibly, previous target saved in
corrected-expert-inputs-20261007-r2/previous-native-assets-link.

Fresh baseline-transfer-duck-20261007-r2 retries the same source260→280,
seed289, LR1e-5 and smoke/fit/qualification budgets from the Decision Note.
No failed checkpoint is used, no PointWorld source/input changes, and no
external assets are modified. Stop on actual geometry/reset issues rather
than treating staging alone as evidence of successful learning.

The fresh duck r2 launcher detected a transient GPU0 PID1449244 and exited
before creating a smoke child; preserve its FAILED record and do not kill
that process. On recheck the PID had naturally exited. Fresh duck r3 at
9294db0 uses the same staged r2inputs/source260/budgets and has entered
SMOKE; qualification-s290 is queued for its successful endpoint only.

## Duck endpoint280: readiness fails, audit before extra fitting

Duck r3 at9294db0 completed20epoch transfer260→280. Fixed seed290
64frame0 qualification:0/64 stable/no-later-drop,8/64 lift>=3cm,5/64
maximum elevated proxy-held runs>=45frames (maximum303), mean max lift
33.57mm and contact fraction.31864. First-step displacement<=2.869mm,
no old reset teleport. This is UNCLEAR substrate readiness and not an
evaluator/world-model negative result. Do not automatically append epochs.

Blocker audit:8env×128frozen duck-policy steps with actual duck URDF,
measured11keypoints and independently sampled hand/object gaps; same
native_reset_probe budget<=180s/1MiB onGPU0. Compare force-proxy/near
agreement and inspect full savedqualification drop/hold traces via CPU
statistics. This distinguishes broken geometry/contact/reset from a
policy that can lift but cannot retain. No PPO, newlabels or reward change.
If wiring passes, choose the next independent expert from qualifieds3260,
and record a separate bounded recipe decision; do not use duck280 as
qualified ancestry. Preserve all failed staging/launch/qualification outputs.

## Recovery recipe Decision Note after duck audit

Question: does20epoch duck failure justify closing specialist recovery?
Evidence: duck reset/actual geometry wiring passes and5/64episodes already
hold45frames but drop later. The old successful duck and cup recipes used
80epochs from s3e260, not20 (P-20260924-duck-specialist and
P-20260925-cup-specialist). Current corrected input/newparent are different
and the old results cannot be inherited. A20epoch gate alone does not
implement the historical specialist training budget.

Root chooses one fresh duck260→340 run from the qualified s3260 source,
not the failed duck endpoint. SameLR1e-5/reward2/10/5/anneal40→80/seed289,
8env2epoch discarded smoke,64env80epoch fit, fixed64frame0seed290 gate.
Retain the original <=300s smoke/900s fit/5GiB and<=900s/1GiB qualification
caps; observed20epoch runtime fits the80epoch ceiling. This separates
insufficient transfer duration from inability to retain, without reward
or heldout-tuned threshold changes. Stop on drift/nonfinite/GPU conflict/
time/cap. If negative, preserve and inspect reference/control opportunities;
no automatic extra epochs beyond340. If positive, freeze as newduck role.

The queue's initial-role caps now reflect separate recipes: mixed12<=40,
train5<=20, balanced5<=40, duck/cup<=80, airplane_base<=20. All initial
sources still need qualified owned random-self-trained ancestry. The
explicit unqualified<=40continuation remains limited to unchangeds3.
This does not pre-authorize all runs or override per-run Decision Notes.
GPU0 was reoccupied by a foreign zyc job; check another idle GPU afresh
with at most3total owned GPUs, never stop that job.

Fresh80epoch duck queue started at1386d51 onGPU0 after it again became
free: outputs/consequence-evaluator/baseline-transfer-duck-20261008-r1/,
queuePID1496208/fitworker1497950, qualifier queue1497869. Discarded
2epochsmoke passes and full64env fit is live at274; this is a verified
wait on the actual owned process, not a completed specialist. The endpoint
qualification will execute only after successful terminal training.
