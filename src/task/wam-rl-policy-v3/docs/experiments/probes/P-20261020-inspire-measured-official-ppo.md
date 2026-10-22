---
schema: ref2dex.probe.v2
probe_id: P-20261020-inspire-measured-official-ppo
experiment_id: P-20261020-inspire-measured-official-ppo
date: 2026-10-20
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 14
seed_pool: probe
seeds: [246, 247, 248]
decision_changed_if_positive: continue the measured goal WAM policy improvement route
decision_changed_if_negative: diagnose measured training dynamics before adding scale or changing the method
status: STOPPED
run_id: inspire-measured-official-20261020-r3
---

2026-10-22 12:4x repair scope: old prefix construction changed473/501 terminal modes; restored complete
declarations retain all501 tasks (490RELEASE/11HOLD), while true UNKNOWN is still refused. Fixing RELEASE
timing changes same-archive full scoring0→77/501 (safe prefix0→9); old53,347 completed PPO episodes
cannot be retroactively rescored without their missing physical traces. Official PPO remains v4, v6 UNWIRED.
Voxel now protects admission/model inputs and no longer directly manufactures task termination; new
teacher/source-PD complete-task qualification still fails0/6. See
[reviewable fixes and remaining blockers](../../../../ptv3-shared-wam-v6/docs/audit/fix-review-20261022.md).
Old run counters below are preserved historical observations, not repaired-run results.

2026-10-21 user-authorized resource switch: r3 normal STOP at1575; all3 ranks
STOPPED, six workers and supervisor exited. Final rank0 checkpoint61,158,850bytes
CPU-loaded with finite model/optimizer and nonempty optimizer state. Training
count53347 completed episodes, full-task successes0; this is not deterministic
evaluation or proof of zero grasp/lift rate. Source mismatches/sync errors0.
Preserved final checkpoint and all evidence; no PPO restart. GPU4 now supports
local WAM adaptation; the subsequent explicit multi-GPU request assigns all three4/5/6 to WAM. Switch verification is in
outputs/ptv3-shared-wam-v6/local-subset-pretrain-20261021-r2/process.json.

User requests starting PPO now onGPU4/5/6 concurrently with independent
trajectory visualization onGPU7. This is an explicit resource allocation for
this run, not a general change to CAMPAIGN. Main branch, frozen code manifests.

Decision: can the actually measured goal+WAM observation support policy
improvement after latest BC? The user requests enough training scale, so a
two-env wiring check is only a Blocker check, not a substitute for training.
Then three ranks with512 environments each,1536 total, matching the previous
Shadow six-rank256 total and global12288 transitions per eight-step rollout.
Global minibatch3072 after four minibatches; five epochs. Use pinned official
UniDexGrasp2 PPO collector/storage/GAE/update, actual-distribution KL correction,
global advantage statistics and gradient averaging already used by Shadow.
No DExplore PPO, reward or hand-reference actor input is used.

Source: corrected r3 teacher-rollout admission, all501 TRAIN sequences/50 objects.
Tasks/events derive from teacher object rollouts. BC source is corrected-data
Scheduled686 DEV-best(step0), retaining learned Current421 goal columns; this
is the declared DEV selection, not post-hoc endpoint selection. Added tau
weights are zero initially but trainable during PPO, with actual WAM80000 input.
CAD encoder is frozen at the BC parent so rollout features do not drift between
sampling and update. Actor/critic MLP1024/1024/512/512 ELU; actor BC initialized,
critic fresh as Shadow; native12 actions/30Hz remain Inspire-specific.

Official optimizer settings: lr3e-4 adaptive desiredKL.016, clip.2, gamma.96,
lambda.95, entropy0, gradient norm1, initial noise_std.8 (preserve upstream
squared-scale sampling convention). Reward is existing causal MeasuredRewardTracker
with the same gamma. Full TaskController task, not imitation of hand references.

Reset contract: execute each action in its observed episode, collect terminal
reward/done, explicitly reset only completed native environments, clear only
their task/reward/history/previous-action state, and return the new observation
before sampling the next action. Actual native clock/episode IDs are checked.
Two-env probe forces different environments to terminate atsteps2/4, then real
official PPO updates; training never uses this engineering forced termination.
Terminal GAE does not bootstrap from a reset state.

Run limits:10k PPO iterations or48h, own STOP at update boundary,20GiB disk
floor, bounded checkpoint retention (last3 plus500 milestones). Rank0 only
saves checkpoint; other ranks log progress. Physics and policy subprocesses
share their assigned GPU; no unknown processes killed. All native motion
sampling is restricted to the admitted TRAIN membership. Runtime graspenv
Torch2.4.1+cu121 policy and pinned dexplore_v120_train Torch2.0.1+cu118 physics.
Log first iteration throughput/memory/ETA and actual multi-episode reset results.
Training success is full TaskController completion, kept distinct from lift
proxies; no WAM gain or held-out generalization claim from this run alone.

## Startup failure and backend isolation

The r1 multi-rank GPU PhysX run failed before its first PPO update with
`convexPlaneNphase_Kernel` illegal GPU access; its logs remain in the r1 output.
The two-env asynchronous-reset engineering test passed2 iterations/40 optimizer
steps, zero source-episode mismatches and zero behavior log-prob replay error.
Native screens r1/r3 include successful full50-object512-env single-rank two-update
runs, but also failed256-env and partitioned512-env cases. This is not evidence
of a simple environment-count/OOM boundary. Explicit synchronization did not
eliminate the error; do not claim the GPU backend fixed.

r2 tests CPU PhysX (8 physics threads/rank), with the same1536 environments,
TRAIN membership, physical parameters, native actions, reward and official PPO
settings. WAM and PPO stay on the requested GPUs4/5/6. Upstream hardcoded CUDA
motion caches are moved to the CPU sim tensor device after creation. This is an
explicit physical-backend change, not a strict GPU-PhysX reproduction. Test the
same asymmetric indexed reset before beginning its three-rank training.

WAM batch256/shared TF32 is checked against FP32 for every full trajectory in
the first inference batch; failure above1mm aborts. A full512-env single-rank
screen measured0.377mm maximum,30.38s collection and0.17s optimization per
iteration, so physics-only timing is not total training throughput. Backend and
inference settings now form part of the strict resume identity. r2 manifests
record the frozen code revision and actual progress; startup or environment
creation alone is never counted as a PPO update.

CPU PhysX full50-object512-env screen r5 also completed2 iterations/40 optimizer
steps,31.31s collection and0.20s update on its second iteration, source mismatches0.
CPU two-env r2 smoke completed the same asynchronous reset fixture with0 source
mismatches and0 log-prob replay error. These verify startup and handoff, not policy
improvement or physical equivalence between CPU/GPU solvers.

## Three-rank startup accepted

r2 frozen code ed62f767 passed the CPU asynchronous-reset smoke and began actual
three-rank updates. At acceptance each rank completed at least2 PPO iterations /
40 optimizer steps; parameter synchronization error0, source-episode mismatch0,
behavior log-prob replay error0. All ranks bind501 TRAIN motions/50 objects.
One natural workspace terminal was reset in this initial interval; this is not
a task-success result. WAM first-batch full-trajectory TF32 errors were
0.330/0.285/0.548mm (ranks0/1/2), below1mm.

Rank0 saved official-iteration-1.pt (61,158,108 bytes), completely loaded on CPU
with finite model/CAD tensors and optimizer state. Second-iteration throughput
was approximately32s per global12288-transition iteration.10k would take about
89h at that rate, exceeding the explicit48h run budget; the budget stops/saves
first, and a subsequent run may resume. No convergence or gain claim yet.

Startup resources are recorded separately in the output's startup-resources.json;
progress/curve files belong to each rank, and bounded checkpoint writing is
rank0 only. All154 Task tests pass. Global tools/verify.py still reports only
pre-existing missing metadata in six trajectory-policy cards; do not describe
the whole repository verification as passing.

Thirty1Hz startup samples measured mean GPU utilization32.6/36.9/46.9% on
physical GPUs4/5/6, peak total allocation11,480MiB per card. Utilization remains
intermittent; the measured causal state/task packing and WAM calls dominate the
collector, so high GPU utilization is not yet achieved. Four synchronized
iterations/80 optimizer steps per rank were reached during this sampling.

## User-requested source/runtime audit and GPU fault recheck

Decision/Blocker diagnostic: distinguish a recent Torch/runtime change from a
native tensor/reset or contact-scene failure before accepting a GPU-backend fix.
Read-only official NVlabs/dexplore main sources were fetched and compared under
`tmp/inspire-physx-diagnosis-20261020-r1/upstream/`; no upstream files overwritten.
Primary sources: [official README](https://github.com/NVlabs/dexplore/blob/main/README.md),
[requirements](https://github.com/NVlabs/dexplore/blob/main/requirement.txt),
[Inspire task](https://github.com/NVlabs/dexplore/blob/main/dexplore/env/tasks/dexplore_inspire.py).
Requirements specify torch>=1.8.1, rl-games1.1.4; README example uses CUDA11.6,
not a unique exact Torch version. Our native environment differs from that
installation example, but did not change between old mixed6/r4 and this run:
both manifests name dexplore_v120_train/bin/python, Torch2.0.1+cu118. Live import
also verifies Python3.8.10, CUDA11.8, NumPy1.23.5, rl-games1.1.4. WAM/PPO runs
separately in graspenv/Torch2.4.1+cu121, also the same as old mixed6/r4.

The vendor is official-derived, not an unmodified official experiment. Local
changes include complete motion export, object-balanced motion sampling and
per-motion table resets, plus a body-to-collision-shape ownership correction in
Inspire. The official PD mapping is unchanged. The host additionally replaces
reset observations with measured/FK caches, coalesces indexed setters, exposes
measured force/state, and bypasses DExplore's policy/reward/termination. Those
adaptations cannot be excluded as fault triggers simply because PhysX reports
the eventual error. The reset-submission implementation and vendor physics
files are unchanged from old dd694523; new inputs/pool/env-count are different.
Old mixed6/r4 actually stopped at cumulative iteration972 with5072 native steps
per rank, beyond the earlier card's interim360 report.

Bounded reproducible diagnostic entry:
`tools/audit/probe_inspire_physics.py`; fresh output, seed246 engineering repeats,
no WAM or learned policy, fixed canonical commands, actual native clocks checked.
Official-PPO check separately uses frozen120279df and the same BC/WAM inputs.
GPU7 was initially free and subsequently shared with monica's new task within
available memory; no other task changed/killed. Short concurrent native probes
on4/5/6 ran alongside the unchanged active three-rank training and exited.

All12 cases in `outputs/wam-rl-policy-v3/inspire-physx-provenance-20261020-r1/result-summary.json`
completed without the historical crash: stamp16; known subset16 objects/512 envs
with zero/noise actions; complete50-object512-env actual official PPO2 updates /
40 optimizer steps; CUDA memcheck16 envs/two controls (ERROR SUMMARY:0 errors);
GPU PhysX with CPU tensors512 envs; two ordinary GPU-tensor512-env repeats;
three simultaneous native512-env tests on4/5/6; and512 envs/128 controls with32
asymmetric partial resets. The stress test checks unchanged live episode IDs
and clocks after each selected reset. Native PD/source checks remain enabled.

GPU-tensor repeats have identical recorded initial sequences and q/dq/physical/
object/hand/tick arrays. GPU PhysX CPU-tensor initial q/dq/object/tick also match;
physical248 max difference1.73e-6 and hand2.38e-7m. This is only initial-state
adapter evidence, not full-trajectory physical equivalence. The new explicit
`--physics-backend gpu-cpu` separates GPU physics from CPU state exchange per
installed Isaac Gym programming/tensors.html; it is a diagnostic option only.

Conclusion: no unique cause identified and no claim the GPU backend is fixed.
These bounded checks do not reproduce the prior intermittent first-step crash,
and do not establish long multi-rank GPU-PhysX training stability. Do not attribute
it to a recent Torch upgrade or generic PPO divergence. Native adaptation and
startup/contact conditions remain candidates. The active CPU-PhysX three-rank
training remains frozen and unchanged; no silent backend migration performed.
All154 existing Task tests pass on the audited changes. Diagnostic artifacts
are bounded and initial snapshots are small; no new neural caches or videos.

## User pause and same-three-GPU diagnostic migration

User explicitly requests pausing production and testing the GPU path again.
r2 received its owned STOP and all ranks exited normally atiteration215, after
4300 local optimizer steps. Source61,158,368-byte checkpoint is fully CPU loaded,
finite, with21 optimizer parameter states; both model and optimizer are retained.
Supervisor/root now records USER_PAUSED, all six owned workers exited. The
production run remains paused unless the user asks to resume.

The bounded new run inspire-measured-gpu-recheck-20261020-r1 freezesfa10fcd5,
retains the source checkpoint by hardlink underinputs, and runsGPU4/5/6×512 with
all501 TRAIN/50 objects, the sameWAM/BC and official optimizer settings. Restore
model+optimizer from215 and explicitly migrate CPU→GPU PhysX with GPU tensors;
16 additional updates (through231) or1200s, then exit. Shared TF32/batch256,
packed extended state and setter synchronization remain explicit. This is a
backend migration diagnostic, not a matched physical-trajectory comparison.

The new --migrate-physics-backend option requires --resume. A unit regression
ensures ordinary resume rejects a backend change and explicit migration still
rejects WAM/env-count/world-size or any other identity change. The migration and
source checkpoint are saved with manifests and output checkpoints. Three
focused reset/identity regressions pass. Acceptance requires all three workers
complete16 updates, actual multi-episode reset checks, source mismatches0 and
rank parameter synchronization0. Preserve failure logs ifGPU PhysX recurs; do
not fall back silently or interpret environment creation as success.

The frozen resumed runner initially records parameter synchronization only at
cumulative multiples of20, so first-resume216 has no measured sync field.
Cumulative220 checks all three ranks at0; behavior log-prob replay and source
mismatches also0,48 natural terminals/resets exercised by that point. Future
code46015eb3 corrects the first-resumed-update logging condition; the active
fa10fcd5 snapshot is not hot-patched. This observability correction does not
change the policy update or native physics.

### GPU recheck result and final paused state

The three-rank test completed all16 updates,215→231,320 optimizer steps per
rank,128 real native controls per rank,global196608 transitions. All six
workers and supervisor exited normally;GPU4/5/6 were released. The production
run remains user-paused atCPU215; theGPU231 test checkpoint is separately
retained. No automatic long-training resume took place.

Across the test335 natural terminals/reset transitions occurred, all workspace
failures; complete task successes0 within this128-control window. This is not
a full-episode policy success-rate evaluation. Native PD error, source-episode
mismatch, behavior log-prob replay and measured rank synchronization errors0.
Indexed reset setter submissions66/70/71 across128 controls per rank. Final
checkpoint official-iteration-231.pt is61,158,880 bytes, fully CPU loaded,
model and optimizer tensors finite, every Adam state advanced320 steps and
frozen CAD encoder is unchanged. Identity differs fromCPU215 only in the
explicitly authorized physical backend.

Steady rank0 iteration time (excluding first2) is 36.75s.103 samples every5s
measuredGPU utilization37.55/38.67/33.25% and peak total15,240MiB per card.
The GPU path is operational in this bounded test; this does not identify or
prove elimination of the prior intermittent first-step failure. Raw acceptance,
resources and per-rank curves are in the recheck output. Current status PAUSED
refers to the research training, while the diagnostic itself is COMPLETED.

## User-authorized GPU long-training resume

User explicitly asks to start training after the passed GPU recheck, revoking
the pause. New production run inspire-measured-official-20261020-r3 freezes
3a0779f6 and resumes the verified GPU231 model AND optimizer, hardlinked under
itsinputs. This is ordinary same-backend resume, no migration override. Source
GPU identity/WAM80000/admission/geometry/frozen CAD and official PPO recipe remain
unchanged. The first-resumed-update parameter sync check is included.

GPU4/5/6,512 environments each,1536 global,12288 global rollout/3072 minibatch;
all501 admitted TRAIN sequences/50 objects. NativeGPU PhysX/GPU tensors,
Torch2.0.1+cu118; WAM/PPO graspenv/Torch2.4.1+cu121, sharedTF32/batch256,
packed extended state and explicit setter synchronization. No native reference
actor/reward/termination introduced. Rollout seeds246/247/248, fresh episodes;
iteration and optimizer state continue, live episode counters restart per run.

Target cumulative10000 (9769 additional updates), bounded by the original48h
training budget:7801.46s CPU r2 +683.64s GPU recheck already used; remaining
164314s (about45.64h), not a reset48h allowance. Save/stop at budget, owned STOP
or20GiB disk floor. Rank0 last3 +500 milestones retained, no source checkpoints
overwritten. Any worker failure closes all owned peers; preserve logs and last
completed checkpoint, with no silent CPU fallback. The previous intermittent
GPU startup issue remains a recorded limitation even though the16-update
recheck passed. Verify actual resumed updates/resources before handoff.

### GPU long-training startup accepted

r3 reached at least235 on all three ranks,80 new optimizer steps each. The
first resumed update232 now directly measured rank synchronization0; source
mismatch and behavior log-prob replay errors0. SavedGPU232 was fully CPU loaded:
identity exactly equals sourceGPU231, no migration override, all model/optimizer
tensors finite, each Adam state advances20, frozenCAD unchanged. Source231 is
retained by hardlink; completed output checkpoints follow bounded retention.
Evidence: resume-verification.json and per-rank progress/curve files under r3.

Current round approximately34s; startup60s/12 samples at5s measured mean GPU
utilization37.8/37.9/43.9%, total peak15,240MiB each. WAM and CPU task packing
remain the collector's major cost. All six workers and supervisor are alive;
this is a long-running handoff, not experiment completion, convergence or a
method-gain claim. The original failed startup remains in limitations.

## 2026-10-21 live status check

At09:54 Asia/Shanghai, r3 ranks all reached cumulative1520,25780 local new
optimizer steps, approximately13.1h elapsed; supervisor and six workers alive.
No new PhysX failure. Native PD/source and policy log-prob/sync errors remain0.
Live snapshot: r3/status-20261021T015435Z.json.

Global ended training episodes51322, full TaskController successes0, workspace
failures26952 (52.5%). These are stochastic online training counters, not a
fixed-panel evaluation. Full task success is not the same as grasp/lift; those
per-episode quantities are computed in the env but not aggregated by this
runner. Other failure reasons are not individually counted in current logs.
The last100 logged iterations average35.93s; value loss near0.01 alone provides
no convergence evidence. Prioritize stage/grasp/reward diagnostics before
interpreting longer training as progress; this status check itself did not
alter or stop the authorized training.

Thirty seconds sampled at2s: meanGPU utilization49.3/33.1/46.8%, total peaks
15240/15240/15306MiB. Remaining original training allowance32.56h; local disk
222GiB free. Rank0 keeps500/1000/1500 plus the latest3 checkpoints, bounded as
specified. Latest progress and checkpoint iteration advance normally.

## 2026-10-21 reward logging audit and historical comparison

At10:24 Asia/Shanghai, all three ranks reached1570, approximately13.59h of
this run. Supervisor and all six workers are alive. Ended training episodes
53174, full TaskController successes0, workspace failures27792 (52.27%).
Rank0 last100 iterations average36.31s; remaining original allowance32.05h,
disk free about221.5GiB. Measured source mismatch/log-prob/synchronization
errors remain0. Snapshot: outputs/wam-rl-policy-v3/inspire-measured-official-20261020-r3/status-20261021T022443Z.json.

Confirmed logging defect: inspire_measured_official_ppo.py replaces ppo.log
without persisting the official collector's mean_reward/rewbuffer or the
environment's episode_results.return_sum. All three curve.jsonl streams lack
reward/return fields, and all three TensorBoard event files are88bytes only.
Reward still feeds official PPO storage and optimization; this finding does
not mean reward was absent from training. Historical training reward cannot
be reconstructed from saved losses or model/optimizer checkpoints. Logging
has NOT been repaired in the frozen active run. No reward trend or convergence
claim is supported by the existing value-loss curve.

Historical comparison, checked against committed cards and actual artifacts:

- Current BC source is Scheduled686 r4 DEV-best(step0), preserving the learned
  Current421 goal parent but with initially zero tau/valid columns. On the fixed
  four TRAIN motions/300 controls, parent and step0 each had2/4 lift5cm30,
  followed by drops; full task completion was0/4 within the bounded window.
  The trained50k endpoint had1/4 lift5cm30 and worse DEV command MAE
  (0.05317855 versus0.04012461). See [BC card](P-20261020-inspire-scheduled686-bc.md)
  and outputs/wam-rl-policy-v3/inspire-wam-goal-bc-20261020-r4/comparison.json.
- Early DExplore shared-controller testing used WAM-v2, not this v4 executor.
  Three fixed sequences had teacher holding but student held0; one airplane
  diagnostic recovered holding with teacher wrist/student fingers, while
  student wrist/teacher fingers still failed. Teacher-assisted holding is not
  autonomous student success. See [shared native card](../../../../ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-shared-native-grasp.md)
  and [component diagnosis](../../../../ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-inspire-component-oracle.md).
- WAM-v3's frozen nine-source recipe includes OakInk2/GRAB/ARCTIC/UniDexGrasp/
  TACO/HOT3D/DexYCB/H2O/HOI4D; it does not include DExplore. This source check
  is separate from downstream DExplore experiments. The concrete recipe is
  outputs/ptv3-shared-wam-v3/nine-source-strict-quality-20261015-r1/recipe.json.
- PPO engine/capacity reuse does not imply identical tasks or reward: current
  Inspire requires the TaskController's ordered SE3 path and terminal behavior,
  using measured TaskReward rather than the Shadow native reward. At saved
  PPO1500, actual Gaussian action standard deviations exp(2*log_std) span
  0.2321–0.4064. Noise-induced failures remain a hypothesis, not a diagnosis.

Next Decision Probe should distinguish BC weakness, PPO policy degradation,
exploration-induced failures and task-stage/reward bottlenecks. The cheapest
useful comparison is a fixed initial-state panel for the BC parent and saved
PPO milestones, deterministic means first, reporting approach/contact/lift/
hold/path progress and termination reasons separately; compare sampled actions
only if needed to isolate exploration. Restore reward/episode telemetry for
future training records. No such new evaluation, logging deployment, reward
change or training restart was performed during this audit/commit request.

## 2026-10-22 负结果 CPU 审计：结论范围修订

独立读取冻结3a0779f6981ee01ffb8e637abb9b7b3fea2437f3入口：导入V4Runtime，由BC config加载ptv3-shared-wam-v4/step80000-model.pt，严格v4 schema；本运行不是WAM v6 PPO。rank最终manifest STOPPED1575，53347结束训练回合/完整任务成功0/27869 workspace失败是一个同步模型的训练累计，不是53347条独立评估、不是抓取率为0；多个rank也不是多个独立训练seed。顶层旧RUNNING为陈旧状态。保留v4执行桥未观察完整任务完成，不能归因v6、解码质量或缺C。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。

## 2026-10-22 12:3x 修复与危害量化（不倒灌旧运行）

UNKNOWN显式拒绝后的准入为437/501，64序列完整记录且原标签不改；native/policy同集及握手、
TaskScheduler前置断言已加回归，旧501分布checkpoint不能默默恢复为同一次437分布实验。
新增有界episode日志及reward/phase/control/return统计，旧return仍无法回补。官方PPO仍v4，
v6未接线。本轮未重新训练或宣称修复后成功率。
最终UNKNOWN ended集合界6212–9314，voxel27869，排二者后已发生子集0/N且N16164–25478；
更新预算16515072，不能把回合份额当预算损耗。完整影响表与测试见
[修复报告](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-fixes-20261022.md)。
