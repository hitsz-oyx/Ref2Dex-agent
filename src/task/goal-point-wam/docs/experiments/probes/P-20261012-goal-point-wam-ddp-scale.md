---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ddp-scale
experiment_id: P-20261012-goal-point-wam-ddp-scale
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: cc434265
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 14
seed_pool: probe
seeds: [275, 276, 277, 278, 279]
decision_changed_if_positive: retain scaled hand-memory candidate for action-world reliability and downstream policy work
decision_changed_if_negative: inspect learning curves and forward-action/input semantics before additional scaling
status: UNCLEAR
run_id: goal-memory-ddp-20261012-r2
---

# Single-model four-GPU Goal-Point-WAM scale-up

Class: Decision. Serves the Task's learned goal-conditioned hand representation
upstream of Mission's robot/Cm/RL claim. Prior hand+memory goal reduced trajectory
EPE and goal ablations on GRAB/ARCTIC. This run asks whether that candidate keeps
learning with broader exposure and yields a useful forward model. User explicitly
replaces the proposed six matched seed/control launches with ONE model on FOUR
synchronously distributed GPUs. The latest instruction is to train directly
without further utilization tuning. No six-arm
training started; their unused configs were removed. The cancelled panel449 was
generated but no outcome was observed, and is not used by this Probe.

Result: UNCLEAR; four-rank engineering preflight passed and the single scientific training run is RUNNING. No formal causal or multi-seed claim is possible.
Decision: Run the selected architecture at larger global batch and fixed budget;
use final real/static goal and world/action diagnostics to choose the next step.

## Remote four-rank scale run: launch protocol

Decision Note (latest user authorization): retain local main0--3 and reduce
local extra methods to6/7; release4/5. User explicitly asks to try large training
on the remote server. Remote `deepcybo-4-4090-48` has four idle48GiB4090 and a
ready graspenv (torch2.0.1+cu118, spconv/flash-attn/scatter available). Run one
remote DDP model on remote0--3, same selected architecture and seed275; this
is a hardware/larger-batch execution run, not a new control or multi-seed test.
Success means actual four-rank finite training after a full-batch compatibility
preflight. Failure changes deployment/batch/environment choices, not the core
research claim. No tuning grid or additional method comparison is requested.

Dedicated NAS root:
`/mnt/ugreen_nas/storage/oyx_storage/goal-point-wam-remote-20261012-r1/repo/`.
Existing `/home/wbcd/workspace/oyx_ws/Ref2Dex-agent` and NAS datasets are untouched.
Export the committed main snapshot97c07d04, its exact PW vendor files, and only
qualified processed OakInk2/GRAB/ARCTIC inputs. Verify SHA256 for every migrated
file. Relocate source/index paths into a NEW mixed manifest, preserving original
manifest bytes, source/index hashes, weights and normalization-source identity.
Copy the same1536-window panel indices; rebind only its relocated manifest hash
and retain original panel hash. No new Touch/EPIC/EgoDex supervision enters.

Remote recipe: strict PW50000 initialization, hand+memory-goal/world1,
BF16/TF32/AdamW, per-rank256/global1024,10000 updates (10.24M sampled occurrences),
same seed275/draw276/panel277 and LR schedule as the local scale candidate.
The remote run starts fresh from PW; it does not resume the local optimizer.
Different batch/occurrence counts and software versions preclude a matched
hardware or method-quality comparison. Full-batch four-update engineering smoke
must finish with finite losses and equal final parameter hashes before training.
OOM or dependency incompatibility stops launch for diagnosis; no silent batch
reduction. Monitor all rank progress, GPU memory/utilization and estimate ETA.
After passing preflight, preserve its logs, input/result/rank-parameter hashes
and checkpoint SHA256; remove only its reproducible four-update engineering
checkpoint files to reserve space for scientific rolling checkpoints.

Authorized temporary concurrency: local4 main +2 extra, remote4; no use of
local4/5. Remote training max6h/1440GPUmin, isolated entire NAS scope <=5GB
including code, migrated inputs, preflight and rolling checkpoints. These are
additional user-authorized remote resources, counted against the shared300GB
artifact boundary. Pre-staging local outputs+tmp~290--291GB; reserve local
remaining rolling-checkpoint growth and enforce combined298GB save/stop through
the local remote-budget watcher. Remote standalone cap remains active if SSH
disconnects. Stop own jobs only on drift, foreign-GPU conflict, nonfinite/OOM,
disk budget or deadline. Remote training survives SSH disconnection; monitoring
requires the user's existing Windows tunnel to stay available.

## Fixed method and budget

- Fresh strict PW50000 scene/projection transfer; current WAM heads, object
  projection and observed-hand adapter freshly initialized with seed275.
  Current-hand XYZ/history + memory object goal32 paired points, coordinate
  scale.05m, world_weight1. No robot, video, tactile auxiliary or L_goal.
- NCCL DistributedDataParallel,4 ranks; initial segment GPUs0/1/4/5, resumed
  segment GPUs0/1/2/3 after those devices became idle. Each rank
  microbatch128; global batch512, accumulation1,10000 optimizer steps,
  5,120,000 sample occurrences from the existing4,378,478 train-window corpus.
  Source weights5/9,2/9,2/9, existing category stratification, draw seed276.
  Occurrences are partitioned by update/rank; there is no separate model/seed.
- Same initial model on all ranks, confirmed by parameter SHA; rank-specific
  stochastic serialization stream after initialization. DDP averages gradients
  of locally masked WAM losses; this is a mean of rank-local objectives rather
  than a monolithic global-valid-count mean. No old/new matched-batch claim.
  broadcast_buffers=false follows the existing PW distributed substrate;
  find_unused_parameters=true covers dormant query adapter parameters.
- BF16 forward/FP32 physical loss, TF32, AdamW lr.001,500 warmup, cosine to10%,
  decay.01, gradient norm cap1 after synchronized backward.4 loader workers/rank.
  GPU queries every100 steps, per-rank progress every10, all-step local logs.
  No synthetic initialization, adaptive LR, extra seed, control or silent batch
  reduction. OOM requires a recorded resource-only adjustment and fresh launch.
- Evaluation1536 unique windows,512/source, seed277. Freeze actual index/hash
  before launch; exclude same-sequence temporal frame overlap with previously
  viewed compatible Task panels, regardless of anchor (abs(tick difference)>=28).
  Windows may overlap each other and reuse val sequences/PW selection data;
  these are fresh temporal windows, not untouched subjects or a final benchmark.
  Evaluate every1000 steps, all4 ranks, aggregate by actual valid-point/object
  counts. Fixed final10000 for conclusions; best.pt is descriptive only.
- The learning decision is exploratory: at final, goal beats its own static-goal
  ablation and zero hand by>=5% on>=2 sources => PROMISING for this candidate;
  otherwise UNCLEAR. No no-goal control/multiseed route superiority claim.
  World moving/static and action sensitivity need separate audit before L_goal;
  no amount of training loss alone proves world or robot usefulness.

## Resources / stopping

The latest user explicitly authorizes exactly four distributed GPUs, consistent
with Campaign's concurrent cap. Foreign tasks are never modified. Memory128
single-rank preflight measured15.65GB allocated/18.62GB reserved; synchronized
compute.882s/update. Four-rank small smoke and full-batch smoke must both pass
before scientific launch, including finite losses and final parameter equality.
Small smoke4 updates completed with all four final parameter SHAs identical.

Max6h wall/1440GPUmin,6GiB new artifacts including smoke and rolling checkpoints;
aggregate outputs+tmp must remain<300GB (prelaunch279.9GB). Stop on nonfinite,
OOM, input/config/source drift, foreign conflict or budget/deadline. Owned ranks
accept SIGUSR1 at the next100-step synchronized stop boundary and save their
state; torchrun owns its four workers. Never kill unknown processes. Checkpoints
store all-rank RNG and optimizer for same-config/world-size resume. Only own
rolling latest/best names are atomically replaced; final/best use hardlinks to
checkpoint snapshots to bound disk. Earlier runs/checkpoints remain untouched.

## Artifacts / limitations

Output roots: `outputs/goal-point-wam/goal-memory-ddp-20261012-r1/` (initial
segment and frozen evaluation panel), and `goal-memory-ddp-20261012-r2/`
(resumed training). The r1 `evaluation/panel.json` binds the panel and excluded
history; each training subdirectory holds global/per-rank indices, config/input identity,
per-rank logs/progress, validation curve, optimizer checkpoints and result.
Fixed code commit and launcher/PIDs are recorded before/at actual launch.
Raw sparse forward-repeat noise remains a separate engineering issue; final
fixed-input world independence audit cannot prove physical accuracy. Any
positive stays single-seed local evidence; multi-seed and matched causal tests
are explicitly absent per user instruction. Scaling alone does not complete
Mission's Cm-on/off RL requirements.

## Prelaunch engineering evidence

Four-rank full-batch128 smoke also completed4 updates, global batch512, with
identical final parameter SHA on all4 ranks. Warm steady update time.95--.96s;
peak observed reserved memory about17.4GiB. Small and full-batch smoke are
engineering-only, not candidate accuracy evidence or scientific initialization.
A read-only independent review confirmed gradient synchronization, occurrence
sharding and count-weighted evaluation. It found resume RNG restoration before
DataLoader iterator creation consumes an extra CPU base seed; restoration was
moved after iterator creation before the fresh scientific launch.

The supervised launcher reuses the existing PointWorld ownership checks,
records exact source hashes/PIDs and polls every30s. It requests only owned
workers save/stop on source drift, foreign context,298GB artifact threshold or
6h deadline. Max180s cleanup grace; no foreign PID is signaled.

## Actual launch

Code cc434265, physical GPUs0/1/4/5, one model/global batch512.
Supervisor PID3904485, torchrun PID3904499; worker PIDs3904547/3904548/
3904552/3904557. Exact startup identity, commands/source hashes and live progress:
[launcher state](../../../../../../outputs/goal-point-wam/goal-memory-ddp-20261012-r1/group_status.json),
[training identity](../../../../../../outputs/goal-point-wam/goal-memory-ddp-20261012-r1/train/input_manifest.json),
[progress](../../../../../../outputs/goal-point-wam/goal-memory-ddp-20261012-r1/train/progress.json).
At56 steps all4 gradient norms match, losses/gradients finite; recent median
update time.98s, remaining ETA approximately2.7--3h. Startup sampling peaks
100% utilization on each selected GPU; device memory about17--20GiB. This is
runtime evidence only. No outcome label upgraded; training remains in progress.
[Evaluation panel](../../../../../../outputs/goal-point-wam/goal-memory-ddp-20261012-r1/evaluation/panel.json)
contains512/source from59 OakInk2,88 GRAB,34 ARCTIC val sequences. Current
artifacts282.387GB, within cap; no remote GPUs, auxiliary data or control runs.

## Saved pause and direct continuation

The reference run saved/stopped cleanly at step400, retaining its original
latest/final checkpoints. A bounded engineering detour measured a fused Hilbert
batch128 benchmark at.893s versus reference.882s, so fusion was not adopted.
An optional vectorized sampling benchmark measured.690s with tested sampling
parity; this also was not adopted after the user instructed direct training
without further utilization tuning. These are engineering measurements only.

The selected action is to resume the exact original trainer/config/model from
step400, preserving optimizer and all-rank RNG, global512/10000-step target and
original absolute6h deadline. No extra scientific seed, control or backend
migration is used. This corrects the unnecessary tuning detour; it introduces
no new accuracy claim. The latest reference-source trainer is byte-identical
to cc434265. Unused performance drafts remain outside the active recipe.

## Active resumed segment

Run `goal-memory-ddp-20261012-r2`, physical GPUs0/1/2/3; supervisor PID3925911,
torchrun PID3925924. The launcher receives `--resume` pointing to r1 step400
and `--deadline` preserving r1's absolute budget. The deadline option belongs
to the supervisor; the trainer retains its existing same-config resume checks.
It restores the state after DataLoader iterator creation. This is continuation
of one model, not another seed or fresh initialization.

[Current launcher state](../../../../../../outputs/goal-point-wam/goal-memory-ddp-20261012-r2/group_status.json),
[training identity](../../../../../../outputs/goal-point-wam/goal-memory-ddp-20261012-r2/train/input_manifest.json),
[progress](../../../../../../outputs/goal-point-wam/goal-memory-ddp-20261012-r2/train/progress.json),
and [console](../../../../../../outputs/goal-point-wam/goal-memory-ddp-20261012-r2/console.log)
record actual execution. At step616 the resumed run is RUNNING, loss/gradient
finite; all four rank gradients match at the latest shared step610. Recent
median update1.138s, remaining training ETA about3h plus evaluation
and checkpoint time. A GPU snapshot showed all4 devices at100%, memory about
18--22.5GiB; point-in-time utilization is not a sustained performance claim.
A subsequent20s sample averaged54.2/49.5/54.2/57.4% across GPUs0/1/2/3
([samples](../../../../../../outputs/goal-point-wam/goal-memory-ddp-20261012-r2/resume_gpu_samples.json));
this is recorded without extending utilization tuning.
Aggregate artifacts283.10GB remain below300GB. Source/storage/ownership guards
remain active; no further throughput benchmarks or implementation changes are
planned while this run trains. Results remain UNCLEAR until actual evaluation.
