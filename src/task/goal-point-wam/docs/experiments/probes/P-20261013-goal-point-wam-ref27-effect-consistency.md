---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref27-effect-consistency
experiment_id: P-20261013-goal-point-wam-ref27-effect-consistency
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: f8b8366c
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 37
seed_pool: probe
seeds: [295, 296]
decision_changed_if_positive: continue one-model mixed-domain C1 after C0; retain trajectory and control while reducing frozen-world effect gap
decision_changed_if_negative: save C0 and stop scale-up if contracts or non-regression fail; preserve A2 and B baselines
status: PROMISING
run_id: wam-effect-ddp-20261013-r3
---

# ref2_7 C: frozen-world predicted-trajectory consistency

Result: COMPLETED at50500 total updates (C0 500 + C1 50000), global400/4GPUs. PROMISING for offline mixed-domain adaptation; Cm causal/physical utility remains UNCLEAR.
Decision: Preserve the completed candidate and frozen A/B baselines. No further overnight training; retain the forecast tradeoffs and amended admission in reporting.

## Decision Note

User explicitly authorizes C0 then large single-model training with video on
a separate server. B2 preserved warm behavior but most paired gains survived
alpha0 ablation; use the stable old interface and Probe32 warm policy here.
Qualified Probe34 Chunk-R stays fixed. This Decision Probe serves manipulation
and checks whether induced effect supervision can adapt the actor without
spoiling forecast/native control. Cheapest contract gate precedes long training;
no B changes, new data acquisition, RL or new seed replicas.

Freeze complete scene/goal/raw adapter/action/World/R and all buffers; only
trajectory decoder learns. Frozen World parameters alone cannot preserve the
GT oracle when shared scene changes, so context is explicitly detached and
scene remains eval/frozen too. Use exact raw executed actions for GT oracle.
Convert predicted displacement to9D=[future XYZ,current-relative displacement,
previous-step increment], not velocity/acceleration. Missing current/previous
points are masked. Future validity is supervision only, never actor input.
Record any policy-vs-oracle action-mask difference; Delta_E then includes that
mask difference as well as trajectory changes. No future goal enters World.

## Fixed training protocol

4 idle physicalGPUs0/1/2/3, one NCCL DDP model,100 windows/rank/global400.
Each rank exactly60 main-three /25 nominal /15 actual recovery. Within human
60 preserve historical source probabilities5:2:2 and moving/category sampling.
Nominal13startup+12later use TRAIN env0..15; recovery5 own expert future24
windows per source8/16/32, role3. Recovery retains259 original training windows;
large exposure is repeated supervision, not new robot interactions.

Main corpus is existing path-relocated data-main-worktree and train-only stats.
Robot9D/action/effect use actual own future observations and applied controls;
planned reference only supplies actor goal, never effect label. Actor future
admission remains restricted to nominal teacher and actual expert takeover.
No pure failed trajectory is relabeled. Human PD loss0; robot Chunk-R loss
is weighted by robot fraction. Loss is valid per-window coordinate MSE/.05
+1*robot-fraction control-aware R objective +.1*existing normalized motion-
weighted World point/translation/rotation objective. AdamW1e-5, decay.01,
clip1, bf16 actor/World, frozen oracle eval mode. No first-night weight search.

Seed295 initialization/296 stateless occurrence base, step/rank substreams;
reuse continuity, not independent replicates. Save exact step and optimizer;
resume recreates identical sample occurrences. No massive generated cache.

C0:500 updates after full-batch4-step DDP engineering smoke. Require finite
loss/gradients, effect-to-tau/decoder and action-to-tau gradients nonzero,
World/R/frozen parameter gradients0 and parameters/buffers unchanged, rebuilt
GT9D max difference<=2e-6m, human PD rows0, fixed GT-oracle metrics drift<=1e-5mm.
Then require h1/first8<=1.1initial+.05mm on all panels; robot first-PD XYZ/
rotation/finger <=1.1initial+(.1mm/.01rad/.01rad), clipping<=initial+.01.
These non-regression gates supplement the user's numerical C0 gate to avoid
immediately scaling up a visibly degraded actor. No statistical qualification.

C1 starts automatically only after C0 passes, continuing the same model and
optimizer for20k additional updates (total20500). Fixed evaluation every1k and
at C0/target/deadline. Two consecutive non-regression failures stop/save.
Extension to50k additional updates (total50500) requires all checks passing,
mean8-panel Delta_E all24 improving by>=max(.1mm,1% of previous absolute gap)
versus the first recorded panel in total16500..18500, and enough measured
wall time for30k more updates. Fixed8h deadline includes C0/C1/evaluation;
no automatic budget extension. It may end before20k if throughput requires.

## Metrics / evidence boundary

Fixed panel:96 existing validation occurrences per human source, nominal64,
startup64, three actual recovery TRAIN pools. Report h1/first8/all24; robot
first applied PD XYZ/rotation/finger p95 and clipping; predicted and exact-GT
World point EPE h24/all24. Delta_E=predicted minus GT EPE, may be negative;
it is a frozen-model discrepancy, not irreducible error or physical utility.
Initial and every evaluation use identical samples/batch shapes and cached frozen scene/goal/raw context; exact oracle is recomputed each time.
Per-rank frozen context packets are persisted so checkpoint resume retains
the same oracle representation, rather than reintroducing sparse replay noise.
All saved arrays/metrics have one immutable step identity.

Long-run loss decline does not alone justify useful effect supervision.
No matched lambda_E0 long control or physical actor execution is authorized
in this stage; those remain necessary before attributing policy benefit to Cm.
Do not upgrade this single-seed mixed-domain run to Validation.

## Resources / operation

Single-model4GPUs (shared max6), wall28800s/32GPUhours, owned outputs+scratch
<=8GB, global outputs+tmp<=400GB. Initial global usage~293.70GB and disk free
~606GB. Stop/save on foreign assigned-GPU process, source edit, deadline,
nonfinite, contract/frozen drift or storage bound. Never stop unknown workers.
Keep immutable decoder/optimizer step checkpoints (roughly tens ofMB each),
latest receipt only; preserve original checkpoints and B2 artifacts.
Supervisor records owned rank PIDs, GPU/VRAM, progress and ETA. No new branch.
Latest user instruction: root polls every30min after startup and sends no
chat updates before2026-10-13 10:00 Asia/Shanghai. Deterministic supervisor
resource checks run independently and consume no model tokens.

## Implementation / reproduction

- `src/task/goal-point-wam/src/goal_point_wam/effect_consistency.py`
- `src/task/trajectory-policy/src/trajectory_policy/wam_effect_data.py`
- `src/task/goal-point-wam/tools/run/train_effect_consistency.py`
- `src/task/goal-point-wam/tools/run/launch_effect_consistency.py`

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/goal-point-wam/tools/run/launch_effect_consistency.py --smoke --output outputs/goal-point-wam/wam-effect-smoke-20261013-r1
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/goal-point-wam/tools/run/launch_effect_consistency.py --output outputs/goal-point-wam/wam-effect-ddp-20261013-r1
```

## Engineering findings / launch

Original engineering runsr1..r4 are preserved and are not scientific negatives.
The first hit DDP find_unused output-sink wrapping: returned tau ceased to be
an ancestor of returned effect although decoder gradient existed. A2s CPU
Gloo reproduction demonstrated the distinction; freeze the unused memory-mode
query adapter and use DDP without that wrapper. Fix8d80e631, then filter frozen
parameters out of autograd.grad (17313a88). Repeated CPU DDP forward/backward
regression passes. Nine existing action/consequence tests and two deterministic
mixture/resume tests also pass (12 including the new DDP regression).

The next run caught fresh sparse-encoder oracle replay drift despite frozen
state. Fixed evaluation now caches frozen scene/goal/raw context, explicitly
isolating actor adaptation. Smoke also measures fresh-context oracle replay
rather than concealing the noise: max absolute h24 discrepancy.05199mm on
nominal; no state change. Cached-context GT oracle drift exactly0.
An empty rank in the three-window human smoke panel required optional diagnostic
aggregation (129d9831); no loss or dataset semantics changed. Scientific panel
has96 human occurrences per domain and is partitioned across all4 ranks.

Full-batch4-step smoke r5 COMPLETED/engineering PASS,36/36 miniature panel
checks pass. All4 effect-to-tau sums363..484, effect-to-decoder1024..1976,
action-to-tau75..85 nonzero; human PD rows0; rebuilt recorded9D max1.937e-7m.
World/R/all frozen differences0. Weighted effect gradient is2.27..3.19 times
geometry gradient, so there is no evidence that lambda_E=.1 is too small.
No increase is made. This checks wiring, not the 500-step scientific gate.
Observed full-batch first-step~1.22s and rank0VRAM3855MiB/utilization69%.

Scientific r1/supervisor604880 started00:23:29 but stopped before update1:
CPU DataLoader fork inherited initialized LLVM threads (`pthread_join failed`).
This is a loader implementation failure, not negative C evidence. Keep logs.
Use4 spawned CPU loaders/rank after LLVM initialization (847452f8); repeat the
same full-batch engineering smoke with this actual loader path. Smoke r6
COMPLETED and all numerical/frozen/gradient checks pass. No method/data change.

Scientific r2/supervisor609910 starts from the original warm model, with the
same global400 and4GPUs. Its original absolute deadline1791851009.5187526
(~08:23:29) is preserved, so engineering retry does not reset the8h budget.
Implementation01add31e also persists frozen validation contexts for resume.
Root will inspect the scientific C0 gate, controls, effect curves and resource
status at30-minute intervals while user sleeps, without chat before10:00.

Startup verification00:31: actual scientific update400 reached, finite loss.
After loader startup, updates100..400 took~.344s/update (~1160 windows/s across4GPUs). Rank0
VRAM5251MiB; supervisor snapshots four-card utilization49..69% and
VRAM5.3..7.7GB, within24GB/card. Early target ETA~2.45h includes
startup amortization; steady update cost projects~1.95h plus evaluations.
No memory/throughput blocker observed. Root next scientific poll~01:02 local; no user notification during quiet hours.

- [Engineering result](../../../../../../outputs/goal-point-wam/wam-effect-smoke-20261013-r6/train/result.json)
- [Live supervisor](../../../../../../outputs/goal-point-wam/wam-effect-ddp-20261013-r2/group_status.json)
- [Live model manifest](../../../../../../outputs/goal-point-wam/wam-effect-ddp-20261013-r2/train/manifest.json)

## Limitations / future evidence

Human object goals are hindsight-achieved endpoints, robot goals are planned
reference endpoints; labels remain separate. Existing World learned mainly
human data and scene was robot-adapted, so robot oracle trust remains empirical.
Recorded hand trajectories are an executed-motion proxy, not a native-command
causal transition contract. Matched effect-off continuation, multi-seed held
out manipulation and causal utility Validation remain deferred.


## 01:02 Decision Note: admit numerical C0 with a disclosed forecast tradeoff

Scientific r2 stopped at500 because root's added 10% trajectory guard failed
startup first8 only:10.1074->11.3837mm, threshold11.1682mm;35/36 passed.
All user-requested numerical/frozen/oracle/gradient contracts passed. All
first-PD checks passed: nominal XYZ p9571.416->17.899mm, startup21.503->1.454mm,
three recovery pools43.197->18.229,44.177->22.677,171.827->151.582mm. Human
all24 and effect gaps improved substantially. Robot effect gaps are mixed.
This is a genuine forecast tradeoff, not an implementation error or a pass
under the original stricter gate; original result/checkpoint remain intact.

Question: stop the requested long C training over a1.276mm startup first8
tradeoff, or obtain the explicitly requested long-horizon evidence? Root
chooses same-model/same-optimizer continuation, with no loss, sampling or
learning-rate change. ref2_7 defines C0 as numerical contracts and requests
first-PD preservation; it does not define the additional10% tau cutoff.
Replace root's extra scale-up rule prospectively: original36 diagnostics stay
visible, native-control gates retain their original limits, and trajectory
h1/first8 has a2x+.05mm gross-deterioration stop. Two consecutive failures of
these continuation guards still stop/save. This is a protocol amendment after
seeing C0, explicitly excludes any pre-registered non-regression claim, and
must be included in final reporting. No weight search or baseline rerun.

Resume immutable step500 and its cached contexts/optimizer into fresh r3;
recheck exact oracle, frozen state and gradient contracts before admission.
A20k continuation costs~2h plus evaluation; optional50k remains conditional
on gap improvement, guard pass and remaining original deadline08:23:29.
Failure saves and stops; success remains offline single-seed evidence only.
No new authorization boundary: same task/claim,4GPUs,400GB and original8h.

Continuation r3 supervisor635812 launched with f8b8366c; original absolute
deadline remains08:23:29. Six targeted action/DDP/mixture/guard tests pass.
[Continuation supervisor](../../../../../../outputs/goal-point-wam/wam-effect-ddp-20261013-r3/group_status.json).


## Completed run and saved-panel audit

r3 completed2026-10-13 05:55:31 Asia/Shanghai, exit0, total50500 updates.
C1 used exactly50000 additional updates from r2 step500 with the same model,
optimizer and stateless occurrence stream; no reset/new seed/loss change.
At20500, continuation guards all passed and mean8-panel gap decreased enough
versus17000; measured time allowed30k more within the original8h deadline.
The recorded curve therefore extended automatically to50500. Actual C1 update
wall time17401s (~4h50m,19.33GPUh excluding initialization/C0), ~.348s/update
including evaluations. Total sample exposure20.2M windows; C1 alone20M,
including12M human/5M nominal/3M recovery repeated draws, not unique data.

| Fixed panel | tau h1 mm, initial -> final | tau first8 mm | tau all24 mm | World gap all24 mm | first-PD XYZ p95 mm |
| --- | --- | --- | --- | --- | --- |
| OakInk2 VAL | 7.105 -> 2.639 | 17.531 -> 5.610 | 43.574 -> 15.761 | 3.130 -> .881 | n/a |
| GRAB VAL | 7.199 -> 5.110 | 34.995 -> 21.392 | 114.145 -> 79.724 | 39.309 -> 16.131 | n/a |
| ARCTIC VAL | 7.023 -> 4.208 | 28.793 -> 18.923 | 84.107 -> 62.485 | 23.092 -> 11.072 | n/a |
| nominal REPORT | 6.947 -> 6.928 | 15.058 -> 16.257 | 21.646 -> 22.032 | 6.497 -> 6.329 | 71.416 -> 14.538 |
| startup REPORT | 6.966 -> 8.104 | 10.107 -> 10.812 | 11.263 -> 10.150 | -.024 -> .103 | 21.503 -> .740 |
| t8 recovery TRAIN | 6.925 -> 7.471 | 14.325 -> 13.683 | 16.682 -> 15.100 | 3.782 -> -1.220 | 43.197 -> 9.247 |
| t16 recovery TRAIN | 7.046 -> 7.893 | 16.256 -> 14.735 | 18.073 -> 16.123 | 2.909 -> -1.620 | 44.177 -> 6.959 |
| t32 recovery TRAIN | 13.626 -> 13.352 | 22.788 -> 19.715 | 24.869 -> 21.437 | 5.911 -> -8.251 | 171.827 -> 148.081 |

Unweighted8-panel mean gap10.5757->2.9281mm (human3 mean21.8437->9.3614;
robot5 mean3.8148->-.9318). Panel averaging is descriptive and not population
weighting. Nominal/startup gaps barely change; recovery TRAIN gaps improve,
including negative values, which can reflect frozen-model bias/compensation
and do not demonstrate physical usefulness or held-out recovery generalization.
Nominal native XYZ p95 improves while all24 tau slightly worsens. t32 remains
148mm control error, so broad reliable manipulation is not established.

All five robot first-PD XYZ/rotation/finger preservation checks pass; all panel
clipping rates0. Final amended continuation guards36/36 pass. Original strict
forecast diagnostics34/36 pass: startup h1 +16.34% and t16 recovery h1 +12.02%
fail the original1.1x+.05mm limits. Initial C0 startup first8 failure is also
preserved; later first8 recovers within its original limit. Do not describe
the experiment as universal non-regression or an unamended prospective test.

Across all recorded evaluations, frozen model/R differences exactly0; cached
GT-oracle per-window arrays exactly unchanged; actor-vs-oracle action masks
identical on these panels. `tools/audit/audit_effect_consistency.py` checks53
saved evaluation panels including resumed step500 and every1k/final; all
stored means/percentiles/gaps reproduce with maxdifference0, optimizer count
50500, four saved contexts present. This is a saved-array/continuity audit,
not fresh model inference or independent raw-label reconstruction. Existing
contract/gradient tests supply the engineering checks; no extra evidence run.

Historical run manifest uses `run_id=train` and `training_allowed=false` due
to trainer metadata defaults; output paths/supervisor/recipe/checkpoint identify
the actual admitted scientific r3 unambiguously. Preserve original records;
audit receipt supplies the correct run_id. Future trainer metadata now uses
supervisor run directory and distinguishes smoke from admitted training.
This metadata repair does not alter completed weights/data/metrics.

Rank VRAM plateau~8.0/10.0/9.1/6.9GB; root half-hour snapshots utilization
~30..84%, no OOM or throughput degradation. GPU0..3 released after completion.
Final r3 artifacts~.655GB including audit, well below8GB; global outputs+tmp
~294.51GB below400GB. Checkpoints are small actor/optimizer patches, not full
WAM duplicates. Latest immutable patch preserves step50500; all prior steps
and original/failure artifacts remain available. No unrelated process stopped.

- [Completed result](../../../../../../outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/result.json)
- [Latest checkpoint receipt](../../../../../../outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/latest_checkpoint.json)
- [Saved-array audit](../../../../../../outputs/goal-point-wam/wam-effect-ddp-20261013-r3/audit-r1/result.json)
- [Training curves](../../../../../../outputs/goal-point-wam/wam-effect-ddp-20261013-r3/audit-r1/curves.png)
- [Half-hour root observations](../../../../../../outputs/goal-point-wam/wam-effect-ddp-20261013-r3/root_monitor.jsonl)

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/goal-point-wam/tools/run/launch_effect_consistency.py --output outputs/goal-point-wam/wam-effect-ddp-20261013-r3 --deadline 1791851009.5187526 --resume outputs/goal-point-wam/wam-effect-ddp-20261013-r2/train/checkpoint_step500.pt
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/goal-point-wam/tools/audit/audit_effect_consistency.py --run outputs/goal-point-wam/wam-effect-ddp-20261013-r3 --output outputs/goal-point-wam/wam-effect-ddp-20261013-r3/audit-r1
```

The experiment has delivered the requested large-exposure C mechanism Probe.
Retain final and20500 checkpoints for a later separately authorized execution
comparison; matched effect-off continuation is still required before attributing
control gains to LE rather than continued tau/action supervision. No physical
rollout, Cm-off extra training, B change or further50k run is performed tonight.


## Subsequent user-authorized physical check

User subsequently requested final C rollout. One full542-control development
rollout under [Probe38](P-20261013-goal-point-wam-ref27-rollout.md) is valid but
actor held0/no lift; teacher/GT-Chunk held434/441, local grasp calibration passes.
Actual command/frame/outcome audit max0. This leaves the offline improvements
intact but demonstrates that they did not establish autonomous grasp for this
checkpoint/reset. Cm causal utility remains UNCLEAR; no further training here.
