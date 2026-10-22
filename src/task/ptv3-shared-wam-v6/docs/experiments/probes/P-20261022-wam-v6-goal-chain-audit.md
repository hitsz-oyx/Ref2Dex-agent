---
schema: ref2dex.probe.v2
probe_id: P-20261022-wam-v6-goal-chain-audit
experiment_id: P-20261022-wam-v6-goal-chain-audit
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: c971bca689fa76845128cecb04c2dc43173f069c
claim_id: C1
hypothesis_family: HF-v6-goal-contract-identifiability
probe_index_in_family: 1
seed_pool: probe
seeds: [172,181,182,183,184,185,186,187,188]
decision_changed_if_positive: prioritize causal waypoint versus realized-endpoint conditioning and intent supervision before further model scaling
decision_changed_if_negative: inspect supervision and sequence generalization without assuming missing contact intent explains the errors
status: RUNNING
run_id: goal-chain-audit-20261022-r1
---

# Goal contract intervention and scaled existing-weight diagnostics

Decision Probe. Distinguish conditioning contract mismatch, event-phase
dispersion and joint forecast failure. Cheapest evidence: immutable existing
healthy4371 and pinned FM8500; no optimizer, new data, model growth or loss edits.

## Independently verified contract, and replay scope

`wam_rl_policy_v3/goal_scheduler.py:OrderedGoalScheduler.update` emits
`goal_semantics='spatial_path_waypoint'` and `reference_endpoint_time_s=None`.
Its positive horizon_s=.8 does NOT mean guaranteed arrival at t+.8. It uses
ordered milestones, observed path progress, proximity/motion feedback and bounded
lookahead (default3–25mm), recovering/timeout modes. V6's inherited
`ptv3_shared_wam_v5/data.py:sample_window` instead supplies the exact realized
pose at t+24, as32 same-ID point displacements in the current object frame.
These are different random variables. Direct replacement is not justified by
matching tensor dimensions; this audit measures the sensitivity, not a training fix.

Planner-G runs this exact, unmodified deployment path module persistently from
the first available observation through each selected tick, at30Hz; never reset
per window. Reference=complete recorded object path supplied as a known
instruction (oracle reference); only current/past poses and valid hand11 enter
`update`. Missing pose observations are skipped and counted; times remain real.
Right valid hand else left; hand11 indices0,1,5,9,13,17,4,8,12,16,20. Reports
retain modes/timeout/reference index. This is NOT full TaskScheduler replay:
the caches lack measured grasp-established, released, supported feedback. The
path module is used by TaskScheduler and deployment adapters, but these missing
inputs prevent certifying the complete deployment pipeline.

Oracle-G=unchanged t+24 endpoint. Perturbed-G=start from legal Planner-G, sample
paired SE(3) residual inv(Oracle-G)*Planner-G from TRAIN only, exclude the same
sequence, right-compose onto Planner-G, project with exact scheduler bounds.
This is a stress sensitivity arm matched to empirical TRAIN planning errors,
not an unbiased simulator of real robot planning error. E_next held identical
across arms to isolate goal; it is the recorded annotation, not separately
inferred deployment E_next. Planner causal replay does not make that label causal.

## Scale, splits, seeds and metrics

All frozen DEV3356 windows (all existing sources); report independently by
source and capture-cluster statistics. A separately flagged nonoverlap core
has disjoint inclusive[t-3,t+24] support28frames within capture. All-window
DEV coverage does not pretend3356 independent examples. TRAIN bank samples
up to16 disjoint supports from EVERY frozen TRAIN sequence (~1064); up to8
selected independently of their temporal order for neural evaluation. Unadmitted
prefixes: one random complete28-frame valid support avoiding EVERY local TRAIN
window in that recording. Complete remote ancestor admission manifest remains
unavailable: label `heldout_prefix_local_recipe`, never lifetime-unseen.

Selection/empirical perturbation seed181; fixed FM per-sample noise seeds
172,181..187,K8 with common random numbers across goal arms. Bootstrap/permutation188.
These are repeated forecast draws, NOT eight independent trained models.
Healthy deterministic K1; FM backbone evaluated once/condition then exactly the
existing4-step ODE repeated8times. Native-forward identity smoke required.
Report sample-mean accuracy and spread; no best-of-K claim. C head is independent
of sampled residual τ, so repeated C scores do not imply C sampling diversity.

All24 EPE, C-head queryF1 threshold.8 and probability-weighted GT-surface contact
centroid error, object best-SE3 residual; bands1–8/9–16/17–24. Phase evaluation
labels: pre-contact; first6 frames after GT event entry=onset; later contact=hold;
clear state following release=post-release; missing=unknown. No causal inference
from phase masks. Geometry prefix8 only, all2048 object queries, observed778/hand
bone-transport skin: distance on soft-positiveGT C>.25; geometricF1 exponential
3mm radius/10mm falloff threshold.8 (~5.23mm); contact centroid error on GT-positive
frames; oriented-nearest-point penetration PROXY and GT-direction-referenced
clearance PROXY on negativeC<.01. These proxies are not physical signed mesh
penetration. Own-object dense CAD uses observed-only affine extension of predicted32,
report affine residual and object rigid residual alongside both anchors. Compare
absolute hand/object EPE before interpreting small relative error as quality.

Identifiability follow-up: same raw H4-hand/H4-object/goal distances and E_next,
20/50/100mm complete-link neighborhoods; independent-capture groups. Add event-
aligned dispersion with fixed relative-event offsets and common valid support,
side by side with raw variance, plus contact-presence/region/finger/action labels.
Report neighborhood support/diameters and label-partition distances; singleton
groups contribute no variance degrees of freedom. Event timing control remains
necessary because E_next timing already enters v6. Ref annotations unavailable.

## Runtime and artifacts

Run root `outputs/ptv3-shared-wam-v6/goal-chain-audit-20261022-r1/`.
Entry `tools/audit/audit_goal_chain.py --phase prepare|bank|evaluate`.
Environment `/home2/wyy/miniconda3/envs/graspenv/bin/python`, strict FP32,
AMP/TF32 off. Two detached inference workers, physicalGPU2/3 exclusively (fresh
idle check required); never touch trainingGPU4/5/6 or GPU7, no process control
outside this run. Each worker allocator18GiB, ≥2GiB device headroom, initially
3h wall-time cap; completed immutable batch shards support resume. If GPU is
claimed, stop only owned diagnostic worker and reassign to an idle0–3 device.
CPU replay6 processes. No credential reads/writes, no push. No checkpoints made.
Estimated additional peak8GiB, initial disk149.45GiB available; floor20GiB.
Exact frozen commit, launch commands/PIDs/progress/ETA recorded in run receipts.

## Executor three-condition design — NOT LAUNCHED, awaits owner confirmation

Decision: does the learned prior improve executable task success versus the
existing smooth reference+geometric-retargeting prior, or only forecast scores?
Freeze one executor checkpoint, action frequency/control gains, observations,
hand/object morphology, reset states, task goals, episode timeout and safety
termination. Three condition providers: predictedτ from one fixed WAM checkpoint;
the owner's existing reference+geometry retargeter at matched replanning rate;
no trajectory prior (zero/masked condition with an explicitly compatible executor).
If the fixed executor was never trained with masked conditioning, no-prior is
an out-of-distribution ablation, NOT RL-from-scratch evidence; report that
separately. RL-from-scratch requires its own trained policy and matched compute,
so it cannot simultaneously be claimed a fixed-executor comparison.

Proposed data: immutable existing capture-disjoint task split,100 reset/task
instances across available objects and contact regimes,3 paired simulator seeds
per reset per arm=900 episodes. No selecting visually favorable demonstrations.
First engineering feasibility smoke5 resets/arm, then freeze100 before success
comparison. For each episode report full-task success, path/terminal achievement,
surface contactF1/penetration/contact-position error, longest continuous valid
hold duration and drop count; report paired task-cluster95%CIs and all failures.
Native robot contact/force vs human surface contact must be distinguished.

Budget proposal only: one idle authorized GPU, up to2h feasibility plus6GPUh
evaluation, max900 episodes×20s simulated time; estimate wall time from smoke
before authorization. Optional RL efficiency arm:3 seeds, success curves versus
environment transitions with equal observation/action/reward/reset/task mix and
fixed maximum10M transitions/seed; budget at most24GPUh total initially and stop
at matched target success or bound. Count all prior/executor training separately;
do not present inherited pretraining as free. No PPO, training or execution of
this design is authorized in this diagnostic run.

## Execution update 11:17 CST — partial, not the final FM comparison

Prepared10454 candidate rows. Neural panels: TRAIN5241/1064 sequences,
DEV3356/113 sequences, local-uncovered-prefix905/905 sequences; DEV nonoverlap
core847. One TRAIN alias-overlap row1167/0/11 was excluded; independent complete
207579-window/cross-slot/alias support audit PASS. `panels.json` and
`support-audit.json` contain exact identities.

Main GT bank6185 valid windows/1063 captures (up to16 disjoint supports); narrow
50mm complete-link only6 groups/32 captures, time3 control zero groups. To obtain
inside-horizon event alignment support, add existing-TRAIN candidate bank with
16 uniform+16 event-inside-window candidates/capture, seed182; seed183 repeats
that SAME candidate selection protocol. Candidate pool may overlap, but each
variance group and gate use a native capture at most once. This changes candidate
coverage, not model/data/training/evaluation panels. Full bank plus event-inside-
horizon-only subpopulation reported separately; no mixture of raw/aligned cohorts.

Seed181 healthy baseline, all3356 DEV: Oracle hand/object8=15.797/14.722mm,
Planner21.122/25.550, Perturbed21.668/25.887; C-head F1=.685894/.619804/.684312.
C-head contact centroid3.883/4.002/3.889mm. Predicted hand vs GT/own object contact
unsigned distance: Oracle14.709/10.537mm, Planner22.980/11.287mm. Small relative
own-object distance accompanies worse absolute EPE; do not call it better quality.
FM full panel pending; its partial subset is deliberately not compared to full
healthy DEV. Exact sums/counts/capture-paired intervals in `large-summary.json`.

Goal contract gaps independently computed: DEV planner/oracle point gap205.037mm
(mean),107.842mm median; target-reference-frame-minus-t median1.032, not24.
69.845% of realized Oracle t24 poses exceed the deployment scheduler's default
50mm/.35rad/50mm translation/rotation/mean-point budget. This is a substantial
conditioning-domain mismatch. It does NOT establish the causal benefit of any
new training recipe; no new training is run.

Full-frame geometry aggregates use oriented-normal proxies. To also give actual
watertight CAD occupancy at scale, queue a t8 endpoint-only pass after each OWNED
GPU worker finishes; exact existing model computation, native-forward identity
check. All DEV inferred; native GRAB CAD assets used where mapped/watertight.
128 sampled transported-skin vertices/draw, K8 FM; CPU BVH SDF with cube and
CAD-registration controls. Predicted affine CAD containment is exact for invertible
maps; world depth lower/upper bounds=positive canonical SDF×min/max singular value.
This is object mesh occupancy of a hand skin proxy, not full robot geometry.
GT-trajectory smoke4 windows/1 capture: GT-object and predicted-object identity
both11/512 inside,9/512 >1mm; depth sums44.68359mm match to1e-5mm. The smoke
correctly permits penetration in the proxy GT skin; it does not assert zero.

GPU event:3 was claimed by PID2626443 between idle checks;32-window smoke had
already completed22.70s, no signal delivered. Initial full workers GPU2/0.
GPU2 later acquired other PIDs2633253/2633564; only owned PID2629688 was stopped
at complete healthy4737 batch boundary, then resumed on freshly idle3 as
PID2639649. GPU0 PID2629690 uninterrupted. Record details in
`gpu3-resource-interruption.json`, `gpu2-resource-interruption.json` and
`launch-worker0-resume1.json`. GPU4/5/6/7 and other jobs untouched.

Commands (all from repository root; full argv/PIDs/time/commit in launch JSON):

```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
OUT=outputs/ptv3-shared-wam-v6/goal-chain-audit-20261022-r1
TOOL=src/task/ptv3-shared-wam-v6/tools/audit
$PY $TOOL/audit_goal_chain.py --output "$OUT" --phase prepare --workers 6 --seed 181
$PY $TOOL/audit_goal_chain.py --output "$OUT" --phase bank
CUDA_VISIBLE_DEVICES=3 $PY $TOOL/audit_goal_chain.py --output "$OUT" --phase evaluate --batch-size 64 --shard 0 --shards 2 --k 8 --memory-gib 18 --max-seconds 10400
CUDA_VISIBLE_DEVICES=0 $PY $TOOL/audit_goal_chain.py --output "$OUT" --phase evaluate --batch-size 64 --shard 1 --shards 2 --k 8 --memory-gib 18 --max-seconds 10800
CUDA_VISIBLE_DEVICES='' $PY $TOOL/prepare_event_enriched_bank.py --output "$OUT" --seed 182
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_event_aligned_variance.py --output "$OUT/event-bank" --seed 188
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_event_aligned_variance.py --output "$OUT/event-bank" --seed 188 --event-only
CUDA_VISIBLE_DEVICES='' $PY $TOOL/audit_goal_contract_gaps.py --output "$OUT"
CUDA_VISIBLE_DEVICES='' $PY $TOOL/summarize_goal_chain.py --output "$OUT"
```

The launchers use `subprocess.Popen(start_new_session=True,stdin=DEVNULL)`;
training=False, frozen inference corec971bca6, later diagnostic helpers as logged.
Shared workspace HEAD can change due other threads; it is not a new model version.
GPU batch64 reaches95–99% utilization,2.83GiB peak torch allocation (~5–6GiB
nvidia process usage). Estimated all primary metrics11:28, followed by CAD pass
~11:35–11:45 CST subject to resource conflicts. Initial3h bound remains14:05;
no budget extension or new training. Native-forward control max2.09e-7m and
4 event-warp/support/variance tests PASS.
