# P-20261020-inspire-scheduled686-bc

This probe warm-starts a Scheduled686 Inspire executor from the best Current421
checkpoint and adds only the newly generated spatial WAM tau plus validity.
It is paired with `P-20261020-inspire-scheduled686-cache`; the v4 checkpoint has
no trained CONTACT/RELEASE event input.

- `experiment_id`: `P-20261020-inspire-scheduled686-bc`
- `run_id`: `inspire-scheduled686-bc-20261020-r1`
- source: audited same-execution teacher rollout
- parent: Current421 best at step 3000
- model: 686 input, 12 output, 1024/1024/512/512 ELU
- optimizer: AdamW 3e-4, batch 2048, 50k steps

## Result (2026-10-20)

The run completed 50,000 updates on GPU6 in 1,546.9 s.  The best fixed DEV
command MAE was 0.036683 at step 1,000 (root 0.016646, finger 0.056720);
the final step was 0.052845.  The best checkpoint is retained for later
closed-loop comparison.  These are offline action errors and do not establish
that tau improves physical task success.


## Corrected training boundary in v3

The historical r1 initialization retained random tau/valid columns. Its small
DEV differences cannot be used as a clean WAM ablation. The repaired trainer
uses `initialize_current`, zeros all 265 new columns, and admits only shared
TaskController-goal layout v2. Step0 is saved as both best/latest before updates,
so the exact parent remains selectable if condition training degrades DEV.
Shape feature distribution diagnostics are logged at evaluation. Capacity,
optimizer and the 50k maximum remain the Shadow recipe; no new long BC run is
started as part of the engineering repair.


The real-parent regression at 94ce4175 used Current421 best step3000 and 256
active/new-tau-valid rows from eight TRAIN objects. Initial output max error was
2.384e-7 and all added weights were exactly zero. One engineering AdamW step
produced nonzero tau/valid gradients (max 0.014186/0.004477) and updated the new
columns; no checkpoint was saved. Parent normalized shape features had p99 2.48
and maximum 3.24 on this panel. This small panel does not exclude broader shape
shift or establish grasping. Report:
`outputs/wam-rl-policy-v3/inspire-current421-warmstart-audit-20261020-r1/report.json`.


## Corrected full-data retraining: r2

- classification: Decision Probe; distinguish whether repaired shared-goal tau
  can improve matched DEV imitation/closed-loop behavior over the exact parent.
- next decision: better DEV and physical progress supports a bounded policy
  improvement probe; a step0 win or no closed-loop progress prioritizes causal
  error/conditioning diagnosis before multi-hour PPO. No WAM attribution claim.
- cheapest test: reuse immutable admitted data and v2 tau layout; retrain only
  Scheduled686 from Current421 best, keep step0 in selection, then paired DEV and
  the same fixed TRAIN native panel. Do not retrain Current421 or recapture data.
- task: wam-rl-policy-v3; branch: main.
- run_id: inspire-scheduled686-bc-20261020-r2.
- git_commit: supplied by REF2DEX_CODE_COMMIT in the frozen run manifest.
- inputs: original measured admission, v3 shared-goal cache r2, Current421 best step3000.
- recipe: 50000 steps, batch2048, AdamW3e-4/weight_decay1e-5, seed247,
  hidden1024/1024/512/512 ELU, joint CAD encoder, gamma not applicable to BC.
- engineering: differentiable CAD encoding once per episode point cloud per
  minibatch and gather back to rows; no frozen shape feature cache, no change
  in loss/sample weighting. Output and all-parameter gradient regression required.
- environment: graspenv Python3.8/Torch2.4.1+cu121; GPU6 training, GPU7 evaluation.
- limits: 8h wall-clock, own STOP, bounded best/latest only; leave other GPUs free.
- outputs: outputs/wam-rl-policy-v3/inspire-scheduled686-bc-20261020-r2/.
- evaluation (superseded by r3): paired tau-valid DEV5693 rows, followed by deterministic native controls;
  this small TRAIN panel cannot establish unseen-object generalization.

## ref1 coverage and recovery repair: r3

r2 was stopped during CPU preparation before any optimizer update. Its stopped
manifest is retained; it is not a failed learned-policy result.

The trainer now retains all task-active real commands, including invalid WAM
startup rows with zero tau and explicit validity false. Expected counts are
TRAIN54162 (52660 valid / 1502 invalid), DEV5879 (5693 valid / 186 invalid).
Paired Current421/Scheduled686 DEV uses exactly the same full set and reports
valid and invalid groups separately. This repairs a concrete coverage gap; it
does not establish that the omitted rows caused previous physical failures.

TaskScheduler no longer consumes a future planned RELEASE on an unplanned drop.
Recovery requests CONTACT without mutating that planned cursor, and regrasp
resumes the pending RELEASE. A planned intermediate release still requires
confirmed path context and observed arrival. These event-only changes do not
alter task10/local goals or v4 inputs, so the shared-goal tau cache is reusable.

- run_id: inspire-scheduled686-bc-20261020-r3; task/branch unchanged.
- git_commit: frozen launch manifest; same inputs, parent and 50k recipe as r2.
- preparation: four ordered spawn workers; serial/parallel arrays and ordering
  tested equal. Shared differentiable CAD output/gradient GPU audit at
  outputs/wam-rl-policy-v3/inspire-shared-cad-audit-20261020-r1/report.json:
  output max1.79e-7, parameter-gradient max2.98e-8.
- outputs: outputs/wam-rl-policy-v3/inspire-scheduled686-bc-20261020-r3/.
- regression: v3 132 tests passed, including startup labels, unplanned
  drop/regrasp/continued tracking and legitimate intermediate release.
- selection/decision/limits: unchanged; retain step0, best/latest only, GPU6
  train/GPU7 evaluation, no long PPO until a positive physical signal.

### r3 matched evaluation protocol

Training is frozen at 57777674; native/paired DEV tools at 8df34303 also save
initial physical snapshots and at most the first100 decisions, with an explicit
mask for actions actually applied. Subsequent 6782d140 tightens terminal-event
acknowledgment only; it does not change task10, local goal or v4 prediction.

- selected-policy run: inspire-ref1-paired-20261020-r1, GPU7; Current421 parent
  versus DEV-selected Scheduled686 best, same seed246, same two TRAIN sequences
  s10_airplane_fly_1 / s10_alarmclock_pass_1, 300 native controls, no PPO update.
- trained-final diagnostic: inspire-ref1-latest-20261020-r2, GPU6 after training,
  same paired DEV and native panel for final50k weights. This distinguishes
  imitation-based selection from physical behavior; it cannot establish a WAM
  benefit against an equally continued Current421 training budget.
- full paired DEV5879 commands, split into valid5693 / invalid186; keep full
  native initial-state comparison rather than relying on the random seed alone.
- all native runs use graspenv policy + dexplore_v120_train physics runtime,
  deterministic clipped means, task-active termination boundary, no video.

Before the learned-policy comparison, Current421 initial measured347 agrees
with its recorded teacher data to 2.98e-7 max on these two sequences; task10 is
exact. Initial command MAE is already0.13543 / 0.07339, and both remain ACQUIRE
with path progress0 at control300. This isolates an initial imitation deficit
on this panel; later time-aligned action differences also include state drift.
The 300-control budget does not constitute full-episode success evaluation.
Report: outputs/wam-rl-policy-v3/inspire-ref1-paired-20261020-r1/current-teacher-first100.json.

The unused latest-r1 wrapper was stopped while waiting for training, before any
GPU evaluation, to correct a metadata key in the wrapper. Its stop record is
retained; it is not an additional policy result.

### r3 completed results and decision

All50000 updates completed in597.23s after56.64s preparation; sampled training
GPU utilization84–93% (checkpoint/evaluation samples sometimes lower), peak
reported1329MiB. Best/latest output is bounded to about39MiB total.

| Paired DEV subset | Current421 | Scheduled best (step0) | Scheduled final (step50000) |
| --- | ---: | ---: | ---: |
| All5879 | 0.03825807 | 0.03825807 | 0.05427545 |
| Valid5693 | 0.03632101 | 0.03632101 | 0.05271482 |
| Invalid186 | 0.09754662 | 0.09754663 | 0.10204255 |

Step0 remains selected. The tiny step0 prediction difference (mean2.42e-8)
is numerical; it is not a learned WAM effect. Latest training loss declines
while DEV error rises, consistent with overfitting, without identifying a unique
cause. Normalized DEV shape-feature max grows from3.60 to9.66; values are
finite and this alone does not prove an encoder defect.

Both native comparison runs completed300 controls on the two fixed TRAIN
motions. All initial q/dq/physical/object/hand/tick/table arrays match bitwise,
including the GPU6 final-policy diagnostic against the GPU7 parent. Native PD
error and episode-action mismatch are0. Both Scheduled runs produce596 valid
WAM predictions including the unused final decision; task10 versus WAM point-goal max errors are6.15e-9m for selected
step0 and1.02e-8m for final50k.

At control300 all three policies remain/return ACQUIRE, with path progress0.
Final50k does improve first-command MAE on these TRAIN motions from
0.13543/0.07339 to0.00778/0.00705. The alarmclock observer enters TRACK at tick60
(40 of the first100 observations) but returns ACQUIRE by tick300. This is a
temporary motion-coupling heuristic signal, not certified stable grasping or
task success. Later time-aligned teacher differences include changed states.

Decision: do not launch long PPO from these results. Diagnose the transition
from initial imitation to loss of coupling and closed-loop state drift; initial
measured/task inputs on this panel are already aligned. Any eventual WAM benefit
claim still needs matched continued Current training and broader physical tests.
This two-TRAIN-motion,10s screen does not establish full-episode success or DEV
physical generalization.

Evidence: selected-policy run's paired-dev.json, initial-state.npz,
first100-decisions.npz, current-teacher-first100.json and first100-analysis.json;
final-policy run's paired-dev.json and native/policy manifests. No PPO updates.
All owned GPU jobs have ended. Full v3 regression133 tests and scoped repository
Python/structured/Markdown checks pass.

## Latest admission r3 and learned goal parent: r4

User explicitly requests training with WAM after the latest Current421 goal BC.
Decision: does shared-goal predicted tau improve imitation or native behavior
on this corrected corpus, or does the closed-loop deficit persist? Positive
physical progress supports a bounded improvement probe; no progress prioritizes
coverage/conditioning repair. Cheapest discriminating run: reuse the learned
Current421 parent, rebuild only its new tau, train one Scheduled68650k arm and
evaluate DEV-best plus the predeclared50k endpoint on the fixed four TRAIN lifts.
This is not matched-budget WAM attribution or full-episode validation.

- task/branch: wam-rl-policy-v3/main; git_commit in the frozen launch manifest.
- run_id: inspire-scheduled686-bc-20261020-r4.
- source: inspire-teacher-rollout-audit-20261020-r3/manifest.json, all admitted
  active56036 TRAIN/6081 DEV commands from501/62 sequences, admission v2.
- parent: inspire-bc-sampling-20261020-r1/startup_half/best.pt, step7000,
  actually trained goal421 on this exact source; no old Current421 parent.
- cache: new inspire-scheduled686-cache-20261020-r4, frozen WAM v4 step80000,
  same-execution measured feedback and task10/WAM goal, no training reference.
- recipe:686 inputs/12 actions,1024/1024/512/512 ELU, joint differentiable CAD,
  AdamW3e-4/wd1e-5, globalbatch2048, standardized L1/clip1, startup_half,
  seed247. Preserve step0 in DEV selection, bounded best/latest only.
- environment: graspenv Python3.8/Torch2.4.1+cu121; two cache shards GPU6/7,
  BC GPU6, later paired evaluation GPU6/7; leave GPU4/5 idle.
- bounds: cache6h, fitting2h, each native screen<=900s/300controls, no PPO/video,
  owned STOP and20GiB free disk floor; whole run output<=500MiB.
- evaluation: same6081 active DEV rows including invalid tau; fixed
  s1_toothpaste_lift/s1_stamp_lift/s1_waterbottle_lift/s1_apple_lift, seed246.
  Compare initial physical snapshots explicitly, and distinguish lift5cm/30frame
  proxy from full TaskController completion. Best/endpoint predeclared; no
  checkpoint reselection using the native outcomes.
