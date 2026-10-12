---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-startup
experiment_id: P-20261012-goal-point-wam-ref2-startup
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: ed53315d
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 19
seed_pool: probe
seeds: [293, 294]
decision_changed_if_positive: run one bounded nominal closure with the fixed final startup head and unchanged teacher/GT calibration rows
decision_changed_if_negative: inspect startup per-tick trajectory/PD errors without launching another adaptation recipe
status: PROMISING
run_id: wam-ref2-head-startup-20261012-r1
---

# Startup coverage before live WAM-R continuation

Decision Probe for the ref2 point-trajectory -> frozen retargeter -> native PD
route. The previous live prefix had identical frame0 observations but WAM PD
XYZ error225.27mm versus teacher, measured wrist drift25.32mm after one action,
and no held frames before the workspace stop at172 controls. At identical
frame0 state, privileged offline replacement of first4/8 tau rows reduced
WAM versus GT-R PD discrepancy225.80mm to38.53/14.59mm. Startup trajectories
therefore materially affect initial control. Original300-update uniform fit
sampled only0.625% tick<3 rows; a bounded continuation tests whether increased
startup coverage can reduce initial PD mismatch without harming real-H4
forecasts. This tests practical usefulness, not a causal sampling effect versus
matched uniform continuation.

## Fixed protocol

Initialize trajectory/observed-hand adapter from the authenticated previous
fixed300 head, same human WAM step10000 base. Freeze scene/PTv3, goal encoder,
action/world and original physical R. Train only trajectory and hand_query,
300 additional updates, seed293/model and294/draw, AdamW lr3e-4 wd.01,
batch128, grad clip1, actor BF16/R FP32, frozen/eval dropout convention.
Same loss: Huber tau/.05 plus1*Huber encoded native PD/R target_scale versus
true encoded PD/R target_scale. Wrist6 are relative to current measured q;
finger independent coordinates absolute. No architecture/loss change.

Every training batch has exactly32 rows sampled uniformly from ticks0:16 and
96 from16:519, with replacement, env0:16 only. Record actual counts tick<3 and
<16. Use original qualified teacher train/validation/report split, untouched
test env28:32. Inputs are measured H4 plus planned object endpoint at t+24;
future hand is labels only, not an input. No live rollout data enters this fit.

Retain identical64 validation windows env16:20 and held report64 env24:28,
initial/final only. Add all first16 ticks x validation env16:20 =64 startup
windows, with per-tick trajectory and PD versus GT-R/actual teacher reports.
Fixed final update300 is the sole saved head; no report-based model selection.
Raw initial/final predictions, labels, PD/action and ticks/envs are saved.

Predeclared preliminary signal: startup validation WAM versus GT-R immediate
PD XYZ p95<=20mm, and held real-H4 nearest8 trajectory EPE<=25.966mm
(previous20.773mm times1.25). If both pass, root may run exactly one nominal
three-row closure using the new frozen head; full physical calibration and
hold/placement outcome remain separate. Otherwise diagnose, no extra recipe.
Original10mm/.05rad shadow screens remain recorded separately and unchanged.

## Engineering, resources and stops

New static entrypoint `src/task/trajectory-policy/tools/run/calibrate_wam_robot_startup.py`
derives the earlier calibration implementation while preserving that recorded
source file unchanged. Authenticate old head/source/results, base/model/stats,
qualified teacher readiness/training_allowed/audit, and all five frozen hashes.
RAM/GPU cache only, head-only checkpoint<10MB, no optimizer/base copy. Existing
bridge gains an optional authenticated head directory propagated to both
runtimes and matched against peer head SHA; default remains previous head.
Native physics/controller/reference and sim seed292 remain unchanged.

GPU2 only; training plus at most one later closure<=10min. Launcher accepts a
remaining deadline<=600s to enforce the combined allowance. All previous/new
ref2 outputs and owned scratch counted in aggregate100MB guard; no video.
Stop on nonfinite, source drift, foreign GPU process, frozen gradient/hash
change, failed cache/direct equivalence or resource limits. Own processes only.
Engineering smoke is at most one update and saves no checkpoint.

## Results

Fixed final300 continuation completed with sourceed53315d in68.25s
(monitor including final artifact guard69.82s). Predeclared preliminary signals
both passed, **PROMISING for one physical continuation**, not control success.
Startup validation PD XYZ p95 versus GT-R199.878 ->18.341mm (<=20), versus
actual teacher199.332 ->20.246mm. Held real-H4 nearest8 EPE16.478mm is below
the fixed25.965923mm allowance; held overall nearest8 improved28.020 ->17.217mm.
Startup validation nearest8 improved34.246 ->13.118mm. Startup per-tick PD
GT-R p95 is13.890mm at0; largest per-tick value19.345mm at8. Original
10mm/.05rad screens remain failed: held PD XYZ p95=21.139mm, wrist-axis
p95=.06281rad, finger-coordinate p95=.05402rad; clipping0. Held true-PD
XYZ p95=25.556mm and GT-R baseline remains reported separately.

Sampling exactly9600/38400 first16 rows (25%),1817 tick<3 rows (4.73%).
Both trainable modules received gradients; frozen scene/goal/action/world/R
initial/final hashes exactly match prior300 source. Cache/direct maximum
passed, authenticated base/model/stats/source/head unchanged; fixed final
only, no optimizer/base copies. CUDA peak allocated2456MiB, physical sampled
memory3133/24576MiB; a steady training sample showed90% GPU utilization.
Combined ref2 artifacts/private scratch~16.43MB, below100MB.

Artifacts: `outputs/trajectory-policy/wam-ref2-head-startup-20261012-r1/`
contains manifest/result/predictions/monitor and head-only checkpoint SHA
2b7eff74991a8ef306b192747d716210d4348d3259f628fa73c78dfc7b617fc2.
Root independently recomputed rawstartup PD18.341158mm andheldreal-H4
16.478322mm; sampling/frozen/source audit passed. Root permits the sole
follow-up `wam-ref2-execution-20261012-r3` under the original execution card,
seed292, same teacher/GT-R/WAM-R native protocol with this fixed head and
remaining530s deadline. Physical continuation completed and remains separate: teacher/GT-R each
held441 frames and passed controlled placement; WAM held0, clipped4.059% and
failed placement. Execution Probe's exact-current-initialization result is
UNPROMISING. Initial executed PD improved225.27 ->11.71mm, wrist state1
25.32 ->1.65mm, but visited-state drift returned before contact (first wrist
>10mm at9, PD>50mm at12). This Probe is still PROMISING only for its fixed
nominal startup-fit criterion; it did not establish successful control.
Full evidence/early forecast-dependency diagnostics and limits are in the
execution card and `outputs/trajectory-policy/wam-ref2-execution-20261012-r3/`.
No extra fit or control followed. Fit plus closure183.50s and total ref2
artifacts/private scratch~21.8MB remain within declared600s/100MB.

## Limitations / future evidence

A single continuation is not a matched sampling ablation or general control
validation. Startup coverage does not supply recovery-state trajectories;
closed-loop visited-state drift may still fail even after nominal startup PD
improves. One motion/seed and semantic robot landmarks do not establish human
retention, real robot transfer or multi-motion generalization. Held report is
observed only at fixed initial/final, and the broader trajectory/PD screens
must not be relabeled from this preliminary startup signal.
