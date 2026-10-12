---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-recovery-retargeter
experiment_id: P-20261012-goal-point-wam-ref2-recovery-retargeter
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 25
seed_pool: probe
seeds: [290, 291, 292]
decision_changed_if_positive: admit a bounded physical GT-tau calibration with adapted R before further actor aggregation
decision_changed_if_negative: inspect R recovery coverage and nominal retention before changing backbone capacity
status: UNCLEAR
run_id: wam-ref2-recovery-retargeter-20261012-r1
---

# Retargeter coverage on actually executed expert recovery trajectories

## Decision Note

This serves ref2_1's robot adaptation and autonomous-state aggregation, prior
to actual consequence learning and RL. Experiment24's actual expert-only
recovery futures provide a new diagnostic that unrestricted tau inversion
could not: fixed R, role3 ticks32..168, all next24 controls actually expert.
Early recovery32:48 XYZ PD p95 is126.638mm versus9.705mm on the same nominal
teacher ticks; all137 recovered windows47.171mm versus nominal16.011mm.
Thus numerically unrestricted tau fitting did not exclude a physical R
coverage problem. Actor capacity alone cannot explain perfect-input R error.

Root chooses bounded R.net adaptation with original normalization and replay
of nominal teacher data. No full WAM backbone thaw. Admission remains derived
and R-only: never change the original source actor-future prohibition, never
use role2 future motion or a copied/forked expert state. This is reversible
and within existing authorization; no claim or resource expansion.

One GPU2, each stage<=600s; all ref2 outputs and scratch<=100MB, shared
artifacts<=300GB. One1000-update fixed-final fit, followed only if its screens
pass by one192-control/takeover32 physical diagnostic using the startup WAM
head and new R. Stop owned jobs on foreign GPU2 process, identity drift,
nonfinite data, deadline or storage bound. Failure means diagnose that local
mapping; success permits physical calibration, not autonomous success or RL.

## Data and scoped admission

Source wam-ref2-execution-aggcollect-20261012-r1/native:192 controls/193
measured states, exact roles teacher/gt_tau_R/wam_R/command-replay-expert32.
Raw source SHA27da16275bdbc2c482ac6bfa164bf206dba4781cebc0c6b07c9a01d180329102,
root audit3468 checks/max0. Actual expert action equals applied action and
actual PD equals query PD reconstructed from CURRENT measured q. Every future
control[t:t+24] is expert, so role3 hand[t+1:t+25] supplies physically measured
R input. Native semantic point order and current object-frame anchoring remain
unchanged. WAM inputs remain own observed H4 and planned object goal only.

137 recovery windows current32..168. Hold40:48 and96:112 (24), trainremaining113.
These adjacent windows have overlapping futures and the same reset, hence
are local coverage diagnostics, not independent generalization. All labels
are actual expert PD. Original nominal training env0:16/all519 ticks=8304;
nominal retention panel16 predeclared ticks x env24:28=64, untouched test28:32.
Training/held row identities and raw outputs are retained. Preserve original
source provenance even if historical producer code changed; authenticate
actually consumed data/audit artifacts and current execution dependencies.

## Fit and offline screens

Warm original physically qualified tau.pt SHA414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f.
Same schema,width256,use_tau,parameter shapes and SIX normalization buffers;
train R.net only. Seeds290/291 reused for deterministic recipe continuity,
not a replicate. Fixed1000 updates/final only, lr1e-4,wd.01,clip1. Batch128:
nominal64 (startup16/later48) + recovery64, replacement. Equal normalized
Huber PD losses using ORIGINAL target_scale. Engineering smoke1 saves no R.
Verify initial weights bitwise, nonzero recovery gradient, finite values and
unchanged buffer hashes; save same R schema checkpoint under a fresh path.

Predeclared screen: recovery held XYZ p95<=20mm, nominal held XYZ p95<=20mm,
startup nominal XYZ p95<=10mm; held rotation/finger p95<=1.5 initial.
Report earlyheld40:48 separately and clipping for every panel. Screens admit
only new physical GT-tau calibration; frozen head was trained with OLD R,
and this mismatch must be explicit in every execution manifest/READY check.

## Physical diagnostic if admitted

Authenticate new R and derived scoped admission, exact architecture/buffers,
base-R physical qualification, source and audit chain. Optional --r-dir must
preserve default old R behavior. GT-tau and WAM use SAME new R; teacher uses
original expert. Startup head remains2b7eff74991a8ef306b192747d716210d4348d3259f628fa73c78dfc7b617fc2.
192 controls/takeover32/execution seed292, all actual queries and consequences
retained. This new R must earn local grasp calibration (teacher and GT held45,
clipping<1%) itself; old R qualification does not transfer. Report pure WAM
behavior and actual expert PD discrepancy without declaring full placement.
No new actor fit under this card: any next adaptation needs its own protocol.

## Limitations / future evidence

One recovery episode and temporally overlapping windows cannot establish R
generalization. Expert queries rely on privileged training-only reference;
none enters the actor. A changed R may invalidate head control compatibility.
Training/GT diagnostic improvements cannot establish autonomous grasp or Cm
causal utility. Independent resets, complete542-step task evaluation and
matched RL/Cm Validation remain future evidence.

## Results

Result: predeclared R coverage probe; implementation and source admission in progress.
Decision: run engineering smoke then fixed1000 updates, and admit physical diagnosis only on the predeclared screens.
