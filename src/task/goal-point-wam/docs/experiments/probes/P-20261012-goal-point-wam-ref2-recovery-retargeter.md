---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-recovery-retargeter
experiment_id: P-20261012-goal-point-wam-ref2-recovery-retargeter
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 77ab8e14
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

Result: fixed1000 updates reduce recovery train53.22->6.05mm but held17.99->19.74mm; startup rotation retention gate fails, so no deployment.
Decision: test ONE predeclared alpha=.5 interpolation of base/final net weights with all original screens unchanged; no repeat training or rewrite of the completed fit.

Fitter77ab8e14, engineering1 update5.13s/PASS; fixed1000 updates64.58s,
GPU2/peak54MiB. Nominal XYZ p9510.59->9.02mm, startup2.28->.959mm.
Recovery train53.22->6.05mm, held17.99->19.74mm and earlyheld19.63->21.29mm.
Clipping0 everywhere; original buffers bitwise unchanged and initial net exact.
Startup rotation p95.002769->.004442rad exceeds1.5x, so offline_screen_pass
false. Final checkpoint is preserved and does not enter physics.

### Follow-up Decision: conservative weight interpolation

The strong training change with slight held worsening suggests local overfit
or too large an adaptation; failure is not a mapping-capacity proof. Cheapest
check is a single alpha=.5 parameter interpolation between authenticated base
and fixed1000 final R.net. No new optimization or selection over a grid. Keep
six buffers identical and all original gates unchanged; fresh derived output
links parent checkpoint/result/raw predictions and scoped admission. GPU2
forward only<=60s/<2MB, same local correlated diagnostic panels. If it passes,
admit the already-declared local GT physical calibration; if not, investigate
recovery support/trajectory supervision before further R investment.

The single interpolation (generator605f6755) also failed unchanged gates:
recovery train31.998mm, held16.721mm/earlyheld18.026mm; nominal11.195mm,
startup3.673mm. Nominal rotation .023772->.053259rad and startup
.002769->.005525rad fail retention. Both final R and interpolation remain
offline diagnostics, neither deployed. Optional strict --r-dir integration
e91be0d5 is available; root33 CPU admission/scene/partial/partition checks PASS.
Root independently recomputed experiment24 raw metrics and actual-query PD
labels, saved root_metric_label_audit.json: all errors0, identities match.

### Follow-up Decision: cross-takeover physical-future diagnostic

Before another adaptation, cheaply distinguish temporal fit from transfer to
a different actual recovery. Existing recovery-r2 source542/t16 has root
audit9768/max0/SHA b0edec52551bc9e7f528c80022f7aad946793d02e450fc234bac8eabde8c32ee.
Evaluate base, final1000 and half R on role3 current16:96, each next24 controls
actually expert; compare source teacher row0 same ticks, fixed16-tick blocks.
GPU2 forward only<=60s/<1MB. This is privileged diagnosis only, allows no
training or physics and does not reinterpret failed complete placement.
If new R degrades this independent-takeover early recovery, collect broader
true recovery support before further R training; if preserved, reassess the
practical rotation constraint and actor trajectory supervision. Same motion/
reset and privileged expert still limit independence/generalization.

Cross diagnostic completed6.03s (fixed22a49ef4). On actually measured t16
expert recovery80 windows, base/full/half XYZ p9544.272/31.027/33.897mm;
same-trace teacher17.310/21.942/19.412mm. Early recovery16:32
38.445/35.406/32.924mm, transition48:64 72.662/49.404/54.076mm. There is
limited cross-takeover change, but nominal degradation and original failed
gates remain; neither candidate is deployed. All PD current-q reconstruction0.
Initial engineering preflight rejected legacy audit's missing takeover_tick;
fixed ONLY missing legacy key by using fixed traceSHA+manifest+actual applied
expert mask, still rejects contradictory explicit key. Failed preflight log
retained separately; it is not method evidence. Original R is used for the
separate direct actor-geometry probe26; R is only a frozen forward diagnostic.
