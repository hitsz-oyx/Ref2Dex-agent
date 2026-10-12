---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-recovery-geometry
experiment_id: P-20261012-goal-point-wam-ref2-recovery-geometry
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 754ff3b0
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 26
seed_pool: probe
seeds: [295, 296, 292]
decision_changed_if_positive: collect new actor's actual states after nominal retention and cross-takeover geometry screens
decision_changed_if_negative: broaden true early expert recovery support before further backbone thaw
status: UNCLEAR
run_id: wam-ref2-head-recovery-geometry-20261012-r1
---

# Direct expert recovery geometry before another capacity change

## Decision Note

Ref2_1 requests true autonomous-state aggregation prior to consequence/RL.
PD-only actor fits damaged nominal controls, and R has imperfect mapping of
physically measured early recovery trajectories. R.net adaptation fits its
training segment but fails original rotation-retention gates; these failed
R candidates are not deployed. Their physical qualification is NOT required
to test whether an actor can learn real expert recovery geometry directly.
Keep original qualified R for any actor execution; other R weights are only
privileged offline diagnostics.

Root chooses a fixed direct-tau supervision probe, no PD gradient through R.
The question is whether measured expert future supervision on recovered own
states transfers across takeover16->32 while retaining nominal geometry.
This separates physically valid trajectory learning from numerical inverse
R fitting, and informs data coverage versus actor capacity. It does not
claim to cover pure autonomous ticks9..15, which lack an expert future label.

Cheapest test reuses audited existing actual takeovers. OneGPU2, smoke1 then
fixed500 updates/final only, <=600s/stage, ref2 outputs+scratch<=100MB and
shared storage<=300GB. No new branch. Stop on drift/nonfinite/foreign GPU2
process/budget. A negative geometry result motivates broader early recovery
support, not automatic full backbone thaw. No new permission boundary.

## Independent derived actor admission

Preserve source actor-future restrictions and mixed-policy deny. The newer
t32 manifest has actor_future_hand_training_allowed=false; legacy t16 lacks
that explicit key and its original admission allowed expert PD only. Neither
absence nor old training_allowed=true grants actor future labels.
NEW actor-specific admission authorizes only
role3 actually measured hand future whose entire24 causing controls are
actually applied unclipped expert. Never use role2 future, copied/forked
state, nominal teacher future as recovery label or cross handoff windows.
Reconstruct actual expert PD from actual action/current measured q; bind
source trace/manifest/result/root audit hashes. Each H4 belongs to its own
source/role, including its observed WAM prefix at handoff.

Training source old recovery-r2:542/t16, independent audit9768/max0, trace
SHA b0edec52551bc9e7f528c80022f7aad946793d02e450fc234bac8eabde8c32ee.
Only current16:96 (80 windows), next24 true expert controls. Expert held435
overall but final placement failed; this LIMITED approach/grasp segment does
not admit late placement examples or claim complete expert recovery.

Cross-takeover diagnostic source aggcollect-r1:192/t32, root3468/max0,
traceSHA27da16275bdbc2c482ac6bfa164bf206dba4781cebc0c6b07c9a01d180329102.
All137 windows current32..168 stay OUT of this actor training. Unlike R's
temporal split, this is a whole-takeover diagnostic, still the same reset and
motion and previously inspected evidence, not independent generalization.
Nominal train8304/env0:16, original held64 panel/env24:28; test28:32 untouched.

## Actor and loss

Warm original startup head2b7eff74991a8ef306b192747d716210d4348d3259f628fa73c78dfc7b617fc2.
Same Git-authenticated human WAM/backbone/H4+plannedobjectgoal contract;
scene/goal/action/world frozen, trajectory/hand_query trainable. Actual
current/H4 points alone feed scene; planned object reference min(t+24,542)
feeds goal. No q, expert observation, teacher hand future or time as INPUT.
Targettau=(actual role3 hand[t+1:t+25]-currenthand)@currentobjectrotation,
native24x33 semantic order. Left hand absent. Targets remain separate from
input placeholders; world forward is not called.

Batch128:nominal64(startup16/later48)+expert-recovery64 replacement.
loss=Huber(pred_nominal/.05,tau_nominal/.05)+
Huber(pred_recovery/.05,tau_recovery/.05). No PD loss or R backward.
lr1e-4/wd.01/clip1, fixed500 updates, seeds295/296 reused for deterministic
recipe continuity, not a replicate. Check real recovery trajectory gradient,
five frozen module/R hashes, cached/direct equality and consumed input hashes.
Smoke1 saves no checkpoint. Retain raw tau/PD/actual labels/row identities.

## Screens and possible next collection

Report train80 and cross-takeover137: h1/nearest8/all24 EPE, wrist h1,
separate early32:48/48:64 and later64:169. Report original frozen R true-PD
XYZ/rotation/finger/clipping; direct geometry improvement does not imply
control accuracy because R is itself imperfect in these regions.
Also report pure-WAM original query first32 actual expert-PD discrepancy,
without inventing future hand labels for it.

Offline geometry screen: cross-takeover nearest8 EPE<=.5 initial and
nominal nearest8<=21.522mm. Any new actor collection additionally requires
nominal original-R PD<=50mm, nominal clipping<1%, all finite/hash checks.
If admitted, one192-control/takeover8 collection queries expert on NEW actor's
actual states; actor row remains autonomous, teacher intervention only role3.
Keep original qualified R and explicitly label partial/local calibration;
no placement success verdict. New actor's PD and actual consequence/held
behavior determine further aggregation. Optional R overrides not used here.

## Limitations / future evidence

Different takeovers at the same reset are not independent task/seed evidence.
Early trajectory labels may rely on teacher-only privileged references absent
from the actor; direct supervision can still be conflicting. No human/world
quality retention established by frozen weights alone. Consequences are
retained for later robot Cm learning; this is supervised geometry, not RL or
Cm causal utility. Full task and matched RL/Cm Validation remain deferred.

## Results

Result: direct geometry fits training and retains nominal; cross-takeover nearest8 improves23.70->19.63mm but fails predeclared halving, so r1 is not deployed.
Decision: aggregate the audited t32 true expert geometry into training explicitly, then diagnose a fresh actor's actual states under nominal retention gates; retain all original failed screens.

Engineering smoke1 update23.74s/PASS, direct recovery gradient910.165,
R_backward_used=false/cache-direct0/frozen hashes unchanged. Fixed500
754ff3b0 completed121.14s/GPU2: nominal nearest8 17.222->15.874mm,
PD25.498->29.020mm/clip0. Recovery train80 nearest8 26.909->11.207mm,
all24 41.494->11.886mm. Cross137 nearest8 23.695->19.635mm,
all24 33.775->27.586mm; early32:48 nearest8 71.158->56.382mm.
Cross original-R PD60.601->52.759mm/clip.7299%->0. Pure original-query32
PD234.937->224.270mm. Geometry-halving failed; no r1 physical collection.
The head learns actual geometry without the prior nominal PD collapse, but
unseen more-severe recovery retains substantial error. Probe stays UNCLEAR.

## Follow-up Decision: true recovery geometry aggregation

Training80 fitting and retention distinguish direct physical trajectory
supervision from prior PD-only inverse fitting. The t32 trace contains exactly
the harder own expert states still missing from geometry training. Explicitly
RETIRE cross137 into training; it is never again held-out actor evidence.
Warm the fixed r1 final and run ONE500-update continuation on old80+new137
true expert-only windows. Each batch nominal64 (startup16/later48) + recovery64
(old32/new32). Same loss/lr/seeds/frozenmodules/originalR. Both actor-specific
derived admissions must now state trainingallowed and previous cross role
retired; source policies/data/old r1 results remain immutable.

This tests whether broader physically valid support can be fitted without
losing nominal controls. Within-source errors are training diagnostics only.
Do not reuse the old cross-halving screen as an independent result. The NEW
evidence is a fresh192/takeover8 collection from THIS fixed learned actor,
whose actual visited state/control/consequence and expert queries are saved.
That diagnostic collection requires nominal nearest8<=21.522mm, original-R
PD<=50mm, clipping<1%, finite/source/frozen checks. The relaxed purpose is
explicit: query a new state distribution, not validate recovery from loss.
Original r1 did not pass its original screen and is not deployed retroactively.
No new R, no full placement verdict; original qualified R on both GT/WAM arms.
This is one additional collection/continuation within600s/stage/100MB limits,
then review true autonomy/recovery support before another fit.

## Actual consequence engineering preparation

In parallel, prepare a small world-only view of the existing192-step trace,
without scientific training. Future own hand points are EXECUTED world
conditions, never actor expert targets; actual future object pose is an effect
label in fixed current-object coordinates. Derive an engineering-only scoped
view with scientific_training_allowed=false, retain all original deny fields.
Do not use GoalWindows, which auto-generates actor future targets and future
object goals. World9D is [future point, displacement from current, frame
increment], all in meters; it is not native18D control or velocity/acceleration.
This is a feedback-affected hand-motion proxy, not yet a native-action causal
model suitable for RL candidate evaluation. Independent CPU frame/unit/leak
checks and eight bounded raw samples (<1MB) prepare the later integration;
they do not prove world quality, grasp benefit or RL/Cm utility.
