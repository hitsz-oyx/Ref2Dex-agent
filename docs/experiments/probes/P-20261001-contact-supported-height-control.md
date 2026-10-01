# P-20261001-contact-supported-height-control

HF10 slot2/2, Decision Probe, new setup and fresh randomized actual control.
Frozen before scalar fitting or new simulator labels; detailed scientific
rationale, proper probability setup gate and causal controls are in
[decision](../../decisions/D-20261001-supported-height-calibration.md).

Input physical NN: contact-trajectory-model-r1/trajectory.pt,
SHA f7c0b0f95d2344947bd872e0ef518bc0fa49ebc592a9c154bb101f55ebdd22e8.
No neural parameter training, PPO, V or actor changes. New joint/release
positive logit-affine calibration uses historical cal only,3initial-frame
folds seed9841,100LBFGS iterations/head, regularization1e−4. Old physical
preparation MAE gate remains failed. Fresh action data determines utility.

Physical score: signed change in last3 jointly-contact-supported object
height above rest, relative to the observed current supported height. Height
trajectories remain signed, and newly acquired-then-released lift is included
in all-window release labels. Never condition evaluation on future acquisition.
Margin=max(.5mm,cal median absolute factual signed-score error); ensemble
relative-gain mean−std, joint-contact loss<=.02, release-effect mean+std<=0.
All models use the same rules, state-only has no numeric candidate input,
shuffled keeps independent numeric corruption; bestfixed uses uniform-fit
actual signed-height consequences only. No scalar/threshold choice uses new
rollout labels or historical old-held targets.

Proper setup gate and prospective gate stay separate. Crossfit joint Brier
<=.8constant, ECE<=.05, release Brier<=constant, class support, nonzero proposals
permit collection. Wholecal parameters are then frozen. Factual class
calibration does not certify candidate effects, risk bounds or policy utility.

Collection: simulator351–356/private allocation7351–7356,96envs each,8windows
per first episode,650ticks, native wall<=240s/parent<=290s. Contact proxy3steps
and history10; observe all5recommendations, randomize policy1/5, record actual
arm probability including duplicate recommendations, execute cached action2
then ownbase8, cooldown6, reobserve/replan. Entirely new rollout outcomes.
Existing six experts/base4/airplane motions read-only. Original windows remain
complete, no trimming/no hot state clones. Raw NN/calibration/source hashes
checked before/after. Total setup+collection<=60min/8GiB including failed runs.

Analysis: IPW common-action policy contrasts, episode/frame descriptive
intervals; actual different-choice matches separately. Cm must improve signed
supported-height>=.5mm overall4controls, base frame90%lower>0, release difference
upper90%<=.02allstates/<=.05initiallylifted, endcontact difference>=−.02;
intervention5–80% and32matched different-choice windows/15episodes per side
for each contrast. Insufficient support UNCLEAR; no threshold rerun. Record
action changes, proposal/actual coverage, replans and inference latency.
Positive permits policy-training design; mixed-history local effects alone
do not meet the final stable-grasp or matched trained-policy claim.

Setup completed on GPU4 in2.535seconds (code ac8aa8e): crossfit Cm joint
Brier .143864 vs constant .222624, ECE .030780; release Brier .009298 vs
constant .010216. Proper setup PASSED; original MAE gate remains FAILED.
Frozen artifact SHA027202015c32ba783aa1bbef5a0a3c501643bfa904e0b460cdf1877193971355;
physical NN weights unchanged. Cal proposal117/1297 (9.02%), margin1.358195mm,
bestfixed cup arm1. Details in supported-height-control-setup-results.json.

Native-command contract fixed before fresh outcomes: Inspire overwrites raw
channels7/9/11/13/16/17. Equivalent constant2step plans are merged using
12independent commands for actual propensity and all effect-support counts.
Log exact native18DOF candidate targets and executed targets at every step.
State-only6slots preserve expert identity and predict candidate-specific
physical futures; this control tests numeric action information, not absence
of all action-consequence modeling. Runtime/collection tests25passed.
