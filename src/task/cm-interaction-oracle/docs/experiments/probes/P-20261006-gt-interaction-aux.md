---
schema: ref2dex.probe.v2
probe_id: P-20261006-gt-interaction-aux
experiment_id: P-20261006-gt-interaction-aux
date: 2026-10-06
task: cm-interaction-oracle
branch: agent/cm-gt-interaction-aux
git_commit: a63c7ef
claim_id: C3
hypothesis_family: HF-gt-interaction-aux
probe_index_in_family: 1
seed_pool: probe
seeds: [292, 293, 294]
decision_changed_if_positive: advance training-only action-conditioned representation to longer matched Probe
decision_changed_if_negative: stop this fixed auxiliary recipe without new Cm fitting or tuning
status: PLANNED
run_id: gt-interaction-aux-s292
---

# GT interaction supervision of PPO actor representation

## Root Decision Note and motivation

User ref15 authorizes a new training-only Cm route after frozen critic ranking:
GT supervision directly shapes actor z, and PPO still learns action quality.
This supersedes previous PPO pause for this bounded route, not final Mission.
Decision experiment: does this cheap h8 action-conditioned auxiliary improve
trained actor stable grasp and same-z source-return readability? Auxiliary
prediction alone cannot answer the decision. No learned teacher, candidate forks,
ranking selector or future inference input. No core claim change.

Choose existing source_e260 native actor/critic checkpoint, preserving its
architecture and source reward/optimizer identity rather than introducing a new
actor architecture. Shared z is existing actor512 latent shared with auxiliary;
native PPO critic remains a separate trunk. Same-z value diagnostic is a frozen
readout on a common source-policy pool, NOT shared-critic PPO or a claim about
current policy advantage accuracy. Source-policy finite-episode MC vs trained critic is
policy-mismatched and truncates timeouts where PPO bootstraps and reported only as a separate diagnostic.

## Frozen four-arm contract

A plain PPO(lambda0); B conditioned GT auxiliary(lambda.05); C same auxiliary
with whole action chunks cycled between valid same-episode windows at identical rollout
clock; D same B with z detached from auxiliary. All four construct identical
512+144→128→19 decoder in private CPU RNG context, without altering CPU/CUDA
rollout RNG. Decoders do not enter actor inference/model state_dict. D trains
head but aux encoder gradient is exactly zero. A/D actor behavior should remain
matched apart from numerical simulator variability, audited explicitly.

H is the native1442 current actor observation including existing reference and
history features, no future. h8 at30Hz, actual executed normalized requests
s[t]→s[t+8]. GT19 contains object positiondelta/.03(3), sign-invariant relative
quaternion small-angle vector/.5(3), change in object minus configured-hand-link
centroid/.03(3), endpoint five force-gated contacts(5), their changes(5).
Contacts: handlink force>.1 and object force>.1. This inexpensive GT target is
physical object effect and interaction, not task Y or stable Z supervision.
No hand FK/learned Cm or privileged GT at inference.

Masked mean SmoothL1 loss, equal dimensions and fixed physical scales.
Discard any window touching done (including endpoint), and last7rollout ticks.
Targets/action chunks/masks aligned through env-major flatten and PPO dataset;
chunks use task.actions after actual step, invert its in-place finger [0,1] mapping
back to normalized requests, checked against clipped Gaussian
request, not actor mu or unclipped PPO sample. No reset/rollout stitching.
C retains H/GT/masks and action marginals, cycles entire8×18chunks by one
between valid envs per rollout; no invalid/cross-reset donor is used; mapping reused for all six mini-epochs, no RNG draws.
Feedback chunks are post-treatment; prediction quality does not prove causal
counterfactual understanding. B>C alone also cannot rule out shuffle label noise.

## Training, endpoints and minimum decision gate

Source checkpoint SHA25616fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f,
source_e260. Four single-seed292 arms, each64additionalepochs(e260→e324),
64env×32horizon=131072fresh transitions/arm, mini256×6updates/epoch=3072updates.
Same source PPO gamma.99/tau.95/clip.2/critic5/bounds10, LR1e−5; source2approach,
10held_lift,5progress/no link bonus, original source env and the three ref14 canonical airplane motions.
Source_e260 itself was trained on one motion; this is the same three-motion
continuation distribution for all four arms, not identical source data scope.
Reuse completed contact backtrack schedule180→220, fractions.5contact/.25lift,
window±3, already progress1. No added hold/drop reward. Only final checkpoint
and standard latest copy, no best-model selection, no epoch/horizon/λ sweep.

Actor-only deterministic evaluation: all four final checkpoints and source on
96frame0 first episodes(three motions, no early termination), seed293. New
cohort, not ref14 selected32anchors. Initial motion/start/object/q/root contract
must match all arms. Track contact/lift45consecutive ticks(1.5s) and later drop
(remaining episode), as well as legacy5tick success. Primary sustained success
is45tick success with no later drop; also report stable45 and drop separately.
Paired counts and descriptive bootstrap(seed294) do not replace Validation.

Common independent source-policy evaluation pool provides complete discounted
source2/10/5 MC returns. Sample observations atstride8, retain actual h8chunk
and GT; split by env_id modulo4 with balanced motion strata, never random rows.
Frozen actor z from each arm: fixed standardized ridge .01, fit using75%envs,
evaluate25%; predict source MC V return and realized-return-minus-sourceV
(residual, not actual trained-policy advantage). Also compare GT heldout
SmoothL1 and action-chunk shuffle sensitivity; native critic MC diagnostic
kept separate from z readout. GPU batched inference/linear solves, no policy
fitting on evaluation pool. Complete pool and protocol frozen before comparison.

PROMISING only if B primary sustained success≥A/C/D+3episodes, B stable45≥each,
and same-z heldout MC readout MSE improves≥5% vs A and C, with valid nonzero
encoder intervention and the ≥5%heldout GT-learning gate below. Otherwise UNPROMISING for this fixed recipe if training
and auxiliary have demonstrably learned (heldout B GT loss below its source
initial decoder, ≥5%); UNCLEAR if short training/coverage/diagnostics cannot
distinguish hypotheses. This is route screening, not global Cm refutation.

## Resources, verification and stop rules

Engineering smoke: one additional PPO epoch for each arm, ≤600s/2GiB total,
source/dev seed42, no scientific results. Main four arms on idleGPUs1–4,
one worker/GPU; ≤1800s wall and≤7200GPU-seconds. Evaluation/audit≤900s wall,
≤3600GPU-seconds, total≤8GiB. Frozen commits/input hashes before launches.
Stop on nonfinite, wrong gradient/RNG/action/mask, unpaired initial contract,
unknown GPU conflict or cap. No interference withGPU0 or external read-only data.

## Limitations / future evidence

One seed, short64epoch update, three known airplane motions and one exposed
checkpoint. Source rollout MC is policy-dependent; cannot establish optimal Q
or cross-policy calibration. z readout is diagnostic, not PPO critic replacement.
If positive, longer matched PPO and multiple-seed independent Validation remain
required. Final matched trained-policy Cm-on/off utility remains OPEN.
If negative, independent implementation review precedes closure; no λ sweep,
new learned Cm or continuation just to improve auxiliary loss.


## Engineering repair before scientific execution

First smoke at88a0eb9 stops before any optimizer update after47.20s: native
Inspire mutates task.actions[:,6:] into [0,1] PD fractions. Stored PPO requests
remain normalized[-1,1]. Correct capture inverts this known mapping after step
and checks against clipped PPO requests (atol2e−7, float32 inverse rounding),
without changing actual actions, PPO likelihoods or labels. All failed outputs
retained. Also explicitly choose ref14's three canonical motions for all arms;
first smoke used the one-motion source training root and supplies no scientific
comparison. New smoke uses the final three-motion contract.

Second smoke at701aa32 captures correct actual actions, then stops on the first
reset: wrapping live env_step in inference_mode turned mutable history into an
inference tensor, incompatible with later native reset. Use no_grad instead
(the original rollout semantics); no label/score/architecture change. Neither
failed smoke reaches PPO optimization, outputs retained and costs charged to
600s engineering cap.

Third smoke at98128f2 reaches first PPO minibatch then audit stops: torch.compile
joint actor/critic autograd materializes unused critic gradients as exact-zero
tensors rather than None. Minimal two-independent-trunk aot_eager reproducer
(tmp/reproduce-compiled-aux-gradient.py) shows nonzero actor grad/zero critic
grad and fails the old None-only guard in3.07s. Check exact norm0 for critic,
nonzero actor for B/C and zero actor for D; scientific gradient gate unchanged.
Only D completed its single epoch; other arms stopped before optimization.
No scientific result from the failed campaign.

R4 smoke completes all four e261 /2048frames /48updates in34.29s, matching
initial source/RMS/decoder/physics hashes. B/C applied actor gradient>0, D=0,
critic aux gradient0, all tensors finite. However A/D first-rollout valid mask
differs before updates; final native states are not exact. Hypothesis: default
torch.compile stochastic graph changes initial sampling schedule across cold
processes (alternative GPU solver variability). Disable the official compile
option uniformly, retain all study variables and use first-rollout hashes as
a sharper paired smoke gate before scientific training. Cost charged to same
600s smoke cap; no scientific method conclusion from these checks.

R5 eager smoke completes in31.16s but first-rollout states/actions/dones hashes
still differ across arms before optimization; disabling compile does not remove
the problem. Therefore use already validated native GPU PhysX/CPU tensor
pipeline, one PhysX CPU thread, aligned immutable reference tables, GPU
actor/decoder/geometry. This is a uniform simulator execution-contract repair
for reproducible pairing, not an auxiliary variable. Source2/10/5 formula
explicitly moves physical arguments to GPU before computing identical bonuses;
no source tracker/drop shaping. Evaluate under the same tensor pipeline. Check
all four first-rollout hashes and A/D final native weights/RMS before main.

R6 stops during initial training reset before source restore: task native loader
keeps reference tables onGPU even with CPU tensor pipeline. Align immutable
tables before native reset, including constructor reset, using a process-local
wrapper around the existing initializer. Engine-owned buffers are not moved.
Move info terminate to PPO device as an adapter. All original outputs retained.
Independent design review also identified invalid donor windows in naive C.roll;
final C only cycles among valid windows at identical time, preserving conditional
action marginals and no reset-crossing donors. Fixed before scientific launch.
Complete-MC diagnostic is finite-episode, timeout-truncated; PPO critic uses
bootstrapping at timeout. Report this target mismatch, do not infer critic bias.

R7 localizes remaining device mismatch: vendor create_rlgpu_env unconditionally
sets args.device=cuda when --horovod is enabled, even with --pipeline cpu.
Each arm is already one GPU/process; remove the legacy --horovod flag uniformly
so tensor device remains CPU, while actor/decoder/geometry stay GPU. Source
optimizer is restored intact; no actual distributed averaging is required.
Engineering artifacts may retain up to2GiB within unchanged8GiB total, preserving
failed runs and matched smoke checkpoints; global Campaign boundary unchanged.

R8 first full-state/history/obs/reference plus CPU/CUDA RNG hash and first
Gaussian action hash are EXACT across all four arms. It stops at later reset
because PPO done_indices live onGPU while task reference tables are CPU.
Convert only env_reset indices to native task device before delegating; reward,
actions, reference and epoch contracts unchanged. No optimizer update.

R9 completed at a28cd30 in29.13s: all four first-action input (fullsim/history/
obs/reference/CPU+CUDA RNG), first requested action and entire32step first
rollout states/actions/dones/masks EXACT. Independent training audit PASS:
all e261/2048frames/48updates, source model/RMS/decoder initialization equal,
B/C actor aux weight-gradient nonzero, D0, critic aux0; A/D entire native model
and RMS EXACT after optimization. This passes wiring AND pairing. All earlier
GPU-pipeline smoke comparisons excluded from scientific interpretation.

Full-episode smoke first attempt fails before stepping: player's symbolic
CUDA device has no index whereas live tensors report cuda:0; geometry bridge
requires exact device identity. Construct it using actual model parameter
device and move state there. This preserves GPU geometry and source formulas.
Training already uses indexed ppo_device and is unaffected.

Full-episode smoke r2 reaches first native step then detects mixed-device done:
player returns done on RL GPU while evaluator reward/tracker state use CPU
native tensors. Explicitly move done to native task device for physical
statistics; actor and geometry remain GPU. No completed scientific evaluation.
