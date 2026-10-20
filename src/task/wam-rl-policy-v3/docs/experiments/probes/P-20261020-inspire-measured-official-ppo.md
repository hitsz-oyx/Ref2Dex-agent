---
schema: ref2dex.probe.v2
probe_id: P-20261020-inspire-measured-official-ppo
experiment_id: P-20261020-inspire-measured-official-ppo
date: 2026-10-20
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 14
seed_pool: probe
seeds: [246, 247, 248]
decision_changed_if_positive: continue the measured goal WAM policy improvement route
decision_changed_if_negative: diagnose measured training dynamics before adding scale or changing the method
status: RUNNING
run_id: inspire-measured-official-20261020-r1
---

User requests starting PPO now onGPU4/5/6 concurrently with independent
trajectory visualization onGPU7. This is an explicit resource allocation for
this run, not a general change to CAMPAIGN. Main branch, frozen code manifests.

Decision: can the actually measured goal+WAM observation support policy
improvement after latest BC? The user requests enough training scale, so a
two-env wiring check is only a Blocker check, not a substitute for training.
Then three ranks with512 environments each,1536 total, matching the previous
Shadow six-rank256 total and global12288 transitions per eight-step rollout.
Global minibatch3072 after four minibatches; five epochs. Use pinned official
UniDexGrasp2 PPO collector/storage/GAE/update, actual-distribution KL correction,
global advantage statistics and gradient averaging already used by Shadow.
No DExplore PPO, reward or hand-reference actor input is used.

Source: corrected r3 teacher-rollout admission, all501 TRAIN sequences/50 objects.
Tasks/events derive from teacher object rollouts. BC source is corrected-data
Scheduled686 DEV-best(step0), retaining learned Current421 goal columns; this
is the declared DEV selection, not post-hoc endpoint selection. Added tau
weights are zero initially but trainable during PPO, with actual WAM80000 input.
CAD encoder is frozen at the BC parent so rollout features do not drift between
sampling and update. Actor/critic MLP1024/1024/512/512 ELU; actor BC initialized,
critic fresh as Shadow; native12 actions/30Hz remain Inspire-specific.

Official optimizer settings: lr3e-4 adaptive desiredKL.016, clip.2, gamma.96,
lambda.95, entropy0, gradient norm1, initial noise_std.8 (preserve upstream
squared-scale sampling convention). Reward is existing causal MeasuredRewardTracker
with the same gamma. Full TaskController task, not imitation of hand references.

Reset contract: execute each action in its observed episode, collect terminal
reward/done, explicitly reset only completed native environments, clear only
their task/reward/history/previous-action state, and return the new observation
before sampling the next action. Actual native clock/episode IDs are checked.
Two-env probe forces different environments to terminate atsteps2/4, then real
official PPO updates; training never uses this engineering forced termination.
Terminal GAE does not bootstrap from a reset state.

Run limits:10k PPO iterations or48h, own STOP at update boundary,20GiB disk
floor, bounded checkpoint retention (last3 plus500 milestones). Rank0 only
saves checkpoint; other ranks log progress. Physics and policy subprocesses
share their assigned GPU; no unknown processes killed. All native motion
sampling is restricted to the admitted TRAIN membership. Runtime graspenv
Torch2.4.1+cu121 policy and pinned dexplore_v120_train Torch2.0.1+cu118 physics.
Log first iteration throughput/memory/ETA and actual multi-episode reset results.
Training success is full TaskController completion, kept distinct from lift
proxies; no WAM gain or held-out generalization claim from this run alone.
