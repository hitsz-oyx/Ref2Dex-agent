---
schema: ref2dex.probe.v2
probe_id: P-20261008-official-generator-screen
experiment_id: P-20261008-official-generator-screen
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 13e749f
claim_id: C3
hypothesis_family: HF-consequence-official-generator
probe_index_in_family: 1
seed_pool: probe
seeds: [239]
decision_changed_if_positive: admit the identified official actor as an additional data generator for a separate bounded geometry-audited rollout collection
decision_changed_if_negative: retain the self-trained expert as the primary generator without an unbounded official-policy search
status: UNCLEAR
run_id: official-generator-screen-20261008-r3
---

# Can the archived DExplore actor supply useful airplane rollouts?

Result: Both64-episode arms completed. Official lift5/hold45 coverage is64/64
versus self-trained56/64 and24/64. Full-episode no-later-drop successes are0/64
official versus18/64 self-trained; the reference ends at table height.
Decision: Official grasp coverage is promising, but the original perpetual-hold
full-end gate failed. The user now chose complete reference with normal controlled
placing counted as success. A separately specified geometry pilot checks that
task before admitting value examples. No evaluator fit started.

## Purpose and evidence

This is a Decision Probe serving ref4 outcome-supervision coverage. The user
explicitly authorized locating and trying DExplore's archived policy in its
dedicated conda environment, and using it as a generator if it performs well.
The historical [expert physical data card](../../../../../../docs/experiments/probes/P-20260924-expert-physical-cm-data.md)
already distinguishes official collection from the final self-trained policy.
[Vendor provenance](../../../../../../third_party/DExplore/UPSTREAM.md) identifies
the external read-only release checkpoint, excluded from the vendor snapshot.

The official teacher is `/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth`,
SHA256 `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`.
Its 1442-dimensional observation and 18-dimensional control match the existing
native task. This compatibility does not prove good physical behavior.

## Decision Note and fixed protocol

Question: Is this fixed official actor a useful additional generator, compared
with our current airplane expert on the same inputs?

Action: Run 64 first, full-frame0 episodes per actor, one airplane motion,
seed239, deterministic actor control, GPU PhysX at30Hz, adaptive termination
and early termination disabled. Use the corrected inputs from
`outputs/consequence-evaluator/baseline-transfer-s3-20261007-r2/airplane_base/`.
Both arms use the same process-local GPU reset/FK repair and input configuration.
Official weights remain read-only and do not initialize any learned policy.

Criterion: At least3cm lift with native hand/object net-force proxies for45
consecutive frames, followed by no drop through the first episode end.
Drop means height below2cm or six consecutive lost-contact frames. This matches
the existing operational readiness screen; geometric labels are a later step.

Positive generator screen: official passes at least32/64 and is no worse in
qualified episode count than the paired self-trained control. This is a
screening decision, not a statistically established superiority claim. If
positive, a separate bounded ref4 collector must still export decision-known
24-step residual schedules, full suffixes and measured geometric diagnostics.
If negative, keep our expert as the primary generator and preserve the results.

Cost: One currently idle GPU0; maximum900s and1GiB per arm (1800s/2GiB total),
plus one native extension build in repository `tmp/`. Stop on occupied GPU,
source/checkpoint drift, nonfinite/incomplete clocks, invalid resets or budget.
No external environment/library/source modifications; no new branch or push.
The working archived runtime is `.runtime_envs/dexplore_v120_train`, with
Torch2.0.1+cu118, rl_games1.1.4 and compatible torch_cluster. User-site loading is
disabled. Isaac Gym is supplied via its existing package path. All installed
environments remain unmodified.

## Execution and provenance

Entry: [probe_official_generator.py](../../../tools/audit/probe_official_generator.py).
Actual executing commit, source hashes, Python/Torch/rl_games versions,
actor identity, resource bounds and checkpoint hashes are frozen separately in
each output `run_manifest.json`. Official runtime commit is `13e749f`; control
runtime commit is `3b31457`. The latter adds strict compiled-key translation
for our checkpoint. Full actual identities are retained in the manifests.

Run IDs: `official-generator-screen-20261008-r3` and
`self-trained-generator-screen-20261008-r3`, under `outputs/consequence-evaluator/`.
The actual self-trained rerun is `self-trained-generator-screen-20261008-r4`:
GPU0/1 became occupied by another user's jobs after the official arm finished,
so the control uses the idle same-model RTX3090 GPU2. The first control attempt
on GPU0 refused occupancy before creating output. The GPU2 r3 control stopped
before stepping because our modern PPO checkpoint uniformly wraps model keys
in `_orig_mod.` (torch.compile). R4 strips this wrapper in memory and loads all
model/RMS tensors strictly, without modifying the checkpoint. This adapter
does not alter the official actor's parameter names or preprocessing.
Native transition exports are diagnostic-only and have `training_allowed=false`;
native actions can be mutated by PD conversion and lack the geometry/known-plan
contract required by ref4 evaluator examples.

The preserved r1 attempt failed before physics: the user-site rl_games1.6.5
imports `torch.amp.GradScaler`, unavailable in dedicated Torch2.2.2. The archived
plan V1.17 identified older dedicated runtimes. Import probes further showed
the native distillation-task import eagerly requires torch_cluster, whose
available extension has an incompatible ABI with Torch2.2.2. The preserved
`.runtime_envs/dexplore_v120_train` provides the complete Torch2.0.1+cu118 /
rl_games1.1.4 / torch_cluster stack; r2 executes that environment directly.
Import probe logs are preserved in `tmp/consequence-official-generator/`.
No r1 episodes were generated. Each r2 arm is capped at800s, keeping the retry group
within1800s including the failed r1 import/extension-build cost.

The preserved r2 attempt also stopped before physics: the repo's newer native
evaluation factory expects `params` while rl_games1.1.4 passes `config`.
The r3 process-local facade forwards the same flat, already-built legacy config,
aliases the deterministic flag, and restores the old BasePlayer preprocessing
so external observation RMS is applied exactly once (the newer repo override
would otherwise apply it twice). A regression check covers factory invocation
and single normalization. This affects both matched arms equally; neither
vendor nor installed library files are edited. Each r3 arm retains the800s cap.

## Results and task-boundary audit

Same input motion and seed239,64 first-frame0 episodes per arm:

| Native physical diagnostic | Official | Self-trained |
| --- | ---: | ---: |
| >=3cm/contact for5frames | 64/64 | 56/64 |
| >=3cm/contact for45frames at any time | 64/64 | 24/64 |
| 45frames then no later drop through full end | 0/64 | 18/64 |
| Mean contact proxy fraction | 0.82994 | 0.41063 |
| Mean hand position tracking error (m) | 0.0777 | 0.3046 |
| Mean object position tracking error (m) | 0.0701 | 0.4087 |

Each first episode lasts542control ticks (18.07s). Official/control process
times were58.49s/58.14s; official GPU memory was18.2GB with approximately54%
utilization during the inspected sample. Both completed jobs released GPUs.
The native `success_rate=100%` is survival with early termination disabled,
and is not an additional stable-grasp result.

Raw source object positions are columns198:201 (not native processed
hoi_data106:109). The reference final height is initial height+0.000495m.
All64 official trajectories subsequently return below the2cm threshold.
This is a reference/task mismatch for a goal requiring perpetual holding:
normal reference-driven placing must not silently be relabeled as successful
holding, nor should the full-end failure count alone be called a bad tracker.

Source-only candidate boundary: the last reference frame elevated at least3cm
is tick481, preceding its final return. A post-hoc prefix audit gives official
63/64 versus self-trained18/64 no-later-drop successes **up to that boundary**.
These counts are diagnostic and do not pass the original full-end gate. A
fresh fixed-prefix reference can change the policy's privileged context and
requires a new frozen screen; no cropped data have been admitted. The user
explicitly chose the entire reference and normal placing as task completion.
The prefix proposal is superseded; no reference is cropped. Follow-up
[full-reference value data](P-20261008-full-reference-value-data.md) specifies
controlled placing geometrically and retains uncontrolled dropping as failure.

Reproduction audit: `tmp/consequence-official-generator/summarize_generators.py`.
Results and provenance: `outputs/consequence-evaluator/official-generator-screen-20261008-r3/`
contains `qualification.json`, `drop-audit.json`, `reference-height-audit.json`
and `matched-generator-audit.json`; control results are in
`outputs/consequence-evaluator/self-trained-generator-screen-20261008-r4/`.
The combined retained screen outputs occupy approximately28MiB.

Validation: Two focused regression tests passed (legacy factory/single RMS,
and strict uniform compiled-key translation). Actual native64-episode runs
verify loading, finite full clocks, FK reset continuity and post-run input
hashes for both actors. The force-only qualification artifacts are still
untrainable; successful prefix counts cannot replace geometric label audits.

## Limitations and future evidence

Single seed, one motion, net-force contact proxies and overlapping native
samples do not establish generalization, reliable collision labels, causal
action utility or the Mission's matched Cm-on/off benefit. Official-origin
pretraining data must remain separately identified if later used. No official
weights may become the final self-trained policy. Expansion to other objects
is deferred until the bounded airplane generator decision changes.
