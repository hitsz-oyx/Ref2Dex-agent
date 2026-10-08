---
schema: ref2dex.probe.v2
probe_id: P-20261008-official-generator-screen
experiment_id: P-20261008-official-generator-screen
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 8571971
claim_id: C3
hypothesis_family: HF-consequence-official-generator
probe_index_in_family: 1
seed_pool: probe
seeds: [239]
decision_changed_if_positive: admit the identified official actor as an additional data generator for a separate bounded geometry-audited rollout collection
decision_changed_if_negative: retain the self-trained expert as the primary generator without an unbounded official-policy search
status: RUNNING
run_id: official-generator-screen-20261008-r1
---

# Can the archived DExplore actor supply useful airplane rollouts?

Result: Pending a fixed matched generator screen.
Decision: Evaluate the existing official actor and the current self-trained
airplane actor; do not train policies or evaluators in this screen.

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
The runtime is `dexplore_repro_py38_torch222_cu121` with Torch2.2.2 loaded before
user-site package fallback. Isaac Gym is supplied via its existing package path;
installed rl_games1.6.5 is explicitly recorded rather than called upstream1.1.4.

## Execution and provenance

Entry: [probe_official_generator.py](../../../tools/audit/probe_official_generator.py).
Actual executing commit, source hashes, Python/Torch/rl_games versions,
actor identity, resource bounds and checkpoint hashes are frozen separately in
each output `run_manifest.json`. The protocol's base commit above precedes the
diagnostic entry; actual code identity is the runtime manifest's commit.

Run IDs: `official-generator-screen-20261008-r1` and
`self-trained-generator-screen-20261008-r1`, under `outputs/consequence-evaluator/`.
Native transition exports are diagnostic-only and have `training_allowed=false`;
native actions can be mutated by PD conversion and lack the geometry/known-plan
contract required by ref4 evaluator examples.

## Limitations and future evidence

Single seed, one motion, net-force contact proxies and overlapping native
samples do not establish generalization, reliable collision labels, causal
action utility or the Mission's matched Cm-on/off benefit. Official-origin
pretraining data must remain separately identified if later used. No official
weights may become the final self-trained policy. Expansion to other objects
is deferred until the bounded airplane generator decision changes.
