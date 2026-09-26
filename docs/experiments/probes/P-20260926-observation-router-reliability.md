---
schema: ref2dex.probe.v2
probe_id: P-20260926-observation-router-reliability
date: 2026-09-26
branch: agent/observation-router-reliability
git_commit: 33848ed3226c51fd4d2f9037ef41724c10906982
claim_id: C1
hypothesis_family: HB01
decision_changed_if_positive: "Keep the nonprivileged six-expert hierarchy as the C1 route and design fixed multi-seed Validation."
decision_changed_if_negative: "Do not validate this router; inspect whether expert-choice fidelity or physical grasp coverage failed before changing the C1 route."
probe_index_in_family: 1
seed_pool: probe
status: PLANNED
classification: Decision
---

# New-seed reliability of the observation-driven six-expert baseline

## Decision question

Can the already frozen initial-observation SVC preserve the self-trained six-expert route on a fresh simulator seed? The existing seed233 Probe passed 60/64 expert-choice fidelity and lifted 27/64 with observation routing against 21/64 with simulator-object-ID routing. One fresh matched seed is the cheapest way to decide whether that nonprivileged hierarchy merits formal C1 Validation. This is independent of the frozen Cm campaign.

## Frozen inputs and control

- Seed `260`, 64 first completed full episodes, the same 12 motions and six self-trained checkpoints in both arms.
- Control: fixed `simulator_object_id` route. Candidate: initial 1442-D actor-observation SVC, frozen model SHA256 `1fa84c94891d63824be0100b5eb78f4aa1e37825c6d71fff65e517c0c7f2fc14`.
- Route config SHA256 `6e3205a8b5fe2be3ab46308678e673d10e62745a7eda8196a7fe0e0d365bb7a8`; evaluator SHA256 `4f4feddb3bbad7d5b0dbd67848f4a9a88fa1923bdfcdeb316d5916640c6b6320`.
- Same motion directory, task/env/train configs, disabled early termination, checkpoint set and simulator seed. Run control then candidate sequentially on one idle GPU. The evaluator checks expert checkpoint and router model hashes before starting.

## Predeclared gate

Require both manifests `COMPLETED`, 64 first episodes per arm, equal motion IDs/start frames/episode lengths per environment, all finite, and no checkpoint/config/model drift. Then label this route `PROMISING` for C1 Validation only if:

1. at least 60/64 initial expert choices equal the fixed route, with every cup environment assigned to `cup_e340`;
2. observation-route held-lifts are no more than 6/64 below the fixed route;
3. observation-route held-lifts reach at least 20/64, preserving a nontrivial physical grasp signal.

If choice fidelity fails, mark the frozen classifier `UNPROMISING` and inspect the confusion matrix. If physical conditions fail despite fidelity, mark `UNCLEAR` about the router and revisit the expert substrate. Do not tune the SVC, seed, thresholds or expert set on this result. Passing is still a Probe, not proof of stable grasp or Cm utility.

## Budget and stop conditions

One idle GPU, sequential arms, target wall time <=10 minutes, output <=200 MB. Stop on occupied GPU, input drift, missing checkpoint, incomplete episodes, nonfinite values, excessive runtime or storage. Save a machine-readable preflight, both native evaluator manifests/results, and a compact result index. No new training, Cm model, PPO or collector.
