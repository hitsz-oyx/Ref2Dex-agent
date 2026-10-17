---
schema: ref2dex.probe.v2
experiment_id: P-20261017-direct-tau-executor
task: trajectory-policy
branch: main
git_commit: e8ea6019
date: 2026-10-17
classification: Decision
status: UNPROMISING
hypothesis: A closed-loop policy can use a future GT tau plan as a high-level plan and emit direct active Inspire actions without the old tau-to-R-to-PD inverse route.
question: Does direct pi_exec(state, GT tau future8, task goal) preserve grasp/lift/hold on an unseen corrected s1 sequence, and does an oracle interaction context C recover it if not?
decision_changed_if_positive: advance from GT tau to WAM-predicted tau; treat tau as a usable high-level plan.
decision_changed_if_negative: do not start PPO or WAM-C training from this probe; use the oracle-C comparison to decide whether interaction context is the missing input.
protocol: BC on official applied native actions from the audited s3 teacher rollout, then frozen direct-action execution with 8 official teacher calibration environments and 8 direct environments on corrected unseen s1; oracle-C arm adds current 10D contact/support context.
train_source: outputs/trajectory-policy/competent-pd-teacher-20261011-r2/trajectory.npz
unseen_source: outputs/trajectory-policy/gt-tau-new-sequence-20261017-r4/teacher/trajectory.npz
unseen_sequence: s1_airplane_lift
action_contract: policy emits 12 active native coordinates; passive coordinates are zero and Inspire coupling is applied once by DExplore.
input_contract: measured state87 + future hand tau8 in the live object frame + task goal pose12 + previous applied active action; oracle-C arm appends contact pair/gap, support gap, object velocity6 and table footprint.
gate: direct arm requires hold45 >= 6/8, success >= 6/8 and clipping < 1%; teacher is calibration only.
runtime: /home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train, Torch 2.0.1+cu118, GPU 0 with pre-existing 0.4 GiB context.
---

## Motivation and decision note

The prior physical probe showed that the old `GT tau -> R -> PD` route does not
hold a coordinate-corrected unseen sequence. This probe tests the proposed
replacement directly: the policy reads the live state every control step and
outputs the native action, while GT tau is only a plan. The cheapest useful
discriminator is one BC fit, one unseen closed-loop run, and one oracle-C arm;
there is no PPO or WAM-v4 training before that discriminator.

## Offline BC

The fit uses `applied` native teacher commands and selects active indices
`(0,1,2,3,4,5,6,8,10,12,14,15)`. The training rollout contains 32 parallel
copies of one s3 sequence, so these are not 32 independent demonstrations.
The direct arm reached active-action L1 0.0155 on the held-out environment
split and 0.1121 on the unseen s1 source replay. The oracle-C arm reached
0.0156 and 0.0796 respectively. These are offline diagnostics, not task
success evidence.

## Closed-loop results

On unseen corrected `s1_airplane_lift`, the teacher calibration was 8/8 hold45
and 4/8 success. The direct GT-tau BC arm was 0/8 hold45, 0/8 success, median
held 0, clipping 0. The oracle-C arm was also 0/8 hold45 and 0/8 success,
median held 0, clipping 0. Therefore the registered decision result is
`UNPROMISING` for this fixed BC contract. The oracle-C arm did not recover the
failure, so this particular 10D contact proxy is not evidence that C restores
the missing behavior.

As an implementation diagnostic only, the same-sequence s3 run produced 6/8
hold45 and 4/8 success for direct BC, while its teacher was 8/8 hold45 and
8/8 success. This separates a working closed-loop action path from the much
larger sequence-generalization failure on s1; it is not a same-motion claim.

Artifacts:

- BC: `outputs/trajectory-policy/direct-tau-bc-20261017-r2/`
- unseen direct: `outputs/trajectory-policy/direct-tau-execution-20261017-r2/`
- same-sequence diagnostic: `outputs/trajectory-policy/direct-tau-execution-s3-20261017-r1/`
- oracle-C BC: `outputs/trajectory-policy/direct-tau-oracle-c-bc-20261017-r1/`
- oracle-C unseen execution: `outputs/trajectory-policy/direct-tau-oracle-c-execution-20261017-r1/`

## Limits and next action

This is a single-seed Probe. The teacher source duplicates one s3 motion, and
the oracle C is a compact current contact proxy rather than a learned or
full-force interaction representation. The result does not refute tau as a
plan in general, but it does not justify starting PPO from this BC checkpoint
or training WAM-C yet. The next useful route is to inspect sequence-normalized
tau/state representation and, if retained, define a richer interaction oracle
before any PPO budget is spent.
