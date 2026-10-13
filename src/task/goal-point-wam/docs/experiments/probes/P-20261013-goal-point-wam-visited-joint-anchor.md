---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-visited-joint-anchor
experiment_id: P-20261013-goal-point-wam-visited-joint-anchor
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 10e7b6f9
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 43
seed_pool: probe
seeds: [292, 295, 296]
decision_changed_if_positive: deploy this bound actor/R pair and keep joint anchored updates
decision_changed_if_negative: stop clocked joint fitting and change the reference-time contract
status: UNPROMISING
run_id: wam-visited-joint-anchor-20261013-r1
---

# Joint actor/R update with a GT-tau action anchor

Result: COMPLETED 3000 updates in 609s on GPU2. No checkpoint passed the
offline gate, so there is no GT qualification and no actor rollout.
Local recipe UNPROMISING.
Decision: Stop this clocked joint update. Do not deploy the pair. The next
change is the reference-time contract, not more epochs or a lower R step.

## Decision Note

The frozen-R realignment lowered training-takeover PD error but the partial
542-step rollout never lifted the object (max +2.4 mm, 61% clipping) before a
one-voxel workspace fault killed the server. Separately, updating R without a
GT-tau action anchor previously moved nominal GT PD from 10.6 mm to 44 mm.
This probe asks whether updating the actor and R together, while the recorded
action of the true hand trajectory still supervises R, can reduce predicted
command error without losing that inverse map. A pass changes the deployed
pair. A GT-map regression or another no-lift rollout stops this clocked
recipe; the reference-time contract is the next change, not more epochs.

Same 19 recoverable own-row takeovers, nominal and human pools, and the
admitted realignment actor plus visited R. Scene, World, and R normalization
buffers stay frozen. No World loss. Losses are trajectory MSE, predicted-tau
action loss through R, and detached GT-tau action loss into R only, equal
weights. One idle GPU, 3000 updates, 1500 s. First-step gradients must enter
the actor and R from the predicted action, and R only from the anchor.

Offline admission requires the training-takeover predicted PD mean to
decrease, nominal/startup/exact GT-tau PD p95 to stay within 1.1x+.1 mm, and
the existing dev/human/clipping guards. Only a passing pair gets the 128-step
GT qualification and, if that gate passes, one 542-step actor rollout.
Success is actor hold of at least 45 frames. Live scene points more than
eight outside the fixed grid still fail the query; fewer are dropped so one
boundary point cannot close the process.

No new branch. Default deployed checkpoints stay untouched until this pair
passes both gates.

## Completed result

GPU2, code started from `10e7b6f9` plus the uncommitted joint trainer.
3000 updates, 608.7s, 1835MiB, utilization about 15–38%. GPU released.
First-step gradient sums: predicted actor 4827, predicted R 3769, anchor
actor 0, anchor R 3860. Frozen scene/World weights and R normalization
buffers stayed unchanged. Every recorded checkpoint from step 250 through
3000 failed the gate. The saved step2000 pair is diagnostic only.

| Panel | Initial | Selected step2000 |
| --- | ---: | ---: |
| Exact TRAIN predicted PD mean | 1.335 mm | 2.489 mm |
| Startup GT PD p95 | 1.516 mm | 2.490 mm |
| Startup predicted PD p95 | 1.419 mm | 3.481 mm |
| Nominal GT PD p95 | 14.599 mm | 10.725 mm |
| Contact TRAIN predicted PD mean | 22.438 mm | 13.807 mm |
| Exact DEV GT PD p95 | 227.806 mm | 288.004 mm |

Exact DEV GT p95 rose at every later checkpoint and was 300.0 mm at step
3000. Nominal GT improved throughout. The anchor therefore retained the
nominal inverse map and did not retain startup precision or the held-out
contact state. Predicted takeover error never fell below its 1.335 mm start.
No 128-step GT qualification and no 542-step rollout were run.
