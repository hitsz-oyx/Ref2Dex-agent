---
schema: ref2dex.probe.v2
probe_id: P-20261024-c2-matched-clip
experiment_id: P-20261024-c2-matched-clip
date: 2026-10-24
task: ptv3-shared-wam-v6
branch: main
git_commit: recorded-in-receipt
classification: Decision
claim_id: c2-regression-is-shared-trunk-update
hypothesis_family: nine-source-supervision-distribution
probe_index_in_family: 4
seed_pool: probe
seeds: [170]
decision_changed_if_positive: retain shared-trunk updates as the working main-cause account on C2 realized support
decision_changed_if_negative: reject shared-trunk updates as the main cause on C2 realized support
status: PLANNED
run_id: c2-matched-clip-20261024-planned
---

# C2 matched flow clipping ablation

This card registers one clipping intervention following
`P-20261024-c2-flow-old-workspace`. GPT confirmed the operator as
OPERATOR_OK. This card does not authorize a launch. Neither arm has started.
No recipe is written to NAS and no launch command is registered here.

## Question and minimum decision experiment

Does freezing the shared trunk remove most of the C2 regression when both
arms use the same independent clipping rule for `flow_future`? The completed
C2-flow-r2 used a different clipping coupling from unrestricted C2, so it did
not identify the shared-trunk claim. The minimum discriminating comparison
is two matched 500-update arms, with all other conditions held fixed.

Both arms use the same C2 realized support and replayed sample path, the same
local3000 weights-only initialization and fresh optimizer, the C2 source
weights and config, world size 4 and global batch 384. Voxel origin remains
[-2, -2, -2] and voxel size remains 0.005 m. Learning rates, warmup and the
configured `clip_grad` value are unchanged. C4-source1 and C2-flow-r2 are not
rerun. There is no workspace-origin change. The known source4 window
s4-34183 tick 61 remains outside C2 realized support; conclusions apply only
to that support, not to a deployable nine-source repair.

## Arms and fixed operator

| Arm | Recipe switches | Trainability |
|---|---|---|
| Unrestricted | matched_clip: true; flow_only: false | All parameters trainable |
| Flow-only | matched_clip: true; flow_only: true | Only names starting with flow_future. trainable |

For each update, call `clip_grad_norm_` separately on (1) trainable parameters
whose names start with `flow_future.`, and (2) all remaining trainable
parameters. Both calls use `max_norm=config['clip_grad']`. Neither group's
norm includes the other group. In flow-only the second group is empty and
must not be passed to the clipping operator. Both arms therefore apply the
same grouped clipping rule to `flow_future`.

Record each group's pre-clip norm per update in `result.json` and at progress
writes in `progress.json`; an empty trunk is null, not zero. With
`matched_clip`, the existing `preclip_grad_norm` list and `grad_norm` field
are the `flow_future` pre-clip return values, not a combined norm. Clip rate is a
diagnostic only and is not an outcome gate. `matched_clip` defaults to false;
when false the existing single global clipping call and output fields remain
unchanged. The flow-only freezing rule is unchanged.

## Decision rule, only after separately authorized execution

Only the flow-only arm relative to local3000 determines the result. The
unrestricted arm supplies the matched intervention comparison and has no
additional outcome gate. Metric order is the same three panel metrics as
the preceding C2-flow card. Use paired mother-cluster CI95 on the fixed panel.

| Observed result | Verdict | Decision |
|---|---|---|
| All three residuals are <= +1.218 / +4.706 / +0.570 mm, respectively, and all three mother-cluster CI95 upper bounds are < +1.827 / +7.059 / +0.855 mm, respectively | PASS | Retain shared-trunk updates as the working main-cause account on this support |
| All three residuals are >= +1.827 / +7.059 / +0.855 mm, respectively, and all three mother-cluster CI95 lower bounds are > 0 | FAIL | Reject shared-trunk updates as the main cause on this support |
| Any other outcome | Not identified | No route verdict |

Status remains PLANNED. No measurements, training authorization or launch
are implied by this card.
