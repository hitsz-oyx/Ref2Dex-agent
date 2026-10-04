# P-20260924-cm-h10-ppo-aux-representation

Date: 2026-09-24. Branch: `agent/cm-h10-ppo-aux`. Classification: Decision.

## Question and decision

Does the frozen, simulator-trained multi-axis H10 Cm improve held-lift
when used only to supervise PPO's shared actor representation? This
distinguishes Cm policy utility from offline effect prediction.

Both arms resume the identical self-trained e260 actor and use the
same reward, curriculum, optimizer, training data, auxiliary head,
teacher computation and random seed. Cm-on has auxiliary coefficient
0.002; Cm-off uses 0. The 12 teacher targets are ±0.1 wrist-x/y/z
central contrasts of predicted ten-step world object xyz translation
and contact fraction, normalized by 0.02 m and 0.1 respectively and
clipped to [-2,2]. Supervision is only at real current contact and
where all candidate actions remain unclipped. Teacher is frozen and
never used to override action or reward; evaluation loads actor only.

## Pre-registered minimal test

Engineering smoke: on/off e260→e262, 16 env, confirm exact resume,
finite teacher/head gradients, and checkpoint actor compatibility.
Then matched 64-env e260→e300 for training seeds 77 and 78, evaluated
on new seeds 166 and 167, 64 first full episodes per cell. Primary
metric: held-lift successes. Fixed source checkpoint SHA
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`;
teacher SHA `3023a6f9715d1b2dbbb0c82b38184e6c84affd57987f6ba4d3d786477b188174`;
motion manifest SHA `2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`.

If first training seed Cm-on minus off is ≤−8pp across both eval seeds,
stop before seed 78. Upgrade toward target-shuffle placebo and formal
Validation only if aggregate on-off ≥+8pp and neither training-seed
total is negative. Do not tune coefficient or target from these eval
seeds. Otherwise classify `UNPROMISING` or `UNCLEAR` and return to
higher-level design. One GPU per run, at most two concurrent; target
≤60 min and ≤5 GB output for this Probe. Stop on input/commit drift,
unsafe GPU occupancy, nonfinite gradients or budget overrun.

## Result

Status: `UNPROMISING` for this fixed H10 multi-axis train-time
auxiliary route; not a refutation of Cm or its offline physical effect.
Implementation commit: `805ada75bcee831b538064d8dd7475763de8fc18`.

Both 16-env e260→e262 smoke runs completed, with a finite nonzero
Cm-on head gradient (`0.0006144` at epoch 261). Actor state-dict keys
matched between arms; the auxiliary head is saved separately.
The four 64-env e260→e300 matched training runs and all eight
64-env full-episode evaluations completed. Each run's command, SHA256,
checkpoint, status and metrics are recorded under the corresponding
`outputs/Dexplore/agent_cm_h10_aux_*` directory.

| Training seed | Eval seed 166 on/off | Eval seed 167 on/off | Combined on/off |
| --- | --- | --- | --- |
| 77 | 39/64 vs 30/64 | 36/64 vs 30/64 | 75/128 vs 60/128 (+11.72pp) |
| 78 | 29/64 vs 40/64 | 39/64 vs 33/64 | 68/128 vs 73/128 (−3.91pp) |

Overall: 143/256 vs 133/256, +10/256 = +3.91pp, below the
pre-registered +8pp threshold. Training seed 78 is negative, so the
per-seed nonnegative condition also fails. At epoch 300, auxiliary
loss on/off was .342/.554 for seed 77 and .369/.533 for seed 78;
this confirms target learning, not policy benefit. The large
train-seed reversal is incompatible with claiming a stable Cm policy
advantage from this Probe. No target-shuffle placebo or formal
Validation was run because the upgrade conditions were not met.

Decision: stop this auxiliary target/coefficient. Do not tune on
seeds 166/167. The next route, if pursued, must change the decision
horizon or supervision alignment, rather than perform local coefficient
search. A higher-cost sequence-conditioned Cm route remains a separate
decision; see `docs/archive/2026-10-04-research-governance/decisions/D-20260924-after-multiaxis-probe.md`.
