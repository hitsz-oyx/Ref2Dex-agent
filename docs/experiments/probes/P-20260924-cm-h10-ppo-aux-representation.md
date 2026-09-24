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

Pending.
