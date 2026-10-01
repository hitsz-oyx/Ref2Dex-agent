# First direct matched policy-learning screen: no Cm-specific utility

Frozen implementation34f78b6, parent `P-20261002-support-feature-policy-training-r1`.
All9216fresh training trajectories529--540 and1536fresh evaluation541/542 complete,
202native ticks each. Three seed752macro heads share initial weights/probabilities,
fresh off-policy data,12fullbatch updates, optimizer and baseline. Short physical
probabilities are features; actual physical105 is the only reward. No greedy
model selection, model reward, source actor action/weights or teacher fallback.

Parent runtime is **FAILED900.343s at its final ANALYSIS**, not during training or
physics. All14per-panel physical/PD/current-state/assignment audits and all12
feature/logit/score-gradient/Adam audits passed. Original scripts, log and manifest
remain unchanged. Corrected analysisdb96433is separately **COMPLETED** in
`analysis_correction_r1`; no new training or physics was run.

The original final analyzer conflated independently reconstructed clearance with
same-input neural arithmetic. Maximum raw clearance discrepancy2.491288e-8m,
within the fixed1e-6raw tolerance, became2.491474e-5after normalization by.001.
V2keeps all tolerances and first checks raw states, then replays the ACTUAL saved
native inputs with independent NumPy math. Maximum feature error1.063828e-6,
logit error7.516463e-6, exact argmax agreement. This repairs the numeric audit
without changing inputs, policies, physical outcomes or scientific gates.

| Fresh all-motion policy | Physical105 count /384 | Rate |
| --- | --- | --- |
| Unchanged initializer | 131 | 34.115% |
| Trained Cm features | 151 | 39.323% |
| Trained state-only features | 153 | 39.844% |
| Trained privileged-global features | 140 | 36.458% |

Cm minus trained state=-0.521pp, versus trained global=+2.865pp. Required>=5pp
against BOTH fails. Each-seed noninferiority also fails:541Cm79/state77/global70
outof192;542Cm72/state76/global70. **UNPROMISING**, stop this exact feature-transfer
learner without additional updates, seed/head/LR/deployment-rule selection.
The older failed prediction/noise/position/disturbance gates stay failed.

| Motion | Initializer /128 | Cm /128 | State /128 | Global /128 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 114 | 113 | 105 | 111 |
| 2 | 17 | 38 | 48 | 29 |

Crucial mechanism audit: query all three already saved head logits on EVERY
one of1536pre-decision states, not only their assigned groups. All heads choose
the SAME deterministic action on ALL1536states: unchanged on motions0/1and
curl+.15rad on motion2. The trained probabilities/weights can differ, but their
tested executable decision rule does not. Different randomized cohorts and
native trajectories yield different finite success counts; those counts cannot
support a Cm-specific decision advantage. This is observed agreement, not a
theorem of stochastic-policy or unseen-state equivalence.

Final head checkpoint SHA256
`a3bbf6f54abe1daa87f7fd1b06f4b1b5bc9c9be783953a291674fff0b70f719d`.
Pretraining4608physical trajectories plus1000updates per neural physical predictor
is separate from9216shared RL collection trajectories. The base scratch policy
also uses19392original teacher rows; no overall data/compute saving is claimed.
All traces and intermediate u00..u12heads/optimizer packets remain retained.

This is a genuine bounded, off-policy contextual macro-policy learning comparison,
with reference-conditioned base control. It is not continuous RL, reference-free
manipulation, formal multi-training-seed Validation, generalization or hardware.
The original journal objective remains unmet: distinctive methodology and actual
Cm training utility are not established. The next route must create and model
physically meaningful conditional decisions beyond a per-motion constant action;
more fitting or Validation of this same decision rule is not justified.
