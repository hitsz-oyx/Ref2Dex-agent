# P-20260924-temporal-contact-cm

- Classification: Decision Probe, CPU only.
- Sources: existing self-trained train3 e320 first-episode transitions
  (airplane/mug/toothpaste fit, seed178) and apple first-episode transitions
  (held-out object, seed174). Apple has been used in prior exploration and
  is not a pristine formal validation object.

## Question and decision

Earlier state/action H20 contact Cm did not gain action information on
held-out apple. Does a ten-step physical interaction history reveal
contact persistence *and* make the current action informative enough to
justify a temporal Cm actor/critic? Fit matched small heads for (a) current
state+action, (b) current state+action+ten-step history, (c) the same history
head without current action, and (d) history head with current actions
shuffled within training object. Predict next-20-step contact fraction and
contact-supported object-z displacement from pre-action transitions.

Use only first-episode, complete-history and complete-future rows at actual
current hand/object contact. Sample at most 1000 rows per train object and
1000 apple rows, with fixed RNG and train-only feature normalization.
The history action-aware model must lower heldout apple contact RMSE by
>=10% versus both current-state action-aware and history action-blind,
and beat the shuffled-action control. If so, train a temporal Cm for a
matched policy auxiliary test on fresh data. If history helps but actions
do not, treat it as a state-estimation signal only; do not claim a policy
decision role. If history does not help, stop this simple ten-step summary.

CPU <=20 minutes, <50 MB output. Stop on missing/invalid transition
provenance, episode-boundary leakage, nonfinite values or unequal sample
contracts. This offline Probe cannot establish Cm policy utility.

## Results

The CPU Probe completed with 3000 first-episode current-contact training
rows (1000 per train motion) and 1000 heldout apple rows, all with full
ten-step history and twenty-step future. Heldout future-contact RMSE:

| Head | RMSE |
| --- | ---: |
| Current state + current action | 0.3822 |
| Temporal history + current action | **0.3320** |
| Temporal history, no current action | 0.3686 |
| Temporal history, shuffled training action | 0.3847 |

History improved over current state by 13.1%, but action-aware improved
over the same-history no-action head by about **9.9%**, just below the
predeclared >=10% gate. More importantly, heldout contact-supported lift
RMSE (scaled by 0.3 m) was 0.1917 action-aware versus **0.1739**
action-blind. The joint action-information gate failed. Status:
`UNCLEAR` for a general temporal Cm representation and `UNPROMISING` for
immediate online attachment of this exact current-action head. Do not tune
on apple or upgrade to PPO from this Probe. The result suggests that recent
interaction history carries state-estimation information, but does not
establish a helpful candidate-action signal for lifting.

Report and frozen fitted heads:
`outputs/CmResidual/agent_temporal_contact_cm_train3_apple_20260924/`.
