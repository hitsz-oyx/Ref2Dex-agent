# P-20260925-cm-crossobject-history-value

date: 2026-09-25
branch: agent/cm-conditional-grasp-value
classification: Decision
status: UNPROMISING for unseen-object ranking

## Question

Does the short-history, action-conditioned Cm learned from randomized
interventions on the cross-object training partition retain treatment-effect
ranking on an unseen object (apple)? This is a representation/generalization
test only: the official actor generated these transitions and is not a final
policy.

## Protocol

Fit the same four frozen variants and four separate outcome heads as
`P-20260925-cm-history-grasp-value` on
`agent_expert_crossobject_randomized_train_s186_h5` and evaluate without
refitting on `agent_expert_crossobject_randomized_apple_s187_h5`. Both runs
are completed randomized `+/-z=0.1` interventions with a five-step follow-up.
The state/history construction, candidate scoring, quartile thresholds and
environment-cluster bootstrap are unchanged. Input SHA256 values are pinned
in the run manifest.

## Decision gate

The route is `PROMISING` for cross-object conditional action information only
if the history-action model has a positive high-minus-low gap of at least
5 mm with a 95% cluster-bootstrap lower bound above zero on an unseen-object
continuous lifting target, and its gap exceeds the action-shuffled control.
Otherwise classify it `UNPROMISING` for this transfer route. Either result
does not establish policy utility or justify PPO integration.

## Budget and limits

CPU two threads, GPU 0, wall <= 30 min, output < 20 MB. This remains a
population effect under sequential randomization, not an individual
counterfactual; no final held-lift label is present.

## Result

Run `agent_cm_crossobject_history_value_probe_20260925` completed with the
same 800 updates and 1000 environment-cluster bootstrap resamples. The model
used 992 randomized rows from the official-actor train partition (`s186`) and
971 rows from unseen apple (`s187`).

The history-action model did not pass the unseen-object physical ranking gate:

| outcome | high-minus-low | 95% cluster CI |
| --- | ---: | ---: |
| one-step object z displacement | +1.70 mm | [−1.79, +4.81] |
| five-step object z displacement | −6.00 mm | [−15.68, +4.02] |
| contact-supported z displacement | −4.22 mm | [−14.81, +6.12] |
| five-step contact fraction | +0.0009 | [−0.0270, +0.0291] |

The action-shuffled five-step z gap was −8.31 mm [−18.67, +2.32], so the
failure is a lack of positive unseen-object effect ranking rather than proof
that the shuffled control is better. The predeclared gate is false. This
contrasts with the single-airplane self-trained split and makes the current
signal object/distribution specific. Do not proceed to an online selector
based on this representation; retain the result as evidence for redesign.
