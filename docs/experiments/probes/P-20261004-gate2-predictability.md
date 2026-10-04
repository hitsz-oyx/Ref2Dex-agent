# P-20261004 Gate 2 consequence predictability

## Question

Can a deployable-input proxy `(H_t, A_t) -> (E_hat, I_hat)` retain the
long-return information that Gate 1 observed when the value bridge receives
ground-truth future effect and interaction?

This is an offline Decision Probe. It uses the audited all18 Gate 1 dataset,
does not replay physics, and does not train a policy. The current `H` contains
the assembled physical snapshot/reference context; deployability of every H
field has not yet been audited, so this is not a final deployment claim.

## Design

For each held-out checkpoint namespace, a small GRU/MLP predictor is trained on
the other namespaces. It predicts all 32 future steps of effect (13 channels)
and interaction (80 channels) from the 10-step history and current action. A
value bridge is trained on GT E/I in the training namespaces. On the held-out
namespace, the same bridge is evaluated once with GT E/I and once with the
predicted E/I. The primary preservation quantity is the fraction of the
GT-E/I improvement over the H-only bridge that remains after substitution.

## Results

| Run | Holdout episodes | Consequence predictor | H-only MAE | GT-E/I MAE | predicted-E/I MAE | GT gain preserved |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| n3, 8 epochs | 134 | effect -0.04%, interaction +0.20% vs train-mean RMSE | 11.864 | 9.603 | 12.885 | -45.1% |
| n3, 32 epochs | 134 | effect -0.19%, interaction +0.09% | 11.864 | 9.603 | 10.786 | 47.7% |
| n1, 32 epochs | 224 | effect -0.49%, interaction +0.23% | 15.327 | 9.610 | 14.055 | 22.3% |

The 8-epoch run was a capacity screen, not a convergence claim. Extending it
to 32 epochs changed n3 from harmful to positive, so the short run is not used
as a route-level conclusion. At 32 epochs, both namespaces show a positive
value gain over H-only, but the raw consequence predictor remains close to a
train-mean baseline and preserves only 22–48% of the GT-E/I gain.

Artifacts:

- `tmp/P-20261004-gate2-predictability-ns3.json`
- `tmp/P-20261004-gate2-predictability-ns3-e32.json`
- `tmp/P-20261004-gate2-predictability-ns1-e32.json`
- implementation commit `881b9bf`

## Decision

**Gate 2 status: UNCLEAR, with a local positive value-preservation signal.**

The result justifies one cheap follow-up on consequence-model capacity or a
lower-dimensional/temporal decoder. It does not justify PPO, distillation, or
an online Cm experiment yet. It also does not refute the Gate 2 hypothesis:
the predictor is still a simple high-dimensional flat decoder, and H's final
deployment availability is not established.
