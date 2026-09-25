# P-20260925-cm-route-local-residual

- Classification: Decision Probe.
- Cm checkpoint: `outputs/CmLite/V1.37/balanced_s1x24_s3_e160_e180/best.pt`
  (SHA256 `396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a`).
- Substrate: the fixed six-expert object route on the 12-motion,
  ten-object self-trained mixture. This is a strong diagnostic substrate,
  not a full-GRAB policy.

## Question and matched arms

Can Cm improve a policy that already has sustained held-lift behavior when it
is used as a local action residual after five consecutive hand-object contact
steps? The Cm-off arm executes the fixed object-route expert. The Cm-on arm
uses the same route and checkpoint, then ranks five candidates consisting of
that action plus deterministic small wrist/finger residuals. The route and
expert-ranking arms use the same simulator, seeds, motion root and evaluation
definition.

The earlier expert-ranking attachment allowed Cm to switch among all six
experts on every contact step. It is retained as a negative control because it
tests a materially different, more invasive policy intervention.

## Results

For the local residual, matched first-episode results were:

| seed | Cm-off | Cm-on local + stable gate | difference |
| ---: | ---: | ---: | ---: |
| 229 | 24/64 | 25/64 | +1 |
| 230 | 27/64 | 24/64 | −3 |
| 231 | 24/64 | 19/64 | −5 |
| 232 | 21/64 | 23/64 | +2 |
| **total** | **96/256** | **91/256** | **−5 (−1.95 pp)** |

The local selector replaced the route action on 10,908–12,677 of roughly
67,000 scored environment steps per run. Therefore the implementation is
active and auditable, but the four-seed Probe is `UNPROMISING` for policy
utility under this checkpoint and residual parameterization.

The invasive expert-ranking control gave 22/64, 24/64 and 17/64 on seeds
229–231, versus 24/64, 27/64 and 24/64 off, for 63/192 versus 75/192.
It is a clear negative control and should not be used as the paper's Cm
接入 claim.

The same ranking implementation on the weaker 59-motion lift-like pool gave
7/128 for both Cm-off and Cm-on across seeds241/242. That result is not a
positive policy-utility signal either.

## Decision and boundary

Do not promote either online action-ranking or local residual Cm to a formal
positive claim. The full 660-motion uniform actor remains an unsuitable
substrate, while the specialist route is useful for matched diagnostics.
Future Cm work must change the information path or training objective (for
example, an explicitly trained contact-supported credit signal) and must keep
the fixed route, exact seeds and same checkpoint as Cm-off controls.
