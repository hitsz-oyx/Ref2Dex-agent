---
schema: ref2dex.probe.v2
probe_id: P-20261010-structured-candidate-bank-audit
experiment_id: P-20261010-structured-candidate-bank-audit
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: fa174ec
claim_id: C3
hypothesis_family: HF-trajectory-conditioned-evaluator
probe_index_in_family: 2
seed_pool: probe
seeds: [401, 402, 403]
decision_changed_if_positive: collect a bounded candidate panel while preserving the exact actor history/observation contract
decision_changed_if_negative: keep H-to-tau candidate generation and evaluator/selector integration frozen
status: UNCLEAR
run_id: structured-candidate-bank-audit-20261010-r2
---

# Can the existing structured rollouts support a nontrivial candidate bank?

Result: `UNCLEAR`. The frozen structured rollouts contain useful candidate
diversity and outcome variation at later query ticks, but they do not save the
actor observation/history needed to establish an H-matched panel. This is a
bounded offline data audit, not an evaluator-ranking, selector, or native
execution result.

## Motivation and decision note

The matched trajectory evaluator Probe remains blocked by the candidate bank:
the existing HA bridge collapses to about 1 mm of spread, while the observed
rolling bank has much larger variation and still fails its information screen.
The cheapest discriminating question is therefore whether the already-frozen
ref7_2 structured rollout data contain locally similar current states whose
future hand trajectories and frozen U32 outcomes differ. If they do, it is
worth one bounded collection change that saves the missing actor history. If
they do not, more evaluator or selector work is premature.

This audit does not change the ref7_2 native R gate. The underlying rollout
packets were already collected and are only read here.

## Frozen input and screen

Inputs are the completed CPU-tensor/GPU-PhysX structured retarget splits:
96 train episodes (seed 401), 64 validation (402), and 64 test (403), each
with 542 commands and zero clipping. Reset rows are checked for bitwise
identity in object pose, hand keypoints, q, dq, and the saved pair proxy.

At ticks 8, 16, 24, and 32 the audit expresses the next 24 hand frames in
each episode's current object frame and computes the frozen 32-state U32
teacher label. A near-current pair is only an engineering proxy:

```
hand RMS <= 3 mm and q RMS <= 0.03
```

The exploratory screen is reset identity, mean candidate tau spread above
5 mm, at least 20 such pairs, and at least 10 near-pair label differences
larger than 0.02. These thresholds are not a scientific validation gate.

## Results

All three splits passed the reset identity check and had zero clipped steps.
The following query rows passed the exploratory screen:

| split | tick | mean tau spread (mm) | near pairs | strict near-pair label differences |
| --- | ---: | ---: | ---: | ---: |
| train | 16 | 18.69 | 410 | 133 |
| train | 24 | 24.29 | 44 | 10 |
| train | 32 | 28.24 | 38 | 33 |
| val | 32 | 19.20 | 46 | 41 |
| test | 16 | 23.38 | 224 | 35 |
| test | 24 | 23.95 | 77 | 21 |
| test | 32 | 25.62 | 57 | 52 |

At tick 8, every split's U32 labels were tied at zero despite nonzero hand
trajectory spread. Labels become nontrivial near ticks 16–32, which is the
useful part of the result: candidate diversity is not confined to a single
split, but the informative windows are phase-dependent.

## Attribution and boundary

The packet has no actor `obs`/history at each query. The near-current rule
matches only hand keypoints and q, not the full H contract (including any
hidden simulator state, object history, or actor normalization). The U32
labels are frozen outcome proxies, not a new same-state causal preference
validation. Therefore the positive rows do not authorize fitting a new
evaluator, claiming H-to-tau ranking, adding a deadzone, or connecting an
online selector/PW/MPC/R controller.

The correct next step, if the route is continued, is one bounded collection
or replay change that saves exact actor history/observation, current q/dq,
future hand trajectory, commanded action, episode identity, and the same
label provenance at selected ticks. The candidate bank should then be
re-audited under the full H contract before any evaluator fit.

## Artifacts and reproducibility

The read-only entry point is
`src/task/consequence-evaluator/tools/audit/audit_structured_candidate_bank.py`.
The output is
`outputs/consequence-evaluator/structured-candidate-bank-audit-20261010-r2/`
(`result.json` and `manifest.json`); the manifest records all input and
script SHA-256 values. Unit coverage is in
`src/task/consequence-evaluator/tests/test_structured_candidate_bank_audit.py`.

This Probe remains `UNCLEAR`: it changes the data-collection priority, not the
scientific status of the trajectory evaluator or the failed native retarget
execution gate.
