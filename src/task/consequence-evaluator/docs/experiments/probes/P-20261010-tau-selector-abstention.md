---
schema: ref2dex.probe.v2
probe_id: P-20261010-tau-selector-abstention
experiment_id: P-20261010-tau-selector-abstention
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: beed0f1
claim_id: C3
hypothesis_family: HF-trajectory-conditioned-evaluator
probe_index_in_family: 7
seed_pool: probe
seeds: [412, 413, 414, 415]
decision_changed_if_positive: prioritize an exact-H candidate bank and retain a baseline-preserving selector contract for later integration
decision_changed_if_negative: keep trajectory selector work frozen and return to data/trajectory-policy design
status: PROMISING
run_id: tau-selector-abstention-20261010-r3
---

# Can an observed-trajectory selector abstain safely near the baseline?

## Motivation and decision note

The current C1 GT-effect branch is unsafe, and the H-to-tau predictor does not
transfer to the held episode split. The remaining low-cost later-chain question
is whether the selector interface itself is worth preserving: when observed tau
scores are close to the baseline candidate, can a baseline-preserving abstention
rule reduce unsupported interventions without losing the informative rows?

This is an offline engineering Probe on the frozen tau-only (`T`) predictions
from the history panel. It does not infer tau, use C1/PW, construct candidates
from labels, or execute a selected plan. A positive result only prioritizes
exact-H candidate-bank collection and a future selector contract.

## Protocol

The audit reuses the held 18x7 panel from
`history-panel-trajectory-utility-20261010-r4/panel-predictions.npz`.
Candidate 0 is the baseline. For each fixed margin
`epsilon ∈ {0, .005, .01, .02, .03, .05}`, it selects argmax(T) only when
`max(T)-T[baseline] > epsilon`; otherwise it selects baseline. Labels are
used only for post-hoc `panel_metrics` and regret reporting. The saved tau
shuffle is the fixed negative control.

## Result

The audit completed on CPU and wrote
`outputs/consequence-evaluator/tau-selector-abstention-audit-20261010-r2/`.
The nominal tau-only panel has 137 strict pairs and 7 informative anchors;
tau shuffle drops pair accuracy from `.7007` to `.4599` (24.09pp).

| epsilon | non-baseline choices | mean regret | informative mean regret |
| ---: | ---: | ---: | ---: |
| 0 | 15/18 | .00637 | .00000 |
| .005 | 11/18 | .00521 | .00000 |
| .01 | 11/18 | .00521 | .00000 |
| .02 | 10/18 | .00521 | .00000 |
| .03 | 8/18 | .00463 | .00000 |
| .05 | 2/18 | .04398 | .10268 |

The `.01--.03` interval removes low-margin non-baseline choices while keeping
zero regret on the seven informative anchors. At `.05`, the selector begins
abstaining on informative rows and regret rises sharply. The exploratory
selector screen therefore passes for observed tau, with a practical margin
around `.01--.03`.

## Interpretation and boundary

This is `PROMISING` only as an observed-tau/offline selector contract. It does
not show that H can generate the required tau, that C1/PW can score it, or that
any selected trajectory improves native success. The result changes the next
data decision: construct an exact-H, non-tied candidate bank and then rerun the
same abstention audit before any C2/PW or online selector work. Do not promote
this Probe to a ranking, H-to-tau, PointWorld, native-control, or Cm claim.

## Reproducibility

Audit entry point:
`src/task/consequence-evaluator/tools/audit/audit_tau_selector_abstention.py`.
The output manifest records hashes of the frozen fit, panel predictions, and
audit script. The run was CPU-only because it is a small deterministic score
replay; no model was trained and no GPU process was started.
