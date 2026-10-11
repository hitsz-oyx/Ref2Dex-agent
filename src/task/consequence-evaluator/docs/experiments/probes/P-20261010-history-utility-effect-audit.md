---
schema: ref2dex.probe.v2
probe_id: P-20261010-history-utility-effect-audit
experiment_id: P-20261010-history-utility-effect-audit
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 57af024
claim_id: C3
hypothesis_family: HF-trajectory-conditioned-evaluator
probe_index_in_family: 6
seed_pool: probe
seeds: [412, 413, 414, 415, 416]
decision_changed_if_positive: retain the C1 effect branch and authorize a separately screened C2/PW offline audit
decision_changed_if_negative: keep the effect branch, C2/PW, selector, and native execution frozen pending a contract or data repair
status: UNPROMISING
run_id: history-utility-effect-audit-20261010-r2
---

# Is the frozen C1 GT-effect branch safe enough for later-chain selection?

## Motivation and decision note

The history-panel evaluator Probe found that matched-init C1
(`H+tau+E_GT`) was worse than both H-only and C0. Before attributing that to
the approximate-H panel or moving to `完整链路.md` steps 8--9, this bounded
read-only audit isolates the effect branch on the exact held test panels. It
reuses the frozen C1 checkpoint; it performs no fitting, no candidate
construction from labels, no PointWorld inference, and no native execution.

The discriminating question is whether removing or permuting `E_GT` leaves
the ranking unchanged (a safe branch), or materially changes/improves it (an
effect wiring/distribution problem). A safe result would justify a separate
C2/PW offline audit. An unsafe result closes the current effect-to-selector
route until the branch contract or panel is repaired.

## Protocol and contract

The input is
`outputs/consequence-evaluator/history-candidate-panel-20261010-r2/` and the
matched-init C1 checkpoint is
`outputs/consequence-evaluator/history-panel-trajectory-utility-20261010-r4/C1.pt`.
The audit uses the 18 held test panels (7 candidates each, 137 strict pairs,
7 informative anchors) and the fit's deterministic donor seed 416. It scores:

* nominal `H+tau+E_GT`;
* `E_GT=0` and the model's `use_effect=False` path;
* within-panel `E_GT` permutation;
* within-panel `tau` permutation;
* both permutations.

All inputs are standardized with the checkpoint statistics. Panel membership,
candidate order, split isolation, finite values, checkpoint hash, and panel
hash are checked before inference. Labels are used only by `panel_metrics` to
score predictions.

## Result

The audit completed on CPU with the frozen checkpoint and produced
`outputs/consequence-evaluator/history-utility-effect-audit-20261010-r2/`.

| variant | strict pair accuracy | top-1 agreement | mean regret |
| --- | ---: | ---: | ---: |
| nominal C1 | .5839 | .0556 | .09433 |
| `E_GT=0` | .7007 | .3889 | .00463 |
| `E_GT` panel shuffle | .4526 | .0556 | .07581 |
| `tau` panel shuffle | .5693 | .1667 | .09144 |
| both shuffle | .4453 | .1111 | .07292 |

The zero-effect and `use_effect=False` predictions are bitwise equal. Removing
the effect improves pair accuracy by 11.68 percentage points, while
permuting it degrades accuracy by 13.14 points. The nominal tau-shuffle drop
is only 1.46 points. The safety screen is false: nominal C1 is below .70, the
zero-effect ablation improves by more than the allowed 3 points, and tau
shuffle does not drop by 3 points.

## Interpretation and boundary

This is `UNPROMISING` for the current C1 effect branch, not a formal negative
claim about GT effects or trajectory evaluators in general. The effect branch
is consumed by the model, but on this approximate-H/small-information panel
its current representation or training distribution damages ranking. The
result does not authorize C2/PW inference, deadzones, selector deployment,
online MPC, or native R execution. A future reopening must first repair or
re-specify the effect contract, or create a larger exact-H panel, and then
rerun a fresh matched audit.

## Reproducibility

Audit entry point:
`src/task/consequence-evaluator/tools/audit/audit_history_utility_effect.py`.
The result manifest records the panel/checkpoint/script hashes and the exact
git commit (`57af024`).
