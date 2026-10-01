# Contact-response standalone research: first results

Worktree: `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-contact-response`.
Branch: `agent/contact-response-cm`. Original working tree restored to
`agent/paired-evaluator`; its existing changes were preserved during migration.

## Physical experiment

`P-20261001-contact-response-resolution-r1`, code466e80d,16/16 processes complete,
four panels of96windows, one idle GPU4,481.9seconds. Original inputs unchanged.

| Horizon | Pulse contrast RMS/mm | Repeat contrast RMS/mm | Descriptive ratio |
| ---: | ---: | ---: | ---: |
| 5 | 7.150 | 0.101 | 70.44 |
| 10 | 18.827 | 0.444 | 42.36 |
| 30 | 105.792 | 9.638 | 10.98 |

The physical-resolution components pass, but vertical effects have mixed signs
across panels. Full predeclared gate: **UNPROMISING**. No lift or grasp benefit
is inferred. Report exposed pre-intervention state mismatch and the retrospective
contact-proxy schedule; native proxies do not establish pairwise contact.

Corrected descriptive analysis is `analysis-v2/analysis.json`. Original analysis
is retained: its arbitrary denominator floor yielded enormous finite ratios when
observed repeat contrast was zero. v2 reports those ratios as null; pooled5/10
values and both gate outcomes are unchanged. This does not imply zero uncertainty.

## Differential vector prediction

Following D-20261001-vector-response-learning, the unused conditional design was
revised to separate measurable full-vector effects from failed monotone lift.
No original simulation gate or result was changed. The test panels were already
in the physical pilot, so learned outcomes remain reused-data exploratory evidence.

`P-20261001-differential-response-learning-r1`, code44f22ea,9/9 fits complete,
three optimization seeds per method,1000updates each,17.2seconds on GPU4.
Fit actor286 contributes192windows; test actor287 contributes192windows.

| Model | 5-step contrast RMSE/mm | 10-step contrast RMSE/mm |
| --- | ---: | ---: |
| Factual | 10.121 | 26.546 |
| Differential loss | 10.114 | 26.990 |
| No-pulse input | 9.922 | 26.456 |
| Global fit-mean effect | 10.061 | 26.259 |
| Privileged per-motion fit mean | 10.137 | 26.340 |

Differential z-sign agreement is32.6%/31.1%, below75%; neither RMSE improvement
gate passes. **UNPROMISING**. Freeze this small-data implementation; do not add
updates or seed search to pursue its gate. No trained policy or Cm utility claim.

## Artifacts and research decision

- Compact physical traces:16`response.pt` files, per-run manifests/results/logs.
- Learning:9network checkpoints, all seed metrics, `predictions.pt`, manifests.
- Standalone vector/PDF/PNG physical figure and CSV; total experiments~70MB.
- `paper/manuscript.tex` and PDF review copy: complete English working draft with
  actual negative results, methods, primary references and limitations.
-14focused tests passed; GPU4 released after the run.

Next decision-worthy question: can verified interacting geometry and broader
contact/action coverage produce predictable effects on a newly frozen independent
test set? This requires a new collection design, not additional fitting of the
current tiny bank. Do not start policy/PPO from these failed predictors.

Journal readiness remains **NOT READY**. Missing evidence includes a distinctive
method beyond established action/controllability-aware modeling, independent
object/task/hand splits, matched policy improvement and real hardware where feasible.
