---
schema: ref2dex.probe.v2
probe_id: P-20261023-tau-fixed-panel-metric
experiment_id: P-20261023-tau-fixed-panel-metric
date: 2026-10-23
task: ptv3-shared-wam-v6
branch: main
git_commit: f30e8d7ba803d1d1b64270636dd807bf1461a5fb
classification: Decision
status: COMPLETED
run_id: tau-fixed-panel-baseline-20261023-r1
seed_pool: probe
seeds: [170, 172]
decision_changed_if_positive: use the fixed panel as the τ improvement gate for the next checkpoint
decision_changed_if_negative: keep the checkpoint from being reported as an overall τ improvement and inspect the failing phase
---

# P-20261023-tau-fixed-panel-metric

| 可能结果 | 含义 | 会改变哪个决定 |
|---|---|---|
| 新 checkpoint 在全面板前进、但接触窗口/前 8 帧退步 | 指标掩盖了关键相位 | 判为**未改进**，别报喜 |
| 全面板前进、接触窗口与前 8 帧同时前进 | τ 真改进 | 写进 8 点报告并排下一轮训练 |
| 只有个别相位动、其它不动 | 只改到局部 | 写明是哪个相位、别当整体改进 |

## Decision and scope

This Decision Probe establishes the fixed TRAIN anchor score that will decide
whether a future checkpoint improved τ. It does not train, replay physics, or
make a generalization claim. The requested distinction is between a true τ
improvement and a whole-panel average that hides early or contact-phase damage.

## Duplicate check and reuse

The following existing artifacts were checked and are retained as separate
contracts:

* [`train-anchor.json`](../../../../../../outputs/ptv3-shared-wam-v6/tau-curated-20261023-r2/train-anchor.json)
  is the equivalent fixed sample set, so the scorer consumes its 512 immutable
  `(sequence_id, slot, tick)` rows and anchor ID directly.
* [`evaluate_staged_progress.py`](../../../tools/audit/evaluate_staged_progress.py)
  scores synthetic or archived executor stage transitions; it has no
  checkpoint prediction input or fixed TRAIN anchor.
* [`prepare_executor_panel.py`](../../../tools/audit/prepare_executor_panel.py)
  selects measured robot rollout prefixes and explicitly does not replay or
  score a checkpoint.
* [`evaluate_executor.py`](../../../tools/run/evaluate_executor.py) runs a native
  closed-loop executor arm. Its physics state, policy arm, and held-out robot
  panel are not equivalent to a checkpoint-independent offline TRAIN score.
* [`run_single_tau_calibration.py`](../../../tools/audit/run_single_tau_calibration.py)
  fits and evaluates one `s1_cylindersmall_pass_1` trajectory; its
  `single-tau-calibration-20261022-r1/manifest.json` is a bounded executor
  calibration receipt, not a multi-phase fixed panel.
* [`staged-eval-panel-v1.json`](../../../configs/staged-eval-panel-v1.json)
  describes six archived native 600-control episodes, not the 512 curated
  TRAIN windows.

No existing scorer was checkpoint-independent and equivalent. The only new
scorer is [`evaluate_tau_fixed_panel.py`](../../../tools/run/evaluate_tau_fixed_panel.py),
with its frozen definition in
[`tau-fixed-panel-v1.json`](../../../configs/tau-fixed-panel-v1.json).

## Fixed panel contract

The panel uses the 512 rows and anchor ID
`94d9287bb6c650cc80e4345515e0589472786bd749981a0b25ea7475118ebc37` from the
curated TRAIN anchor. It uses seed `170`, v6 `V6Windows` H4/K24 preprocessing,
the frozen event recipe with dropout disabled, and flow evaluation seed `172`.
The order and rows do not depend on the checkpoint. A deterministic CUDA mode
(`CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`torch.use_deterministic_algorithms(True)`) is part of the command contract.

The primary metric is hand future-point endpoint error in millimetres, pooled
over valid hand points and frames. The JSON reports mean, median, p95, and
count for the full panel, `approach`, `close`, `contact`, `hold`, and
`unknown` phase groups. It also reports the separate first eight future frames
and contact-window subset. A contact-window frame is one with any valid surface
contact target above confidence `0.25`; this is independent of the phase label.

## Baseline result

Command (repository root):

```bash
python3 src/task/ptv3-shared-wam-v6/tools/run/evaluate_tau_fixed_panel.py \
  --checkpoint outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt \
  --output outputs/ptv3-shared-wam-v6/tau-fixed-panel-baseline-20261023-r1 \
  --batch-size 8 --gpu 7 --max-seconds 1140
```

The machine-readable receipt is
[`baseline.json`](../../../../../../outputs/ptv3-shared-wam-v6/tau-fixed-panel-baseline-20261023-r1/baseline.json),
and the one-line summary is
[`summary.txt`](../../../../../../outputs/ptv3-shared-wam-v6/tau-fixed-panel-baseline-20261023-r1/summary.txt).

| subset | valid hand values | mean mm | median mm | p95 mm |
|---|---:|---:|---:|---:|
| full panel | 467,208 | 23.409973 | 18.326221 | 60.462026 |
| approach | 105,840 | 29.897741 | 22.739629 | 80.631971 |
| close | 101,808 | 21.873276 | 18.141441 | 53.408049 |
| contact | 105,840 | 22.194105 | 16.589462 | 61.445312 |
| hold | 103,320 | 19.982422 | 16.610634 | 49.644929 |
| unknown | 50,400 | 22.469606 | 18.194866 | 53.755281 |
| first 8 frames | 156,135 | 9.357826 | 7.325660 | 24.244514 |
| contact window | 288,960 | 19.834665 | 16.518689 | 48.541784 |

The contact window contains 7,416 of 12,288 panel sample-frames. The run used
GPU7, torch `2.4.1+cu121`, completed 64 batches in 17.5 seconds, wrote about
12 KiB, and passed the deterministic repeat check. No optimizer update,
checkpoint write, training, or physics replay occurred; `/home2` remained
above the 20 GiB safety floor.

## Interpretation boundary

These are fixed TRAIN prediction-quality numbers for checkpoint comparison.
For a future checkpoint, overall improvement counts only when the first-eight
and contact-window means also move in the favorable direction. A phase-only
change must be reported as local. The panel is not a validation/generalization
or executor-success qualification.

`git_commit` for the scored implementation is
`f30e8d7ba803d1d1b64270636dd807bf1461a5fb`.
