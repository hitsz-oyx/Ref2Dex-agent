# 最近

九源手物诊断，任务 `ptv3-shared-wam-v6`。大训练仍然 HOLD。下一刀还没有新卡，不要开跑。

2026-10-24，旧原点 `[-2,-2,-2]` 上的 C2-flow-r2 跑完 500 步。它和已完成的 C2 只差 `flow_future` 可训练。相对 local3000 的三项误差是 +1.862 / +6.941 / +1.066 mm，母序列置信区间下界都大于 0。相对 C2 已测到的退化 +2.436 / +9.412 / +1.140 mm，大约还剩 76% / 74% / 93%。500/500 步的裁剪前梯度范数都大于 1。裁决是没有路线结论：这不是修复，也不能据此否定「共享主干更新是主因」。权重在 NAS `diagnosis/C2-flow-r2/step000500.pt`。卡是 `src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261024-c2-flow-old-workspace.md`。

同一天，把体素原点改到 -4 m 的预检是否定的。已知坏窗口能进去，但初始面板不再等价，最大点位移 41.351 mm。不要在 -4 m 网格上训练，也不要把它当成同一次 C2。卡是 `P-20261024-workspace-origin-preflight.md`。source4 窗口 s4-34183 tick 61 在 -2 m 仍然非法，它不在 C2 实际训练到的支持里。

GPT 用 5.6 极高看过 r2，不是 Pro。它同意停在这个裁决上，不要用原设置再跑 500 步。若还要回答主因，只留一刀：同一支持、同一初始化、同样 500 步，两臂对 `flow_future` 用同一条事先固定的裁剪，主干分开裁。不改原点，不重跑 C4-source1。当时这刀还没有卡。

2026-10-24: GPT confirmed the operator (OPERATOR_OK). Card: `src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261024-c2-matched-clip.md`. Recipe switch: `matched_clip` (default false).

GPT 现在负责拍板。它先 HOLD 两臂一起开，因为还没有「分组裁剪下的全训练臂仍复现 C2 退化」。随后 RUN_CONTROL：只跑 `diagnosis/C2-matched-control`，`matched_clip` 开、`flow_only` 关。复现成立当且仅当相对 local3000 的三项都 ≥ +1.827 / +7.059 / +0.855 mm。flow-only 臂先不跑。

2026-10-24: The matched-clip unrestricted control has completed; reproduction holds (REPRODUCED). Recorded in `src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261024-c2-matched-clip.md`. The flow-only arm remains unauthorized and unlaunched; do not start it.

```text
MEASURED
directory: diagnosis/C2-matched-control
updates: 500
means_mm: +11.387 / +33.992 / +6.778
delta_vs_local3000_mm: +2.406 / +9.333 / +1.139
ci95_mm: +2.082 / +2.714, +7.718 / +11.099, +0.938 / +1.374
thresholds_mm: +1.827 / +7.059 / +0.855
script_token: REPRODUCED
flow_preclip_gt1: 500/500
trunk_preclip_gt1: 500/500
```
