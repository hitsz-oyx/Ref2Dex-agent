# P-20261003-cm-residual-policy

**Status:** `INVALID_IMPLEMENTATION` historical Probe; freeze its world-height and provenance
claims until the corrected v2 route is rerun.

**Decision question:** Can a Cm policy improve the frozen HF15 rotation-Cup controller by adding a small object-frame translation residual for two native control steps, while preserving contact and mesh-clearance safety?

**Implementation:** branch `agent/cm-residual-policy`, final route commit `5bc3cb1` (design contract in `362621f`). The baseline is the Cup expert with trigger-anchored wrist orientation. The policy emits only a bounded object-local translation in `[-2,2] mm × [-2,2] mm × [-3,3] mm`; all other native action channels remain baseline-owned. Cm and actor weights are frozen in `cm_residual_policy.pt` (SHA256 `2dcefa6ee5ecaef4cc34c1a6ef7478ab16c7356e54f6a6ebe84cce99e660cfcc`).

**Source and fit:** Four native source attempts contributed 592 valid rows after rejecting partial/reset-contaminated or saturated windows. Fit/calibration/held buckets contained 263/156/173 rows. Three-model Cm and matched shuffled ensembles were trained from the same fit rows. Source execution used real Isaac Gym PhysX, 96 environments, 30 Hz, airplane motions, and the frozen six-expert bank.

**Native contrast:** Five formal native runs (`probe_s734` through `probe_s738`) produced 1167 complete windows. The held hash split (`bucket >= 70`, hash seed 12651) contains 103 baseline, 130 residual, and 130 shuffled rows across 21, 18, and 20 motion/start groups respectively. The held metrics are recorded independently in [the audit JSON](P-20261003-cm-residual-policy-results.json):

| arm | held rows | H10 object-height change (mm) | last-three-step joint contact | contact loss | clearance loss | executed residuals |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 103 | -5.18 | 0.670 | 0.466 | 0.845 | 0 |
| residual | 130 | 10.05 | 0.805 | 0.331 | 0.785 | 4 |
| shuffled | 130 | 8.30 | 0.651 | 0.485 | 0.831 | 4 |

The apparent residual-minus-baseline height difference is descriptive and unpaired. Only four held residual windows actually changed the native action; the rest fell back exactly to baseline because the ensemble margin, uncertainty, or OOD gate rejected the actor proposal. The shuffled arm has a similar descriptive height shift despite the same low action coverage, so these arm means cannot support a Cm utility claim.

**Gate and decision:** The row and motion/start support gates pass, but the predeclared residual coverage gate requires at least 48 changed held residual windows and only 4 changed. The result is `UNPROMISING` for this implementation route. Do not start PPO, do not promote this to a supported scientific claim, and do not interpret the arm means as evidence that residuals improve the strong controller.

**Next decision:** Treat fail-closed coverage as the blocker. If this route is revisited, first run a cheap policy-coverage diagnosis on held states (actor output scale, calibration margin, and OOD distribution) and require an independently justified coverage target before another native comparison. The current route supplies native wiring and risk accounting but does not establish Cm residual utility.

**Coverage diagnosis:** To distinguish fail-closed coverage from a useful residual, a diagnostic
checkpoint copied the frozen policy and changed only the predicted-lift margin from the calibrated
3.99 mm to 0.5 mm. OOD, ensemble-disagreement, contact-loss, and clearance-loss filters stayed
unchanged; this checkpoint was never treated as the calibrated policy. Twelve additional native
runs (`probe_m050_s741` through `probe_m050_s752`) produced 867 held windows with 55 actual
residual changes, passing the predeclared coverage gate. The corrected independent audit
(`audit_margin050_all12_corrected.json`) applies the predeclared episode and motion/start cluster
bootstrap checks in [the corrected audit JSON](P-20261003-cm-residual-policy-margin050-results.json): residual-minus-baseline height was -1.69 mm at the motion/start level, the 90%
lower bounds were -1.39 mm (episode) and -7.15 mm (motion/start), and clearance-loss increases
were +3.44 and +1.31 percentage points respectively. The diagnostic is therefore also
`UNPROMISING`; lowering the safety margin does not reveal a stable Cm residual benefit.

The audit implementation now enforces the predeclared lift, bootstrap, contact, and clearance
criteria instead of treating coverage alone as `PROMISING`. Do not lower the margin further or
start PPO on this route.

## 审查冻结（2026-10-03）

这份记录的 residual 世界高度结论撤回并冻结为 `INVALID_IMPLEMENTATION`，不是新的
`UNPROMISING` 证据：训练/选择评分把 object-local `z` 当成了 world `z`，而 native 指标
使用 world `z`。同时，旧 collector 在 reset 后复用了旧 motion/start/rest 元数据，且把
`PYTHONHASHSEED` 记成了 seed，因此相关分组和复现 provenance 不可信。修正版改为统一
世界系高度目标、每个 trigger 快照 metadata、显式记录并传递 simulator seed，并移除
`shuffled` residual control；修正版 source/fit/native Probe 使用新 experiment ID，尚未
重跑。旧 checkpoint 和 records 保留作审计输入，不得用于 Cm utility claim。
