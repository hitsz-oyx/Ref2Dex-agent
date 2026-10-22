---
schema: ref2dex.probe.v2
probe_id: P-20261022-grasp-geometry-coverage
experiment_id: P-20261022-grasp-geometry-coverage
date: 2026-10-22
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-grasp-geometry-coverage
probe_index_in_family: 1
seed_pool: probe
seeds: [260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291]
decision_changed_if_positive: retain measured size contract and relative SE3 control; learn drift recovery residual above frozen R
decision_changed_if_negative: diagnose positive control and physical approach before attributing failure to WAM or task difficulty
status: COMPLETED
run_id: grasp-v11-coverage-20261022-r1
---

# 尺寸几何与新增初态覆盖

## 结果解读表

| 结果 | 结论 | 下一步 | 来源 |
|---|---|---|---|
| source接近→同伺服成功，R链失败 | 该初态接口不足 | 漂移恢复 BC/有界残差 | Hermes + 外部审计 |
| A1/A2都成功 | 共享执行修复有用，不能归因τ | 保留简单链；τ按n40另判 | Hermes |
| 全失败或source准入失败 | 正控/执行/任务难度仍混杂 | 不否定WAM、不宣称任务不可能 | 外部审计 |

Result: 36/36 completed and raw-matched; new-object A1=2/8 A2=0/8 A0=2/8, new-sequence A1=0/4 A2=0/4 A0=2/4; three-arm initials12/12 matched.
Decision: three source-only strict witnesses localize a reference/intent/execution bottleneck; do not promote global geometry qualification or retrain WAM; three-arm RL HOLD.

分类 Decision；假设、最便宜检查、三处diff与命令、环境/限制及具名待裁决均见
[执行报告](../../grasp-geometry-coverage-20261022.md)。不重训v6，不做RELEASE资格。
注册器无policy成绩筛选：相对v9/v10新增8对象+已有面板对象4新序列，每初态三链各一次，
≤21600controls，600/20s，GPU4/5/6每worker≤2h，0updates。不是新holdout或显著性检验。
产物：`outputs/wam-rl-policy-v3/grasp-v11-coverage-20261022-r1/`。

实际16046controls、0updates。A0仅作不同执行链见证，不是GTτ→R上界；鸭子A1成功而A0失败，
不能把source当普遍上界。原生静态准入仍有动态限位：A1 2/12、A2 3/12、A0 5/12；
A1/A2桌面中止2/12、1/12。cubesmall source持握但8.4879mm误差未达新4.0132mm，
旧15mm下source严格5/12与新4/12分别报告。全部阈值未根据本批结果调整。
并发n40正式覆盖另卡排队；此工程面板不是其替代，分母不合并。
