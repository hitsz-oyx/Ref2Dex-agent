---
schema: ref2dex.probe.v2
probe_id: P-YYYYMMDD-topic
date: YYYY-MM-DD
branch: agent/<route>
git_commit: <execution-commit-or-manifest-reference>
claim_id: C3
hypothesis_family: HF02
decision_changed_if_positive: <next route>
decision_changed_if_negative: <next route>
probe_index_in_family: 1
seed_pool: probe
status: PLANNED
---

# Probe: <decision question>

## 最小方法与边界

要区分的假设与最便宜的可判别方法；GPU/时间/磁盘预算和停止条件。
本问题一个 experiment_id；对照、seed、技术重试放 run_id。设计修订明确留痕。

## 结果与下一步

PROMISING / UNPROMISING / UNCLEAR；最关键指标、有效性与结论范围。
说明结果改变了哪个决策；无结果时保留 PLANNED，不预写结论。

## 证据

manifest、结果表、代码提交和产物链接。命令、配置、输入哈希保存在 manifest。
技术失败/无效执行保留且标明，不作为有效 Probe 或成功证据。
