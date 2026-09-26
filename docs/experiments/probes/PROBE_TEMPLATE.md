---
schema: ref2dex.probe.v2
probe_id: P-YYYYMMDD-topic
date: YYYY-MM-DD
branch: agent/<hypothesis-family>
git_commit: <commit-or-working-tree>
claim_id: C3
hypothesis_family: HF02
decision_changed_if_positive: <the next route if the signal is positive>
decision_changed_if_negative: <the next route if the signal is negative>
probe_index_in_family: 1
seed_pool: probe
status: PLANNED
---

# Probe: <short question>

## Question

这次最小实验要区分什么假设？

## Hypothesis

H1:

Alternative:

## Decision

如果 H1 成立：

下一步：

如果 H1 不成立：

下一步：

如果两种结果都不会改变下一步，不要运行该 Probe。

## Minimal protocol

只写区分假设所需的最小实验。

## Budget

GPU:

wall time:

storage:

## Stop condition

什么时候立即停止？

## Result

Status:

`PROMISING` | `UNPROMISING` | `UNCLEAR`

Key evidence:

## Decision update

实验以后路线如何变化？若 family 已耗尽预算，更新
`docs/RESEARCH_QUEUE.yaml` 的状态并切换到更高层假设；不要只换 metric 或
horizon 继续消费同一个 family。

## Artifacts

必要路径即可。
