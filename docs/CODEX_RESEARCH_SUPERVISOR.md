# Legacy supervisor compatibility note

状态：**RETIRED**。当前工作流不再使用固定巡检消息或
`scripts/codex_research_supervisor.py` 作为研究调度器。

当前入口是：

- root 研究循环：[`ROOT_AGENT.md`](ROOT_AGENT.md)
- Broker 与任务合同：[`AGENT_BROKER.md`](AGENT_BROKER.md)
- worker/watchdog：[`AGENT_POLLER.md`](AGENT_POLLER.md)
- 本机一致性检查：`python3 scripts/workflow_doctor.py`

`scripts/agent_result_poller.py` 只保留旧状态文件和旧 queue 的显式兼容入口；当前
poller/watchdog 直接使用 `scripts/runtime_support.py`。`--root-goal-resume-once`
不能与 `root_watchdog.py` 并行运行。

用户暂停或恢复监督授权时，使用唯一控制入口：

```bash
python3 scripts/researchctl.py supervisor pause \
  --registry docs/AGENT_REGISTRY.json \
  --broker-tasks-db .runtime/tasks.sqlite \
  --broker-state-db .runtime/AGENT_STATE.sqlite
python3 scripts/researchctl.py supervisor resume \
  --registry docs/AGENT_REGISTRY.json \
  --broker-tasks-db .runtime/tasks.sqlite \
  --broker-state-db .runtime/AGENT_STATE.sqlite
```

本文件不定义新的状态来源或恢复语义；若与当前 Broker/watchdog 文档冲突，以当前
入口为准。
