# Workflow 文档

这些文档描述当前固定角色池、Agent Broker、runtime binding、任务交接和 root
liveness。新对话从根目录 [`AGENTS.md`](../../AGENTS.md) 进入；不要把旧 plan、log 或
历史 handoff 当作当前 workflow。

当前规范入口仍保留在 `docs/` 根目录，供脚本、测试和已有链接兼容：

| 主题 | 当前入口 |
| --- | --- |
| 固定角色和权限 | [`AGENT_ROLES.yaml`](../AGENT_ROLES.yaml) |
| Broker、队列、租约、四类消息 | [`AGENT_BROKER.md`](../AGENT_BROKER.md) |
| root/worker 委派和集成 | [`AGENT_COORDINATION.md`](../AGENT_COORDINATION.md) |
| root 决策循环 | [`ROOT_AGENT.md`](../ROOT_AGENT.md) |
| poller/watchdog 运行合同 | [`AGENT_POLLER.md`](../AGENT_POLLER.md) |
| 连续监督兼容说明 | [`CODEX_RESEARCH_SUPERVISOR.md`](../CODEX_RESEARCH_SUPERVISOR.md) |
| 历史 runtime 审计记录 | [`AGENT_REGISTRY.json`](../AGENT_REGISTRY.json) |

`.runtime/AGENT_BINDINGS.json`、`.runtime/tasks.sqlite` 和
`.runtime/AGENT_STATE.sqlite` 是本机运行状态，不提交到 Git。

## 操作前检查

在派发新任务前运行：

```bash
python3 scripts/workflow_doctor.py
```

它只读检查角色定义、registry、binding、Broker 数据库、supervisor lease 和活动租约。
`WARN` 可以是故意未绑定的可选角色；需要完整角色池时使用
`--require-all-bound --strict`。新的 dispatch/claim 只有 Broker 状态为 `RUNNING` 时
才会通过；暂停允许已有任务收尾。

`scripts/agent_runtime_adapter.py` 是显式执行边界。它消费一个已 claim 的任务，向
调用者提供标准化 JSON 和 lease 环境变量，并要求 runtime launcher 通过 Broker 完成
handoff；没有 launcher 时只会排队，不会声称任务已经启动。

详细设计见 [`workflow-repair-design.md`](../superpowers/specs/2026-10-01-workflow-repair-design.md)。
