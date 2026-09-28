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
