# 主代理连续监督规范

## 状态

旧的 `scripts/codex_research_supervisor.py` 固定巡检模式已 **RETIRED**。它只保留
兼容测试用途，不得通过 cron、setsid 或 `--legacy-fixed-message` 重新变成研究调度器。

当前工作流由三层组成：

1. root 的一个长期 Mission-level Goal 负责研究决策和任务派发；
2. `scripts/worker_event_poller.py` 只读观察固定 worker，并在变化时发送 `POLL_EVENT`；
3. `scripts/root_watchdog.py` 只处理 root liveness、lease-authorized bounded resume
   和平台预算终态。

worker poller 和 root watchdog 都不选择研究路线、不读取科学指标、不启动实验。它们
不能替 root 形成结论或替 worker 创建子代理。

## Root 循环

root 读取 `docs/AGENT_REGISTRY.json`、`.runtime/AGENT_BINDINGS.json`、`MISSION`、
`STATE`、`CAMPAIGN` 和当前交接，随后：

1. 审计 worker 的 evidence、资源归属、停止条件和终态；
2. 将候选行动分类为 Blocker、Decision、Evidence 或 Curiosity；
3. 只向固定角色派发一个包含 objective、decision test、branch、预算、停止条件和
   deliverables 的 `GOAL_DISPATCH`；
4. 收到 `POLL_EVENT` 或用户消息后再次审计并选择下一任务。

child task 的终态不会结束 root Goal；root 必须回到任务选择循环。没有可执行行动时
保持 Goal active，记录 `SUPERVISOR_IDLE`，等待真实事件，不发送固定心跳。

## Root watchdog

watchdog 读取 root Goal、root rollout、root queue 和
`.runtime/SUPERVISOR_LEASE.json`：

* active Goal 在 grace period 后没有 turn 或 queued input：发送一次
  `ROOT_LIVENESS_WAKE`；
* paused Goal 只有在 lease `enabled=true` 且 `allow_root_resume=true` 时才通过
  app-server bounded resume 一次；
* `usage_limited`/`budget_limited` 输出 `ROOT_BUDGET_LIMITED`，不自动恢复；
* complete、failed、blocked 或未知状态只记录，不重启。

lease 由以下命令管理：

```bash
python3 scripts/researchctl.py supervisor pause
python3 scripts/researchctl.py supervisor resume
python3 scripts/researchctl.py supervisor status
```

缺少 lease 文件等同于禁用。`pause` 先撤销自动恢复授权；若要立即中断当前 turn，
用户仍通过 Codex Goal 控制入口执行暂停。

## `CODEX_HOME` 和旧状态

thread ID 只在对应 `CODEX_HOME` 下有意义。更换 runtime 时更新本地 binding，保留旧
身份到 registry 的 `retired_conversations`，不要复用 thread ID。旧 poller state 不
能直接给新 worker poller 使用；新程序使用独立 schema 和 state 路径。

只有 root 集成已验收 worker branch。若 linked worktree 无法写 Git metadata，执行
`BYTE_EXACT_ROOT_IMPORT` 的逐文件 hash 交接；root 不在集成时改写研究内容。
