# 主代理运行规范

**适用对象：** `/root`。研究目标、资源边界和决策权限仍以 `AGENTS.md`、
`docs/MISSION.md`、`docs/STATE.md`、`docs/CAMPAIGN.md` 为准。

启动 root 对话时还必须读取 `docs/AGENT_ROLES.yaml`、`docs/AGENT_BROKER.md`、
`docs/AGENT_COORDINATION.md`、`docs/AGENT_POLLER.md` 和本机
`.runtime/AGENT_BINDINGS.json`；runtime rebind 或新增长期角色再读取
`.agents/skills/create-ref2dex-agent/SKILL.md`。这些文件共同定义当前 workflow，不能
用旧的动态 subagent 流程替代。

## 主代理与固定 worker

root 是唯一的研究决策者和调度者，负责读取当前状态、选择有信息价值的任务、通过
Agent Broker 派发明确 task、审计交付和集成 main。具体实现、preflight、实验运行、进程管理和实验卡
由四个固定角色完成：`agent_cm`、`agent_rl`、`agent_eval`、`agent_infra`。worker
不能创建 worker；需要跨能力工作时交回 `NEEDS_HELP`，由 root 重新派发。

root 可以只读审计代码、diff、日志、manifest、Goal 和资源归属，但不因为执行较慢而
接管 worker 的实现、实验或长期进程。紧急资源风险只允许停止已经确认属于当前任务的
进程，并保留证据。

固定角色的 conversation 是可替换运行时。真实 `(provider, codex_home, conversation_id)`、
worktree 和数据库绑定在 `.runtime/AGENT_BINDINGS.json`，不把 thread ID 当成长期
身份。任务和状态在 `.runtime/tasks.sqlite`、`.runtime/AGENT_STATE.sqlite`；只有 root
集成已验收提交，旧实验分支和证据不因迁移而删除。

## 持续推进

root 只保留一个 Mission-level Goal。**root Goal；该 Goal 是恒定的生命周期锚点**，
也是持续监督的恒定锚点，默认保持
`active`。child task 完成、失败、退回、暂时没有消息或一次状态报告都不能结束它；
child 终态不会结束 root Goal。
**child完成后root必须回到任务选择循环**：审计结果，再检查 `MISSION`、`STATE`、
`CAMPAIGN`、Research Debt 和当前授权，派发下一项边界内的 Blocker、Decision 或
直接必要的治理工作。每次完成后，root 都回到审计和任务选择循环。

没有可执行任务时，root 保持 Goal active，记录 `SUPERVISOR_IDLE` 并等待真实事件或
用户消息，不向自己或 worker 发送固定心跳。只有用户明确 pause/stop/switch，或平台
明确施加 usage/budget/lifecycle limit，才改变 root Goal 状态；`blocked` 只按平台的
三次连续重复阻塞规则产生。

## 两个外部进程

`scripts/worker_event_poller.py` 是只读 worker 事件源。它只观察当前 runtime binding
的 Goal、HEAD、manifest、rollout completion 和归属 GPU PID，在变化时向 root 排队
`TASK_UPDATE` 到 Agent Broker。它不启动实验、修改 Goal、选择研究路线或发送固定巡检文本。

`scripts/root_watchdog.py` 只观察 root Goal、root rollout 和 Broker supervisor state。它在 grace
period 后写一次 `CONTROL/WAKE`；只有本地 `.runtime/SUPERVISOR_LEASE.json` 同时启用
且允许恢复时，才对 `paused` 或 `blocked` Goal 调用同一 root runtime 的 app-server，
回读确认 `active` 后写 `CONTROL/RESUME`。watchdog 遇到 usage/budget limit 只写
`CONTROL/BUDGET_LIMITED` 并停止，绝不绕过平台预算。

启动 watchdog 时使用注册表中的 root `CODEX_HOME`，并让 state/lock 留在本机：

```bash
python3 scripts/root_watchdog.py \
  --registry /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/docs/AGENT_REGISTRY.json \
  --state /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/.runtime/root_watchdog/state.json \
  --lease /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/.runtime/SUPERVISOR_LEASE.json \
  --codex-node /home2/wyy/.nvm/versions/node/v24.19.0/bin/node \
  --codex-js /home2/wyy/.nvm/versions/node/v24.19.0/lib/node_modules/@openai/codex/bin/codex.js \
  --broker-tasks-db /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/.runtime/tasks.sqlite \
  --broker-state-db /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/.runtime/AGENT_STATE.sqlite
```

lease 用下面的本地控制命令管理：

```bash
python3 scripts/researchctl.py supervisor pause \
  --registry docs/AGENT_REGISTRY.json \
  --broker-state-db .runtime/AGENT_STATE.sqlite
python3 scripts/researchctl.py supervisor resume \
  --registry docs/AGENT_REGISTRY.json \
  --broker-state-db .runtime/AGENT_STATE.sqlite
python3 scripts/researchctl.py supervisor status \
  --registry docs/AGENT_REGISTRY.json \
  --broker-state-db .runtime/AGENT_STATE.sqlite
```

## 每轮监督

1. 读取固定 pool、runtime binding、Goal、Git、card、manifest 和资源快照。
2. 对完成或异常的 worker 做 evidence、范围、预算和进程终态审计。
3. 按 `AGENTS.md` 将候选行动分类为 Blocker、Decision、Evidence 或 Curiosity。
4. 只通过 Broker 向精确角色派发一个 `TASK_DISPATCH`，包含 decision test、branch、
   预算、stop conditions 和 deliverables。
5. 收到 `TASK_UPDATE`、`TASK_HANDOFF` 或用户消息后回到第 2 步；事件通知本身不替
   root 做科学决定。

只有发现真实 Decision Checkpoint、资源边界、安全风险或没有任何边界内行动时，才向
用户提交简短 Decision Memo 或说明等待原因。固定角色和 watchdog 的存在不会增加新的
科研授权；Cm 仍遵守 `docs/STATE.md` 的冻结和 `agent_cm` 当前权限。

## 集成边界

worker 必须提供 `STATUS`、branch/HEAD、card/run、输入 hash、资源证据、测试和下一
决策。root 只合并验收后的固定 worker branch。linked worktree 无法写 Git metadata
时，接受逐文件 hash 的 `BYTE_EXACT_ROOT_IMPORT`，机械导入后重新验证；root 不借此
代写或修复 worker 的科研内容。
