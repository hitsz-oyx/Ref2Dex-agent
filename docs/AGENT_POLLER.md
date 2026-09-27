# Worker event poller 运行合同

**角色：** 外部 `worker_event_poller` daemon  ·  **所有者：** `/root`

该进程是无模型、无研究决策的事件源。它读取固定 pool 当前 runtime binding 对应的
worker；启用 Broker 时只写 `TASK_UPDATE`，由 root 读取 Broker。root 负责审计证据、
做决策、派发下一任务和合并；poller 不把任何终态解释为科学结论。

## 观察范围

由 [`scripts/worker_event_poller.py`](../scripts/worker_event_poller.py) 观察：

* worker Goal、固定 branch HEAD 和 worktree；
* worktree `outputs` 下最新 `run_manifest.json`（按 canonical 目标去重）；
* 归属该 worktree 的 GPU 计算 PID；
* 当前 runtime rollout 中新的 own `task_complete` ID。

首次运行只建立 snapshot。之后只有这些字段变化才记录事件。rollout 使用追加游标，
Broker 写入失败时保留事件和快照，下一轮重试，确保至少一次通知。事件 payload 保留
worker、role、变化字段、前后快照和新增 completion ID。

poller **不能**：

* 读取实验指标来选择研究路线；
* 启动、停止、重启训练、collector、PPO 或评估；
* 修改任何 Goal 或实验 SQLite；Broker 自己的本地状态库是唯一例外；
* 发送固定心跳、`continue` 或“全局巡检”文本；
* 创建代理、改变 branch、合并 main。

## 启动

在拥有该 runtime 进程的机器上运行：

```bash
python3 scripts/worker_event_poller.py \
  --registry /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/docs/AGENT_REGISTRY.json \
  --state /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/.runtime/worker_event_poller/state.json \
  --codex-node /home2/wyy/.nvm/versions/node/v24.19.0/bin/node \
  --codex-js /home2/wyy/.nvm/versions/node/v24.19.0/lib/node_modules/@openai/codex/bin/codex.js \
  --broker-tasks-db /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/.runtime/tasks.sqlite \
  --broker-state-db /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/.runtime/AGENT_STATE.sqlite \
  --interval 300
```

`--once --dry-run` 可用于只读预检。默认 interval 不低于 120 秒；检测到归属 GPU
进程时下一轮使用 120 秒。state、lock 和 binding 均为本机文件，不提交 Git。

## 事件交接

root 收到消息后以事件里的完整 ID、HEAD、Goal、manifest、GPU PID 和当前 Git 状态交叉
核验。已审计或已合并的 ID 可以标记为 stale，但不能把历史 ID重新解释为新结果。worker
终态仍须交接：

```text
STATUS=<terminal state>  BRANCH=<fixed branch>  HEAD=<sha>
CARD=<id>  RUN=<manifest>  INPUTS=<hashes>
RESOURCES=<GPU/process/time/storage>  EVIDENCE=<tests/limits>
NEXT=<single decision or blocker>
```

## Root watchdog 的边界

poller 不负责 root liveness。`scripts/root_watchdog.py` 是独立进程，读取 root Goal、
root rollout、Broker supervisor state 和 `.runtime/SUPERVISOR_LEASE.json`，只写
`CONTROL` 的 `WAKE`、lease-authorized `RESUME` 或 `BUDGET_LIMITED`；lease-authorized
`RESUME` 会通过同一 root runtime 的 app-server 验证 Goal 从 `paused` 或 `blocked`
回到 `active`。它不能读取指标或
替 root 选任务。lease 缺失等同于禁用；用户通过
`python3 scripts/researchctl.py supervisor pause|resume|status` 管理授权。

未提供 `--broker-*` 参数的旧 queue 路径仍可用于迁移期预检；它产生的 `POLL_EVENT`
不属于当前四类 Broker 消息。
