# 主代理连续监督规范（取代旧外部唤醒器）

## 状态

`scripts/codex_research_supervisor.py` 的“空闲时向 `/root` 投递固定巡检句子”
模式已 **RETIRED**。不要再用 `setsid`、cron、watchdog 或定时 `codex queue`
让主代理自我唤醒；这样只会制造重复报告，不会形成新的研究 goal，也不会可靠地
给子代理派发下一项工作。

旧脚本保留在仓库中，仅用于历史状态审计和兼容性测试。脚本默认拒绝真实 queue
操作；除非明确使用 `--legacy-fixed-message`，否则不会向任何 thread 写入消息。
不应重新启用该兼容模式。

## 正确的监督方式

主代理 `/root` 使用注册表和自己的活动 turn 建立连续监督循环：

1. 从
   [`docs/AGENT_REGISTRY.json`](/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-workflow-v22/docs/AGENT_REGISTRY.json)
   读取目标的 `codex_home` 与 `conversation_id`；
2. 在同一条主对话中读取 thread/Goal、branch/HEAD、experiment card、manifest、
   归属进程和 GPU；
3. 没有 blocker 时只等待到下一轮轮询，不发送固定心跳，也不结束监督 turn；
4. 发现已授权 blocker 时，向精确的 child thread 派发一条包含 objective、
   branch、允许路径、预算、停止条件和交付物的 `GOAL_DISPATCH`；
5. 等待 child 的真实终态或新的 blocker，再做 completion audit。

默认节奏：waiting/blocked 和 CPU 分析每 5 分钟；GPU collection 每 2 分钟，
最长不超过 5 分钟；终态或异常立即审计。轮询不是实验授权，也不自动恢复
`paused`/`blocked` Goal。

### 派发消息的最低格式

```text
GOAL_DISPATCH
TARGET_AGENT_KEY=<registry agent_key>
CODEX_HOME=<registry codex_home>
CONVERSATION_ID=<registry conversation_id>
OBJECTIVE=<one concrete blocker or decision>
DECISION_TEST=<cheapest discriminating test>
BRANCH=<exact branch>  BASE_COMMIT=<sha>
ALLOWED_PATHS=<paths>
EXPERIMENT_OR_CARD=<id or NONE>
RESOURCE_BUDGET=<gpu/time/storage/process>
STOP_CONDITIONS=<explicit conditions>
DELIVERABLE=<commit/card/manifest/handoff>
NOT_AUTHORIZED=<what must not start>
```

不要发送“继续”“再巡检一次”或固定的全局巡检句子作为 goal。没有已授权
blocker 时，正确状态是内部 `SUPERVISOR_IDLE`，而不是向主对话排队一条消息。

## `CODEX_HOME` 规则

thread ID 只在对应 `CODEX_HOME` 下有意义。当前环境的默认目录是：

```text
CODEX_HOME=/home2/wyy/oyx_ws/.codex_oyx_NewAPI
```

如果切换目录，必须同时更新注册表中的 `codex_home`、`conversation_id`、
`session_root`、`state_db`、`goal_db` 和 `queue_db`，并重新验证身份键
`(codex_home, conversation_id)` 的唯一性。

## 与 overload watchdog 的边界

`codex_overload_watchdog.py` 只处理结构化的 `server_overloaded` 事件；它不负责
研究巡检、goal 设计、子代理调度或主代理唤醒。不得把两个工具合并成新的定时
心跳系统。

## 停用与恢复

若历史 research-supervisor 进程仍在运行，只停止已核实属于本项目的精确 PID；
不要触碰实验进程、其他用户进程或 overload watchdog。若未来确需恢复旧兼容模式，
必须由用户明确授权，并同时提供非默认的 thread、`CODEX_HOME` 和具体 message。

停用时还要只读检查对应 `queue_db`。如果旧调度器留下固定巡检文本，先把数据库
做一致性备份，再按 `thread_id`、消息 ID 和固定文本三项精确匹配删除；不得清空
整个 queue，也不得删除其他用户或 child thread 的排队消息。删除后重新只读核验
队列为空或只剩明确授权的 `GOAL_DISPATCH`。
