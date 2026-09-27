# 结果轮询代理运行合同

**角色：** `agent_poller`（对话名 `agent/poller`）  ·  **所有者：** `/root`

轮询代理使用 `/home2/wyy/oyx_ws/.codex_oyx_NewAPI`，只观察 `docs/AGENT_REGISTRY.json` 中登记的子代理。它发现状态变化后向 root 的精确 `(codex_home, conversation_id)` 发送一次 `POLL_EVENT`；root 负责审查证据、决定下一步和合并。轮询代理不能把提交、manifest 或 Goal 终态解释为科学结论。

## 观察范围

由 [agent_result_poller.py](../scripts/agent_result_poller.py) 在等待/CPU 阶段每 5 分钟、发现归属 GPU 计算进程后每 2 分钟读取：

* 每个子代理工作树的 HEAD；
* 对应 thread 的 Goal ID/状态；
* 该工作树 `outputs` 下最新 `run_manifest.json` 的路径和修改时间（按 canonical
  manifest 目标去重共享 outputs symlink）；
* 工作目录归属该工作树的 GPU 计算进程 PID。
* 已登记 thread 自己 rollout 中尚未观察过的 `task_complete` turn。
* rollout 中未匹配 `task_complete` 的本 thread `task_started` CPU turn。

首次运行只建立快照。之后只有这些字段变化才排队 `POLL_EVENT`：`HEAD`、Goal、归属 GPU PID、canonical manifest，或 thread 自己 rollout 中新的 `task_complete` ID。rollout 使用追加游标，正常轮询只读取新增 JSONL；旧 state 没有游标时会做一次静默 catch-up。rollout 中重复的 `task_complete` 只产生一个 ID。

全闲判定还要求 rollout `status=known` 且 `active_turns` 为空；Goal=`NONE` 但有未完成的本 thread `task_started`、rollout 尾部 partial line、文件不可用或读取失败都按 `rollout_unknown`/`cpu_turn` 保守视为非全闲。真实 `turn_aborted` 也是该 turn 的终止事件：清除匹配 active turn，但不生成 `task_complete` ID。只有对应 `task_complete` 或 `turn_aborted` 到达后才恢复可判定 idle，避免 CPU turn 被误唤醒。

同一轮发现多个 child 变化时只排队一条消息：单个变化沿用旧的 `{agent_key, changed, before, after}` 载荷；多个变化使用 `{coalesced: true, event_count, events}`，每个 event 保留完整的新增 turn ID。若 root 的 `queue_db` 中已经有未消费的 `POLL_EVENT`，本轮不推进发生变化的 child 游标，待 root 消费后再把积累变化合并成一条消息。队列失败或 queue 无法只读检查时不确认变化，下一轮重试。该规则保持 HEAD、Goal、GPU 和 manifest 异常的至少一次通知。

脚本不能观察未写入 manifest、未提交且未改变 Goal/GPU 状态的内部想法；子代理仍须按 [协作合同](AGENT_COORDINATION.md) 提交 handoff。

## 启动和运行

先按 [新建代理 skill](../.agents/skills/create-ref2dex-agent/SKILL.md) 启动 NewAPI 对话并登记。`proxy.py` 的 18080 服务已运行时复用，不启动第二份。在 poller 工作树中运行：

```bash
python3 scripts/agent_result_poller.py \
  --registry /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/docs/AGENT_REGISTRY.json \
  --state outputs/agent_poller/state.json \
  --codex-node /home2/wyy/.nvm/versions/node/v24.19.0/bin/node \
  --codex-js /home2/wyy/.nvm/versions/node/v24.19.0/lib/node_modules/@openai/codex/bin/codex.js \
  --interval 300
```

进程应归属于 poller 对话的长任务；按 `long-running-tasks` skill 稀疏检查其输出。脚本的普通无变化轮询不调用模型，也不发送队列消息；poller 对话检查进程时仍会消耗 token。`NOTIFIED` 和 `POLL_ERROR` 需要代理立即检查。运行前确认 `STATE.md` 的 Cm campaign 仍冻结。轮询代理没有 GPU、训练、评估、进程终止、分支合并或修改其他工作树的权限；只可维护自己的去重状态和运行记录。

## 通知与恢复

`POLL_EVENT` 只表示“有变化需要 root 审查”，包含 agent key、变化字段、旧/新 HEAD/Goal/GPU/manifest 摘要；`task_complete_added_ids` 是本次新完成 turn 的完整 ID 列表，状态文件中的 `task_complete` 和 `rollout_cursor` 是去重游标。root 可将已经合并或已审计过的 ID 标为 stale，仍保留消息中的 ID 供审计，不把 stale 事件当作新的研究证据。不得对 root 发送固定心跳。通知使用注册表中的 root 身份及 root 的 `CODEX_HOME`，而非 poller 的 NewAPI home。若 root 对话暂时停下，新结果由队列唤起审查；若队列不可用，保留事件并报告 `POLL_ERROR`。

## 状态与恢复

去重状态位于 poller 工作树的 `outputs/agent_poller/state.json`，锁文件是同目录的 `state.json.lock`。state 顶层的 `deferred_events` 保存 root queue 忙时尚未确认的合并 digest，`pending_snapshots` 保存发送前观察到的完整快照，避免短暂 GPU/资源异常在等待期间消失；事件的 `observed_gpu_pids` 保留延迟合并期间出现过的精确 PID；`supervision` 保存全闲转换边沿与一次性 wake 的 pending 位。canonical live registry 是 `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/docs/AGENT_REGISTRY.json`，root 的待发送消息位于其 root 记录的 `queue_db`（当前为 `/home2/wyy/oyx_ws/.codex_oyx_frj/queue_1.sqlite`）；poller 的 NewAPI `queue_db` 只保存本线程的排队输入，不能用来判断 root 是否已收到事件。

升级或重启前先读取 state 并确认旧进程。首次看到 child 或缺少 `task_complete`/`rollout_cursor` 的旧快照时只建立 baseline，不追报历史 turn。发送前先持久化 `deferred_events` 与 `pending_snapshots`；队列失败或 root queue 已有 `POLL_EVENT` 时保持它们，待可发送时以保存的快照确认游标。root 审查时以消息中的 `task_complete_added_ids`、`observed_gpu_pids`、HEAD/Goal/GPU/manifest 前后快照和 state/rollout 交叉核对；不要按消息到达时间把已审计的历史 ID重新解释成新结果。

重启前检查旧进程和 `outputs/agent_poller/state.json`；同一时间只允许一个 poller 进程。停止时只停止已确认属于该代理的轮询进程，不触碰其他 Codex 或实验进程。

## Owner handoff 与 Goal re-entry

`agent_poller` 不扫描自己的 thread；本轮修复或维护任务进入终态时，只向注册表中
root 的精确 `(codex_home, conversation_id)` 发送一次定向完成 handoff，内容应包括
`COMMIT`（或 `COMMIT=none`）、允许文件的校验和/模式/大小、测试结果、live poller
归属状态和后继迁移步骤。不得用固定心跳替代该 handoff。

Goal 与执行代理状态只作只读证据。只有在执行代理由非全闲变为全闲、root Goal 明确为
`active`/`running`、且 root queue 没有待处理 turn 时，最多排队一次去重的
`ROOT_DECISION_WAKE`，供 root 选择下一步；`supervision.wake_pending` 在队列失败时
保留以便下一次稀疏轮询重试。root Goal 为 `paused`/`usage_limited`/`UNKNOWN`，或
没有可证明的全闲转换边沿时，不自动唤醒；暂停来源也无法从当前 Goal 表证明时，视为
显式/未知用户暂停，只提供 event re-entry 说明并等待用户/系统恢复。poller 不修改
Goal 数据库、不冒充恢复，排队 wake 本身不会使 Goal 变为 active。root 应以事件中的
精确 ID 和当前 Goal 快照区分已集成的 stale 事件与真正的新结果。
