# Root Goal 生命周期审计（2026-09-27）

**STATUS:** COMPLETED（只读审计）
**BRANCH:** `agent/root-goal-lifecycle-test-20260927`
**BASE/HEAD（审计前）:** `026988ae8a7b07273756b92b86984df927783966`
**CARD:** NONE · **GPU:** 0 · **新研究数据:** 0

## 结论

用户授权恢复后，`thread/goal/set(active)` 加同一 thread 的
`thread/goal/get` 回读只保证一个时间点的状态，不是永久租约。证据显示 root
曾明确从 `blocked` 回读为 `active`，随后多次 turn 以
`server_overloaded`（`codex_error_info=server_overloaded`）结束，root rollout
又报告平台将其置为 `blocked`；之后同一 Goal 再次成功回读为 `active`。这最符合
平台容量/usage/lifecycle 状态重新施加，而不是 child 完成或研究“三次 Probe”规则。
平台没有在 Goal DB 中保存转移原因或计数，因此“究竟哪一内部策略触发 blocked”仍为
**UNKNOWN**；下面的因果判断是基于 rollout 事件的最强可验证解释。

当前最后一次只读快照（`2026-09-27T14:50:07Z`）中 root 为 `active`；
`agent_cm_temporal` 仍独立为 `blocked`，`agent_poller` 也为 `blocked`，而其轮询
进程仍在运行。这证明 child 状态不是从 root 继承，也不会由 root 恢复自动改变。

## 当前状态快照（UTC）

来源：只读打开 `/home2/wyy/oyx_ws/.codex_oyx_NewAPI/goals_1.sqlite`；`updated_at_ms`
是 Goal 表原值。

| 身份 | thread / goal | 状态 | `updated_at` |
| --- | --- | --- | --- |
| root | `01a0e1ad-01de-7031-8749-8e55426bb9db` / `94e8d8af-2bdf-4e75-8d7c-e01047d8676d` | `active` | `2026-09-27T14:49:46.831Z` |
| `agent_cm_temporal` | `01a0d7cf-3a82-7a93-b05c-ffecb361f036` / `35ea053b-1b9a-4164-bcc0-f97f3c3ffb71` | `blocked` | `2026-09-25T16:34:15.321Z` |
| `agent_poller` | `01a0dca7-9af9-7951-b9c3-58975c4b470b` / `98a563de-1f28-495a-a4af-628c3a30d79c` | `blocked` | `2026-09-26T08:22:25.198Z` |
| `agent_baseline` | `01a0d6bf-ba61-7f92-8af7-d68c91f4007d` / `2e9de2b4-45e5-4bb7-83a0-7bae01ded1ac` | `paused` | `2026-09-26T04:08:56.165Z` |
| `agent_c1_validation` | `01a0de76-3a3f-7293-88f8-18c140024f9f` | no `thread_goals` row (`NONE`) | — |
| `agent_workflow` | `01a0d86b-d735-7100-b229-d7a3ff0f83cc` | no `thread_goals` row (`NONE`) | — |

轮询状态文件 `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-poller/outputs/agent_poller/state.json`
在 `2026-09-27T14:49:06Z` 记录 `root_goal_status=active`、
`all_execution_idle=false`，并观察到 workflow 的 CPU turn
`01a0e34f-2c16-7880-99ca-2a048e33de61`。只读观察到的 live poller 命令 PID 为
`2400681`；未启动、停止或重启它。

## 可复核的状态转移时间线

证据文件：
`/home2/wyy/oyx_ws/.codex_oyx_NewAPI/sessions/2026/09/27/rollout-2026-09-27T15-03-26-01a0e1ad-01de-7031-8749-8e55426bb9db.jsonl`。

1. `14:12:53Z`（rollout 行 1335）读取到 root 为 `blocked`。
2. `14:14:03Z`（行 1369–1370）同一 Goal 的 app-server 操作明确输出
   `before=blocked, after=active, objective_unchanged=true`；Goal ID 与 thread ID 未变。
3. `14:16:21Z`（行 1392）、`14:20:51Z`（行 1439）、`14:21:36Z`（行 1455）、
   `14:30:11Z`（行 1502）和 `14:34:47Z`（行 1572）的 `task_complete` 均记录
   `Selected model is at capacity` / `server_overloaded`。
4. `14:23:32Z`（行 1477）和 `14:37:03Z`（行 1609）的 root 消息分别报告平台
   再次将 root 置为 `blocked`；`14:37:34Z`（行 1613–1614）又输出同一 Goal
   `before ... blocked after ... active`。
5. `14:43:45Z`（行 1698–1701）以及本审计末次 DB 读取均为 `active`。child
   `agent_cm_temporal` 的旧 `updated_at` 在这些恢复之间没有变化，root 也明确记录
   “不静默唤醒”该 blocked child。

## 四类状态的区分

* **平台 usage/capacity/lifecycle：** `server_overloaded` 与 root 随后的 blocked
  报告在时间上成组出现，是目前最强解释；但没有平台转移日志，不能宣称内部计数规则已
  被证明。
* **child blocked：** Cm child 自 `2026-09-25` 起就是独立 Goal 状态；poller child
  也是旧 blocked 状态。root 恢复只作用于 root thread，未改变它们。
* **poller stale state：** 早先 `14:39Z` 快照仍显示旧的全闲/唤醒待处理状态，而
  rollout 已推进到 `14:42Z`、Goal DB 到 `14:43Z`；这能造成通知延迟或表面状态落后，
  但不会自行把 Goal DB 写成 `blocked`。当前快照已显示 workflow active turn。
* **真实重复阻塞（研究三次无进展）：** 没有证据。上述错误是模型容量事件，不是三次
  有效 Probe；Goal DB 只有当前行，没有转移历史，`thread_goal_continuation_deferrals`
  也为空，无法验证平台是否把容量错误计入其自身重复阻塞计数。

## 规范与实现是否足够

规范已明确 root/child 独立：`docs/ROOT_AGENT.md:29-35,41-42`、
`docs/AGENT_COORDINATION.md:167-172,204-207`；poller 的事件/状态语义和 stale
快照规则在 `docs/AGENT_POLLER.md:44-50,61-89`。`scripts/agent_result_poller.py`
的 `goal_details`（约 70 行）只读 Goal DB，`app_server_resume_root_goal`（约
730-800 行）只接受精确的 root paused Goal、校验同一 thread 回读 active，并不修改
child。回归测试 `tests/governance/test_root_goal_lifecycle.py` 只验证上述静态合同。

因此，**意图层面的 root/child 独立有测试保障；运行时平台在恢复后再次 blocked 没有
保障**。最小后续修复建议（本轮未实施）：

1. 在审计/通知模式中把 `platform_blocked`（及原因未知）与研究 blocker 分开，记录
   `goal_updated_at`、最近 `task_complete` 错误、resume/readback 结果和
   `state_observed_at`，不写 Goal DB。
2. 用隔离 fake app-server 增加 `blocked → active → server_overloaded → blocked` 的
   回归测试，同时断言 child 状态不变；另测 stale poller state 不产生重复 wake。
3. 保留现有“blocked child 不静默唤醒”规则。`AGENT_POLLER.md:90` 已注明 one-shot
   resume 不检查 root 是否已有活跃 turn，这是仍需 root 评估的残余风险，不是本次观察到
   的转移原因。

## 范围与交接

本次只读访问 Goal DB、rollout、poller state、规范、脚本和既有回归测试；没有写 DB、
修改进程、启动实验/GPU、改变科学 claim、修改 main 或其他工作树。本文件是唯一新增
文件；root 可按本文件审阅平台状态与治理缺口，**不需要为本审计恢复 child 或重跑任何
研究任务**。
