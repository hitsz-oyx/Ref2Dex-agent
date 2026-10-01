# Ref2Dex 多代理协作、注册与连续监督契约

**状态：** ACTIVE  ·  **schema：** `ref2dex.agent_coordination.v5`  ·  **所有者：** `/root`

根入口是 [`AGENTS.md`](../AGENTS.md) 第 2 节；本文件与
[`docs/AGENT_ROLES.yaml`](AGENT_ROLES.yaml)、[`docs/AGENT_BROKER.md`](AGENT_BROKER.md)、
[`docs/ROOT_AGENT.md`](ROOT_AGENT.md) 及
[create-ref2dex-agent skill](../.agents/skills/create-ref2dex-agent/SKILL.md) 一起构成当前
workflow。它规定固定角色池、运行绑定、任务交接和监督边界。实验结果仍只写入
experiment card、manifest 和 `docs/STATE.md`；本文件只描述 orchestration。

固定角色规则在 [`docs/AGENT_ROLES.yaml`](AGENT_ROLES.yaml)，只描述长期身份、职责和
权限。机器可读的旧注册表 [`docs/AGENT_REGISTRY.json`](AGENT_REGISTRY.json) 仍保留
历史审计兼容性；可替换的 conversation、provider、`CODEX_HOME`、数据库和工作树绑定
只存放在机器本地的 `.runtime/AGENT_BINDINGS.json`。运行状态、任务租约和 handoff
只存放在 `.runtime/tasks.sqlite` 与 `.runtime/AGENT_STATE.sqlite`。

## 1. 固定角色池

组织永远只有一个 root 和四个执行角色：

| 稳定角色 | 长期职责 | 固定分支 |
| --- | --- | --- |
| `root` | 研究决策、派发、验收、main 集成和用户沟通 | `main` |
| `agent_cm` | Cm、representation、world model 和离线机制分析 | `agent/cm` |
| `agent_rl` | DExplore、PPO、collector、online 和 GPU 执行 | `agent/rl` |
| `agent_eval` | matched evaluation、统计、provenance 和 Validation | `agent/eval` |
| `agent_infra` | 数据 plumbing、工具、测试和 workflow | `agent/infra` |

固定角色不是固定 conversation。对话、`CODEX_HOME` 或工作树损坏时，只替换该
角色的本地 binding，保留 `agent_key`；身份键始终是
**`(codex_home, conversation_id)`**。不得只改显示名来掩盖身份迁移，也不得复用
旧 thread ID。

`agent_poller` 不再是 LLM worker；新的观察程序是无研究决策的
`scripts/worker_event_poller.py`。历史 `agent_*` 记录保留用于审计，不能因为它们
仍在 registry 中就自动扩展角色池。

## 2. 权限和委派

```text
root → agent_cm
root → agent_rl
root → agent_eval
root → agent_infra
```

只有 root 可以选择研究问题、创建 task、改变资源预算、接受交付和合并 main。root
通过 [`scripts/agent_broker.py`](../scripts/agent_broker.py) 派发；它不能动态创建
agent 或调用 subagent/create-agent API。
worker 不能创建 worker，也不能把一次 Probe 升格为 Validation。worker 需要另一种
能力时交回 `NEEDS_HELP`，由 root 重新派发。

新建 `agent_key` 默认禁止。创建 skill 的默认动作是为既有角色 rebind runtime；
只有新增长期能力且用户明确同意时，才建立新角色、分支、工作树和 registry 记录。

## 3. 运行绑定和状态来源

每个 binding 至少记录：

* `agent_key`、`conversation_id` 和 `codex_home`；
* session、state、Goal、queue 数据库和 rollout 定位；
* worktree、branch、角色权限和 handoff 要求。

动态 `RUNNING`、`BLOCKED`、`COMPLETED` 不写入稳定 pool；它们从 Goal 数据库、
rollout、Git、manifest 和归属进程只读推导。修改 registry 或 binding 后运行：

```bash
python3 -m json.tool docs/AGENT_REGISTRY.json >/dev/null
python3 scripts/workflow_doctor.py --json
python3 tools/verify.py --changed
```

`workflow_doctor.py` 负责固定角色、branch、provider、身份、数据库和 lease 的一致性
检查；不再需要在文档中维护另一份手写注册表校验脚本。

## 4. Task dispatch

root 发出的每个任务必须是一个可判别目标，不发送“继续”“再看看”或固定心跳：

```yaml
type: TASK_DISPATCH
task_id: T-YYYYMMDD-short-name
target: agent_cm|agent_rl|agent_eval|agent_infra
objective: one concrete decision or blocker
context: {state_revision: <sha>, decision_test: <cheapest discriminating test>}
constraints: {branch: <fixed branch>, gpu: 0, stop_conditions: [...]}
done_when: [commit/card/manifest/handoff]
```

同一 worker 同时只运行一个 task。task identity 用 `TASK-*`、Probe ID 或 Validation
ID 表达，不创建 task 专用 branch。Broker 只允许四种消息：
`TASK_DISPATCH`、`TASK_UPDATE`、`TASK_HANDOFF`、`CONTROL`。worker 完成时交接：

```text
TASK_ID=<id>  STATUS=HANDOFF_READY
BRANCH=<fixed branch>  HEAD=<sha or none>
CARD=<id>  RUN=<run id or manifest>
INPUTS=<hashes>  RESOURCES=<GPU/process/time/storage>
EVIDENCE=<tests/metrics/limits>
NEXT=<single decision or blocker>
```

## 5. 事件源与 root watchdog

`worker_event_poller.py` 每 2–5 分钟只读观察当前 binding 对应的 worker：Goal、HEAD、
manifest、rollout completion、归属 GPU PID。启用 Broker 时，变化写成一条
`TASK_UPDATE`；Broker 负责持久化和 root 消费，不能确认事件时 poller 保留自己的
cursor。未传 Broker 参数的旧 `POLL_EVENT` queue 路径仅用于兼容，不是当前协议。
它不读取指标来选择路线，不启动或停止实验，不修改任何 Goal，也不发周期性“全局巡检”文本。

`root_watchdog.py` 是另一条极小的 liveness 进程，只观察 root 自己的 Goal、root
rollout 和 Broker supervisor state。启用 Broker 时它写 `CONTROL`（`WAKE`、`RESUME`
或 `BUDGET_LIMITED`）；lease-authorized 恢复会先通过注册 root runtime 的 app-server
验证 Goal 状态。它只允许以下动作：

* root active 但超过 grace period 没有 turn 或 queued input：写一次 `CONTROL/WAKE`；
* root Goal 为 `paused` 或 `blocked`，且本地 `.runtime/SUPERVISOR_LEASE.json` 同时声明
  `enabled=true`、`allow_root_resume=true`：通过同一 root runtime 的 app-server 将 Goal
  恢复为 `active`，再写一次 `CONTROL/RESUME`；
* Goal 为 `usage_limited` 或 `budget_limited`：写一次 `CONTROL/BUDGET_LIMITED` 并停止，
  不绕过平台预算；
* complete、failed、usage/budget limit 或未知状态：记录状态，不重启。

恢复周期合同：

* 同一 root Goal 在一个连续的 `paused` 或 `blocked` 状态周期内，最多执行一次
  lease-authorized app-server resume，并最多写一次对应的 `CONTROL/RESUME`；未发生
  状态变化时，后续 bounded check 不得重复恢复或重复唤醒。
* 当同一 Goal 被重新观察为 `active` 时，清除该周期的恢复消费标记；它之后再次进入
  `paused` 或 `blocked`，属于新的状态周期，可以在 lease 仍有效时再次恢复。
* 这个 guard 只约束 liveness 操作，不把 `paused`、`blocked` 或恢复失败解释为科研
  任务完成，也不改变 root 的 Mission-level Goal 生命周期。

恢复控制平面由 `scripts/runtime_support.py` 中的纯函数
`recovery_control_decision` 统一判断 Goal 状态、容量/终态和周期 guard；它不写
数据库、不调用 app-server。`root_watchdog.py` 是默认且唯一的 lease-authorized
恢复 owner，负责通过 app-server 验证 `active` 后写 `CONTROL/RESUME`；
`researchctl.py` 只管理 lease 与 Broker supervisor desired state。兼容
`agent_result_poller.py` 只有显式 opt-in 才能走旧 queue/app-server 路径，不能与
watchdog 并行作为默认 owner；`worker_event_poller.py` 只写 `TASK_UPDATE`。

这里共享的是判断算法，不是持久化状态：watchdog 与 legacy poller 各自维护本地周期
guard，helper 不提供跨进程去重或共享 persistent guard。因此部署只能选择一个恢复
owner，默认启用 watchdog，legacy 入口不得与其并行运行。

`usage_limited`、`budget_limited`、终态和未知状态都不得转换成恢复动作。外部
`server_overloaded` 等容量错误若使 Goal 回到 `blocked`，仍受同一 Goal 周期 guard
约束；恢复失败保留 `blocked` 观察状态，不得伪造完成或绕过平台上限。

watchdog 不读取 Cm、PPO、reward 或实验指标，也不选择下一任务。lease 缺失等同
于禁用。使用：

```bash
python3 scripts/researchctl.py supervisor resume \
  --registry docs/AGENT_REGISTRY.json \
  --broker-tasks-db .runtime/tasks.sqlite \
  --broker-state-db .runtime/AGENT_STATE.sqlite
python3 scripts/researchctl.py supervisor pause \
  --registry docs/AGENT_REGISTRY.json \
  --broker-tasks-db .runtime/tasks.sqlite \
  --broker-state-db .runtime/AGENT_STATE.sqlite
python3 scripts/researchctl.py supervisor status \
  --registry docs/AGENT_REGISTRY.json \
  --broker-tasks-db .runtime/tasks.sqlite \
  --broker-state-db .runtime/AGENT_STATE.sqlite
```

`pause` 先关闭 lease；如果还要立即停止当前 turn，再由用户通过 Codex Goal 控制
入口暂停 root。`resume` 重新授予 lease，watchdog 在下一次 bounded check 中处理
暂停或 blocked 的 Goal。

## 6. root Goal 生命周期

root 只保留一个 Mission-level Goal。**root Goal 是持续监督的恒定锚点**；child task
可以完成、失败或退回，但 child 终态不会结束 root Goal。每次 child 完成后，**child
完成后root必须回到任务选择循环**，审计 evidence、资源和停止条件，再从
`MISSION`、`STATE`、`CAMPAIGN` 与当前授权选择下一项 Blocker、Decision 或直接必要的
治理工作。

只有用户明确 pause/stop/switch，或平台明确施加 usage/budget/lifecycle limit，才
改变 root Goal 生命周期。三次重复无进展仍按 `AGENTS.md` 触发路线复盘；普通空闲、
状态报告、单个 task 终态和 watchdog wake 都不是暂停理由。

## 7. preflight、handoff 和集成

非 smoke 运行必须由执行 owner 在自己的 branch 写入 machine-readable preflight，
固定 code/route/checkpoint/motion hash、seed、matched control、输出目录、资源归属、
停止条件和 storage 预算。root 只读复核，不代替 owner 启动实验、补数据或修复实现。

只有 root 在终态审计、范围审计和验证通过后合并 worker branch。若 linked worktree
的 Git metadata 只读，owner 交接 `COMMIT=none`、允许文件的 sha256/mode/size，root
只能 byte-exact 导入后提交，不能在集成时改写研究内容。

固定 branch 下一 task 开始前同步 main：

```bash
git checkout agent/<role>
git merge --ff-only main
```

旧实验 branch、checkpoint、manifest 和运行数据不因本次 workflow 迁移删除或覆盖；
它们继续由实验 ID 和历史 handoff 作为证据索引。

## 8. 新建或迁移角色

新对话和角色注册流程见
[create-ref2dex-agent skill](../.agents/skills/create-ref2dex-agent/SKILL.md)。先检查
固定 pool 是否已有所需能力：已有角色就更新 `.runtime/AGENT_BINDINGS.json`；只有
pool 中没有该能力并得到用户明确批准时才创建新的 `agent_key`。迁移时保留旧身份到
`retired_conversations`，不复用 thread ID，并运行本文件第 3 节的校验。
