# Ref2Dex Agent Broker

新对话的入口是仓库根目录 [`AGENTS.md`](../AGENTS.md) 第 2 节；本文件必须与
[`docs/AGENT_ROLES.yaml`](AGENT_ROLES.yaml)、[`docs/AGENT_COORDINATION.md`](AGENT_COORDINATION.md)、
[`docs/ROOT_AGENT.md`](ROOT_AGENT.md) 和
[create-ref2dex-agent skill](../.agents/skills/create-ref2dex-agent/SKILL.md) 一起读取。

`scripts/agent_broker.py` 是固定 worker 池的本地、无模型 orchestration 层。它只做
角色校验、任务队列、一次性租约、消息持久化、handoff 和 liveness 状态；研究判断仍
属于 root，provider 只属于 runtime adapter。

Broker 的 supervisor desired state 是派发闸门：只有 `RUNNING` 接受新的 dispatch
或 claim；`PAUSED`/`STOPPED` 会拒绝新任务，但允许已经持有租约的任务收尾。暂停不会
杀死训练进程。

当前身份分层如下：

| 层 | 文件 | 内容 |
| --- | --- | --- |
| tracked policy | `docs/AGENT_ROLES.yaml` | 固定 `agent_key`、职责、权限和 branch |
| local binding | `.runtime/AGENT_BINDINGS.json` | provider、profile、`CODEX_HOME`、thread、worktree |
| runtime state | `.runtime/AGENT_STATE.sqlite` | worker 状态、心跳、root desired state |
| task queue | `.runtime/tasks.sqlite` | task、lease、四类消息和 handoff |

`.runtime` 全部为机器本地状态，不能作为研究证据提交。角色身份不会随 thread、
provider 或 worktree 迁移；迁移只替换 binding，并保留旧 thread 的审计记录。

## Message contract

Broker 只接受以下四类消息：

* `TASK_DISPATCH`：root 发给一个固定 worker。包含 `task_id`、objective、context、
  constraints 和 `done_when`。
* `TASK_UPDATE`：worker 或只读 event poller 发给 root。包含进度、状态或观察快照。
* `TASK_HANDOFF`：worker 发给 root。包含 handoff status、commit、evidence、结果和
  单一下一步建议。
* `CONTROL`：root、watchdog 或本地控制命令发出的 `PAUSE`、`RESUME`、`WAKE`、
  `BUDGET_LIMITED`、`CANCEL` 等控制信号。

worker 之间不能直接通信；跨角色依赖必须用 `TASK_HANDOFF` 退回 root，由 root 重新
发出 `TASK_DISPATCH`。Broker 不创建 agent，不调用模型，不推断科学结论。

worker 必须先 claim 获得 lease；每次 `TASK_UPDATE`、`TASK_HANDOFF` 和 lease renewal
都必须携带同一有效 token。任务退出而没有 handoff 会被 runtime adapter 标记为失败。

## Root recovery control plane

`researchctl.py` 是 lease/supervisor 控制入口，只能写 `CONTROL/PAUSE` 或
`CONTROL/RESUME` 来改变 Broker 的 desired state；它不直接恢复 Goal。默认恢复 owner
是 `root_watchdog.py`：经过 runtime identity、lease 和 Broker desired-state 检查后，
它才通过同一 root app-server 做一次 `paused`/`blocked` readback，并写对应的
`CONTROL/RESUME`。兼容入口 `agent_result_poller.py` 的 `--root-goal-resume-once` 是迁移期兼容
开关，`worker_event_poller.py` 只能写 `TASK_UPDATE`。

所有入口使用同一纯 `recovery_control_decision` 算法；该 helper 不持久化 guard，也不
提供 watchdog 与 legacy poller 之间的跨进程去重。每个进程的周期 guard 仍保存在各自
状态文件中，因此部署时只能有一个 recovery owner：默认使用 watchdog，legacy
`--root-goal-resume-once` 仅作显式兼容入口，不能与 watchdog 并行运行。容量/预算状态
（`usage_limited`、`budget_limited`）、终态和未知状态只记录或写 `CONTROL/BUDGET_LIMITED`，
不绕过平台上限；外部容量错误导致的 `blocked` 仍最多恢复一次，直到观察到 `active` 才
开启下一周期。

## Provider adapters

代码内置四个确定性 adapter 名称：`codex_app_server`、`codex_cli`、`newapi` 和
`other`。它们把任务转换成统一的 runtime request；真正的 Codex App Server、CLI 或
OpenAI-compatible NewAPI 客户端由对应外部 runtime 消费 binding，不能通过共享登录凭证
或隐式继承 root account。替换 provider 只修改本机 binding，不修改 root 研究逻辑。

## CLI smoke

先通过 `researchctl` 或 Broker control 将 supervisor 置为 `RUNNING`；默认状态是
`PAUSED`，暂停时新的 dispatch/claim 会被拒绝。

```bash
python3 scripts/agent_broker.py \
  control --action RESUME --target root --reason 'broker smoke'

python3 scripts/agent_broker.py dispatch \
  --task-id T-20260928-infra-broker \
  --target agent_infra \
  --objective 'Run the broker contract smoke test' \
  --constraints '{"gpu": 0}' \
  --done-when 'test output' \
  --done-when 'handoff'

python3 scripts/agent_broker.py claim --agent agent_infra
python3 scripts/agent_broker.py status
```

运行前先让目标角色拥有一个有效的本机 binding；没有 binding 的固定角色会被拒绝，
不会静默创建新的 runtime。

实际 provider 投递由显式 runtime launcher 完成。它可以通过 stdin 接收标准化的
`TASK_DISPATCH` JSON，并使用 `REF2DEX_TASK_ID`、`REF2DEX_LEASE_TOKEN` 等环境变量
回写 Broker：

```bash
python3 scripts/agent_runtime_adapter.py \
  --agent agent_infra \
  --command '/path/to/provider-launcher --stdin-json' \
  --request-dir .runtime/provider_requests
```

该命令不猜测 provider，也不会共享 root 凭证；没有显式 launcher 时，Broker 只排队，
不会把排队状态伪装成已启动。
