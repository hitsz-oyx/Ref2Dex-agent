# Ref2Dex 多代理协作、注册与连续监督契约

**状态：** ACTIVE  ·  **schema：** `ref2dex.agent_coordination.v4`  ·  **所有者：** `/root`

本文件规定固定角色池、运行绑定、任务交接和监督边界。实验结果仍只写入
experiment card、manifest 和 `docs/STATE.md`；本文件只描述 orchestration。

机器可读注册表是 [`docs/AGENT_REGISTRY.json`](AGENT_REGISTRY.json)。注册表中的
`pool.roles` 是稳定组织身份，`agents` 是当前或历史运行记录。可替换的
conversation、`CODEX_HOME`、数据库和工作树绑定存放在机器本地的
`.runtime/AGENT_BINDINGS.json`，不会把运行迁移写进研究历史。

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

只有 root 可以选择研究问题、创建 task、改变资源预算、接受交付和合并 main。
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
rollout、Git、manifest 和归属进程只读推导。修改 `AGENT_REGISTRY.json` 后运行：

```bash
python3 -m json.tool docs/AGENT_REGISTRY.json >/dev/null
python3 tools/verify.py --changed
```

注册表中的所有 legacy runtime record 仍须通过以下身份检查：

```bash
python3 - <<'PY'
import json
from pathlib import Path

d = json.loads(Path("docs/AGENT_REGISTRY.json").read_text())
agents = d["agents"]
keys = [(a["codex_home"], a["conversation_id"]) for a in agents]
assert len({a["agent_key"] for a in agents}) == len(agents)
assert len(set(keys)) == len(keys)
for a in agents:
    home = Path(a["codex_home"])
    assert home.is_absolute() and a["conversation_id"]
    assert Path(a["session_root"]) == home / "sessions"
    for field, filename in (("state_db", "state_5.sqlite"),
                            ("goal_db", "goals_1.sqlite"),
                            ("queue_db", "queue_1.sqlite")):
        assert Path(a[field]) == home / filename
print("AGENT_REGISTRY_VALID", len(agents))
PY
```

## 4. Task dispatch

root 发出的每个任务必须是一个可判别目标，不发送“继续”“再看看”或固定心跳：

```text
GOAL_DISPATCH
TARGET_AGENT_KEY=agent_cm|agent_rl|agent_eval|agent_infra
CODEX_HOME=<from runtime binding>
CONVERSATION_ID=<from runtime binding>
OBJECTIVE=<one concrete decision or blocker>
DECISION_TEST=<cheapest discriminating test>
BRANCH=<fixed role branch>  BASE_COMMIT=<sha>
ALLOWED_PATHS=<paths>
EXPERIMENT_OR_CARD=<id or NONE>
RESOURCE_BUDGET=<gpu/time/storage/process>
EXECUTION_OWNER=<same target role>
STOP_CONDITIONS=<explicit conditions>
DELIVERABLE=<commit/card/manifest/handoff>
NOT_AUTHORIZED=<what must not start>
```

同一 worker 同时只运行一个 task。task identity 用 `TASK-*`、Probe ID 或 Validation
ID 表达，不创建 task 专用 branch。worker 完成时交接：

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
manifest、rollout completion、归属 GPU PID。发现变化后向 root queue 一次性发送
`POLL_EVENT`；队列失败时保留自己的 cursor，不能确认事件。它不读取指标来选择
路线，不启动或停止实验，不修改任何 Goal，也不发周期性“全局巡检”文本。

`root_watchdog.py` 是另一条极小的 liveness 进程，只观察 root 自己的 Goal、root
rollout 和 root queue。它只允许以下动作：

* root active 但超过 grace period 没有 turn 或 queued input：发送一次
  `ROOT_LIVENESS_WAKE`；
* root Goal paused 且本地 `.runtime/SUPERVISOR_LEASE.json` 同时声明
  `enabled=true`、`allow_root_resume=true`：通过受支持的 app-server bounded resume
  恢复一次同一 Goal，然后发送一次 liveness wake；
* Goal 为 `usage_limited` 或 `budget_limited`：输出一次
  `ROOT_BUDGET_LIMITED` 并停止，不绕过平台预算；
* complete、failed、blocked 或其他终态：记录状态，不重启。

watchdog 不读取 Cm、PPO、reward 或实验指标，也不选择下一任务。lease 缺失等同
于禁用。使用：

```bash
python3 scripts/researchctl.py supervisor pause
python3 scripts/researchctl.py supervisor resume
python3 scripts/researchctl.py supervisor status
```

`pause` 先关闭 lease；如果还要立即停止当前 turn，再由用户通过 Codex Goal 控制
入口暂停 root。`resume` 只重新授予 lease，watchdog 在下一次 bounded check 中处理
暂停的 Goal。

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
