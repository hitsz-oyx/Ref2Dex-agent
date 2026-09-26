# Ref2Dex 多代理协作、注册与连续监督契约

**状态：** ACTIVE  ·  **schema：** `ref2dex.agent_coordination.v3`  ·  **所有者：** `/root`

本文件规定 Ref2Dex 当前已登记的 Codex 对话如何注册、分工、交接和接受监督。它不
保存实验结果。当前对话 ID、`CODEX_HOME`、角色和工作树的机器可读注册表是：

[`docs/AGENT_REGISTRY.json`](AGENT_REGISTRY.json)

`/root` 的持续推进、用户打断与停下条件见
[`docs/ROOT_AGENT.md`](ROOT_AGENT.md)。

## 1. 注册制：身份与运行目录

`AGENT_REGISTRY.json` 是代理身份映射的唯一来源。每个登记项至少包含：

* 稳定的 `agent_key`；
* Codex `conversation_id`（即 thread ID）；
* 该对话所属的 `codex_home`；
* rollout、state、Goal 和 queue 数据库的定位；
* 角色、worktree、branch、权限边界和 handoff 要求。

身份键是 **`(codex_home, conversation_id)`**，而不是单独的 conversation ID。不同
`CODEX_HOME` 下可以存在相同格式的 ID；读取或派发消息时必须使用注册表中同一
条记录的两项，不能依赖默认目录或自动发现。
每个 agent 的 `codex_home` 独立登记；当前多个 agent 可以碰巧使用同一个目录，
但注册表没有共用的 `default_codex_home`，也不允许从 root 或环境变量继承。

当前登记五个稳定的 `agent_key`：

* `root`：全局主代理；
* `agent_baseline`：GRAB 全池 baseline / provenance 代理；
* `agent_cm_temporal`：Cm 路线只读证据复核与新假设设计代理（显示名为 `agent_Cm/selective`）；
* `agent_workflow`：工作流与治理代理（显示名为 `agent/workflow`）。
* `agent_poller`：只读结果轮询与事件通知代理（显示名为 `agent/poller`）。

注册表不记录动态的 `RUNNING`、`BLOCKED` 或 `COMPLETED` 状态；动态状态应从
对应 `codex_home` 下的 rollout、Goal 数据库、Git 和 run manifest 读取。角色迁移
时先在 `retired_conversations` 记录旧身份，再登记新身份；不要复用旧 thread ID。

修改注册表后，至少执行：

```bash
python3 -m json.tool docs/AGENT_REGISTRY.json >/dev/null
python3 - <<'PY'
import json
from pathlib import Path

d = json.loads(Path("docs/AGENT_REGISTRY.json").read_text())
agents = d["agents"]
keys = [(a["codex_home"], a["conversation_id"]) for a in agents]
assert "default_codex_home" not in d
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
    assert a["conversation_id"] in a["rollout_locator"]
print("AGENT_REGISTRY_VALID", len(agents))
PY
```

若 supervisor、CLI 或外部脚本显式写入了旧 `CODEX_HOME`/thread ID，必须同步替换。
不得通过修改显示名来掩盖身份迁移。

## 2. 当前角色与边界

| 角色 | 主要责任 | 允许做的事 | 明确禁止 | 必须交接 |
| --- | --- | --- | --- | --- |
| `/root` | 全局监督、目标管理、资源分配、验收和 main 集成 | 在 `MISSION`/`CAMPAIGN` 与当前授权内主动选择有决策价值的任务；派发明确 goal；只读审计；撰写 Decision Memo；审计后合并已接受提交 | 固定心跳式自唤醒；盲目重启/杀进程；重复实验；把 Probe 升格为 Validation | 证据化状态、commit、manifest、资源证据和下一决策 |
| `agent_baseline` | CPU-only canonical baseline 与 provenance | evaluator/config/checkpoint/motion hash 审计；matched-off preflight；CPU smoke、静态审计和测试 | GPU、Isaac Gym、collector、PPO；消费 Cm/HF02 预算；未经新 goal 开启路线 | branch/commit、manifest 路径与 hash、审计结论、终态 |
| `agent_Cm/selective` | 已完成 HF05 的证据保全与新假设设计 | CPU-only 只读复核既有 card、manifest、结果索引并草拟 Decision Memo | 新 Cm Probe/fit/collection、GPU、Isaac Gym、online/PPO；重扫 HF05；复用别的 run | 证据入口、候选假设、最小判别测试、资源估计和下一决策 |
| `agent/workflow` | 治理、上下文效率和验证工具 | 修改工作流文档/模板/治理测试；运行治理测试和 `tools/verify.py` | GPU/Isaac Gym；科研实验；未经 Decision Checkpoint 改科学 claim | commit、测试输出、改动范围和已知取舍 |
| `agent/poller` | 登记代理的只读轮询和事件通知 | 读取 Goal/Git/manifest/rollout completion/归属 GPU；仅在变化时向 root 排队 `POLL_EVENT` | GPU 实验；修改其他工作树；科学判断；合并；固定心跳 | 变化字段、旧/新快照、通知状态和错误 |

代理之间可以并行，但权限不继承：一个代理的授权不能被另一个代理解释为
新实验、换资源或改变研究问题的授权。

## 3. 启动任务的最小上下文

新任务默认按以下顺序读取：

1. `AGENTS.md`；
2. 本文件与 `docs/AGENT_REGISTRY.json`；
3. `docs/MISSION.md`、`docs/STATE.md`、`docs/CAMPAIGN.md`；
4. 当前角色相关代码和 active experiment card。

只有在创建/批准 Probe 或分配 seed 时读取 `RESEARCH_QUEUE.yaml`、
`SEED_LEDGER.yaml`。不要批量读取历史 plan、全部 Activity、全部 experiment
或无关 Git 历史；只有证据冲突、重复实验核查或正式 Validation 才追溯历史。

`STATE.md` 是当前路由索引，不是实验日志。seed、指标、traceback、run 路径
放在 experiment card 和 manifest 中。

## 4. Goal 与明确派发合同

每个委派 goal 必须在一条短消息中说明：

* `TARGET_AGENT_KEY`、`CODEX_HOME` 和精确 `conversation_id`；
* 决策问题与最便宜的可判别测试；
* 精确 branch、commit 和允许修改的路径；
* experiment/card ID、seed、control 和资源预算；
* 停止条件与交付物；
* 明确未授权启动的内容。

“继续”“再看看”或固定巡检句子都不是 goal。新路线、新 claim、新资源类别或
online follow-up 必须建立新的 goal；若触发 `AGENTS.md` 的 Decision Checkpoint，
先提交 Decision Memo。实验必须先区分 Blocker、Decision、Evidence 或 Curiosity；
探索阶段只能形成 `PROMISING`、`UNPROMISING` 或 `UNCLEAR`。

派发不要求先出现 blocker。`/root` 应在 `MISSION.md`、`CAMPAIGN.md` 和当前授权
内主动选择具体、可判别的 `Blocker` 或 `Decision`；直接支撑当前决策的工程/治理
任务也可以派发。`Evidence` 留在 Research Debt，`Curiosity` 默认不做；任何新
授权、资源升级或研究路线变化仍按 Decision Checkpoint 处理。

### 4.1 目标派发模板

`/root` 选出一个边界内、具体且可判别的 `Blocker`、`Decision` 或直接必要的
工程/治理任务后，向对应 child thread 发送如下类型的定向消息：

```text
GOAL_DISPATCH
TARGET_AGENT_KEY=agent_baseline|agent_cm_temporal|agent_workflow
CODEX_HOME=<from AGENT_REGISTRY.json>
CONVERSATION_ID=<from AGENT_REGISTRY.json>
OBJECTIVE=<one concrete decision/blocker>
DECISION_TEST=<cheapest discriminating test>
BRANCH=<exact branch>  BASE_COMMIT=<sha>
ALLOWED_PATHS=<paths>
EXPERIMENT_OR_CARD=<id or NONE>
RESOURCE_BUDGET=<gpu/time/storage/process>
STOP_CONDITIONS=<explicit conditions>
DELIVERABLE=<commit/card/manifest/handoff>
NOT_AUTHORIZED=<what must not start>
```

同一 goal 运行期间不重复投递；由 `agent_poller` 在登记状态或 rollout completion
变化时发送 `POLL_EVENT`。`/root` 收到完成事件后立即审计，并回到当前目标的主动
选择循环，而不是等待新的 blocker。不得用“继续”消息制造活动。`paused` 或
`blocked` 的 Goal 不得被静默唤醒，除非用户建立新授权或 Decision Memo 明确
解除阻塞。

如果任务符合上述边界，`/root` 可以用注册表中的两项身份做**一次性**派发（命令
中的 `<codex-cli>` 由当前环境解析，不能写死另一个 `CODEX_HOME`）：

```bash
CODEX_HOME=<registry.codex_home> <codex-cli> queue \
  --thread <registry.conversation_id> \
  --message '<完整 GOAL_DISPATCH 消息>'
```

这条命令只对应一个明确 goal；不得把它包进 cron、setsid、循环脚本或固定心跳。

## 5. 主代理的连续决策循环（由 poller 通知事件）

研究决策、任务选择和资源授权由 `/root` 负责；外部 `agent_poller` 只负责登记
状态和 completion 的事件通知。不得使用 watchdog 向 `/root` 周期性投递固定的
“执行一次全局 supervisor 巡检”文本。外部脚本 `codex_research_supervisor.py`
的旧队列唤醒模式已经 retired；它不是新的任务调度器，也不应重新启动。

`/root` 开始监督时建立一个连续循环：

1. 做一次 preflight，读取注册表并解析每个已登记 thread 的 `CODEX_HOME`；
2. 结合 poller 事件只读检查 thread/Goal、branch/HEAD、card/manifest、归属进程和
   GPU，将每个 child 分类为 `RUNNING`、`WAITING`、`TERMINAL`、`BLOCKED` 或
   `UNKNOWN`；
3. 初始审计或 completion audit 后，检查 `MISSION`、`STATE`、`CAMPAIGN` 和当前
   授权，按 `Blocker`、`Decision`、`Evidence`、`Curiosity` 分类候选任务；
4. 若存在边界内的 `Blocker`、`Decision` 或直接必要的工程/治理任务，只按第
   4.1 节派发明确 goal；不以 blocker 已经出现为前提；
5. 只有候选扫描后没有可执行任务，才记录内部状态 `SUPERVISOR_IDLE` 并等待
   `POLL_EVENT` 或用户消息，不向自己或 child 发送固定心跳；
6. `agent_poller` 报告终态或异常后，`/root` 立即做 completion audit，再回到第
   3 步选择下一项工作；只有触发 `docs/ROOT_AGENT.md` 的停下条件时才交给用户
   决策或结束当前活动。

`agent_poller` 的默认事件采集节奏：

| 阶段 | 间隔 | 检查 |
| --- | --- | --- |
| queued / waiting / blocked | 5 分钟 | thread turn、Goal、branch/HEAD、最新 handoff、资源归属 |
| active CPU analysis | 5 分钟 | 进程、manifest/log heartbeat、输出增长、时间预算 |
| active GPU collection | 2 分钟，最长不超过 5 分钟 | owner 进程、显存/利用率、manifest、停止条件 |
| terminal / exception | 立即一次 | card、manifest、hash、commit、归属进程和下一决策 |

“对话本身一直不中断”指 `/root` 在产品允许的单次活动 turn 内等待真实事件并
持续完成审计和任务选择，不因一次巡检或子任务完成而主动结束。若平台或用户
中断 turn，恢复时必须从注册表、Goal 数据库和最后一次状态快照继续；不得假设
child 已经完成，也不得自动重启实验。这个连续循环不改变任何资源或实验授权。

[结果轮询代理](AGENT_POLLER.md) 是默认事件通知 owner；root 不与它并行启动重复
的长期轮询进程。poller 只在登记状态或 completion 变化时定向排队 `POLL_EVENT`，
root 被唤起后负责审查、选择下一项任务并继续。这类事件通知不等同于已退役的
固定心跳式自唤醒。

现有 overload watchdog 仍只处理结构化的 `server_overloaded` 事件；它不负责
研究监督、目标派发或唤醒 `/root`。

停用旧调度器时，先只读检查注册表中对应的 `queue_db`。若存在遗留固定巡检文本，
必须先做 SQLite 一致性备份，再用 `thread_id`、消息 ID 和文本内容精确匹配清理；
不得清空整个队列或触碰其他 thread 的排队消息。清理后再次只读确认没有遗留固定
心跳，未来只允许明确授权的 `GOAL_DISPATCH` 入队。

## 6. 非 smoke 运行前的 preflight

运行前由 owner 写入 machine-readable `preflight.json`，至少固定：

1. branch、HEAD、worktree 范围、card schema 和 experiment ID；
2. route、config、evaluator、checkpoint、motion 与代码 hash；
3. seed ownership、matched control 和唯一输出目录；
4. GPU/process ownership、设备、显存快照和 campaign 限制；
5. dry-run/测试、输出 schema、wall-time、storage、非有限值、漂移和行数停止条件。

失败的接线或 import 修复保留为 `INVALID_IMPLEMENTATION`；不能把它包装成
科学结果。没有完整 preflight、manifest 或明确进程归属时，停止并交接。

## 7. 交接格式与证据边界

每个 owner 结束时提供以下紧凑记录：

```text
STATUS=<terminal state>  LABEL=<if applicable>
BRANCH=<name>  HEAD=<sha>
CARD=<experiment/card id>  RUN=<run id or manifest path>
INPUTS=<key hashes>  RESOURCES=<GPU/process/time/storage>
EVIDENCE=<tests/metrics/known limits>
NEXT=<single decision or blocker>
```

只有 commit、card、manifest、hash 和资源审计可以相互复核时，supervisor 才接受
终态。单 seed、单 rollout、loss、视频或工程 smoke 不能被描述为正式科学结论。
连续三个有效 Probe 没有 North-star 进展时，切换高层路线并触发路线复盘，不得
在同一局部问题上无限换 seed/metric/horizon。

## 8. mainline 集成归属

规范集成工作树为：

```text
/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent
branch: main
```

`agent/grab-full-baseline` 的工作树为同级的 `Ref2Dex-agent-baseline`。

只有 `/root` 可以把已接受的子代理提交合并到 `main`。合并前必须核对：子线程
终态、diff 范围、测试/verify 输出、工作树干净、归属进程已结束，以及分支是否
含有无关研究历史。混合 ancestry 的分支只能挑选明确接受的提交，不能因为分支
tip 通过测试就整体合并。合并后重新运行 mainline verification，并向用户报告
新的 main commit。

## 9. 如何更换对话或 `CODEX_HOME`

新建对话或迁移运行目录后：

1. 在 `docs/AGENT_REGISTRY.json` 找到目标 `agent_key`；
2. 同时更新 `codex_home`、`conversation_id` 以及该记录下的数据库/rollout 路径；
3. 保留角色、worktree、branch、资源边界和 handoff 要求；
4. 运行第 1 节的 JSON/身份键唯一性检查；
5. 更新任何显式 supervisor/dispatch 参数；
6. 在下一次 handoff 中说明旧身份、新身份和生效时间。

不要在本文件、experiment card 或 `STATE.md` 另行维护一份会漂移的 ID 清单。
需要变更身份职责时，先修改注册表和本节角色契约，再创建 goal；不能只改显示名。

## 10. 如何新建代理

新代理的对话命名、重名数字后缀、按 `CODEX_HOME` 选择网络入口、工作树创建和
注册步骤见 [create-ref2dex-agent skill](../.agents/skills/create-ref2dex-agent/SKILL.md)。
先完成独立身份和权限登记，再派发明确 Goal；新对话不自动继承旧对话的身份、
资源或当前研究授权。
