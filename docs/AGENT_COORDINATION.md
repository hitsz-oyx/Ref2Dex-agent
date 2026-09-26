# Ref2Dex 多代理协作、注册与连续监督契约

**状态：** ACTIVE  ·  **schema：** `ref2dex.agent_coordination.v3`  ·  **所有者：** `/root`

本文件规定 Ref2Dex 的四个 Codex 对话如何注册、分工、交接和接受监督。它不
保存实验结果。当前对话 ID、`CODEX_HOME`、角色和工作树的机器可读注册表是：

[`docs/AGENT_REGISTRY.json`](AGENT_REGISTRY.json)

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

当前登记四个稳定的 `agent_key`：

* `root`：全局主代理；
* `agent_baseline`：GRAB 全池 baseline / provenance 代理；
* `agent_cm_temporal`：Cm temporal Probe 代理（显示名为 `agent_Cm/temporal`）；
* `agent_workflow`：工作流与治理代理（显示名为 `agent/workflow`）。

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
assert len({a["agent_key"] for a in agents}) == len(agents)
assert len(set(keys)) == len(keys)
assert all(a["codex_home"] and a["conversation_id"] for a in agents)
print("AGENT_REGISTRY_VALID", len(agents))
PY
```

若 supervisor、CLI 或外部脚本显式写入了旧 `CODEX_HOME`/thread ID，必须同步替换。
不得通过修改显示名来掩盖身份迁移。

## 2. 当前角色与边界

| 角色 | 主要责任 | 允许做的事 | 明确禁止 | 必须交接 |
| --- | --- | --- | --- | --- |
| `/root` | 全局监督、目标管理、资源分配、验收和 main 集成 | 在同一条对话中轮询；按需派发明确 goal；只读审计；撰写 Decision Memo；审计后合并已接受提交 | 固定心跳式自唤醒；盲目重启/杀进程；重复实验；把 Probe 升格为 Validation | 证据化状态、commit、manifest、资源证据和下一决策 |
| `agent_baseline` | CPU-only canonical baseline 与 provenance | evaluator/config/checkpoint/motion hash 审计；matched-off preflight；CPU smoke、静态审计和测试 | GPU、Isaac Gym、collector、PPO；消费 Cm/HF02 预算；未经新 goal 开启路线 | branch/commit、manifest 路径与 hash、审计结论、终态 |
| `agent_Cm/temporal` | 一个明确授权的 Cm Probe | 只执行当前 card 冻结的 collection/fit，并遵守 GPU/时间/存储预算 | online/PPO follow-up；改 seed、route、metric、horizon；失败后扫描同一假设；复用别的 run | card 状态、run manifest、输入 hash、资源证据、Probe 标签和 blocker |
| `agent/workflow` | 治理、上下文效率和验证工具 | 修改工作流文档/模板/治理测试；运行治理测试和 `tools/verify.py` | GPU/Isaac Gym；科研实验；未经 Decision Checkpoint 改科学 claim | commit、测试输出、改动范围和已知取舍 |

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

### 4.1 目标派发模板

`/root` 只有在发现一个**已授权且可判别的 blocker**时，才向对应 child thread
发送如下类型的定向消息：

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

派发后 `/root` 等待该 thread 的真实终态或新的 blocker；不再用“继续”消息制造
活动，也不重复投递相同 goal。`paused` 或 `blocked` 的 Goal 不得被静默唤醒，
除非用户建立新授权或 Decision Memo 明确解除阻塞。

如果确实满足授权条件，`/root` 可以用注册表中的两项身份做**一次性**派发（命令
中的 `<codex-cli>` 由当前环境解析，不能写死另一个 `CODEX_HOME`）：

```bash
CODEX_HOME=<registry.codex_home> <codex-cli> queue \
  --thread <registry.conversation_id> \
  --message '<完整 GOAL_DISPATCH 消息>'
```

这条命令只对应一个明确 goal；不得把它包进 cron、setsid、循环脚本或固定心跳。

## 5. 主代理的连续监督循环（取代外部固定唤醒）

研究监督由 `/root` 在**同一条主对话的活动 turn 内**完成，不使用外部 watchdog
向 `/root` 周期性投递固定的“执行一次全局 supervisor 巡检”文本。外部脚本
`codex_research_supervisor.py` 的旧队列唤醒模式已经 retired；它不是新的任务
调度器，也不应重新启动。

`/root` 开始监督时建立一个连续循环：

1. 做一次 preflight，读取注册表并解析每个已登记 thread 的 `CODEX_HOME`；
2. 只读检查 thread/Goal、branch/HEAD、card/manifest、归属进程和 GPU；
3. 将每个 child 分类为 `RUNNING`、`WAITING`、`TERMINAL`、`BLOCKED` 或 `UNKNOWN`；
4. 若没有已授权 blocker，记录内部状态 `SUPERVISOR_IDLE`，使用等待/定时轮询
   保持当前 turn，不向自己或 child 发送固定心跳消息；
5. 若有 blocker，只按第 4.1 节派发一个明确 goal，然后继续轮询其证据；
6. 在终态、异常或 Decision Checkpoint 时立即做 completion audit，并向用户交接。

默认轮询节奏：

| 阶段 | 间隔 | 检查 |
| --- | --- | --- |
| queued / waiting / blocked | 5 分钟 | thread turn、Goal、branch/HEAD、最新 handoff、资源归属 |
| active CPU analysis | 5 分钟 | 进程、manifest/log heartbeat、输出增长、时间预算 |
| active GPU collection | 2 分钟，最长不超过 5 分钟 | owner 进程、显存/利用率、manifest、停止条件 |
| terminal / exception | 立即一次 | card、manifest、hash、commit、归属进程和下一决策 |

“对话本身一直不中断”指 `/root` 在产品允许的单次活动 turn 内使用等待和
轮询保持监督，不主动发送最终结束消息。若平台或用户中断 turn，恢复时必须从
注册表、Goal 数据库和最后一次状态快照继续；不得假设 child 已经完成，也不得
自动重启实验。这个连续循环不改变任何资源或实验授权。

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
/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-main
branch: main
```

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
