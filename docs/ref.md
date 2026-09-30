有，而且现在 GitHub 上已经出现了几套和你想法非常接近的项目。**没有一套和你现在定义的 Ref2Dex workflow 100% 一样，但已经不需要从零造 broker/runtime transport 了。**

我按“接近你的目标程度”来看，最值得研究的是下面几套。

| 项目                                    | 和你需求的匹配点                                                                          | 主要差异                                                            |
| ------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **zetbrush/multiagents**              | 多 provider、Broker daemon、SQLite、持久 session、Codex App Server、自动重启、agent 消息转发       | 默认偏动态 spawn team，不是固定员工池                                        |
| **backnotprop/orchestrator**          | parent 决策、CLI 做确定性执行、durable task store、多 runtime、Codex persistent session/resume | 更偏 task-oriented orchestration，不自带你的 research state machine     |
| **hyspacex/harness-cli**              | 固定 role、不同 role 可以绑定不同 provider、durable state、resume                              | 偏软件开发流水线，不是长期科研 supervisor                                      |
| **formiat/multi-agent-orchestration** | Codex 主导、Claude/OpenCode 执行、session reuse、inbox/outbox handoff                    | 更像 workflow library，不是完整 daemon                                 |
| **codex-mission-ledger**              | durable control plane、background research、worktree 隔离、层级 agent                    | 更偏 Codex native hierarchical agents，不符合你“固定跨 provider worker”目标 |

### 1. 最像你 runtime 层的是 `zetbrush/multiagents`

它已经有：

```text
Claude Code
Codex CLI
Gemini CLI
      │
      ▼
Broker Daemon
SQLite + HTTP
      │
      ▼
Orchestrator
```

而且 Codex 不是靠简单 CLI prompt，而是直接用 **Codex App Server**：

* persistent thread；
* `turn/start`；
* turn 中 `turn/steer`；
* 卡住后 `turn/interrupt`；
* broker 每几秒转发消息；
* idle 时重新启动 turn；
* session 重启后仍保留消息历史；
* agent crash 后自动 restart + handoff。([GitHub][1])

这其实已经解决了你现在自己最费劲的：

> **Broker → runtime transport → 唤醒 Codex**

这一整层。

它和你最大的区别在这里：

```text
multiagents:
orchestrator → spawn team dynamically
```

而你想要：

```text
root → 固定 agent_cm
     → 固定 agent_rl
     → 固定 agent_eval
     → 固定 agent_infra
```

所以如果借它，我会保留其：

```text
Broker
CodexDriver
session persistence
message delivery
liveness
```

但把动态：

```text
create_team()
spawn_agent()
```

换成固定：

```text
bind(agent_key)
dispatch(agent_key)
```

[zetbrush/multiagents](https://github.com/zetbrush/multiagents?utm_source=chatgpt.com)

---

## 2. 我认为最值得你直接借架构的是 `backnotprop/orchestrator`

这个项目的哲学和我们前面讨论的几乎一致：

> **calling agent owns judgment；CLI owns process supervision and task state。**

也就是：

```text
root / parent agent
        │
        │ decides
        ▼
Orchestrator CLI
        │
        ├── Claude Code
        ├── Codex
        ├── Copilot
        ├── Grok
        ├── Pi
        └── custom process
```

它明确把：

```text
研究/任务判断
```

和：

```text
process launch
status
logs
resume
interrupt
task storage
```

分开。([GitHub][2])

这就是你现在的：

```text
root
+
Agent Broker
+
runtime_bridge
```

只不过他们已经把后两者实现得比较成熟了。

尤其是它有一个机器级：

```text
~/.orchestrator/tasks/
```

作为 durable task store，而且每个 worker 都有：

* task ID；
* status；
* logs；
* output；
* follow-up；
* resume；
* interrupt。([GitHub][2])

---

### 它对 Codex 的支持尤其值得我们借

它区分：

```text
codex
```

一次性任务，和：

```text
codex-app-server --session
```

长期 session。

persistent session 下，它管理 Codex thread，并支持继续给同一个 thread 发任务、follow-up、Goal 和 interrupt。([GitHub][3])

这基本就是我们现在正在重新实现的：

```text
.runtime/AGENT_BINDINGS
conversation_id
resume
wake
goal
```

所以我现在反而**不建议继续自己实现完整 Codex transport**。

可以让：

```text
agent_cm
→ orchestrator persistent session A

agent_rl
→ orchestrator persistent session B
```

然后我们的 Broker 只记录：

```yaml
agent_cm:
  runtime: orchestrator
  session: cm

agent_rl:
  runtime: orchestrator
  session: rl
```

这样 Goal/turn/session resume 这一层交给已有项目维护。

[backnotprop/orchestrator](https://github.com/backnotprop/orchestrator?utm_source=chatgpt.com)

---

# 3. 它甚至已经支持我们需要的跨 provider

`backnotprop/orchestrator` 当前 runtime 包括：

```text
claude-code
codex
codex-app-server
copilot
grok
pi
shell
custom process
```

并且明确把 runtime registry 和 agent/model provider 分开。([GitHub][2])

所以：

```text
root = subscription A Codex
agent_cm = subscription B Codex
agent_eval = Claude
agent_rl = NewAPI/custom runtime
```

从架构上完全可以实现。

NewAPI 如果没有现成 runtime，也可以作为：

```text
custom process runtime
```

接进去，而不用改整个 supervisor。

---

# 4. `hyspacex/harness-cli` 有你想要的“固定角色绑定 provider”

这个项目值得参考的点不是 runtime，而是：

```json
{
  "provider": "claude-sdk",
  "roleProviders": {
    "planner": "codex",
    "generator": "codex"
  }
}
```

也就是说：

```text
role
→ provider
```

是显式配置，而不是 parent 动态创建子 agent。([GitHub][4])

而且它明确强调：

> durable state 属于 harness，而不是 model session。

这和我们说的：

```text
agent_key = identity
thread = replaceable runtime
```

几乎完全一致。([GitHub][5])

这部分设计我会直接借：

```yaml
agent_cm:
  provider: codex-a

agent_rl:
  provider: newapi-b

agent_eval:
  provider: codex-c
```

---

# 5. `formiat/multi-agent-orchestration` 和你的 handoff 思路非常像

它的架构是 Codex 作为 orchestrator，然后把 planning / investigation / implementation / review 派给 Claude 或 OpenCode。

比较有意思的是它明确把：

```text
session binding
dispatch
request fingerprint
cooldown
result collection
```

标准化，而且优先 **reuse existing session**。

跨 agent 的传输甚至就是：

```text
.codex/inbox.md
.codex/outbox.md
```

这种非常简单的 transport。([GitHub][6])

它说明了一个重要点：

> 我们的 `TASK_DISPATCH/TASK_HANDOFF` 不需要设计得特别复杂。

关键是 durable state 和 session binding，而不是协议本身有多少字段。

[formiat/multi-agent-orchestration](https://github.com/formiat/multi-agent-orchestration?utm_source=chatgpt.com)

---

# 6. 还有一个和你“跨 provider Codex”直接相关的项目

`Gan-Xing/CodexProvider` 的目标就是：

> 让非 OpenAI 模型参与 Codex 原生 tool-call loop。

它让 Codex App Server 保持工具执行、approval、MCP 和 continuation，而把模型请求转发到：

```text
DeepSeek
OpenRouter
Claude-compatible
其他 OpenAI-compatible upstream
```

([GitHub][7])

这个对你的 NewAPI 特别有参考价值。

因为你现在实际上想做的是：

```text
Codex harness
     │
     ▼
不同 provider/account
```

不一定需要为每个 provider 重写整个 worker runtime。

---

# 所以：你的系统并不是一个没人做过的方向

现在社区里已经逐渐收敛成这样的模式：

```text
             Parent / Supervisor Agent
                       │
                 judgment only
                       │
                       ▼
              deterministic control
              ┌─────────────────┐
              │ task store      │
              │ broker          │
              │ session manager │
              │ runtime drivers │
              └─────────────────┘
                 │      │      │
               Codex Claude  others
```

你特殊的部分主要是上面再加了一层：

```text
MISSION
RESEARCH_QUEUE
hypothesis budget
Probe / Validation
matched causal evidence
```

也就是：

> **现有项目主要解决“多 agent 工程 orchestration”；你真正需要自己保留的是“自主科研决策层”。**

---

## 如果现在重新设计 Ref2Dex，我不会再自己造完整 runtime

我会考虑：

```text
Ref2Dex Research Layer
│
├── MISSION
├── STATE
├── RESEARCH_QUEUE
├── Probe / Validation
└── root research policy
          │
          ▼
    thin adapter
          │
          ▼
backnotprop/orchestrator
or zetbrush/multiagents
          │
 ┌────────┼────────┐
 ▼        ▼        ▼
cm       rl       eval
```

其中我们自己保留的代码可能只剩：

```text
research_scheduler.py
research_policy
experiment contracts
role bindings
```

而这些已经有人解决的问题：

```text
Codex thread start/resume
turn steering
task store
message delivery
worker liveness
process supervision
provider adapter
```

全部交给下层项目。

### 我的优先调查顺序

**第一：`backnotprop/orchestrator`**

最符合：

> root 决策、外部 deterministic harness 执行。

**第二：`zetbrush/multiagents`**

最符合：

> Broker + persistent agents + CodexDriver + 自动重启。

**第三：`hyspacex/harness-cli`**

重点借：

> stable role → provider binding + harness owns durable state。

所以结论是：**有，而且我们现在自己写的 orchestration 层已经和几个现有项目明显重复。比较合理的下一步不是继续修当前 Broker/watchdog，而是先做一次 build-vs-adopt audit，看看能不能直接把 runtime 层替换成 `orchestrator` 或 `multiagents`，只保留 Ref2Dex 特有的科研 supervisor。**

[1]: https://github.com/zetbrush/multiagents?utm_source=chatgpt.com "GitHub - zetbrush/multiagents: Multi-agent orchestration for Claude Code, Codex CLI & Gemini CLI — spawn AI agent teams that communicate, review code, and coordinate via MCP · GitHub"
[2]: https://github.com/backnotprop/orchestrator/blob/main/doc/operator-guide.md?utm_source=chatgpt.com "orchestrator/doc/operator-guide.md at main · backnotprop/orchestrator · GitHub"
[3]: https://github.com/backnotprop/orchestrator/blob/main/doc/codex-app-server.md?utm_source=chatgpt.com "orchestrator/doc/codex-app-server.md at main · backnotprop/orchestrator · GitHub"
[4]: https://github.com/hyspacex/harness-cli?utm_source=chatgpt.com "GitHub - hyspacex/harness-cli: CLI harness for long-running app development with Claude Agent SDK or Codex App Server · GitHub"
[5]: https://github.com/hyspacex/harness-cli/blob/main/CLAUDE.md?utm_source=chatgpt.com "harness-cli/CLAUDE.md at main · hyspacex/harness-cli · GitHub"
[6]: https://github.com/formiat/multi-agent-orchestration?utm_source=chatgpt.com "GitHub - formiat/multi-agent-orchestration: LLM-driven (Codex-centric) orchestration workflow library for delegated software engineering agents · GitHub"
[7]: https://github.com/Gan-Xing/CodexProvider?utm_source=chatgpt.com "GitHub - Gan-Xing/CodexProvider · GitHub"
