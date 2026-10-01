# Ref2Dex Autonomous Research Agent

本仓库采用“自主探索、少量人工决策、正式验证后再形成结论”的科研工作方式。

核心原则：

> Optimize for research progress, not procedural completeness.

AI 的目标不是把每个可能的问题都研究完整，而是在资源和时间有限的情况下，持续获得能够改变研究决策的信息，并向 `MISSION.md` 中定义的最终目标推进。

---

## 1. 默认工作模式：自主探索

除非触发本文件定义的 Decision Checkpoint，否则默认允许 AI 自主：

* 阅读代码与研究记录；
* 修改当前仓库中的代码、配置和测试；
* 创建和切换 `agent/*` 分支；
* 编写诊断脚本；
* 执行 smoke test；
* 运行小规模训练、评估和离线分析；
* 根据结果修复 bug、调整实现并继续尝试；
* 放弃明显失败的局部方案；
* 设计下一轮最小实验。

不需要因为普通研究变量、网络结构、reward、Cm 接法或实现细节变化而逐次请求用户批准。

研究探索本身允许失败。

默认行为应是：

> 在安全且可逆的情况下继续推进，而不是因为存在不确定性而停止。

---

## 2. 启动时只读取最小上下文

每次开始新任务时默认读取：

1. `AGENTS.md`
2. `docs/MISSION.md`
3. `docs/STATE.md`
4. `docs/CAMPAIGN.md`
5. `docs/AGENT_ROLES.yaml`
6. `docs/AGENT_BROKER.md`
7. `docs/AGENT_COORDINATION.md`
8. `docs/ROOT_AGENT.md`（root）或与当前 worker 角色对应的交接规范
9. 与当前问题直接相关的代码
10. 当前活跃实验记录（如果存在）

如果任务涉及代理身份、runtime binding、任务派发、工作树或 provider 迁移，还必须
读取 [`docs/AGENT_REGISTRY.json`](docs/AGENT_REGISTRY.json)、本机
`.runtime/AGENT_BINDINGS.json`（若存在）以及
[`.agents/skills/create-ref2dex-agent/SKILL.md`](.agents/skills/create-ref2dex-agent/SKILL.md)。
如果任务是 workflow 迁移或规范审计，仓库中的 `docs/ref.md` 也属于直接相关上下文；
它是设计输入，不替代本文件和当前状态。

文档分类入口见 [`docs/README.md`](docs/README.md)；workflow 和 research 的索引分别
见 [`docs/workflow/README.md`](docs/workflow/README.md) 与
[`docs/research/README.md`](docs/research/README.md)。这些索引不替代上面的最小上下文
清单，只用于定位当前入口和区分历史资料。

不要默认批量读取：

* 历史 plan；
* 历史指导；
* 全部 Activity；
* 全部 experiment；
* 旧 decision log；
* 无关 Git 历史。

只有以下情况才追溯历史：

* 当前证据存在冲突；
* 需要确认某个历史设计为什么存在；
* 怀疑已经重复做过同一实验；
* 正式验证需要固定历史 baseline；
* 准备修改一个长期 invariant。

历史文档是证据库，不是默认 prompt。

### 2.1 当前 workflow 的唯一入口

本文件是所有新对话的总规则入口；代理协作的具体实现必须与以下规范一起读取，不能
只读取其中一份后自行拼接旧流程：

* [`docs/AGENT_ROLES.yaml`](docs/AGENT_ROLES.yaml)：tracked 的固定角色、职责、权限和 branch；
* [`docs/AGENT_BROKER.md`](docs/AGENT_BROKER.md)：本地无模型 Broker、任务租约、runtime
  状态和四类消息；
* [`docs/AGENT_COORDINATION.md`](docs/AGENT_COORDINATION.md)：root、worker、handoff、
  liveness 和集成边界；
* [`docs/ROOT_AGENT.md`](docs/ROOT_AGENT.md)：`/root` 的决策、派发、验收和持续监督循环；
* [`docs/AGENT_POLLER.md`](docs/AGENT_POLLER.md)：worker event poller 与 watchdog 的运行合同；
* [`docs/AGENT_REGISTRY.json`](docs/AGENT_REGISTRY.json)：固定 pool 和历史 runtime 审计兼容信息；
* [create-ref2dex-agent skill](.agents/skills/create-ref2dex-agent/SKILL.md)：只有 runtime
  rebind 或经用户批准的新长期角色才使用的身份与启动流程。

当前主路径是：root 通过 `scripts/agent_broker.py` 向固定 `agent_key` 写入
`TASK_DISPATCH`；worker 只能写 `TASK_UPDATE` 或 `TASK_HANDOFF`；watchdog 和本地控制
只写 `CONTROL`。Broker 只管理队列、租约、provider adapter 和运行时状态，不做研究
 决策，也不创建动态 subagent。worker 不能直接互相通信，跨角色依赖必须退回 root。
运行时 binding 属于本机 `.runtime/AGENT_BINDINGS.json`，任务队列和状态属于本机
 `.runtime/tasks.sqlite` 与 `.runtime/AGENT_STATE.sqlite`，不能把它们当作研究证据提交。
只有 supervisor desired state 为 `RUNNING` 时才允许新 dispatch/claim；已持有租约的
任务可以在暂停后收尾。真实 provider 投递必须经过显式
`scripts/agent_runtime_adapter.py` launcher；仅写入 SQLite 不代表 worker 已启动。
派发前可运行 `python3 scripts/workflow_doctor.py` 检查角色、binding、数据库和 lease。

---

## 3. 每个实验必须服务于一个决策

在运行一个非纯工程 smoke 的实验前，AI 必须回答：

1. 这个实验想区分什么假设？
2. 实验结果会改变哪个下一步决策？
3. 获取该信息的最便宜方法是什么？

然后将实验分类为：

### Blocker

不解决就无法继续主路线。

处理方式：立即解决。

### Decision

结果 A 与结果 B 会导致不同的下一步路线。

处理方式：优先执行最小可判别实验。

### Evidence

无论结果如何，当前近期路线都不会改变，但未来论文、复现或完整论证需要。

处理方式：写入 `docs/RESEARCH_DEBT.md`，当前不做。

### Curiosity

只是增加理解，并不会明显影响路线或最终 claim。

处理方式：默认不做。

“实验科学上有意义”不是运行它的充分条件。

---

## 4. Probe 优先，Validation 延后

探索阶段默认运行 Probe。

Probe 的目的不是证明结论，而是判断：

> 这个方向是否值得继续投入？

Probe 应尽量满足：

* 单 seed 或极少 seed；
* 少量环境；
* 小数据；
* 短训练；
* 小规模离线分析；
* 尽可能低的 GPU 和时间成本。

只有满足以下条件之一时才进入 Validation：

* Probe 出现明显正向信号；
* 即将基于某结果改变核心架构；
* 即将形成正式科研结论；
* 结果需要进入论文或对外汇报；
* 两条路线需要正式比较后才能选定。

Validation 才要求 matched control、多 seed、固定 metric、预先定义的判定条件和完整证据。

不要把 Validation 的严谨程度施加到每一个探索想法上。

---

## 5. Decision Checkpoint

以下情况必须在继续前记录简短 Decision Memo，并重新核对证据、资源和安全边界：

1. 准备改变 `MISSION.md` 中的核心研究问题；
2. 准备放弃一个核心研究假设；
3. 准备改变最终需要证明的 claim；
4. 两条路线都合理，而任一路线后续成本都明显较高；
5. 需要突破 `CAMPAIGN.md` 中的 GPU、时间、磁盘或权限限制；
6. 操作不可逆或可能影响当前仓库之外的数据、checkpoint、进程或环境；
7. 连续三个有效 Probe 都没有使 North-star scoreboard 获得进展，也没有排除重要路线；
8. 准备把探索性观察升级为正式科学结论。

Decision Memo 必须简短，只包含：

* 当前需要决定的问题；
* 当前最关键证据；
* root 选择的行动及其理由；
* 预计成本、成功/失败后的下一步和停止条件；
* 仍需外部授权的边界（如果存在）。

在 `MISSION.md`、`CAMPAIGN.md` 和当前用户授权允许的范围内，root 自主选择并执行
记录过的行动，不要求用户在预设的 Option A/Option B 之间插入选择。只有改变核心研究
问题或 claim、突破资源/权限边界、执行不可逆外部操作，或需要新的长期身份时，才暂停
并向用户请求相应授权。

不要提交长篇计划让用户审批。

---

## 6. 三次无进展规则

如果连续三个有效 Probe：

* 没有提高核心目标指标；
* 没有排除重要路线；
* 没有解决核心不确定性；

则禁止继续向同一局部问题无限细化。

必须重新检查：

* 当前实验是否仍服务于 `MISSION.md`；
* 是否已经陷入局部诊断；
* 是否应该换接法、换 representation、换优化目标或返回更高层问题。

此时触发一次路线复盘；必要时形成 Decision Memo。

---

## 7. Git 规则

`main` 保存已经值得长期保留的状态。

新研究路线默认使用：

`agent/<short-description>`

例如：

`agent/cmv2-online-adapt`

`agent/cm-action-ranking`

分支代表“实现路线”，不是单个超参数实验。

不要因为：

* seed 改变；
* reward coefficient 改变；
* epoch 改变；
* 一次评估；

而新建分支。

同一实现路线的 Probe 可以在同一分支完成。

当一条路线产生值得保留的代码、工具或证据后再合并回 `main`。

除非用户明确要求，否则不要向远程仓库 push。

不要 reset、覆盖或删除用户已有修改。

---

## 8. 身份与版本

代码版本、实验和研究阶段必须使用不同的身份，不要用同一个连续版本号同时表示三者。

### 8.1 Code Identity

具体实现由 Git 管理。

使用：

* branch：表示一条实现或研究路线；
* git commit：表示具体代码状态。

例如：

`agent/cm-action-ranking`

以及：

`git commit: <sha>`

不要为了记录一次实验而人为创建新的代码版本号。

---

### 8.2 Experiment Identity

实验使用独立的 experiment ID。

探索性 Probe：

`P-<YYYYMMDD>-<short-topic>`

例如：

`P-20260923-cm-action-ranking`

正式 Validation：

`VAL-<YYYYMMDD>-<short-topic>`

例如：

`VAL-20260924-cm-ranking-online-ab`

不同 seed、repeat、checkpoint 或运行实例通常不创建新的 experiment ID，而由 experiment 内部的 `run_id` 区分。

因此：

* `experiment_id` 表示“研究问题/实验设计”；
* `run_id` 表示“该实验的一次具体执行”。


---

## 9. 科研结论边界

工程 smoke 只能证明：

* 代码能运行；
* tensor 有限；
* checkpoint 能加载；
* wiring 正确。

Probe 只能形成：

* `PROMISING`
* `UNPROMISING`
* `UNCLEAR`

正式 Validation 才允许形成：

* `SUPPORTED`
* `REFUTED`
* `INCONCLUSIVE`
* `INVALID_IMPLEMENTATION`

不要把：

* 单 seed；
* 单 rollout；
* 训练 loss；
* 单次成功视频；
* 工程 smoke；

描述成正式科学结论。

---

## 10. 保护边界

必须始终遵守 `docs/CAMPAIGN.md` 中的机器和资源约束。

默认禁止：

* sudo；
* 系统级修改；
* 杀死未知或他人的进程；
* 修改授权工作区之外的项目；
* 覆盖已有 checkpoint；
* 删除无法确认归属的数据；
* 无界生成 cache、视频和 checkpoint。

外部数据和代码如果被标记为 read-only，只能读取或软链接。

---

## 11. 当前状态记录

`docs/STATE.md` 是项目当前唯一默认事实入口。

它只保留：

* 当前 North-star 状态；
* 已经确认的重要事实；
* 当前活跃假设；
* 当前 blocker；
* 当前最值得做的下一步。

不要把每一次命令、每个失败 traceback 或所有历史实验追加进去。

历史实验保存在 experiment card 和 Git 中。

当新证据改变研究判断时，更新 `STATE.md`。

---

## 12. Research Debt

对于：

* 未来论文需要；
* 当前不改变决策；
* 但最终不能永久遗漏；

的实验，写入：

`docs/RESEARCH_DEBT.md`

Research Debt 的存在意味着：

> 这个实验没有被忘记，只是当前不值得占用探索预算。

不要为了“严谨”而立即偿还所有 Research Debt。

---

## 13. 多代理协作

多代理身份、固定角色、Broker、工作树、交接和 main 集成规则见
[docs/AGENT_ROLES.yaml](docs/AGENT_ROLES.yaml)、
[docs/AGENT_BROKER.md](docs/AGENT_BROKER.md)、
[docs/AGENT_COORDINATION.md](docs/AGENT_COORDINATION.md) 与
[docs/AGENT_REGISTRY.json](docs/AGENT_REGISTRY.json)。
主代理的持续推进与轮询规则见 [docs/ROOT_AGENT.md](docs/ROOT_AGENT.md)。
worker 事件与 root liveness 规则见 [docs/AGENT_POLLER.md](docs/AGENT_POLLER.md)。
新建或迁移 runtime 时使用 [create-ref2dex-agent skill](.agents/skills/create-ref2dex-agent/SKILL.md)；
普通任务派发不调用该 skill 创建代理，而是由 root 通过 Agent Broker 派发。
其中的资源授权必须同时遵守本文件和 `docs/CAMPAIGN.md`。

本文件的自主探索权限由获派任务的执行代理在其独立工作树行使。`/root` 负责
选题、派发、监督、验收和 `main` 集成；具体分析、实现、preflight、实验及
长任务进程归属交给相应执行代理，详见上述主代理规范。

现有 Cm 工作必须派发给固定角色 `agent_cm`；如果该角色当前 binding 仍指向历史的
`agent_cm_temporal` conversation，则沿用该 binding 的冻结、CPU-only 和其他资源权限，
但不得把历史 conversation 当成新的组织身份，也不得由 `/root` 临时兼任执行 owner。
需要新增身份或能力时，按 `create-ref2dex-agent` skill 建立并登记代理后再派发。
