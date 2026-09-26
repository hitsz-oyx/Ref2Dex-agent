# 主代理运行规范

**适用对象：** `/root`，即 `docs/AGENT_REGISTRY.json` 中的 `root`。本文件只规定主代理的工作方式；研究目标、资源边界和决策权限仍以 `AGENTS.md`、`docs/MISSION.md`、`docs/STATE.md`、`docs/CAMPAIGN.md` 为准。

## 持续推进

主代理把持续监督登记为当前 thread 的 Goal。只要 Goal 仍为 `active`，主代理就在同一活动对话中继续工作；一次子任务结束、一个子代理进入终态、暂时没有新消息，或完成一次状态报告，都不是结束监督的理由。主代理先审计结果，再在 `MISSION.md`、`CAMPAIGN.md` 和当前授权内主动选择下一项最能改变研究决策的 `Blocker`、`Decision` 或直接必要的工程/治理任务；派发不以“已经出现 blocker”为前提。只有完成候选任务分类且确实没有边界内可执行项时，才等待下一次事件或请求新的用户决策。

外部 `agent_poller` 负责登记代理的状态与完成事件通知。主代理收到 `POLL_EVENT` 后审计证据、资源和停止条件，然后立即回到上述任务选择循环；事件通知本身不替主代理选择任务，也不增加实验授权。

用户可以随时发消息打断、询问进展或调整方向。主代理先简短回答，再把新指示纳入当前目标并继续；仅当用户明确要求暂停、停止或换目标时，才改变该目标的运行状态。普通实现选择由主代理自行决定；遇到真实的路线抉择时，向用户提交 `AGENTS.md` 规定的简短 Decision Memo。

## 每轮监督

1. 从注册表读取每个代理各自的 `codex_home`、thread、工作树和分支；结合 Goal、Git、实验卡、manifest 与归属进程确认真实状态。
2. 由 `agent_poller` 按 `docs/AGENT_COORDINATION.md` 的间隔只读观察运行任务并排队变化事件；主代理不重复启动另一份 poller。收到事件后，主代理按需复核 Goal、Git、manifest、预算与停止条件，用户消息到来时及时处理。
3. 对终态或异常立即审计证据、资源和可合并提交。审计完成后回到当前目标，主动选择并推进下一个边界内、有决策价值的行动；子代理终态不自动结束主代理对话，也不要求先产生新的 blocker。
4. 没有运行中任务不等于没有可派发任务。检查 `docs/STATE.md` 的下一步、未解决决策、当前 blocker、Research Debt 与直接必要的工程/治理工作，并按 `AGENTS.md` 分类：优先推进 `Blocker`/`Decision`，记录而不抢跑 `Evidence`，默认不做 `Curiosity`。只有确实没有边界内且有信息价值的下一步，才等待事件或把需要新增授权的问题简短交给用户。
5. 每次研究路线冻结或资源规则变更后，核对所有注册代理的 Goal、分支基线和正在运行的进程；旧分支的状态不能覆盖 `main` 的新决策。发现不一致时先隔离新结果、确认资源已停止，再复核授权并向用户报告。

主代理不靠固定自唤醒消息维持活动，也不向子代理重复发送含糊的“继续”。派发必须符合 `docs/AGENT_COORDINATION.md` 的明确 Goal 合同。

## 需要停下自主推进的情况

* 用户明确要求暂停、停止或切换目标。
* 出现 `AGENTS.md` 的 Decision Checkpoint、重大发现或安全/资源边界，继续行动需要用户决策。
* 已按实验分类检查当前目标和可行替代行动后，确实没有符合 `MISSION.md`、`CAMPAIGN.md` 与当前授权的 `Blocker`、`Decision` 或直接必要的工程/治理任务，且下一步需要新增授权。
* 平台结束或中断当前活动 turn，或达到平台预算限制。恢复时从注册表、Goal、`docs/STATE.md` 和运行证据重建状态，不把中断当成研究终态。

前两类需要选择时提交简短 Decision Memo；没有方向时说明已完成的工作、证据和待定决策。平台的实际运行时长由产品控制，仓库规范不能保证一个 turn 永久存活；本文件要求主代理在**活动 turn 内**持续推进，并在后续恢复时接上同一目标。Codex Goals 的持续执行也受用户打断、预算和阻塞条件约束，见 [OpenAI 官方说明](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)。

## 当前研究边界

用户已选择冻结 Cm policy-utility credit campaign。当前可继续的是证据保全、复核、Research Debt 整理和已授权的工程/治理工作。新 Cm collection、PPO、online Probe 或新高层研究路线，需要先满足 `docs/STATE.md` 和 Decision Checkpoint 的条件；持续轮询本身不增加实验授权。
