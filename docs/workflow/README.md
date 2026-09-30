# Ref2Dex 科研工作流

这是科研角色唯一日常工作流说明。普通用户会话默认独立，直接执行用户任务；只有显式
加入的 root/worker 接受本合同。历史设计、迁移资料和运行 registry 不属于启动必读。

## 角色与自主推进

Codex root 负责选题、派发、证据验收、main 集成和 MISSION 完成判断。cm、rl、eval、infra
按需启动，职责与资源权限只定义在 [AGENT_ROLES.yaml](../AGENT_ROLES.yaml)。账号、provider、
CODEX_HOME、工作树与执行 store 留在本机配置。worker 不互相派发或改变组织权限。

研究循环：用户目标 → root 选择最小有价值任务 → worker 执行 → 验收证据 → 更新判断并继续。
前台通过持久指令入口联系后台，只有一个派发 owner；关闭前台后继续。默认无总运行时长、
总任务数或总 root 回合截止。明确设置的专项上限仍有效；单任务必须有 timeout，恢复次数有限。

root 自主处理失败、改变局部路线、分配已授权资源，不主动联系用户或请求路线选择。
超出授权的动作不执行，记录阻碍并选择其他工作；用户查询时解释当前结果与重要决定。
Pause 始终有效：停止新 root/worker 派发，既有实验收尾。用户明确 interrupt 时才中断指定
且可确认归属的实验。后台不会自动解除用户的暂停。

独立会话不归 root 管理。科研任务使用各自工作树；派发前检查工作树根与未提交修改，
有独立/未提交工作时拒绝进入，不能清理、覆盖、自动合并或停止其他会话进程。用户未
明确交付的成果不进入科研记录。此检查是合作边界，不是对所有文件访问的系统级沙箱。

## 科研判断与记录

每个实验先说明区分的假设、影响的决策和最便宜的区分方法。Blocker 立即处理；Decision
优先最小实验；Evidence 记入 [RESEARCH_DEBT](../RESEARCH_DEBT.md)；Curiosity 默认不做。
探索先 Probe，正式比较、改变核心架构或形成正式结论时做 Validation。

工程 smoke 只证明运行和接线；Probe 只给 PROMISING/UNPROMISING/UNCLEAR；正式 Validation
才给 SUPPORTED/REFUTED/INCONCLUSIVE/INVALID_IMPLEMENTATION。三次有效 Probe 无信息增益
时先自主复盘，寻找更高层替代路线。确实没有已授权、可判别的方向时 blocked 等待，不重复
实验制造进展，不将受阻标为完成。新证据/指令可触发复盘；没有变化不反复调用模型。

一个决策问题对应一个实验；同一设计的 seed、对照臂和技术重试使用 run_id。工程修复不
单独创建科研 Probe。实验卡给短摘要，manifest 保存命令、配置、哈希与执行版本。设计或
判定条件改变时明确记录新实验/修订，不能隐藏失败或事后筛选。[实验索引](../experiments/INDEX.md)
按路线生成，原始卡片仍是证据入口。STATE 只记改变当前判断的结果；handoff 给链接。

自主决策记录只写主要路线变化、重要假设处置、较大资源重分配、影响路线的冲突与阻碍。
每项说明决定、证据、理由、成本/停止条件、结果；不写普通命令、修复、重试或心跳。
本机 state_dir/AUTONOMOUS_DECISIONS.md 是可重建的阅读视图，原记录持久保存，换目标后旧记录归档到 decision-history。需要提交
研究历史时通过 decisions --export 写出新文件，再由 root 验收；不把控制数据库当实验数据。

## 安装与本机配置

需要 graspenv/Python 3.8+、PyYAML 和 Node 24+。执行层固定为 Orchestrator CLI 0.1.0：

```bash
npm ci --prefix tools/workflow_runtime
```

npm 必须来自 Node 24 的 PATH；不修改系统 Node 或全局包。参考
[workflow.example.json](../../configs/workflow.example.json)，填写绝对路径并存为本机
.runtime/workflow.json。command 是 Node 与 CLI 的 argv 数组，env.PATH 明确包含 Node/Codex。
各逻辑角色工作树必须不同；同一角色的 provider 备选保留工作树，使用不同 CODEX_HOME/store。

不同 provider 使用独立 Codex 配置。fallbacks 是按优先级排列的完整绑定列表，每项必须
verified:true，表示已经完成真实账号身份与能力验收。额度耗尽或 provider 不可用时，入口
记录原因并选列表中的可用备选。任务 provider 字段若明确指定绑定，就不得跨 provider。
账号额度耗尽仍可按已验收备选切换。用户后续确认的恢复策略优先于旧连接故障切换：

- 明确 `Selected model is at capacity` 且执行已经失败结束：等待 60 秒后续接原任务、原 provider/模型；重复容量故障再等待 60 秒，不设默认次数截止。
- 确认连接/网关故障：初始失败后最多 3 次恢复尝试；成功清零，耗尽后停止该任务恢复，不自动切 provider，其他角色继续。
- 用户暂停/完成/明确预算截止不恢复；运行中、取消、超时或状态未知不按容量/网关错误重启。
- 仅后台唯一 owner 执行恢复。恢复时刻与次数持久保存；失败交付在恢复等待中不能提前验收或驳回。
- 后端 `resume` 使用原 provider 会话，生成新执行 ID，逻辑任务 ID、工作树、绑定和历史执行链保留。会话尚未建立时携原任务与保存上下文新建执行，先检查已有产物/进程，避免重复实验。
- 原生 Goal 已暂停、完成或达到预算时保留其停止状态；不靠发送 `/goal resume` 文本改变状态。
- 恢复投递前持久保存唯一名称；响应丢失先核对，无法确认则等待，不重复恢复。恢复同样占用明确派发/root 次数预算，不重置已消耗额度。

未知执行不推断失败。
无备选时该角色等待，其他角色继续；已恢复可用的绑定由 provider-ready 通知，不盲目重试。
未知投递先 reconcile，不能换账号重复启动。其他原生模型 harness 的账号隔离未验收，当前
拒绝启用；自定义工程 process 必须 engineering_only:true，不能接 Probe/Validation。

max_gpu 与 allowed_workspace_roots 必须明确，配置不得突破 CAMPAIGN 授权。总时长/派发数/
root 回合限制仅在明确专项授权需要时填写，默认样例不含这些截止。恢复不重置消耗。
worker 负责检查现有 GPU 进程、磁盘与任务产物成本，不抢占未知资源、不覆盖 checkpoint。
本地控制只验证可测合同；模型执行不是系统级权限隔离，科研真实性仍需独立证据审查。

## 日常命令

以下 python 指 graspenv 的 Python；可显式使用 /home2/wyy/miniconda3/envs/graspenv/bin/python。

```bash
python scripts/researchctl.py --config /absolute/workflow.json instruct --request-id U-001 --text '沿当前 MISSION 继续，先解决数据 blocker'
python scripts/researchctl.py supervisor resume --legacy-dispatch-disabled
python scripts/researchctl.py supervisor run --detach
python scripts/researchctl.py supervisor status
python scripts/researchctl.py supervisor pause
python scripts/researchctl.py supervisor stop
python scripts/researchctl.py instruct --request-id U-002 --text '新的研究目标和范围' --new-goal
python scripts/researchctl.py provider-ready --role agent_cm --provider configured-account-name
python scripts/researchctl.py reconcile
python scripts/researchctl.py decisions --export /absolute/new-decision-record.md
python tools/experiment_index.py
```

前台只有明确收到用户科研指令才调用 instruct；普通独立任务不写入口。request-id 用于
幂等提交；status 显示 instruction_version、processed_version、指令与决策记录位置。
暂停、恢复和新指令使旧推进决定失效；恢复后重新判断。new-goal 更新目标版本，旧证据
不能直接完成新目标；完成后的新目标从 paused 开始，需要明确 resume。

后台持有 owner 锁时，其他前台 dispatch/accept/tick 被拒绝，status/pause/instruct 保持可用。
stop 只暂停派发并退出本工作流的后台 owner，已有 worker 不被 kill；resume 后重新 run --detach。
手动有界 smoke 可用 [task.example.json](../../configs/task.example.json) 通过 dispatch 提交。
accept 需要终态、验收理由，失败执行只能拒绝。任务契约与执行 ID 分离，不能把排队当启动。

日志在 state_dir/supervisor.log。attention 是有限故障恢复耗尽或未知投递；budget_limited
是显式专项预算耗尽；blocked 是科研方向受阻；completed 是正式目标验收。修改状态前先读
证据，未知投递不可换 task_id 重发。后台停止后需要重新 run --detach；不会自动扩张权限。

## 最终完成的证据合同

完成必须对应 [MISSION](../MISSION.md) 的自训练抓取与 Cm 策略因果增益，不能用局部成功
替代。root 提交 complete 时提供 reason、evidence_task_ids。所有 worker 交付须已处理；
证据必须是本目标版本已接受的 Validation，并包含执行来源及独立 agent_eval 审查。

agent_eval 的输出必须是 JSON，schema=ref2dex.validation-evidence.v1，含 goal_version、
reviewed_task_ids 和 checks。checks 必须包含 self_trained_grasp 与 cm_policy_utility；每项
提供 verdict:SUPPORTED、validation_id、card、scope、pre_registered:true、至少两个不同
整数 seeds。抓取项还需 self_trained:true；Cm 项需 matched_control:true 和 arms:[Cm-on,Cm-off]。
card 指向 root 工作树 docs/experiments/validations 中的已集成正式验证卡。执行来源任务必须
被独立审查覆盖。root 检查实际方法、预注册门槛、指标与证据后验收；JSON 字段不证明结论。

完成后保存摘要并停止新派发，不自动 push/发布。预算耗尽或路线受阻仍显示未完成。

## 迁移与验收

旧任务由旧系统收尾，新系统只接新任务。--legacy-dispatch-disabled 表示操作者已经关闭
旧的未来派发/恢复 owner；命令不会代你停止未知进程。旧命令通过 --legacy 保留兼容。
账号身份、有效配置与 store 绑定封印；换账号/配置用新 store，不接管未知历史执行。

先运行 [工程 smoke](../../tools/workflow_smoke.py)，再验证真实多账号/provider、root JSON
回合和后台续接，之后才切换活跃系统。当前 npm audit 的 6 项传递依赖公告未修复；普通
overrides 对 bundled 依赖未生效。上线前核对上游修复或安全重打包，锁定版本不是安全验收。
工程验证与实际限制见 [VALIDATION.md](VALIDATION.md)。设计/实施记录按需查 docs/superpowers。

重要决定可以附带稳定 `decision_id`，后续结果变化时沿用该 ID 更新同条记录。
决策文档显示当前目标范围和启停状态；完成时保留归档，普通启停命令不追加流水账。
