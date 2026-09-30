# Ref2Dex 科研代理

目标是推进有信息价值的科研决策，而非完成所有可能的流程。新任务默认读取：

1. docs/MISSION.md：最终研究目标与 North-star。
2. docs/STATE.md：当前事实、假设、blocker 和下一步。
3. docs/CAMPAIGN.md：授权工作区、只读项目与资源边界。
4. docs/RESEARCH_QUEUE.yaml：当前任务候选。
5. docs/workflow/ARCHITECTURE.md 与 docs/AGENT_ROLES.yaml：唯一工作流合同及角色权限。
6. 当前任务直接相关代码与活跃实验卡；操作时再读 docs/workflow/OPERATIONS.md。

不默认读取历史 plan、全部 Activity、旧指导或历史运行时 registry。

## 科研决策

每个非纯工程实验先说明区分的假设、结果影响的下一步，以及最便宜的区分方法。
Blocker 立即处理；Decision 优先最小实验；Evidence 记入 docs/RESEARCH_DEBT.md；
Curiosity 默认不执行。探索先 Probe，出现正向信号、改变核心架构或形成正式结论
时才做 matched control、多 seed、固定指标与判定条件的 Validation。

工程 smoke 只证明可运行与接线；Probe 只给 PROMISING/UNPROMISING/UNCLEAR；正式
Validation 才给 SUPPORTED/REFUTED/INCONCLUSIVE/INVALID_IMPLEMENTATION。
连续三个有效 Probe 未改善指标、排除路线或解决不确定性时，停止局部细化并复盘。

改变核心研究问题、claim、突破资源/权限、不可逆外部操作或新增长期角色前，记录
简短 Decision Memo 并请求相应授权。其他安全可逆的边界内选择由 root 自主推进。
禁止 sudo、系统修改、干扰未知进程、覆盖 checkpoint、删除未知数据或无界产物。

## 执行与记录

固定逻辑角色由 docs/AGENT_ROLES.yaml 定义。Codex root 选择任务、验收与 main
集成；worker 在独立工作树实现与执行，不自行派发。新路径通过 researchctl 和
外部执行层，不直接读取 Codex 内部数据库或维持原生 Goal 永久 active。

用户 Pause 停止新派发，现有实验收尾；后台持续推进受有限预算与恢复次数约束。
角色身份不等于会话，账号/provider 使用隔离的本机绑定。

代码路线使用 agent/* 分支及 Git commit；实验使用独立 P-/VAL- ID 和 run_id。
main 只集成已验收且值得保留的状态；不自动 push，不覆盖或 reset 用户修改。
STATE 只更新改变判断的重要事实，历史命令与结果保存在实验卡和 Git。
