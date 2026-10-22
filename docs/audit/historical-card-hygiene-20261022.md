# 历史卡片卫生报告 · 2026-10-22

范围：17 张目标 Task-local Probe 卡、`docs/STATE.md` 与本报告。仅补注记/metadata；
没有改代码、重算历史实验、改 gate、状态或历史数值，没有启动训练/评估。
本轮按机主已核实事实修订引用边界，历史 CPU 审计已有注记继续保留；顶部新增注记明确处置。

## 已核实的共同事实及适用范围

1. 官方 PPO 为 **v4**：53,347 结束训练回合/0 完整成功不是 v6 或 τ 无效证据；
   是同步单模型的训练计数，不是独立评估。[逐卡原始审计](../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
2. 同501条真实轨迹，经任务语义修复完整评分 **0→77/501**，安全prefix **0→9/501**。
   旧全零含评分假象；77/501是同档案重新评分，不是新policy成功率，也不能因缺轨迹
   反推旧PPO的修后成功数。完整声明恢复490RELEASE/11HOLD；旧64UNKNOWN与437准入
   仅保留为旧协议史。[修复复核](../../src/task/ptv3-shared-wam-v6/docs/audit/fix-review-20261022.md)。
3. 旧 **64/501 UNKNOWN** 在旧准入下无成功分支；不能认为其余任务全可成功，也不能
   认为UNKNOWN解释全部零。[危害量化](../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-harm-20261022.md)。
4. voxel表示边界曾逐tick终止；1570快照27,792/53,174=**52.27%**，最终
   27,869/53,347=**52.24%**。不是机械安全边界，也不等于浪费同等比例交互/时间。
5. 三rank共 **4,032条曲线缺reward/return/phase**，不支持回报趋势、收敛或精确阶段归因；
   reward实际进入PPO，缺日志不等于无reward。[缺项及分母](../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-fixes-20261022.md)。

后四项按对应执行链限定，不能未经证据给其他早期实验统一归因。
工程检查通过、局部负现象和部分正信号均保留；本轮不把历史负证据撤销解释成方法已成功。
P1继续HOLD；正式v6/τ收益与修后执行资格仍需独立证据。

## 逐卡结论处置

“撤销”撤销错误的方法归因/无效运行的证据资格；“降级”限为描述、固定配方或代理指标；
“加注”保留原局部结论并明确边界。卡头历史status不改，当前引用以新增有效性边界为准。
所有17张卡均明确不得再作为“v6/τ无效”的证据。

| 卡片 | 结论处置 | 能支持 | 不能支持 |
| --- | --- | --- | --- |
| [P-20261011-goal-conditioned-bc](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-bc.md) | 加注 | 单 motion、环境留出下动作蒸馏 L1 0.00672288 通过离线门槛；动作标签来自自有 tracker，工程 packet 仅提供特权几何 goal。 | 不能支持闭环抓持、独立序列泛化或 v6/τ 有效或无效；13,008/4,336 行不是独立任务，低 L1 不认证低层执行资格。不能把 goal-only packet 的 training_allowed=false 误传成所有动作 label 非法。 |
| [P-20261011-goal-conditioned-execution](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-execution.md) | 降级 | 有效 16×542 完整执行中 hold45/long-held 均 0/16，有限命令和所验接线可复算；long-held 是 433 帧且终点 held 的专用门槛。 | 不能支持 v6/τ 或 goal-policy 一般无效；tracker 校准来自另一波，非本 wave 同初态对照。audit PASS 只覆盖所验接线，不能认证 covariate shift/hidden state 为唯一失败原因。 |
| [P-20261011-goal-conditioned-finger-residual](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-finger-residual.md) | 加注 | 有效 r2 hold45 7/16、long-held 0/16、clipping 13.5148%，说明该 finger-only 分解未恢复组合门槛。 | 不能支持完全没有持握、v6/τ 无效或所有 wrist/finger 分解无效。r1 resource guard 中断无轨迹不能作负例；仍是单 motion/seed 和特权 nominal。 |
| [P-20261011-goal-conditioned-ppo](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-ppo.md) | 加注 | 实际 24 次有限 PPO 更新、49,152 训练交互和 actor 改变，可作为训练工程证据；原 UNCLEAR 保留。 | 不能支持冻结评估成功率、PPO 收敛或 v6/τ 无效。训练 reward 和少量 episode held 不替代独立评估，后续 r2 失败也仅约束单 motion 短预算策略。 |
| [P-20261011-goal-conditioned-ppo-execution](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-ppo-execution.md) | 降级 | 可审计 r2 的 16×542 无干预执行 hold45/long-held 均 0/16，接线检查覆盖项通过；r1 角色标记有误，不能替代 r2 身份。 | 不能支持 v6/τ 或 PPO 普遍无效，也不能把本波接线 PASS 当完整物理资格或唯一机制诊断；单 motion/seed 与固定预算限制外推。 |
| [P-20261011-goal-conditioned-residual-integration](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-residual-integration.md) | 加注 | 有效 r4 long-held 10/16、hold45 15/16、clipping 6.6075% 是真实混合结果；组合门槛未过，placement 另列。 | 不能支持完全不会抓持、v6/τ 或残差控制无效。nominal q/velocity 来自特权 geometry cache，非可部署无 future-q planner；r1–r3 setup 失败不算物理负例。 |
| [P-20261011-goal-conditioned-residual-offset](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-residual-offset.md) | 加注 | 固定 offset 下 clipping 0.8187% 通过半门槛，hold45 12/16，而 long-held 0/16，完整组合门槛未过。 | 不能支持完全不会抓持、完整资格已通过或 v6/τ 无效；特权 nominal 参考下单 motion 的 offset 迁移不代表所有 residual 参数化。 |
| [P-20261013-wam-residual-ppo](../../src/task/trajectory-policy/docs/experiments/probes/P-20261013-wam-residual-ppo.md) | 撤销旧受污染归因；保留修后局部结果 | 修后 r7（a2f85d4b）20 更新、frame0 四环境 held0 是局部短预算结果；phase62 已持握 reset 主导训练 held，工程 bootstrap/reset 审计只支持所验合同。 | 不能支持 v6/τ 或残差 PPO 无效；r2–r5 受 bootstrap、reset H4/previous、lift 基准和 optimizer/KL 回退问题污染，r6 更新前失败。r5→r7 也改变 reset 来源，不能当单变量因果改善，训练已有 held 不等于自主起抓。 |
| [P-20261016-wam-coordination-bonus](../../src/task/wam-rl-policy/docs/experiments/probes/P-20261016-wam-coordination-bonus.md) | 加注 | 同初态 DEV 的 BC/none/geometry/WAM 保持代理 28/26/23/25，当前 bonus 配方未过 +5 且不退 BC 的 gate；弱 WAM 特异 bonus 诊断仍有效。 | 不能支持 v6/τ 或 WAM 一般无效、BC-free 成功或纯 reward 因果结论。实际为 Shadow 旧 v3-step80000 WAM 副本加冻结 BC；保持是 goal-position/lift 代理，非实测接触/force 资格。单 seed，参考/观测/bonus 同变。 |
| [P-20261016-wam-learned-short-goal](../../src/task/wam-rl-policy/docs/experiments/probes/P-20261016-wam-learned-short-goal.md) | 加注 | 旧混合行为终点回归的 held-object 离线 gate 未过，且部分 hand/scene 预测误差改善；448 held 状态来自 8 物体且窗口相关。 | 不能支持自主闭环失败、目标可达性/最优性、v6/τ 无效或 RL 样本效率无增益；本卡没有新闭环/PPO 评估，旧 WAM 也不读取动作。 |
| [P-20261017-direct-tau-executor](../../src/task/trajectory-policy/docs/experiments/probes/P-20261017-direct-tau-executor.md) | 撤销方法否决；降级为故障观测 | 旧固定实现的 learner 0/8 hold45 与 teacher 8/8、同序列诊断及离线拟合数值，仍可作为历史实现/分布故障观测。 | 不能支持独立参考 τ、v6 或交互信息 C 无效：旧实现混用 reference future 与 live-current 手，mean-pool 丢失时间顺序，训练是单 motion 多环境复制；10D oracle-C 不代表一般 C。本波 teacher placement 4/8 或 5/8 也未过完整校准。 |
| [P-20261017-gt-tau-new-sequence](../../src/task/trajectory-policy/docs/experiments/probes/P-20261017-gt-tau-new-sequence.md) | 降级 | 仅坐标修正后的 r4：teacher/source-PD hold45 各 8/8，固定 GT-τ→R 为 0/8，说明该冻结 R 在这一序列未通过 hold 资格。 | 不能支持 learned executor、v6/τ 或 PPO 无效；source-PD placement 5/8 未过完整校准。r1 runner 长度/reset 与 r2 partial_filtered 错坐标不能算方法结果，特权 GT 也未检验 learned WAM 预测。 |
| [P-20261018-absolute-plan-executor](../../src/task/trajectory-policy/docs/experiments/probes/P-20261018-absolute-plan-executor.md) | 降级 | 绝对参考锚点/控制空间损失下的离线误差、query 分布和 normalized Huber 筛选未过；teacher warm-up 改变了收集分布。 | 不能支持自主抓持成功、v6/τ 无效或绝对锚点无用。本卡没有最终自主物理评估；120 控制 teacher warm-up 加 beta=.25 的 8/8 hold 含实际约 46.11% 教师干预。原生单位误差没有独立预注册物理门槛，不能把离线筛选叫物理资格。 |
| [P-20261018-direct-tau-dagger](../../src/task/trajectory-policy/docs/experiments/probes/P-20261018-direct-tau-dagger.md) | 加注 | 一轮 learner-state 当前动作/PD 查询可复算，离线 action L1 改善；无教师干预的本序列 learner hold45 为 0/8，teacher hold45 为 8/8、placement 为 7/8，满足本波旧校准。 | 不能支持 v6/τ 无效或 DAgger 普遍无效。查询是当前状态动作/PD 标签，没有生成离轨状态的新未来 τ；同 motion 环境拆分不能认证任务泛化，producer r1 保存失败不计方法结果。 |
| [P-20261018-direct-tau-dagger-pd](../../src/task/trajectory-policy/docs/experiments/probes/P-20261018-direct-tau-dagger-pd.md) | 降级 | 复用 query 下 PD loss 降低离线 PD 误差，无干预 learner hold45 仍为 0/8；99.30% catastrophic query 的分布诊断有效。 | 不能支持 v6/τ 无效或 PD 监督无用；teacher placement 4/8 未达到完整任务校准，终点误差 0.0402 m 不能替代抓持成功。修 loss 与修恢复分布是不同问题。 |
| [P-20261018-direct-tau-executor-r3](../../src/task/trajectory-policy/docs/experiments/probes/P-20261018-direct-tau-executor-r3.md) | 降级 | 修正 reference-frame τ 和时间顺序后的固定 BC，在单条 unseen s1 上 learner hold45 为 0/8、teacher 为 8/8；40 序列 BC 的离线误差和已验接口一致性仍有效。 | 不能支持 v6/τ 无效、缺 C 是唯一原因或完整 placement 正控制已通过；本波 teacher placement 5/8 未达 runner 的 6/8 校准条件。八个同 seed 环境不是多 seed 验证，接线 PASS 不认证状态/动作覆盖。 |
| [P-20261020-inspire-measured-official-ppo](../../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-inspire-measured-official-ppo.md) | 降级；撤销 v6/τ 方法否决 | 仅能支持冻结 v4 桥的这次训练累计结束 53,347 回合、旧完整任务计数为 0，以及当时的终止/接线观测。三个同步 rank 是同一模型，回合数不是独立评估样本数。 | 不能支持 v6 PPO 失败、τ 无效、抓取/抬升率为零或策略已收敛。旧 64/501 UNKNOWN 在旧准入下结构性没有成功分支；同 501 条真实轨迹完整评分经任务语义修复为 0→77/501，安全 prefix 为 0→9/501，说明旧全零含评分假象；这不是该 PPO 的重评分或新策略成功率。voxel 表示边界曾充当每 tick 终止条件，iteration1570 为 27,792/53,174=52.27%，最终为 27,869/53,347=52.24%，不能解释成机械安全越界或按比例推算浪费步数。4,032 条曲线缺 reward/return/phase，不能从 loss 重建回报、阶段或收敛，也不能说 PPO 没收到 reward。 |

## 6张metadata格式修复

保留`ref2dex.probe.v2`，仅补原来缺少的字段。claim_id=C3沿用相邻GT-τ执行资格卡，
family沿用`HF-trajectory-policy-executor-transition`；已有计划资格卡索引1不变。
以下2–7只是档案编号，不表示新实验、独立统计样本或已完成计划卡。
`seed_pool: probe`标明探索池类别，不补造实际seed，也不修改唯一SEED_LEDGER。

| 卡片后缀 | 新增probe_id | 新增claim/family/index/pool | 其他补缺 |
| --- | --- | --- | --- |
| 20261017-gt-tau-new-sequence | 等于原experiment_id | C3 / 上述family / 2 / probe | date取原ID的2026-10-17；原decision的pass/fail两分句分别复制为positive/negative字段 |
| 20261017-direct-tau-executor | 等于原experiment_id | C3 / 上述family / 3 / probe | 无 |
| 20261018-direct-tau-executor-r3 | 等于原experiment_id | C3 / 上述family / 4 / probe | 无 |
| 20261018-direct-tau-dagger | 等于原experiment_id | C3 / 上述family / 5 / probe | 无 |
| 20261018-direct-tau-dagger-pd | 等于原experiment_id | C3 / 上述family / 6 / probe | 无 |
| 20261018-absolute-plan-executor | 等于原experiment_id | C3 / 上述family / 7 / probe | 无 |

## Diff与复现

运行环境：仓库main，系统`python3`；纯文档/CPU校验，无conda激活和GPU作业。
系统`python`为旧Python，不能运行本仓库verify，复现须用`python3`。

修改前：`python3 tools/verify.py --changed`退出1，仅上述6张卡schema缺口。
修改后：全库卡片schema/seed与Markdown检查、选中的治理测试均通过（VERIFY PASS）；
`git diff --cached --check`通过，暂存范围仅本轮19个文件。18份原有文档的每一原始行均按
原顺序保留，diff无删除行。逐卡新增12行有效性边界；6卡另补metadata，STATE新增31行。

```sh
python3 tools/verify.py --changed
# 显式核对文档范围，不包含并行会话的代码改动：
git diff --check -- docs/STATE.md 'src/task/*/docs/experiments/probes/*.md' 'docs/audit/*hygiene*'
# 提交后可复现完整本轮diff（从git log找到本报告所在提交）：
git log -1 --format='%H %s' -- docs/audit/historical-card-hygiene-20261022.md
# 将上一步SHA设为 card_hygiene_commit，再执行：
git show --format=fuller --stat "$card_hygiene_commit"
git show "$card_hygiene_commit" -- docs/STATE.md 'src/task/*/docs/experiments/probes/*.md' 'docs/audit/*hygiene*'
git ls-remote origin main
git rev-parse HEAD
```

提交前只stage本轮19个明确文件。并行会话有未提交代码；为不stash/reset/覆盖这些修改，
在本轮提交的临时 **detached sparse worktree** 执行`git pull --rebase origin main`，
不创建分支。同步后检查该HEAD与主工作区HEAD一致才推送main；若不同则先停止直接推送、
处理上游差异。稀疏工作树只checkout文档，完成后删除本轮临时工作树。
HEAD与远端SHA以交付回执为准，报告不自引用其提交SHA。


实际验证环境与命令（临时文件也限定本轮hygiene目录，检查后清理）：

```sh
python3 --version  # Python 3.8.10
mkdir -p docs/audit/hygiene-verify-tmp-20261022
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/docs/audit/hygiene-verify-tmp-20261022" PYTEST_ADDOPTS="-p no:cacheprovider --basetemp=$PWD/docs/audit/hygiene-verify-tmp-20261022/pytest" python3 tools/verify.py --changed
# 以上在19文件stage后执行，返回VERIFY PASS；pytest由verify按文档变更自动选择。
# 提交之后，在隔离的文档工作树同步：
git worktree add --detach --no-checkout docs/audit/hygiene-sync-20261022 HEAD
git -C docs/audit/hygiene-sync-20261022 sparse-checkout set --no-cone '/docs/STATE.md' '/docs/audit/*hygiene*' '/src/task/*/docs/experiments/probes/*.md'
git -C docs/audit/hygiene-sync-20261022 pull --rebase origin main
git -C docs/audit/hygiene-sync-20261022 rev-parse HEAD
git rev-parse HEAD  # 必须等于同步工作树HEAD
git worktree remove docs/audit/hygiene-sync-20261022
git push origin main
git ls-remote origin main
git rev-parse HEAD  # 推送后两个SHA必须一致
```


Diff摘录（实际metadata补缺；全量diff用上方git show命令查看）：

```diff
diff --git a/src/task/trajectory-policy/docs/experiments/probes/P-20261017-gt-tau-new-sequence.md b/src/task/trajectory-policy/docs/experiments/probes/P-20261017-gt-tau-new-sequence.md
index 42f0db11..09b67d92 100644
--- a/src/task/trajectory-policy/docs/experiments/probes/P-20261017-gt-tau-new-sequence.md
+++ b/src/task/trajectory-policy/docs/experiments/probes/P-20261017-gt-tau-new-sequence.md
@@ -1,5 +1,13 @@
 ---
 schema: ref2dex.probe.v2
+probe_id: P-20261017-gt-tau-new-sequence
+claim_id: C3
+hypothesis_family: HF-trajectory-policy-executor-transition
+probe_index_in_family: 2
+seed_pool: probe
+date: 2026-10-17
+decision_changed_if_positive: A pass permits moving to a learned executor on this sequence.
+decision_changed_if_negative: A fail blocks that route and triggers implementation/data review before PPO.
 experiment_id: P-20261017-gt-tau-new-sequence
 task: trajectory-policy
 branch: main
```
