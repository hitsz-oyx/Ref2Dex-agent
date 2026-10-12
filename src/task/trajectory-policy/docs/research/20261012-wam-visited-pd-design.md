# WAM 访问状态上的 PD 监督：实施前核对

日期：2026-10-12。只读核对原始文献、已有代码和实验卡；没有训练或仿真。

当前 [WAM 执行记录](../experiments/../../../goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-execution.md) 中 teacher/GT-R 各持物 441 controls 并完成放置，WAM 持物 0；startup 修复把第一步 PD XYZ 误差从 225.27 降到 11.71mm，但 state9 再次漂移。这支持先检验访问状态是否可被专家恢复，不能用 nominal 离线精度代替恢复能力。

## 文献允许的解释

[DAgger 原论文 §3 / Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) 的关键是执行当前 learner 或 expert/learner 混合策略，在其**实际访问的状态**查询专家动作，再汇总数据；混合策略执行的动作与该状态的专家标签不同。§2.2 明确讨论专家能从错误状态恢复的条件；保证还依赖策略类和学习误差，不能直接套到一个受限观测、冻结 R 的短轮 Probe。

[PointWAM §3.3–3.4、B.4](https://arxiv.org/html/2610.02840v1#S3.SS3) 从预测手轨迹及当前 robot state 输出 action；robot 阶段使用 trajectory+action 监督并训练 retargeter。原文未找到 DAgger 协议，也没有规定冻结已学 R 后只用访问状态即时 PD 标签；其 learned retargeter 不显式保证关节限位或碰撞约束。因此我们可以借鉴 point-trajectory→action 分工，不能称当前 PD-DAgger 是 PointWAM 已验证的训练配方。

## 本仓库已有合同及应复用的部分

[reference-goal-dagger 卡](../experiments/probes/P-20261011-reference-goal-dagger.md) 和 [采集代码](../../tools/run/collect_competent_pd.py) 在动作施加前，让官方 actor 看到每一行自己的 live observation；保存 `expert_action/expert_pd/intervention`，校验独立 PD 重建与 native controller 一致，再执行选择的 learner/expert action。[训练代码](../../tools/run/fit_competent_actor.py) 对访问行只使用 PD 标签，保留原 teacher 的 tau/PD 数据；没有把混合策略未来手当成 expert tau。

该历史路线 actor 输入是 H328 + planned human/object anchors408。当前 WAM actor 合同仍是 measured H4 点/动力学 + planned **object** goal，不能悄悄加入旧路线的人手 anchors、teacher observation、clock 或 source future hand。专家训练期可使用已授权 planned reference 和完整 native observation；这与 learner 部署接口不同，应明确记录。[当前构造器](../../src/trajectory_policy/wam_shadow.py) 不接受未来 measured hand/q/object 作为 scene 输入。

**标签必须是**访问状态上专家查询得到的 PD[t]，与该行 live q[t] 同时记录；PD[t] 导致 hand[t+1]。将 source 名义手轨迹平移到访问状态只是 privileged substitution，不是恢复轨迹标签；同 tick 的 source PD 也不是该访问状态的专家动作。完整专家恢复 rollout 才可为其中真实专家控制起点构造未来 tau，并记录干预起点；不能为 learner 先前历史随意填整段恢复 tau。

## 冻结 R 时的最小训练语义（本地设计推断）

令 `tau=A_theta(H4,Gobject)`，`z=R_frozen(s_live,tau)`，`y=encode_target(expert_PD,live_q)`。按 [PD 编码合同](../../src/trajectory_policy/pd_retargeter.py)，腕部六维减当前实测 q，手指独立维是绝对目标。使用现有 R 的固定 `target_scale` 计算归一化 PD loss；不要重新拟合统计或重复加 native feedback/feedforward。

R 参数 `requires_grad=False` 且 eval，**R 前向保留 autograd**，不对 tau detach，不包 `no_grad`；只有 current state、标签、冻结特征缓存可 detach。直接在 unclipped encoded PD 上监督，并另外报告 intended/applied clipping；饱和 clip 后的相等不能证明 PD 预测准确。核验 PD loss 实际到达 trajectory/hand adapter，同时 R/scene/goal/world 哈希与梯度保持冻结。

即时 12-D PD 标签足以约束复合策略 `R∘A` 的当步动作，**不足以唯一识别 24×33 的中间手轨迹**，也不提供整个未来 action chunk 标签。保留可信原 teacher tau/PD 锚点，可限制 actor 利用 R 的空方向或分布外误差；真实恢复 tau 只能来自对应干预 rollout。低 PD loss 本身不能证明物理恢复、人类轨迹语义或 goal 条件有效。

## 当前的决策门槛（建议，不修改实验卡）

先做 tick16 从该行真实 WAM 状态继续的 expert takeover，保留相同 episode/reference 与 takeover 标记；它测恢复，不是自主 WAM 成功。如果专家也无法恢复，先核对专家 observation/reference、不可逆状态及控制边界，不直接启动 actor PD 拟合。专家能恢复之后，再考虑有界访问状态 PD 聚合。

查询/native 编码不一致、数据 `training_allowed` 或审计不合格、R 对有效 tau 无可用梯度、所需动作明显超出固定执行器范围时不训练。若 privileged teacher 在近似相同 H4/G 下给出冲突 PD，先核对 learner 缺失的任务进度信息；不能用训练 loss 掩盖观测混叠。训练/验证按完整 episode 分开，保留难状态和全部原失败；再次无干预 rollout 才判断闭环改进。
