# Airplane 物理转移与长期价值学习设计

本设计检验动作条件 Cm 的短期预测能否帮助训练出更好的抓取 actor。用户已确定研究目标：先在 airplane 上展示收益，最终必须形成策略训练闭环，评价持续保持与掉落；跨数据集预训练可选。用户已于2026-09-30审阅并明确同意开始工作，具体实现与工程检查现已获授权；尚无新科学结果。

## 研究问题与证据边界

核心假设是：在足够规模的完整交互数据上学习物理转移，再用长期价值评估候选后果，比同条件的普通 PPO 和直接动作价值监督更能改善训练所得 actor。瞬时物体位移不作动作价值。冻结 actor 的选择器收益仅作为机制检查，不替代最终 RL 评价。

此前 effect-rank、critic loss 加权、冻结 Cm 辅助标签及直接 held-lift 预测均未建立稳定收益。本路线显式连接后继物理状态与长期价值，并在完整过程采集数据；这构成新的高层机制，不复活旧 family。模型准确性和规模效应仍是待检验假设，不是已证结论。

方法依据为 [MVE](https://arxiv.org/abs/1803.00101) 与 [TD-MPC](https://proceedings.mlr.press/v162/hansen22a.html) 的短期模型与长期价值结合思想。本实现是一步候选评分与策略监督，不声称复现完整 MVE 或 TD-MPC。

## 固定任务与可复用底座

复用 canonical airplane 的三条 motion：s3、s7、s9。全部训练臂从同一 source_e260 actor、critic、归一化和优化器状态恢复，核对 checkpoint SHA256 `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`。不使用六专家路由，不为不同 motion 分别挑 baseline。

actor 保留当前 1442 维 observation 与 18 维动作接口。参考信息是已提供的任务条件，不是未来仿真结果；所有臂使用相同参考、随机/确定性采样混合和 rand_action_mask。原始 critic 继续用于真实 rollout 的 PPO/GAE，另建可在预测物理状态上评价的价值网络，避免伪造完整 actor observation。

已核对的时间尺度是 simulation dt 1/60 秒、controlFrequencyInv 2，即控制步 1/30 秒。三条 motion 原文件分别 543、653、488 帧。source_e260 已用 approach=2、held-lift=10、lift-progress=5 的 shaping，不能将其描述为仅靠模仿奖励训练。

## 状态与数据合同

物理输入 x 包含 q、dof velocity、object pose/linear/angular velocity、真实手和物体接触指标、由固定 URDF 得到的手物相对几何，以及持续保持计数和奖励事件状态。quaternion 使用归一化与符号不变损失；不对 quaternion 直接做普通加法。历史长度固定 16 个控制步，episode 开始用 mask 补齐，绝不跨 reset。

任务上下文 c 单独保存 motion 身份、绝对参考帧、初始帧、motion 长度及对应参考 hand/object/contact 信息。其后继由参考索引更新规则确定，不让 Cm 猜测时间或参考。新价值头明确依赖参考上下文，不称作无参考的通用价值函数。

已有 fit_s279 首回合 34608 条转移可用于物理预训练；这些数据没有逐步 reward、显式 terminal next-observation 和完整 termination 合同，不用于伪造 TD 标签。新采集逐步保存实际执行动作、pre/post 物理状态、参考上下文、基础 reward 与各 shaping 分量、done、terminate、timeout、episode/env/motion ID、初始帧与保持/掉落事件。

terminal 后继须在 reset 前取值。自然 motion 终点是本有限任务的终止；意外工程截断单独标记并排除完整 return，不能误当成功或静默补零。采集失败不是科学负向结果。

新数据目标为 100 万条真实控制转移，三 motion 均衡，完整 episode 采集，记录独立 episode 数、起始阶段、成功与失败比例及实际动作覆盖。50% episode 从帧 0 开始，50% 使用固定的参考起始采样，所有后续策略臂相同。公共采集策略使用 source actor 原有采样，按 episode 固定四种额外噪声标准差 0、0.05、0.10、0.20，合法动作裁剪与实际剂量均记录；噪声 RNG 不影响策略或环境 RNG。

按完整 episode 划分 fit 与开发 holdout，不按相邻帧随机拆分。采集 seed 283/284，开发检查 285；执行前扫描现有 experiment/manifest，若冲突则在 Probe seed 池预先重新登记，不能根据结果换 seed。正式 Validation seed 后续另选，禁止复用已有 C1 holdout。

## 模型与价值训练

Cm 用历史编码器 GRU 128 和两层 256 宽 MLP，建立三个独立初始化的动力学成员。预测后继 q、dof velocity、object state、接触，以及即时奖励和真实终止概率；后继手物几何由预测 q/object state 经相同 FK 计算，不独立预测一套互相矛盾的几何。参考上下文直接作为奖励预测条件。连续量按 fit-only 统计归一化，接触预测使用概率，不强行拼成确定的接触真假状态。历史推进使用预测后继与候选动作，事件计数按同一规则更新。

保留一步监督和真实连续片段上的五步预测损失；五步用于约束动力学，不把规划视野扩成五步。损失按归一化后各物理组分别取均值再等权合并，避免高维 q 淹没接触与物体量；reward 与 termination 各为一个独立组。动作无关同容量预测器是物理诊断对照，不直接替代真实策略对照。

从同一 episode 池建立嵌套的 10 万、50 万、100 万条数据档，按完整 episode 截断，统计实际行数。各档固定相同架构和训练更新数，报告真实转移误差、接触校准、reward 误差和价值加权误差；不按小档结果否决完整预算，不按评估策略表现挑模型。正式候选模型固定采用最大档的开发 holdout loss 选择规则。

长期价值 V(history,c) 与直接 Q(history,c,a) 使用相同历史编码器宽度和 256×256 主干。公共预训练的 value label 是已记录采集策略下的完整折扣 return，须明确其后续策略，不称为任意策略价值。进入 PPO 后，以各臂自身真实 rollout 的 lambda-return 更新 V/Q；不把其他臂的后续轨迹无条件当作本臂 on-policy return。discount 与 lambda 从共同训练配置读取并冻结。

V 的真实状态预测与 Cm 预测后继采用同一物理字段、历史推进和参考上下文。为处理非线性价值与接触不确定性，评分平均三个动力学成员各四个接触/终止样本的 return，再减一个样本标准差。独立 teacher RNG，固定每臂初始化和采样流；采样只能在已覆盖动作附近，不把模型当成可靠的远距离反事实模拟器。

## 如何参与策略训练

每个训练状态以当前 actor mean 为中心，候选是原 mean 及每一动作维度 ±0.1，共 37 个，裁剪后去重并保存实际差值。直接 Q 臂与 Cm 臂使用完全相同候选、tie break 和监督形式。

Cm 分数为三个成员的预测即时 reward 加 gamma 乘预测非终止概率及后继 V 的估计；物理状态、接触与终止共同采样，不把平均后继状态直接丢给非线性 V。一步预测之外的收益由 V 承担。直接 Q 臂按自己的 Q 排序。原动作永远保留，不强制接受一个比原动作差的候选。

teacher 不参与环境动作替换。rollout 仍执行原 actor 的 actions，记录其原始 logprob、mu、sigma 与采样 mask。teacher 给出候选标签，梯度不穿过候选选择、Cm 或 V；actor 在 PPO objective 之外加入固定系数 0.01 的动作均值监督，label 为 mean 向最高分候选移动 0.1 倍距离。只有候选分数高于原动作时才监督。监督使用按固定 actor sigma 归一化后的动作均值 MSE，不重写 PPO action/logprob。

候选评分增加的 KL 与 PPO clipping 统计必须记录；如果旧 PPO KL 停止机制触发则三臂沿用同样处理，不单独为 Cm 放宽。推理评估只运行训练得到的 actor，不调用候选选择器或 Cm。

| 训练臂 | 真实 PPO | 额外策略监督 | 目的 |
| --- | --- | --- | --- |
| plain_off | 相同任务、奖励和起点 | 无 | Cm pipeline 相对普通 PPO 的收益 |
| direct_q | 相同 | 直接 Q 选择候选 | 排除额外候选监督自身的解释 |
| cm_value | 相同 | Cm 短期预测加 V 选择候选 | 检验物理模型辅助长期决策 |

公共物理数据与 value/Q 预训练池向三个臂开放；plain_off 不消费 Cm 的模型输出。各臂 actor/critic 起点、PPO 更新数和真实交互数相同。Cm 的额外模型容量与计算是方法成本，单独报告，不能声称三臂总计算严格相同。同容量动作无关监督是否解释收益属于正向后 Validation 的必要消融，不用 direct_q 取代纯 off。

## 共同奖励与稳定抓取评价

保留 source 的 tracking、approach、held-lift 和 progress 项及其原系数，增加每步 0.5×稳定保持进度和一次掉落事件的 -1 反馈。稳定保持进度为连续满足高度至少高于该 episode 初始物体高度 3 cm 且现有 hand/object contact 判据的控制步数除以 30，上限 1。此前已连续保持 30 步后，高度低于 2 cm 或连续六步失去接触触发一次掉落惩罚；episode 内只罚一次，事件状态输入模型和价值头。奖励规则所有臂相同，不按试验结果调权重。

接触判据沿用已核查的 hand/contact body 任一受力及物体受力 >0.1，这是接触代理而非精确手物接触力配对，结果解释须保留这一限制。高度基准保存在 reset 时，不使用变化后的参考高度冒充初始高度。

新的主指标为连续 45 个控制步高度≥3 cm且满足接触，即 1.5 秒：前 30 步是保持，后 15 步是明确的掉落观察窗口。按每个控制步代表的时间区间计时，不按首末采样差算。成功后到自然 episode 结束的掉落率另报，允许参考中后续有意放下，不用整段末尾高度强制否定已完成的保持。

评估关闭提前终止，三个 motion 均从帧 0 开始，固定完整首 episode，不允许按成功与否选择起始帧或剔除。历史五步成功率、最长连续保持时间、平均抬升和接触分量保留为次指标。初始化 object height、实际 control dt、episode 长度必须写入结果，避免不够观察窗口的 episode 被静默当成功。

## Probe 判定与预算

预定两个训练 seed 286/287，三个臂各训练 160 个追加 epoch，horizon 32、64 environments，即每臂每 seed 327680 次新交互；restore 后终点为 e420，不把 160 当绝对 epoch。在追加 0/40/80/160 epoch 固定评估，主判定只看终点，曲线只用于预先定义的学习效率分析。

评估 seed 288/289，每个 seed 96 个首 episode，三 motion 各 32；两个训练 seed 汇总每臂 384 episode。PROMISING 要求 cm_value 相对 plain_off 和 direct_q 的主指标均≥+5个百分点，且两个训练 seed 各自相对两对照均不为负。掉落风险门使用全体 episode 中成功后又掉落的比例，不能相对任一对照恶化超过5个百分点；同时另报条件于成功的掉落比例与分母，分母为零时标为不可估计。完整有效矩阵其他结果为 UNPROMISING；预算不足或数据/模型合同未完成为 UNCLEAR/工程失败，不能据此否定高层假设。此门只决定是否投入 Validation，不作正式科学结论。

Probe 总预算最大两张 GPU、六小时 wall time、20 GiB 新产物，公共采集最多90分钟、模型/价值预训练最多120分钟、六臂训练最多90分钟、矩阵评估最多60分钟。超过默认60分钟的理由是要检验规模化完整交互学习，避免重复百余事件筛选；单一实验不超过 Campaign 的六小时边界。上述为拟定硬上限，实际吞吐须用工程 smoke 校验，不是保证完成时间。

小档模型诊断不消耗新的 policy Probe slot，不启动不同规模的多个 RL sweep。预算耗尽后不偷偷缩减 epoch/seed 来组成科学矩阵。成功率曲线面积以固定 0/40/80/160 epoch 的独立评估点按真实交互次数做梯形积分，仅作为辅指标；不按曲线择优终点。正向立即冻结数据/模型/奖励/终点，优先新训练 seed 的 matched Validation；无正向则停止本实现，记录模型误差、价值排序和策略监督哪个环节失败，不扫描原 gate 的参数。

正式 Validation 的 seed 矩阵、功效/置信区间与动作无关消融在 Probe 正向后另行固定。十月三号23:59前若不能完成，只报告 Probe 边界，不升级为 SUPPORTED。

## 实现与验收边界

复用 DExploreApproachAgent.env_step、DexploreAgent.play_steps 的真实 reward/next_obs/terminate 通道和 source restore bootstrap。新增完整转移 adapter、持续保持 evaluator、物理模型/value teacher 以及旁路 actor 监督模块；不直接修改旧 family 的模型或实验判定。

有意义的工程检查包括 reset/timeout 后继合同、reference phase 推进、quaternion 符号不变性、真实 reward 与模型 label 相等、保持/掉落跨 reset 清零、PPO 执行动作/logprob 未替换、teacher CUDA/CPU RNG 隔离、模型预测后继与 value 输入字段一致，以及每臂 actor 初始权重 hash 和评估配对。

设计已获用户批准。实现遵循本合同，工程修正记录于相应 manifest 与实验卡，不把拟定参数当作已经运行的事实。
