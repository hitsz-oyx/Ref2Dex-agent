# 手—物预测的一致性：主源核对与 WAM-v4 建议

核对日期：2026-10-20。问题：已有独立 contact 预测，为什么手与物体轨迹仍会脱离；其他方法如何把 contact 变成运动约束。只查文献与公开源码，没有训练、复现或改变模型。本笔记是 Task 内设计依据，不是已验证的新研究结论。

## 当前判断与适用范围

已经有工作明确研究这个问题。需要区分三件事：预测 contact 标签；使预测 contact 与预测几何相容；使几何变化符合真实动力学。共享网络、多任务 loss 可以帮助第一件事，但不自动满足后两件事。下面先列直接 WAM 证据，再列相邻的**人—物交互预测/生成**方法；后者的机制可以借鉴，不能直接引用为我们的闭环执行已获保证。

本次视频和 train probe 使用的是 WAM-v4 step80000。当前 [model.py](../../../ptv3-shared-wam-v4/src/ptv3_shared_wam_v4/model.py) 的 contact 分支读取共享特征与 object latent，没有读取解码后的未来手/物体坐标；[objectives.py](../../../ptv3-shared-wam-v4/src/ptv3_shared_wam_v4/objectives.py) 分别监督位置与 contact，没有几何一致性项。已有 [train rollout probe](../experiments/probes/P-20261020-wam-mano-preview.md) 显示四个训练样例的真实历史短期误差较小，而预测历史回填会漂移；这支持检查 rollout 鲁棒性，但尚未区分 contact 的贡献。

同日新增24→1对照后，四例真实历史平均误差降至0.82–2.44mm，而持续预测回填误差全部高于24→8。针对漂移的近期优先级应调整为生成历史下的转移/时间一致性；以下几何一致性方案仍是待审计的独立方向，不能称已找到漂移主因。实验结果与回填审计见同一张 probe 卡。只把 τ 输入一个可学习 contact MLP 也不自动约束几何；应由预测手与预测物体共同重算 proximity，并让监督/一致性误差向两种运动输出传梯度。

## 直接 WAM 证据

### PointWAM：共享表示与联合监督，未给出接触硬约束

**PointWAM: 3D World Action Modeling for Dexterous Robotic Manipulation**，arXiv [2610.02840v1](https://arxiv.org/html/2610.02840v1)。共享 3D 特征后分别用 MLP 预测手与场景轨迹；机器人阶段联合微调轨迹预测与动作 retargeter。核对的 Eq.7 是场景/手轨迹误差与动作误差，未加入显式接触一致性项。联合监督鼓励共同运动，并非每帧手物几何必须相容的保证；场景预测不直接输入 retargeter。[§3.2–3.4](https://arxiv.org/html/2610.02840v1#S3.SS2)。

### PhysWAM：用同时依赖两种预测的几何误差耦合输出

**PhysWAM: Physically Consistent World Action Model for Autonomous Driving**，arXiv [2609.37970v1](https://arxiv.org/html/2609.37970v1)。作者明确指出联合生成不自动产生共享几何约束。CPP 用预测深度反投影，再由预测车辆运动变换，与观测 LiDAR/记录运动构成的参照比较；同一误差向深度和运动两个输出传梯度。可借鉴的是“输出间可计算的关系”，不是照搬驾驶几何或把结果外推到抓取。[§3.3 Eq.6–7](https://arxiv.org/html/2609.37970v1#S3.SS3)。

### DreamZero：执行后真实观测更新历史

**World Action Models are Zero-shot Policies**，arXiv [2602.15922v1](https://arxiv.org/html/2602.15922v1)。联合去噪视频与动作；每个动作 chunk 执行后，用真实观测替换 KV cache 内的生成帧，限制生成历史持续积累。这是闭环反馈机制，不是手物表面的接触约束，也不能据此认定模型有长期自由滚动能力。[§3–3.1](https://arxiv.org/html/2602.15922v1#S3.SS1)。

## 相邻 HOI 领域的四个方法

### InterDiff：在接触参考系预测并修正物体运动

**InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion**, ICCV 2023，arXiv [2308.16905](https://arxiv.org/abs/2308.16905)，[作者项目](https://sirui-xu.github.io/InterDiff/)，[官方代码](https://github.com/Sirui-Xu/InterDiff)。

从历史预测未来整个人体和物体。去噪中间结果先重建手物几何，检查穿透和接触距离；需要修正时，在接触人体 marker 的参考系中预测物体相对运动，再变回世界系并混回生成过程。corrector 的训练包含表面接触和穿透惩罚。它不只是输出一个 contact 分类结果。[§3.2、Eq.5–8、Algorithm 1](https://arxiv.org/html/2308.16905v1#S3.SS2)，[附录 B.2](https://arxiv.org/html/2308.16905v1#A2.SS2)。

边界：是学习的修正器与几何软约束，不是刚体接触求解器；论文明确骨架表示下不使用 mesh contact/penetration loss，说明仅有稀疏关节点时，表面约束并不能原样搬过来。[附录 B.2](https://arxiv.org/html/2308.16905v1#A2.SS2)。

### CoopDiff：同一个接触锚点必须跟随双方运动

**CoopDiff: Anticipating 3D Human-object Interactions via Contact-consistent Decoupled Diffusion**, arXiv [2508.07162v1](https://arxiv.org/abs/2508.07162v1)。任务是从历史预测未来 HOI。

人体与物体用不同分支；人体分支输出接触点位置，物体分支从预测刚体位姿计算同一物体局部接触点的世界位置：`C_O = R_hat p + t_hat`。训练对有接触的区域惩罚 `||C_H - C_O||²`；另以人体运动条件引导物体分支。这里的 contact 是跨分支共同锚点，不能只靠各自拟合标签而完全互不约束。[§3.4 Eq.3、§3.5–3.6](https://arxiv.org/html/2508.07162v1#S3.SS4)。

边界：软一致性 loss 不保证每个采样都满足接触；官方论文实验设置还采用 InterDiff correction，因此其完整结果不能全部归因于 contact loss。[§4.1 implementation details](https://arxiv.org/html/2508.07162v1#S4.SS1)。本文未核对到官方源码，不作代码实现承诺。

### JointHOI：显式 contact 与运动重算出来的 contact 对齐

**JointHOI: Jointly Generating Contact Maps Enhances Hand–Object Interaction Generation**, arXiv [2607.01768v1](https://arxiv.org/abs/2607.01768v1)，[作者项目标注 ECCV 2026](https://smk11602.github.io/JointHOI/)。它从文本生成双手—物体运动，使用 GRAB/ARCTIC。

将动态接触距离图与手、物运动联合生成。推理中根据预测 MANO 手表面和预测物体位姿，重新计算每个物体 anchor 到手表面的距离；用显式预测与重算距离的 log-L1 差产生梯度，引导后续去噪。论文明确联合生成本身仍可能产生几何不一致，所以增加这一步。[§3.4 Eq.15–16](https://arxiv.org/html/2607.01768v1#S3.SS4)。

边界：无符号距离相符仍不等于无穿透、摩擦可行或抓稳；指导是软的，不能称为严格物理保证。作者项目的 Code 按钮目前指回本页，未验证到可用源码；这里仅为论文方法核对。

### HOI-Dyn：手的运动应能解释物体的响应

**HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion**, [作者标注 NeurIPS 2025](https://wulin97.github.io/hoi-dyn/)，arXiv [2507.01737v2](https://arxiv.org/abs/2507.01737v2)，[官方代码](https://github.com/wulin97/HOI-Dyn)。

先学习物体响应模型 `D(手的相对运动, 物体状态, 接触/形状上下文)`；生成模型训练时，用 D 检查生成手物运动之间的动态相容性。其 loss 对比生成序列与真值序列上的 dynamics residual，以减轻 D 自身误差造成的误导。D 仅用于训练；它超出了“手和物体表面靠近”的约束。[§3.1 Eq.2–6、§3.2 Eq.10–11](https://arxiv.org/html/2507.01737v2#S3.SS1)。

公开实现 `compute_dynamics_loss_for_generated_hoi` 中 `dyn_loss_type='res'` 确实计算生成分支误差与真值分支误差之差；默认使用生成 contact，提供 GT contact 对照。调用方仍需冻结 D 参数，`eval()` 本身只切换 dropout/BN 行为。[core/model.py L454–479](https://github.com/wulin97/HOI-Dyn/blob/main/core/model.py#L454)。

边界：学习动力学的监督信号不构成真实力学保证；论文中的 driver/responder 说法也不应被理解为“没手接触，物体就绝不能动”，重力、惯性与支撑均需考虑。

## 对我们模型的建议：先让 C 对 τ 负有可检查的约束

以下是设计推断，尚无本仓库 ablation 支持。应同时保留“真实历史下的短期预测”和“完全回填预测的多轮 rollout”两个评价口径；contact 一致性与 rollout 稳定性是两条问题，不能用一次改动预设二者都会解决。

1. **先审计现有输出，不立刻加大训练。** 在现有 train 四例上同帧展示原始手点、预测物体、C 与部位；同时计算真值 C 的精度，以及从预测轨迹重建的 proximity 与 C 的矛盾率。必须沿用当前标签的坐标、尺度、部位定义；先测真值轨迹自身的重建误差，建立稀疏表示下的误差下限。没有真实物体表面或 FK 时，这个量叫 proximity consistency，不能叫真实接触一致性。

2. **最小模型 Probe 优先增加输出之间的几何一致性。** 用预测手几何与预测物体查询位置可微地重算 proximity，再与预测/监督 C 对齐；近接触的距离要比远处更重要。对我们 11 点、128 query 的规模，直接广播计算有限的点/骨架段距离即可，不需要引入 KNN 编码器。如果当前 voxel 标签基于骨架段而非点，就必须对应骨架段；不要把 MANO 拟合表面当原始真值。

3. **防止接触 loss 的简单逃避。** 训练中使用可信 GT contact mask / part target 给“接触必须靠近”的梯度，并保留位置监督；若只用预测 C 作为距离权重，模型可能全部预测无接触以降低 loss。不要把所有无接触点强行推远；接近、释放、滑动都允许短暂近距离。部位一致性应把指定手部区域与对应物体 anchor 连起来，而不是让任意最近手指代替。

4. **再做短 rollout 训练。** 保持既定 24→8 执行口径，先用 2–3 轮预测回填、历史扰动或混合真实/预测历史的小 ablation，检测误差放大是否改善；同时检查真实历史下的前 8 帧是否退化。C 与轨迹完全相符也可能整体走错，因此这项不能被 contact loss 替代。

5. **后续结构可以让接触进入生成过程。** 先形成随时间变化的物体局部接触 anchor/part intent，再让手轨迹与物体位姿 decoder 都读取它；由同一几何状态重建 contact。一开始可以不增加人体网格：骨架约束的能力明确限定为骨架 proximity。需要 robot physical contact 时，用目标手实际 FK/碰撞几何实现接触与关节限制；不能从 11 点唯一恢复表面。已有 [CEDex audit](../../../../../paper/cedex/contact-morphology-primary-audit-20261019.md) 给出接触意图→形态映射→几何优化的静态例子，但不能代替动态验证。

最低成本的近期决策是：**若 C 准而重算接触与 C 矛盾，优先试几何一致性；若二者相符而共同偏离目标，优先试 rollout 鲁棒性/条件歧义。** 若 C 本身不准，先核对标签可辨识度与校准，不能直接让它强力牵引 τ。

## 证据缺口

- 尚未完成当前 checkpoint 的 C—τ—物体逐帧矛盾审计；既往 train rollout 漂移不能单独证明 contact 耦合不足是主要原因。
- 未复现任何引用论文；公开结果的任务、网格质量、条件输入与我们的模型不同。
- 几何距离/共享锚点可以减少悬空和分离，不能替代接触力、摩擦锥、质量惯量、支撑、关节可达性与执行器动力学。若要求严格数值满足某组约束，需要投影/优化/仿真；实际物理正确性仍受模型与观测误差限制。
