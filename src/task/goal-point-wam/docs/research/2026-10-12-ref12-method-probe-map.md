# ref1_2 方法拆解与可执行 Probe

日期：2026-10-12。任务：`goal-point-wam`。目的：把文献的中间机制转成能改变下一步选择的实验，而不是只沿用 ref1_2 的最后建议。本文只做原始来源核验与实验设计，未运行 GPU，未修改当前训练的模型、数据或 trainer。

## 核验范围

先读了已有 `paper/act2goal/schema.json` 和 `goal-policy-primary-audit-20261012.md`，再核对下列原文。`checked` 只表示原文或代码确有此机制；`adaptation` 是针对本仓库的推断，不能写成原论文在本数据上的结果。

| 文献 | 本次核验的版本与一级来源 | 代码证据边界 |
| --- | --- | --- |
| Act2Goal | [arXiv v1，§III-A–D](https://arxiv.org/html/2512.23541v1#S3) | [作者项目页](https://act2goal.github.io/)本次浏览无可读正文；[作者 GitHub](https://github.com/act2goal)公开页面只显示项目网页仓库，未核验训练实现 |
| DreamTrajectory | [arXiv v2，§3.2–3.4](https://arxiv.org/html/2608.01381v2#S3) | 有原文方法与参数；本次未找到并核验作者训练代码，不能声称代码复现 |
| Fast-WAM | [arXiv v2，§3](https://arxiv.org/html/2603.16666v2#S3) | [官方 `fastwam.py`](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/fastwam.py)的 `_build_mot_attention_mask`、`infer_action` 已读；是访问当日 main，未固定 commit |
| InternW0-Δ | [arXiv v1，§3.4–3.8](https://arxiv.org/html/2609.31394v1#S3) | [官方仓库](https://github.com/InternRobotics/InternW0-Delta)与[4D 缓存/训练指南](https://github.com/InternRobotics/InternW0-Delta/blob/main/tools/4D_distillation/README.md)可读；未逐行审计全部 MoT/CI 实现 |

GitHub API 目录树读取遇到匿名限额 403；用原文和直接官方文件继续核验，没有下载模型、数据或克隆大仓库。

## 1. Act2Goal：目标条件的中间过程与分阶段优化

**Checked.** It initializes a pretrained goal-conditioned visual generator, concatenates current and goal latents, and supplies layered world features to an action expert. MSTH uses dense proximal and sparse distal predictions. Stage 1 jointly optimizes visual/action flow matching with action weight 0.1; stage 2 optimizes the entire pipeline using action loss only. Online LoRA updates use collected actions paired with the actually achieved observation as goal. This is not evidence that relabeling alone creates several goals for an identical history. [原文 §III](https://arxiv.org/html/2512.23541v1#S3)

**Adaptation：中间几何计划。** 可单独训练 `P(H,G) → target-object motion[1:24]`，使用原样本实际的 object transforms/paired points 做密集监督；将**预测**的中间计划或其特征交给 actor。actor 的推理输入只能是 H、G 和 P 的输出。GT 中间轨迹作为 loss target；若喂给 actor，只能标为 privileged upper-bound 工程诊断，不能混入可部署结果。它测试的是“中间进度表示是否让 goal 更容易用于手轨迹预测”，不是 `W(H, executed hand)` 的前向物理正确性。

**Adaptation：短 horizon 多目标。** 对同一合法 anchor 选择 h∈{6,12,18,24}，把实际 `E_h` 当 G，同时提供明确 h，动作监督仅使用原轨迹前 h 步，h 后 mask 掉；若 world 用同一短动作，则其目标也截到 h。独立 goal-blind teacher-forced world 分支若仍输入完整真实24步动作，可以继续监督完整24步，必须明确与 actor 前缀任务分开。若强制固定 K24，则不能只换成 E6/E12 并继续监督完整24步；应改为由对应真实可兼容24步片段产生目标。复制同一个样本几遍不是增加 goal diversity。

**限制。** 上述是 horizon-conditioned prefix learning；不同 h 的动作前缀仍可能相同，不能据此证明同 H、同 horizon 存在多模态可辨识动作。真正 HER 的新信息来自实际 rollout 的动作和结果，当前离线 observational corpus 不自动获得这种反事实覆盖。MSTH 的长期规划收益也不能直接迁移到仅24步窗口。

## 2. DreamTrajectory：计划与执行分离，先预测后排序

**Checked.** A policy predicts planned end-effector poses and robot commands; an independently trained GRU world model predicts realized poses from those commands and observations. Its training data include successful and failed interactions with aligned post-action outcomes. At inference it retains the original candidate and adds 29 AR(1) perturbations, with σ=0.05 and ρ=0.9 in denormalized robot-action space. Candidates are scored by plan agreement and smoothness, without backpropagating through the world model into the policy. [原文 §3.3–3.4](https://arxiv.org/html/2608.01381v2#S3.SS3)

**Adaptation：冻结 PW 的 candidate ranking。** 当前 W 是 hand trajectory→object effects，而论文是 robot commands→executed EE poses；借鉴的是搜索、预测、评分这个机制。以 actor 的24步手点预测为中心，生成时序相关的小扰动，保留原预测，给冻结 PW 批量预测物体结果，再按 target-object endpoint-to-goal 距离加动作先验偏离与平滑代价选择。噪声幅度应以米为单位在我们 hand 合同内重新确定，不能照搬机器人不同通道共用的0.05。预测 xyz 变化后，9D action 中的 XYZ、相对当前手点位移和逐步位移必须按本地真实合同重建（后两项不是速度/加速度）；不能留下互相矛盾的输入。

**先做最便宜的 Blocker 检查。** 固定同一 H、scene tokens、eval mode 与采样点，只改 candidate actions，检查 W 的结果确实依赖动作且数值有限；GT executed action 的 held-out prediction 应优于静止预测，分别报告 source 与运动分桶。排序若受采样噪声支配，先修 wiring/固定采样，不能当方法负结果。

**不能证明的事。** 当前未执行的扰动候选没有真实物理标签；surrogate endpoint 变好不等于物体实际更接近 goal。将 GT action 放入候选仅提供 oracle diagnostic，部署候选池不能含 GT。相近 H 的其他录像片段也不是同一 H 下扰动动作的真实执行结果。可先报告预测改进、排序稳定性、候选离训练分布的距离；需要仿真或真实执行才能形成物理改进结论。

## 3. Fast-WAM：保留未来监督，推理移除未来分支

**Checked.** The pretrained Wan video expert and action expert share masked attention. During training action tokens access the clean first-frame context, but cannot access future video tokens; the clean context cannot attend to future or action tokens. At inference future tokens are absent and actions use a single observation-context pass. The paper separates this from joint imagination, IDM, and removal of the video objective. Official `_build_mot_attention_mask` and `infer_action` agree with that information-flow contract. [原文 §3](https://arxiv.org/html/2603.16666v2#S3), [官方代码](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/fastwam.py)

**Adaptation：共享表示而不是可微 planner。** 当前 Goal-Point-WAM 已有世界辅助监督、action inference 不必调用 world decoder；这与训练/推理分工相容，但不是 Fast-WAM 的 MoT 复现。可在隔离实现中保留 actor 的输入与输出，仅增加未来物体运动的密集 training-only objective；或者在保留共享世界表征后移除推理世界头。重要检查是改变 GT future、executed-action labels 时，同一 H/G 的 actor forward 输出完全不变，且梯度不从 future label 输入通道回到 actor。共享参数通过辅助 loss 更新是允许的。

**限制。** Fast-WAM 的未来预测不是 `W(H, clean executed_action)` teacher forcing，不能将其 attention mask 当作 action-specific dynamics 已解决的证据。已有小规模 world-weight 实验可提供起点，但跨 recipe 的训练差异不能直接归因于“世界模型没有价值”。这条机制不要求今晚再起一组长期 world-on/off 对照。

## 4. InternW0-Δ：Causal Imprint 与训练期未来特征蒸馏

**Checked.** Learnable Causal Imprint tokens aggregate observed context, cannot attend to realized futures, and condition the action expert. They receive adjacent clean-future-latent difference targets and cosine alignment to detached future features. A separate frozen 4D teacher provides geometry/motion descriptors as training-only auxiliary supervision; its branch is absent at action inference. The official guide caches and seals teacher features before continued training. These mechanisms are distinct from differentiable endpoint planning. [原文 §3.4–3.8](https://arxiv.org/html/2609.31394v1#S3), [官方4D指南](https://github.com/InternRobotics/InternW0-Delta/blob/main/tools/4D_distillation/README.md)

**Adaptation：几何 change tokens。** 增加由 H 计算的少量 tokens，监督其预测24步相邻 object-point displacement/translation increments，再交给 actor；object 身份、canonical points 与当前 anchor 必须保持一致。若希望这是 goal-specific plan，则显式让 G/h 进入 change predictor；若只用 H，称为未来变化先验，不声称迫使模型使用 G。预测中间结果和 actor 都不能直接读 GT future。

**Adaptation：冻结 PW 特征老师。** 可以把冻结 PW 在真实执行动作下得到的、与 object/time 对齐的未来表征作为 detached teacher target，用 H/G-side student tokens 回归或 cosine 对齐。先核对 PW teacher 能否预测真实运动，target 是否真的携带未来变化而非恒定/静态特征。更小版本可直接使用当前已有的真实几何变化标签，避免新下载 Track4World 和生成大缓存。采用本地 PW teacher 的实验只能标 PW feature distillation adaptation，不能称为原论文4D基础模型复现。

**限制。** detach 只阻止 teacher 的梯度，不自动阻止前向泄漏；必须检查 actor 输入路径。未来监督改善表征可能提高 BC，但不等于具有准确动作因果模型。不同来源 validity、帧率、motion normalization 应沿用冻结统计并分别审计，不能只优化 aggregate loss。

## 5. 今晚可独立推进的实验优先级

下表是研究设计建议，资源和具体规模由 root 按空闲卡与当前用户授权确定。每条只改变一个机制；单 seed Probe 足够决定是否投入下一轮。4卡 DDP 主线的 live-guard 文件、config 和数据保持原样，新实现放隔离工具/模块并使用独立输出目录。

| 优先级 / 分类 | 想区分的假设 | 最便宜且语义合法的测试 | 结果改变什么决定 |
| --- | --- | --- | --- |
| P0 / Blocker | frozen PW 对 candidate action 可分，且在真实动作上的动态预测足以用于 ranking | 固定128–512个不重叠窗口，按source/运动分桶；真实动作前向误差、固定采样下 action interventions、重复排序 | 不满足则先修输入或 forward model；不进入actor通过W反传 |
| P1 / Decision | 显式中间几何计划比 endpoint side condition 提供更有效的 goal 信息 | 先训练小型 `P(H,G)→object motion`；看 held-out trajectory/endpoint error、静态基线、goal perturbation；通过后接actor预测plan | 决定投入 plan-guided action expert，或保留当前端点路线 |
| P1 / Decision | 当前 H-only 表征缺少可学习的 future-change 信息 | 独立 change-token head + 密集 adjacent-motion target；首先测变化预测及标签泄漏，然后看 actor BC/goal-response | 决定采用 training-only CI-style supervision，或无需新增token通道 |
| P2 / Decision | 合法 horizon/achieved-goal 增广有可辨别额外信号 | 对实际前缀一致改 G、h、action/effect masks；审核三者一致后测各h条件误差与同H跨h响应 | 决定扩展合同支持可变h；不会把任意endpoint替换当HER |
| P2 / Decision | 非可微 ranking 能选出更合理的模型内候选而无需再训actor | 用 P0 通过的冻结W、保留原预测、少量平滑候选、有限米制扰动；报告surrogate指标与支持范围 | 正信号后做执行验证；不凭surrogate关闭/确认物理路线 |
| 后续 / Decision | frozen W 约束能使actor学会goal，而非钻模型漏洞 | 只在P0及独立执行验证足够后，引入小权重 endpoint consistency、BC/trust-region与动作有效性约束 | 决定是否扩展可微闭环；独立forward误差/动作异常恶化即停止 |

这些路径可以使用剩余卡分别做 plan、change-token、合法增广 Probe；不用等待10k主线最终结果才开始。主线中间 checkpoint 只应读已原子完成的独立文件并记录 step/SHA；主线训练 loss 或单次surrogate提升不是正式结论。

## 6. 最小报告合同

每个实际实验卡记录初始化checkpoint/SHA、输入与标签合同、单seed、source配比、有效样本量、冻结panel、训练步数与停止条件。除训练 loss 外至少报：按source的手点EPE、target-object endpoint error、静态/运动分桶、同H合法goal/h干预响应、泄漏检查。ranking额外报原候选与被选候选的预测误差和动作偏离，但明确真实执行结果是否存在。

本次文献核验支持“这些机制值得分开测试”的设计判断；不形成它们在当前 corpus 上有效、无效或能够保证收敛的科研结论。
