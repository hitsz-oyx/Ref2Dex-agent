# ref8：H → 候选 τ 的方法选择

日期：2026-10-10。依据：[当前 ref8](../user/ref/ref8.md)。范围：离线 proposal；冻结 PointWorld、E/I 必要性实验和在线执行。本次只阅读论文与作者源码，没有训练、下载数据、安装或克隆仓库。以下建议是研究判断，不是实验结论。

## 先排除可修复的输入问题

[现有 H→τ Probe](../experiments/probes/P-20261010-history-to-tau.md) 在 train/val 上改善，在 seed-414 test 上 point RMSE 为 302.80 mm，高于 persistence 的 262.14 mm。6144 个 train windows 来自 96 个 episode，不等于 6144 条独立示范。模型训练用 L1，单输出更接近条件中位数；仅凭这个结果不能判定“唯一未来回归必然失败”或证明多模态性。

先由主任务核对旧权重、输入标准化、seed 分布和窗口/坐标合同。多模态模型同样依赖这些输入，不能掩盖归一化 bug、缺少任务目标或分布外状态。observed-τ ranking 的 70.07% 与 shuffle 45.99% 是有用的局部线索，但只有 7 个 informative、近似 H 匹配 anchor；不能等同正式 same-current-state rolling gate 已通过，更不能假设它在生成候选上仍准确。[历史 utility 审计](../experiments/probes/P-20261010-history-utility-effect-audit.md)、[abstention Probe](../experiments/probes/P-20261010-tau-selector-abstention.md)

## 方法比较

| 方法 | 原方法需要什么 | 能否部署时不读真实未来参考，产生本项目 24×11×3 τ？ | 当前决策 |
| --- | --- | --- | --- |
| ACT / CVAE | 示范 observation 与未来 action chunk；posterior 训练时读未来 chunk，decoder 读当前 observation 与 latent | 可改为监督未来手位移；未来 τ 只作训练 label/posterior 输入，推理从 prior 采样。作者标准推理却是 z=0，直接运行只有一个候选 | 最小多模态训练候选，必须明确改造 |
| Diffusion Policy | observation-history / future action sequence 配对；条件去噪生成整段 action | 可将每步 33-D 手位移当生成目标，当前 H 作全局条件，多个随机噪声给多个 τ；这是本项目改造，不是作者现成手轨迹输出 | CVAE 确有覆盖瓶颈后再考虑小模型 |
| PointWAM | 当前彩色点云、手点、语言指令；human/robot 手轨迹及 scene tracks 监督 | 原法已由当前输入预测手/场景轨迹，无需推理时 future GT；但原手点为每手 10 点，本项目 11 点需适配；轨迹头是确定性 MLP，不能直接得到多模态 K 候选 | 借坐标/手场景拆分思想，当前不引入整套预训练 |
| DexWM | 图像历史、候选动作；训练 next-latent 与 hand heatmaps；规划还需 goal image | 原法是动作条件 transition；需先提供动作，CEM 结合 goal 优化关节，再 FK 得手路径。不是 H→τ 的现成 proposal | 不作为当前 blocker 的直接替换 |

ACT 的 chunk/CVAE 合同及 deterministic z=0 见[论文 §IV](https://arxiv.org/html/2304.13705v1)。更具体地，[固定作者源码 L105–129](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/detr/models/detr_vae.py#L105) 视觉分支传 latent，去视觉的 state-only 分支却没有把 latent 送入 transformer。因此不能直接删图像后声称得到 state-only 多模态 ACT。要显式实现 `decoder(H, current_hand, z)`，检查改变 z 是否改变完整 τ。

Diffusion Policy 用观测条件生成 action sequence，适合多模态分布与滚动执行；[论文](https://arxiv.org/abs/2303.04137)与[固定 lowdim 源码](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_lowdim_policy.py)显示从随机噪声采样、条件限于 observation 前缀，normalizer 在输出时逆变换。复用时须显式改 horizon/33-D 输出与 `t+1:t+24` 对齐，不能沿用作者 action slice 造成一帧偏移；同样必须适配本项目有效窗口 mask。

PointWAM 的[论文 §3/B.5](https://arxiv.org/html/2610.02840v1)使用初始时刻手/场景点的位移、分 horizon 标准化，共享 backbone 上两个 MLP 头用轨迹平方误差监督。它提供 H→手轨迹 的方法先例，**没有提供“改成多模态就能修复我们的 seed 泛化”的证据**。作者[项目页](https://chrockey.github.io/PointWAM/)截至本次读取仍标 `Code soon`；搜索所得 xiaooai/PointWAM 不据此视作该论文作者实现。当前没有可核实的作者源码可迁入。

DexWM 的[论文 §3.2–3.4/Appendix B](https://arxiv.org/html/2512.13644v1)是 action-conditioned latent dynamics + goal-image CEM；官方[源码入口/数据说明](https://github.com/facebookresearch/dexwm/tree/74ccc30a3f390d49445b63ec6f7469805d5bd954)有 EgoDex/DROID 与 RoboCasa exploratory 数据。其 action 条件与视觉目标不能凭本项目 H 向量替代，当前也不需要为 proposal 增加整套 visual world model。

## 最低成本、能改变下一步的检查

输入审计先完成，随后在同一 episode split 上比较 persistence、修复后的 deterministic displacement predictor，以及 train-only retrieval 候选。检索只用当前允许的 H/current hand，存 train 片段相对其起始手点的位移，再附着到 query 的当前手点；K 固定为 8，episode 去重防止八个相邻窗口冒充八种模式。它不需要新训练，可便宜检查现有数据是否包含近似可用未来；也不能假定这种附着自动保留接触或动力学可行性。

如果检索覆盖或修复后的单预测已足够，先保留最简单 proposal。如果审计正常、单预测不足、近邻片段显示有多条合理延续，则值得跑一个小 CVAE Probe：沿用当前可部署条件与 24-step 标签，输出相对当前 11 手点的位移，latent 如 16-D、网络约现有 width 256，reconstruction + KL；推理固定 K=8 prior samples，保留 persistence 与 z=0。只做单个预先指定训练，不由 test 选 KL、K、checkpoint 或随机样本。是否优于检索是待检验假设，不能提前宣称。

必须分别报告：K=1 prediction；best-of-K 离线覆盖上界；样本间距离与几何/速度异常；以及 future-label-free 选择结果。best-of-K 用 GT τ 的选择只作诊断，**不是可部署 selector**；若覆盖好而选不出，就进入 τ→Y 的分布迁移问题，不再盲目增大 proposal。CVAE 的 posterior 重建良好而 prior samples 无覆盖，或 z 不改变输出，都不能叫多模态成功。不能平均多个候选后冒充可执行计划，平均可能落在各条实际轨迹之外。

若无覆盖信号或部署选择失败，记录 UNCLEAR/UNPROMISING，当前停在离线边界。Diffusion 的小型 temporal denoiser 可作为后续有动机的替代；完整 ACT 图像架构、PointWAM 大规模 human pretrain、DexWM goal-image CEM、同一轮架构/K/KL sweep 现在成本更高且不能优先解释旧 seed-414 失败。不启动在线 R/PW/selector 控制，也不从这些论文推广全局 Cm claim。

## 追溯

本次后续[固定诊断/最小匹配Probe](../experiments/probes/P-20261010-history-tau-proposal-diagnosis.md)已完成：旧H包含未来参考/contact差；纯测量H下位移target的held point RMSE为212.66mm，优于persistence262.14mm，位置target463.30mm。八个train-only检索候选的GT覆盖171.91mm，但top1为242.78mm，未过固定门槛。它支持先保留位移预测基线、区分覆盖与选择，尚不支持必须增加CVAE/DP、生成候选能执行或新的Cm结论。两个target的normalization也不同，不能把差异全归因于纯anchoring；旧H与新H结果同样不是匹配比较。

作者源码固定身份：ACT `742c753c0d4a5d87076c8f69e5628c79a8cc5488`；Diffusion Policy `5ba07ac6661db573af695b419a7947ecb704690f`；DexWM `74ccc30a3f390d49445b63ec6f7469805d5bd954`。书目信息见 `paper/act/schema.json`、`paper/diffusion-policy/schema.json`、`paper/pointwam/schema.json`、`paper/dexwm/schema.json`。
