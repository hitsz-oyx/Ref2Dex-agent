# Act2Goal：对 ref1_2 的补充核验

日期：2026-10-12。只核对与当前实验决策相关的 [arXiv v1 §III-A/C/D](https://arxiv.org/html/2512.23541v1#S3)，未运行代码。

目标视觉 latent 与当前 observation latent 组成 world 条件，world 生成中间视觉状态，action expert 用分层 cross-attention 接受其特征。它是 goal-conditioned visual generator，而非本仓库 `W(H, executed hand trajectory)` 的同义实现。

两阶段离线训练：初始化**已预训练**的 world model；stage 1 joint visual/action flow matching，action 权重 0.1；stage 2 使用 action loss 端到端更新整个模型。不能把它转述为先独立训练一个新 world，再首次 joint。

在线阶段将实际执行后到达的 observation 作为 achieved goal，用采集动作更新 LoRA。它提供新的 rollout 数据支持，但论文没有证明同一个精确 H 配对多种不同 goal；HER 本身不保证任意 goal 的因果可辨识性。[原始协议及算法](https://arxiv.org/html/2512.23541v1#S3.SS4)

**对本仓库的推断：**可以借鉴预训练后分阶段适配、保留中间过程监督与 achieved-goal 数据语义；当前三臂注入 Probe 不需要同时引入视频生成、HER 或长时轨迹尺度，避免改变过多变量。
