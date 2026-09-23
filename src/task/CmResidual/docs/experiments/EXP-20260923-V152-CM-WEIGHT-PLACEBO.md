# V1.52: Cm 一步效应 PPO 权重的样本对应关系安慰剂

- experiment_id: `EXP-20260923-V152-CM-WEIGHT-PLACEBO`
- branch: `agent/v152-cm-placebo`
- status: `PREREGISTERED`
- official actor checkpoint: **never**

## 前提与待检验问题

V1.51 的单轨迹 s3/未见 seed119–123 成对实验中，Cm-on 396/640 对
Cm-off 317/640，+12.34pp，五 seed 合计均为正，seed 聚类95% CI
+7.66至+17.03pp；但抓取仍远非稳定，且这个结果只能归因于“CmLite
模型+权重配方”的联合处理，不能单独证明学到的动作条件世界模型表征。

安慰剂臂沿用完全相同的冻结 V1.37 CmLite checkpoint、输入、预测与
`w=1+p_contact*clip(abs(predicted_dz)/0.003,0,1)`，但在每个 rollout step
中仅对 `rand_action_mask=1`、真正参与 PPO actor 更新的环境样本随机置乱
权重。置乱保持这些样本的权重多重集、均值、模型调用与边际计算成本，
打断具体状态-动作和权重的对应。使用独立 PyTorch RNG seed152，不消耗
策略采样 RNG；环境奖励、critic、actor 网络和推理均不改。若原 Cm-on
优于置乱，才有更直接的“预测与动作对应关系有用”证据；仍不能声称已
证明原始 Cmv2 几何 token 架构或跨手表征。

## 冻结训练、评估与门槛

原 Cm-on/off 只用 V1.51 已冻结 e300 checkpoint，不用 seed119–123
选择权重或 epoch。两者 SHA256 分别为
`21d4972eb1956258b6d36b12fc31bbcf2ba3dd4dbe622949ee20ffc51547aad8`
和 `d61a8fdec5948dd3a7df7d3021c0e8a98d0fa6dce9dd00d86d1e805af89c4521`。
置乱臂从**同一个自训练** V1.39 s3 e260 checkpoint SHA256
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`
继续到固定 e300；训练 seed70、64env、horizon32、minibatch256、
学习率1e-5、相同 reset 课程/奖励、Cm 权重系数1.0。
输入 corrected_r2 manifest SHA256
`2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`；
CmLite SHA256
`396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a`。

先对纯置乱函数和 launcher 做测试，再跑 e262 两 epoch smoke，核对 checkpoint
恢复、有限性、`permuted=true` 日志、活跃样本权重边际精确不变；失败则
修工程而不视为科研反证。正式训练从 e260 **重新**到 e300，固定最终 epoch。
评估使用全新 seeds124–128，每 seed 每臂两次64环境完整首 episode、
关闭提前终止，总计30次。三臂都只运行同一普通 PPO actor，无 Cm 推理。
先完成全部固定矩阵，再计算结果；不根据途中结果改权重函数或 seed。

预注册“学到的样本对应关系有益”门：原 Cm-on 相对置乱在总成功率
≥8pp、五 seed 各自两次合计均高于置乱、seed 聚类10000次 bootstrap
双侧95% CI 下界>0。另独立检查原 Cm-on 相对 off 在新 seed 是否按同样
门槛复制 V1.51 结果。稳定抓取门仍要求原 Cm-on 的10次运行各≥58/64。
若任一门失败，分别否定相应主张，不把 placebo 或 off 当作“Cm架构无用”。

最多同时 GPU5/6 两卡，启动前各≤1GiB，绝不碰其他进程；产物<20GB，
项目总产物<300GB。用户同期可能修改仓库基本层，训练/评估必须记录
起始代码提交和关键 DExplore 源码指纹；若评估依赖源码中途变更，停止
并将受影响矩阵标为无效，不能拼接不同代码版本。
