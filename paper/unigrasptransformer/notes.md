# UniGraspTransformer 阅读笔记

核查日期：2026-10-19；来源为官方论文、代码与 README，后续已运行发布state策略的本地兼容评估，尚非论文环境严格复现。

微软亚洲研究院的 CVPR 2025 工作。相对 UniDexGrasp++ 的复杂课程与反复 generalist/specialist 训练，它先用 PPO 训练逐物体专家、采集成功轨迹，再离线监督训练统一 Transformer。专家平均成功率 94.1%；每项采集 1000 条，共 320 万条、每条 200 步。论文称 3200 objects；不要直接将结果配置中的物体—尺度条目解释成 3200 个不同 object code。[论文 §3](https://arxiv.org/html/2412.02699v2#S3)

论文表 2 的成功率如下，顺序为已见物体／已见类别的新物体／新类别物体。属于论文报告值，不是本仓库复现结果。

| 输入设置 | UniDexGrasp++ | UniGraspTransformer |
|---|---|---|
| State | 87.9 / 84.3 / 83.1 | 91.2 / 89.2 / 88.3 |
| Vision | 85.4 / 79.6 / 76.7 | 88.9 / 87.3 / 86.8 |

规模代价：专家训练 16×V100、80 小时；统一策略 8×A100、70 小时；点云编码器另需 A100、40 小时。简化流程不等于小算力方法。[论文 §4、表2](https://arxiv.org/html/2412.02699v2#S4)

## 接口与实现要点

六个 token 是当前帧的特征分组：本体状态、上一动作、物体状态、手物距离、时间、点云特征。注意力发生在这些分组之间；含上一动作和时间编码不等于输入整段历史，更不是预测未来 τ。`ObsPatchTransformer.forward()` 将观测切片编码成 token，随后注意力与 MLP 输出当前 24 维动作。手物距离按表1为36维，§3.3正文的26维与表不一致，不采用该数字。[官方实现](https://github.com/microsoft/UniGraspTransformer/blob/main/dexgrasp/algorithms/rl/dagger_value/module.py)、[配置](https://github.com/microsoft/UniGraspTransformer/blob/main/dexgrasp/cfg/train/universal_policy_state_based.yaml)

其观测、点云编码器和 reward 都经过改造：显式手物距离与时间、完整／部分点云双设置，reward 区分接触前张手与接触后抬升。它并非只把原始 UniDexGrasp++ MLP 换成 Transformer；效果不能全部归因于注意力架构。[论文 §3.1–3.4](https://arxiv.org/html/2412.02699v2#S3)

## 对当前路线的意义与边界

研究判断：适合借鉴“高质量专家数据→较大执行器离线 BC”，以及把 native observation 与 WAM τ 分组编码的设计；现成策略没有 τ 条件，不能直接替代当前 τ executor。它研究通用抓取，也不直接解决我们的阶段调度、释放或几何轨迹跟踪。

官方 README 指定论文使用 IsaacGym3、Python3.8、CUDA11.6、Ubuntu20.04、驱动550.107.02。IsaacGym4 提供3015个 dedicated policies，另185项加载失败；Gym4 universal policies 仍标为 Coming Soon。两版发布情况提示版本兼容值得核查，不能据此断言本仓库 UniDexGrasp++ 低成功率的根因。[官方 README](https://github.com/microsoft/UniGraspTransformer#v-test-pre-trained-models-isaacgym4)

本地评估记录：Preview4/替代动态初态下，最终jar4/256、camera46/256；官方静态初态包缺失是协议限制，不能与论文表2直接比较。详见[实验卡](../../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-unigrasp-published-eval.md)。
