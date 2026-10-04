# 2026-09-30：显式物理预测作为推理输入

HF06 的最后一次 MLP 检查未过 contact coverage 和 arbitration stability；
关闭依赖置信下界认证 teacher 标签的实现。Cm 核心问题不变。区别于冻结的
训练期 auxiliary（评估不调用 Cm），本路线在真实策略推理时显式提供固定的
物理预测，训练策略自己学习何时使用它，不用置信区间硬认证候选。

固定机制：在 canonical 三条 airplane motion 上，用 self-trained source_e260
采集完整首回合 observation/action 轨迹。用同一批数据、同架构和同初始化
训练四个 BC actor：Cm-on 接收预测的一步物体局部位移 3D + contact q50，
Cm-off 的对应四维为零；随机未训练同架构物理 head 为 generic placebo；
冻结 teacher action 的前四维为 action-information control。所有臂都计算同一
source teacher action，只有四维输入的内容不同。动作仍由 BC actor 输出。
actor 由零训练，不使用官方 actor；source expert 和 physical predictor 固定。

策略推理仍需要 self-trained source expert 生成 Cm 的候选动作，因此不是
单 actor 部署或完全独立学生的主张。source action 不直接传给 on/off actor。
random 和 action controls 用于区分物理监督收益与额外 teacher-action 信息。
预测模型仅在 airplane 接触触发分布上训练，完整轨迹有分布外风险；完整策略
评估失败就是停止信号，不调目标、阈值、学生宽度或 seed。

这是 Decision Probe：正向才继续多训练 seed Validation；负向关闭该 bottleneck
实现。固定拟合 seed 278，数据 seeds 279/280，实际评估 seeds 281/282；
不使用 validation seed 池。四臂采用同一个 1446→256→256→18 ReLU actor，
2000 steps、batch 512、Adam lr=1e-3，fit-only 输入归一化。holdout 数据只诊断
BC fidelity，不选 epoch。真实策略主指标是原 evaluator 的首回合 held-lift。

升级门预先固定：Cm-on 相对 off、random、action 各合计 >=5pp，且两个
evaluation seed 相对 off 都不负。否则 `UNPROMISING` 或技术无效 `UNCLEAR`。
结果只支持这个固定 airplane 任务和单训练 seed；不能形成正式因果结论。

预算：1 张空闲 GPU，整个采集/训练/四臂双 seed 评估 <=60 分钟，输出 <=5 GiB，
无视频。不突破 Campaign。输入 hash 漂移、first-episode pairing 失败、非有限
数据、未知 GPU 冲突或超预算立即停止。采集和评估 stdout 写日志文件，避免
旧任务 BrokenPipe 的输出管道失效。现有 checkpoint 和研究工作树只读。

执行后处置：有效 r2 的八臂配对通过，on/off 均 18/128，random 21/128、
action 14/128，升级门失败；关闭这个 BC inference-input 实现，不进入 Validation，
不调参。r1 的 random CUDA reseed 造成配对失败，保留为执行无效；修复后整组
重新训练、评估。两次执行累计 19.81 分钟、单 GPU，输出约 0.5 GiB，无遗留进程。
结果见 [实验卡](../experiments/probes/P-20260930-cm-inference-bottleneck.md)。
