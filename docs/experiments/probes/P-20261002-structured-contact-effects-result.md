# HF21 slot1：后果信息 PROMISING，实际控制收益待检验

固定[Probe](P-20261002-structured-contact-effects.md)的全部信息门通过，标签
PROMISING；这仅是复用数据的模型信息，不是独立Validation或真实策略收益。
代码分支`agent/cm-structured-contact-effects`，拟合实现`344b5db`。

已审计HF19/HF20共5018实际H10窗口，fit2532/cal1295/reused-held1191；fit-only
归一化和监督。四模式同参数容量、三seed各1000次GPU更新，全部进程exit0。
本原型把末3步hand/object力存在代理与mesh clear编码为8种联合事件，显式
条件高度/抬升，并加入当前原生参考、六专家PD差及参考附近的动作输入。

| 模型 | reused-held 支持高度 MAE mm | 联合支持 Brier | 生成分布95窗口高度 MAE mm |
| --- | ---: | ---: | ---: |
| Cm | 9.980 | .06901 | 14.884 |
| state-only | 14.365 | .08458 | 17.455 |
| action-shuffled | 14.516 | .08480 | 17.235 |
| direct-score | 9.935 | 未训练事件头 | 14.485 |

Cm相对两动作信息控制高度改善30.5%/31.3%、联合支持Brier改善18.4%/18.6%。
lift/loss、第一步object-dv/节点position不劣化门通过，生成分布高度门通过，
support/lift概率包含关系误差0。direct-score高度略优于Cm，必须在下一次实际
控制检验保留；不能仅由Cm优于state/shuffled声称物理后果提供独特策略收益。
完整[结果](P-20261002-structured-contact-fit-results-r1.json)。

[独立审计](P-20261002-structured-contact-fit-audit-r1.json)对5018窗口从rawforce/mg
重建联合事件、世界物理观测重建首步标签和当前相对几何、独立PD映射及fit-only
均值/方差，全部通过。任务标签误差0，物理标签归一化误差1.55e-4，当前几何
差3.66e-7m，规范化统计最大差6.39e-7。全部事实及8候选完整NN复现0差，独立
指标最大差1.56e-6，原固定信息门与标签一致。

CUDA拟合与预测155.79秒、完整审计18.95秒；加排除seed590两次输入工程5.24秒
与60秒准备计入总239.98秒，拟合输出47.87MiB。借用HF19源/预训练28.165
GPU分钟/498.84MiB与HF20源+审计101.58GPU分钟另报；不把复用视作免费数据。

进一步的[冻结生成工程](P-20261002-structured-action-engineering-r1.json)也通过：
32步输入优化加入joint-support/几何风险约束、置信增益与OOD abstain；排除
seed590的96当前状态中Cm54产生不同cup的初始PD，32个当前OOD全部返回cup。
完整NN、规划头、score/loss/support/PD复现0，权重凸域、固定旋转和最终可行性
门通过，所有网络参数指纹未变。6.07秒；预测平均改变窗口增益2.675mm不是
真实抬升收益，不对工程seed的未来后果调参。

下一步是[原生执行工程](P-20261002-structured-native-engineering.md)，直接在GPU
物理任务执行新H10程序并逐步审计。它通过后才登记HF21 slot2新独立随机执行
机会/风险检验，再检验反复观察与重新决策，随后策略训练和最终稳定抓取。
完整goal和C3仍OPEN，原HF19/HF20/HD03事实与门全部保留。
