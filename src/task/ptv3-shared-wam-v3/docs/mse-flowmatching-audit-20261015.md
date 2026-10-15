# MSE 与 Flow Matching：现有权重诊断（2026-10-15）

当前判断：**UNCLEAR；保留 MSE 主基线，尚不足以因多模态平均化而改用 FM。** 发现幅度和速度衰减，但缺少相似条件下多个独立真实未来的支持。此处没有训练 FM，不能宣称 FM 输赢。

## 输入与执行

按用户给定《mse和flowmatching的选择.md》实现。冻结远端 strict 运行的 step16000（目标80000步），不修改运行中的训练/normalizer/数据。8个实际有 DEV 的来源各128查询，共1024窗、570次原始录制；每窗只取第一只完整观测的11点手。旧配方的 DexYCB 无 DEV，因此本次不覆盖它；全被试扩充版不是这批权重的训练输入。候选库来自 TRAIN，共11216窗；每源目标2048，HOT1795/H2O1229因准入和抽样去重不足目标。不得解释为全量数据评估。

GPU缓存代码 b3e6d009，最终聚合代码56a1da34；抽样/聚类seed214。远端GPU0与本会话自己的预训练共享，3GiB硬上限，batch1/8/32逐级检查。主缓存r2耗时273.6秒，峰值约1.37GiB；连同烟测及主动停止的早期批次，总GPU阶段约8.5分钟。父训练复查step18780、1.466updates/s，四卡82–96%利用率；诊断显存已释放。CPU仅做数组检索、聚类、统计和绘图。

## 普通预测统计

以下均为查询窗中位数；幅度指24步各手点相对当前的位移模长比，速度为相邻步位移模长比。GT幅度<5mm不计入幅度比；速度另排除24步GT幅度<5mm或GT速度<0.01m/s。各源等量抽样，不是训练混合分布的加权总体。

| 来源 | ADE mm | 零运动 ADE mm | 匀速 ADE mm | 运动幅度/GT | 速度/GT |
|---|---:|---:|---:|---:|---:|
| OakInk2 | 12.21 | 16.16 | 17.44 | 0.88 | 0.82 |
| GRAB | 50.27 | 90.70 | 139.39 | 0.90 | 0.80 |
| ARCTIC | 36.08 | 67.07 | 84.60 | 0.79 | 0.65 |
| UniDex | 19.97 | 34.64 | 79.53 | 0.74 | 0.46 |
| TACO | 29.97 | 53.58 | 85.54 | 0.80 | 0.60 |
| HOT3D | 48.11 | 72.61 | 91.12 | 0.81 | 0.67 |
| H2O | 17.12 | 22.50 | 30.12 | 0.72 | 0.57 |
| HOI4D | 17.29 | 41.57 | 42.23 | 0.92 | 0.64 |

所有源幅度中位数<1、速度中位数<1，衰减确实存在。所有源的MSE ADE中位数也均低于两种简单外推；这只说明模型学到了有效预测，不证明机器人收益。欠拟合（仅16k步）、标签噪声/高频抖动、不可观测因素和多模态均可造成衰减，单看这张表不能区分。

## 条件邻域与平均化诊断

用 frozen pre-head scene/hand/goal摘要z检索，TRAIN-only whitening；同源、同几何/场景物体身份、同手存在模式。比较完整H4、物体历史与已提供的h24物体目标G，全部使用当前目标坐标下的米单位。未来手GT不进入检索；把未来标签清零不改变模型输出的检查通过。每邻域只保留每次原始录制一个邻居，至少8次录制，最多32。

2cm紧阈值下，6个具有跨split几何支持的来源共768查询，**0个**满足至少8个独立邻居。5cm下ARCTIC仅2个、H2O仅2个（均不足16邻居），其余0；raw与latent检索一致。UniDex/HOI4D的TRAIN/DEV物体几何完全不重合，另外256查询无法在这一定义下条件诊断，不能视为单峰证据。

在仅有的4个宽邻域中，未发现同时通过绝对轨迹与位移轨迹判据的多模态。ARCTIC的pred→条件均值/最近真实轨迹/3NN中位距离分别20.98/16.08/21.27mm，GT→3NN为22.76mm；H2O相应为37.27/37.88/39.25mm，GT→3NN39.33mm。没有稳定的“预测更靠均值且远离真实轨迹”的信号，但样本太少，不能推广为不存在平均化。

不能画出可信的low/high ambiguity比较并据此选架构：4个支持点不足以支撑分层结论。JSON保留所有距离、宽/紧阈值、两种检索、两种坐标空间、逐capture bootstrap以及接近物体表面的指尖代理。指尖代理是稀疏点云无符号距离，不是mesh接触、穿透或抓取成功率。

复审修正：H2O一个原始take按标签缺口拆成多个文件，使用manifest.source_sequence重新合并，仅做CPU重算，无需重跑GPU。其他源保留唯一文件身份，避免UniDex多个独立rollout批次重用episode相对名导致误合并。初版按文件身份的H2O 10个宽邻域降为2个；中间r1/r2/r3报告用于追溯，**本报告与analysis-r4为最终结果**。570次录制不等于570个独立人；native另附group权重敏感性（HOT为participant，HOI为object），bootstrap条件于固定候选库，不含候选抽样/聚类不确定性。独立只读审查与4项合同/合成控制测试通过。

## 当前决策与下一次可判别对照

当前继续MSE训练，保留完整训练预算；本轮不因衰减就更换主干或启动大规模FM。完成训练后可在同一面板复跑，区分欠拟合与持续衰减。若下一轮要实际比较架构，冻结同一个backbone，以相同数据/预算重训MSE头与条件FM头；同时报告单样本ADE/FDE、minADE/FDE@8、有效模式precision/coverage和采样延迟，不能拿FM的best-of-8直接同MSE单输出比较后就宣布胜利。可先用TRAIN中重复条件更密集的录制做机制探针，必须与DEV泛化结论分开。

MSE手头在平方误差下趋向条件均值是理论机制；本次没有识别出足够可靠的多模态条件。FM通过回归条件概率路径的向量场学习可采样分布，依然需要实际训练与匹配比较；参见[原始论文](https://arxiv.org/abs/2210.02747)，它本身不保证我们的任务效果。

## 产物

[完整机器报告](../../../../outputs/ptv3-shared-wam-v3/mse-multimodality-analysis-20261015-r4/report.md)、[summary.json](../../../../outputs/ptv3-shared-wam-v3/mse-multimodality-analysis-20261015-r4/summary.json)、[逐查询指标](../../../../outputs/ptv3-shared-wam-v3/mse-multimodality-analysis-20261015-r4/queries.json)、[邻域控制](../../../../outputs/ptv3-shared-wam-v3/mse-multimodality-analysis-20261015-r4/conditional.json)。

![逐源诊断](../../../../outputs/ptv3-shared-wam-v3/mse-multimodality-analysis-20261015-r4/prediction_diagnostics.png)

![轨迹例子；此例未通过多模态判据](../../../../outputs/ptv3-shared-wam-v3/mse-multimodality-analysis-20261015-r4/case-00.png)

复现：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v3/tools/audit/analyze_multimodality.py \
  --cache outputs/ptv3-shared-wam-v3/mse-multimodality-cache-20261015-r2 \
  --output outputs/ptv3-shared-wam-v3/<fresh-analysis-directory> \
  --native-manifest outputs/3d-data-preparation/native3d-clean-motion-20261015-r2/manifest.json
```
