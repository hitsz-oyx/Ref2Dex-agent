---
schema: ref2dex.probe.v2
probe_id: P-20261005-conditional-consequence
experiment_id: P-20261005-conditional-consequence
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 5db4f08
claim_id: C3
hypothesis_family: HF-conditional-consequence
probe_index_in_family: 1
seed_pool: probe
seeds: [237, 238]
decision_changed_if_positive: prioritize bounded same-state candidate mechanism before matched trained-policy utility
decision_changed_if_negative: identify conditional prediction versus task-transfer bottleneck without treating reconstruction as Cm utility
status: UNCLEAR
run_id: conditional-consequence-s237
---

# Can predicted action consequences retain oracle task information?

Result: UNCLEAR: action-sensitive I predictions, but A/B/C fail; predicted consequences retain31.18% oracle gain without stable added value over direct Ha.
Decision: Preserve action sensitivity; stop this fixed fit without selector/PPO, diagnose generalization rather than append epochs or rescue with shuffled results.

## Motivation and root Decision Note

Ref9 explicitly requests H vs Ha vs shuffled-a consequence prediction on ref7,
then task value against direct Ha. Ref8 GT prognosis PROMISING and noisy arm
bridge positive, full sufficiency UNCLEAR. This Decision serves GT→Cm→trained
policy utility; it does not retry the ref7 negative own-I gate or redefine
Mission. Cheapest discriminator: frozen existing854 windows and ref8 split,
one declared fit budget per model, no simulation or seed/width/horizon sweep.

Root keeps E12/I14 and14D intended arm-onehot EXACTLY ref8. Counterfactual
predictions use decision-time intended action only; future actions/PD/feedback
correction not inputs. New held categorical arm cannot be meaningfully inferred
from an unseen onehot dimension; do not silently swap action representation
just to report held-arm generalization. Instead PREDECLARE two-direction
cross-half extrapolation, holding both outer-test environments and opposite
five-wave half out of fitting. Held-arm extrapolation deferred to a future
physically continuous action representation if this Decision warrants it.
Signed18 extension is deferred sensitivity, not fitted to rescue main results.

Resource: idleGPU6 only, main timeout600s, synthetic engineering smoke≤60s,
combined cap660s, outputs≤100MiB. CPU label/hash/OLS/bootstrap/review only.
Stop on dataset/source drift, nonfinite, OOF/environment leakage or resource
conflict. Keep failed runs. Positive leads to a separately frozen candidate
mechanism Probe, not immediate PPO; negative diagnoses which link failed,
with independent review before closing this local method. User requires a
review agent for anomalous results; reviewer already checking design, and
will explicitly review the delivered anomalies. Boundaries unchanged.

## Frozen data and fitting contract

DatasetSHA `138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149`,
collectiona81b36d; prior ref8 `gt-consequence-s231-r2`, source08975e1.
Use its EXACT train688/test166 indices,125/31 environments and normalizers/PCA
for full outer-train fits. H=PCA32 historical/actor/context/base+physical72,
14action slots, MLP64/32tanh. ALL854 risk windows retained including193 early
failures. Mediatorstep8 E12/I14, Y eight old continuation/physical heads at9..32.
Do not change labels, model widths, horizon or physical action dose.

Training: FIXED1500 fullbatch epochs, AdamW.002/weightdecay.001, clip2,
seed237 initialization; no test-selected checkpoint or extra epochs. This
addresses ref8's small200epoch warning by a7.5× larger predeclared budget,
not a convergence assumption. Record every trainloss and last200 fractional
change; a model still improving>5% is flagged as fitting-limited and cannot
by itself justify closing the general predictor/representation hypothesis.
Same architecture and init within H/Ha/shuffle predictor pair and all new
scorers, same1500updates. Fit budget/time/params reported independently of RL
sample efficiency. Neural model computation usesGPU.

Three full predictors: H→E/I, Ha→E/I, H+shuffled-a→E/I. Permute assignment
within wave×motion×phase-quarter, separately for fit/hold/test, seed238.
Also evaluate the FROZEN Ha predictor with permuted TEST actions; report
permutation changed rates. Targets and states stay fixed in all shuffles.
E/I target standardization train only. Report E12/I14/joint prediction MSE,
per-axis rawunit MAE, train/test gaps and physical/prognostic transfer.

Three motion-stratified environment OOF folds seed237 inside outertrain;
each H/Ha/shuffled predictor fitted ONLYother-fold environments. Each fold's
H PCA+all feature/target normalizers fit ONLYfold-fit indices; save provenance.
Convert raw predictions to outertrain-standardized E/I for downstream. All
outertrain scorer inputs use OOFPREDICTIONS, outertest uses full-train predictor.
Test Y never enters predictor, preprocessing or early stopping.

Downstream primary: fit ONCE then freeze H,Ha,GT_HEI,P_H,P_Ha,P_Shuffled,
Ha_P_Ha using same162D slots (signed18 zero), init237/1500epochs, allY8 heads.
OOF predictions for training P arms. Frozen Ha/permutation evaluation changes
prediction E/I only, never re-trains downstream. Separately reuse ref8's
frozen200epoch GT_HEI scorer as plug-in diagnostic:replace GT with full/OOF
predicted E/I in OLDnormalization; no downstream training or selection there.
This distinguishes oracle-to-predicted input shift from OOF-trained task value.
Old H/Ha/GT scores remain reference, not a matched1500epoch comparison.
Re-estimate same-budget GT oracle for the NEW R denominator; report old-oracle
plug-in R separately. Compare old/new budgets transparently, do not use old
weak baseline to advertise a stronger newly fitted mediated model.

## Pre-outcome metrics and gates

Main primaryY axes3/6/7 (late contact, physicalheight failure, height-held
fraction), failureAUC and supported-stratum factual prognosis ranking.
Paired2000 held-out environment bootstrapsseed238; training-seed uncertainty
not included. Support inherits ref8: test166/31env/3motion counts100/45/21,
allarms≥3 and physicalfailure90/nonfailure76. OOFfolds≥20hold environments,
full outertest contrast design rank14. Crosshalf contrast rank is audited
separately; missing14-arm identification makes ONLY that descriptive contrast
UNCLEAR, not the adequately supported main comparison.

Gate A conditional prediction: Ha vs H I14 and joint normalized MSE gains≥5%,
both95% lower>0; Ha vs trainedshuffled I14 gain≥5%, and frozenHa testshuffle
I14 penalty≥5%. E separately reported, not required to exceed5% because ref8
showed I dominates. Shuffled labels must change≥60% of fit/test assignments.

Gate B action contrasts: on OUTERTEST estimate current-only adjusted factual
GT and factual P coefficients; separately compare SAMEH mean model candidate
arm-minus-zero vectors to adjusted GT. Report14×E12/I14 signed vectors, units,
sign agreement (GTnormalized axis magnitude≥.1), correlation, gain vs zero,
amplitude ratio/error. Main predicted Ha candidate I contrast correlation≥.50,
sign agreement≥.65 and MSE gain vs zero≥.10. Candidate-H contrasts must be
exactlyzero (no action slot). Descriptive cross-half predictors trained on
outertrain first5/last5waves each forecast outertest oppositehalf: report
E/I MSE and candidate-I vs marginalGT contrasts, do not replace main gate
with whichever half works. Different-state randomized contrast estimates
are not true individual counterfactual labels; shared-zero uncertainty remains.

Gate C task value: P_Ha OR Ha_P_Ha must improve directHa primary normalized
MSE≥5% with paired95% lower>0, physicalheight failure cannot worsen>2%,
P_Ha must improve P_H primary≥3%, and frozenactionshuffle must worsenP_Ha
primary≥2%. R=(L(H)−L(P_Ha))/(L(H)−L(GT_HEI)) point≥.25, denominatorGT gain
positive with bootstrap95% lower>0 and ratio valid≥95% draws. This compares
representation value with direct intended-action value, not reconstruction.
Report allcomparisons/ranking; alternative Ha_P_Ha is prespecified additional
consequence value, cannot ignoreaction-insensitive P_Ha to rescue the chain.

OverallPROMISING only adequate support+A+B+C, elseUNPROMISING for the fixed
contract if adequate implementation/support and observedgatesfail. If key
fit remains improving>5% in last200updates or support/implementation fails,
overallUNCLEAR with per-link failures preserved. No posttest epoch/signed/seed
rescue. EvenPROMISING is not identifiedmediation, same-state selection or
matched trainedpolicy Cm utility.

## Deferred evidence

Singlefit/sourceactorcohort; initialHcompression, sourcepolicy feedback and
noise maylimit predictability. ContactI usesaggregate netforce/proximity,
not pairedslip/friction. Onehotdoesnot permit arbitraryunseenactiongeneralization.
Finite sample bootstrap doesnot coverfitseedvariation. Future conditional
predictability improvements requirea new Decision and fixed protocol, not
an unbounded continuation. Validation and policyintegration remain deferred.


## Pre-main engineering support record

46 Task tests pass. Frozen ref8 split audited before ANY fit: OOFhold env
42/42/41, eachOOFsource arm≥17. Main outertest treatmentrank14. Each
opposite-half outertest has83rows; one lacksindexplus, adjusted14arm ranks
13/9 because nuisance support is too sparse. Retain eachhalf E/I prediction
MSE and sameH model candidate vectors, but set adjustedGT contrast metrics
NULL/UNCLEAR when rank<14; pseudo-inverse coefficients are not identified
individual effects. Do not add data/change split or silently use those
coefficients as evidence. Main gates rely on full outertest; crosshalf
contrast limitation is pre-run, not post-outcome rescue.
Pipeline additionally uses physical supervision and predictor compute; matched
162D downstream capacity does not match total pipeline compute/capacity.

## 完成结果与判断

实际采集仍为 ref7 的 `a81b36d` 数据；本轮运行代码 `5db4f08`，
run `conditional-consequence-s237`。21 次固定 1500-epoch 拟合全部完成：
3 个完整预测器、9 个 OOF 预测器、2 个跨半包预测器、7 个任务 scorer。
沿用 ref8 的 688/166 窗口、125/31 环境划分，854 个窗口和 193 个早期失败全部保留。
预测器各 10,554 参数、scorer 各 12,776 参数，同阶段初始化一致；没有按 test 选择权重。

**总标签 UNCLEAR，预设 A/B/C 均未过。** 原因是预测泛化、动作差异保留和额外任务价值
都不足，且预先规定的拟合限制仍触发。不是实现失效，也不否定动作效应或 Cm 核心假设。

### A：动作条件预测

| 预测器／测试动作 | E12 normalized MSE | I14 normalized MSE | 联合 MSE |
| --- | --- | --- | --- |
| H | 1.580799 | 1.900143 | 1.752754 |
| Ha | 1.909467 | 1.738606 | 1.817465 |
| Ha_test_shuffled | 1.815989 | 1.946551 | 1.886291 |
| Shuffled | 2.027548 | 1.854938 | 1.934604 |

Ha 相对 H 的 I 误差改善 **8.50%**，环境 bootstrap 95% 区间为 −10.71–23.58%，
下界未过零；E 误差反而增加 **20.79%**，联合误差增加 3.69%。相对训练时打乱动作的
预测器，Ha 的 I 改善 6.27%，区间也跨零。因此 A 不通过，不能只用 I 的点估计宣布成功。

但冻结 Ha 预测器并置换 TEST 动作，I 误差增加 **11.96%**，区间 **4.17–22.24%**。
这个局部正向证据说明模型使用了动作；它不等于可靠地预测全部动作后果。
E 在置换后点估计更好，说明动作通道对各物理头的帮助并不一致。

所有 permutation 都限制在各自 fit／hold／test 和 wave×motion×phase-quarter 内，
不改变 factual 状态与标签。完整源／test 的标签改变率为 84.16/72.89%；三折分别为
80.22/78.97%、81.13/70.93%、80.65/67.54%，均超过预设 60%。训练打乱和冻结后测试打乱
是两个不同对照，均保留。

### B：逐臂后果差异

主读数是在同一批 166 个 H 上改变 intended arm，再取 predicted candidate−zero。
GT 是同一测试集的 current-only 调整后 factual arm−zero；没有逐状态 GT 反事实标签。
GT nuisance rank93、treatment rank14、残差 df59、条件数约7.45，数值上可识别；
但 zero 只有7个窗口，所有对照共享这个有噪声的参考，精度有限。

| 预测器 | I contrast 相关 | 方向一致率 | 相对零预测的 MSE 改善 | RMS 幅度比 |
| --- | --- | --- | --- | --- |
| H | NA | 0.00% | 0.00% | 0.0000 |
| Ha | 0.2927 | 62.29% | 2.22% | 0.3689 |
| Shuffled | 0.1393 | 74.29% | 5.27% | 0.4669 |

Ha 的 I 相关0.29275、方向62.29%、误差改善2.22%，分别未达到 .50/.65/.10；B 不通过。
E 的相关 −0.0240、误差改善 −35.83%。无动作输入的 H candidate−zero 精确为零，连线符合预期。
全部14臂的 E12/I14、有符号差分、单位、factual prediction 差分和幅度指标均导出，见下表及 JSON。

跨半包使用源半的 outer-TRAIN 拟合，预测另一半的 outer-TEST，始终保持环境隔离：

| 源采集半包 | 目标 test 窗口 | E12 MSE | I14 MSE | GT 对照 rank | 逐臂 GT 对照 |
| --- | --- | --- | --- | --- | --- |
| 0 | 83 | 1.575567 | 1.756718 | 13 | NULL / UNCLEAR |
| 1 | 83 | 2.670072 | 2.135412 | 9 | NULL / UNCLEAR |

缺臂和低 rank 已在运行前记录，因此只报告预测误差与模型 candidate 向量，不解释完整
14 臂 GT 差异。原 diagnostic 中的 `crosshalf_GT` 是未识别的伪逆数组，不可当作臂效应；
报告导出时将它们抑制为 NULL。没有通过补数据或换 split 补齐证据，也没有宣称跨半包
动作差异已保留。未见类别的 one-hot held-arm 外推明确延后。

### C：OOF 后果的任务价值

三折源／held 环境为83/42、83/42、84/41，每个源集都覆盖15臂，最少24/17/23窗口。
每折 PCA、feature 和 target 尺度仅用源集拟合。OOF 输出先还原物理单位，再转外层尺度；
下游训练行全部用 held-fold 预测，test 行用完整 outer-TRAIN 预测。
所有 scorer 训练一次后冻结评价，保留直接 Ha 和 Ha+predicted 后果对照。

| 任务输入 | Primary test MSE | Physical-failure MSE | Failure AUC | Primary train MSE |
| --- | --- | --- | --- | --- |
| H | 0.793056 | 0.792324 | 0.8548 | 0.001082 |
| Ha | 0.647753 | 0.736756 | 0.8396 | 0.000322 |
| GT_HEI | 0.408042 | 0.477182 | 0.9333 | 0.000247 |
| P_H | 0.799347 | 0.928069 | 0.7974 | 0.000471 |
| P_Ha | 0.673008 | 0.766781 | 0.8322 | 0.000257 |
| P_Shuffled | 0.610808 | 0.611305 | 0.8939 | 0.000257 |
| Ha_P_Ha | 0.599499 | 0.596108 | 0.8931 | 0.000143 |

P_Ha 相对 H 的主误差改善15.14%（95%：1.39–28.26%），但相对直接 Ha **更差3.90%**
（gain 区间 −25.05–13.47%），额外价值未建立。Ha_P_Ha 的改善点估计7.45%，区间
−7.15–20.54%，仍不足。P_Ha 相对 P_H 改善15.81%，区间也跨零。
P_Shuffled 的点估计0.61081反而优于 P_Ha0.67301；不能选它救援 Cm 主结果。
冻结 Ha 后置换动作使 P_Ha 任务误差点估计增加9.21%，区间跨零。

本轮同预算 GT oracle 仍强：H0.793056→GT0.408042，改善48.55%，区间37.32–58.81%。
使用同预算分母，oracle gain retention **R=0.31180**，即31.18%；区间 **2.90–56.56%**，
2000次 bootstrap 分母均为正。R 单项通过，但不能替代 direct-Ha 额外价值门槛，C 未过。
ref8 的旧200epoch结果保留作参考，没有用旧弱基线归一新模型。

将预测后果直接代入冻结的旧 GT scorer，P_H/P_Ha/P_Shuffled 主误差为
1.00770/0.89431/1.02538，均弱于旧 H0.73135；P_Ha 的旧 oracle R 为−0.48177。
另一次 GPU replay 把原 GT 输入送回旧 scorer，与 ref8 保存预测最大差仅2.38e−7。
所以 plug-in 失效不是 H/PCA 基或尺度漂移；预测质量／输入分布变化会影响 oracle scorer。
OOF-trained scorer 的结果不同，二者不能混报。

| 预测器 | OOF-train E MSE | OOF-train I MSE | Full-fit test E MSE | Full-fit test I MSE |
| --- | --- | --- | --- | --- |
| H | 2.058599 | 2.011352 | 1.580799 | 1.900143 |
| Ha | 2.273091 | 1.843603 | 1.909467 | 1.738606 |
| Shuffled | 2.366574 | 1.993274 | 2.027548 | 1.854938 |

OOF/full 质量差异不是泄漏，但可能限制任务迁移；逐轴均值／std 也已保存。
下游同容量不能控制整条管线额外的物理监督、模型和计算。所有8头误差、factual ranking
和 ≥10 pair 分层分母均在 result.json；它们不是同状态候选动作 ranking 或策略收益。

## 反常结果审查与 root 归因

按用户要求，`ref5_engineering_review` 对实际反常结果做了只读独立 review。
它逐项核对 step8 E/I、step9..32 Y、跨 step8 的 lost-contact、OOF 环境与尺度、
所有动作排列、candidate−zero、36组 gain/CI、OLS 和 R，未见关键实现缺陷。
root 还在 GPU 重放全部保存的 predictor、scorer 和旧 scorer：预测及 candidate 数组
精确一致，独立 joint-design OLS 差1.36e−12，bootstrap一致。
数据、原权重和主结果未修改，没有实现问题需要补跑。

反常结果后的无拟合诊断（不改主 gate）：

| 固定基线 | E12 MSE | I14 MSE | Joint MSE |
| --- | --- | --- | --- |
| 训练集均值 |0.753523|0.890693|0.827384|
| E=0 / I保持before |0.882805|1.045766|0.970553|

两者均明显优于全部 MLP 预测器。完整 Ha 的 train MSE0.06530、test1.81746；
任务 scorer 的 train primary 接近零而 test 仍大。1500epochs 后完整 Ha 的末200步
损失仍降6.19%、OOF降5.8–9.7%、下游降12.7–34.0%，但绝对训练残差已经很小。
预设 fitting-limit 规则触发 UNCLEAR；**训练 loss 下降不能证明再加 epochs 会改善测试**。
证据更指向当前拟合的泛化问题，而非直接支持“动作效应不存在”。

训练打乱模型的原始方向一致率74.29%也不能解释成它恢复了更好的物理规律。
GT I 对照约71.55%能量来自跨臂共同偏移，和共享 zero 的不确定性一致。
事后逐轴去除跨臂均值，仅作诊断：Ha I相关0.19766／方向60.27%／gain−4.53%；
Shuffled相关−0.07562／方向43.15%／gain−52.18%。独立 reviewer 复算一致。
这个敏感性解释了原始 sign 指标为何可能被共同参考偏移抬高，不能救援原 B 门槛。

**Root Decision Note：** 当前模型有 I 动作敏感性，但这套固定 PCA／类别动作／MLP1500
不能提供可靠的动作差异保留和额外任务收益。结束本次固定拟合，不启动 selector/PPO，
也不继续加 epoch、换 seed／width／signed 表示救援。保留 GT 预后和真实 action→I 证据。
下一项有决策价值的问题应是：受控的泛化训练（仅用训练内环境 validation 的正则化，
或更有约束的物理 innovation）能否改善对直接 Ha 和简单物理基线的比较。
该新协议本轮未执行；不是关闭所有 conditional-prediction、E/I 或 Cm 研究。
最终训练所得 matched Cm-on/off policy utility 仍 OPEN。

## 产物、资源与范围核对

GPU6 主运行59.99s，合成工程 smoke3.93s；权重审计仅 inference、各次 timeout60s，
累计明显低于660s。主模型 peak allocated 71.14MiB，新增产物<12MiB。
全部进程结束，GPU6空闲。46项 Task tests 和 scoped repository verification 通过。
没有仿真、checkpoint 覆盖或正式 Validation；用户 guidance 文档保持未追踪。

输出：`outputs/cm-interaction-oracle/conditional-consequence-s237/`。
manifest/result/fit_records/diagnostic 保存21套权重、OOF来源／尺度／排列／数组；
engineering_audit/engineering_replay/independent_engineering_review 保存审查；
frozen_oracle_alignment_replay 保存旧 scorer 坐标系复验；anomaly_diagnostics 保存
固定基线与中心化诊断；per_arm_contrasts JSON/MD 保存完整14臂有符号 E12/I14 和单位；
finger_amplitudes JSON/MD 保存逐指幅度；`conditional_consequence.png` 为独立图，
灰色虚线是 train-mean I 基线。旧 ref7/ref8 输出不变。

Ref9 要求已逐项覆盖：H/Ha/shuffle 和 frozen testshuffle；E/I/联合与逐臂方向／幅度；
冻结 split、主表示和 intended action；signed sensitivity 明确延后；严格 fold-only OOF；
直接 Ha 与 Ha+pred；同预算 R；不作 mediation 结论；保留早失败；固定更充分训练预算；
实际反常结果 review。跨半包预测已完成，GT 差分不可识别部分明确记为证据不足。

## 逐臂 GT／预测有符号差异


GT=current-state-adjusted factual randomized arm-minus-zero contrasts in166test windows. Pred=sameH mean model candidate arm-minus-zero; no individual GT counterfactual labels. All26axes and factual prediction contrasts remain in JSON.

| Arm | GT dz mm | Pred dz mm | GT own logforce | Pred own logforce | GT own distance mm | Pred own distance mm | I correlation | I amplitude ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| index_plus | -58.94 | -13.13 | -0.529 | -0.011 | +29.49 | +13.03 | 0.561 | 0.344 |
| index_minus | -12.90 | -0.88 | +0.041 | -0.130 | +19.64 | -2.43 | -0.290 | 0.292 |
| middle_plus | -15.23 | +0.61 | +0.773 | -0.142 | +22.01 | +1.80 | -0.349 | 0.274 |
| middle_minus | -9.85 | -18.12 | -0.030 | -0.344 | +24.73 | +19.94 | 0.432 | 0.600 |
| pinky_plus | -24.40 | -8.60 | -0.629 | -0.122 | +34.20 | +5.16 | 0.505 | 0.393 |
| pinky_minus | -10.24 | -5.40 | -0.440 | -0.133 | +25.40 | +11.96 | 0.347 | 0.506 |
| ring_plus | -30.61 | -15.09 | +0.972 | -0.229 | +28.44 | +3.63 | 0.265 | 0.349 |
| ring_minus | -21.65 | +13.56 | +0.540 | -0.331 | +32.87 | +4.58 | -0.049 | 0.260 |
| thumb_yaw_plus | -19.71 | +0.63 | -0.170 | +0.062 | +16.22 | +2.81 | 0.165 | 0.285 |
| thumb_yaw_minus | -26.89 | -2.60 | -0.340 | -0.134 | +20.07 | +3.89 | 0.368 | 0.358 |
| thumb_pitch_plus | -19.89 | +1.63 | +0.110 | +0.021 | +4.35 | -0.36 | 0.621 | 0.285 |
| thumb_pitch_minus | +49.41 | +2.99 | -0.775 | -0.230 | +20.96 | -1.94 | 0.062 | 0.562 |
| synergy_plus | -19.86 | -10.35 | +0.636 | -0.170 | +17.27 | -4.52 | 0.199 | 0.327 |
| synergy_minus | -9.01 | -5.18 | -0.513 | -0.395 | +55.09 | +19.16 | 0.639 | 0.346 |

## 逐指物理扰动幅度复审

复用同一 ref7 数据，没有新动作采集。以下保留 commanded／actual 与单位区别；
原始 min/mean/max 和所有12 native joints、5 true tips 在 JSON。

## Per-finger perturbation amplitude audit

Units: native action dimensionless; PD targets/actual q radians; tip displacement mm.
PD target changes are commanded, not actual executed joint angles. Driver table order: index, middle, pinky, ring, thumb-yaw, thumb-pitch. JSON PD/action arrays use all12 native joints6..17, including followers.

| Driver | Native index | Physical range (rad) | Native mimic followers |
| --- | --- | --- | --- |
| index | 6 | 1.6000 | 7: ×1.05 |
| middle | 8 | 1.6000 | 9: ×1.05 |
| pinky | 10 | 1.6000 | 11: ×1.05 |
| ring | 12 | 1.6000 | 13: ×1.05 |
| thumb_yaw | 14 | 1.1500 | none |
| thumb_pitch | 15 | 0.5500 | 16: ×0.6;17: ×0.8 |

### Delivered PD target magnitude per arm

Signed offsets per active step, not cumulative K-times angles. Finger targets are absolute native PD targets with a per-step baseline offset. Min/mean/max per joint, degrees and range fractions are preserved in `finger_amplitudes.json`. Composite arms exclude thumb-yaw. Clip % counts active commanded finger coordinates.

| alpha | Arm | n | index rad | middle rad | pinky rad | ring rad | yaw rad | pitch rad | clip % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | zero | 56 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | index_plus | 49 | +0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | index_minus | 56 | -0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | middle_plus | 51 | +0.00000 | +0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | middle_minus | 63 | +0.00000 | -0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | pinky_plus | 72 | +0.00000 | +0.00000 | +0.32000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | pinky_minus | 46 | +0.00000 | +0.00000 | -0.32000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | ring_plus | 60 | +0.00000 | +0.00000 | +0.00000 | +0.32000 | +0.00000 | +0.00000 | 0.00 |
| 1 | ring_minus | 55 | +0.00000 | +0.00000 | +0.00000 | -0.32000 | +0.00000 | +0.00000 | 0.00 |
| 1 | thumb_yaw_plus | 49 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.23000 | +0.00000 | 0.00 |
| 1 | thumb_yaw_minus | 54 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | -0.23000 | +0.00000 | 0.00 |
| 1 | thumb_pitch_plus | 78 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.11000 | 0.00 |
| 1 | thumb_pitch_minus | 42 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | -0.11000 | 0.00 |
| 1 | synergy_plus | 62 | +0.32000 | +0.32000 | +0.32000 | +0.32000 | +0.00000 | +0.11000 | 0.00 |
| 1 | synergy_minus | 61 | -0.32000 | -0.32000 | -0.32000 | -0.32000 | +0.00000 | -0.11000 | 0.00 |

For composite ± at a20% driver range dose, intermediate followers6→7 etc receive ±0.336rad (10.70% of their3.14rad range); thumb-pitch followers receive ±0.066/0.088rad (2.10/2.80% of their ranges). At5% driver dose these are divided by4. Yaw has no target coupling.

### Measured native driver response at step8

Current-state-adjusted arm-minus-zero contrasts of actual q(step8)−q(before), radians. Not PD targets.

| Arm | index | middle | pinky | ring | yaw | pitch |
| --- | --- | --- | --- | --- | --- | --- |
| index_plus | +0.12875 | +0.03508 | +0.01204 | +0.00308 | -0.00562 | -0.00613 |
| index_minus | -0.15809 | -0.02648 | -0.00266 | +0.00306 | +0.00053 | -0.02202 |
| middle_plus | +0.05977 | +0.16934 | +0.02540 | +0.06805 | -0.00172 | -0.01241 |
| middle_minus | -0.03351 | -0.14832 | +0.00320 | -0.03896 | +0.00427 | -0.00623 |
| pinky_plus | +0.01215 | +0.02435 | +0.26734 | -0.00121 | +0.00015 | -0.00239 |
| pinky_minus | -0.00205 | +0.01421 | -0.26966 | -0.03733 | +0.00081 | +0.00415 |
| ring_plus | +0.00987 | +0.04957 | +0.01291 | +0.23303 | -0.00184 | -0.00111 |
| ring_minus | -0.00517 | -0.00878 | +0.00370 | -0.22309 | +0.00526 | -0.00427 |
| thumb_yaw_plus | -0.02100 | -0.01908 | -0.00647 | -0.01102 | +0.21367 | -0.00504 |
| thumb_yaw_minus | +0.00048 | +0.03201 | +0.00474 | +0.00392 | -0.20989 | -0.00512 |
| thumb_pitch_plus | -0.02185 | +0.00034 | +0.00140 | -0.00016 | +0.00352 | +0.08702 |
| thumb_pitch_minus | +0.01170 | +0.02278 | +0.00628 | -0.00391 | +0.00023 | -0.10148 |
| synergy_plus | +0.20111 | +0.25448 | +0.29047 | +0.25572 | -0.01000 | +0.08441 |
| synergy_minus | -0.21090 | -0.23692 | -0.30172 | -0.29002 | +0.00434 | -0.12145 |

### Measured five-finger tip response at step8

True tip positions in the measured hand-base frame. Values are norms of adjusted displacement vector contrasts (mm), not differences of average travel. Raw within-arm travel means are in JSON and include ordinary baseline evolution.

| Arm | index mm | middle mm | pinky mm | ring mm | thumb mm |
| --- | --- | --- | --- | --- | --- |
| index_plus | 21.69 | 2.43 | 0.28 | 0.12 | 1.05 |
| index_minus | 24.38 | 2.07 | 0.14 | 0.11 | 6.85 |
| middle_plus | 4.03 | 25.71 | 0.80 | 4.12 | 2.96 |
| middle_minus | 2.42 | 26.85 | 0.13 | 2.70 | 1.34 |
| pinky_plus | 0.80 | 1.61 | 15.19 | 0.05 | 0.48 |
| pinky_minus | 0.11 | 0.58 | 17.56 | 2.66 | 1.68 |
| ring_plus | 0.74 | 3.27 | 0.45 | 27.62 | 0.96 |
| ring_minus | 0.45 | 1.05 | 0.37 | 29.06 | 0.35 |
| thumb_yaw_plus | 1.55 | 1.74 | 0.21 | 0.92 | 9.33 |
| thumb_yaw_minus | 0.09 | 1.65 | 0.19 | 0.21 | 9.87 |
| thumb_pitch_plus | 1.48 | 0.26 | 0.27 | 0.42 | 10.15 |
| thumb_pitch_minus | 0.73 | 1.44 | 0.26 | 0.18 | 13.94 |
| synergy_plus | 25.86 | 30.97 | 15.86 | 29.79 | 8.58 |
| synergy_minus | 27.79 | 32.82 | 18.92 | 33.61 | 19.81 |
