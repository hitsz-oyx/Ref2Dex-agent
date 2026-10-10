---
schema: ref2dex.probe.v2
probe_id: P-20261010-retargeter-local-motion
experiment_id: P-20261010-retargeter-local-motion
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: 49013f69b06bee53436f0fc731a730e4a0c94bc7
claim_id: C3
hypothesis_family: HF-trajectory-policy-retargeter
probe_index_in_family: 3
seed_pool: probe
seeds: [293, 296, 297, 298]
decision_changed_if_positive: collect competent native data and evaluate the learned local-motion retargeter in physics
decision_changed_if_negative: inspect inverse conditioning and native execution semantics before another architecture change
status: UNCLEAR
run_id: retargeter-local-motion-20261010-r1
---

# Can action-aligned local hand motion improve native inverse precision?

## Motivation / Decision Note

服务Mission A/ref1_1 learned τ+state→A路线；Cm-on/off真实训练收益仍是最终要求。
首个generic Transformer fit有τ信息，但val first-PD XYZ p9549.71mm，train也38.78mm；
同数据局部标定的手点恢复准确，但简单affine inverse仍32.6mm。两个诊断均UNCLEAR，
不定位旧R为主故障、不把小数据负结果当作架构不可行。
分类Decision：检验generic action query需要从attention提取local motion，是否限制
当前1500updates预算内的精度。最便宜方法是同数据/算法的有界局部条件分支。

Root选择一次固定训练双臂，只有τ条件提取结构改变：加入每动作query的对齐motion。
若原screen通过才进入competent采集/物理；若未过先审查conditioning/动作执行语义，
不增加epochs/数据量或继续盲目换模型。条件分支的参数增加是架构干预的一部分，
不能把改善归因于alignment单因素；本轮不是正式架构Validation。
预算一张GPU5/8GPUmin/64MiB，fit360s+独立audit120s，无新物理/RL/WM/teacher数据。
不改Mission/claim、不新建分支或push，无新外部授权需求。

用户追问已有研究后补充去重前置：旧Task已有actualτ+q/dq的2层Transformer与
structured96/64/64episodes，不能把本轮描述为首次learned R。
先只读旧v1 frozen checkpoint，在新Task同16576 validation窗口重算首8 active
native误差/PD，<=60s/4MiB。旧v1无previous action/force输入，符合当前边界；
旧v2含previous action，不作为等输入对照。先检查旧v1代码关键class与其Git snapshot
一致及checkpoint SHA，再决定是否需要local-motion训练；不重新启动旧Task。
若继续fit，其deadline收紧到300s，基线60s+fit300s+audit120s仍共用8GPUmin/64MiB。
旧模型训练数据/预算不同，只作reuse baseline，不作matched architecture结论。
详情见Task research `20261010-retargeter-prior-work.md`。

## Fixed architecture / contract

保持current s87和24x33实际future hand displacement τ，current object frame。
不新增future q/object、phase/clock、force/tactile或动作历史。desired τ是action interface，
future actual τ只用于监督诊断，不称为可部署预测意图。
保留原width128/2layers/4heads/dropout0 Transformer及state/trajectory memory。
每个action slot k=0..7的新query为原learned query + MLP(local66)，其中
local66=[τ[k], τ[k]-τ[k-1]]/.01m，τ[-1]=0=current手点；对应first native A[t+k]
与hand[t+k+1]对齐。MLP66→128 GELU→128，不引入固定PD系数或显式IK。
head仍零初始化输出train action均值；原网络公共参数在seed296下初始化一致。
state-only臂保留相同模型容量，把global normalizedτ与local motion同时置零。
相同normalizer/init/minibatch/优化器/更新数，所有网络梯度仍可回到τ。

## Fixed data / training / metrics

沿用前一inverse Probe两源low/inventory绑定，wave0--3全部16env train、wave4
全部16env validation，partial排除，66304/16576窗；旧teacher packet禁止训练。
相同seed296、CUDA FP32 highest、batch256、1500updates、Adam3e-4、gradnorm1。
train-only mean/std以及floor保持s/τ .001、A .01；loss8x12 standardized L1。
每250updates按每臂自己的val L1选best，唯一snapshot+最后一次best symlink。
同tick跨env derangement297/298 τshuffle，两臂模型仅forward结构不同于原版。
保留原base learned_retargeter.py、retargeter_data.py、fit_learned_retargeter.py，
不改变旧fit输入SHA；新入口fit_local_retargeter.py固定同algorithm。

沿用原screen：τ L1<=.8state-only且shuffle>=1.1τ，all/startup first-PD
XYZ p95<=5mm、rotation/fingers<=.05rad为PROMISING；τ不优于state-only或
shuffle增加<=2%为UNPROMISING；否则UNCLEAR。报告原版与本版误差/耗时/参数数目。
独立audit入口按checkpoint schema选择loader，其数据/标签/FP64 statistics/PD/
val replay/selection/screen仍独立重构，固定容差不变；新版base/local源码均绑定SHA。
训练/审计都hash-end guard；审计PASS后才解释结果，无物理成功claim。

## Resources / stop

GPU5 preflight<=512MiB/util<=10%，训练时每250updates记录util/memory/ETA，
foreigncompute>512MiB停止。fit<=360s/audit<=120s，新增8GPUmin/64MiB，总<=4GPU/
300GiB。非有限、source漂移、split/control错误、GPU冲突立即停止保留FAILED，
不自动重训/补seed/调参，无新physics/旧Task/WM实验。
产物`outputs/trajectory-policy/retargeter-local-motion-20261010-r1/`；
审计`outputs/trajectory-policy/retargeter-local-motion-audit-20261010-r1/`。

## Results

Result: `UNCLEAR` — local-motion improves offline τ usage, but the fixed native-precision screen still fails.

`49013f6` fixed fit completed in 40.02s on GPU5 (Torch peak 58.65MiB; monitor
reported 1025MiB device usage and no foreign compute). The τ arm and matched
state-only arm both selected update 1500. Validation standardized L1 was
`0.241886` versus `0.332077` (ratio `0.7284`), and same-tick τ shuffle was
`0.519620` (shuffle/full ratio `2.1482`). Thus the aligned branch uses the
provided local motion signal under the fixed data contract.

The physical-error screen did not pass: τ all/startup first-command XYZ p95 was
`44.15/51.21 mm`, rotation `0.3688/0.1101 rad`, and fingers
`0.1019/0.1210 rad`. The matched state-only values were
`58.09/74.47 mm`, `0.3974/0.1702 rad`, and `0.1174/0.1591 rad`.
The independent audit completed in 5.55s with `PASS`; requested/applied labels,
native PD reconstruction, split, train-only statistics, checkpoint selection,
all saved validation predictions, and shuffle mapping were independently
reconstructed within the fixed tolerances.

This is a positive offline conditioning signal but not a deployable retargeter,
physical grasp result, or Cm utility result. Since the fixed physical screen did
not pass, do not collect competent native data or enter physics with this
checkpoint. Inspect inverse conditioning and native execution semantics before
another architecture change; do not add epochs, seeds, data, or physics in this
Probe.

Fit output: `outputs/trajectory-policy/retargeter-local-motion-20261010-r1/`.
Independent audit: `outputs/trajectory-policy/retargeter-local-motion-audit-20261010-r1/`.

Decision: retain the local-motion evidence as `UNCLEAR`, close this fixed Probe,
and use a separate bounded diagnosis to distinguish command-identification error
from native execution/preload semantics before deciding whether another learned
retargeter intervention is justified.

## Limitations / future evidence

只有单motion/同initial/失败多的PPO轨迹，validation参与checkpoint selection，
不是独立test。inverse多解/接触约束/hidden solver仍可能影响native动作复现。
local branch同时增加参数与改变alignment，Probe不单独定位其因果成分。
即使screen通过，competent coverage、predicted τ鲁棒性、闭环完整持有/drop及
joint forecaster/action、Cm-on/off策略训练收益仍分别需要实际证据。
