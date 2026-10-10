---
schema: ref2dex.probe.v2
probe_id: P-20261010-learned-retargeter-inverse
experiment_id: P-20261010-learned-retargeter-inverse
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-trajectory-policy-retargeter
probe_index_in_family: 1
seed_pool: probe
seeds: [293, 296, 297, 298]
decision_changed_if_positive: collect bounded competent native data and evaluate the learned inverse in physics
decision_changed_if_negative: diagnose labels or inverse conditioning before more data or physics
status: UNCLEAR
run_id: learned-retargeter-inverse-20261010-r1
---

# Can actual future hand motion supervise a trajectory-dependent native retargeter?

## Motivation / Decision Note

服务于Mission A和用户ref1_1 learned retargeter路线；Mission B matched Cm训练
收益仍后续必需。当前GT/dense经旧R有效而高层未可用，不据此定位R为主故障。
具体猜想是：真实future hand比仅当前state为native action提供可学习的额外信息，
小型Transformer可以在当前预算内利用它，先支持进一步物理/competent采集投入。
分类Decision。最便宜方法是已存actual hand/applied action配对的匹配离线训练，
不用fork、不新增高层PPO、teacher action标签或完整PointWAM。

Root选择一次固定1500update双臂fit及独立audit。正向才补competent data和GT/predicted
τ物理评价；负/不清楚先核对inverse/标签链，不根据单次fit关闭τ架构或复制大模型。
在当前用户持续主线授权内，不改Mission/claim、不新建分支，无新权限需求。

## Data / fixed split

只读两轮trajectory PPO `trajectory-ppo-20261010-r1` 和
`trajectory-ppo-long-credit-20261010-r1` 的low；二者原采集seed293。
先验证 `retargeter-data-contract-20261010-r1` inventory SHA；旧成功teacher packet
`engineering_only=true/training_allowed=false` 不打开、不作为输入标签。
每源完整reset wave0--3全部16env用于train，wave4全部16env为selection/validation，
partial wave5排除。两源同split，避免它们前128相同样本或平行近似前缀跨split。
训练66304窗口/128environment episodes，validation16576窗口/32episodes。
同seed/初始化以及大量滑窗重叠不等于独立示范，validation参与checkpoint选择，
没有独立test或正式generalization claim。无根据held成功状态过滤行。

输入current s87 = actualq18+dq18+当前object frame内hand33+当前object3x4+objvel6。
τ24x33是**actual** hand[t+1:t+24]减当前对应hand，再转当前object rotation frame。
没有future q/object、reference、phase/clock、action history或触觉输入。
标签为actual applied native command[t:t+7]的12active坐标；passive6列推理恢复0。
请求/applied逐元素一致；禁止跨reset或补缺失末帧，future actual与proposal不可混。

## Matched models / interventions

两个相同初始化的小型Transformer decoder：width128、2blocks、4heads、24trajectory
step tokens+1state token、8action queries，dropout0。train-only归一化：sstd floor.001、
τ各horizon/point/coord std floor.001m、A各step/coord std floor.01 native units。
线性头初始化0，首先输出train动作均值。normalization与参数初始化完全相同。
一臂使用τ，一臂zero normalizedτ输入，仅state提供观测；保留同参数容量。
相同每update256样本、Adam3e-4、gradnorm1、1500updates，初始化/采样seed296。
GPU highest FP32，2threads。每250updates两臂按各自最低validation standardized L1
保存新唯一snapshot，末尾best只是一次性symlink，不覆盖既有checkpoint。

loss为全部8步12active command的standardized L1。推理恢复native units后clamp[-1,1]，
仅报告raw越界比例，不把这种输出clip与旧R的PD目标裁剪混为同一语义。
选定τ model固定validation τshuffle：在每源同一tick的16不同episode间作derangement，
seed297/298，只换完整actual displacement τ，不换currentstate或action标签。
保留donor mapping和每个window source/tick，以排除“只打乱了phase分布”的解释。

## Metrics / fixed exploratory screen

全部held-out8步standardized L1，以及第一native control的clipped command MAE。
PD target误差由actual native scale换算：wristΔA*scale，fingerΔA*scale/2；
因samecurrentstate差分，offset/currentq抵消。分别报告all和startup tick0:63，
每窗口XYZ/Eulerrotation/6activefingers的最大绝对误差p95。

机制PROMISING仅当τmodel L1<=.8state-only、同tick shuffle L1>=1.1τmodel，且
all/startup first-control p95均XYZ<=5mm、rotation<=.05rad、finger<=.05rad。
UNPROMISING若τmodel不优于state-only或shuffle损失增加<=2%；其余UNCLEAR。
这是有界离线screen，不是可用抓取controller或正式因果证明。
独立audit从原low重建全部validation s/τ/A和shuffle mapping，用保存checkpoint重放
全部预测与PD换算/指标、检查train-only statistics/split/sourcehash/推理passive列。
审计入口 `tools/audit/audit_learned_retargeter.py` 不调用训练的feature/window构造。
独立NumPy FP64 train statistics与保存normalizer绝对误差<=1e-5；全部validation
预测重放<=2e-5、实际native PD<=1e-6、重新计算指标<=1e-5；超出即审计失败，
不根据结果放宽容差。checkpoint选择也与六条固定monitor记录独立核对。

## Resources / stop

GPU4启动前已有其它计算，记录并选择GPU5：当前util0/memory137MiB，只有已有图形
初始化context约130MiB，无活跃模型计算；不影响他人进程。默认preflight<=512MiB/
util<=10%，训练期间foreigncompute>512MiB停止。仅用一张卡，全局<=4GPU/300GiB。
新增8GPUmin/96MiB，fit<=360s、audit<=120s。GPU网络与batch，CPU文件/窗口/统计；
记录250update utilization/memory/ETA，异常及时报告。原数据/源checkpoint只读。
超时、非有限、输入/模型源漂移、GPU冲突、标签/applied或split错误立即停止保留FAILED，
不自动重训，不增加epochs/seed/hyperparameter sweep或新physics/WM预算。

## Results

Not run yet; fixed protocol before training.

## Limitations / future evidence

仅failure-rich小数据、单motion/同initial/seed，完整稳定held训练数据极少。
Inverse可能多解，当前τ在teacher当前策略延续条件下发生，24步与前8动作不必唯一。
τshuffle sensitivity不等于正确控制不同τ；GT实际τ不等于predicted部署意图。
后续competent coverage、不同可执行τ控制响应、GT/predicted τ完整持有/drop、实测
actionloss到forecaster及Cm-on/off训练收益仍需各自有预算的Probe/Validation。
