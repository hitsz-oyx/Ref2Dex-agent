---
schema: ref2dex.probe.v2
probe_id: P-20261015-inspire-action-representation
experiment_id: P-20261015-inspire-action-representation
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: 076b5c31
claim_id: C1
hypothesis_family: HF-inspire-action-representation
probe_index_in_family: 1
seed_pool: probe
seeds: [195, 192, 148]
decision_changed_if_positive: prioritize original action parameterization and dissect geometry loss and normalization
decision_changed_if_negative: prioritize teacher input sufficiency and fitting coverage before attributing failure to q_PD
status: RUNNING
run_id: inspire-action-representation-20261015-r1
---

# Inspire shared：原生action与q_PD几何编码的配对Probe

Class: Decision。用户目标是定位Inspire shared不能抓取、Shadow可抓取的原因，明确提出
q_PD和原生归一化action差异，并临时授权剩余卡做实验。服务独立执行底座；最便宜判别
是在同一个Inspire任务内固定输入/数据/父模型，只改变动作参数化，避免跨手任务混杂。

## 现有证据与假设

独立CPU审计660序列165677控制，q_PD[t]+q[t]逆映射ACTIVE命令max5.96e-7；66序列
2190窗口geometry首拍roundtrip max5.66e-7，未见粗大单位/clock/旋转branch错误。
两边动作已TRAIN std标准化，不能解释为只有Shadow归一化。首拍XYZ与native平移线性
等价；变化主要为finger current-relative监督、额外SO3旋转3维、loss维度权重和尺度，
以及未来8拍“固定PD目标vs固定增量命令”的监督含义。K1不直接受未来buffer语义影响。
详见根级research/2026-10-15-inspire-shadow-action-representation.md。

假设：使用原生12D独立命令可以减轻动作拟合/闭环的困难；若共同首拍物理误差和闭环
出现明显改善，优先改用native并进一步拆分该表示包；若两边仍差，不能仅归因q_PD，
下一步检验输入完整性（Inspire87D vs teacher1442D，Shadow300D即完整teacher输入）。

## 固定协议

两臂共同remote WAM-v2 step80000、fresh query actor、seed195、同strict补采660，
TRAIN93680/DEV10376、同DEV256 indices/seed192、序列均匀+startup/later平衡采样。
同state87+shared768 token/mask，保留原已知物体goal[t+24]；不加完整teacher输入或接触。
同三LR5e-6/1e-5/5e-5、AdamW wd.01、AMPbf16/TF32、gradient clip1、无GT tau warmup/anchor。
forecast hand/scene normalized MSE和action normalized L1权重均1。

G：8×15 wrist-geometry-continuous-lift.v1原q_PD；N：8×12 native-command.v1原记录ACTIVE
applied命令，ACTIVE=(0..5,6,8,10,12,14,15)，其他六个被原生coupling覆盖的teacher输出
不作为独立自由度。只改变输出表示及其维数/原TRAIN-only动作统计，其他actor trunk相同。
保留既有normalizer floor惯例：native .01、geometry1e-6；所以结果检验表示/监督加权包，
不能单独归因SO3或归一化一项。不得横比两种坐标下action loss绝对值。

每臂单GPUbatch256、2000updates，并行执行，固定终点为主要读出，不按各自best挑结果。
每250评估共同DEV；保留500/1000/1500/2000及best/forecast/control。输出initial-actor.npz，
核对除action mean/scale与输出head形状外的初始trunk张量、state/tau统计、父/输入/panel一致。
Head形状改变消耗不同初始化RNG，故用共同training_rng_seed在audit/训练前重新对齐随机流。
先同batch4-update smoke；正式重新从80k/fresh actor启动，不继承smoke。

主要共同离线指标：首拍原生intended/applied command MAE、PD XYZ mm mean/p95、SO3 rad、
finger rad、clipping及hand/scene ADE mm。native预测通过原生PD映射换算，geometry通过原
freeze/consume解码，两者比较同一控制器。geometry raw lift误差不与native Euler误差混比。

两臂固定step2000进入相同原生三行环境/seed148/30Hz K1，飞机149/小球141/杯子172控制，
沿用P-20261015-shared-native-grasp的三条val和hold45短序列代理。row0 teacher，两student
各自实测H4/87D，重复行不当独立seed。不混teacher命令/GT tau；独立核对原生chunk或
geometry chunk→实际命令、PD、输入、clock和held数。

## 判别与局限

离线方向信号：native首拍XYZ mean和p95均比geometry低>=20%，或applied command MAE
低>=20%且XYZ不恶化>10%。行为信号：teacher正控制可复现的序列上native新增至少1/3
hold45、执行审计PASS。满足则局部PROMISING，优先进一步排除额外旋转/手指坐标/normalizer
及all8目标差异；全部失败或拟合尚不足则UNCLEAR，不在短预算里refute表示路线或数据量。
完整收敛、多seed、matched正式成功率及first-only目标对照延后，不预设训练loss能证明抓取。

## Decision Note / 资源

用户临时明确允许GPU6/7实验；原Inspire0..3和Shadow4/5继续，允许本轮8卡同时占用。
先等待原Shadow2k单次评估释放GPU6，再两臂各占一空卡。每臂训练<=2h、总GPU pipeline<=3h，
每臂产物<=9GB、两臂及native总<=24GB，20GiB磁盘底线。稳定吞吐后更新ETA，预算不静默延长。
外来GPU进程、非有限、输入/初态不匹配、转换审计失败、30min无进展或资源超界停止自己的组。
固定提交/detached工作树，现有训练源码/数据/checkpoint不覆盖；不新增branch。


## 实际启动及独立审查 / 2026-10-15

固定代码076b5c31，detached tmp/inspire-action-repr-code-20261015-r1；PointWorld/PTv3
复用原Inspire训练相同固定依赖。15项相关单测PASS，包括NATIVE动作梯度到shared、
动作target选择、同物理命令的跨表示指标和rotvec/lift不一致反例；workflow verify PASS。
原有GPU0..3和Shadow4/5均未改动。原Shadow2k评估已结束后才取得GPU6/7。

两臂同batch256/4-update smoke均COMPLETED，实际matched-smoke.json为PASS：
除输出head/action统计外的初始actor trunk+state/tau统计完全相同，DEV索引与准入相同，
初始hand/scene loss差均0。正式重新从原80k+fresh actor启动，不继承smoke更新。
Driver2298258拥有两臂pipeline：native GPU6、geometry GPU7，各2000步，之后自动固定
终点三条native执行及独立审计。当前RUNNING、约1.83–1.86s/update、显存约10.4GiB/卡，
预计训练另需约1h；暂无DEV/闭环结果，不横比两组TRAIN动作loss做结论。

数值核对first-XYZ-normalization-audit.json：两组首拍XYZ mean最大差2.33e-10、std差0，
std均[.06822,.07376,.13319]m。即首拍XYZ项的标签和归一化已等价；任何表示包收益应进一步
检查finger编码、额外旋转、多拍监督和共享特征梯度，而非仅称改变XYZ单位。

另已核对split：此前Inspire三条都val；Shadow startup1k held TRAIN3/8、DEV1/8，
expanded2k held TRAIN4/8、DEV1/8。不能混用Shadow TRAIN+DEV与Inspire val结果比较。
train-panel-audit.json提供三条按字典序/同pass动作/有strict startup窗选择的TRAIN候选，
源补采无surface/support/table字段，未伪造源held认证；需要后续teacher正控制。
若两臂val均失败，优先补TRAIN/val资格对照以区分拟合与泛化，再决定输入完整性实验。

中途500步共同DEV：native/geometry首拍PD XYZ mean101.23/94.43mm、finger MAE
0.10145/0.11318rad、applied command MAE0.10351/0.11404。手指及总体命令略好、XYZ略差，
尚不能认为native全面改善；最终仍以固定2000步及后续K1抓取判读，不按此改选点/提前停。


## 并行CPU诊断：forecast相对运动基线

Class: Decision，先判断跨手122.5mm与30.5mm差距有多少来自目标运动幅度。固定此前
Inspire DEV256和Shadow common/startup DEV64，计算零位移与当前速度外推的hand ADE，
与已保存模型指标形成相对比值。若幅度解释绝对误差差距，不再把mm差距当表示质量的
直接证据；若归一化相对基线后Shadow仍明显好，再优先查forecast域覆盖。纯读取数组
和统计，无神经网络forward，使用CPU以免争用已授权的训练GPU；不新增训练/数据准入。

已完成NumPy统计与独立只读复核：

| 同30Hz/未来24拍 hand ADE | Inspire DEV256 | Shadow common DEV64 | Shadow startup DEV64 |
| --- | ---: | ---: | ---: |
| 零位移基线 mm | 279.393 | 63.380 | 141.632 |
| 当前速度外推 mm | 328.020 | 85.248 | 141.632 |
| 已保存模型 mm | 122.517 | 30.466 | 28.449 |
| 模型/零位移 | 0.439 | 0.481 | 0.201 |
| 模型/速度外推 | 0.374 | 0.357 | 0.201 |

Inspire与Shadow common面板的绝对forecast误差差4.02倍，零位移目标幅度差4.41倍；
相对基线质量相近，暂不据mm差距认为WAM表征差4倍。各自面板/任务不匹配，这不证明
两手表征同等好，也不是控制成功率证据。脚本的wrist_motion字段是landmark0掌部运动
代理，不等于腕部PD控制误差。采样身份、24×11覆盖计数、固定query物体坐标的位移范数
不变性均核对通过。输出hand-motion-scale-audit.json；工具compare_hand_motion_scale.py。
