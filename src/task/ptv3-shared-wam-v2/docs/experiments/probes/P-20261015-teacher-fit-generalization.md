---
schema: ref2dex.probe.v2
probe_id: P-20261015-teacher-fit-generalization
experiment_id: P-20261015-teacher-fit-generalization
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: c01f4bfe
claim_id: C1
hypothesis_family: HF-full-teacher-input
probe_index_in_family: 2
seed_pool: probe
seeds: [180, 181, 162]
decision_changed_if_positive: resolve student optimization before replacing teacher reference inputs with tau
decision_changed_if_negative: prioritize cross-sequence coverage and generalization rather than longer fixed-data fitting
status: UNCLEAR
run_id: teacher-fit-generalization-20261015-r1
---

# Teacher输入：拟合程度与跨序列泛化

Result: COMPLETED，UNCLEAR；学习率退火明显改善TRAIN，但dev基本不变。 Class: Decision。用户继续人工审阅660数据，并授权继续对照实验。
服务MISSION独立执行底座；不在审阅过程中动态改变数据、标签或dev，人工标签独立保存。

## 研究问题与最便宜判别

前轮full1442、无tau学生的TRAIN约14mm/dev43mm，小集合509样本可拟合至3mm，
而原teacher重放准入122299 tick误差不到0.001mm。尚不能将完整teacher输入未达20mm
简单归因WAM、信息不足、采集错位或动作随机性。先以原缓存统计误差分布：dev中位
26.54mm、最差5%误差占22%、其余95%仍mean35.31mm；不是仅几个异常值抬高均值。
TRAIN startup mean9.84mm而dev startup51.54mm，需区分拟合残差和跨序列覆盖。
这些是描述统计，不以dev误差删除样本，不构造更容易的新dev。

Decision Probe：同一随机学生、same cached observations/q_PD、same TRAIN/dev512面板，
比较继续优化和降低后期学习率是否能让TRAIN/dev同时改善。比换模型或补采便宜。
四臂：full1442/current721 × constant/cosine_tail；都用teacher固定RMS/clamp输入预处理，
current721第二分支归一化后强制0，无tau。one-step relativePD12与固定live-q decoder，
MLP4107788参数、初始化180/draw181/panel162，原样保留batch512/AdamW3e-4/decay1e-4/
clip1/标准化L1。各臂从scratch运行24000更新；前12000完全相同，后12000一臂保持3e-4，
另一臂余弦退火至3e-6。相同输入两schedule的12k checkpoint逐tensor复核作为干预门。
不是不同seed重复；与旧12k prefix比较只能作为工程一致性，原12k不构成新独立control。
每500评价，保留原p95优先选择；额外保存12k/24k完整学生权重及对应预测以比较同一步。

若退火后TRAIN mean<=5mm且dev仍>20mm，优先判泛化/覆盖为未解问题；若TRAIN/dev
都明显改善并dev<=20mm，局部PROMISING优化路线，再重估full/current输入差异；
TRAIN也未充分拟合时仍UNCLEAR，不声称存在不可消除的信息下界。报告selected与24k
两种视角、mean/p95/median/startup/later，避免事后换选点。有限Probe不升级全局claim。

## 资源、版本和停止

根STATE最新显示远端九源预训练使用4卡，本地仅GPU0/1两卡：full/current配对同时，
constant与cosine_tail分两轮；合计最多6卡，不重复使用已结束的7卡例外。
每臂最多1200s、1GB checkpoint/日志，四臂总5GB上限；不复制旧cache、保留20GiB
可用磁盘。非有限、foreign GPU PID、无界输出、超时停止自己的worker，保留原结果。
运行前commit代码、实验卡/seed；manifest记录真正代码身份和schedule。首500更新观察
GPU利用率/显存/ETA。完成后停止本次优化预算，不自动继续无界训练/模型搜索。

## Limitations / future evidence

单seed、旧dev已反复使用，原teacher RMS为特权预处理；归纳不等于独立验证。
全部label仍来自原coarse筛选数据，人工审阅尚未纳入。本轮只区别一种优化干预，
不独立识别网络容量、损失函数、状态分布或训练数据质量各自贡献；即使TRAIN拟合
很好而dev差，也不能用单实验证明唯一原因。物理执行延后，不拿离线误差冒充抓取。

## 运行完成与结果

运行代码42b052fe（核心模型/退火实现ad66e8b8），supervisor1851796；本地GPU0/1，
远端4卡保留。四个worker全部正常结束，单臂约76–79秒，24k步；源码/数据未因人工
标注改变。每组约2.34GiB显存，GPU采样约10–41%（小MLP，host dispatch及检查亦占时）；
神经计算均GPU。完整前12k模型参数逐tensor相同，通过实际干预对照门，未用不同
初始化解释后半程差异。每组12k/24k和dev改善点权重、预测均保留。

固定面板512、sequence-uniform/半数startup，以下均为同一24k末步：

| 输入 | 后半程学习率 | 24k TRAIN mean mm | 24k dev mean mm | 24k dev p95 mm |
|---|---|---:|---:|---:|
| full_relative | constant | 10.14 | 43.80 | 134.02 |
| full_relative | cosine_tail | 6.28 | 43.22 | 133.77 |
| current_relative | constant | 13.24 | 49.73 | 165.72 |
| current_relative | cosine_tail | 8.71 | 49.51 | 155.46 |

退火使full1442 TRAIN mean10.14→6.28mm（降低38.1%），但dev43.80→43.22mm
（仅降低1.34%）；current721 TRAIN13.24→8.71mm，dev49.73→49.51mm。
在此优化干预下未过dev20mm门，也未达到预设TRAIN<=5mm分支，保持UNCLEAR；
不把“优先定位泛化”升级为已排除优化/容量的结论。继续加同类固定数据训练优先级降低。

各臂dev按p95→mean选择点（原选择门保留）：

| 输入 | schedule | selected step | TRAIN mean mm | dev mean mm | dev p95 mm |
|---|---|---:|---:|---:|---:|
| full_relative | constant | 21500 | 12.13 | 43.92 | 129.91 |
| full_relative | cosine_tail | 16500 | 11.21 | 43.39 | 132.07 |
| current_relative | constant | 9500 | 19.51 | 48.97 | 144.02 |
| current_relative | cosine_tail | 9500 | 19.51 | 48.97 | 144.02 |

## 全量窗口核对与归因边界

使用GPU0对4个末步权重读取全部110250 TRAIN/12049 dev窗口重推，固定面板误差与
保存预测独立复算一致；all-window-audit.json为PASS。均匀窗口权重不同于主要面板，
只作为代表性检查，不能用更低的all-window mean事后替换主要门。

- full1442固定LR：全部TRAIN/dev mean13.02/35.01mm；退火后11.33/34.90mm。
- full1442退火：TRAIN startup9402帧mean0.466mm/p951.774mm；dev startup1134帧
  mean47.213mm/p95131.742mm。TRAIN later mean12.339mm，dev later33.625mm。
- current721退火：TRAIN startup0.832mm，而dev startup53.193mm。

这直接显示学生可精确拟合见过序列的起始状态，跨序列起步没有随训练精度推广。
TRAIＮ/dev都覆盖同50物体、同10subjects，但序列严格不交叠；不能称未见物体/受试者
的泛化测试。完整teacher输入仍有少量相对收益，尚不能认定缺未来信息是唯一/主要瓶颈。
高精度startup也不证明后续动态状态全部充分拟合，完整TRAIN11.33mm残差仍待解释。

Decision：本轮固定预算停止，不扩展同类优化搜索。后续优先区分同序列未训练帧的
插值与整序列新动作/初态外推，并在人工审阅完成后设计固定clean TRAIN对照；保留
原始完整dev基准并单列质量分层，不随已看到的误差删dev。人工标注当前未进入任何
准入或训练，Viser与标注文件继续可用。本轮没有新的物理抓取结论。

输出teacher-fit-generalization-20261015-r1/summary保留results.json/csv、table.md、
lr-controls.png、manifest.json；all-window-audit.json保存所有窗口/phase统计和相同
prefix检查，initial-error-distribution.json保存先验尾部诊断。各组curve/result/预测和
stage权重可追溯。原source、旧权重、人工标签和远端训练进程均未改写。
