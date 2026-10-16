---
schema: ref2dex.probe.v2
probe_id: P-20261016-wam-source-control-transfer
experiment_id: P-20261016-wam-source-control-transfer
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-full-teacher-input
probe_index_in_family: 4
seed_pool: probe
seeds: [180, 181]
decision_changed_if_positive: localize WAM transfer benefit to pretraining source or semantic readout before further joint adaptation
decision_changed_if_negative: stop the tested frozen source and pooling route and diagnose state-space coverage and control-relevant representation
status: UNPROMISING
run_id: wam-source-control-transfer-20261016-r1
---

# 冻结WAM来源与腕部语义读出

Result: COMPLETED，UNPROMISING。parent/robot与mean/wrist四个WAM臂均更好拟合seen、恶化两组留出；无确定性实现错误。
Decision: 结束当前来源/聚合扩展，优先区分BC状态覆盖与控制学习归纳偏置；保留WAM核心假设。

Class: Decision。用户明确授权自主探索。服务WAM控制效用；沿用刚完成的完整teacher
隔离划分，检验原80k与机器人7k表征是否有不同控制迁移能力。

## 假设、成本与决策

H1：原80k优于7k，优先定位联合微调的表示变化。H2：两个来源mean都弱、wrist读出
改善，则优先修读出。H3：真实特征不优于等宽raw随机投影，则当前接法没有明确迁移收益。
最便宜方案为现有63条cup/mug序列9543个seen/held/unseen状态的一次冻结特征提取，
缓存mean与wrist两种2304D读出；只训练小控制头，不重训WAM或仿真。

原始成员：seen6849/interpolation1707/unseen987。完整teacher1442、teacher RMS/clamp、
relativePD active12、MLP1442→1024→1024→1024→512→12，与原实验一致；新增零初始化
2304→1024输入支路。各臂raw初始参数必须逐tensor等同原step0，抽样180/181复用，非独立重复。
训练样本/标签不因WAM来源改变；任何原成员越出encoder域先失败，不静默筛掉。
输入teacher统计沿用原特权诊断；输出与特征统计只用seen，隔离和unseen不拟合统计。

第一阶段none、parent-mean、robot-mean各24k更新，原batch512/AdamW3e-4/WD1e-4/
clip1、12k后cosine至3e-6，固定终点。若mean没有明确跨两留出组信号，再做random
raw-only固定投影、parent-wrist、robot-wrist，同预算。六臂上限，不追加无界搜索。
wrist只替换hand768块，其余scene/goal均值、2304维与支路参数不变。
none支路实际无梯度；random臂检验额外有效容量/归纳偏置，不称严格随机语义等价对照。

预定Decision信号：同来源/读出的隔离与unseen序列等权XYZ均改善>=10%，seen拟合
不明显恶化，且p95/旋转不出现大退步，才值得扩充物理验证。否则UNCLEAR/UNPROMISING，
不形成正式WAM无效/遗忘结论。主要读出沿用原三组macro，报告tick均值、p95与phase。
原none应复现旧结果；若漂移明显，先检查数值/初始化，不进行来源归因。

## 资源与版本

原方案仅本地GPU5、保留6/7；实际按下方用户授权临时使用GPU6/7。整个探索<=30min、
产物<=2GiB、磁盘保留20GiB；每stage<=1800s，记录GPU显存、利用率与elapsed/ETA。
GPU计算优先；prepare和数组汇总CPU。原数据/checkpoint只读，当前main仅提交本轮
新增工具/测试/卡片，不新建分支或覆盖用户修改。入口
`tools/run/probe_wam_source_transfer.py`，输出`outputs/ptv3-shared-wam-v2/<run_id>/`。

停止：非有限、原成员/权重加载/冻结/初始化合同失败、未知GPU消费者、超时、产物
或磁盘越界停止自己的stage并保留证据。必要时只修有明确复现的接线问题。

## Limitations / future evidence

单seed、完整teacher特权输入、一个任务族、已反复使用DEV。不是少示范或跨手型结论，
也不是物理抓取成功。mean/wrist差异改变可用语义，不能凭正信号单独证明原pooling
完全丢失信息。来源效应不等于已证明灾难遗忘；正式结论需匹配Validation。
WAM分支额外接收H4/CAD和已知t+24物体goal；raw1442主要为原teacher的t+1/t+16参考。
因此none/random与WAM的信息集并非完全相同，正信号不单独证明预训练因果；本轮首要
来源对照parent/robot的信息集相同。random线性投影主要控制额外参数化，而非任意非线性容量。

历史暴露边界：parent是OakInk2/GRAB/ARCTIC三域80k，不是九源v3。robot7k此前已在
旧strict TRAIN上联合学习；本轮seen6849中5071、隔离1707中1670条曾属于旧TRAIN准入。
unseen987中800条属于旧DEV，未梯度训练，但用于原7k选点。因此本轮隔离只针对新控制头，
不能称整个系统未见状态泛化；parent/robot对比包含历史监督暴露差异。来源差异不能单独
证明灾难遗忘，robot的正信号也不能直接当作未见状态迁移。独立只读review核对这些成员。

## Decision Note：临时使用空闲卡

用户本轮明确允许临时占用剩余卡。GPU5正在既有PPO100评估，改用GPU6/7分别提取
parent/robot特征并训练匹配头；已占用0–5任务不动。CLI显式`--temporary-idle-use`只允许
6/7且仍硬拒绝已有消费者。例外仅本轮<=30min/2GiB，完成释放，不改全局CAMPAIGN。
此前GPU5启动在分配前被保护拒绝，日志extract-parent.log保留，无模型计算/科学结果。

## 结果：六臂完成

实际计算提交`e98b9a48`，全部固定24k终点；不是按DEV选择最优步。序列等权腕部
PD XYZ误差如下，单位mm；隔离特指本轮控制头未训练的同序列片段。

| 特征臂 | seen | 同序列隔离 | unseen序列 | 隔离逐帧p95 | unseen逐帧p95 |
|---|---:|---:|---:|---:|---:|
| none | 0.564 | 32.652 | 42.436 | 76.961 | 101.488 |
| raw固定随机投影 | 0.308 | 30.816 | 41.652 | 75.834 | 96.651 |
| parent80k mean | 0.171 | 36.925 | 48.588 | 95.590 | 135.794 |
| parent80k wrist | 0.186 | 37.311 | 48.887 | 96.486 | 128.691 |
| robot7k mean | 0.169 | 36.636 | 51.777 | 95.910 | 135.460 |
| robot7k wrist | 0.162 | 37.333 | 52.225 | 99.370 | 140.525 |

四个真实WAM臂seen更准，但两组留出macro及p95均更差；未达到两组均改善10%的预定门。
random相对none仅改善隔离5.62%、unseen1.85%，本身也未达到门限。它与raw第一层相加，
可代数合并成一个线性层，主要改变参数化/优化偏置，不能称已增加函数类容量。

这削弱了“只因联合微调毁掉了原表征”与“只因hand均值丢掉腕部身份”对本轮结果的单因解释。
不证明联合微调没有负影响，也不排除其他完整token读出、控制损失或九源v3能有收益。
较符合当前观察的机制是：该2304D支路促进训练拟合，却没有提供可泛化的控制约束；
这仍是候选解释，不是表征信息量或过拟合机制的正式证明。

## 实现与资源核验

- 六臂样本、首draw与原始PD标签一致；逐帧数组独立重算的报告误差为0。
- none终点raw参数与旧`step24000.pt`全部tensor逐位相等；macro差<0.00014mm。
  不同推理分组/batch形状下seen/隔离单坐标最大差0.108/0.032mm，unseen为0；不称预测逐位相等。
- 两源严格加载、冻结且无梯度；全部原成员workspace通过，无静默筛样；seen-only特征统计逐项核对。
- 独立review复核来源、成员历史暴露、wrist身份和mean/wrist只改hand块；
  `reviewer-audit.json`当时只复算了none/parent-mean，六臂终点复算见`summary/array-audit.json`。
- 两源提取74.1/73.0s；六个24k头各74.7–76.8s，总stage计算约601s，单卡合计约10min，
  两卡实际并行；提取约2.4GiB/卡、头训练约0.64GiB/卡，采样利用率峰值提取82–85%、头54–58%。
  最终产物约567MiB，剩余磁盘>160GiB；GPU6/7均已释放，原任务未终止。

产物根：`outputs/ptv3-shared-wam-v2/wam-source-control-transfer-20261016-r1/`；
六个`fit-*/final.pt`、`predictions-final.npz`、固定统计、曲线、process/resources记录保留。
汇总：`summary/results.json`、`metrics.csv`、`table.md`、`comparison.png`、`array-audit.json`。
工程smoke2步仅验证运行，不进入上述结果。3项新合同与原teacher划分共8项测试通过。

## Decision Note：停止当前冻结来源/读出扩展

问题：是否继续靠换来源、换hand聚合或增加同类冻结支路来救当前控制泛化？
证据：原父权重与robot权重、mean与wrist均未改善两留出组，且p95同步恶化；raw终点精确复现。
root选择：本轮标UNPROMISING，结束六臂，不进入这四个WAM头的物理评估或长训；
保留WAM核心假设，转向分辨BC状态覆盖与控制学习的归纳偏置，而非继续扩大此支路。
下一步应先核查固定留出面板之外可用的真实teacher状态，设计训练覆盖度对照；若缺少
可用覆盖，则优先用单变量控制学习Probe，不用重复更多更新替代。新实验另记预算与停止门。
本轮无新增资源或外部授权需求。跨手型、少示范、对象轨迹控制和最终PPO增益仍需后续证据。
