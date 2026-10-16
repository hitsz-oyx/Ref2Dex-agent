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
status: PLANNED
run_id: wam-source-control-transfer-20261016-r1
---

# 冻结WAM来源与腕部语义读出

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

仅本地GPU5；0外部工作、1–4既有PPO保留，6/7始终空闲。整个探索<=30min、
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
