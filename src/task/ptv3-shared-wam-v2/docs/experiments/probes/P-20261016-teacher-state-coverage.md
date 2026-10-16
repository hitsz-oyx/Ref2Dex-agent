---
schema: ref2dex.probe.v2
probe_id: P-20261016-teacher-state-coverage
experiment_id: P-20261016-teacher-state-coverage
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-full-teacher-input
probe_index_in_family: 5
seed_pool: probe
seeds: [180, 181]
decision_changed_if_positive: prioritize relevant state coverage and sample-efficient feedback learning
decision_changed_if_negative: inspect effective coverage and control inductive bias before collecting more trajectories
status: RUNNING
run_id: teacher-state-coverage-20261016-r1
---

# 固定留出面板的教师状态覆盖对照

Class: Decision。服务MISSION自训控制器与WAM可评价底座。用户已认可上一轮提出的覆盖
对照并要求持续推进。问题：旧杯类模型的泛化缺口能否由现有其他物体真实teacher状态减小？
最便宜方法是复用全660缓存，不仿真、不重新采集、不训练WAM。

## 协议与决策（训练前固定）

复用teacher-sequence-generalization的seen6849/held1707/unseen987身份。原杯类57序列
与扩大589序列对照，扩大训练106005行；原held及其[t-3,t+24]隔离带2538行全部排除，
DEV不进入训练。扩大只来自其他物体，不能把它等同新增杯类近邻状态或因果恢复覆盖。
初始模型使用旧step0随机权重，不使用teacher actor权重；原teacher RMS与原seen-only
目标mean/std全部固定，防止归一化改变损失权重。1442→1024→1024→1024→512→12，
relative wrist6/absolute active fingers6，L1，AdamW3e-4/WD1e-4/clip1，24k更新，
12k后cosine到3e-6。sequence-uniform、half-startup、batch512，初始化180/抽样181。
原始组重新运行并与旧终点比较；本轮相同seed仅用于匹配，不是独立重复。

固定同组sequence-macro腕部XYZ mean为主指标，两留出均改善>=10%，p95不恶化>5%，
旋转p95不恶化>10%，且原seen拟合<=5mm或显著恢复，则标PROMISING并进入小闭环/覆盖
后续判别。均无改善且拟合合格时，本次扩大配方UNPROMISING；若扩大组训练欠拟合或
原杯类曝光显著稀释，记UNCLEAR，不能由此否定覆盖假设。保留全训练拟合与分阶段指标。
中期曲线只描述，不使用DEV选点。检查全1442及物理反馈子块的近邻覆盖以解释数据是否相关；
近邻不保证控制等价。后续配方另记Decision Note，不做无界参数搜索。

## Decision Note

关键证据：原学生1000→24000步seen18.40→0.56mm，unseen41.51→42.44mm；
冻结WAM六臂未改善。原cache教师回放max0.000795mm，故完整输入足以再现教师标签。
root选择先比较已有数据覆盖，避免重复读出/容量搜索。目标是定位而不是用大量示范替代
少样本目标；新Shadow v3早期正信号属于独立Task，不覆盖本轮Inspire结论。

## 资源、运行与停止

本地GPU0，启动时空闲，GPU6/7保留空闲；不干预GPU1–5既有任务。每臂1200秒/512MB，
本轮两臂+独立smoke/数组审计合计<=45分钟/2GiB；磁盘至少20GiB。
工程smoke2步不进入科学结果。训练记录GPU显存/利用率/ETA。非有限、外来consumer、
成员泄漏、原始组不能复现或预算超限停止自己的run。TMPDIR=项目tmp。
实现入口tools/run/probe_teacher_coverage.py，产物outputs/ptv3-shared-wam-v2/
teacher-state-coverage-20261016-r1/{original,expanded}，运行manifest记录实际代码commit。
保持main，不新建分支，不覆盖其他会话修改或旧产物。

## Limitations / future evidence

单训练seed、反复使用DEV、特权完整教师参考、固定杯类面板；不证明跨手型或闭环收益。
全量抽样减少原杯类曝光，负结果须检查欠拟合；更多其他物体可能不覆盖关键反馈状态。
若信号明确再做物理执行和后续匹配WAM对照，多seed正式证据延后。
