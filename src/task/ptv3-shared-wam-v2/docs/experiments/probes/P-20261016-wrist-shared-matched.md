---
schema: ref2dex.probe.v2
probe_id: P-20261016-wrist-shared-matched
experiment_id: P-20261016-wrist-shared-matched
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: b209a2e7
claim_id: C1
hypothesis_family: HF-wrist-shared-matched
probe_index_in_family: 1
seed_pool: probe
seeds: [196, 197, 199, 148]
decision_changed_if_positive: retain a direct811D control path and test aligned shared conditions in native grasp execution
decision_changed_if_negative: prioritize control training and feature readout before joint WAM adaptation
status: RUNNING
run_id: wrist-shared-matched-20261016-r1
---

# 同一腕部MLP有无shared的匹配对照

Class: Decision。服务解释Inspire控制映射误差，以及预训练交互表示是否能改善机器人执行。
此前MLP与shared actor同时改变融合、目标、网络与优化，不能用其差距归因shared。
用户已授权执行同一MLP±shared，并利用本地/远端并行或分布式加速；本地仍保留两张空闲卡。

## 固定协议与能改变的决策

raw输入始终原811D（state87+已有goal24位置差3+当前721），同一strict
TRAIN93680/DEV10376窗口，同TRAIN-only归一化、原评估身份、半数startup/半数later且
phase内序列均匀抽样、model196/draw197、global512、fresh初始化、6000步固定终点。
仍只监督当前native腕部6D命令，L1按TRAIN标准化，不增加未来手参考或teacher动作干预。
原MLP隐藏1024/1024/1024/512保持；shared额外线性投影只加至第一层preactivation，
初始化全0，因此三臂初始模型和输出逐项相同。none臂使用零shared，但保留相同容量。

冻结共享7k WAM，在eval模式FP32/TF32对每个实际窗口提取shared，以有效mask分别对
scene/hand/goal求均值再拼接2304D；不把未来标签送入encode。每split独立缓存，
shared统计只在TRAIN拟合；shuffle在TRAIN/DEV内分别固定置换（199/200），不跨split。
pooling是本Probe具体readout限制，负结果不能否定其他token readout或整个WAM。

主三臂：none / aligned / shuffled，LR5e-4、WD1e-4。相同逐draw及模型初始化认证后执行。
附加两组none控制分别仅改LR为5e-5、或仅改WD为0.01，区分旧优化设置的影响；
这些不作为shared效用的配对臂，不混入主三臂因果比较。

主要读出为同一DEV256首控XYZ mean/p95与startup分项；TRAIN256检验拟合，另保存全部
TRAIN/DEV指标，不用更好看的all-window均值替换主面板。aligned相对none的mean和p95
均改善至少10%、startup不恶化，且优于shuffled，才记离线PROMISING并进入固定native；
否则UNCLEAR或UNPROMISING，保持探索结论边界。单seed不构成正式效用结论。

## 执行、资源与停止条件

本地GPU1/2/3/4连续分片并行提取同一冻结模型，GPU0外部作业不动，5/6/7空闲。
每分片<=1800s，完整缓存与训练<=8GiB，磁盘保留20GiB；非有限、身份/边界/覆盖不符即停。
合并必须每窗口恰好覆盖一次、checkpoint和pooling一致；原660及权重只读。
五组MLP可在GPU1–5并行（0外部、6/7空闲），每组<=900s/0.5GiB，仅保留固定终点。
阶段始末记录GPU利用率、显存和ETA；模型训练默认GPU。

远端四卡当前空闲，但portable迁移仍RECEIVING，不据目录存在宣称同数据可用。
先在独立remote run目录运行同形状MLP的1/2卡DDP工程吞吐检查（相同global512，<=300步）；
只决定执行方式，无学习结论。小MLP若DDP更慢，优先每卡独立对照；数据准备不阻塞本地。
不停止/改写既有上传、远端预训练或其他进程。正式运行保存本次实际代码提交和输入路径。

Decision Note：优先隔离shared条件而非新增模型路线。缓存冻结特征将神经计算摊销到所有
匹配臂，成本低于逐臂重复WAM；短DDP benchmark避免更大通信开销。主对照有信号再做
物理闭环及后续joint更新；无信号先定位融合/优化，不追加原joint长训。用户授权内执行，
没有改变最终claim或扩大数据权限，不需新增身份/分支。本轮不自动启动PPO。

入口：tools/run/probe_wrist_shared.py；模型/采样：src/ptv3_shared_wam_v2/wrist_shared_probe.py。
输出：outputs/ptv3-shared-wam-v2/wrist-shared-matched-20261016-r1/。
