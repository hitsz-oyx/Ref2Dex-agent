---
schema: ref2dex.probe.v2
probe_id: P-20261015-current-wrist-mlp
experiment_id: P-20261015-current-wrist-mlp
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: pending
claim_id: C1
hypothesis_family: HF-current-wrist-mlp
probe_index_in_family: 1
seed_pool: probe
seeds: [196, 197]
decision_changed_if_positive: add captured current teacher information or repair wrist optimization before further joint training
decision_changed_if_negative: prioritize generalization or missing temporal intent over another equivalent joint training extension
status: RUNNING
run_id: current-wrist-mlp-20261015-r1
---

# 当前腕部监督 MLP 配对 Probe

Class: Decision。服务解释Inspire/Shadow差异。混控定位腕部，stored TRAIN256误差66.42mm，
所查reset和batch差不足0.1mm；线性加入当前腕姿误差DEV82.13→82.53mm没有改善。
最低成本使用补采strict相同93680/10376窗，直接拟合当前native腕部6D，不再跑WAM。

A=当前state87+已有物体goal[t+24]位置差3；B=A+同一执行帧捕获的DExplore delta0 721D。
没有增加未来手参考t+1/t+16，无tau。检查源current-only/future-invariance审计及frame_tick。
A的额外721维置零，两臂完全相同811→1024→1024→1024→512→6 ReLU网络和初始权重、
零输出头、TRAIN-only输入/标签统计、AdamW LR5e-4/WD1e-4、归一化L1、batch512、
seed196初始化/197训练采样。每半batch按startup<16/later、各自sequence均匀取样。
这里的采样按可用phase序列均匀；不声称与旧joint逐个draw相同。两臂逐draw完全相同。
不拟合future8/fingers/forecast，故与旧shared比较只能诊断拟合，不能归因为单一学习率。

固定6000步终点评估，不选DEV最优点。复用execution-gap相同TRAIN256/DEV256和四条
序列面板，另终点全TRAIN/DEV，分startup/later。XYZ是未clip意图native命令经过controller
scale的PD误差；Euler报MAE。保存1/1000/…/6000权重与optimizer，不覆盖旧产物。

Decision Note：A准确则优先原joint优化/接口；B显著优于A（DEV均值相对>=20%，startup
也改善）则当前teacher信息值得接回策略；仅TRAIN精确DEV差则优先覆盖/泛化。两者差
则不足以区分信息不足和本次拟合不足，不重复盲目延长。正向离线信号仍需闭环测试。
GPU6/7并行、仅Shadow4/5继续，剩余四卡空闲。每臂<=900s、<=1GB，总<=2GB，保留
20GiB磁盘。预算/非有限/设备冲突只停止本轮，不改现有作业。单seed Probe，无成功率claim。

产物：outputs/ptv3-shared-wam-v2/current-wrist-mlp-20261015-r1/{state_goal,plus_current}/。
脚本：tools/run/probe_current_wrist.py。执行提交由各manifest记录。
