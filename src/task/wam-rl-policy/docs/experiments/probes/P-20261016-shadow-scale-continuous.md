---
schema: ref2dex.probe.v2
probe_id: P-20261016-shadow-scale-continuous
experiment_id: P-20261016-shadow-scale-continuous
date: 2026-10-16
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-shadow-scale-continuous
probe_index_in_family: 1
seed_pool: probe
seeds: [241, 242, 243, 244, 245, 246, 247, 248, 249]
decision_changed_if_positive: retain a competent self-trained Shadow BC bootstrap and scale online improvement before frozen-WAM token validation
decision_changed_if_negative: diagnose coverage objective reward and exploration instead of treating a short two-object run as method failure
status: UNCLEAR
run_id: shadow-scale-continuous-20261016-r1
---

# Shadow大规模持续推进

Class: Decision。服务自主抓取基座与预训练物理表示辅助RL的最终目标。用户明确授权大规模
持续推进、充分利用本地/远端GPU。前20轮仅2TRAIN对象、127k步、DEV24起点，不能判断充分
训练效果；旧query BC还混合8拍动作监督，当前单拍拟合/容量可能限制RL bootstrap。

## Decision Note（扩大前）

选择并行两线：现有冻结WAM/BC的PPO扩大对象/并行环境/训练；同一全量Shadow缓存直接
训练current300→当前canonical24的自训练BC MLP，争取得到更强、更快的reference。
不使用官方actor权重初始化学生；官方既有动作日志允许作为监督。若MLP提高真实闭环
DEV保持，再接同一PPO；它不替代最终WAM效用对照，不变更MISSION claim。

第一阶段在线最多128个TRAIN对象、4GPU同步收集；环境数由64/128等真实吞吐smoke确定，
目标200次更新，50/100/200读出，单次训练<=12h。已有2对象20轮权重/optimizer/RNG
可作为单卡200轮持续性对照，最多4h；不声称恢复PhysX内部状态。
BC MLP使用全量strict TRAIN1536897/DEV354761当前标签，fresh seed241，后续242/243
只在正向信号后配对复核；batch2048、AdamW3e-4/WD1e-5、标准化L1、最多50000步/2h，
保留5k/10k/20k/50k等端点。离线DEV分对象固定面板，不用TRAIN归一化之外的信息。

扩大评估：先冻结所有388DEV对象的native资产与分批面板，4起点/对象，总1552rollouts
每策略；连续读出用预先选择的64DEV子面板（256rollouts），不按结果挑选。全部物体
明确来自本项目DEV，仍属官方TRAIN集合，不冒称官方test-unseen泛化。
对照相同对象/seed/env配置，实际九类初态须逐值相同；baseline也有数值波动，native保持
代理不能当接触认证。主指标连续60帧goal<=5cm且lift>=10cm，另报保持到终点、reach/lift
和退步病例。DEV>=10个百分点改善且TRAIN不明显退步记PROMISING，再固定协议进入Validation。
阶段曲线/行为无进步则诊断或调整，不因单个成功视频、训练loss或GPU数量宣布成功。

## 实现与资源边界

代码仅当前main下wam-rl-policy，原BC/数据/检查点只读；新产物保存本run专用目录。
补齐训练恢复、分阶段checkpoint、早停/监控；多卡必须全局adv归一化、同步actor/critic
梯度、全局KL与共同accept/rollback，不能把独立训练进程称作DDP。
提速先验证等价：省略未使用forecast heads、批量构造H4输入，与原控制器同输入逐tensor
比较，允许既有GPU推理数值波动但不悄悄改变动作/goal/时钟含义。

本地GPU1–4用于主训/吞吐smoke，GPU5用于BC和评估；GPU0外部作业不动，6/7保持空闲。
远端4卡可用，先同步新权重/代码和所需原生资产、通过实际仿真smoke，再运行不同seed
或方案；跨Torch版本不混作单变量比较。初期总产物<=64GiB、任何时刻磁盘保留20GiB，
每个长作业有明确max_seconds，资源/非有限/合同失败立即停本组并保留记录。
GPU利用率、显存、吞吐和ETA实际记录；冻结WAM/BC可逐阶段抽审state_dict且收尾全量审计。
新runtime目录的资产只读软链接至已有材料，不修改其他Task数据或第三方项目。

## Limitations / future evidence

本卡是探索性规模推进，不是正式Validation。多个训练seed、reward对照与learned RL-token
只有在会改变下一步决策时启动，不能把资源充足作为无界实验理由。正式效用claim仍需
匹配none/shuffled/aligned及冻结评估协议；暂不更新整个WAM。
