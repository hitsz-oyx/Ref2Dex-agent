---
schema: ref2dex.research-report.v1
report_id: P-20261023-bc-ppo-usefulness-preflight
date: 2026-10-23
task: wam-rl-policy-v3
experiment_id: P-20261023-bc-ppo-usefulness
branch: main
run_id: bc-ppo-usefulness-20261023-r1
status: PREFLIGHT_COMPLETE_BUDGET_REDUCED
---

# 结果解读表

| 实测结果 | 含义 | 决定 |
|---|---|---|
| BC 3.1236 / PPO 4.0602 controls/s，均<8 | 原1e5/arm顺序需15.73h，预算账不成立 | **预算缩小：两arm各14,400 controls**，不硬跑1e5 |
| 22 updates；exact joint KL 0.0001121–0.0084083；clip 0–0.1317；回滚0 | 旧首步KL异常未复现，数值门通过 | 同BC起点重新开始配对；不继承preflight权重 |
| GPU7采样峰值6045MiB；写盘49.85KiB/s；总输出预测0.679GB | 显存/磁盘预算可承受 | 固定GPU7顺序，不借GPU6；总输出硬门4.8GB |
| 正式成功率/限位/≥3初态改善：尚未运行 | preflight不证明任务收益，未完成1e5无改善分支 | 原判据保留，不能下“PPO无效”结论 |

preflight从04:08:06开始，报告时间2026-10-23T04:27:13.216195+08:00，累计19.11min。
PPO 3600 controls / 886.657s，BC 600 / 192.086s。
这是含模型/原生启动、reset、更新和落盘的端到端速度；不是优化器吞吐。
PPO实际优化器速率0.024812 updates/s。
BC短测只覆盖第一个初态，PPO覆盖全部6初态且重复；因此速度不视为所有相位的常数。
原计划preflight两arm各3600，首完整回合低于8/s后取消第二个3600，再补600 BC controls，
使工程preflight在20分钟内完成；这不是正式配对预算。

## 顺序 total vs 08:00 截止

| 项目 | 实测推算 |
|---|---:|
| 原每arm100000，BC+PPO顺序 | 15.734h |
| **预算缩小**：每arm14400，顺序 | 2.266h |
| 两个6初态确定性评估，按各3600 controls上限 | 33.99min |
| 原始轨迹/更新重放与报告预留 | 15min |
| 缩减预算+评估+预留预计完成 | 2026-10-23T07:32:09.050102+08:00 |
| 机主截止 | 2026-10-23 08:00 +08:00 |

预算用`floor(((08:00-now - assessment_estimate - 900s)*.85)/(1/r_BC+1/r_PPO)/600)*600`，
即从剩余时间扣掉评估/复核/报告，再留15%速度裕量，并对两arm同比例缩至14.4%。
固定这次规模，不根据收益途中修改学习率、KL、奖励或成功阈值。
supervisor总截止07:50，保留partial；工作区可用<20GiB或输出>4.8GB停止。
初检GPU6=4110MiB，不满足借用门；本runner仅接受GPU7。

## 数值、显存与写盘

全22 epochs均accepted；行为logprob重放最大误差0.000018120<1e-4。
熵区间-12.972808838至-12.972740173，是12D Normal differential entropy，允许为负。
KL低端有1e-4级的小更新，不把“低于target”自动等同任务有效；无NaN/Inf、无拒绝/回滚。
GPU采样利用率0–64%，
显存是30s采样峰值，非逐毫秒峰值；完整数据在`resources.jsonl`。
观测写盘13110.8 bytes/control，比历史10KiB估计更大；
按实测×1.25且至少16KiB/control估计，含preflight/评估/32MiB额外预留，总0.679GB<5GB。
无视频，checkpoint每40回合及最终，独立新文件；不覆盖旧checkpoint。

## PPO逐回合序列

每格KL/clip按epoch0、epoch1排列；回报为当前采样回合reward sum，不与最终确定性成功率混用。

| 回合 | controls | exact joint KL | clip fraction | 末epoch熵 | 回报 | retries |
|---|---:|---|---|---:|---:|---:|
| p0000-A1 | 210 | 0.0014918, 0.0058966 | 0.0000, 0.0762 | -12.972776 | -1.164566 | 0 |
| p0001-A1 | 600 | 0.0018810, 0.0084083 | 0.0050, 0.1317 | -12.972800 | -0.740928 | 0 |
| p0002-A1 | 203 | 0.0014275, 0.0051166 | 0.0049, 0.0345 | -12.972786 | -1.204617 | 0 |
| p0003-A1 | 84 | 0.0002074, 0.0009390 | 0.0000, 0.0000 | -12.972775 | -1.090193 | 0 |
| p0004-A1 | 600 | 0.0004207, 0.0020085 | 0.0000, 0.0083 | -12.972787 | -0.370710 | 0 |
| p0005-A1 | 600 | 0.0002981, 0.0015452 | 0.0000, 0.0017 | -12.972797 | 1.560738 | 0 |
| p0006-A1 | 92 | 0.0001121, 0.0004396 | 0.0000, 0.0000 | -12.972809 | -1.076075 | 0 |
| p0007-A1 | 203 | 0.0002482, 0.0012269 | 0.0000, 0.0049 | -12.972795 | -1.210521 | 0 |
| p0008-A1 | 405 | 0.0005848, 0.0027956 | 0.0000, 0.0296 | -12.972772 | -1.130226 | 0 |
| p0009-A1 | 600 | 0.0002800, 0.0011157 | 0.0000, 0.0033 | -12.972745 | -0.193688 | 0 |
| p0010-A1 | 3 | 0.0004180, 0.0023032 | 0.0000, 0.0000 | -12.972742 | -0.003380 | 0 |

最后预算截断回合不算完整600-controls评估。正式runner保持原600-controls成功条件，
预算末尾只做时间截断、bootstrap和标记partial，不提前发成功奖励。

## 运行环境、冻结和评测口径

策略Python `/home2/wyy/miniconda3/envs/graspenv/bin/python`，Python 3.8.20、Torch 2.4.1+cu121、CUDA 12.1。
原生Python `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`，
各episode `physics/manifest.json`实测Torch2.0.1+cu118、GPU PhysX、1 env、packed state、synchronized setters。
原生Python与策略环境分别记录，不原地升级；TMPDIR仅项目`tmp/`。
preflight代码commit `330ac19dc1f93fabb2392d57e9a287920a550be5`；后续main使用已提交专用runner，
模型/评分/物理代码复用`preflight-ppo/runtime-code`同一只读快照。

[专用runner](../tools/run/run_bc_ppo_probe.py)只读导入原GeometryGraspConsumer、StagedProgress、
physical_safety和原生host，未修改source_ceiling、pose_consumer、tools/audit、run_source_ceiling或grasp-v13文件。
与原geometry worker的差异：允许GPU7/规模预算、两arm同记reward、有限checkpoint、
finite/精确KL门日志、预算截断和快照复用。完整回合仍是600 controls、尺寸相关位置容差、
实测连续hold、原物理安全门；不是旧15mm协议，也不借source-ceiling资格结论。
关节主要比较`physical_joint_limit`回合数（原URDF外0.1安全裕量），同时单列名义URDF越界状态数，
不把0 abort冒充0机械偏离。逐初态改善=成功增加或joint abort减少；
回退=成功减少或joint abort增加；两项冲突也计回退，失败签名另报。

正式两arm从同一`drift-intent-residual-20261022-r1/bc/bc.pt`开始，
R/WAM不更新；模型seed293，原生seed294–299，初始化196，全部probe seed复用。
preflight第二轮模型seed294，仅工程稳定性，不进入主训练。
最终两arm同一六初态/seed、均值动作、相同观测/动作/phase合同，逐数组核对初态。
原判据仍需≥3初态改善且无回退；不以训练采样回报或仅失败类型迁移替代完整报告。

原始指标：`outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/preflight-summary.json`、
各arm manifest、`resources.jsonl`；注册为`paired-registration.json`及`assessment-registration.json`。
相关回归12 passed。全仓verify当时被另一条路线的卡片metadata/链接阻断，未改其文件。
