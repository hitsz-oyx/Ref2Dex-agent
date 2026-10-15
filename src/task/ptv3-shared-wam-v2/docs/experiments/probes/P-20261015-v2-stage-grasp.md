---
schema: ref2dex.probe.v2
probe_id: P-20261015-v2-stage-grasp
experiment_id: P-20261015-v2-stage-grasp
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: b8542ef6
claim_id: C1
hypothesis_family: HF-v2-stage-grasp
probe_index_in_family: 1
seed_pool: probe
seeds: [148]
decision_changed_if_positive: qualify successful stages on a broader physical panel and compare action versus forecast stage selection
decision_changed_if_negative: separate R execution failure from predicted trajectory failure before further fine-tuning
status: RUNNING
run_id: v2-stage-grasp-20261015-r1
---

# WAM-v2 微调阶段的实际抓取 Probe

Result: pending engineering smoke and native panel.
Decision: 用户要求直接拿保留的微调权重沿用旧Task物理评估；不再延长训练。

## 动机与协议

Class: Decision。服务于MISSION自主执行底座，区分离线动作loss降低能否改善抓取，以及
GTτ→R与预测τ→R在哪一层失效。最便宜方法是复用已有三路native执行，逐个完整权重
严格加载同阶段WAM与R（含normalizer），不跨阶段混配，不替换原80k parent。

固定完整阶段1000/3000/3500/4000/8000/9000/10000/11000/12000；权重只读位于
`outputs/ptv3-shared-wam-v2/robot-finetune-stage-archive-20261015-r1/`。
固定原dev面板s10_airplane_pass_1、s10_mug_pass_1、s4_spheresmall_pass_1，沿用旧
字典序选择，不按新结果筛样；控制长度分别149/172/141，留足GTτ的未来24帧。
每组3环境frame0重置，角色teacher、GT_tau_R、shared_WAM_R。K4刷新、8步chunk、
30Hz、原PD增益/coupling/clamp；query冻结目标并以live q逆转换。GTτ取原RL660实际
未来手轨迹（oracle诊断）；自主行只取自己的H4状态、原有参考物体t+24目标和预测τ。
本轮没有添加新的输入，也没有把参考未来手轨迹送给自主行。

v2 checkpoint保存原source而没有重复保存controller；从其精确source manifest读取
controller并在每组与真实native controller逐项比较。仅内存补齐执行元数据，不覆盖权重。
显式v2 schema/robot stage/完整WAM+R/goal/表示检查；v1默认行为保留。

每组记录原始手物轨迹、命令、实际PD、query/chunk时钟、WAM历史/目标和R输入输出。
复用独立NumPy/SciPy审计核对控制转换、连续角分支、输入时序和held统计。
主要描述性指标为连续held帧与held45：抬升>=3cm、手物表面距离<=1cm且脱离桌面支撑。
它是持物代理，不能当正式任务成功率；另报最大抬升、相对原记录手点误差、相对参考物体
位置误差和clipping。teacher未达到held45的序列标为校准不足，失败不可独立归罪模型。

若某阶段自主行在teacher合格序列达到held45且clipping<1%，记局部PROMISING并择优
做后续扩面；全部自主行未达门且teacher合格时仅记本面板UNPROMISING；teacher不合格
或关键工程问题使归因不清时UNCLEAR。不得从三个已用dev序列宣称泛化或Cm有效。

## 资源与停止条件

本机GPU0/1/2三卡并行，各卡一个序列逐阶段；远端80k和本地四卡微调已完成。
先4控制工程smoke<=180秒，再27组每组<=180秒、面板<=1800秒（最多90GPU-min），
输出<=2GB、文件系统余量>=20GiB；不复制checkpoint，不操作其他任务进程。
每个launcher拥有其policy/native两个进程组；超时/foreign GPU PID/非有限/控制不符
停止该组保留日志。独立失败不得悄悄换阈值/目标/数据；执行问题先定位后只补受影响组。
具体运行提交/进程/预算见process.json与各组group_status.json；初始入口code b8542ef6。

## Limitations / future evidence

单物理seed148沿用旧面板，非独立复制；dev曾用于离线模型选择。完整episode末尾24帧
被截去；held45不含最终放置/任务成功。原记录未来τ和当前执行状态可能分布偏离。
完整多序列、多seed、不同K及人工视觉确认待本Probe改变下一步决策后执行。
