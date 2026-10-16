---
schema: ref2dex.probe.v2
probe_id: P-20261016-wrist-shared-grasp
experiment_id: P-20261016-wrist-shared-grasp
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: a5beb5b0
claim_id: C1
hypothesis_family: HF-wrist-shared-grasp
probe_index_in_family: 1
seed_pool: probe
seeds: [148]
decision_changed_if_positive: retain the matched live shared-conditioned wrist interface for a larger fixed behavior panel
decision_changed_if_negative: retain direct current-input wrist control and diagnose remaining finger/recovery limitations
status: RUNNING
run_id: wrist-shared-grasp-20261016-r1
---

# 同一腕部MLP±shared的实际抓取评估

Class: Decision。用户明确要求评估新版本抓取，覆盖上轮“离线正向再进native”的近期安排。
本次不因离线负结果跳过行为验证，也不把更低离线误差自动当作能抓取。
服务Inspire控制映射与交互表示效用；检验实际接入shared是否改变部分持物行为。

最便宜的可判别协议：固定原三条val飞机s10_airplane_pass_1（149 controls）、
球s4_spheresmall_pass_1（141）、杯s10_mug_pass_1（172）。seed148、K1、同初态。
各组3个隔离环境：teacher校准 / none腕部MLP+原shared手指 / aligned腕部MLP+原shared手指。
两个MLP固定上轮seed196/draw197的6k终点，不按DEV挑选seed或checkpoint；原WAM及
手指均用同一shared7k control-step7000.pt，不加载两臂不同joint WAM。

raw811D使用每行各自同tick87D状态、goal24位置差与当前721D；shared2304D由冻结WAM
从各自H4手/物体和既有goal在线计算，仍为scene/hand/goal三组mask均值，严禁回放离线缓存。
server查询顺序own(env2)/repeat(env1)，进入none/aligned时明确逆序，原手指不变。
两模型均输出当前腕部6D native命令，替换clamp前intended[:6]；不使用teacher动作辅助。

首个4 controls工程smoke必须通过实际PD/完整chunk→原手指/六维替换/当前raw输入/
shared行顺序/CPU模型重放审计，再启动三个对象并行。新增接口绑定同一feature checkpoint，
加载MLP不得改变teacher RNG；各行state/H4/goal和执行时钟仍沿用独立audit。
shared模型推理每tick一次，同时保留原chunk与pooled特征用于回放检查。

主指标沿用partial_hold45_only_no_placement：离桌>=3cm、表面距离<=1cm、排除桌面支撑，
连续至少45帧。报告最长保持、抬升/接近、clipping；teacher未过门时单列校准不足。
初始teacher/学生状态必须一致；workspace退出和执行失败单列，不补成0或当作成功率。
此固定三对象单seed只能形成PROMISING/UNPROMISING/UNCLEAR，不证明泛化或完整放置。

本地GPU1/2/3并行、GPU0外部作业不动、4–7空闲。smoke<=600s，每组<=600s，总本轮
<=2400s/2GiB，磁盘保留20GiB。无新训练、不扩大远端运行、不改原数据/checkpoint。
任一工程检查失败先保留原产物并修复，只补受影响组；不放宽重放容差掩盖问题。

Decision Note：本轮用户要求以行为核对离线判断，选择固定小面板而非先扩展训练或
换全部控制器。学生原手指共用、实时shared从各自状态计算，能回答腕部接法是否值得
继续，但不能隔离手指误差与恢复能力全部影响。持物有正向再扩大行为面板；全失败则
优先分析接近/接触/抬升阶段，不据单次Probe否定整套表示。

输出outputs/ptv3-shared-wam-v2/wrist-shared-grasp-20261016-r1/，实际代码提交记录于运行回执。
