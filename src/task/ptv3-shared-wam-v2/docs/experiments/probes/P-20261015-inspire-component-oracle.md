---
schema: ref2dex.probe.v2
probe_id: P-20261015-inspire-component-oracle
experiment_id: P-20261015-inspire-component-oracle
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: 509a47f8
claim_id: C1
hypothesis_family: HF-inspire-component-oracle
probe_index_in_family: 1
seed_pool: probe
seeds: [148]
decision_changed_if_positive: prioritize the deficient wrist or finger control group identified by teacher assistance
decision_changed_if_negative: retain combined control and observation mismatch hypotheses rather than infer one component is sufficient
status: PLANNED
run_id: inspire-component-oracle-20261015-r1
---

# Inspire shared7k 腕部/手指 teacher 混控定位

Class: Decision。服务当前用户目标：解释Inspire失败而Shadow出现抓取信号的原因。
固定已在三条val序列失败的shared7k，区别腕部轨迹控制与手指闭合哪个子系统更限制
闭环；最低成本是一次三环境短episode，同时测试两个组件干预，避免先扩大训练再猜因果。
这不是自主学生策略，也不用于训练准入或成功率宣传。

## 固定协议

输入权重为outputs/ptv3-shared-wam-v2/shared-native-grasp-20261015-r1/inputs/control-step7000.pt
已有硬链接快照。先4-step工程smoke+独立audit；随后s10_airplane_pass_1、149 controls、
seed148、K=1，三行角色固定：

1. 原始完整teacher正控制。
2. 本行teacher腕部6维 + 本行student手指6个独立命令。
3. 本行student腕部6维 + 本行teacher手指6个独立命令。

teacher每步使用各行自己的1442D观测计算；禁止从row0复制动作。student两行仍使用
本行实测H4/87D、原已知goal[t+24]、原shared模型和geometry-lift freeze/consume。
混合在解码后的原生命令空间、最终裁剪之前进行。腕部indices0..5，手指6/8/10/12/14/15，
其余耦合命令仍保持student零值，耦合由原生环境处理。记录混合前student_intended、
teacher_actions、逐行teacher_obs、混合后intended/applied/实际PD与既有预测/测量日志。
manifest显式teacher intervention、training_allowed=False和三行角色。

独立audit分别验证未混合student输出解码、组件替换、最终命令裁剪和实际PD、两行本行
H4/87D/goal/clock与held计数；不得因加入oracle跳过旧student预测检查。

## 判读

执行前CPU轨迹定位：读取此前shared7k三条固定面板的已审计原始日志，检查学生腕部
相对teacher的偏离时刻是否早于teacher首次接近物体，及学生是否曾接近物体。若已在
接触前大幅漂移，则优先解读腕部oracle能否恢复到达物体的能力；若接触前正常而接触后
失稳，则把手指/接触作为主候选。这是同episode描述统计，不把teacher后续动作与学生
不同状态的动作逐帧差当作监督误差。纯数组计算使用CPU，无模型forward。

该CPU检查已完成：三行initial q/hand/object完全相同。shared7k腕部相对teacher首次
偏离5cm在飞机/球/杯第3/4/4帧，teacher首次surface gap<=3cm在42/27/43帧。
学生全episode最小gap为0.444/0.250/0.536m，均未接近；首拍相同状态的实际PD XYZ误差
为58.55/51.12/22.37mm。说明此三条失败在接触前已出现，不能只解释为握不牢；不等于
证明唯一根因是腕部或信息缺失。结果及图在动作表示run的approach-timing目录，
工具summarize_native_approach.py。该图已查看，与数组统计一致。

本次row0 teacher连续held>=45帧才解释两种干预，不能沿用此前teacher62帧。
沿用surface gap<=1cm、抬升>=3cm、排除桌面支撑的held定义。
若只有teacher腕部救回held，优先研究student腕部跟踪/参考信息；若只有teacher手指救回，
优先研究闭合命令/接触输入。若两者都不能救回，仅说明当前单组件干预不足，保留交互
和输入缺失假设；两者都救回也不能据此认定任何组件原本正确。它们均为teacher-assisted
诊断，不计作学生自主成功。最小判别信号出现后再考虑第二条sphere141确认；不先扫面板。

## Decision Note / 预算与停止条件

当前动作表示2k配对Probe优先，GPU0..7均忙时不启动本项GPU计算。用户已临时授权剩余卡实验，
因此只在该配对pipeline成功完成、两臂所有学生均未held45、GPU6/7释放且nvidia-smi确认
空闲后执行；若配对已有held45正信号则跳过本项，优先分析该信号。不停止其他训练。
监督器只附着已确认存活的upstream PID和/proc启动身份，最多等2h，不自行重启上游；
上游失败/退出/替换则本项失败退出。代码/输入固定在执行前提交与工作树。

4-step混控smoke及audit通过后，两卡并行：GPU6上述149步组件诊断；GPU7使用同一shared7k
做一条自主TRAIN资格对照s1_airplane_pass_1、195controls、seed148/K1，无任何teacher混入。
该TRAIN候选在主动作表示卡的train-panel-audit.json中按字典序/同pass/strict startup选择，
不是挑结果。自身row0 teacher held45认证后才解读训练集表现；teacher失败只记未认证，
不自动追加序列。若TRAIN能抓而已有val不能，优先泛化；若TRAIN也不能，不能只归因val
split，继续检查闭环拟合。它不是新增数据量/跨手因果实验。

首条及TRAIN含smoke预计<=30min，单组native launcher<=600s、每组CPU audit<=120s、总产物<=4GB、
磁盘保留20GiB；失败、非有限、合同不匹配停止自己的组。无需新长期身份或外部写入。
当前只是已实现并测试的待执行诊断，git_commit字段为准备前HEAD，实际执行前必须更新。

准备验证：native_execution 8项单测通过，包含own-row组件替换/耦合维保持/输入无修改；
新audit在原control-airplane和forecast-mug日志回归通过，原18项合同与held复算均保留；
语法和仓库verify通过。尚未运行GPU工程smoke或采集oracle结果，不能据此形成方法结论。
条件接续监督器已独立只读审查通过：上游条件门、PID身份、GPU空闲、smoke顺序、固定
输入与两组角色均未见blocker。其CPU audit增加120s timeout，native沿用原有600s监督。

## Limitations / future evidence

teacher反馈改变学生状态分布，结果只能定位本面板上哪类辅助更有帮助，不能证明唯一
根因。只有一seed且短episode，不作跨手成功率比较。TRAIN/val差异另由动作表示Probe
的TRAIN候选资格对照处理，不能用本oracle代替泛化检验。
