---
schema: ref2dex.probe.v2
probe_id: P-20261016-wrist-shared-grasp
experiment_id: P-20261016-wrist-shared-grasp
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: a2c83098
claim_id: C1
hypothesis_family: HF-wrist-shared-grasp
probe_index_in_family: 1
seed_pool: probe
seeds: [148]
decision_changed_if_positive: retain the matched live shared-conditioned wrist interface for a larger fixed behavior panel
decision_changed_if_negative: retain direct current-input wrist control and diagnose remaining finger/recovery limitations
status: UNCLEAR
run_id: wrist-shared-grasp-20261016-r1
---

# 同一腕部MLP±shared的实际抓取评估

Result: UNCLEAR（shared行为收益）：两组在固定三轨迹上均有两条观察到hold45，shared增加飞机最长保持、减少球/杯保持；none飞机末段仍持物，不能只凭阈值当作完整失败。
Decision: 保留811D直通腕部MLP作为行为基线，不将aligned统一替换到全部对象；若继续比较，扩大预先固定的行为面板，并检查接近/保持/释放阶段，而非用离线均值或单次hold45选最终策略。

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

## 实际执行与结果

7项腕部输入/保留原路径/源绑定/随机状态/行序测试通过；4 controls smoke完成，
实际PD、六维替换、原手指、raw811当前输入、server pooled行序与CPU模型重放均通过。
独立只读review未见阻断问题。随后GPU1/2/3固定三个对象并行，约78.70s（含三个完整
CPU执行审计），均COMPLETED/independent_audit PASS，无workspace退出或早停。

| 对象 | teacher最长保持 | none腕部MLP | aligned shared腕部MLP |
| --- | ---: | ---: | ---: |
| 飞机 | 63 | 40 | 60 |
| 球 | 79 | 75 | 56 |
| 杯子 | 98 | 58 | 23 |

单位为30Hz帧。均使用同一个原shared手指控制器，但各自腕部造成状态分化后，手指输出
可以不同；没有teacher动作干预学生。上述是局部连续持物指标，不是完整参考跟踪/放置
成功率。三个对象均teacher-positive，两组各两条观察到>=45帧，不能将2/3外推为总体成功率。

### 保持区间和观察截断

none飞机有[54,93]40帧及[113,149]37帧两段保持，末段到评估终点仍持续，终点抬升0.699m、
手物表面gap0.0020m；因此“未观察到连续45帧”不等同无法抓起或后续不会达到45帧。
aligned飞机[57,116]连续60帧，终点已失去保持、回到近初始高度；不能只凭最长保持
说它全过程更好或把释放自动判断成掉落。是否满足任务释放/放置须另按参考目标审查。

none球[42,116]75帧，aligned球[46,101]56帧且末段[137,141]5帧仍保持。
none杯[65,122]58帧，末段[167,172]6帧保持；aligned杯仅[62,84]23帧，随后明显落下，
终点低于初始1.070m、手物gap2.817m。这是本面板shared的明确退化个案，仍非全局负结论。

两学生的腕部clipping均0；手指clipping为飞机21.48%/3.36%、球8.51%/7.80%、
杯4.07%/37.21%（none/aligned）。不能仅据手指饱和把原因归给手指；腕部状态偏离也
可能改变接触和下游手指控制。这些是需要定位的伴随现象，不是因果干预证据。

### 数值与执行核对

所有三行初始q/dq/hand/object/velocity逐项相同。全过程CPU腕部命令重放max<=2.39e-7，
实际intended/clamp/PD合同及共享特征的own-row顺序均通过原容差。held在独立汇总中
从物体高度、surface/support/footprint重新计算，全部与保存结果一致。

另在GPU1对每条已存轨迹的0/32/64/末拍重建own-row H4、状态和goal，重新执行冻结WAM。
与保存live pooled的最大绝对差0.00804、base chunk差max0.00248（混合单位），将重算
pooled送入两MLP后的腕部XYZ差max0.069mm。没有把latent或chunk误差冒充物理位移，
也不声称推理逐位确定；具体逐tick值见live-wam-replay.json。该检查为所选12个tick，
不能替代所有时刻/所有状态的独立重算。没有放宽执行或CPU命令重放的容差。

每卡约19GiB，开始/64/128步采样GPU利用率、显存和ETA；已释放GPU1–3，GPU0未操作。
本轮约24MiB产物，低于2GiB预算；未新增训练，没有修改原示范或权重。

### 证据与适用范围

同run下comparison.json保存3组完整结果和audit；summary/results.json/csv保留每条件
保持区间、终点删失、腕/指裁剪及高度；summary/native-traces.png为实际高度和表面gap。
工具tools/audit/summarize_wrist_shared_grasp.py可向fresh目录重导出，无新仿真。

本轮说明新版腕部MLP替换后已有实际抓起/持物行为，不能把shared的离线退化直接当作
所有行为都退化；也未观察到shared在三个对象上的一致优势。单seed、三个预选轨迹、
单模型初始化、固定短horizon、近邻/高度代理指标仍限制结论；末段仍持物的片段存在
右删失。更多seed/固定对象面板、可视化动作质量、参考释放/放置度量作为future evidence，
当前不为补齐统计而无界扩跑。进入最终策略或正式claim之前仍需Validation。
