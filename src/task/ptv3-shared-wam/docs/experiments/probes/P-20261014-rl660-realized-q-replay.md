---
schema: ref2dex.probe.v2
probe_id: P-20261014-rl660-realized-q-replay
experiment_id: P-20261014-rl660-realized-q-replay
date: 2026-10-14
task: ptv3-shared-wam
branch: main
git_commit: 625e3c135e5ff65e71390b9e439e84ec0511e61f
claim_id: C1
hypothesis_family: HF-rl660-realized-q-replay
probe_index_in_family: 1
seed_pool: probe
seeds: [148, 149]
decision_changed_if_positive: fit a small realized-q retargeter before considering residual RL
decision_changed_if_negative: test trajectory tracking compensation before relabeling or expanding supervised training
status: UNCLEAR
run_id: rl660-realized-q-replay-20261014-r1
---

# RL660 realized-q可执行性与原PD重放校准

Result: UNCLEAR，待工程门及独立审计。
Decision: 依用户ref2_2后的继续探索授权，先做无R/forecaster的最小物理上限检查；现有联合微调和远端预训练继续。

## Decision Note 与研究动机

服务于MISSION独立执行底座。Class: Decision。区分“实现未来q可直接作为位置目标”与
“相同低层控制器仍需动力学补偿”。最便宜方法为现有dev记录的纯命令物理重放，不训练新网络。
前轮标签审计确认PD不是实现q，但没有证明标签噪声下界或任何K的必然发散。
需要先校准原始命令重放；校准失败只说明此实验无法解释，不关闭R或q-label路线。

首批固定dev的各物体字典序第一条：s10_airplane_pass_1、s10_mug_pass_1、
s4_spheresmall_pass_1；不按重放结果选样。148为配对物理seed，149留作必要工程修复，
不是独立统计重复。保持30Hz、原腕部Kp200/Kd20、手指Kp100/Kd10、原coupling/clamp。

## 协议与指标

每序列6个frame0相同重置的独立环境：recorded_native（原applied逐tick）、recorded_PD
（原绝对PD用live q逆转换）、realized_q_full（实际q[t+1]作为目标），另3行先原native
roll-in至准入ticks的首/中/末，再各替换8 controls的实际q目标，随后恢复原命令。
物理q目标先按12独立维投影到原controller耦合；不直接绕开coupling驱动18维。
K4查询冻结8个绝对q目标，4拍后重取；K1会消费相同oracle目标，不构成独立方法臂。
记录输入/实际命令、PD、query clock、q/dq/手点/物体/速度、接触力与表面/桌面支撑。
无actor推理，legacy player只构建环境。不存在官方权重初始化独立策略。

首先检查frame0五类状态max误差<=1e-4、实际PD独立复算误差<=1e-6。原native连续重放
手点EPE和物体位置mean均<=5mm方可解释全程q门。窗口则要求原native在同窗口校准，
且替换前fork与原native五类状态max差<=1e-4。保留失败和所有时间序列，不事后换阈值。
hand通过暂定mean<=10mm/p95<=20mm；物体位置mean<=10mm，并单列held45连续帧。
原native无hold45时该序列不能检验抓取保持，但仍可检验tracking。
q全程若hand/物体/保持均通过为局部A；hand通过物体/保持失败为B；hand失败为C。
这些是Probe信号，不证明误差下界或监督/RL分工的唯一因果解释。8拍替换仅报告窗口
保持/接触，不能冒充45拍持续持物；全程按已有partial_hold45语义不判断最终放置。

## 资源与停止

GPU6，开始前确认空闲。根STATE明确远端重建1/3已STOPPED，远端训练0/2、本地微调5/7
加本Probe和另一路GPU4合计最多6卡；不停止其他任务。SSH暂时banner超时，以最新
远端shadow与明确STOPPED记录核对，若状态出现冲突则不启动/停止本Probe。
全实验最多20 GPU-min（含smoke/失败修复/追加判别），每run<=300s，最多6个科学运行；
首个smoke4 controls<=120s。输出与scratch合计<=500MB，剩余空间>=20GiB。
源码固定且不修改运行中训练的src包；新工具独立于两个训练进程。超时、非有限、
外部GPU PID、状态/时钟不一致则停止自己的进程并保留失败；不提高Kp或启动PPO。

## Limitations / future evidence

仅3序列、单物理seed、使用真实未来状态的oracle上限；不评价学得R或forecast。
重置可匹配可观测状态，接触solver状态无法从旧npz精确恢复；原native校准是解释前提。
realized-q在当前输入下的可辨识性、FK误差、R-only拟合、predicted-tau执行与residual RL
均延后，只在此gate能改变决策时推进。若q失败先区分阻尼/跟踪滞后，不能直接称接触力缺失。

## 首批结果与下一项最小判别（追加运行前）

实现0680af03/0d1a072b。首个smoke8.72s在legacy Runner无load_path失败，未进入physics；
修复无checkpoint的runtime壳后r2四控制10.99s、独立审计通过。三条科学运行均完成，
actual command/PD/clock独立审计通过。原native重放hand/object mean为15.72/12.13、
28.72/26.45、6.39/6.49mm，均未过预定5mm校准，因此保留原full分类UNCLEAR，
不将其升级为C或关闭q标签路线。相同初态/首拍PD不保证不同环境行的下一拍完全相同；
接触solver隐状态/环境行差异尚未隔离。当前target wrapper已有持久tensor引用，
生命周期假设可由代码排除，未为此修改主仿真或正在训练的源码。

另一个预先声明的原绝对PD臂可以重现记录：hand/object mean分别1.220/4.547、
0.939/2.590、0.000091/0.00399mm；q_full则hand mean85.65/60.97/63.49mm，
held分别5/0/74，PD臂63/89/74。纯q在起步8拍也有明显手部滞后，不能直接称“仅缺接触预紧”。
旧窗口中/晚段fork误差超过1e-4，仍不作为精确物理fork的因果比较。

Class: Decision。下一项只改确定性的trajectory tracking补偿：同3序列、相同seed和
控制器，在q目标上增加(Kd/Kp)*v_desired；所有组Kd/Kp=.1s，v_desired仅由相邻实际q
以30Hz差分得到，故目标=q[t+1]+3*(q[t+1]-q[t])，不使用teacher PD/native或真实dq。
之后照常couple、用live q逆转换/clamp。比较与原q臂的手/物体误差及held，原PD臂
保持不变作为每次执行校准。没有调Kp/Kd、没有gain搜索、不训练R、不启动PPO。
总科学运行数现在最多6（原3+FF3），仍计入原20 GPU-min/500MB，不新增预算。
原native校准和原分类不追溯放宽；新增单列supplemental_PD_calibration_pass，只有
hand/object均<=5mm才解释对应FF与q跟踪对照。轨迹跟踪mean改善>=30%且不降低
持物才记局部PROMISING，否则保留UNCLEAR或UNPROMISING（仅此控制律）。

## FF后追加的单项定位Decision（运行前）

FF三个手点mean为19.05/15.35/16.09mm，相对raw q降低77.8%/74.8%/74.7%；
held为62/0/70。mug手误差下降但物体仍约171mm/held0，sphere held74降70，
未通过“不降低持物”以及原hand门，不宣布nominal已合格。FF airplane的PD臂
object mean6.70mm超过补充5mm校准，保留该限制，不追溯放宽。

Class: Decision。在同一mug/seed148仅把q臂的手指部分改回原记录PD，腕部保持
q+固定阻尼补偿。若恢复held，优先定位手指闭合/预紧映射；若仍失败，不能只把问题
归给手指，须继续分离腕部姿态/接触时序。它是oracle分组替换，不是独立学习策略。
这是第7个、也是本卡最后一个科学运行，追加预计25s；最多运行数6改7，20 GPU-min
和500MB原资源上限均不增加。完成后汇总，不继续扩展controller sweep或长训。
