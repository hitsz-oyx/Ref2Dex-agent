# HF25：接触状态条件的动作响应信息

experiment_id: P-20261002-contact-dependent-motor-information
family: HF25
probe_index_in_family: 1
kind: Decision / reused-fit-cal nonlinear physical response Probe
status: DRAFT

封存说明（2026-10-02）：用户准备开始新路线。本卡保留未执行设计，不是活跃
实验或结果。HF25没有登记到研究队列、没有消费科学Probe，也没有任何非线性
模型权重。以下训练结构、参数与预算是原草案，未通过源数据支持预检。

实际预检：旧cal989条第一周期转移有16条接触丢失，低于预设24例支持门；
当前mesh clear325条，真正clear→unclear29条。655条next unclear不等于655
次掉落。完整计数及12个源文件SHA见
[标签预检](P-20261002-contact-dependent-motor-information-preflight.json)。

[逐周期随机采集器草稿](../../../scripts/collect_randomized_motor_cycles.py)已编写，
仅Python语法检查通过；独立审计器与runner尚未实现，未运行原生GPU smoke，
未产生新数据。未来恢复须先补执行验证并重新固定数据源合同，不能直接使用
以下旧第一周期源草案启动拟合。

本实验区分：接触历史/指端几何条件的非线性响应是否比state与shuffled提供
可用的native动作后果信息；通过才固定新的实际短片段控制，失败停止本配方。
最便宜方法是复用真实随机化源，先避免新增仿真或直接做成功率。设计见
[接触响应决定](../../decisions/D-20261002-contact-dependent-motor-response.md)。

只用HF19 source-r1/571–582的第一实际控制周期；原SHA12651 group bucket
fit<50/cal50–69；held>=70不构造特征/目标，排除570仅工程、不用于科学训练。
原actual概率[.2,.2,.1,.1,.1,.1,.1,.1]，验证float32存储后归一到数学总和1。
每个当前候选真实nativePD的12独立通道减其概率均值；不能把这个初始随机
分配假设施加到后续已被动作改变的pre-state。

current输入：history10x69；53维原state物理上下文加共享mean_PD-current_q12；
6节点object-frame相对位置/速度、native节点q/dq各6、身份6、对应finger
intermediate/distal力signed-log/mg3、norm-log1及availability1，共29。
Palm没有该收集力，availability=0；五指对应当前收集body净力，非tip接触对。
未来只作监督。history/context/nodes归一均仅fit，std>=.001、clip8，中心动作
按fit实际中心PD RMS>=1e-4缩放，不裁剪以免破坏中心化。

state encoder：GRU69→64，context65→64，shared node29→64→64，pool mean；
三者concat192→128→128，state head15（物理7+事件8）。response复制已训练
encoder，node head192→128→90，按6节点x6通道与当前中心native动作做线性
读出到15；每候选修正减完整候选的概率加权均值。所有候选平均physics/logits
回到冻结state，zero所有中心动作精确回到state。event softmax概率均值不作
该保证，event joint/coherence仍须成立。

state成员seed17681–17683，各800更新；各成员独立response cm/shuffled各
500更新。response只学相对冻结state的修正，encoder可训练，state完全冻结。
shuffled只打乱fit中心候选动作与actual选择（seed=member+1000），当前状态/
标签/候选平均责任不变，cal均用真实候选。batch128、Adam(lr state1e-3,
response5e-4,weight_decay1e-4)、clip1，response最后层zero初始化，coefficient
L2 .001。最终固定更新checkpoint，不按cal选epoch、seed或成员。

物理标签7：actual dv3-gravity*dt(.1m/s)、actual dp3-current_v*dt(.005m)、
dCLR-current_vz*dt(.002m)。8类下一周期event=hand_presence*4+object_presence*2+
clear；presence依据rawforce/mg>.1、CLR>=2mm。loss=smoothL1物理7+event CE，
response另加系数L2。当前force不当周期平均力。支持高度为P111乘next正高度，
next高度由当前真实高度+CV位移+预测dp残差构成；不增加直接任务收益头。

cal支持>=256record/24episode/8组，实际joint-loss>=24且clear-loss>=24案例。
原门：支持高度MAE比state/shuffled各低5%，contactloss Brier各低10%，object
dv向量RMSE各低10%；dp向量与CLR RMSE均不高于state的105%。事件概率联合
一致、数值有限、base冻结、全候选期望physics/logit修正<=2e-5必须通过。
全部门通过PROMISING，支持不足UNCLEAR，否则UNPROMISING；不降低门或追加
更新。cal上的known-propensity政策排序只描述性报告，不作正式真实控制收益。

描述性selector：3成员支持高度均值减1.645std，候选相对Cup支持高度的
paired成员增益下界>0且joint概率不降>2pp、clear-loss概率不增>2pp才改变，
否则Cup；state-only物理预测候选恒定，明确回Cup。shuffled相同规则；base0与
fit-only最佳固定候选并列，已执行所选candidate的支持计数与IPW值保存。只用
actual转移作为政策估计，没有全候选反事实真值/regret。

排除570工程训练state5/response5，仅证明current-only/梯度/中心化/冻结/事件
和实际PD接线；未来污染不得改变输入。科学模型从新seed重新初始化，不继承
工程权重。总slot<=900秒/512MiB，准备60秒、排除工程、科学拟合、完整独立
GPU审计全部计入；旧源/HD04成本并列另报。任意input/hash/预算失败停止保留。
通过后slot2另固定独立原生重新决策及全部对照，之后才策略训练/完整稳定抓取。
