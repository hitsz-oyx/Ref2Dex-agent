# HD04：随机化中心动作的短周期物理响应资格

experiment_id: P-20261002-randomized-motor-response
family: HD04
probe_index_in_family: 1
kind: Decision / reused-fit-cal linear response qualification
status: ACTIVE

问题与下一步见[动作响应决定](../../decisions/D-20261002-randomized-motor-effects.md)。
只用HF19 `P-20261002-contact-geometry-source-r1` 全12seed571–582已审计记录，
原SHA12651 group split：fit<50，cal50–69，held>=70只读取组身份以剔除，
不构造held特征/目标。排除工程、HF24新反馈及后验失败数据不混入此诊断。
只用每H10第一步：所有候选起始PD已在分配前生成，实际assignment按已知
[.2,.2,.1,.1,.1,.1,.1,.1]。中心动作z=actualPD-概率加权候选PD均值，12个
独立DOF由真实执行器定义；完整候选中心残差的概率均值必须数值为零。

监督7个实际物理残差：object dv3扣gravity*dt、object dp3扣current_v*dt、
mesh dCLR扣current_vz*dt；归一尺度.1m/s、.005m、.002m。先验不是未来真值，
力仅当前观察特征，不当周期平均力。第一实际PD、rawforce联系、组身份、当前
特征与实际post状态独立复算；物理信息不等于任务收益。

state基线线性ridge输入q18/qvel18/objectquat4/objectv3/omega3/object-palmXYZ3/
log(hand_maxforce/mg,objectforce/mg)2/CLR1/positiveheight1（53维，加截距）。
geometry系数phi=[1,object-palmXYZ3,objectv-palm_velocity3,两个logforce,
CLR,positiveheight]11维。fit-only归一/clip8；中心动作用fit观测RMS、最低1e-4
按各DOF缩放，不裁剪中心动作以破坏零均值。
先拟合state基线，再拟合G(s)z的132维响应到真实残差减冻结state输出。
ridge固定lambda=.01，截距也同样正则，所有模型一次solve，无超参/epoch扫描。
shuffled只对fit中心动作跨行打乱(seed16671)，phi/目标不动，state与归一共享；
cal都输入真实当前动作。比较cm、state-only、shuffled与CV/重力先验，保存完整
所有候选预测的加权均值应严格等于state基线以及响应矩阵谱/动作覆盖。

cal支持>=256第一步记录/24episode/8组，真实|dv|>.05m/s动态记录>=64。
资格门：全cal与动态cal速度向量RMSE都比state/shuffled各低10%；物体位移
向量和meshCLR RMSE各比两控制低5%，且全cal速度不劣于零dv常速度基线。
分组500次90%bootstrap(seed16672)只描述误差变化、不作为新的正式科学结论。
通过PROMISING，支持不足UNCLEAR，否则UNPROMISING；不改门、补数据或调ridge。
失败停止线性配方，不从简单线性失败推出非线性物理响应不存在。

GPU5默认，整诊断<=300秒/64MiB，准备预留30秒；旧源采集/模型成本另报。
CPU仅文件/统计，GPU做拟合与批量模型预测/分解。工程与独立统计审计必须通过，
未来数据只作训练监督；未执行候选只有模型预测，不当真值。没有候选收益、
Cm实际控制、PPO或稳定抓取结论；所有旧标签及最终Mission保留。
