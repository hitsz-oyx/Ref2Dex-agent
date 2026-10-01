# 可学习分数与实际执行之间的缺口

2026-10-01，HF11 slot1关闭，机械诊断 Decision。新研究目标/claim不变。

事实：matched on/off各207123有效首episode步/840更新；稳定45ticks并无
后续drop为17/384vs13/384，四eval seed描述性t95差[−.872,+2.956]pp。
on的5065个独立评价决策、off5014个决策，原始option-ID argmax相对各自
先验全为0变化。不能把同seed的trained6/96 vsprior3/96当学习收益：访问
状态上未改变策略实际贪心命令，仿真/观察不重复仍能产生标签差异。

需要区分：网络根本没有学到相对动作偏好，还是学到的偏好被固定先验压住？
最便宜方法：仅在已保存评价输入上GPU重算最终policy分数，核对实际选择；
分开记录可学习分数与固定log先验。先验优势log(.90/.02)=3.80666，比较
学到的最大相对优势及移除先验后的离线argmax变化。不执行新动作、不使用
stable/release目标调参、不把离线不同argmax描述成实际收益。

结果若有学到的偏好但不足以越过3.806，下一路线把Cm知识初始化进可学习
策略，而不在执行层永久叠加固定先验；若没有偏好，优先检查目标/梯度和
物理输入，不追加同路线epochs。另一个已确认实现事实：原生reward是
reference body/object/interaction/contact imitation与energy乘积，不显式
奖励保持45ticks或防止后续drop；它与本次retention目标不能视为同一东西。
下一路线需先固定与保留抓取一致的共同任务reward，再进行独立matched设计。

只做一次GPU机械审计≤60秒/16MiB，不更新任意NN/actor/checkpoint，旧输入
与结果保留。此为工程路径诊断，不是额外HF11utility Probe或阈值重扫。
旧门UNPROMISING不改，C3OPEN；当前授权允许，无外部审批边界。
