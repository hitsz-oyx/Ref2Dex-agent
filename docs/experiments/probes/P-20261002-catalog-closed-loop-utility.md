# HF15 slot2：冻结 Cm 的真实短片段闭环作用

Decision Probe，HF15 2/2，与原始9模型拟合和6对照模型拟合共用60min/8GiB；
不重置预算。训练模型/校准/门限在新数据前冻结。问题：程序条件物理未来预测
在留出接触状态直接控制动作，是否优于状态局部收益评分、打乱程序标签的物理模型、
base和最佳固定程序？正向才进入策略训练；充分支持下失败则关闭本slot，不扫阈值。
最便宜实验：冻结已有六专家/Cm，随机化完整10步控制器窗口，执行2步后重观察。

- 数据：96env，每phase最多650tick，H10完整无terminal；先由base执行至
  连续3tick归一化双净力存在、中心高于rest3cm、全source mesh离桌2mm。
  只在原始`SHA256(9851/motion/start)%100>=70`留出初始组启用，其他组不参与。
  最多8window/firstepisode，6tickcooldown，所有部分/terminal窗口使采集失败。
- 五个独立控制器：catalog_cm、state_policy、catalog_shuffled、always_base4、
  always_fixed7；观察后私有随机分配完整窗口owner，各p=.2，命令巧合不合并p。
  前三者共享冻结calmargin+相对ensemble std、risk<=base+2pp、ret>=base−5pp、
  fit标准化abs>8回退。物理模型预测H10完整程序，但每次只执行2步再重规划；
  真实H10窗口标签属于这个闭环控制器，不冒充单一H10程序的训练真值。
- 程序0..5每帧按新obs调用六专家；6/7只锁2step当前rotation3..5，XYZ和
  fingers继续base/cup反馈。fixed7也每2步重取锚点，与原始整10步固定锚点不同。
- 每次重规划对同一实际观察保存五推荐、三模型分数/风险/std/OOD，以及history、
  原生pre-step obs、实际raw与PD、原始净力/质量/几何。影子推荐没有反事实后果。
  对实际12独立PD坐标与同状态base/fixed7命令比较，容差2e−5；命令变化不能
  替代收益。没有逐状态oracle，不能报告真实regret。
- 工程seed480/private11480，2window；科学seed481–492/private11481–11492，
  serial fresh GPU admission，仅允许GPU0/1，单native<=240s、parent<=290s。
  科学运行要求source/input与通过的工程匹配。3nets/mode及专家全部冻结，无PPO/V。

预先固定判别：每个policy>=96完整window、>=24episode、>=8初始组；Cm真实
改变base和fixed7的window分别>=48、>=12episode、>=8初始组，每window5次
重新观察决策。主收益是末3帧joint净力存在且meshCLR>=2mm时的正保留高度减
起点正高度，mm。已知p的IPW Cm-minus-control对四对照分别要求>=2mm、
初始组与episode两种cluster bootstrap1000次90%下界>0；几何CLR任一帧丢失
风险差point<=+2pp，末3帧joint净力存在差point>=−5pp。所有门通过PROMISING；
支持不足UNCLEAR；充分支持下任何效用/覆盖门失败UNPROMISING。风险门是
探索点估计，不是正式安全结论。最终持续抓取/策略学习收益仍待后续验证。

前置成果：Cm255/704留出clear提议，state-policy279/704，shuffle0；这是
覆盖率，不是真实改变或收益。原始raw-action解码器0提议的失败保留。
预算起点85.238s/11,738,437B +GPU selector audit3.289s；另计launcher审计/
工程/所有失败尝试/科学采集，截止执行末加终态回放与统计。
