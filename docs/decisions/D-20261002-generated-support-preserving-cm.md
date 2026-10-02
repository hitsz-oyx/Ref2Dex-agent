# Decision Memo：生成分布适配与显式接触损失

问题：HF21已打通真实控制，但尚未优于强Cup/direct，下一步怎样增强Cm的职责？
Mission/C3不变，不对已用完2slot的HF21追加seed、重训或改门。

关键证据：8seed1245H10窗口/563episodes完整接受，原门UNPROMISING。
held实际45/73Cm窗口/450步改变Cup；高度+3.420mm，group90[-3.325,9.280]，
不足9.616mm噪声门。cal相对direct-3.164mm、held+3.487mm均区间跨0。
信息门曾PROMISING，但新cal实际生成Cm高度MAE17.628mm，旧整体9.980mm
不能代表新分布。平均生成臂高度偏差仅+1.932mm，不能归结为统一严重过估。

实现缺口：当前loss头是几何clearance损失，不是hand/object存在损失；生成门
约束P111（共同存在且clear），不能保证hand/object共同存在本身不下降。
缺少显式contact-loss职责，与用户的接触保持目标尚不完全对应。

选择新高层假设：在真实生成动作分布上适配，并让Cm显式预测和保护H10接触
损失。仅新source bucket<50的624事实窗口可加入训练，旧fit作回放；不把8候选
预测当真值，不用新cal/held拟合。增加H10中任一步hand/object代理丢失的实际
标签，终端joint概率由原8事件得到；使contact-loss概率至少为1-Pjoint以保持
必要关系，再加入共同存在与接触损失的直接生成约束。仍保留原物理/高度预测
及Cup、state-only、shuffled、direct对照，所有旧代码/checkpoint/结果冻结。

先固定新模型/标签合同和GPU信息Probe，检验适配与接触损失是否可学；新cal/held
已用于路线决定，只能称复用数据Probe。通过才预算新的独立真实源验证动作
收益，再设计策略训练，最终独立稳定抓取验收；失败复盘高层控制职责。
新路线最多2个Probe，首个模型/工程目标<=1800秒/512MiB/一张空闲GPU，
不得在训练时侵占当前他人GPU任务。精确seed/更新/指标在首个新卡启动前固定；
本Memo不是已启动实验或正向结论，不把再拟合当作最终效用证明。

无新外部授权边界；单会话直接执行。策略学习与完整Cm-on/off仍未完成。
