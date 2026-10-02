# 联合后果生成器的原生执行工程

classification: ENGINEERING
status: ACTIVE

HF21 slot1的信息门PROMISING且完整独立审计通过。这里不增加科学Probe，也不
把预测进步升级为控制收益。先验证新冻结模型能真正执行H10程序，且所有pre-only
输入、分配、规划、实际PD/全mesh/原始force和标签可独立复算。

排除seed610，96原生GPU环境，assignment15610，最多180tick，每episode的
early/clear各最多一个H10窗口；选择条件沿用当前joint3、rest>=5mm、warmup10、
剩余至少11步，末10tick不启动新片段。动作分配仍是10slots映射[0,1,2,3,4,5,6,7,0,1]，
候选先生成再随机分配。三生成器只优化输入logits32步、lr.15/clip1，模型不更新。
score为joint-support概率×条件高度，置信增益mean−1.645std必须>0；最终候选
几何loss增量<=.02、support降幅<=.05且当前/候选规范化输入不超过8，才可
替代cup；否则cup。direct-score只用独立评分头，风险为state-only静态预测。

当前六专家反馈每步重算，系数H10固定、旋转锚定当前目标；真实env.step执行
raw命令，native映射及联动关节保持原任务合同。记录current1442原生观测及
两侧每步物理观测，模型/专家冻结，schema使用独立structured来源，不冒充旧源。

单空闲GPU1，native内部wall<=420秒/parent480秒，完整独立审计<=360秒；
native+audit整个工程<=900秒/128MiB。本工程及梯度/拟合/准备成本分别记录，
借用旧源/预训练另报；范围仍在HF21 slot1的1800秒/512MiB之内。不新增训练、
不看工程seed的效果来调参，工程通过后再固定slot2的新独立真实机会合同。
