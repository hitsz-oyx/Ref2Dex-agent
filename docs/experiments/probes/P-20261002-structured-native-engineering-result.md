# 结构化 Cm 的实际控制路径已通过原生工程

排除seed610，GPU1原生96环境/180tick，共72个完整H10窗口、53episode、720
实际步，含19个已离桌窗口。[工程合同](P-20261002-structured-native-engineering.md)
不变；没有根据该seed的任务后果调参，没有科学收益标签。

实际6个Cm分配窗口中4窗口/40步的PD指令不同于同一实际状态上重算的cup。
这证明新Cm输出能改变实际执行，不意味着Cup另一条轨迹的反事实后果已知。
47批全current输入、proposal/allocator RNG、三生成器各32步优化、最终权重/
score/loss/support、完整NN和硬可行性独立复现均0；first-candidate指令差
1.49e-8。所有模型/专家冻结，direct-score风险明确来自state-only静态先验。

全部720步真实native PD、联动关节、旋转、rawforce/mg、全mesh CLR/任务标签、
两侧世界/原生相对几何均通过。PD最大差1.20e-7，旋转/mesh/score标签差0，
pre/post相对位置差2.17e-7m；没有用endpoint净力推演速度或冒充手物接触对。
native/audit/parent均正常exit0，GPU已释放。

完整native+审计308.04秒；加输入工程/准备/12模型拟合/预测/全审计239.98秒、
梯度生成工程6.07秒，slot1累计554.09秒，在1800秒/512MiB内。输出包含原生
记录、全部规划、Gym缓存、模型和预测；共享旧源/预训练另报。完整成本与冻结
hash见[完成索引](P-20261002-structured-native-engineering-completion-r1.json)。

现在可固定HF21 slot2新独立真实机会合同：保持已通过的信息模型和生成器，
比较Cm与强cup/state-only、direct-score、shuffled、base和fit选定最佳固定。
新源需要重复观察/重新规划及足够实际分配支持；旧held、这次工程、预测分数
均不能替代真实收益。完整作用链条与策略学习/最终稳定抓取目标仍未完成。
