# 冻结Cm生成动作原生工程：通过，收益待测

[完成记录](P-20261002-optimized-contact-native-engineering-completion-r2.json)：
GPU1、96个完整H10窗口/78episodes、960实际步、79规划批次；源进程与独立
审计均exit0。原240秒超时记录保留；r2相同模型/规则/seed，改420秒原生上限，
原生父进程305.98秒，完整独立审计192.71秒。
含失败、GPU梯度工程和准备成本共853.32秒/20.05MiB；共享HF19预训练
源/模型成本28.17分钟/498.84MiB另报，不把它当免费输入或新模型训练。

[审计](P-20261002-optimized-contact-native-audit-r2.json)逐窗口复算实际命令、
耦合PD、专家范围、固定旋转、原始力/mg、全meshCLR和全部pre/post几何；
960步最大PD差1.19e-7、旋转差0、mesh/保留支持高度标签差0。
全部79批pre-only输入复算0；proposal及allocation RNG复现0；三模式
32步规划、最终权重/轨迹/预测复现0；完整NN与缓存后果头复现0；参数冻结。
补充依赖与旧fit输入hash也通过，不重写旧HF19信息/选择失败门。

Cm在71/96起点生成不同cup权重，direct86/96、shuffled92/96。
这些是候选提议。实际分配到Cm的7个窗口中，6个窗口/60实际步的PD不同
于同一实际状态下的cup指令；direct7/7、shuffled14/14。
[实际执行差异](P-20261002-optimized-contact-actuation-engineering-r2.json)仅证明
指令变化，比较使用同一实际轨迹上的即时cup目标，不是cup反事实后果。

工程确认生成器能直接影响真实物理指令；没有候选收益、闭环提升、最终策略
学习或稳定抓取结论。下一步固定独立随机源机会Probe，比较Cm生成、direct/
shuffled生成、state-only、base、最佳固定与随机程序，并保留重复槽噪声门。
完整goal和C3仍OPEN。
