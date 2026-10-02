# 冻结Cm动作生成：GPU梯度工程已通过，原生执行待接入

[路线决定](../../decisions/D-20261002-cm-optimized-action-generation.md)保留
完整目标的真实候选机会、后果学习、短片段执行/重新观察、后续策略训练要求。
HF19 slot1原门UNPROMISING、slot2关闭不改；本工程没有登记新科学Probe。

新[生成器](../../../src/task/CmResidual/optimized_contact_actions.py)只接受当前
history/native、原始力、mg、meshCLR/rest；独立live输入与既有已审计起点
最大误差7.63e-6（scaled skip）、physical4.77e-7、node5.96e-8。
冻结HF19三成员Cm/direct-score/shuffled，每次先缓存当前state/history编码；
目标为H10联合支持高度相对cup的均值，扣1.645倍成员分歧与100mm的预测
几何风险超额惩罚。已知law保持旋转锚定，六块专家权重softmax simplex。
Adam输入权重logits固定32步、lr .15、clip1，初始化.95cup+.05/6；cup作为
可行基准保留，目标没有改善时回退cup。所有网络eval/requires_grad=False。
不对模型增加训练、不读取未来、不依赖长期V。

GPU1上的[工程记录r2](P-20261002-optimized-contact-actions-engineering-r2.json)
验证24个真实seed570状态/完整六专家bank：Cm生成14/24个不同权重，direct
21/24、shuffled21/24。三种模式均32步；权重和最大误差1.79e-7；全部XYZ/
手指命令在对应专家范围，原生PD旋转锚定误差0；冻结参数fingerprint不变。
同一在线输入下缓存后果头与完整冻结网络误差0；也测试了原生collector的
no_grad调用条件，输入梯度仍能运行。总GPU工程4.62秒。
所报预测score提升不是实际抬升/收益；本工程没有执行新生成动作。

首次缓存工程检查将独立CPU组装的特征与GPU在线组装比较，score差3.05e-5mm
触发极小绝对限；[失败快照](P-20261002-optimized-contact-actions-engineering-failure-r1.json)
保留，保守计10秒。修复比较范围：在线输入与真实起点独立核对，缓存与完整
网络则使用相同在线输入。没有调整网络、优化参数或科学门。

下一工程步骤是接入新的原生源采集，记录Cm/直接评分/shuffled生成程序的
完整参数/规划轨迹，实际随机分配与执行H10；核对真实PD、当前输入、概率、
全pre/post/force/mesh标签及冻结NN回放。原生工程通过后才固定并登记新的
候选机会科学卡。完整goal与C3仍OPEN，当前没有后台实验或训练进程。
