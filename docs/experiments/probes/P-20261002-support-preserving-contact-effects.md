# HF22：生成分布适配与显式接触损失信息

experiment_id: P-20261002-support-preserving-contact-effects
family: HF22
probe_index_in_family: 1
kind: Decision / reused actual-source model Probe
status: ACTIVE

区分：真实生成分布适配与显式contact-loss是否改善可用于动作生成的后果信息，
还是增加一个逻辑输出仍不足。通过进入带接触保护的生成/独立真实执行，失败
停止这套适配配方，不扫更新/seed/门限。最便宜是复用全部已审计事实转移。
依据[决定](../../decisions/D-20261002-generated-support-preserving-cm.md)。

union固定6263窗口：HF19/HF20旧5018加HF21新1245，origin19/20/21。旧数据
与新数据均SHA12651 motion/start bucket<50 fit、50–69 cal、>=70 reused-held。
训练旧fit2532、新fit624；所有cal/held不进训练、不改变归一化或模型选择。
旧及新cal/held已经用于路线决定，本卡是复用信息Probe，不是独立Validation。
所有actual source/planning/full audits/statistical audits/checkpoint/代码/本卡hash固定；
工程seed623排除，不把其33窗口加入训练。无未执行候选伪标签。

输入保持pre-only69×10 history、force/mg、whole-mesh CLR、1442当前原生
观测、当前六专家bank/Cup PD、候选PD与权重差、已知H10律。future仅实际标签。
旧事件8类/条件高度/支持抬升/第一步43维物理/几何loss保持，新增H10中任一步
hand或object netforce/mg<=.1的contact-loss。joint末3步共同存在为event6+7，
contact-loss概率=1-Pjoint+Pjoint*sigmoid(新条件损失头)，必不低于1-Pjoint。
此为接触存在代理，不是识别接触对；几何loss与接触loss分开，非最终掉落验收。

新增global第20行，旧19行/全部backbone和其他head从HF21各模式各成员原样
初始化；新行weight0/bias-3。cm/state-only/shuffled/direct四模式均同容量。
原三成员12811–12813对应原checkpoint，Torch适配seed13811–13813；各500次
Adam更新、lr1e-4/weight_decay1e-4/clip1/batch256，每批128旧fit+128新fit，
同成员各模式初始化对应旧模式、采样索引相同。规范化完全继承旧fit-only。
shuffled在fit origin×early/clear内打乱node-action/law，context保留。
Cm/state/shuffled在旧原目标上增加接触loss BCE及终端joint存在时的条件损失
logit BCE；direct仍仅训练独立支持高度分数，不使用未训练事件/物理/接触头。

保存全部actual三成员预测与8候选预测（候选无反事实标签）。冻结旧Cm的完整
NN预测作为未适配参考；旧Cm没有contact-loss头，不把1-Pjoint当受训contact-loss。
纯CPU标签/统计允许，所有训练与重复NN预测GPU5（启动时重新检查空闲）。

固定信息门：新fit/cal/reused-held各>=150窗口/32episodes/8groups；new-held
contact-loss真/假各>=32；实际generated arms2–4>=60。Cm在new-held整体和
generated subset的高度MAE均比未适配旧Cm改善>=5%；new-held终端joint Brier
和contact-loss Brier均比适配state-only/shuffled改善>=5%。支持事件/lift/几何loss
Brier及object-dv/relative-position RMSE不差于未适配Cm及适配state/shuffled10%；
旧reused-held高度与joint-support Brier不差于未适配Cm10%，避免丢掉旧能力。
direct高度误差照实报告，不要求其未训练物理头非劣。所有包含关系误差<=1e-6，
输出有限。全门通过PROMISING，支持不足UNCLEAR，否则UNPROMISING。

先用排除623验证原始force标签、PD/物理输入、future-poison、20行warm-start
保留旧输出、接触损失逻辑和有限动作梯度；模型无更新。随后固定GPU fit与
独立rawforce/世界物理标签/完整NN/原门审计，whole<=1800秒/512MiB，预留60秒
准备、工程实际计费；不得覆盖旧checkpoint。共享旧源及HF21累计2261.93秒另报。
不启动新rollout、PPO或最终成功率矩阵，只有信息过门才执行后续生成验证。
