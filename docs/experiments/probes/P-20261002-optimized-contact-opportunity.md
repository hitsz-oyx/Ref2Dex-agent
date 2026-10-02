# HF20：冻结Cm主动生成程序的真实机会

experiment_id: P-20261002-optimized-contact-opportunity
family: HF20
probe_index_in_family: 1
kind: Decision / Probe
status: ACTIVE

问题：已有H10后果模型的动作梯度能否产生比强固定cup及直接评分/打乱模型
更好的真实接触程序？区别于HF19随机候选排序：Cm生成系数并直接执行。
原HF19信息/选择门失败及slot2关闭保持，不通过修改旧metric重置预算。
最便宜的判别是冻结已有模型/专家，随机执行H10，按已知概率评估新源留出
效应。通过才投入本family slot2的适配/短片段重决策；否则停止此配方。
完整goal、Mission、Cm策略学习和最终稳定抓取验收不变。

## 固定执行与资源

使用已通过完整规划/真实PD工程的生成器与新source schema。
12科学phase：seed591–602、assignment15591–15602；工程590不进科学数据。
每phase96env、650tick、early/clear各最多1窗口，原生420秒；完整审计300秒。
生成器三模式各3冻结成员，Adam输入logits32步/lr .15/clip1，不训练网络。
目标/风险惩罚/simplex/旋转锚定与已通过工程完全相同，不扫超参数。
候选为base、cup、Cm-generated、direct-generated、shuffled-generated和三个
独立随机程序；10槽均匀，base/cup各重复2槽，合并p=.2/.2/.1×6。
全部候选及规划形成后随机分配；H10保持权重、每步更新真实专家反馈。
触发仍当前joint-proxy3步/rest>=5mm/warmup10/首episode/足够剩余步，
不按速度或未来筛选。触发、全候选、每批规划、真实pre/post/PD/force/mesh全存。

预算：本slot<=3600秒/8GiB，计入新工程保守900秒；源+完整审计pipeline
另限2500秒，至少留200秒作支持/IPW分析。旧HF19共享预训练数据
及计算28.17分钟/498.84MiB另报。实际单phase工程源306秒+完整规划审计193秒，
重复12phase单GPU会超预算。因此在CAMPAIGN最多4卡以内，用3张空闲3090
并行独立phase（计划GPU1/5/6）；每卡依次源采集与审计，绝不同时超过3卡。
分别报告wall time及源/审计累计GPU占用时间，不将并行加速当计算成本减少。
任一资源冲突、输入漂移、不完整/reset、专家/模型变化、执行/审计错误、
预算耗尽立即停止自己有句柄的任务。没有因效果不足追加seed或丢坏行的路径。

每phase终态exit0且完整原生/规划审计通过后接纳；全部12phase通过才分析。
SHA12651按motion/start分fit<50/cal<70/held>=70，继承旧模型fit-only训练边界；
只在新fit的base/cup中选最佳固定。新cal仅描述，held不调参数或门。
fit/cal/held各>=150窗口/32episodes/8初始组；每split base/cup>=24，三个
生成器各>=24，随机池合计>=60；不足UNCLEAR，不补采。early/clear另报。

## 机会与归因门

分配arm2就是冻结Cm生成策略，arm3/4分别direct/shuffled；并非从新结果拟合
目录。state-only的动作梯度为0、可行cup回退保留，执行等同cup；另报always-base
及fit选出的最佳固定base/cup。随机池是等概率选择三个随机程序，不选随机序号。
新源只保存实际后果，不提供逐状态所有候选真值，不报告伪oracle/regret。

主score是末3步最低rest-relative高度，在末3步joint-force代理且meshCLR>=2mm
时保留，减初始正高度，单位mm。risk是H10曾clear后失去2mm间隙；另报末3步
joint存在。净力存在代理不是已识别手物接触，间隙损失不是最终掉落率。
所有策略用实际arm匹配/p的HT-IPW；随机池用arm>=5匹配/.3。
held比较Cm对state-only、always-base、最佳固定、direct、shuffled、随机池。
每策略匹配>=24窗口/12episodes/8初始组；随机池匹配定义为其三臂实际执行。
1000次episode及motion/start组bootstrap90%，固定seed12753/12754。

base重复槽0/8及cup重复槽1/9分别按原槽p=.1做IPW差；报告组90区间。
门槛max(1mm,2*abs(base-null),2*abs(cup-null))；必须两对都有实际支持。
这是重复槽随机分配差，包含有限样本波动，不称同状态可重复仿真噪声。
每比较score差>=该门槛，两种bootstrap90下界>0，risk point差<=+.02，
末3步joint point差>=-.05。Cm对cup的实际原生PD起点差>1e-5覆盖至少10%；
另要求实际分配Cm中>=12个窗口的H10 PD与同一实际状态上的cup指令不同。
提议覆盖与实际执行覆盖分开报告，不把指令变化当收益。

全部源/支持/风险/收益/执行门通过为PROMISING，支持充分但未过为UNPROMISING，
支持不足为UNCLEAR。旧HF19一步信息失败不会改写；本新控制接口只检验已有
部分H10信息是否能生成更好的动作。正向仅允许slot2独立适配和重规划Probe，
不是正式utility、RL学习收益或稳定抓取结论。不直接启动PPO/最终成功率矩阵。
