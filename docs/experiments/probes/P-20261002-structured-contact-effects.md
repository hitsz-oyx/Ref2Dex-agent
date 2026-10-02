# HF21：联合事件后果与当前专家动作效果

experiment_id: P-20261002-structured-contact-effects
family: HF21
probe_index_in_family: 1
kind: Decision / reused-source model Probe
status: ACTIVE

区分：结构化任务后果与当前参考反馈信息能否提供比state-only/shuffled更可靠
的实际动作信息，还是只修复输出逻辑。前者进入新的独立真实执行生成/重观察，
后者停止这个原型，不扫epoch/seed/门限。最便宜方法复用已全审计HF19/HF20
源，不新增rollout。依据[决定](../../decisions/D-20261002-structured-contact-consequences.md)。

合同：SHA12651 motion/start bucket<50 fit、50–69 cal、>=70 reused-held；
两个科学源共5018窗口，工程seed570/590均排除。源manifest/records/各phase
完整独立audit、旧模型/源代码合同与新代码/本卡/工程结果全部hash固定。
旧held已用于研究路线决定，这里只是复用数据的机制Probe，不是独立Validation。
未来新的真实收益须新独立随机执行源，原HF19/HF20标签和门不改。

输入只有pre history、force/mg、全mesh CLR、原生1442观测（当前actor可见的
1/16步参考差，不是未来实际物理状态）、当前六专家动作bank、cup参考PD目标，
当前候选PD相对cup差、每节点专家权重差及已知H10反馈律。以后实际bank/观察、
未来force与label均不能进入输入。未来只作实际分配的标签，不构造反事实oracle。

目标：末3步hand/object存在代理及全mesh clear的8种联合事件；support为111；
非负条件高度和条件3cm抬升；期望支持高度=support概率×条件高度，支持抬升
概率=support概率×条件抬升概率。另预测H10几何损失、真实第一步相对节点
position/velocity、object dx/dv及H10末3步minCLR。proxy不是识别的手物接触对。
direct-score仅训练独立非负高度评分，其未训练事件/物理头不参与物理对照。

四模式cm/state_only/shuffled/direct_score，同架构/参数数量，三seed12811–12813，
各1000次Adam更新，batch256，lr3e-4/weight_decay1e-4/clip1。fit均匀采样，同seed
初始权重/采样序列相同；shuffled在fit的源HF19/HF20及early/clear内打乱候选
action/law，context不打乱。state-only保留相同当前专家bank与参考状态，清零
规范化候选action/law。所有归一化fit-only、std下限.001、clip8；不再调参。

预设首个信息门：全部fit/cal/reused-held>=150窗口/32episodes/8motion-start组；
reused-held实际support和非support各>=32；生成分布HF20 arms2–4>=60事实窗口。
Cm在全reused-held支持高度MAE和joint-support Brier均至少比state/shuffled
改善5%；lift/loss Brier及第一步object-dv/relative-position RMSE不差于这两对照
10%；生成分布事实高度MAE不差于两对照2%。包含关系误差<=1e-6、输出有限。
门均通过PROMISING，支持不足UNCLEAR，否则UNPROMISING；一致性通过本身只是工程。
直接评分同样报告高度MAE，不将其未训练事件头当作学过物理。

先用排除seed590做独立rawforce/PD标签及future-poison/梯度工程。首次梯度
check在eval模式触发cuDNN限制，失败原样保留，改train-mode backward后全部
输入与标签复算0；模型没有更新，不混入fit。两次成本5.242秒。
证据：[工程](P-20261002-structured-contact-engineering-r2.json)、
[失败](P-20261002-structured-contact-engineering-attempt-r1.json)。

首个模型Probe单空闲GPU1、wall<=1800秒/输出<=512MiB，包括工程准备与运行
审计。CPU仅处理文件/标签/算术，所有NN训练和重复预测GPU。共享HF19/HF20
源和旧预训练成本另报；无新仿真、无actor/PPO或最终成功率矩阵，不改Mission/C3。
slot2只有信息门通过后开展独立真实执行，失败不重训本配方。单会话直接完成。
