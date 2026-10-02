# HF19：接触几何与动作组合的信息/机会 Probe

experiment_id: P-20261002-contact-geometry-action-information
family: HF19
probe_index_in_family: 1
kind: Decision / Probe
status: ACTIVE

问题：把每指可执行动作与物体坐标下的相对几何绑定，并用真实随机组合
训练短期物理作用模型，是否能提供超过强rotation_cup的条件动作信息？
正向才进入slot2的实际闭环；负向停止此配方。最便宜的判别是一次有界随机
源采集、固定拟合和留出IPW选择器检查，不先做PPO/最终成功率。
区别于HF18的全局固定目录：此处有新连续组合真实监督、每指动作—局部
几何绑定和密集一步物理监督；旧门/旧失败结果不改，不把模型名称变化当进展。

## 源合同与预算

冻结使用已通过原生工程的collect_contact_geometry_source.py与源schema v1。
科学seed571–582，共12phase；assignment_seed15571–15582。每phase96env，
early/clear各最多4窗口、650tick/240秒。GPU1空闲时使用GPU1，不干扰他人。
八程序：base、rotation_cup、六个独立随机组合；10槽中的重复base/cup只测
噪声，合并propensity .2/.2/.1×6。随机组合序号不代表六种固定动作。
完整候选在分配前生成，只保存实际执行后果；未知候选没有反事实真值。
SHA12651按motion/start组fit<50/cal<70/held>=70。工程seed570不进科学拟合。

slot1总执行预算3600秒/8GiB，包括此前工程194.824秒、失败、审计、拟合、
分析；源采集最多2500秒，预留审计/拟合。运行器只启动源采集，不自动训练。
任何输入漂移、跨reset、不完整窗口、专家变化或实际PD不符停止。
全部12phase终态/hash、真实命令/PD/概率/force/mesh及每步pre/post几何审计
通过后才接纳。只看split/支持统计，不因held效果追加seed、改变采样或丢坏行。

源支持门：fit/cal/held各至少150窗口、32episodes、8个初始组；每层的base
和cup各至少24实际匹配，随机组合合计至少80。不足为UNCLEAR，不调门补采。
每层指数据split，不是early/clear子组。early/clear分层另外报告。

## 固定信息合同

输入只用pre/history、原始力存在代理、物体相对掌/五指端link位置/速度，
各节点当前q/qdot与实际PD目标差、已知组合系数及旋转控制律。未来只作标签。
一步监督：相对位置/速度变化、物体实际dv/dx、mesh间隙变化和两个存在代理。
不用端点净力积分；link原点不是精确接触点，净力代理不是已识别接触对。
H10监督：末三步最低高度/间隙、手/物存在保持、几何损失、支持形成抬升、
支持高度。H10输入是已知反馈律/初始目标，不包含未知未来动作。

共享节点MLP64与历史GRU64，池化后256/128物理头；同容量cm/state_only/
action_shuffled，三个seed12601–12603，各1000更新、batch256、Adam lr3e-4、
weight_decay1e-4、clip1。从头训练，fit-only归一化，GPU。另匹配一个直接
任务评分控制，只监督H10支持高度，防止用弱state-only物理模型替代强策略对照。
具体张量索引、标签单位和工程审计在拟合前固定到实现；不基于held调结构/更新数。

信息门：held一步物体dv与相对位置delta的RMSE均比state_only和shuffled
至少改善5%；间隙RMSE与存在Brier不恶化超过2%；H10支持高度MAE比两者
至少改善5%，支持抬升Brier不恶化超过5%。常速度/状态保持基线另报。
失败关闭此配方；部分信息不替代原门。不得追加epoch、换seed或修改门。

## 真实候选机会与选择器门

fit只在base/cup中选最佳固定参考；随机组合池均值单独报告，不能把随机序号
选择当最佳固定动作。重复base/cup槽给出参考噪声，保留bootstrap区间。
固定均值没有增益不等于没有条件机会；一次物理拟合后的选择器才判断条件机会。
所有候选共享起点与可执行域，按预测H10支持高度选；ensemble的不确定差异
门为预测优势至少1mm+1.645*成员差值std，预测几何损失不得高于参考+.05。
不确定时执行fit选出的固定参考。直接评分/shuffled用同样候选与回退条件；
state-only没有动作辨别力时固定返回参考，另报always-base。

按实际分配与propensity，在held计算所有策略的IPW支持高度/风险/覆盖；
比较cm与state-only、shuffled、always-base、最佳固定、直接任务评分。
参考重复噪声阈值max(1mm,2*abs(held参考重复槽差))。对每个比较要求：
score差至少阈值，motion/start组及episode bootstrap90下界均>0；几何损失
point差<=+.02，末三步joint存在point差>=-.05。每策略实际匹配>=24、
12episodes、8初始组，且cm至少10%窗口改变固定参考。
通过为PROMISING，支持足够但门失败为UNPROMISING，否则UNCLEAR。
只有信息与条件机会都通过才用slot2跑实际重新观察/执行；离线IPW不是已证明
闭环收益，不报告逐状态真实regret。排序只报告实际观测与策略价值，不能伪造
未执行候选的排序真值。最终稳定抓取及Cm策略训练仍是后续验收。
