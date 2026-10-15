---
schema: ref2dex.probe.v2
probe_id: P-20261015-shadow-data-scale
experiment_id: P-20261015-shadow-data-scale
date: 2026-10-15
task: shadow-finetune
branch: main
git_commit: 57737b7b
claim_id: C1
hypothesis_family: HF-shadow-data-scale
probe_index_in_family: 1
seed_pool: probe
seeds: [214, 215, 216]
decision_changed_if_positive: prioritize larger native Shadow dataset and broader paired success panel
decision_changed_if_negative: inspect fitting and closed-loop distribution shift before longer scaling
status: RUNNING
run_id: shadow-scale-controls-20261015-r2
---

# ShadowHand：固定计算预算的小数据/全量数据对照准备

Class: Decision。用户要先在ShadowHand自身环境微调并看成功率，以检验现有DExplore数据覆盖
不足的猜想。该Task服务独立策略资格，Shadow收益不能自动成为Inspire失败的数据量归因。
原pilot只用7TRAIN物体/41轨迹，学生TRAIN1/8、DEV0/8，不能检验数据量假设。
全量候选已通过12456条数据合同审计，故现在准备最小fixed-compute对照，隔离TRAIN支持集合。

## 设计与判别

- small：pilot cache的7TRAIN物体、41轨迹、6232窗；full：1567物体、10118轨迹、1536897窗。
  small保存状态、标签与完整tick集合必须是full的物理同一子集，按raw episode/env匹配，
  不能因metadata行顺序不同误配。两者均使用相同终点5cm/安全窗准入，不使用DEV误差选训练。
- 同80k父权重、fresh Shadow query actor、seed214、AdamW/各模块LR/forecast权重不变；
  两臂各1000更新、batch64。采样均为序列均匀，再在序列内均匀，故full不保证一个epoch或充分拟合。
- 两臂state/action mean/std均仅由small TRAIN拟合并冻结，避免normalizer改变成为额外干预，
  small不读取full TRAIN或DEV标签拟合统计。预训练WAM numerical normalizer保持父权重值。
- 两臂使用同一个full DEV reader、seed215预先抽取的64窗，保存索引和raw/env/tick身份；
  不重新抽各自面板。TRAIN/DEV按物体分开；不把小集原DEV64与这里的新DEV面板直接比较。
- 同16env固定2TRAIN+2DEV对象面板、seed216，teacher/student完整初始状态配对。
  闭环主读出用相同1000步终点，分别报告官方success、末帧5cm和同步抬升保持代理，
  TRAIN/DEV分开；原pilot seed213和step500结果不能作为该对照的一臂。
- 保存每250步权重、best action和final；离线全DEV面板动作L1/root/finger/首命令MAE及forecast。

最便宜先完成共享controls和真实数据合同核验；GPU执行按实际空闲和预算另行启动。
局部PROMISING信号：full终点相对small的共同DEV首命令MAE降低>=20%，且DEV闭环多成功
至少2/8个起点；这只是后续扩大面板的触发条件，不能当正式成功率结论。否则UNCLEAR，
先看TRAIN/DEV拟合与分布移位，不能以full尚未充分学习为数据量路线负证据。

## 资源与当前状态

准备阶段先完成代码、controls和CPU小集统计；随后按下面Decision Note进入临时单卡短Probe。
常规资源须本地留2卡，本次使用用户明确允许的一张临时空卡：原Inspire三臂GPU0–5继续，
Shadow GPU6两臂各1000步、总1小时，GPU7空闲。每训练入口2400秒/12GB checkpoint/20GiB空闲
保护继续生效；native每worker600秒。不得为对照暂停既有任务或扩成长训。

输入manifest、父checkpoint、native panel保存size/mtime身份；启动拒绝settings/panel/input漂移。
`tools/run/prepare_scale_comparison.py`只生成controls和命令，`train.py --controls`才消费同一份配置。
6项Task合同测试涵盖物理同一子集与标签变动拒绝、同DEV/同归一化，以及设置/输入漂移拒绝。
当前已进入GPU训练，尚无本实验最终对照或成功率结果。

## 真实输入准备回执

固定代码c3f8f7bc，已生成
`outputs/shadow-finetune/shadow-scale-controls-20261015-r1/{controls,commands}.json`。
逐条按raw episode/env复核51条small缓存与full缓存，全部保存字段逐值相同，small tick为full子集；
包含small DEV数据是为了验证沿用记录身份，并未用其拟合统计。两臂共同DEV354761窗中的
同64个query身份及normalizer逐数组完全一致，启动接口已在真实cache上重放并通过。

CPU加载/拼batch64、每臂6批：small均值0.076秒（热缓存约0.049秒），full均值0.311秒，
点数512；所有batch浮点tensor有限，state/action shape符合合同。full随机数据I/O比small慢，
尚未包含GPU模型计算，不将这次测量当训练吞吐。初估1000步仍应在原2400秒训练guard内，
真正启动后须重新观察GPU利用率与ETA。未因低吞吐测量自行修改matched batch或训练步数。
证据`readiness-audit.json`位于上述目录，CPU复核脚本保存在`tmp/shadow_scale_readiness_20261015.py`。
以上准备回执记录时未启动GPU；准备通过不是数据量效应的结果。

## Decision Note / 从准备进入有界GPU Probe

当前决定：按用户“先写代码，然后可以运行暂时占用一张空卡跑小规模测试”的持续授权，
继续同范围的小规模比较。上一阶段主动只准备输入，并不把授权解释为只能运行一次命令；
此次仍限制同一张GPU6、两臂各1000步/batch64，整体≤1小时/14GB，不启动长期微调。
证据：全量合同审计、nested逐值核验、共享DEV/normalizer均通过；CPU加载开销可在短Probe预算内。
原三组Inspire GPU0–5继续、GPU7保持空闲，临时第七张卡只供此有界比较；外来进程使用GPU7则
释放本Task的GPU6，绝不终止他人。两臂依次运行满足只用一张额外卡的用户边界。

执行：先固定提交与detached工作树，记录真实code root；每臂训练后立即原生16env终点评估。
保存初始actor全部张量供逐值比较，原forecaster严格加载同父权重；比较器核对控制配置、
实际DEV IDs、代码/参数/父统计、初始actor和跨臂仿真初态，最后按预定指标汇总。
监督器只管理自己的子进程组，源代码漂移、超时、磁盘<20GiB、超14GB、外来GPU进程或
任一stage失败即停止，不自行扩大预算、换输入或把未完成实验当负结果。未知结论保持UNCLEAR。
本次是用户已明确允许的临时单卡小Probe，其他资源和研究claim均沿用现有约束。

启动r1在任何训练更新前因固定工作树PTv3依赖缺失退出：detached worktree中的第三方gitlink
目录已经存在为空，首次整目录软链接保护拒绝后，shell仍执行了启动命令。已核验supervisor
2081606和child2081619均终止，保留原process/log；这不是拟合或方法负证据。
修复仅在新工作树第三方目录内挂载原固定ptv3包，启动入口先检查依赖文件，另建r2输出，
不改实验配置、不覆盖r1、不重新启动任何Inspire或远端训练。

## r2实际启动回执

依赖补齐后r2已启动，固定57737b7b、detached工作树`tmp/shadow-scale-code-20261015-r2`，
supervisor2083589、首个train-small child2083598。r1/r2 scientific controls逐字段相同，
仅代码身份/执行目录改变；原始失败记录保留。任务层7项合同测试和工作流验证通过。

已实际更新到small step180/1000，loss0.77581（hand0.09330/scene0.44575/action0.23675），
峰值allocated2991MiB，稳态约0.55秒/步，剩余small约7.6分钟；整轮暂估25–35分钟，
full吞吐须在真正启动后再测。50秒采样GPU6平均77.8%利用率、最大3637MiB，GPU7始终0%/2MiB。
真实初始actor51个state tensors已保存，四个normalizer buffer与controls逐值一致，DEV IDs完全
一致、父权重step80000。`startup-control-audit.json`与`startup-observations.json`在r2目录。
完整跨臂初始actor/仿真初态核验仍待full臂完成；不把small启动核验当全部matching通过。

已对small step250的实际完整checkpoint做CPU反序列化核验：1192294103字节、模型和
optimizer张量全部有限，586个参数具有optimizer state，三个参数组学习率为5e-6/1e-5/5e-4。
actor四个normalizer与controls逐值一致，forecaster四个统计buffer与父TRAIN统计逐值一致。
`checkpoint-audit-step250.json`明确只覆盖所列step250，不把尚未生成/未检查阶段视为已验收。
这是序列化/合同审计，不调用CPU模型推理；脚本`tmp/shadow_scale_checkpoint_audit_20261015.py`。

补充CPU Decision诊断：对TRAIN候选统计近目标/保持阶段的时间覆盖，区分“轨迹少”与“完整
forecast24支持把关键晚期动作丢掉”。对比现行H4/K24+动作8的query与只要求安全动作8的query；
近目标仅定义为当前物体距任务目标<=5cm，不等同接触稳定。若多数序列完全没有保留近目标
监督，下一步优先检验尾段action-only/partial-forecast支持；否则保持数据量对照解释边界。
最便宜直接读已审计cache的pose/goal，CPU≤10min/200MB，不调用模型、不改变当前准入或DEV面板。

## small终点完成 / full正在更新

small1000步完成，564.8秒，250/500/750/1000完整权重保留。固定共同DEV64的1000步
action L1为0.27629，首命令MAE0.11705（root0.21301/finger0.08506），hand0.19778/scene1.15844。
seed216原生16env结果如下，官方success是sticky位置标志，不能与终点或保持混报：

| 角色 | 官方success TRAIN / DEV | 末帧5cm TRAIN / DEV | 同步抬升保持60帧 TRAIN / DEV |
| --- | --- | --- | --- |
| teacher | 7/8 / 1/8 | 6/8 / 1/8 | 6/8 / 1/8 |
| small学生 | 6/8 / 1/8 | 2/8 / 0/8 | 2/8 / 1/8 |

全部初始q/dq/root/hand/object/observation300/goal/reset一致、无invalid；独立NumPy从raw重算
sticky/终点/同步连续保持计数与报告完全一致，teacher/student的observation[t+1]上一拍命令
与action[t]均max0，命令有限且[-1,1]。证据r2下`native-small-independent-audit.json`。
该面板teacher DEV只有1/8，预设DEV增加2/8门槛的功效有限；仍保留原门槛，不据此改判定或
用small成功与旧seed213/旧selected500结果作同因素改善归因。

full已真正启动，step100/1000，稳态约0.80秒/步，ETA约12分钟、峰值allocated3017MiB，
GPU6读出93%/3623MiB，GPU7空闲。实际51个初始actor张量逐值相同，代码/父checkpoint/
父统计/配置/预算和共同DEV IDs均一致，回执`cross-arm-startup-audit.json`。尚待full终点及
跨臂native初态核验，当前不能判断数据量收益。

## TRAIN晚期监督覆盖诊断

CPU21.9秒遍历10118条TRAIN候选，未改变训练。现行完整forecast支持保留1536897个query，
其中当前物体距目标<=5cm的351321个；若只要求安全8步action和同workspace，则有1941617个
query/706676个近目标query。完整forecast额外舍弃404720个query，其中355355个近目标，
约占安全action-only近目标候选的一半。末尾action-only支持是审计假设集合，尚未准入或训练。

现行7868/10118条含近目标query，action-only为9791/10118；两者差1923条。首次进入5cm的
状态tick分位数25%/50%/75%为93/115/147，完整200状态轨迹通常只训练到query151。
观察到连续近目标60帧的序列7091条，现行query中仍连续保留60帧的2436条，action-only6496条。
这是位置阶段覆盖，不证明接触质量或说明训练必须包含连续60条query。

证据`outputs/shadow-finetune/full-phase-coverage-20261015-r1/{summary.json,sequences.jsonl}`。
这排除了“近目标监督完全没有”的说法，同时暴露明显晚期覆盖缺口；若full固定预算仍不能改善
保持，下一轮优先以相同full数据做partial-forecast/尾段BC最小对照，而非立即归因总轨迹数不足。

## 终点结果后的标签有效性检查

paired pipeline已正常完成，所有matching检查通过；full首命令MAE仅改善6.21%，DEV官方success
仍1/8，整体success2/16、同步保持0/16（small7/16、3/16）。阶段结论UNCLEAR，不能关闭数据量路线。
最便宜先排除全量teacher标签与预期确定性策略不一致：按官方固定300D actor严格重载权重，
穷举当前TRAIN准入query的首命令，与采集canonical action[t]比较。此前仅小集153点重算，
不能替代这项全量检查；不会以observation[t+1]上一拍命令一致冒充teacher函数复算。

这是Blocker诊断，若不一致先修数据，若一致才继续拟合/尾段覆盖问题。临时GPU6、≤10min、
模型分配≤2GiB、报告≤50MB，不训练、不改任何cache/指标/原checkpoint，GPU7保持空闲。
该检查沿用用户的一张空卡小测试授权；上轮pipeline实际25.8分钟已退出并释放GPU6，不延长它。
动作clip与原采集一致，FP32关闭TF32，容忍1e-4以容纳批大小/后端GEMM浮点差异。

## Limitations / future evidence

固定计算预算衡量更大支持集合在同预算下的收益，不能证明完整数据的收敛性能。只一个seed、
64个可能相关的DEV窗口、2个DEV物体闭环，统计功效弱。更广闭环面板、多seed、充分拟合、
teacher300-only MLP对照、接触稳定性，以及与Inspire的差异归因都留待有判别信号后执行。
