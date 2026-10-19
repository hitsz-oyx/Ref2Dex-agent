---
schema: ref2dex.probe.v2
probe_id: P-20261019-inspire-shadow-parity
experiment_id: P-20261019-inspire-shadow-parity
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 11
seed_pool: probe
seeds: [246]
decision_changed_if_positive: continue the full-data Shadow-recipe Inspire PPO on the ordered object-path task
decision_changed_if_negative: repair the first failing robot adapter without changing PPO algorithm or shrinking BC coverage
status: RUNNING
run_id: inspire-shadow-parity-20261019-r1
---

用户修订：Inspire必须大部分遵从Shadow成功路线，不使用DExplore原ref输入PPO，
不再以小网络/单序列试跑替代正式BC路线。PPO任务使用完整已知物体SE3路径，经scheduler接WAM；
不是直接给actor人体手ref、参考姿态误差或未来实测手。原小BC结果保留为独立工程尝试。

Decision：全数据、同网络及同PPO算法在Inspire的适配是否能够稳定训练和执行？
最低成本的必要步骤是先完成无ref物理输入、全量BC及小批实际官方UniDex PPO更新/错开reset检查，
通过后才放大环境数；不能由单序列BC失败推断方法不成立。

保持项：共享NormalizedCommandMLP隐藏层1024/1024/512/512、ELU、零输出初始化；
TRAIN-only状态/动作std下限.05、仅current裁到[-5,5]、tau固定.05m缩放/valid尺度1；
先current BC再零扩展WAM条件BC、AdamW3e-4/weight_decay1e-5、标准化L1、batch2048、
两阶段各50000更新、半启动/半全TRAIN采样、同object-balanced DEV选择。
PPO直接调用Shadow的prepare_official和UniDexGrasp2官方引擎/GAE/clip/adaptive KL分布；
rollout8、5epoch、4minibatch、gamma.96、lambda.95、LR3e-4、KL.016、clip.2、初始noise.8。
不调用DExplore PPO/rl_games更新、不加载官方DExplore actor为策略基础。

机器人适配：18原生DOF、12独立命令、30Hz；H4[-3,-2,-1,0]，tau8×11×3世界系。
无ref current312=q18+dq18+physical248+object pose/velocity13+task endpoint xyz3+previous action12，
拼tau264和valid1后577维。physical248来自已审核current721[9:257]；0:9及257:721舍弃。
在线纯仿真host直接从实测刚体构造physical248，并做原生函数/参考扰动负控制。
启动采样窗口按Shadow8/60秒换算为4个Inspire控制拍，H4有效需4帧。
PPO的网络/优化完全一致；完整物体路径任务的奖励/成功需要有序进度与终点门控，
这属于任务目标适配，不冒充与Shadow静态终点任务奖励逐项相同。

数据：完整现有660序列（50物体）的同执行示教，总165677拍原生命令；
589 TRAIN/150044拍、71 DEV/15633拍，保留既有划分，DEV不进梯度/统计。
current BC使用全部有限命令，不要求未来24帧；WAM条件使用全部同刻可构造H4，
逐帧检查workspace，超出模型工作区的帧明确剔除scheduled训练并报告，不造τ。
每个序列都入当前BC；几何/已知物体路径与native来源一致。

预算：GPU6/7，既有monica低占用授权共享；GPU0–5的活动Shadow训练不改。
全量准备/BC阶段≤8小时、缓存+权重≤2GiB、可用空间<20GiB停止新增写入；
native工程PPO每次≤30分钟，训练若通过smoke遵从已有10000更新上限/48h并采用滚动保留。
不存视频、不重采660、不生成大token缓存。模型graspenv/Torch2.4.1+cu121，
物理dexplore_v120_train/Torch2.0.1+cu118；AF_UNIX有界RPC、临时文件在项目tmp。

### 全量current完成与WAM精度回归

r1/c5dcaf4d已完整提取660序列165677命令，current BC的50000更新完成，约168秒，
GPU6利用率约78–95%、总显存约1GiB；object-balanced DEV最佳command MAE0.03851。
这不是行为成功率，DEV后期误差回升，保持与Shadow一致的best选择规则，不据此额外训练。

r1首次WAM缓存bf16的batch64/single差4.1168mm，超过1mm合同而停止；同输入重复batch
差3.3161mm。固定实测H4/goal的GPU7回归check_inspire_wam_precision.py可稳定抓到该模式；
模型eval已禁用随机serialization（未发现eval随机点序），仅切FP32后batch/single差7.2µm、
repeat batch22.1µm、repeat single5.8µm，全部通过原1mm门槛。
当前证据定位为这组输入的bf16数值敏感，未声称定位底层稀疏算子根因。
Inspire新cache和在线adapter统一使用FP32并写入合同，不修改WAM权重、网络或Shadow活动版本。

r2复用已完成current BC和不可变physical cache，在新输出重新生成FP32条件；
589 TRAIN/71 DEV划分、全量覆盖与其他超参不改。失败r1的空scheduled数组只在确认
两个done均全false、所有原writer退出后清理，保留首批数值审计、日志、current缓存及全部权重。

### 原生PPO接线工程协议

复用同一prepare_official，PPO.run/storage/GAE/优化器与Shadow相同，actor直接替换为
共享大BC均值、critic同官方大MLP。纯物理host只用DExplore资产/初始化/PD/物理步进，
不构造DExplore actor、不加载teacher、不跑rl_games PPO；post-step不读hand-ref观测、
不使用DExplore参考模仿reward/reset/curriculum。physical248直接从实测刚体算，原生函数
比对及参考扰动负控制在启动时执行。显式indexed reset先返回新episode/tick状态，再采动作；
旧terminal transition保留，无隐藏初始化控制拍、无跨回合action。
完整TRAIN路径通过按物体固定mesh、每次reset轮换该物体的motion，覆盖全部589条。

奖励保留Shadow goal_cond=False手掌/五指距离项和接近门控，针对完整物体路径将目标距离
替换为局部SE3路径误差，成功bonus必须scheduler按序complete，并加2×正向confirmed进度增量。
高度项仅将Shadow固定世界高度阈值平移到各回合桌面；不使用人体手参考误差或姿态模仿loss。
这是任务目标适配，不宣称与Shadow静态抓取reward完全相同。几何路径完成不证明抓稳/释放，
路径成功与保持代理分别记录。路径horizon为参考长度加30控制拍、scheduler另保留stall保护。

工程最先允许已完成全量current BC零扩展的2-env/4-update检查（与Shadow条件BC初始化相同），
明确标engineering_current_expansion；不可将其称WAM条件BC最终结果或作为正式warmstart。
actual learning必须使用完成的全量scheduled BC。检查官方20个optimizer steps/update、
均值/log-prob回放、原生命令/PD、两环境错开reset与WAM history清空，再扩大到256环境。

### 全量条件完成与真实PPO工程回归

r2两个FP32分片覆盖全部660条，分别626.9s/655.5s，首批single/batch误差均约12µm，
原1mm门槛通过。逐帧workspace排除354+811=1165个条件帧；这些帧仍已进入current BC，
没有换小数据或把不可预测τ造为零后纳入scheduled loss。已核对原几何目录恰有660条，
与当前corpus缺失/多出均0。完整两阶段各50000更新、网络/损失/采样和Shadow一致。

两环境纯物理+实际Shadow PPO的r1已完成4轮80更新，均值与logprob回放0、PD误差0，
但最终审计写入status重复字段触发退出失败；修复后r2完整重复通过，包括正常关闭双peer。
r2原生physical248与原函数max差2.98e-7、参考扰动不变；32个真实物理控制拍、0 workspace
failure，跨回合动作不匹配0。tick5只reset env0，episode[1,1]→[2,1]、ticks[0,5]；
tick11只reset env1，episode[2,1]→[2,2]、ticks[6,0]。真实终止transition保留，
新动作使用新episode观测/H4。此回归为current大BC零扩展，仅证明工程，不报任务成功率。
54项Task测试通过，包括原Shadow兼容、实际官方PPO超参/分布/BC均值和collector AST不变。

与Shadow相同，scheduled BC选择应包括初始current BC扩展点。r2训练遗漏了初始点的
best保存，故在同一个固定DEV面板重新对比初始点与训练过程中best，写入独立bc-selection，
不覆盖既有训练产物；后续trainer已补齐启动时初始DEV记录/保存。选择只用DEV命令MAE，
不读取闭环成功率、hold或测试集来选checkpoint。

### 256环境通过与吞吐检查

ppo256-r1/f1b13c60已用全部589 TRAIN/50物体、256环境实际完成4轮/80优化器更新，
均值/logprob回放误差0，物理动作episode错配0；混合reset检查通过。仅32个控制拍，
共3次结束（含两次强制工程reset、一次workspace失败），没有路径成功；不作为成功率评估。
原CPU输入构造下每轮collection约16.8–17.3秒、learning约0.18秒。

bc-selection按同一800条object-balanced DEV面板选择初始current扩展点，MAE0.03850862，
优于条件微调best的0.03868201。两阶段训练均已完整执行；入选点的tau列为零，
PPO可继续学习，但不存在BC阶段WAM增益证据，不改用闭环结果选点。

后续ppo256-r2复用Shadow已验证的GPU online_batch/workspace_valid输入构造，保持FP32和
相同H4/goal/模型/动作。CPU/GPU全部字段及原Shadow合同共16项测试通过。
同时限制OPENBLAS线程为2，记录scheduler/input/model计时和tau输入权重范数。
先做20轮无强制reset的工程/计时运行；通过后从该checkpoint续至累计10000轮。
两次使用同一seed和BC起点，计时仅作运行配置诊断，不作任务效果配对结论。
Shadow六卡原训练已正常完成10000终点，GPU0–5空闲；新Inspire只占GPU6/7，满足预留两卡。

ppo256-r2/9c37932b现已正常完成20轮、400优化器更新和160原生控制步；均值/logprob误差0，
PD误差0、动作episode错配0。steady collection中位15.958s、learning中位0.175s，
相对原CPU输入约16.8–17.3s提速有限；FP32 WAM推理为主要开销，不据此降低精度门槛。
GPU6观察值28–100%、总显存约4.36GiB，物理GPU7约8.28GiB、通常12–39%；
新路线不修改PPO超参追求利用率。tau列范数已从0增长，证明参数被更新，不证明有益。

已结束26回合，其中workspace终止23、路径成功0；多数长路径尚未结束，不能报告
成功率/收敛，也不能忽略早期越界失败。sampler共初始化442/589条（包含初始化与reset），
不等于442条已完整执行。后续长训记录实时初始化覆盖；BC已覆盖全部660条不受此影响。

下一运行inspire-shadow-parity-20261019-ppo-r1从ppo256-r2的iteration20模型/优化器续训，
新物理回合，追加9980轮至累计10000；不从含强制reset的工程版本续训。GPU6模型、
GPU7纯物理各256环境，48h/3GiB上限，滚动最近3份及每500轮/最终checkpoint。
按当前吞吐估计约45小时；定期查看workspace失败和路径完成，尚无Inspire成功claim。

验证：完整Task测试54项通过，另16项含CUDA输入/τ/Shadow回归通过。根verify仍被
6张既存trajectory-policy实验卡的39项v2元数据错误阻挡，与本次Inspire修改无关；
未改写这些历史卡片或用户的CAMPAIGN变更。

长训ppo-r1已实际启动，冻结88b4c445；第21轮正常完成（恢复20轮+新增1轮），
均值/logprob回放误差0，GPU6/7分别约4.4/8.3GiB显存，GPU0–5空闲。
首轮因H4冷启动为12.69s collection，ETA以此前steady16.1s/轮估计，不用首轮短时外推。
运行状态见outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-ppo-r1/process.json，
逐轮记录在policy/curve.jsonl；累计iteration、workspace失败与路径成功继续分别记录。

### BC自主抓取质量检查（用户追加问题）

Decision：全量BC warmstart在无探索噪声时是否已有抬升/保持能力？若已有而PPO频繁失败，
优先审查探索/动作尺度；若BC也不行，先审查示教与实测输入/动作对应及BC质量，不能只等长训。
既有DEV MAE0.03851及有噪声PPO启动回合不能回答这个问题；早期单序列小网络也不代表本次BC。

最低成本Probe：inspire-shadow-parity-20261019-bc-eval-r1，在GPU4模型/GPU5原生仿真、
64环境覆盖50 TRAIN物体，seed246，记录每个环境第一个完整回合，最多1200控制拍/1800秒。
不停止GPU6/7主PPO，仍有GPU0–3空闲；产物≤3GiB，不存视频，不做optimizer更新。
相同selected BC、scheduler、workspace终止与原生动作/PD，确定性actor均值，无随机动作。
记录初始完整测量供配对复用；分别统计物体最大/终态抬升、曾/最终连续30拍抬升≥10cm、
保持后掉落、路径完成与workspace失败。仅是物理抬升保持代理，不是接触力抓稳证书或未见物体泛化。

入选BC已核验首层整个tau+valid块严格为零，故仅在该条件成立的BC评估可省略WAM predict，
仍构造同一H4/scheduler输入并检查workspace；已用非零输出网络回归逐位动作相同和越界规则相同。
若权重非零或评估PPO，则正常预测WAM。该优化在评估manifest中显式记录，不能称WAM闭环增益。
完整Task测试55项通过；当前长训冻结版本不热改。

bc-eval-r1/78e1ba98已正常完成64/64个首回合、50个TRAIN物体、473全局控制拍，
模型评估约171秒、0 optimizer更新；原生PD误差0、episode动作错配0。
高度≥初始10cm且连续30拍为6/64，但其中5个后来掉落；剩余cup_drink回合末高度
达到3.65m且workspace失败，不能称稳定抓取。最终高度保持且无workspace失败为0/64，
目标附近连续保持1秒0/64，完整有序路径完成0/64，workspace失败15/64。
63/64的参考路径曾要求≥10cm高度变化，故低结果不能主要解释为面板全是低高度任务。
高度阈值可以把抛飞也算入，6/64只报告为高度代理，不能升级为抓取成功率。

初始测量与各自示教首帧核对：64条不同序列，q/dq、object xyz/velocity误差0；
physical248逐序列RMSE最大5.96e-8，object quat最大4.22e-8。
这排除了明显初态输入漂移的解释，尚未定位闭环失败，也未验证整个采集/当前动力学逐拍一致。
同名示教60/64曾升高10cm只是数据描述，不能当本次配对teacher成功率。
结论：全量BC离线动作拟合已完成，但尚未形成可靠自主抓持/完整路径执行能力；
PPO长训仍在运行，后续优先审查BC闭环误差与示教/执行动态对应，不能仅凭DEV MAE认可warmstart质量。
汇总及逐回合证据在outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-bc-eval-r1/
的summary.json、policy/evaluation.json和initial-state-diagnostics.json。

### 用户要求停止旧运行、转六卡混合训练

16:41用户停止旧两卡训练，STOP已触发且所属policy/native/supervisor退出，保留最后完整
checkpoint iteration161；旧launcher将主动STOP记为FAILED，stop-request.json记录了用户请求，
不将它误报成算法数值失败。新run使用该模型/optimizer继续，新物理回合。

六卡版在GPU0–5每卡同置graspenv WAM/PPO和dexplore_v120_train纯物理两个进程；
每卡256环境，总1536，六rank同一个策略，沿用Shadow官方梯度/KL/优势统计归并。
50个TRAIN物体按master seed246分为9/9/8/8/8/8个，每条589 TRAIN motion恰好属于一个rank；
DEV不入PPO，BC仍是全660两阶段。局部几何身份合并50物体后校验resume绑定；
广播参数与normalization buffers，单份rank0 rolling checkpoint，记录参数同步误差。
native派生随机流seed246+rank，policy初始化246、采样流246+rank；不是追加独立实验seed。

Blocker工程检查先在GPU0/1、每卡64环境运行4轮，显式错开reset并验证实际官方PPO
80 optimizer steps/rank、回放/PD误差、跨rank参数完全同步。通过后从原iteration161
启动六卡无强制reset的正式续训，累计10000上限，不从带工程reset的smoke策略续训。
新launcher主STOP在更新边界归并，rank0先保存完整checkpoint，barrier后各peer正常退出；
每run48h/3GiB、20GiB磁盘底线，GPU6/7预留。诊断GPU测试安排在六卡启动之前，
后续先做CPU日志/权重读取，不在六卡训练时额外占满两张预留卡。
代码网络/奖励/动作/WAM精度不变；合并训练batch改变为原来的6倍，manifest显式记录。

两rank smoke(cc2b84c0)已完整通过4轮：各80 optimizer steps、所有参数同步误差0、
均值/logprob误差0、原生PD/episode动作错配0，含两环境错开reset。它只用于工程验证，
正式六卡从原161继续，不使用此带强制reset的165权重。

退化审计发现继承的官方adaptive KL与实际采样scale_tril不一致，
详见P-20261019-ppo-checkpoint-evaluation的诊断追加。新Inspire修正KL度量，保留原采样
分布、noise.8、desired_KL.016、clip/GAE/20 optimizer steps及网络/奖励，不重写PPO。
optimization_contract显式绑定actual-KL版本；从161旧optimizer迁移的源/目标写入resume-optimization。
这使新运行与旧运行还存在KL校准变化，不将结果差异单归因于六卡batch扩大。
60项Task测试通过；真实checkpoint方差探针旧公式失配红灯(exit2)、修正后与Torch真KL一致绿灯。
模型更新后另记录整个rollout的实际KL/旧proxy KL，不将其误称每minibatch硬约束。
该修正尚未用匹配Shadow续训证明成功率恢复，不能说已完全解决退化。

六卡inspire-mixed6-20261019-r1(fd79db1a)已从161实际续训，首轮162以及随后多轮正常；
六rank首轮参数同步误差0，均值/logprob误差0，native PD/episode错配0。
源成员审计coverage-audit.json核对589条motion恰好覆盖一次，50物体无重叠/缺失；
每rank物体9/9/8/8/8/8、motion134/87/81/91/95/101。不是已完成全部轨迹的成功证明。
新权重单份保存于rank-0/policy，首次162已发布；optimization_contract与从legacy161的
迁移均有记录。主训练累计10000上限，GPU0–5总1536环境，GPU6/7未被本项目占用。
观察到约19.7–20秒/轮、每卡约8.7–9.1GiB；10000终点预计约54小时，超过当前48h预算，
不自动扩预算，保留周期checkpoint可继续。actual rollout KL首轮0.0140，随后例0.0137，
旧proxy同轮仅0.00852；这里只确认校准度量运行，不据此声称任务成功或退化已恢复。
旧两卡的主动停止另写stop-result.json，原始FAILED/InterruptedError日志保留，不覆盖历史证据。


### 六卡在线吞吐修订（2026-10-19，用户授权）

工程问题：提高相同1536环境、官方PPO配方及WAM权重下的实际吞吐，不以额外计算抬高利用率。
当前旧mixed6-r1的最近20轮无干扰计时约18.56s/update，采集16.5–17.9s；
30秒采样GPU0–5平均利用率分别67.1/80.5/51.1/61.0/68.0/55.2%。
该运行已在完整更新边界停止，保存iteration300，12个peer均正常退出。

最低成本工程对照：只用既有50 TRAIN物体、每物体8个实测H4，400窗口，无策略更新。
旧训练停止后的GPU0独占模型计时（graspenv/Torch2.4.1+cu121，3次重复）：
full_fp32/batch128中位2.25195s，hand_fp32/batch256为2.14039s；
仅共享Transformer启用TF32、hand-only/batch256为1.20616s，推理吞吐约1.87倍。
相对完整FP32最大轨迹偏差0.42784mm、repeat0.35611mm、batch/single0.23855mm，
均通过原1mm门槛；同实测current及物体旋转下iteration300均值动作最大变化0.00059044。
这不是动作逐位相同或学习效果等价的证明，也不从离线推理倍数推断整轮倍数。
全模型TF32在前一争抢计时中误差接近1mm，因此没有选用；保留FP32几何编码和head。

另一个实际CPU热点为reset时逐帧参考位姿校验：212帧参考构造100次，旧实现
2.456/2.489/2.531s，批量等价检查0.0901/0.0902/0.0904s，约27.6倍；
poses/path_points/arc/milestones逐位一致，单实例状态及interaction history不共享。

新入口默认仍为full_fp32/128，优化模式须显式指定并作为execution identity记录；
旧checkpoint仅在显式performance migration时允许该项差异，WAM/几何/奖励/动作等不匹配仍拒绝。
TF32仅包围共享Transformer调用，异常和正常返回均恢复；PPO math、采样分布及actual-KL合同不变。
启动和每300次预测对真实状态重跑完整FP32，超过1mm即拒绝继续；没有静默降低门槛。
70项Task测试通过，新增批量参考验证、precision作用域异常恢复及真实adapter门槛反例。
固定输入与计时产物：outputs/wam-rl-policy-v2/inspire-throughput-20261019-r1/，当前约77MiB；
不生成视频，不修改外部数据，保留两张卡。拟从iteration300用新冻结版本续训，在线比较整轮耗时。
