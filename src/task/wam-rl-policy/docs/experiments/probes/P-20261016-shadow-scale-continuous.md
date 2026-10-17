---
schema: ref2dex.probe.v2
probe_id: P-20261016-shadow-scale-continuous
experiment_id: P-20261016-shadow-scale-continuous
date: 2026-10-16
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-shadow-scale-continuous
probe_index_in_family: 1
seed_pool: probe
seeds: [241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251]
decision_changed_if_positive: retain a competent self-trained Shadow BC bootstrap and scale online improvement before frozen-WAM token validation
decision_changed_if_negative: diagnose coverage objective reward and exploration instead of treating a short two-object run as method failure
status: UNCLEAR
run_id: shadow-scale-continuous-20261016-r1
---

# Shadow大规模持续推进

Class: Decision。服务自主抓取基座与预训练物理表示辅助RL的最终目标。用户明确授权大规模
持续推进、充分利用本地/远端GPU。前20轮仅2TRAIN对象、127k步、DEV24起点，不能判断充分
训练效果；旧query BC还混合8拍动作监督，当前单拍拟合/容量可能限制RL bootstrap。

## Decision Note（扩大前）

选择并行两线：现有冻结WAM/BC的PPO扩大对象/并行环境/训练；同一全量Shadow缓存直接
训练current300→当前canonical24的自训练BC MLP，争取得到更强、更快的reference。
不使用官方actor权重初始化学生；官方既有动作日志允许作为监督。若MLP提高真实闭环
DEV保持，再接同一PPO；它不替代最终WAM效用对照，不变更MISSION claim。

第一阶段在线最多128个TRAIN对象、4GPU同步收集；环境数由64/128等真实吞吐smoke确定，
目标200次更新，50/100/200读出，单次训练<=12h。已有2对象20轮权重/optimizer/RNG
可作为单卡200轮持续性对照，最多4h；不声称恢复PhysX内部状态。
BC MLP使用全量strict TRAIN1536897/DEV354761当前标签，fresh seed241，后续242/243
只在正向信号后配对复核；batch2048、AdamW3e-4/WD1e-5、标准化L1、最多50000步/2h，
保留5k/10k/20k/50k等端点。离线DEV分对象固定面板，不用TRAIN归一化之外的信息。

扩大评估：先冻结所有388DEV对象的native资产与分批面板，4起点/对象，总1552rollouts
每策略；连续读出用预先选择的64DEV子面板（256rollouts），不按结果挑选。全部物体
明确来自本项目DEV，仍属官方TRAIN集合，不冒称官方test-unseen泛化。
对照相同对象/seed/env配置，实际九类初态须逐值相同；baseline也有数值波动，native保持
代理不能当接触认证。主指标连续60帧goal<=5cm且lift>=10cm，另报保持到终点、reach/lift
和退步病例。DEV>=10个百分点改善且TRAIN不明显退步记PROMISING，再固定协议进入Validation。
阶段曲线/行为无进步则诊断或调整，不因单个成功视频、训练loss或GPU数量宣布成功。

## 实现与资源边界

代码仅当前main下wam-rl-policy，原BC/数据/检查点只读；新产物保存本run专用目录。
补齐训练恢复、分阶段checkpoint、早停/监控；多卡必须全局adv归一化、同步actor/critic
梯度、全局KL与共同accept/rollback，不能把独立训练进程称作DDP。
提速先验证等价：省略未使用forecast heads、批量构造H4输入，与原控制器同输入逐tensor
比较，允许既有GPU推理数值波动但不悄悄改变动作/goal/时钟含义。

本地GPU1–4用于主训/吞吐smoke，GPU5用于BC和评估；GPU0外部作业不动，6/7保持空闲。
远端4卡可用，先同步新权重/代码和所需原生资产、通过实际仿真smoke，再运行不同seed
或方案；跨Torch版本不混作单变量比较。初期总产物<=64GiB、任何时刻磁盘保留20GiB，
每个长作业有明确max_seconds，资源/非有限/合同失败立即停本组并保留记录。
GPU利用率、显存、吞吐和ETA实际记录；冻结WAM/BC可逐阶段抽审state_dict且收尾全量审计。
新runtime目录的资产只读软链接至已有材料，不修改其他Task数据或第三方项目。

## Limitations / future evidence

本卡是探索性规模推进，不是正式Validation。多个训练seed、reward对照与learned RL-token
只有在会改变下一步决策时启动，不能把资源充足作为无界实验理由。正式效用claim仍需
匹配none/shuffled/aligned及冻结评估协议；暂不更新整个WAM。

## 已执行 / 运行中（2026-10-16）

实现commit：53dccda5（fresh单拍BC），92b4231c（同步PPO/面板/提速），53fcd641（整套DEV），
b73ecf74（portable身份），9adc6693（阶段自动评估），a3d804f7（失败退出码/扩展cache）。
六项Task测试通过，包括两进程PPO与拼接全局batch的优化更新等价，以及随机H4/ragged CAD
各字段与原输入路径等价。原生两卡2轮后，actor/critic逐tensor跨rank相同，597字段冻结审计通过。

已冻结128TRAIN、388DEV、64core_DEV面板；每32对象一片，实际完整资产闭包3.95–38.95MiB，
含手模型/纹理/所有linked目标，均在原128MiB界内。仅runtime独立副本的guard支持已审计
512MiB，不改第三方或原collector；本轮未实际需要更大资产预算。

| run_id（均在outputs/wam-rl-policy） | 状态/要点 |
|---|---|
| shadow-current-cache-20261016-r1 | COMPLETED，原strict源窗TRAIN1536897、DEV354761 |
| shadow-current-bc-seed241-20261016-r1 | COMPLETED，50k×2048，163.35s，GPU5峰值2.39GiB allocated；GPU2.73GiB used |
| shadow-scale-wam-smoke-20261016-r1 | 两卡64对象/128env，2轮，211.6s，退出码0/0；native/同步/freeze通过 |
| shadow-scale-wam-128obj-4gpu-20261016-r1 | RUNNING，GPU1–4，128TRAIN/256env；200轮=10188800 envsteps，<=12h，初步95s/update约5.3h |
| shadow-scale-mlp20k-core00-20261016-r1 | COMPLETED，32DEV/128起点，held37、final55、official61 |
| shadow-scale-wam9k-core00-20261016-r1 | COMPLETED，同对象/seed/env，held37、final64、official70；九类初态全部逐值同一 |
| shadow-scale-mlp50k-alldev-20261016-r1 | COMPLETED，388DEV/1552起点，held396(25.52%)、final590(38.02%)、official679(43.75%) |
| shadow-scale-wam9k-alldev-20261016-r1 | RUNNING，同全DEV对象/seed/env的旧WAM基线；等待后再做配对比较 |
| shadow-scale-eval-queue-20261016-r1 | RUNNING，GPU5：完整旧BC→固定core两种BC→WAM PPO50/100/200；有界13h |

50k MLP离线DEV固定6208窗动作MAE=.0193933、root=.0330181、finger=.0148517。
这些为标准canonical命令单位，不是手腕位置误差；50k native保持25.52%不是好效果或RL增益证明。
32DEV面板采用MLP20k，不与50k全DEV比例相减；完整评估面板对象排序/重置相同才可配对。

远端部署到已有本项目scope的`repo/outputs/wam-rl-policy/`，98MB包包含新Task代码、精简
MLP50k权重与全部所选原生资产；旧数据/权重不覆盖。全部native链接检查通过，外部vendor只读。
首轮四卡smoke未编译出gymtorch，四rank退出1；supervisor快速失败误标COMPLETED的bug已修，
原始日志/exit_codes保留，不算成功smoke。单进程fresh项目cache构建+加载PASS，重跑r2。
仍在native工程核对阶段，不声称已开始远端有效训练。工程问题不构成方法负证据。

## 后续最小判别（200轮读出前记录）

远端50/100轮core held均71/256，对首次BC69/256，未见明显信号；独立审查发现其中4个
warm后实际初态改变，故不作为配对增益。两个runtime/config/对象分配相同，剩余252初态
逐值同一；相关3个mesh在BC首次VHACD、PPO读取cache，当前仅为候选因果。
保持原始记录；安排在既有warm cache上补BC core/full基线，不删除cache、不剔除失败物体。

Decision：若200轮仍无清晰DEV改善，优先区分“探索幅度不足”和“BC缺乏尾段标签”而非
继续无条件增大预算。已测100轮确定性执行残差RMS约.004，初始latent log_std=-2.5；
增加一个可审计的-1.5初始化选项，保持网络初始mean/权重/数据/奖励不变，先不启动科学运行。
尾段问题先以最便宜离线GPU测量：固定388DEV每对象一条序列(seed244)、每4拍安全current
标签，分别报告startup(<8)、原admitted时段(8..151)、late(152..198)的24D/root/finger MAE。
如果late没有明显异常，不投入全量重训；若late误差明显高，则再考虑明确的新current-safe
准入训练，不得称其与旧forecast准入数据完全匹配。该审计不改变训练、不使用未来输入。

上述离线审计已完成（388DEV每对象一序列，GPU2，1.99s）：startup MAE .005789，admitted
period .020785，late .005250；对应root .009196/.035539/.015350。因此不启动尾段准入重训。
这只说明所采离线安全尾段拟合未见异常，不证明所有闭环分布外尾段都没有问题。

warm-cache BC再次得到与cold BC相同的4环境初态，cold/cache本身不充分解释。独立审查
未见全局RNG分支错误；发现vendor reset内和pre_physics_step末尾在同次simulate前两次
调用同一GPU position-target setter，与本机官方Gym文档建议不一致，尚未证实因果。
下一最小工程probe：只在新输出开`synchronize_warm_step`，把原warm step展开为相同zero
pre_physics→Torch synchronize→simulate/fetch→post；保存pre-warm完整root/DOF/targets/
yaw/forces。固定shard01/128env/seed249，同GPU分别BC与PPO，区分reset输入差异和warm
物理步差异。实际首轮只开同步、不额外CPU读pre-warm；保存pre-warm为独立的capture开关，
避免CPU拷贝隐式同步混淆。若仍有问题再加capture并检查重复setter/foreign-stream临时索引
生命期，不先大改RNG。

同步warm Probe未消除BC/PPO初态差异（hand .05684m、object .16519m等），不能宣称修复。
首次PPO子运行因相对checkpoint在vendor切换cwd后找不到而失败，保留日志；修复CLI提前
resolve后用新evaluate-r2目录完成。下一工程判别为两个互相独立的可选runtime修改：
`retain_reset_indices`保持传给Gym的int32索引引用直到simulate完成；
`single_reset_target_setter`仅移除reset中的中间target setter，保留pre_physics末尾的最终
setter。均只改自有runtime副本，不改vendor、奖励或物理参数。每个条件同GPU/同shard01
BC与PPO50各一轮；先检查实际起点，再决定是否用于后续训练/评估，旧结果全部保留。

retained/single-setter独立Probe均未消除起点差异，当前不能归因为任一单因素。
下一最小控制：BC和PPO采用同一evaluate路径、都加载同一自训练RL checkpoint，仅BC臂
`force_zero_residual`将latent置0，逐步断言实际canonical action==frozen BC action。
此控制不改变BC执行语义，统一checkpoint加载/辅助网络构造/初始内存路径；仍须逐值审计
实际起点后才允许配对解释。若成立用于阶段读出，仍保留初始化问题的根因未明限制。
同时每个新native worker在慢资产初始化前复制自有wam_rl_policy包到输出runtime-code，
避免正在初始化的旧入口后加载到新模块；旧worker缺省可选诊断字段以向后兼容。

## Decision Note：200轮之后

远端MLP 128TRAIN/512env已完成200轮=20377600步；core持握BC69、PPO50/100均71、
PPO200为66（各256），未见预定义10个百分点的信号。252个相同起点之外有4个初态不等；
不同加载路径、warm同步、索引保留、单setter和同checkpoint零残差控制均未消除，因此
根因仍未明，暂不把任一候选解释升级结论。即使这4个结果全部有利翻转，也不足以产生
本面板所需26条改善；该限制不阻止进行最小的探索幅度判别，但正式归因必须补齐。

选择下一最小并行对照：同一MLP50k、前64个固定TRAIN对象（两个原32对象分片），
每臂2GPU×128env、200轮/10188800步，参数/初始化/draw/reset/reward均同，只改初始
latent log_std=-2.5 vs -1.5（标准差2.718倍）。每臂<=2h；各自固定core_DEV进行249/250/251
三个reset读出，不把初态不匹配的逐轨迹转换当配对收益。若强探索出现跨reset清晰改善，
再扩128物体/多seed；否则优先考虑策略可训练容量/BC约束和任务reward，而非无界加轮数。
这是Decision Probe，不是正式Validation；补充250/251仅为exploratory reset，未触及保留池。
初始化深挖先保留现有诊断与限制，最便宜的下一定位是pre-warm完整状态与Gym刷新同步，
待它能改变后续决策时再做，不能让流程完整性替代有效行为推进。

GPU5完成BC基线后距WAM50轮读出约25min，可用这段明确窗口做廉价BC优化判别：从同一
MLP50k checkpoint/Adam/全部RNG恢复，分别再训50k步至100k，constant3e-4 vs cosine降至
3e-5，网络/数据/抽样完全相同。每臂<=600s，串行总预算1200s；固定core_DEV seed249
比较端点，不按DEV挑endpoint。如果annealing带来明显native保持提升，换用更好reference；
否则不投入更多BC重复训练。此处不变更全量数据准入，也不新建分支/替换原checkpoint。

## 本地WAM32轮范围外中断与恢复

旧run `shadow-scale-wam-128obj-4gpu-20261016-r1` 在第32轮采集，rank03 env53触发
`BC H4 workspace invalid`，整组按保护逻辑退出；没有OOM或PPO非有限报错。四个latest
checkpoint均为31轮，actor/critic跨rank逐tensor一致，因此只从共同31轮恢复，不重训旧轮。
原ShadowController已对超范围env返回零BC并保持invalid至reset；Stage0 strict reject对小面板
够用，但放大在线探索必须定义范围外行为。修复FastController只对有限的out-of-support
环境采用相同零BC回退，其他env照常推理；非有限物理仍拒绝。残差执行与native reward不变，
固定199步不截断/拼接轨迹。逐轮记录fallback env/ticks，第一次触发保存原始状态/身份；
checkpoint中的reference明确为“原frozen BC＋范围外零fallback”，不得隐去此语义。
该修复不说明方法有负结果；从共同31轮继续至200，原run和失败日志只读保留。


## Decision Note：探索噪声之后检查表征（2026-10-16）

问题：在线增益有限是否主要来自随机初始化小头难以从混合尺度raw current300学习？
关键证据：原128TRAIN MLP 200轮core held69→66/256；64TRAIN两卡200轮，标准噪声
三reset合计202→226/768，强噪声209→231/768，仅+3.13/+2.86个百分点，且标准第三reset退步。
实际噪声RMS确实增至约2.43倍、800个actor步骤均接受、学习率未塌陷。因此不继续盲目放大噪声。
BC50k→100k constant/cosine，离线MAE改善但core held69→68/74，单纯BC训练量暂不优先。

root选择最便宜的表征Decision Probe：远端两组各2GPU、64TRAIN、128env/rank、200轮、
10,188,800 env steps/组。`shadow-representation-{normalized,bc-hidden}-64obj-20261016-r1`。
normalized用与BC完全相同的raw clamp/训练集标准化300D，再补212个零；bc-hidden用同一次
冻结BC forward的512D末隐藏层。两者均无参数LayerNorm512，拼显式world-relative goal3/.1
和相同BC reference24，得到539D；Actor/Critic均539→256→256相同容量、相同种子和zero mean，
原残差上界、reward、GAE、PPO步数和初始log_std=-2.5保持。没有对标准化输入再额外截断。
它同时改变actor/critic输入，不分离两者效用，也不等价于WAM效用验证。
增加只读critic explained-variance诊断，不改变loss/更新。

预定两臂终点均做64DEV×4起点、reset249/250/251，与同一个RL packet强制零残差BC比较。
远端4/256初态不一致问题经5个工程probe仍未定位，停止无判别性的继续修补；所有实际初态
审计和失败证据仍保存，失败面板不得称为严格配对。该少数异常不足以单独产生26/256的
预定10个百分点信号，但小幅改善仍UNCLEAR，不升级正式结论。
若表征出现跨reset稳定明显增益，扩大128TRAIN和全388DEV；若没有，结合critic拟合诊断，
转而检查长时credit/每批更新预算，而非关闭RL。预算每臂<=2h训练/<=4h全队列；空间保留20GiB，
本地6/7继续空闲，无新增外部授权边界。零残差/冻结性/实际执行/跨rank一致性任一硬合同失败
立即停止本组、保留checkpoint。恢复训练必须显式保留`--observation-kind`；evaluate从packet自动读。

独立只读审查用真实MLP50k权重重放确认：新BC action逐位等于原forward、512D特征冻结，
两臂后27D goal/base一致，同seed初始Actor/Critic逐位一致。11项测试通过；这些只证明工程接线。


## Decision Note：相同交互预算的PPO采集/更新节奏（2026-10-16）

问题：200轮交互不能回答优化是否充分；当前整199步才做4次full-batch actor更新，
总800次。只读核对既有原版
`outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1/plus/dexgrasp/cfg/ppo/config.yaml`
为nsteps8、noptepochs5、nminibatches4。原版网络/环境数/优化器等也不同，不据步数比例宣称
当前实现有错，但优化节奏是有依据的下一项Decision变量。
表征两臂200轮均完成，末段critic EV约.85/.78，暂未见价值完全不学；行为读出仍进行中。
本地WAM50 core held80→83/256，official117→139、final101→114，不能称保持已明显改善。

选择远端fresh两卡full199 vs fresh两卡chunk16；两者固定同BC50k的冻结512D特征、539D小头，
同seed、64TRAIN、256global env、200个完整199步episode/10,188,800 env steps；
不按DEV选择checkpoint/表征。`shadow-cadence-{full199,chunk16}-64obj-20261016-r1`。
chunk16每episode12×16+7，最多52次actor步，200episode最多10400次；每样本仍4个actor epoch。
chunk16同时改变采集/更新节奏、GAE段边界bootstrap与段内全局advantage归一化，不作单一
“optimizer步数”因果归因。相同BC、残差范围、奖励/种子/网络容量/学习率/KL保护不变。

执行前先远端两卡两episode工程smoke，确认26批/最多104个actor步、更新前critic实际next-state
bootstrap、最后7步真正terminal零bootstrap、无内部reset/BC重复forward、跨rankactor/critic一致、
冻结reference不变、latent重放及执行合同。smoke只证明接线；通过后完整200episode。
两臂都待旧表征队列完成后才占对应GPU，不与已有作业竞争。每臂训练<=2h、队列<=4h；
固定core64DEV×4、reset249/250/251读出，沿用10个百分点/方向稳定才继续全DEV的判定。
若无明显收益，不把此结果升级为RL无效；下一步结合实际策略偏移/奖励与保持的差异再判断。
硬合同/非有限/资源越界停止本组并保留产物；本地6/7空闲、旧checkpoint不覆盖，无新授权边界。

测试13项通过；含默认full199与原更新后actor/critic逐tensor相同、非terminal零bootstrap拒绝、
实际next-state bootstrap和更新后新策略重放。独立只读审查未见阻断问题。
`stats.ppo`是最后一段摘要；分段整episode必须读`ppo_segments`及
`actor_optimizer_steps_this_episode`，不得拿最后7步的EV/loss冒充全199步。

补充工程审计：KL保护若最终拒绝候选，原诊断可能记录已回滚候选的KL；现只读重算
最终保留策略的KL，训练loss/accept/rollback不变。新增强制拒绝测试，14项通过。
两卡smoke结束用`tools/audit/audit_distributed_ppo.py`审计跨rank权重、真实chunk长度、
terminal/bootstrap标签、重放误差、冻结性及实际actor更新数后才进入训练。


## Decision Note：训练配方资格验证与WAM直接z输入（用户2026-10-16扩张授权）

当前证据：仅16步分段、其余legacy配方不变，200episode实际10400 actor步，对照full199的800步，
core三reset held202→210/768（full199为202→197）；没有稳定10点收益。
分段曾将LR降到6.25e-6，说明“多更新”不能代替有效的整套训练配方。
BC隐藏表示207→233/768，raw归一化210→188；仍不是WAM直接输入，也不能据此关闭WAM。
用户要求并行扩大网络/优化/奖励/终止/动作自由度，并直接输入WAM z。root选择下面两波，
不修改MISSION claim，不将旧简化基线的负信号当WAM效用的负证据。

第一波：远端四卡各1个训练臂，固定train[0]的32TRAIN物体、128env、200完整episode，
5,094,400真实env steps/臂；每16步更新、5epoch×4等大小随机minibatch，critic同预算。
R0 small(256x256)/residual/native；R1 large(1024x1024x1024x512)/residual/native；
R2 large/reference-gaussian/native；R3 large/reference-gaussian/hold。
同BC50k、同actor种子246/探索247/reset245，不新增未登记seed，
同Tanh激活；raw状态采用同一BC-hidden539D输入。R0→R1检查容量，R1→R2检查动作参数化，
R2→R3检查逐拍保持塑形。full action以base+零head为初始Gaussian均值，执行完整[-1,1]
canonical动作，不再永久±.15；显式log_std=-4.3971199849，匹配legacy -2.5经.15缩放的
一阶初始实际噪声（tanh下并非逐值相同，报告实际RMS）。latent未clip的Normal计算logprob。
BC reference penalty按实际确定性命令偏移计算，三者同权重.1，不更新BC。
hold仅加1/tick的低速位置代理：goal<=5cm、lift>=10cm、object speed<=.1m/s；不使用
连续计数隐状态，不改official success；native reward与bonus分别保存，不以新reward高作为行为改善。

minibatch新路径在整个当前chunk上检查全局KL；首次步过大可quarter-LR校准，已有接受步骤后
到KL边界则回滚该候选并结束actor更新，避免反复用同旧batch把LR无限缩小。它是新资格配方
的一部分，不声称仅minibatch不同于legacy。随机shuffle使用独立保存/恢复的generator，
seed由探索247+1000000+rank派生，不消耗环境reset RNG。各rank仍等有效样本，未增加失败mask。
所有臂均先2episode工程smoke+freeze/重放/执行/实际chunk审计，通过后fresh200（不从smoke挑选）。
终点固定64DEV×4、reset249/250/251，同RL packet加载强制BC对照，实际初态审计失败仍保留
描述性读数且不称严格配对。10个百分点且方向稳定再扩388DEV/训练预算，无信号则继续诊断。

第二波：WAM reference相同、状态/目标/BC动作显式相同，仅z为none/aligned/shuffled，
三组均2631D且同大头/相同初始参数。z从本拍同一次冻结WAM H4 forward的scene/hand/known-goal
有效token分别均值汇总（3×768），逐类型LayerNorm；原始current300完整保留，无state128D瓶颈。
不运行未来readout、不读未来GT、不离线缓存z、不用teacher动作；shuffled donor固定为另一个物体，
不跨TRAIN/DEV。none仅指不直接给RL z，公共WAM BC reference仍包含WAM，不能称完全无WAM。
第一轮是便宜的type-summary接入Probe，不是learned query；负结果不能否定更丰富readout。
z还携带H4与CAD信息；正结果不能单独证明预训练优于等信息原始历史/几何。
拟先GPU5运行2episode工程检查，远端第一波各队列释放GPU后再运行三臂200episode，
每臂32TRAIN、64env（实际smoke吞吐决定是否可128env），rollout16、5×4mini、大头、
完整动作+hold奖励。单臂<=2h，整队列<=4h；smoke若ETA超预算先调整环境数并记录，不能无界延长。
不抢占本地100轮阶段评估；本地新smoke<=10min，预计100轮尚需约40min以上。

失败终止作为下一项独立Decision：不能直接写task.reset_buf，否则native下一拍reset/splice。
需要alive/learning_done/valid吸收状态，失败后不再进入actor/critic/adv/KL，逐env bootstrap归零，
多卡按有效样本sum/count加权。这一实现尚未完成，不把当前固定199步称已具备失败终止。
先跑上述现成可审计的变量；确定保守跌落/飞离定义再补1GPU资格probe，无接触认证不称接触失败。

资源：使用远端0/1/2/3，本地5只做有界smoke/空隙作业；本地0外部作业保护、6/7留空。
每臂独立输出，旧checkpoint/轨迹不覆盖，空间保留20GiB，总产物仍限64GiB；任一硬合同失败
停止本组。23项测试通过（含新动作、大容量、WAM状态保留/类型mask、双rank mini/global等价、
shuffle不消耗环境RNG），这些只是工程资格，不是抓取正结论。


## Decision Note：加入v3，独立特征源与共同BC（用户2026-10-16追加）

已读v3 Task README/STATE/model/normalizer/checkpoint及80k同面板比较。v3 final80k在既有544抽样
预测面板相对v2 best，新增五域手/scene宏误差约-35%，旧三域基本保持；不等于Shadow收益。
原三域物理读数80k优于其31k/41k，因此首个固定下游候选选final80k，41k保留但不在机器人DEV
按结果搜索端点。v2固定同为final80k；两者训练seed/数据曝光不同，不能单独归因九源数据。
原权重只读，远端本scope导出model/stats/config/step供推理，不携带/修改原optimizer或checkpoint。

为真正隔离“给RL的z”，更新第二波设计：所有臂统一fresh自训练MLP50k BC，而非共同WAM BC；
独立FeatureController只载入冻结v2/v3 encoder，使用同H4/已知goal/CAD，独立于BC权重与输出。
v3-none / v3-aligned / v3-shuffled / v2-aligned四臂对应远端GPU0/1/3/2（具体以queue为准），
各32TRAIN、64env、200完整episode=2,547,200步，2631D同大头、fullGaussian显式匹配初始噪声、
hold奖励、rollout16/5epoch×4mini不变。none实际仍计算v3作计算/内存控制，但给RL全零z；
此时BC也不含WAM，因此是无直接/间接WAM信息的动作对照。aligned/shuffled的采集时z入buffer，
PPO不重算encoder。第一波配方结果未稳定前，这仍是探索性信息Probe，负结果不关闭WAM路线。

独立特征模型有单独精确freeze/grad审计；feature支持域外只置零z直至本episode reset，
BC仍执行自己的同拍动作，记录feature fallback env/ticks，不能将其混作BC fallback。
resume必须保留特征来源；evaluate自动从packet恢复。v3使用自己的TRAIN normalizer，
不偷换v2 normalizer；当前只encode不用future head，但依然严格加载完整forecaster state。
没有把v3权重灌入原BC，也没有用官方actor初始化学生。

本地GPU5完整WAM-z/large/fullGaussian/hold/mini两episode工程smoke已通过：26批、109个实际
actor更新、maxKL.019892、重放3.05e-5、冻结597字段不变，elapsed约213s；物理和token路径均执行，
不作为抓取结论。按此实测，原第二波2h/臂预算可能不足200episode；当前用户扩张授权与CAMPAIGN
没有更紧时间边界，root将第二波明确改为每臂<=4h、队列<=6h（含3组固定DEV读出），
仍4远端卡/64GiB总产物/20GiB磁盘保留、本地保留6/7。smoke测实际吞吐后报告ETA，若4h仍不足，
保留共同milestone并停止，不无界延长。各GPU先完成本卡第一波配方队列，再2episode新源smoke，
审计通过后fresh200，终点仍core64DEV×4、249/250/251，不按评估挑端点。


终止优先级的最便宜诊断（已有轨迹CPU复算，无新增模型推理/仿真）：全1552DEV的MLP/WAM BC，
按曾lift>=10cm后跌至<2cm、低于初态10cm、物体平移逃逸>.75m、手物距离>.75m四类合并，
分别150/158条出现，首次后剩余控制帧仅占3.49%/3.57%；WAM PPO50核心256条为16条/2.21%。
这些保守阈值不能覆盖参考偏离/真实contact loss，也不能证明终止无效；它们不足以解释
主要无进展，因此当前降低复杂absorbing-mask实现的优先级，先完成配方/直接WAM特征Probe。
条件terminal及不等样本加权仍保留为后续Decision，不把未实现项隐藏。产物
`outputs/wam-rl-policy/shadow-termination-opportunity-audit-20261016-r1/summary.json`。


v3数据暴露边界已做源4物体身份CPU核对：Shadow388个DEV物体与v3原native源4的1653个TRAIN
物体交集为0；388个全部属于其413个DEV物体。strict recipe只在此基础删除窗口，不新增训练物体。
这只证明源4没有TRAIN对象泄漏；这些物体参与过预训练DEV验证，不冒称独立TEST。报告
`outputs/wam-rl-policy/shadow-v3-pretrain-exposure-audit-20261016-r1/summary.json`。

延后诊断：若当前WAM/配方仍无稳定收益，可检查有限199步horizon但current300没有显式剩余时间
时，critic在终段的拟合歧义，以及直接微调自训练BC/critic warmup的资格对照；当前先完成已启动
的特征决策，不因这些合理疑问立刻继续扩展GPU网格。完整Gaussian的噪声下界和旧residual
具有不同物理单位，若长训触及std clamp必须先匹配实际动作单位边界，不能隐去此差异。

## Decision Note：critic资格诊断，不将失败配方归因WAM（2026-10-16）

四臂small/large/free/hold均已完成200episode及固定三reset评估，没有方向稳定的10点保持增益。
独立只读检查确认实际actor更新27788/18822/22394/25404、critic各52000次，非未训练；
但free最终critic在2048个真实state上逐值输出53.2031326，最后两层Tanh近100%饱和。
hold末层98.63%单元跨state近常数；small内部16步EV中位.972而终末7步为-5.081。
full-action已移除旧cap，但三reset所有确定性动作相对BC偏离仍<.14，不能认为cap已被挑战。
原始运行保留，未发现执行/GAE/冻结硬合同失败；这些负信号归为UNCLEAR：配方尚未资格化，
不作为容量、动作自由度、保持奖励或WAM路线无效的证据。

正在跑的四WAM臂各13段在线EV多数仍为正，并非已确认同样塌缩；初始64states上的低输出方差
不能代表当前采集分布。保留四臂200轮和固定读出队列，不因旧BC-hidden的诊断擅自停止。

最便宜的Decision Probe固定free的first/last两条完整TRAIN采集批次，不新增仿真，GPU拟合
实际finite-horizon MC return（gamma=.99、真实done、终点0、无16步bootstrap/GAE）。
同BC-hidden539+一维clock，Tanh/ELU critic × zero/remaining clock四臂，同large宽度、Linear初值、
batch顺序、Adam3e-4/clip1、1000步/batch1024；只改变critic激活与显式已知剩余动作比例。
按object_code排序每4个留1个，共24TRAIN内拟合/8TRAIN内留出对象；相邻帧与同物体全部起点
保持同组，完全不读DEV。报告训练及对象留出MSE/EV、内部192步/终末7步、饱和率、时间条件均值
基线。轨迹来自段内更新的随机混合策略，因此MC回归资格不等于固定策略真V，也不保证PPO改善。
若ELU显著改善拟合且未牺牲留出误差，下一步短在线matched资格Probe；若只有clock改善，则先
隔离clock；若均无改善，检查目标噪声/尺度，不直接重跑全部WAM。不得根据该离线结果选DEV端点。
run_id=shadow-critic-fit-20261016-r1；输入来自远端本scope既有free产物；fresh四模型不改旧checkpoint。
远端GPU0与本任务v3-none有界共享（约<1GiB新增、总<=10min），不占额外GPU，不占本地GPU5，
不干预本地6/7现有外部任务。超时/非有限/磁盘<20GiB停止本诊断。checkpoint仅保存四份小critic，
输出<256MiB，无新增授权边界；代码提交与输入路径写入manifest。

本地WAM第100轮core读出held80→75/256（第50轮83），lift124→143、到达117→130、终点101→116。
实际首帧审计2/256不一致，故不是严格配对退步证据；保留独立comparison产物，不宣称收敛。

离线资格Probe已完成，实际GPU拟合四臂共12.64s（含输入重构）；smoke20步亦通过。
按物体留出的TRAIN MSE：Tanh-zero830.106、Tanh-remaining571.856、ELU-zero832.982、
ELU-remaining434.903；对应fit MSE455.592/401.573/408.686/322.752。
留出终末7步MSE817.451/92.337/1055.837/126.564。
整体EV .336/.569/.325/.655。单换ELU没有留出改善，clock两种激活下均明显改善；
clock-only时间均值基线1089.249，仍差于完整状态模型，不能说状态不重要。
结果仅PROMISING于时间可观测性的回归修正，未证明PPO或机器人行为已改善。
全部JSON已同步本地`outputs/wam-rl-policy/shadow-critic-fit-20261016-r1/`，模型留远端。
独立只读审查确认目标对齐/按物体分组/四臂Linear初值与batch匹配；默认旧网络初始化逐值不变。
这些fresh540D critic并非原在线539D critic初始化重放，且BC见过这些TRAIN留出物体，边界保留。

### Decision Note：两臂后续持续塌缩，释放GPU做短在线资格（2026-10-16）

新证据改变上一条保留判断：v2-aligned ep30–37连续8episode、v3-shuffled ep26–35连续10episode，
全部13段|EV|<1e-4。实际64×2631输入上v2输出逐值同63.1915817，shuffled仅59.6794853..59.6794891；
两臂最后两Tanh全100%饱和、512末层单元跨state std<1e-5，终末loss约2000–2900。
none/aligned尚无全段近零episode，在线EV仍.426–.961/.554–.979。
root选择通过本任务queue STOP保留并停止v2-aligned、v3-shuffled，标记INTERRUPTED_RECIPE_COLLAPSE；
不能将不等训练预算、未完成审计/未完成DEV面板作为WAM比较，也不认为v2无效。none/aligned继续200。

下一步在释放的远端GPU1/3顺序各跑两臂，Tanh/ELU × zero/remaining clock，fresh25episode、
各32TRAIN/128env、636800envsteps，540D同large actor，fullGaussian/native、16步/5×4mini，
相同BC50k/seed245,246,247/Linear初值、每臂先2episode工程smoke+分布式审计。
run_id=shadow-value-qualification-{tanh-zero,tanh-remaining,elu-zero,elu-remaining}-32obj-20261016-r1。
clock此处同时提供actor和critic，不将任何行为差异单独归因critic时钟。全部臂固定25终点，
用TRAIN内在线EV/终末误差/真实动作偏移/饱和验证优化资格；不按DEV挑配方，不在此时启动正式结论。
每臂<=20min、每卡两臂队列<=1h；沿用冻结/原生执行/重放/真实terminal/bootstrap硬合同。
若末10轮大部分chunk EV>0且无连续3轮全13段近零，可继续其200episode与固定DEV读出；
若四臂都失败，保留结果，检查value target尺度/critic warmup，不追加盲目大规模训练。
两卡通过STOP确认原owned进程结束、nvidia-smi空闲才启动，不能干预其他GPU或外部作业。

在线资格r1在import IsaacGym前失败：新启动脚本漏继承scope/env.sh中的PYTHONPATH，
尚未创建仿真或执行PPO。保留失败日志；r2显式继承同一已知IsaacGym路径/运行root后fresh重启，
不修改模型或实验设计，实际run_id后缀使用20261016-r2。

追加固定中期行为读出：在上述资格队列结束、GPU1/3空闲后，对仍完整训练的v3-none/v3-aligned
各自固定update50.pt做core64DEV×4、reset249的同packet BC强制零偏移/候选评估，使用现有
evaluate_scale_pair与真实初态审计。两组仍继续200；不根据中期DEV选checkpoint或改训练超参。
run_id=shadow-wam-feature-{v3-none,v3-aligned}-u50-core-seed249-20261016-r1。
目的是获得及时的闭环初步读数，若无稳定明显收益仍等固定200及后续修正配方，不追加DEV网格。
每组配对评估<=1h、含等待队列<=2h；相同64物体/256轨迹，非全388DEV，不作正式结论。

停机后可重放检查由`tools/audit/audit_critic_saturation.py`保存至
`outputs/wam-rl-policy/shadow-feature-saturation-audit-20261016-r1.json`（本地已同步）。
最终保留v2/shuffled checkpoint为39/37轮，native训练在STOP响应期间完成了少量额外episode。
v2第39轮后半chunk EV已从接近0恢复到.89，但终段仍.00015；不能描述为“从此不可恢复”的塌缩。
shuffled终点仍连续12轮全13段近零。停止依据是先前长段学习失效和优先诊断配方，而非证明永久失效。
none/v3-aligned第56/51轮仍在线EV正；陈旧初始输入上深层饱和较高不单独触发停止。
四组actor在同一实际初始输入上的四层Tanh饱和率均为0，未发现与critic相同的actor饱和证据；
因此当前只修critic激活/显式时钟，不扩展无证据的actor激活网格。

在线25轮四臂全部完成且各freeze/原生执行/实际chunk审计PASS。末10轮全chunk EV中位
Tanh-zero/.remaining=.866/.916，ELU-zero/.remaining=.953/.966；终末7步EV中位为
.517/.559/-.114/-.204，对应末minibatch物理MSE中位43.1/32.2/368.4/220.5。
TRAIN held均值39.4/38.9/38.2/38.1（每轮128条），没有行为提升信号；不能按最后一轮49条挑选。
clock稍改善Tanh，ELU终段仍不合格；没有证据支持立即将ELU配方推广为WAM“修复”。
全部curve/manifest/audit及摘要已同步本地`shadow-value-qualification-summary-20261016-r2.json`。

### Decision Note：先便宜检验value target尺度（2026-10-16）

目的：区分“必须更大网络”与“深层critic被数十量级的原始target推向饱和”；在线clock/ELU尚不足。
只追加已有free同两批TRAIN轨迹的GPU回归，每臂1000步/同540D/同初值/batch/物体分组，
target除以固定100，推理乘100还原为原物理return单位；所有报告MSE/EV仍用原单位。
与既有scale1比较，完整四臂重复总预计<30s；它同时改变回归参数化与有效gradient clipping尺度，
不宣称只是无影响的数值单位变换。run_id=shadow-critic-fit-scale100-20261016-r1。
只在本任务远端GPU0短暂共享，<=600s/新增<128MiB；不改PPO、不改native reward、不重跑仿真。
若明显缓解饱和并改善留出误差，再实现正确物理bootstrap/GAE的短在线scale1/100对照；
否则不扩张此变量。当前v3两组50轮行为评估继续，本地与远端未停止的200轮训练继续。

scale100离线已完成，四臂GPU合计12.14s。原物理单位留出MSE835.468/444.969/887.441/427.826；
Tanh-remaining相对scale1的571.856降低22.2%，最后Tanh饱和率.657→0；
但终末MSE92.337→171.009变差，不能称全面改善。无clock的Tanh/ELU仍无留出整体改善。
因此只将“scale+clock避免Tanh饱和”视为PROMISING资格线索，不修改reward，不使用此结果宣称抓取收益。

### Decision Note：本地GPU5有界检验physical value缩放（2026-10-16）

本地0/5/6/7现均空闲，1–4主训继续；重新检查GPU后使用5，始终保留6/7。
同机fresh Tanh+remaining clock两臂value_scale1/100，控制Torch/PhysX/BC/32TRAIN/128env/
540D大网络/fullGaussian/native/16步5×4mini/seed245,246,247，固定25episode/636800步。
每臂先2episode smoke+审计，再fresh25+审计，单臂<=20min、队列<=1h，不占本地200轮评估窗口。
run_id=shadow-value-scale{1,100}-32obj-20261016-r1。
实现保证Critic.forward始终返回物理V=value_scale*net(obs)，actual-next-state bootstrap、GAE、
returns和EV仍原native reward单位；critic优化MSE除value_scale²，日志保留物理value_loss并另记
normalized_value_loss。scale默认1，旧checkpoint缺字段自动1，resume必须保留scale，eval从packet恢复。
它改变critic参数化/优化尺度，不是reward scaling，也不是把不同单位的value塞给GAE。
两条单位协变测试覆盖full-batch和minibatch：reward/value/bootstrap同步×100时GAE/return×100、
normalized loss及actor/critic更新相符。该测试只证明单位接线，不证明实验效果。
判定看末10轮真实全chunk与终末EV/物理MSE、TRAIN保持、饱和及执行/冻结合同，不按DEV选端点；
明显改善才将通过的方案用于新一轮matched WAM特征Probe，否则不扩张盲训。

第50轮v3中期抓取读出已完成，两组各256条完整199步轨迹，均为固定core64DEV×4/reset249。
共同BC held69/end66、lift141、到达121、终点106；none PPO held76/end74、lift121、到达109、终点99；
aligned-v3 PPO held85/end83、lift136、到达126、终点111。
两组各自BC/候选初态审计PASS；追加直接none-PPO→aligned-v3-PPO的九类实际首帧审计也全逐值相同。
直接比较held双方44、none-only32、v3-only41、均失败139；不能只报净增9而隐藏32条退步。
因此v3较none多9/256=3.516个百分点，较BC多16/256=6.25点；尚未达到预设10点且只有
单训练seed/单reset，当前UNCLEAR，有小幅正向读数，不作为v3胜v2或正式WAM有效性结论。
两臂继续固定200轮及原249/250/251，不因这个端点读数挑checkpoint/增加DEV搜索。

官方 initialized PPO 复核进一步限定了本卡结论范围：本卡只检验小型 Shadow continuous
scale 配方的 critic/拟合和闭环读出，不能用来判断官方 PPO baseline，也不能据此预期把
WAM 输入直接接入官方大网络后仍保持抓取。
结果与直接比较已同步本地相应u50目录和
`outputs/wam-rl-policy/shadow-wam-feature-v3-aligned-vs-none-u50-audit-20261016-r1/comparison.json`。

scale缩放代码d37bcea3的30项测试通过，含双精度单位协变（避免近零梯度经Adam放大的float32
微量舍入差异）；GPU5原生scale1两episode smoke及实际执行/冻结/13段审计PASS。
当前队列`shadow-value-scale-campaign-20261016-r1`继续fresh25对照，supervisor PID178810，
实测约11s/episode、3GiB显存，两个smoke+25臂预计合计约12min（受资产初始化影响）。
本地主训仍1–4、0/6/7空闲；没有使用外部进程或扩张磁盘预算。

## 2026-10-16 20:12 全部预定队列结束后的读出

本地WAM主训200轮已完成，256global env、累计10188800env steps；本次resume段curve只含32–200，
不能将169行误读成只训练169轮。200轮core实际初态九项全部PASS：BC held80→78/256、
end73→74、lift124→144、到达117→130、终点101→109；未改善持续保持，不称收敛成功。

远端none/v3-aligned均完成200与freeze/分段/执行审计PASS，并完成249/250/251各256条DEV：
none BC→PPO held69→65、59→68、76→70；三面板均有少数实际初态不一致，不称严格配对。
v3 BC→PPO为69→72（初态1条不一致）、58→72（PASS）、75→64（PASS）。
两个有效配对面板方向相反，不能把第50轮+9宣称为稳定复现，或跨不同BC初态拼总和证明v3有效。
none200轮中33轮全部13段|EV|<1e-4（首105），v3为35轮（首86）；v3最后10轮均近零，
none最后10轮9轮近零、200最后几段有恢复。不是不可恢复的永久判决，但现配方没有长时资格。
数据JSON已同步本地对应remote run目录；这是训练配方失效证据，不作为WAM路线有效负证据。

本地scale1/100各25轮及两次smoke均完整审计PASS；末10轮全段EV中位.783→.875、
终末EV.441→.508，物理末minibatch MSE43.05→45.84；TRAIN held均值41.8→38.5/128。
因此缩放有价值拟合信号，却没有闭环改善证据；还没有在WAM2631D输入上检验长期稳定。
20:12查询本地8卡/远端4卡全部空闲，旧队列均按预算正常结束，无新的后台长训仍在运行。

### Decision Note：WAM输入下的critic长时资格，四卡匹配2×2

Blocker：短25轮不能覆盖旧none/v3在105/86轮首次全段近零的问题；下一步不能继续原配方盲训，
也不能直接把BC-hidden上的缩放效果推广到WAM。最便宜可判别设计是保持相同WAM输入形状、
共同已知remaining clock、比较none/aligned-v3 × value_scale1/100四臂。
各32TRAIN/64env、125完整episode=1592000交互步，固定fresh MLP50k BC、v3 final80k encoder，
2632D（raw300+type-summary2304+goal3+clock1+base24）、large Tanh actor/critic、fullGaussian、
hold奖励、16步/5×4mini、seed245/246/247及独立shuffle seed不变。none计算同encoder后置零z。
只有特征条件与value缩放两个因素不同；125覆盖旧105/86轮失稳区，不按DEV挑端点。
run_id=shadow-wam-value-{none-scale1,none-scale100,v3-scale1,v3-scale100}-32obj-20261016-r1。
远端GPU0/1/2/3分别对应上述四臂。先2episode工程smoke+原生/冻结/真实chunk审计，通过后fresh125。
每臂训练<=2h、全队列<=4h；预计约80min训练+固定core64DEV×4、reset249/250/251读出。
本地卡保持释放（至少6/7始终空闲），新增产物估计<8GiB，仍遵守总64GiB/磁盘保留20GiB。

预先停止条件：若连续8个完整episode全部13段EV均已知且|EV|<1e-4，则保存本轮checkpoint、
critic-health-stop.json与原始curve，标记STOPPED_CRITIC_QUALIFICATION后结束该owned训练及队列；
未知EV、短chunk日志或单轮低EV不得触发。停止是配方资格失败，不是WAM负结果。
正常完成125才运行冻结/执行审计和固定DEV；未完成臂不与125臂混作等预算有效性比较。
判断：scale100能避免长时失稳且保留状态区分，才有资格扩大；是否改善抓取另看固定DEV，
若无改善仍UNCLEAR，不把EV改善包装为行为收益。两条scale100 WAM臂都通过时，可在同配方内
评估直接z；v2/shuffled推迟到这一Blocker通过之后，不增加当前无判别性的全网格。
新gate默认关闭，旧运行/旧checkpoint行为不变；仅本轮显式设8。源版本写入实际manifest。

20:22执行快照：实现87d14c7c，本地31项测试PASS，远端Torch2.0的单位协变/停止条件3项PASS。
四组2episode原生smoke、独立encoder/BC冻结、动作执行、13段bootstrap/重放审计全部PASS，
现均进入fresh125（约第2轮），不是从smoke挑checkpoint。四个supervisor PID2112316–2112319。
实测smoke总92–94s/两轮（含初始化），fresh第1轮含初始化约49s；稳态预计约40–45s/轮，
训练约85–95min，三固定DEV评估再约20–30min（若资格失败则提前停止）。
GPU利用率67–80%、约5.24GiB/卡，无OOM/非有限错误；本地全卡空闲，不占6/7。
本地可用167GiB，仍高于20GiB保留。已有失败和200轮产物未覆盖。
