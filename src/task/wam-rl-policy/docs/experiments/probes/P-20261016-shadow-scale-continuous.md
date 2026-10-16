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
