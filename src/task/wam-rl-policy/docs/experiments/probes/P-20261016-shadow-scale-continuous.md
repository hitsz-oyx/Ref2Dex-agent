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
seeds: [241, 242, 243, 244, 245, 246, 247, 248, 249]
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
