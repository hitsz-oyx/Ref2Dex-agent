---
schema: ref2dex.probe.v2
probe_id: P-20261022-wam-v6-samples-efficiency-3arm
experiment_id: P-20261022-wam-v6-samples-efficiency-3arm
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-wam-v6-sample-efficiency
probe_index_in_family: 1
seed_pool: probe
seeds: [191, 192, 193, 194]
decision_changed_if_positive: retain WAM pretraining if C reproducibly improves interaction efficiency over both A and geometric B
decision_changed_if_negative: reject current sample-efficiency claim and prioritize executor or representation changes
status: HOLD
run_id: samples-efficiency-3arm-20261022-r1
---

# 三臂抓取—抬升样本效率 Decision Probe

**13:0x v2继续HOLD/协议冻结**：本环境单独生成`synthetic-grasp-lift8cm-position-HOLD1s-timeout20s.v2`，
测的是当前初态向上8cm的精确位置抓取—抬升任务，**不是501条原始完整任务完成率**。
保留原TaskSpec作为来源，source_full_task_success=null；不将UNKNOWN重标成HOLD。
修复复制过来的每tick voxel误终止：编码不可用只给同executor τ=0/valid=false，物理安全/
任务成功独立；真实native三臂总60controls的故意outage smoke PASS，C覆盖13/20。
这是工程检查，不是小规模样本效率结果。

正式RL现在必须通过匹配任务/executor/策略/初态序列的物理资格门。单轨迹另外的5cm
elapsed1s校准source-PD1/1，但GTτ原R/校准R均0/2，且不同于本卡精确8cm；不能放行本卡。
旧A286720/B149504交互受旧reward/编码终止影响，事后重算不能消除训练污染；旧checkpoint
不得作为新协议续跑。见[一页结算](../../audit/negative-result-audit-one-page-20261022.md)。

**12:4x继续HOLD**：prefix完整声明与release时序已修（同501记录评分0→77），物理动作/
indexed reset已过工程检查；新teacher/sourcePD6例完整任务仍0/6，95%上限资格未过，
raw释放意图/连续腕角/动作范围与跟踪误差仍须解决。此时三臂“无收益”无法归因，禁止恢复。
新task10/τ cache必须重建，不能沿用旧prefix任务条件。最新证据：
[共同复核包](../../audit/fix-review-20261022.md)。下文历史“按设计直接跑”不覆盖本HOLD。

这是设计先行的 Probe，不是结果。P0 qualification 优先；已有 executor 未证明
能稳定执行未见机器人 GT-τ 时，本卡不能把 C 的失败归因为预测质量，也不能拿历史
PPO 的训练计数冒充本实验成功率。用户授权按设计直接运行首批，无需另行确认。

## 问题、决策与最低成本

区分“未来预测预训练减少达到抓取抬升目标所需的在线交互”与“额外模型/τ输入并无
可重复收益”。若 C 同时胜过 A 和可优化几何的 B，继续投入预训练表示；否则先修执行
链或删除在线解码。复用已经跑过真实多回合训练的官方 UniDex PPO 与纯 Inspire
物理 host，固定任务/BC/奖励，只改变参考提供方式，避免重写模拟器或并行复现 PointWAM。

首批是 **3 arms × 3 independent policy seeds × 1,048,576 control transitions**，
合计 **9,437,184** 在线训练交互，不以两步 wiring smoke 代替实验。
训练 seeds 191/192/193；固定共同初态评估 seed 194，不计作第四个训练种子。

## 手、物体与任务

- 手：DExplore Inspire，30 Hz、12独立动作，原生18 DOF/coupling保持。
- 物体：`cubemedium`、`cylindermedium`、`mug`。分别覆盖平面/棱边、连续曲面、
  非凸带柄形状；三者均独立核到本地object key；旧cache实际512点，不能冒充2048点。
  root已从同一原生OBJ产生独立dense2048 cache，P1再次实际打开NPZ核到三者
  surface/normals=(2048,3)、query=(32,3)；未用旧512冒充2048。
- 每回合从该次物理初始化的物体位姿产生显式任务：目标世界z增加8cm（position-only，不约束物体朝向），
  终态HOLD至少1秒；任务不取teacher实际未来物体路径。计划CONTACT事件在起点。
- 20秒/600 controls episode上限，15秒acquire timeout，原TaskController负责
  measured grasp/arrival/HOLD判据；成功是TaskController完成，另报lift≥5cm连续30帧。
  后者是抬升代理，不等同稳定接触或全任务完成。
- 本卡测试固定3物体上的学习效率；不是未见物体泛化。P0另行承担未见序列资格。

## 三臂与公平性

| 臂 | 在线参考 | 共享输入/训练 |
|---|---|---|
| A | 从训练第一步起tau=0、valid=0 | 当前q/dq、物体状态、H4、CAD、同一显式goal/CONTACT/HOLD计划 |
| B | 当前CAD/手位置的几何抓取锚点，12轮近表面+proximal优化，生成短参考 | 同A；不读取未来实测状态或未来示教 |
| C | 固定healthy v6 checkpoint4371预测τ；执行器消费前8帧 | 同A；同goal/next-event，不接GT未来；FM不加入本三臂以免增加变量 |

B允许几何优化，当前实现是基于已有CAD与手锚点的确定性近端优化，不宣称已达到最强
完整机器人碰撞约束IK。若B本身无法提供可执行抓取，C>B不能被宣传为打败机主的最强
几何基线；需结合P0和B独立闭环数字解释。三臂均允许PPO修正参考，均不按tau误差奖励。

所有臂的公共shape encoder、任务/奖励CAD与B优化使用同一原生OBJ的dense2048 surface，
C也用该surface；不只给C更高分辨率几何。shape encoder架构/权重相同且冻结，
其点数从BC历史512统一变为2048属于共同输入变化，不能单独归于C。

三臂使用同一`Scheduled686 r4/best.pt`共同初始化：独立CPU加载核到其step=0，
新增tau/valid输入权重为0。保留同一Current421 BC父权重和冻结CAD encoder，critic fresh；
运行入口强制断言step0与新增列全0。所有臂同样继承该父BC的完整示教成本，不能把C解释为
纯零示教RL，也不能仅给C新示教。亲本BC实际训练数据/50物体来源见原卡和checkpoint config；
本实验没有重训BC或添加示教。A从头在无tau分布训练，不是推理时置零消融。

同一846维MLP1024/1024/512/512（686旧输入+160共同H4历史）、12动作、优化器/PPO超参/噪声约定、初始化分布、
奖励与终止、256env、8步rollout、512updates完全相同。B/C新增参考列在共同父策略为0，
只有在线交互训练能建立使用方式；actor当前状态输出在初始时逐位一致。
历史160维为4×[object位置相对当前/.05、符号一致quaternion、11手锚点相对同锚点当前位移/.05]；
startup仅左填充最早已观测状态，不读取未来。三臂actor/critic同846维，历史新增权重为0，
旧BC的686列逐位拷贝；相同task10包含当前phase，CONTACT/HOLD计划对所有臂已知。
沿用pinned official UniDex PPO的实际分布KL、GAE、global advantage实现。
奖励沿用同一`MeasuredRewardTracker`/`TaskReward`，gamma=.96，禁止C额外tracking reward、
contact loss或GT未来teacher forcing。C预训练用未来标签属于预训练成本，单独报告。

## 评估与数字定义

在训练controls **0、262144、524288、786432、1048576**（updates0/128/256/384/512）
评估确定性actor均值。各臂/各训练seed评估使用独立物理host、同一eval seed194、同一
256env/物体清单、同样初始化/重置顺序，保存`initial-state.npz`。跨臂与跨checkpoint必须
逐项比较q/dq/object/hand/sequence、physical248、tick、table pose、
DOF/contact/object force与valid、episodeID全部相等；不相等则该比较无共同初态资格，标记无效并修复。
评估不更新policy，不计入训练环境步数；评估物理调用与耗时单独报告。
只计每个env的第一个episode，后续自动reset回合不重复进入成功率分母。

主要阈值预注册为**50%任务成功率**，且连续两个评估点达到；阈值交互量取首个达标点，
未达标写`>1048576 (right-censored)`，不把未达标当作0步。固定预算主结果为1048576步
successes/256，另报lift5cm30/256、每物体分母、平均return、失败原因和max lift。
学习曲线面积用共同交互坐标上成功率梯形积分除以1048576，范围[0,1]；缺checkpoint不插值
跨大段冒充完整AUC。3seed各自数字全部列出，再报mean±sd，配对seed差值；不要把256env
当作256个独立训练seed。仅3seed的置信度有限；本卡结果仍属Probe。

支持样本效率收益的探索门槛：C相对A和B均至少降低20%阈值交互量，且3seed方向一致；
阈值均未达时只报告固定预算和AUC差，不宣布已证明“更少步达到目标”。
**推翻条件**：相对A和B没有可重复效率收益；或收益来自更多BC数据、不同奖励、额外GT
未来条件/teacher动作、不同终止和预算。若所有臂在零附近，结论是当前实验未建立可学的
执行任务，不以更长预训练解释失败。不得事后选择只报告C较好的物体或checkpoint。

## 资源与执行

物理GPU2/3为本线程用户授权资源；每次新队列启动先`nvidia-smi`确认空闲。0/1仅在
root明确确认空闲后选用；4/5/6 flow长训与7禁止使用。不停止现有进程，不push，不碰凭据。
每GPU同一时刻一个独立seed/arm，9个作业排队；arm顺序随seed轮转。
每run上限6h或512updates，任一先到保存停止。任一训练未达相同预算，则只能比较共同
已完成交互预算，不能将不等预算终点当公平固定步数结果。最多512updates的入口硬界，
首批不自动扩展；输出≤8GiB，磁盘可用低于20GiB停止。每run保留0/128/256/384/512及终点
checkpoint，全部9run约3GiB模型，禁止视频和全量tau缓存。

C模型推理按64个环境分块（容量实测后采用），避免256env一次PTv3 dense2048推理OOM。新吞吐须实测；
历史约36s/12288全局交互不能外推v6 dense2048。训练开始后记录nvidia-smi显存/利用率、
每update collection/learning时间、首个稳定阶段ETA。若ETA超过6h，保存已完成曲线与共同
预算，不静默缩小成功率分母/种子数。GPU被占则保留checkpoint，不杀外来进程。

环境：policy/WAM `/home2/wyy/miniconda3/envs/graspenv/bin/python`，Torch2.4.1+cu121；
physics `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`，Torch2.0.1+cu118。
使用原生GPU PhysX，不静默换CPU；已知历史GPU PhysX间歇性非法访存风险保留。
项目tmp放socket/extension cache；输出全部在Task outputs下。

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/run/launch_samples_efficiency.py \
  --output outputs/ptv3-shared-wam-v6/samples-efficiency-3arm-20261022-r1 \
  --geometry-cache outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry \
  --gpus 2 3 --seeds 191 192 193 --arms A B C --envs 256 --updates 512
```

入口生成每个policy/native的完整命令、process.json、manifest.json和curve.jsonl，
checkpoint内保存model/optimizer/BC来源与运行环境。新launcher只管理自己创建的子进程。
主入口、环境和物理host分别是`tools/run/launch_samples_efficiency.py`、
`tools/run/train_samples_efficiency.py`、`tools/run/samples_efficiency_env.py`、
`tools/run/samples_efficiency_physics.py`；共享历史脚本不修改。

## 成本与解释边界

C预训练成本不能写0：它继承远端九源checkpoint与本地适配4371；完整祖先GPU-hour若无法
从现有manifest可靠核到，则明确“不可完整核到”，报告可核到的本地steps/elapsed和源路径，
不得编造。在线推理每次调用保存batch、wall seconds，另报ms/env、吞吐与P50/P95；B也报
几何优化耗时，A报policy-only。训练交互收益与总wall/GPU成本分别报告。

执行状态与结果由实际运行产物回填。设计时训练runs=0、环境交互=0、成功率/AUC未测，
不把尚未执行的计划数字当作效果数字。

## 启动前工程核查（不是训练结果）

2026-10-22：4个汇总/共同历史/几何合同测试通过（连续两点评估阈值、缺点不算完整AUC、共同初态精确比较）；
几何参考平移等变性实数组检查通过。4个入口py_compile通过、train --help通过。
仍需真实GPU异步reset/PPO/三provider smoke；通过前不声称训练启动成功。

工程smoke run_id=`samples-efficiency-engineering-20261022-r1`，work_version=`samples-efficiency-v1`，
GPU1启动前实测6MiB/0%；3env、每臂2updates、seed191，强制异步reset只是接线检查，
不进入学习效果结果。P0当前机器人GT0/16、原动作回放15/16的口头状态由root提供；
P1不得据此将任何臂的失败归因为生成器。

### 工程smoke已完成

`5bfee65c`工程实现、`samples-efficiency-engineering-20261022-r1`：A/B/C各3env、
2 PPO updates、40 optimizer steps、48训练交互，三臂native与policy全部COMPLETED。
source/episode动作错配、PD误差、logprob重放误差、参数同步误差均0；异步reset0/1均通过。
第2轮collection为A0.338s、B0.389s、C1.183s，learning0.142/0.141/0.161s；
模型CUDA峰值192/192/576MiB。不是256env吞吐估计或学习效果。
可核数字：`outputs/ptv3-shared-wam-v6/samples-efficiency-engineering-20261022-r1/engineering-summary.json`。
GPU1退出后6MiB/0%，没有动其他作业。后续首批实现收紧保存：只保留0/128/256/384/512
及至多一个partialstop，不受官方PPO自己的log-save schedule影响；actualforecast identity记录健康4371
（A为no-tau，B为geometry）而不是BC亲本历史v4路径，arm/846/H4/workspace/task合同严格绑定。

预训练成本独立核对（root读healthy4371实际checkpoint）：elapsed_s=5210.235427，
即1.4473小时累计wall，当前world_size=3；累计elapsed跨不同早期拓扑，不能直接×3
冒充完整GPU-hours，完整祖先九源训练成本UNKNOWN。该时间不是本P1在线训练成本。

### 256env真实规模工程容量已完成，尚非学习曲线

`efb1b408` / `samples-efficiency-capacity-20261022-r1`，GPU1，A/B/C各256env、
2updates、40optimizer steps、4096训练交互；native与policy全部COMPLETED，
source/PD/episode/logprob/参数同步误差全部0，异步reset0/1通过。每臂只保存0/2两份
checkpoint，实际验证official logsave guard。第二轮collection为5.485/8.532/40.642s，
learning约0.16s，模型CUDA峰值204/204/716MiB。以当前C吞吐512轮约5.8h，超过原4h，
因此原首批不能直接按4h发车；先优化v6推理batch或明确修改公平预算，不静默缩水。
实际资源采样和数字见`outputs/ptv3-shared-wam-v6/samples-efficiency-capacity-20261022-r1/engineering-summary.json`。
该容量工程仍用旧公共512 CAD，非科学A/B/C结果；正式实现已统一dense2048公共输入。
统一2048后独立GPU1实测50物体共用shape encoder输出(50,64)、全finite、peak110.53MiB；
所有4合同测试继续通过。首批须绑定修正后的代码版本，不混用工程曲线。

性能适配：runtime d7145b24支持inference_batch64+forecast_only；healthy只跳未被policy消费的独立C/scene头，
完整24帧手τ路径保留；root独立做full/only数值parity。P1 identity绑定batch64/forecast_only，
这不是压缩τ/P2消融。新容量run_id=`samples-efficiency-capacity64-20261022-r1`，C256env2updates，
统一新dense2048；通过前不启动科学长批。

### 启动预算与冻结修订

root授权将所有臂的run wall上限统一6h，训练交互目标仍512updates/1,048,576，
评估每个独立host也受6h硬界。capacity64 C2562updates已COMPLETED，40optimizer steps、
4096交互、source/PD/logprob/episode/同步误差0，第2轮26.921+0.162s，模型peak1714.87MiB。
root独立parity：同一encoder激活下healthy/FM hand decoder maxdiff均0；独立encoder重算
漂移0.0381mm/0.000536mm，不能宣称端到端bitwise一致。

按256env容量估计：A约0.80h/run、B约1.24h/run、C约3.85h/run，9训练jobs约17.7 GPU-hours；
双GPU训练约9–12h（任务离散/吞吐变化会延长）。评估另算：C每256共同初态panel约34min，
15个C panels约8.5 GPU-hours；A/B评估较快但也需计物理调用。总预计约13–17h wall，
这是容量外推，实际ETA以正式curve/resources更新，不能写成已运行完成时间。

启动时复制完整依赖代码到output/runtime-code，记录snapshot.json的commit/文件清单；
policy/native/后续seed均从snapshot读取实现，data/checkpoint/cache仍读原项目的固定路径。
root之后修改main不会改变排队seed的实现。官方PPO源也来自snapshot。
优先调度9个训练作业，空闲lane才处理已完成checkpoint评估，防止静态切片把三个C全部压同卡。
每run写interaction-costs.json，native.steps×envs为实际交互，包含STOP前采集但未更新的rollout；
checkpoint/manifest另记实际环境步，不能把只计已完成update的步数冒充全部成本。
最终自动汇总JSON、learning-curves.png/svg；缺点/缺seed标INCOMPLETE，不填0。

### 冻结与真实评估通路工程验证通过

冻结smoke r1在native首次创建前失败：snapshot跳过data/cfg symlink，缺inspire.yaml，
训练交互0；失败产物保留。4f02b123显式冻结选中训练YAML，未复制大assets，fresh r2重跑。
`outputs/ptv3-shared-wam-v6/samples-efficiency-frozen-smoke-20261022-r2/`已COMPLETED：
A/C各3env1update、各24训练交互；各自checkpoint0均通过真实独立host评估450controls。
两份14字段初态NPZ逐字段完全相同；自动summary退出0并生成JSON/PNG/SVG，summary正确标
INCOMPLETE（只是工程通路，不是9job科学结果）。r2全实现/依赖从snapshot读取。

正式启动另加项目内XDG_CACHE_HOME/MPLCONFIGDIR/CUDA_CACHE_PATH，避免driver/图形缓存落home；
C冻结推理用fork_rng恢复CPU/CUDA策略RNG，不让PTv3随机序列化消耗PPO探索流。
正式批次独立run_id仍为samples-efficiency-3arm-20261022-r1，不混入以上任何工程数据。

## 正式首批已启动（结果待定）

run=`samples-efficiency-3arm-20261022-r1`，冻结代码fe81b6b5，supervisor PID2708769，
独立session、GPU1/2。启动前两卡均6MiB/0%，未动4/5/6或他人作业；其他空闲卡未占用。
完整命令/启动receipt在`tmp/samples-efficiency-3arm-20261022-r1/launch.json`，supervisor.log同目录；
运行数据根`outputs/ptv3-shared-wam-v6/samples-efficiency-3arm-20261022-r1/`。

首个正式大规模状态：A-seed191迭代10、200optimizer steps、20480训练交互；
B-seed191迭代5、100optimizer steps、10240训练交互。source/behavior logprob/参数同步误差均0。
实际dense2048并行采集A8.25s/iter、B16.25s/iter，learning约0.16s，模型CUDA峰值202MiB；
整卡约4.95–5.01GiB。30s资源采样A0–17%、B0%，采集/几何CPU路径主导、GPU间歇空闲，
没有因此改变冻结设计或减少样本。A当前剩约1.17h，B约2.31h；正式C尚排队。

上述A/B比早期单job512点容量更慢，旧17.7GPU-hour总训练估计偏低。
以当前A/B与新C容量估算9训练jobs约22GPU-hours，双卡训练约11–14h；评估另计，
整批暂估16–22h wall。此为在线更新ETA，不是完成承诺；C正式运行后应再次更新。
目标仍9job各1,048,576训练交互、3训练seed、45共同初态评估panels。
目前完成科学训练runs=0/9、完成评估panels=0/45，阈值/AUC/固定预算成功率均待定。
不能用上面的工程数据或训练计数给出样本效率收益。

交接时独立读取正式progress：A-seed191=31 updates/63488交互，collection8.19s+learning0.15s，当前剩余ETA1.11h；B-seed191=16 updates/32768交互，collection16.37s+learning0.16s，当前剩余ETA2.28h。
训练episode计数不等同独立共同初态评估，本轮尚无完成的科学评估panel或样本效率结论。
独立supervisor将继续有界运行并自动生成最终JSON/PNG/SVG，不需要本对话持续轮询。

## 2026-10-22 owner HOLD — negative-result audit

Owner prioritizes task-success reachability and action/reset contracts before any RL sample-efficiency attribution. Root STOP requested for this owned finite queue; preserve partial checkpoints and physical-interaction costs. No further train/eval jobs authorized in this queue until audit resolves these blockers. Partial or missing seed/arm results are not a no-benefit conclusion.

受控STOP已核对：A191 STOPPED139更新/284672已更新交互/286720实际物理交互；
B191 STOPPED72/147456/149504，合计实际436224。两者各多采一个未更新rollout。
C、其余训练seed与科学评估均未启动（0/45 panels）。没有阈值步数、AUC或三臂无收益结论。
顶层FAILED是STOP后继续索取缺失milestone checkpoint的编排状态，不能算学习失败。
此前“supervisor将继续”及RUNNING/ETA均为历史记录，已由本HOLD覆盖。
物理同状态逐轴q/dq与原生双环境reset隔离尚未认证；CPU逻辑检查不能替代它们。
完整原因与证据：[负结果审计](../../audit/negative-result-audit-20261022.md)。

### 12:3x 修复后仍HOLD

UNKNOWN准入、身份标识、scale防线与奖励统计已修；不代表真实成功路径与native动作/reset
资格已经通过，不据部分A/B结果否决三臂。v6官方PPO未接线；独立CPU到PD的接口smoke
不替代物理认证。[修复与下一验收](../../audit/negative-result-fixes-20261022.md)。
