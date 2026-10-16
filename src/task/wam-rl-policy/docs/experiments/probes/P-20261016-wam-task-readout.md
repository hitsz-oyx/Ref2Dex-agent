---
schema: ref2dex.probe.v2
probe_id: P-20261016-wam-task-readout
experiment_id: P-20261016-wam-task-readout
date: 2026-10-16
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-wam-task-readout
probe_index_in_family: 1
seed_pool: probe
seeds: [246, 247]
decision_changed_if_positive: qualify a goal-conditioned spatial readout for direct-action PPO with transition auxiliary learning
decision_changed_if_negative: diagnose transfer and observability before spending on another large PPO comparison
status: UNCLEAR
run_id: task-readout-20261016-r1
---

# WAM任务读出：空间交互先验能否帮助实际回报学习

Class: Decision。服务物理预训练辅助低机器人数据RL；不更改MISSION claim。
用户要求先改进大WAM的使用机制，不再把BC拟合作为主线前置关卡。

## Decision Note

现有PPO仅使用scene/hand/goal三组768D平均特征；没有任务读出学习，未用未来预测头。
现有V3物理预测改善不等于控制收益，旧PPO还有critic资格问题。选择先实现小型状态/目标
查询模块，从冻结V3空间token读取任务信息，并利用真实短期物体运动和回报监督。无教师
动作监督、无BC权重或BC隐藏态输入。WAM没有动作条件，不当作反事实动作模拟器。

最便宜判别：重用已完成、审计通过的32TRAIN对象原生轨迹，按对象24/8拆分；
拟合实际有限时域MC回报和16/32控制步后的物理量。全部同一组训练数据、预算、seed。
先做离线资格，再决定是否接直接输出动作的PPO；不把离线误差改善当作策略提升。

## 固定协议

- 输入：`outputs/wam-rl-policy/shadow-value-scale1-32obj-20261016-r1/rank00/`
  的first/last episode、environment-objects.json；两条轨迹来自不同阶段的旧BC/PPO，
  同episode内也会更新策略，因此目标是实际混合行为回报，不是固定策略V。
- 32对象按object_code排序每四个留出一个，整对象所有环境/两episode共同拆分。
  这是TRAIN内部读出泛化，不是预训练未见对象或独立DEV。
- 每episode取8个预先固定tick（6至166等距），共2048条；H4为[-6,-4,-2,0]，
  原生60Hz→WAM30Hz。current300、当前目标及remaining clock为公共旁路。
- WAM冻结V3 final80000，来自既有远端完成checkpoint，只读下载到本run inputs。
  CAD来自既有policy-data-full缓存；不重新采集教师动作。
- raw-query获得与WAM相同的全部CAD xyz/normal/动力学、hand当前/历史、goal点信息，
  24D统一原始token；WAM-query使用全部encoder tokens，不用未来观测输入。
  所有query为128D、4head，由当前状态和目标生成；mean为原三类型均值投影。
- 五臂：raw-query+aux、wam-mean+aux、wam-query仅return、wam-query+aux、
  wam-query+forecast+aux。最后一臂还输入冻结预测头16/32原生步的手/物体位移预测。
- aux为16/32步的物体局部位移xyz、到当前goal距离、最近hand landmark到物体中心距离，
  共10维；后者不是接触真值。回报gamma=.99，episode终点bootstrap0。
  输入/标签尺度只用拟合对象统计；不读取action/base/latent用于监督或输入。
  原生current300自身含上一拍已执行动作，作为共同历史状态保留，不是待模仿的动作标签。
- fresh初始化seed246，batch seed247（复用既有PPO seed作匹配，不是独立复现），
  AdamW3e-4、batch128、固定1000步，梯度clip1；return标准化MSE，aux标准化MSE权重.2。
  最终端点比较，不按留出挑checkpoint。所有臂保留相同公共状态编码和回报头宽度。
- Gate：query+aux或forecast+aux留出对象return MSE同时比raw-query+aux、mean+aux
  至少低10%，且物理误差不比mean差超过5%，才记PROMISING并优先接在线PPO。
  对象宏平均MSE为主，另报逐对象/fit/aggregate/相对TRAIN时钟均值。
  query-only用于判断aux是否有贡献，不能靠增加aux笼统归因空间读出。

## 资源与停止

使用启动时空闲本地GPU0；用户已授权空闲卡，仍不占用未知进程。模型拷贝约0.4GB，
缓存、日志和模型总产物<=4GiB，磁盘保持20GiB以上。提取<=20min，五臂总训练<=15min；
任何非有限/输入合同错误立即停并保留原产物。记录GPU占用、显存、吞吐、ETA。
成功后只进入有界在线资格；失败先查读出学到什么，不扩大seed/步数追分。

## 参考与边界

[RLT官方说明](https://www.pi.website/research/rlt)启发小型可学习RL读出，但它使用
VLA参考动作与离策略学习，不能证明本路线无BC或PPO有效。
[Masked World Models](https://arxiv.org/abs/2206.14244)启发任务相关辅助监督，
并不把本项目未动作条件化的WAM变成可规划模拟器。

## Limitations / future evidence

单seed、小面板、旧行为分布、有CAD和原生完整状态；收益尚需在线PPO、固定机器人
交互预算、预训练随机初始化/打乱对照及独立对象面板验证。不同token维度导致参数量
不同，须报告；本probe最多决定是否投入任务读出，不证明预训练因果作用。
已有轨迹利用不等于从零PPO成功，也不满足完整小数据通用策略claim。

## 2026-10-16 首轮结果与下一项最小判别

固定1000步结果（fit/留出对象回报MSE，物理标准化MSE）：raw-query+aux
40.65/771.38、.709；wam-mean+aux 15.09/594.18、.596；query-return
11.71/668.98、.985；query+aux 16.57/614.03、.567；query+forecast+aux
16.65/665.94、.628。TRAIN时钟均值留出1119.10。完整数字以audit/summary.json为准。
mean比raw降低22.97%，但新query+aux比mean差3.34%；forecast进一步变差。
两条新接法均未通过预设gate，固定资格为UNPROMISING；不能否定WAM表示或PPO路线。

Decision Note：不扩大该读出训练或直接进入新PPO对照。先利用已提取的冻结预测，
和同轨迹真实16/32步点位移逐项比较，加入零位移/恒速基线、运动>5mm子集。
该统计区分预训练未来输出是否在当前行为分布可用，决定下一步偏向预测校准/动作落地，
还是继续改任务读取。纯数组/坐标统计CPU<=2min，新增输出<=5MiB，不新训模型。
不是追加seed或按留出挑选checkpoint。若预测失配，保留latent先验，不能直接用原预测
做reward/规划；若预测本身有信息，才进一步检查融合。原有gate和结果不追改。

### 目标语义核验后的Decision Note

冻结forecast在留出对象上，16/32步scene误差72.49/141.00mm，零位移26.52/45.75；
hand63.68/125.97mm，零位移36.49/56.76。运动>5mm子集方向相同。
但追溯`goal_point_wam.data._object_goal`及`pose_mesh_data.windows`发现预训练条件是
真实t+24（30Hz，原生48步）物体终点；`fast_bc.batched_inputs`提供整个任务goal位置、
保持当前姿态。当前实际轨迹不保证在0.8s达到整个任务终点。因此以上误差只描述部署
接法的不匹配，不是同条件下预测能力负证据。诊断不能借未来标签修复线上输入。

下一项最低成本：固定已有2048样本中有完整48步未来者（7ticks×128×2=1792），
冻结V3、同实际H4，比较4种条件：原task-goal；causal-scheduled（当前到task goal位移
乘min(48/remaining,1)，不指定未来旋转）；zero-displacement；oracle完整t+48点终点。
后三者新增推理，原task-goal缓存重用。所有方法同一subset、16/32步误差、整对象留出；
oracle仅诊断，不进入策略、回报拟合、数据统计或可部署结果。若oracle误差较task至少
降低30%，优先研究短期目标生成/可实现性桥接；若未改善，回到域迁移/时钟/decoder问题。
scheduled须同时胜zero才能视作可部署目标桥接线索，否则不把缩小运动幅度当改进。
GPU0<=5min、新产物<=10MiB，纯推理无训练。保持第一轮结论，补充其适用条件。

oracle终点天然含未来信息，须再用完全相同终点作线性几何插值基线：物体点按终点
位移×h/48，手点按物体终点刚性变换的位移×h/48。只做CPU坐标统计，不新训模型。
这区分终点本身的信息与WAM学到的手物协同；oracle不构成线上可部署方案。
