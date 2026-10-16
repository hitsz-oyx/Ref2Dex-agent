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
