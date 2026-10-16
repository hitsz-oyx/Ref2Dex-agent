---
schema: ref2dex.probe.v2
probe_id: P-20261016-official-unidex-ppo
experiment_id: P-20261016-official-unidex-ppo
date: 2026-10-16
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-official-unidex-ppo
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 249]
decision_changed_if_positive: use the qualified official PPO as the common trainer before testing WAM
decision_changed_if_negative: diagnose official engine, asset/bootstrap compatibility, and training budget before attributing failure to WAM
status: UNCLEAR
run_id: official-unidex-{smoke,train32,evaluate}-remote-20261016-r1
---

# 先接通官方UniDexGrasp++状态PPO

Result: 尚未完成；不能把入口接通或短smoke当作抓取能力复现。
Decision: 按用户要求先用远端官方PPO训练，底座通过后再接WAM。

Class: Blocker/Decision。用户怀疑自写PPO实现或配方导致失败，要求先跑官方路径。
服务总目标A的训练底座与B的共同对照；本地UniDex任务资格不替代最终DExplore目标。

## Decision Note / 最小判别设计

问题：保留上游状态PPO采集、GAE、分布、优化和奖励，能否从随机初始化学会抓取？
最便宜方法是复用已有32TRAIN物体资产，直接运行固定上游PPO类，不继续修改自写循环。
上游commit `6c9e710b04769bf5d9765e3233820101cb3d718f`。
用户已明确指定远端训练；远端四卡启动前均空闲，先用GPU0，其他卡留待固定评估/后续对照。

第一步工程smoke：256env、32物体、32个8步rollout、seed45、<=600s。
必须验证actor/critic参数改变、有限loss/参数、640次实际Adam step、保存/重新加载。
第二步fresh训练：32TRAIN×32env=1024，seed245、10000个8步rollout（81920000环境步），
训练<=7200s、checkpoint每1000更新，任务总产物<=4GiB，每个run<=2GiB，磁盘保留20GiB。
若smoke显存不足先降工程env数；正式环境数/预算改变须在本卡记录，不静默追结果。

官方`1024,1024,512,512` ELU actor/critic，正交初始化，原300D观测、24D动作，
nsteps8、5epoch×4minibatch、共同Adam3e-4、gamma.96、lambda.95、native reward，
adaptive LR/desiredKL.016，init_noise_std.8。原module实际scale_tril=exp(log_std)^2，
有效初始标准差.64；原KL使用另一尺度约定，首次原样保留，不静默修复。
不加载BC或官方actor初始化；官方示例脚本默认model_dir非空，此处显式取消。
官方训练random_time=True，只随机第一次episode计时，不是物理中间态RSI。

## 启动适配与证据边界

入口`tools/run/official_unidex_ppo.py`（相对Task）调用原`process_ppo`及
`PPO.run/update`/`RolloutStorage`，逐run复制冻结源码。
状态分支不使用PointNet；仅移除两个无条件vision import，并补未使用的backbone config键。
既有原生环境bootstrap只保留最大异构aggregate容量修复，物理/奖励/动作实现不改；
runtime工作目录与assets软链接位于新run，不能向共享资产目录写入。
progress/log/optimizer保存为被动观测；不改变采样、损失、梯度或参数更新。
依赖只安装到远端本任务独立目录，不能覆盖现有conda环境或他人任务。

固定终点为10000；另记录预定1000中间点作训练进度诊断，不能按DEV挑模型。
确定性评估原始完整199步、random_time=False，报告native sticky success及
held60+lift10cm，保存真实初态。第一轮baseline不构成WAM比较或小数据claim。
至少有8/128个实际保持病例且参数加载/执行通过，才视为足够支持下一项WAM Probe；
否则先区分训练不足与执行缺陷，不关闭PPO/WAM假设。

## 独立审查

AGENTS第14节只读审查未发现自写GAE明确错误：相同输入与gamma下return最大差4.77e-7。
两边均对未裁剪样本计算log-prob、再裁剪执行。关键配方差异包括官方有效探索std.64、
大ELU网、短rollout与纯adaptive LR；自写冷启动std.223、小Tanh网、硬KL回滚，
第25episode三臂只执行55/70/36次actor更新（名义上限260）。这足以动摇此前归因，
但不是已经定位自写实现bug。保持官方原貌建立有信息价值的基线。

## Limitations / future evidence

32物体缩小面板、1024环境，与官方全对象论文结果不是等规模复现。单seed、复用探索面板，
只形成Probe判断；正式WAM收益要在相同官方底座、初始化、交互预算和实际初态下另做对照。
WAM接法尚未启动；不把官方checkpoint评估或微调混成从零学习。
deadline到达时保留已有checkpoint，标记未完成，不把TIME_BUDGET当正常完成。
