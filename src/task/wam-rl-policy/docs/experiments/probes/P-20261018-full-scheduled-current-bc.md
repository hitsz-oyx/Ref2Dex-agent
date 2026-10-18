---
schema: ref2dex.probe.v2
probe_id: P-20261018-full-scheduled-current-bc
experiment_id: P-20261018-full-scheduled-current-bc
date: 2026-10-18
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 2
seed_pool: debug
seeds: [0]
decision_changed_if_positive: use full-data scheduled BC as the PPO warmstart candidate
decision_changed_if_negative: retain full-data current BC and inspect scheduled conditioning before PPO
status: COMPLETED
run_id: full-scheduled-current-bc-20261018-r1
---

# 全量现有BC接入调度WAM

Class: Decision。用户明确复用已有全量BC，不以Camera小样本的拟合结果作为扩展门槛。
假设：保持已有控制能力的初始化可以学习真实调度WAM条件，并作为PPO起点。
最便宜的方法：复用既有缓存和50k checkpoint；先做批处理接口/吞吐工程检查，
然后全量条件缓存和BC继续训练。物理性能需后续同池native评估，离线误差不证明成功率。

数据：`outputs/shadow-finetune/policy-data-full-20261015-r1` 全部已准入窗口。
TRAIN 10118序列/1567对象/1536897行；DEV 2338序列/388对象/354761行。
这是筛选后的1955对象，不是官方所有2066对象无筛选。复用原TRAIN归一化和DEV panel。
现有 `shadow-current-bc-seed241-20261016-r1/best.pt` 模型权重迁移；新训练随机种子0。

输入565维：current300 + world tau264 + history-valid1；第一层新增列置零，
初始化保持原BC均值输出。前6拍tau=0/valid=0，由原BC控制而非零动作。
调度器每60Hz实测帧更新，初态到任务位置目标的路径保留初始姿态；
H4取当前及过去2/4/6拍。冻结WAM step12000，使用其绑定dense overlay，
30Hz预测插值到60Hz前8拍；无未来teacher轨迹输入。C暂不作为actor输入。

入口：`tools/run/train_current_bc.py prepare-scheduled`，逐记录flush完成标记，可恢复；
只有全量COMPLETED缓存可训练。`train --init-current`复用原训练器；保存初始DEV指标和
best checkpoint，避免较差的继续训练覆盖父模型起点。PPO actor支持该565维模型。

输出：远端 `/mnt/ugreen_nas/storage/oyx_storage/scheduled-current-bc-20261018-r1/outputs/wam-rl-policy/`；
本地运行记录 `outputs/wam-rl-policy/full-scheduled-current-bc-20261018-r1/`。
环境：远端graspenv Python3.8/Torch2.0.1+cu118；GPU0与本项目WAM共享，
检查占用身份，记录峰值显存/ETA；缓存24h、BC2h、磁盘余量20GiB停止线，最多保留既定端点。

## 当前证据与限制

权重迁移、PPO概率回放、原BC和tau合同10项测试通过；实际首条序列调度/历史构造通过。
全量旧geometry中不足512点的对象全部被绑定dense overlay覆盖，无需删除对象。
全量条件生成、继续训练、565维native执行和真实PPO训练尚待运行结果；不声称已完成。

## 执行位置修订

用户要求改为本地训练，并明确允许共享GPU。使用本地GPU6（3090 24GiB），
不向其余卡派发本任务；已有monica任务不停止。环境graspenv Python/Torch详见运行manifest。
远端流水线在WAIT_CURRENT_CACHE阶段已终止（PID2667797），未开始条件GPU推理/BC；
本地直接复用既有current缓存，不重建。输出根
`outputs/wam-rl-policy/full-scheduled-current-bc-20261018-local-r1/`，
条件与训练子目录分别cache、fit。共享显存/吞吐实测后记录ETA。

本地工程检查：真实父checkpoint step50000，128条DEV状态的迁移动作最大差
8.34465e-7；新在线startup/history适配器在内共11项合同测试通过。
4条记录/608行条件缓存完成；batch32与单条tau最大差0.000244m（阈值0.001m），
PyTorch峰值713MiB。检查仅覆盖一次batch，不声称所有状态批处理等价。
全量生成接续运行，完成后自动进入50k BC；运行状态/吞吐见本地产物的
`pipeline-status.json`、`cache/progress.json`，失败会阻止后续训练。

用户质疑全量条件预计算的成本后，主动STOP结束本地流水线，已完成183条记录，
可恢复缓存保留，未进入BC训练。现有全量current BC本身不依赖新增缓存；
穷举每个状态的WAM条件是新增条件BC的方法选择，不是PPO warmstart的必要条件。
实测约150行/秒；端到端包含CPU调度/输入构造、batch32完整WAM forward、逐记录flush，
未做分项profiling，不能把全部耗时归因于GPU推理。

## 缓存吞吐优化（工程）

用户澄清要求加速同一份全量缓存，不改成跳过缓存/PPO优先。
`tools/audit/benchmark_scheduled_cache.py`在固定6条序列测batch32/128/256：
每条146个有效状态，batch256实际只运行146条；稳定forward从约0.65s降至0.33s，
显存峰值约1610MiB；与batch32的tau最大差0.000415m，未超工程阈值0.001m。
CPU逐序列调度约0.49–0.55s、在线输入构造约0.09s、tau转换约0.034s。
这些是组件计时，不冒充端到端加速。

优化实现：worker同时准备调度状态和完整WAM输入tensor，pin-memory预取，
GPU整条序列一次推理；使用4个worker，与GPU重叠。原数据范围/划分/归一化不变。
新增wait/transfer/forward/tau/write计时和排除首条冷启动的吞吐/ETA。
worker输入与原主进程构造逐tensor精确相等测试通过。
保留原183条部分缓存，新的batch协议输出到同一run的`cache-fast`，不混写。
先测50条工程端到端吞吐，再接续全量生成与BC。

优化端到端50序列/7600行：排除首条冷启动后2.121序列/s，约322行/s，
相较旧1.065序列/s约1.99倍；预计剩余5849s（约1.6h），需随对象和共享负载更新。
PyTorch峰值1857MiB。50序列累计forward17.43s、tau1.77s、write4.94s；
wait8.49s几乎全部来自首次worker启动，稳定CPU预取已能跟上GPU。
全量生成已恢复到`cache-fast`，`tmux full-bc-cache-fast`持久运行，完成自动接续BC。
测量记录`optimization-check.json`和`benchmark.json`。未宣称所有状态严格批次不变。

用户进一步授权多GPU并行：选择GPU2/3/4/5/6，GPU1/7不分配本任务，GPU0已有任务不动。
`parallel_scheduled_cache.py`仅恢复已初始化缓存；各rank拥有global_record_id % 5
对应的不重叠行区间，分离done位图、日志、process/manifest/progress；主进程锁定协调权，
只在全部rank完成且合并位图全真后发布COMPLETED和启动BC。既有已完成行直接跳过。
单卡旧流水线先通过STOP退出，避免两个写入系统并存。时间上限6h、每卡4CPU worker。

## Native接口工程检查

新增`closed_loop_scheduled_shadow.py`独立入口，不改正在迭代的旧GT-tau runner。
固定TRAIN bottle、seed0，两次reset各96个60Hz控制步，最长300s；本地GPU6/graspenv。
初始使用旧全量BC扩展565（新列为零），验证完整接口启动、真实动作和反馈、episode reset，
以及每个发出目标与48步后实测姿态的对齐。尾部不足48步的目标显式censor，不能当零误差。
该工程检查没有teacher查询，也不用于宣称目标可达：初始actor尚未学习利用tau。
产物`outputs/wam-rl-policy/full-scheduled-current-bc-20261018-local-r1/native-parent-r1/`。

原生父BC接口检查完成：两次96步；每次前6拍valid=False、第7拍起True，
reset初始300维观测逐值一致；actor均值相对父BC最大误差7.153e-7/5.960e-7。
终态距目标3.43/6.09cm、最大抬升20.45/20.42cm，第一条仅末尾9帧达5cm门槛，
不称稳定成功；同初态GPU物理rollout仍出现差异，不作为精确配对因果比较。
每条有49个0.8s目标审计样本；均值误差14.01/14.72cm，父BC尚不利用tau，
只能说明测量链路可用，不能据此修改goal或宣称Scheduler不可达。

新增`--ppo-smoke`：同一native入口采样原始Gaussian动作，保存真实reward/done，
截断从实际末态重新构造565维输入并bootstrap；单次4-minibatch PPO更新检验概率重放、
actor/critic梯度与KL回滚，reward不改。属于工程资格，不是PPO效果实验。

## 完成结果

五卡条件缓存完成12456序列：TRAIN1536897行，DEV354761行。独立全量扫描确认
所有输入有限、current300与原缓存逐值相同、history-valid和startup零tau逐行正确。
结果`completion-audit.json`；原始split和标签未变，未丢弃对象。
BC续训50k步耗时171.46s，best在46000；DEV动作MAE从0.0193933至0.0170210（下降12.23%），
root0.0330182→0.0291519，fingers0.0148516→0.0129774。继续训练本身与WAM条件效果混杂，
没有matched state-only续训对照，不能将改善归因于WAM。选择按既定DEV panel，无native选模。

best checkpoint的native两次96步完成，终态4.454/3.151cm，最后仅5/7帧在5cm门槛内，
最大抬升20.26/19.36cm；是单TRAIN对象的接口检查，不是稳定保持或泛化验证。
`native-conditioned-r1/`保存完整测量、当前目标、tau、canonical/applied动作和reset证据。
0.8s目标端点均值误差15.09/13.14cm：期间目标持续重规划，不能把该指标直接解释为
固定目标跟踪失败或目标不可达，只说明局部目标和执行结果的监测链路已可用。

`native-ppo-smoke-r1/`为父BC扩展起点的真实64步on-policy工程更新：
behavior log-prob最大误差5.34e-5（通过既有1e-4门限）；actor接受1步、critic4步，
2次候选回滚后actor lr2.5e-6，joint KL0.004803，first-layer最大变化2.503e-6。
bootstrap来自实际末态；使用未修改原生reward。新critic explained variance为负，
不声称价值学习完成或PPO改善。独立工程checkpoint不覆盖BC。

结论：全量scheduled BC已具备后续PPO warmstart的接口资格；长程PPO和效果评估未完成。
后续优先接批量原生rollout和持续训练，保留父BC/state-only对照；任务阶段/可达性保证
仍不由本次接口资格推导。参考正式接口文档。
