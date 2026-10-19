---
schema: ref2dex.probe.v2
probe_id: P-20261019-native-ppo-comparison
experiment_id: P-20261019-native-ppo-comparison
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 7
seed_pool: probe
seeds: [246]
decision_changed_if_positive: expand matched-training controls if current route beats historical official policy
decision_changed_if_negative: investigate task coverage and executor before claiming WAM benefit
status: COMPLETED
run_id: native-ppo-comparison-20261019-r2
---

Decision：在当前相同16物体×4环境、seed246、确定性200步首回合协议下，
比较历史official initialized model_10000.pt与当前自训练BC→PPO5500+WAM80000。
GPU6/7各一臂，graspenv，每进程900秒，主训练不改。
native分支严格加载官方300D actor/critic，直接接原生300D观测，不运行WAM推理，
不添加tau、不使用BC actor、不使用WAM workspace失败终止。共用成功/保持指标。
当前启动器为代码复用仍加载BC/WAM对象，但native策略和观测路径均不使用它们。
评估初始状态逐字段核对；训练初始化、面板和交互预算仍不同，
此比较是历史系统表现重测，不是WAM净收益的matched-training消融。
产物：outputs/wam-rl-policy-v2/native-ppo-comparison-20261019-r1/。

## 结果与对齐审计

r1发现重置初态不匹配，不用于配对结果。r2代码bd0817ae：仅评估reset随机数
改为独立CPU RandomState(seed246)生成同均匀分布，再送GPU；避免CUDA/JIT
初始化历史对reset随机流的影响。预热后重设seed、清除goal-reset标志，
正式reset后把桌面root恢复saved初态并清零速度（仅初始条件，后续物理不改）。
两臂采用同协议，初始原生300D、手点、object pose、goal、完整root逐位一致。
当前565D附加WAM轨迹条件，官方300D路径完全不运行WAM推理；新测试覆盖该隔离。
37项测试通过，严格state_dict加载成功。

|模型|成功|抬高10cm且目标5cm内保持60步|平均回报|
|---|---|---|---|
|历史官方初始化PPO10000|45/64 (70.3%)|33/64 (51.6%)|136.07|
|自训练BC→PPO5500+WAM80000|47/64 (73.4%)|38/64 (59.4%)|140.25|

当前路线多2例成功、5例保持，是小幅正向Probe，不能宣称显著优于官方或隔离WAM净贡献。
两臂optimizer_steps=0、workspace_failures=0，逐回合记录在r2对应evaluation.json。
官方微调32物体与本次16物体无重合；当前路线96训练物体包含本次16物体。
这不代表官方发布模型预训练未见这些物体，但说明两臂微调训练覆盖不匹配。
初始化（官方发布权重 vs 自训练BC）、交互预算也不同，因此它是历史系统同面板
评估而不是matched-training消融。保留继续路线的资格，不据此声称WAM有因果增益。

运行耗时原生约7.19秒、当前约49.08秒（64环境首回合），不含启动，
证明接入完整WAM有明显推理开销，不能把成功指标的小幅改善说成计算效率优势。
主六卡训练未动，GPU6/7评估进程已结束。证据r2/comparison.json与initial-state-audit.json。

## 官方发布原始模型补测

用户要求直接重测state_based_model.pt，不再把我们追加10000轮的模型称为官方原始模型。
run_id=published-ppo-evaluation-20261019-r1，GPU6、64环境、相同16物体/seed246/
cpu-reset-rng-and-canonical-table.v1协议，900秒上限，graspenv。
权重来自本地已有outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1/
plus/dexgrasp/example_model/state_based_model.pt；native300路径无WAM推理和PPO更新。
报告iteration=null，因为官方发布权重的训练迭代数未知，不标记为本地10000轮。
与r2历史微调/current WAM的初始状态和指标比较，其他已述训练覆盖限制仍适用。

### 原始发布模型重测结果

r1原始模型30/64成功、22/64保持，但与此前历史比较run的初始状态不匹配，
不作跨run严格配对。r2同步重跑两臂（代码8a93bb34），初始native300、手点、
object pose、goal、完整root状态逐位一致；不以同seed代替实际初态核对。

|模型|官方成功|抬高并保持60步|平均回报|
|---|---|---|---|
|官方发布state_based_model.pt|30/64 (46.875%)|20/64 (31.25%)|89.57|
|自训练BC→PPO5500+最终WAM80000|47/64 (73.4375%)|38/64 (59.375%)|129.08|

增加17个成功、18个保持；当前路线有1个workspace失败，按失败回合计入64分母，
未剔除；两臂都无优化器更新。此结果支持当前完整系统在该TRAIN面板上优于
未经本地微调的官方发布模型，不隔离WAM净增益，也不代表全对象泛化。
官方发布模型内部训练迭代数未知，报告null；明确区别此前本地追加10000轮的模型。
GPU6/7已释放，主训练不变；证据published-ppo-evaluation-20261019-r2/
comparison.json、initial-state-audit.json及两臂evaluation.json。

## 发布权重来源与复现差距审计（2026-10-19）

Decision：官方结果表与本地发布模型成功率存在大差距，先排除权重、actor适配及最近评估重置修改，决定是否必须修正baseline后再比较。
最便宜的动态诊断是使用已有原生official_unidex_ppo.py入口，不使用新增CPU reset RNG/预热重置/桌面规范化；同16物体64环境、seed246、官方发布权重、GPU6、graspenv、600秒/2GiB上限，不训练。
run_id: published-legacy-reset-audit-20261019-r1；git_commit: 218d6362。沿用已有环境资产，不能作为完全上游复现；初态不配对，仅用于排查整组近期重置改动。

审计结果：
- 官方README明确将state_based_model标为trained policy；论文状态策略通过GeoCurriculum和GiGSL（specialist PPO微调、DAgger蒸馏）得到。发布权重本身没有逐轮训练履历，不能把“未经过我们追加微调”说成“官方未训练/未微调”。
- 官方train结果表有3200个object-scale条目；当前16个code+scale全部在官方TRAIN中，其公开成功率算术平均92.5375%。2145为该split去重object code数，不能混同object-scale任务数。历史32物体追加微调集与当前16物体不相交，不意味着官方预训练未见过当前物体。
- 从固定上游commit下载原始checkpoint，17220395 bytes，与本地文件逐字节相同。64个实际初始观测经过原始上游actor和本地适配actor，动作max abs diff=0。
- 旧原生评估入口完成：29/64成功（45.3125%），15/64保持，33/64抬高；仍远低于官方表。它不采用近期CPU reset RNG/预热重置/桌面规范化，说明这组改动不是低分的唯一必要原因。不同初态，不能将29与30的差解释为重置改动的精确效应。
- 该旧入口manifest中的BC_loaded只是bool(model)历史字段，iterations=10000只是未使用默认值；本次mode=evaluate，未训练、未加载我们BC。草率读取这些字段会误报实验身份。

结论修正：保留此前本地配对结果，但撤回“已经复现官方基准并优于官方”的解读。当前存在很大的官方发布模型复现差距。相同初态证明两臂本地比较可比，不证明仿真/资产/协议与官方一致。针对本地分布追加PPO可能补偿差距，尚无因果证据，不能解释为WAM已胜过官方方案。权重文件和actor前向已排除；共享仿真/资产/评估协议尚未隔离。后续优先用单物体原始上游--test与官方资产/物理配置对齐，之后再做同初始化、同训练覆盖/预算的有无tau消融。主训练未更改。

审计产物：outputs/wam-rl-policy-v2/published-provenance-audit-20261019-r1/{panel-audit,checkpoint-identity,actor-equivalence}.json；动态诊断：outputs/wam-rl-policy/published-legacy-reset-audit-20261019-r1/。环境graspenv，Python3.8、Torch2.4.1+cu121，GPU6。

来源：[官方README](https://github.com/PKU-EPIC/UniDexGrasp2)、[论文](https://arxiv.org/html/2304.00464)、[官方逐object-scale结果](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/results/state_based/train_set_results.yaml)。公开结果是不同采样协议的参考值，不与本地64场景视为配对实验。
