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
