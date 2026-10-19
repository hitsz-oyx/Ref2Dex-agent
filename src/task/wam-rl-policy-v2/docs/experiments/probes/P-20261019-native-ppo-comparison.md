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
status: RUNNING
run_id: native-ppo-comparison-20261019-r1
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
