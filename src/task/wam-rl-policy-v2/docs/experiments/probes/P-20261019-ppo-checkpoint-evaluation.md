---
schema: ref2dex.probe.v2
probe_id: P-20261019-ppo-checkpoint-evaluation
experiment_id: P-20261019-ppo-checkpoint-evaluation
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 5
seed_pool: probe
seeds: [246]
decision_changed_if_positive: continue training if fixed-panel performance improves
decision_changed_if_negative: investigate plateau or regression before increasing training budget
status: RUNNING
run_id: checkpoint-eval-20261019-r1
---

Decision Probe：判断不同checkpoint在固定初始场景上的确定性策略表现是否改善。
GPU6/7并行独立评估1000/3000/5000/5500轮；每个64环境、同rank0的16 TRAIN物体，
同seed246，首次完整native回合（200步上限），random_time=False，动作act_inference，
不调用PPO更新、不生成checkpoint。graspenv，每个进程900秒硬上限，四次总预算30分钟。
初始状态保存并逐字段对比，首次done后不再纳入该环境统计，避免短回合偏置。
官方成功和抬高10cm、目标5cm内保持60步分别报告。不是DEV或holdout评估；
64场景小样本、单seed，不能据此宣称收敛或泛化。主六卡训练不变。
产物：outputs/wam-rl-policy-v2/checkpoint-eval-20261019-r1/。
