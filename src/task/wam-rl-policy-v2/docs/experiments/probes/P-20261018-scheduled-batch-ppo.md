---
schema: ref2dex.probe.v2
probe_id: P-20261018-scheduled-batch-ppo
experiment_id: P-20261018-scheduled-batch-ppo
date: 2026-10-18
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 3
seed_pool: debug
seeds: [0]
decision_changed_if_positive: expand the scheduled BC warmstart PPO to a training object panel
decision_changed_if_negative: retain BC and repair continuous PPO before scaling
status: PLANNED
run_id: scheduled-batch-ppo-20261018-r1
---

# Scheduled BC 的批量 PPO 试跑

Class: Decision。区分持续 on-policy 更新是否可运行且保留基本控制，与单次工程更新
不能暴露的累计退化。先在一个物体的多环境运行，是扩展训练对象池前最便宜的判别。
不把此结果解释为跨对象成功率或 WAM 条件的因果增益。

初始化：全量 scheduled BC 的 fit-fast/best.pt (step46000)，冻结 WAM step12000。
每拍每环境独立 scheduler/H4 历史，就绪环境合并一次 WAM forward。
565D current+world tau+valid，24D unclipped Gaussian 用于 PPO likelihood；
原生动作 clamp 后另取副本交给模拟器，native reward 不改。

首轮 GPU6，16环境，180拍/rollout，4次 PPO update，各2epoch/4minibatch，
actor LR1e-5、critic LR1e-4、joint KL上限.01；前后各一次确定性均值策略评估。
每条 rollout 在 native terminal 前显式截断，以实测末态 bootstrap，不把截断当失败终止。
随后显式重置全部环境/历史。若提前出现 native terminal 则失败退出，不跨 episode 拼历史。
评估复用 seed0，训练 reset种子为 seed0+update；GPU physics 不承诺完全相同轨迹。
这会限制长时保持结论，后续需要真正跨 chunk 持续 episode 和异步 reset。

预算1800秒、磁盘余量20GiB、最多2个PPO端点；STOP文件/非有限/概率回放漂移/
无接受的actor更新停止。父BC永不覆盖。用户已允许共享GPU；GPU1/7不分配本任务。
环境 graspenv，Python3.8/Torch2.4.1+cu121/IsaacGym，运行版本写manifest。
输出 `outputs/wam-rl-policy-v2/scheduled-batch-ppo-20261018-r1/`。
入口 `tools/run/closed_loop_scheduled_shadow.py --envs 16 --ppo-updates 4 --ticks 180`。

批量历史的部分重置、合并推理、60Hz插值以及原控制合同/目标指标共4项测试通过。
运行结果待记录。
