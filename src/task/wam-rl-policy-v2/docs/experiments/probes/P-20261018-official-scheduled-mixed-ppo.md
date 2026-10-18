---
schema: ref2dex.probe.v2
probe_id: P-20261018-official-scheduled-mixed-ppo
experiment_id: P-20261018-official-scheduled-mixed-ppo
date: 2026-10-18
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 4
seed_pool: debug
seeds: [0]
decision_changed_if_positive: retain upstream PPO as the common trainer and expand mixed-object training
decision_changed_if_negative: diagnose the BC/WAM interface under upstream PPO before changing the algorithm
status: RUNNING
run_id: official-scheduled-mixed-20261018-r1
---

# 六卡混合物体：复用官方 PPO + 自训练 BC + WAM-v4

Class: Decision。用户要求六卡混合训练，并明确要求优先复用已有官方PPO实验，
避免重新实现训练算法。此协议取代尚未启动的自定义六卡PPO计划。
假设：既有全量scheduled BC可通过官方PPO底座在混合TRAIN物体中继续学习。
最低成本路线：已有官方run/update/storage + 薄观测适配；先六卡32iteration工程检查，
随后直接启动1000iteration训练，不用小样本表现作为大规模训练前置门槛。

依据旧Task的 P-20261016-official-unidex-ppo 与 P-20261017-wam-runtime-correction：
固定上游6c9e710b；复用 `third_party/UniDexGrasp2-official-ppo` 和 process_ppo，
不是重写GAE/collector/update。旧实验曾有重复residual和全批history清空缺陷；
本次没有residual adapter，每环境独立history/reset，三条动作均值路径测试一致。

保留官方cfg：nsteps8，5epoch×4minibatch，共同Adam3e-4，gamma.96，lambda.95，
adaptive KL .016，原reward，native episode/reset/random_time=True。
actor/critic隐藏层1024/1024/512/512 ELU，critic官方初始化。
官方scale_tril=exp(log_std)^2和官方KL尺度约定均保留；init参数.8，实际std.64。
不移植上一轮自定义PPO的硬KL回滚、小critic或std.05。

必要差异：
- actor均值替换为全量scheduled BC (best46000)，包括其冻结归一化，全部权重可训练；
  不加载官方actor checkpoint。
- 565D current300+world tau264+valid1，冻结WAM-v4 step12000。
- 六卡平均梯度、全局KL、全局advantage均值/无偏std；单卡公式与上游一致。
- 多卡总192环境，六个互斥TRAIN分片各16物体×2环境，共96物体，非全量在线物体池。
- 每run记录上游源码的精确最小替换列表；PPO.run不改，GAE递推不改。

资源：本地GPU0–5，GPU6/7不分配本任务，已获共享monica轻负载授权。
先32iteration/96物体/192环境工程检查（每卡640optimizer steps），<=900秒。
随后1000iteration，共1536000训练环境步，<=7200秒；每200iteration及首/末端点保存，
仅rank0保存模型和optimizer，原BC不覆盖；小型各rank曲线与资源日志保留。
运行代码复制冻结，旧数据与模型只读复用；磁盘20GiB余量停止线；任一rank失败则
supervisor终止本任务其余进程组，不触及其他任务。STEP与checkpoint版本分别记录。

入口：`tools/run/launch_mixed_ppo.py`，默认使用官方engine。
旧自定义PPO入口仅留作历史工程诊断，不作为本轮六卡训练器。
23项测试通过，包括上游PPO.run AST不变、分布/动作路径一致、TRAIN分片与实测物体绑定。
性能、成功率与稳定保持均待实测；官方探索噪声可能损害BC控制，须记录而非事后改配方。

## 六卡工程检查通过

r1资产路径预检未进入训练；实际worker r1因原生scale_buf为dict而停止，
修正身份读取后r2完成，代码ce93103b。六卡32iteration共49152环境步，
每rank严格640个官方Adam step，全部learning-audit PASS；动作路径误差0，
首/末端点六卡参数误差0，各rank跨过43次环境reset。
全程136.6秒，纯训练115.5秒，约3.61秒/iteration；单卡约4.1GiB总显存。
预计1000iteration约1小时。注意smoke过程manifest的legacy ticks=180未参与官方循环，
实际nsteps由保存的官方training-config.json确定为8；正式训练metadata已明确记录8。

正式运行 `outputs/wam-rl-policy-v2/official-scheduled-mixed-20261018-r1/` 从同一BC重新初始化，
不从32iteration工程端点继续选模。加入被动episode计数/native success/held60+lift10cm，
不改reward或更新。tmux持久化supervisor，固定代码副本与最多2小时预算。
