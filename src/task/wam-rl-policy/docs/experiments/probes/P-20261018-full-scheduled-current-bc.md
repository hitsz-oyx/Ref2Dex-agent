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
status: RUNNING
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
