---
schema: ref2dex.probe.v2
probe_id: P-20261019-inspire-scheduled-executor
experiment_id: P-20261019-inspire-scheduled-executor
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 10
seed_pool: probe
seeds: [246]
decision_changed_if_positive: integrate the DExplore official PPO after the measured WAM BC bridge passes
decision_changed_if_negative: fix the first failing native or conditioning boundary before scaling
status: RUNNING
run_id: inspire-scheduled-20261019-r1
---

Decision工程Probe：能否将冻结WAMv4→空间goal scheduler→自训练BC执行器部署到
DExplore Inspire真实物理闭环。用户授权在Shadow评估后尝试，使用GPU6/7共享；0–5不动。
最低成本方案：一条TRAIN的s1_spheresmall_lift、300控制拍；先原生teacher正控制确认
该任务可执行，再用已有/本次同执行TRAIN示教拟合BC。只有通过32拍原生动作/PD审计，
才扩展到完整lift；之后再接官方DExplore PPO的有界更新，不启动大规模训练。
仅工程可行性，不把单序列拟合、teacher正控制或旧WAMv2结果作为新方法成功/泛化。

新合同：current delta0 721 + world displacement tau[8,11,3] + valid1=986，
30Hz H4[-3,-2,-1,0]，right slot，WAM点序输入/原生点序τ输出，单位米，
8未来时刻1/30..8/30秒；startup历史不足时tau=0/valid=0。
原生18动作中只训练12独立维度，六个coupling维度明确展开、由原生控制器覆盖。
冻结最终WAM80000、新Inspire canonical512/flow32/interaction128身份显式替换并绑定。
使用已知完整物体参考SE3路径，非未来实测手轨迹；当前原生721仍含同刻reference。

预算：单次900s、全流程≤3600s、≤2GiB输出、低于20GiB停止新增写入。
原生固定dexplore_v120_train(Torch2.0.1+cu118)，WAM/BC为graspenv(Torch2.4.1+cu121)，
有界本地AF_UNIX IPC，socket在项目tmp，不存视频/全量预测cache/多checkpoint。
原生执行复用collect_rl660_batch的初始化、capture和动作/PD审计，不更改vendor；
新adapter不得用官方teacher接管学生失败。记录动作、实测hand/object、原生contact、
终态/曾抬升与tracking，不能仅用物体z上升宣布稳定抓取。
