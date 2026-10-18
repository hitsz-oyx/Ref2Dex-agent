---
schema: ref2dex.probe.v2
probe_id: P-20261018-goal-wam-viser
experiment_id: P-20261018-goal-wam-viser
date: 2026-10-18
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-ordered-goal-scheduler
probe_index_in_family: 3
seed_pool: debug
seeds: [0]
decision_changed_if_positive: inspect real scheduled-goal WAM tau before a bounded executor integration
decision_changed_if_negative: locate goal insensitivity or coordinate/input mismatch before physical execution
status: RUNNING
run_id: goal-wam-viser-20261018-r1
---

# Viser审阅实际goal条件下的冻结WAM预测

Class: Decision / bounded inference Probe。区分同一实测状态下，改变局部goal是否产生可见的
手物预测变化，以及当前调度goal是否出现明显不合理的阶段／幅度。正向仅进入执行探针；
无响应或明显异常则先定位条件／接口。最低成本为已有4个TRAIN物体记录与冻结checkpoint，
不运行新仿真，不训练或修改WAM，不以预测误差或敏感性宣称物理可行。

输入沿用前一工程回放的Camera、Clock、Sheep、Pillow四个物体；每个实测tick先按60Hz
更新调度，t-6/t-4/t-2/t构造30Hz H4，每12tick缓存一个状态。参考是已知的整条录制
物体轨迹，不是独立生成的任务轨迹。调度重新使用模型原始32个flow query身份，不能直接
复用旧HTML为显示而重采样的32点delta。目标三臂为0.5倍前视、原调度和2倍前视，均受
同一个下个检查点和位姿跨度上限约束；受限时允许重合，恢复连接段不伪装成路径内目标。

固定快照来自修复版`pretrain-nine-v4-contractfix-warm13000-20261018-r1/step2000.pt`。
推理副本只保留model/config/stats/step。原dense manifest包含远端绝对路径，因此导出副本
把路径改为相对路径并重新绑定副本的manifest digest，保留原manifest、原digest和来源；
几何payload字节与原32点query逐一验证不变，不修改原训练checkpoint或训练进程。

预算：远端本项目机器单GPU、推理<=600s、产物含模型副本<=2GiB、空盘>=20GiB。
本地八卡均有其他用户进程，因此本地不占GPU；远端在本项目训练所在机器做短时推理，
不停止或更改训练，记录资源共用及时间。只传输四个环境的object/hand/time/reset，数值
保持原样，并记录original_env与原始来源。逐样本推理固定PTv3 batch context，未来24帧保持30Hz；当前物体
anchor下的位移旋转回世界并加到当前手点／flow query。实际环境、精度和提交写manifest。
Viser只读缓存、CPU运行，端口8770、最长24h，SIGTERM/SIGINT或到期退出；原8768服务保留。

展示包括实测手骨架／CAD、输入goal、模型预测手骨架和轨迹、模型物体32点未来及C着色。
当前观测和预测未来时刻分开控制。C固定在当前物体表面锚点，不当作未来表面位置。
录制真实未来仅为可选审阅对照，不进入模型或调度反馈；越过记录末端则隐藏。

Result: pending inference and browser review.
Decision: pending user visual review; no robot-benefit claim.
