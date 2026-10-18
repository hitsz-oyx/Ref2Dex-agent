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
status: COMPLETED
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

Result: 4个TRAIN物体68状态204组冻结预测完成，43.54秒；20项几何相关测试、全部缓存独立复核及Firefox真实WebGL2交互检查通过，仅条件敏感性与展示资格。
Decision: 保留Viser作为goal与真实预测的审阅入口，先人工检查准备/随动/丢失阶段，再决定有界执行探针；不改WAM训练，不声称物理可达或机器人收益。

## 实际运行与检查

- 实际推理代码`cfe01df6`，远端graspenv / Python3.8.20 / PyTorch2.0.1+cu118 /
  NumPy1.24.4，RTX4090 GPU0，bfloat16 autocast、物理量输出float32。
- 4物体各17状态，204次单样本forward用时43.54秒，PyTorch峰值allocated398MiB。
  与本项目训练短时共用GPU0，采样总显存约7.86→8.79GiB；完成后回到7.86GiB，
  原训练PID2638833仍运行。本地无CUDA推理，Viser只用CPU。
- 本地缓存约20MiB；checkpoint推理副本保留在远端本次独立临时目录，不必下载到浏览服务。
  `snapshot.json`记录step2000原始来源、路径迁移的manifest身份和逐payload验证范围。
- 相对调度臂，短／长前视的预测手点末端平均差分别为：Camera3.34/4.92mm、
  Clock3.78/5.74mm、Sheep3.65/3.49mm、Pillow3.10/4.05mm。
  只说明改变goal能影响本组网络输出，不解释为更长更好、预测正确或可执行。
- `data-audit.json`复核全部68状态204预测：原始观测、goal锚点、SE(3)跨度、预测t0、
  真实未来可用mask、C概率和30→60Hz原生τ。锚点最大误差1.12e-16m，τ误差1.87e-9m。
- 三项新边界测试与原17项调度测试通过；Viser骨架／轨迹连线单独smoke通过。
- `browser-audit.json`记录Firefox136 / Xvfb / llvmpipe真实WebGL2检查：四物体、
  首末观测与首末预测时刻16组合，观测／未来滑块、固定观测的预测播放、单／三栏布局、
  跳到随动、真实鼠标旋转。截图`viser-review.png`已查看。
- 原几何HTML服务8768保留；Viser端口8770，进程、命令、24小时停止条件写
  `viewer-process.json`，当前观测状态写`viewer-state.json`。

Limitations: 参考与实测来自同一条已有轨迹，不能检验偏离参考后的真实恢复；固定稀疏
观测之间的目标更新未在界面中插值成新的模型推理。几何上限、接触proxy与目标条件敏感性
均不认证可达或抓稳。仅实测右手11点和模型预测骨架，不虚构完整机器人手网格。
仓库全局`verify.py --changed`仍被trajectory-policy既有六张卡的v2元数据缺失阻断，
本次新增卡与工具未出现在报错中；日志保存在`tmp/goal-wam-viser-20261018-r1/verify.log`。
