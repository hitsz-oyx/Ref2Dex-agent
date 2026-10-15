# UniDexGrasp / ++ ShadowHand 数据用于 WAM-v2 微调：官方来源审计

日期：2026-10-15。范围：官方论文、官方代码和本地已固定的官方源码；不下载大数据、
不运行 GPU、不把现有几何 provider 的通过当作控制监督已经通过。

结论：它适合扩大共享 WAM 的手物运动/几何预测数据；现有 ShadowHand action **不能直接
拼入 Inspire q_PD 动作监督**。要利用其大量 policy 执行数据，可以给 ShadowHand 增加独立
控制头，或在 Inspire 中重新执行并采集 q_PD；后者才得到与当前抓取控制同语义的标签。
这是由手型和控制定义推导的工程判断，尚无本轮收益实验。

## 1. 三种容易混淆的数据

| 项目 | 官方提供/研究的内容 | 能否等同于大规模执行示范 |
| --- | --- | --- |
| UniDexGrasp 的抓取生成集 | 静态物体、手根姿态、关节姿态、抓取质量/接触相关标签 | 不能；稳定终点不含接近、闭合、举起时的状态—控制序列 |
| UniDexGrasp 原始 policy | simulator、goal-conditioned teacher/student 训练代码及物体 split | 公开 README 未给固定执行数据集的 episodes/frames 总数 |
| UniDexGrasp++ | 复用物体/初始化资产；公开 state-based policy checkpoint，可自行 rollout | 可以采集真实仿真执行；总量由采集次数决定 |

原项目把抓取生成和执行策略分为两个独立子目录；生成 README 明确 DFCData 是 grasp
labels；++ README 的发布项是代码、state-based checkpoints、结果摘要，而非某个固定
轨迹包。[原项目](https://github.com/PKU-EPIC/UniDexGrasp)、
[生成数据说明](https://github.com/PKU-EPIC/UniDexGrasp/tree/main/dexgrasp_generation)、
[++ 发布说明](https://github.com/PKU-EPIC/UniDexGrasp2#update)

2026 的 `unidex-ai/UniDex` 是另一个人类视频/多手基础模型项目；现有
`paper/unidex/schema.json` 不应改成 UniDexGrasp。
[2026 UniDex 官方仓库](https://github.com/unidex-ai/UniDex)

## 2. 官方规模应怎样读

UniDexGrasp 论文当前公开 HTML §4.1 报告 **1.12 million valid static grasps、5519 object
instances、133 categories**；其静态生成划分为 3251/754/1514。这里不是连续 trajectory。
本次没有把常见的 5355/1.32M 数字套用到它，也不把静态实例数当作当前 policy 可采 mesh。
[论文 §4.1、Appendix A](https://arxiv.org/html/2303.00938)

原 policy README 写 train 3200、test seen 141、test unseen 100 个 object-instance 条目。
这些文件可重复出现同一 mesh 的不同 scale，**3200 不等于 3200 个不同 mesh**。
[官方 policy 数据说明](https://github.com/PKU-EPIC/UniDexGrasp/tree/main/dexgrasp_policy#dataset)

对本地固定的 ++ commit `6c9e710b` 对应 split 文件保留重复键再取并集，本地输入准备报告为：

| split | 文件条目数 | 不同 mesh | 不同 mesh-scale |
| --- | ---: | ---: | ---: |
| TRAIN | 3200 | 2145 | 3200 |
| test seen | 141 | 137 | 141 |
| test unseen | 100 | 43 | 78 |

这是**发布文件的本地统计**，不是重新声明论文 benchmark 的正式实例数。不能普通
`dict` 解析就覆盖同 mesh 的早期 scale。代码和本地报告：
[`prepare_unidexgrasp_assets.py`](../../src/task/video-data-preparation/tools/run/prepare_unidexgrasp_assets.py)
的 `split_file`；
[`selection.json`](../../outputs/video-data-preparation/unidexgrasp-policy-assets-20261014-r1/selection.json)。
原文件可查
[TRAIN](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/cfg/train_set.yaml)、
[seen](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/cfg/test_set_seen_cat.yaml)、
[unseen](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/cfg/test_set_unseen_cat.yaml)。

++ 论文本身 §5.1 写 3165 instances，附录超参数写 N_train=3200；因此正式比较时应报告
我们实际读取的 mesh/scale/episode/frame，不用一个笼统的“3000+”混合这些单位。论文
成功率 85.4%/78.2% 是其 vision policy 结果，不能保证当前公开 state checkpoint、本地
初始化和采集子集也有这个比例。[++ 论文](https://arxiv.org/html/2304.00464)

**官方已经公开的固定动态示范包的精确 episode/frame 数：本次未找到，未知。**
不能从 PPO 的 num_envs、8-step rollout、DAgger buffer2000、或论文训练步数反推出一个
已经发布、可直接下载的离线行为克隆数据集。

## 3. 控制接口：与 Inspire q_PD 差在哪里

固定官方源码 `third_party/UniDexGrasp2/dexgrasp/tasks/shadow_hand_grasp.py`：

- `numObservations=300`，`numActions=24`。观测包含 q/dq/DOF force、五指尖刚体状态与
  force/torque sensor、hand/root pose、上一拍动作、物体状态/目标偏移和64维形状特征。
- ++ 的29维 grasp-goal 槽清零；它是 non-goal-conditioned grasp-and-lift，不是原始
  UniDexGrasp 的指定 grasp-pose 条件策略，更不是 DExplore 指定参考轨迹跟踪器。
- `pre_physics_step` 默认分支把 action[6:24] 映射到18个受控关节的 limits，再作为 position
  target；action[:3]、action[3:6] 经 z-canonicalization 的逆变换后，分别变成根部施加的
  **force/torque**，不是根部绝对 position/orientation target。

[官方 state task 源码](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/tasks/shadow_hand_grasp.py)

默认 `useRelativeControl=false`、`actionsMovingAverage=1`、`controlFrequencyInv=1`；
配置写60Hz，episodeLength200，goal_cond=false，random_time=true。我们的完整回放为
不从随机中段开始而显式设 random_time=false，并固定 simulator dt=1/60，是真正60Hz
执行。MJCF 自己的 `option timestep=0.002` 不能拿来替代 IsaacGym 运行时 sim.dt。
[官方任务配置](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/cfg/shadow_hand_grasp.yaml)、
[本地采集器](../../src/task/video-data-preparation/tools/run/replay_unidexgrasp.py)

手型为 ShadowHand：实际 robot.xml 含22个活动 hinge joints（包括4个由 tendon 耦合的
远端关节），shared.xml 只有18个 position actuator，外加浮动根部6维外力/力矩控制。
原独立 wrist body/joints 在这个发布模型中注释掉，根 landmark 是 palm，不能伪造 wrist。
[robot.xml](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/assets/mjcf/open_ai_assets/hand/robot.xml)、
[shared.xml](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/assets/mjcf/open_ai_assets/hand/shared.xml)

与本仓当前 actor 合同对照：

| 项目 | UniDexGrasp++ ShadowHand | 当前 DExplore/Inspire actor |
| --- | --- | --- |
| 手指测量状态 | 22 q + 22 dq | 保存q18/dq18；根6维+6个主动手指维，另有耦合关节 |
| 动作 | 6维根外力/力矩 + 18维关节 position targets | 根 xyz/rotation + 6指的 q_PD；8拍编码为8×15 |
| 语义 | 通用抓取举升 | 跟踪参考执行得到的反馈补偿 PD target |
| 是否可直接共享 label | 不可：手型、维度、控制物理量均不同 | 现有动作头只接该合同 |

当前本地 actor 读取 `pd_targets[t:t+8]` 并从几何源读 Inspire q_ref，不能把 Shadow 的
`target` 数组改名就满足合同。
[`robot_data.py`](../../src/task/ptv3-shared-wam-v2/src/ptv3_shared_wam_v2/robot_data.py)

## 4. 本地采集确实保留了哪些信息

现有采集器的 `snapshot()` 和 episode 保存路径保留 hand11、object pose、q/dq、hand_root13、
policy observation300、goal、success/reset，另保存 actor action、实际 applied action、根部
force/torque、关节 target 和真实时间。静态点云特征64维并不等于完整 CAD，但已有独立
geometry 产物可供我们 WAM provider 使用。这是本仓采集 schema，不是作者发布的示范格式。
[`replay_unidexgrasp.py`](../../src/task/video-data-preparation/tools/run/replay_unidexgrasp.py)

紧凑轨迹手物状态已是30Hz（原60Hz每隔一帧），动作仍可保留60Hz时间；要切 actor chunk
必须按 `time/action_time` 明确对齐。每一个30Hz区间的两个根力作用改变整个物理过程，
不能只取每隔一个 action 就宣称是等价30Hz控制。WAM几何序列可沿现有经过时钟核验的读取器。
[`audit_unidexgrasp_replay.py`](../../src/task/video-data-preparation/tools/audit/audit_unidexgrasp_replay.py)、
[时钟记录](../../src/task/3d-data-preparation/docs/research/2026-10-15-native-clock-cleaning-evidence.md)

本地全量计数与动作键逐层核验见 [本地 policy 数据审计](local-policy-audit.md)：
root 本轮从83个完成回放根复算2066mesh、33056 trajectories，原60Hz状态6611200帧、
30Hz compact3305600帧。原始文件有 observation/target，compact没有这两项；后者仍有
199拍60Hz动作及 action_time。因此可以从现有raw重导Shadow控制数据，无需先重新回放。
official_success14671、末帧距目标5cm内12490，均不直接证明稳定抓持；actor BC不能把
所有成功和失败执行不加区分当成同质量 expert demonstration。

## 5. 能改变近期决策的建议

1. 若目的是检验“660条太少”：固定 shared/actor 接口后，在 **同一 Inspire 手型与 q_PD
   合同** 上做数据规模曲线（同对象/轨迹划分）。ShadowHand混入会同时改变手型、控制和任务，
   因此不能单独识别数量原因。
2. 若目的是利用已采的大量 ShadowHand 数据：先把它作为 WAM shared 的手物 forecast 辅助源，
   DExplore 继续提供 Inspire action loss。root 本轮核验当前v2父80k的main3不含source4，
   远端v3已包含它作为forecast来源；精确recipe见[本地审计](local-policy-audit.md)。
   不要把“远端v3已使用”误说成当前三臂的v2父权重已经使用。
3. 若希望 Shadow action 也更新 shared：单独22关节状态/24动作 Shadow 头，并在原仿真接口
   做 action replay/闭环资格检查；保留 Inspire 头。这个实验测试跨手共享表示，不能承诺提升
   Inspire抓取；不应现在修改正在运行的匹配三臂。

这些是面向本仓的推断和最小可判别设计，不是上述论文已经证明的结论。本轮仅调研，不启动训练。

## 6. 发布和许可边界

++ README 声明 work/dataset 为 CC BY-NC4.0，而同仓 LICENSE 文件是 MIT（软件）。
应同时保留这两个事实，不把 GitHub 自动显示 MIT 等同数据无限制商用；物体来源及
NVIDIA/IsaacGym相关组件仍保留各自许可文件。本仓科研用途与对外商用是不同问题。
[原 UniDexGrasp 许可](https://github.com/PKU-EPIC/UniDexGrasp#license)、
[++ README 许可](https://github.com/PKU-EPIC/UniDexGrasp2#license)、
[++ LICENSE](https://github.com/PKU-EPIC/UniDexGrasp2/blob/main/LICENSE)
