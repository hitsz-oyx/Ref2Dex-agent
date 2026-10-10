# ref7_3: reference tracking control 与现有 R 的比较

日期：2026-10-10。范围：核实 ref7_3 的论文、作者开源与迁移边界；本调研阶段未训练、未仿真、未安装依赖。本文是方法调研与下一步建议，不是本仓库控制收益 Validation。用户随后授权的实现与训练结果见 [reference-tracking Probe](../experiments/probes/P-20261010-reference-tracking.md)。

## 判断与需要纠正的前提

**方法层面，reference-conditioned closed-loop tracking 比继续单独压低 full-action inverse regression 的 L1 更值得最小 Probe。** 这是基于公开方法和本 Task 未通过执行 gate 的研究判断，尚未证明在 Inspire 上优于现有 R。现有 ref7_2 已要求 current state，且已经试过每步重新查询；不能把对比简化成“开环 τ-only vs 闭环”。真正变化是：训练目标由复现某条 action label 改为在模拟器中获得跟踪与物体行为回报，并允许在 reference 周围调整 commanded targets。

源码还揭示一项直接相关的差异：`anchored_future_batch` 把 source 的 future displacement 平移到 live current hand 上，因而丢掉 source-current 与 live-current 的绝对偏差；若用于绝对 reference tracking，actor 需要额外读 reference-current error。这需要作为实现假设独立核实，不能将原执行失败直接归因于纯几何不够。[当前 anchoring 源码](../../tools/run/run_hand_action_retargeter.py)

当前 best checkpoint 约 2.4 MiB、约 0.6M 参数，ref7_3 的“giant decoder”不是主要问题；重点是信息合同和实际闭环目标。

本地证据：[ref7_2 原始要求](../user/ref/ref7_2.md)、[执行及归因卡](../experiments/probes/P-20261010-hand-action-retarget-data.md)。低离线 MAE 与未通过 held gate 并存，只能说当前方法未通过，不足以证明 τ 不可用或 preload 不可恢复。

## DexTrack

**已核实：** [论文 §3.1/附录 A.2](https://arxiv.org/html/2502.09614v1) 每步根据当前 hand/object state、reference 和误差输出 PD positional targets；奖励同时约束 object position/orientation、wrist/fingers 与 hand-object affinity。原 action space 为 reference joint targets 加累积 residual；relative targets 也提供。它不是仅对未来 hand geometry 做 action label regression。

**公开范围：** [作者 README](https://github.com/Meowuu7/DexTrack/blob/1005fad6158c3fa0e4e8df01323de2f62d539f7a/README.md) 公开 tracking environments、single/multiple-trajectory train/eval，但明确没有公开 homotopy optimization 和 IL/RL generalist 迭代的两个关键组件。支持 floating Allegro 与 LEAP+Franka，Isaac Gym Preview 4；新 hand 的 keypoint 对应须手动定义。因此不能把论文完整 flywheel 当作现成复用代码。

**源码核验：** [control 路径](https://github.com/Meowuu7/DexTrack/blob/1005fad6158c3fa0e4e8df01323de2f62d539f7a/isaacgymenvs/tasks/allegro_hand_tracking_generalist.py#L12032) 在 `use_kinematics_bias_wdelta` 分支读取下一帧 reference，累积 `speed_scale * dt * action`，叠加在 nominal target 上，并在 joint limits 内裁剪，最后 dispatch positional targets（L12303–12324、L12400–12406）。[single tracking script](https://github.com/Meowuu7/DexTrack/blob/1005fad6158c3fa0e4e8df01323de2f62d539f7a/isaacgymenvs/scripts/run_tracking_headless_grab_single.sh#L475) 默认 0.0166s、22000 env；这些是作者设置，不是本项目的运行预算。脚本存在多次覆盖参数，迁移时应提取最终配置而非直接执行。

固定 commit 的六份只读源码（总计 979415 bytes）与许可证保存在 [`third_party/DexTrack-source-audit`](../../../../../third_party/DexTrack-source-audit/)，逐文件 URL/SHA256 在 `SOURCE_MANIFEST.json`；未安装或执行。论文索引见 [`paper/dextrack/schema.json`](../../../../../paper/dextrack/schema.json)。

**推断：** DexTrack 是当前 Gym 工程最接近的参考，但需将 22-DOF hand 改为本项目 18-DOF/coupled Inspire、60Hz 改为既有控制频率。残差宜定义在明确单位的 PD-target 层，再经 native adapter 转换 wrist translation/rotation 与 finger targets；不能把通用 joint-target residual 直接当作我们混合单位的 18-D normalized action。保留低层 tracking 自由度，而非硬固定 wrist：已有 fixed-wrist/live-teacher-finger audit 仍 held 8/10/9，尚未隔离 wrist/source-state 的作用。

## REGRIND：方法可借，不能直接迁入 Gym

**已核实：** arXiv:2607.11874 是 *A Minimalist Retargeting-Guided Reinforcement Learning Recipe for Dexterous Manipulation*，2026-07-13；作者代码 [yunhaif/regrind](https://github.com/yunhaif/regrind) 已提供 retargeting、数据及 RL 入口，依赖 IsaacSim 5.1.0 / IsaacLab 2.3.0。默认环境是 LEAP/WUJI 的 scissors/screwdriver，不是 Inspire。

论文采用机器人 configuration reference 加 residual，再交低层 PD；object keypoint tracking 为主奖励，还包括 object velocity、wrist tracking 和动作正则。actor 使用物体配置、当前/过去机器人位置、上一动作及 phase；critic 额外读 fingertip 与 velocity。它没有要求 actor 读取真实接触力，也不能据此说网络显式“学到了 force”。[论文 §3.3/附录 B](https://arxiv.org/html/2607.11874v1)

**源码补充：** actor 配置还显式包含 reference action base 的 wrist pose 和 finger target；motion command 是 phase。finger 默认以 motion target 为 base，加 bounded residual，root 则用 reference pose 加 SE(3) residual 后施加 PD wrench。因此并非直接从任意未来 11-point τ window 预测完整 24-step actions；把我们的 τ 解成可行 wrist/finger reference、补充 τ-window conditioning、适配 Gym native actions 都需要实现与验证。[观测/奖励配置](https://github.com/yunhaif/regrind/blob/38347a9e30184620df04e19c63c7c72378cae103/source/regrind/regrind/tasks/manager_based/dexterous/dexterous_env_cfg.py)、[动作实现](https://github.com/yunhaif/regrind/blob/38347a9e30184620df04e19c63c7c72378cae103/source/regrind/regrind/tasks/manager_based/dexterous/mdp/actions.py)、[phase command](https://github.com/yunhaif/regrind/blob/38347a9e30184620df04e19c63c7c72378cae103/source/regrind/regrind/tasks/manager_based/dexterous/mdp/commands.py)

**推断：** 更适合借 reference/residual、object reward、RSI 与扰动训练，而非换模拟器来跑作者整套工程。对象参考是训练条件的重要部分，不能以只有手 τ 的部署接口冒充同等信息。

## Contact-Anchored Retargeting：论文存在，训练源码尚未发布

**已核实：** arXiv:2609.24093 对应 *Dexterous Robot Manipulation from Human Demonstrations via Contact-Anchored Retargeting and Residual Policy Learning*，2026-09-21。三阶段为 MANO physics refinement、contact-anchored retargeting、robot residual learning；论文使用 Isaac Gym PPO。robot residual 调 wrist translation/finger controls，wrist orientation 保持 reference；correction 有累积、衰减和 clipping，最后接触后 finger correction 不再衰减。条件含 task ID、object geometry 和 reference motion；部署读当前 joints/object pose。[论文 §3](https://arxiv.org/html/2609.24093v1)

**代码边界：** 作者 [DexGEM-Lab/real2sim2real](https://github.com/DexGEM-Lab/real2sim2real) 当前只有 assets、LICENSE、README，明确写“Code release coming soon.” 因此只能作为方法来源，不能承诺直接复用 actor/reward/retargeter。论文报告 morphology profile/coupling 适配；本次没有源码可核实 Inspire 的实现或完整 privileged observation schema。

**推断：** 保留 accumulated target correction 对稳定 finger preload 有启发，但可能漂移，须保持 native action bounds，并独立记录 reference error、target 与实际 q；不应仅凭几何误差就假定握持力恢复。

## DexMV：不是通用 τ tracking policy

**已核实：** ECCV 2022 / [arXiv:2108.05877](https://arxiv.org/abs/2108.05877)，作者 [dexmv-sim](https://github.com/yzqin/dexmv-sim) 与 [dexmv-learn](https://github.com/yzqin/dexmv-learn) 已发布。它将 human hand/object poses 翻译成 robot demonstrations，再用 DAPG 等 imitation/RL；依赖 MuJoCo 2.0。示范生成包括坐标对齐、时间插值、joint q/dq/ddq inverse dynamics 得 torque，以及 hindsight goal。[生成文档](https://github.com/yzqin/dexmv-sim/blob/d857c16e4d6888a60560b12f5eca20c36e4073f6/docs/demo_gen.md)、[项目](https://yzqin.github.io/dexmv/)

**推断：** 它支持“retarget geometry 不等于可执行 action”的区分，不能用来直接论证我们的 τ-only general controller 已有现成实现。对 Inspire/Gym 的 action/observation/reward 合同仍需重新适配，本次未逐行审计 DexMV actor。

## AnyTeleop / dex-retargeting：可用的几何起点

作者关联 [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) 已开源；[Inspire 右手配置](https://github.com/dexsuite/dex-retargeting/blob/3f56141bc8bd2760d5e452e382937269554ebb21/src/dex_retargeting/configs/teleop/inspire_hand_right.yml) 是 vector retargeting，含六个 active finger joints、wrist-to-tip 向量与 joint limits/coupling 所属 URDF。对应 [AnyTeleop 论文](https://arxiv.org/abs/2307.04577) 是 teleoperation。[README](https://github.com/dexsuite/dex-retargeting) 要求按名称处理 simulator joint ordering。

**推断：** 可作为 nominal finger reference 的候选；须核对我们的 11 points 是否覆盖所需向量、URDF 耦合和 native actuator 数。已有 Inspire 配置不等于 pretrained manipulation controller，也不保证接触与动态可行。

## 下一步最低成本判别建议

先核对 source/backend/layout、native command 单位与对齐，并建立同协议下独立完整 episode 的 teacher 行为基准；已观察到 GPU PhysX 跨 row 即使 reset/action 相同也会分歧，不要求 bitwise clone 成为新的学习前置。训练与评价以每个环境自己的 live feedback 为准，不能把跨 row 复制 teacher 失败解释为任何学习算法无效。

若进入下一轮，可先用已有 physically executed teacher 的 robot-state/object reference 测试 tracker 控制上界，再将 nominal reference 替换为 11-point τ 的几何 retarget。这两步区分 control learning 与 geometry-to-joint reference 的误差，不能将 teacher future commands 提供给部署 actor。比较 nominal geometric targets 与相同 targets 加 bounded one-step reference-conditioned residual tracking；object GT reference 若进入 actor 要明确为 oracle 条件，若只作 reward 也应声明训练信息。部署时实际能获得的 task goal/reference 才能用于推理。未来手 τ 必须进入 actor conditioning，并用替换 reference/扰动恢复检查排除单纯 phase memorization。

最小 Probe 的问题是“同一参考下，基于当前反馈优化实际跟踪/保持是否修复现有监督 R 的执行缺口”。positive 才扩展多轨迹；negative 先检查 reference可达性、动作合同、奖励是否惩罚必要 preload、RSI 是否遮蔽 frame-0 acquisition。不能将 reference residual 等同于回到旧 baseline-policy residual：nominal 项应来自 τ 的几何参考，而非 teacher actor 未来动作。

本次只推荐这个最小 Decision Probe，未切换实现路线或启动训练。开始前应在 Task-local experiment card 冻结初始化、资源预算、reference/privileged-input 合同、保持与后续掉落指标以及停止条件。额外 RL 交互/计算不能与离线 R 的成本混为一谈；tracker 成功仍只解决 execution，不替代 Mission 中 matched Cm-on/off 的策略训练收益证据。

另一个方法边界：已有 [DExplore 论文 arXiv:2509.09671](https://arxiv.org/abs/2509.09671) 本身也是 kinematic-to-dynamic RL 控制路线。建议先借用本仓库已有 Inspire/native 基础，加入显式 τ conditioning；这只是工程成本判断，不把“已有相关第三方代码”当作本次运行或完成证据。

源码身份：REGRIND `38347a9e30184620df04e19c63c7c72378cae103`；DexGEM `97f89a265968904f2a7e5a9ba2c82e88ae9afc7a`；DexMV-sim `d857c16e4d6888a60560b12f5eca20c36e4073f6`；dex-retargeting `3f56141bc8bd2760d5e452e382937269554ebb21`。
