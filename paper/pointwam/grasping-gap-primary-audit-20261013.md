# PointWAM 与当前抓取失败的差距：一手来源复核

日期：2026-10-13。只读核对论文、作者项目页、当前源码和实验卡；本轮没有训练、仿真或修改既有代码。工作树基准为 `ea2575aa`，存在用户/其他会话未提交修改；下列本地行号指本次查看的工作树，不声称所有内容均已提交。本文是诊断分析，不把 Probe 升级为正式结论。

## 一手来源能确认什么

以下压缩表保留论文的关键条件。它们描述作者的研究设置，并非本任务保证。

| Contract | Verified paper facts |
| --- | --- |
| Input/output | Current colored scene, semantic hands, language; deterministic scene/hand displacement forecasts; no object-goal input. |
| Human data | 1,148,324 EgoDex/VITRA episodes; CoTracker3 plus hand-scaled VGGT-Omega; missing hands masked. |
| Representation | Ten points/hand; shared spatial backbone; frozen Mosaic3D and Recap-CLIP. |
| Training | 651M trainable pretraining parameters; 655M robot stage; 60k updates/stage, batch192, 4 B200, 34h/19h. |
| Robot supervision | 959 replay-successful DexJoCo demonstrations; joint backbone, trajectory heads, retargeter updates; scene/hand MSE plus recorded-action L1. RoboDojo:100 demonstrations/task, eight tasks. |
| Embodiment | Franka/16-DoF Allegro; RoboDojo dual ARX-X5 grippers; real OpenArm/6-DoF Inspire. Real demonstrations: banana54, insertion64. |
| Evaluation | DexJoCo: 50 episodes/task/seed, three seeds;50Hz, async30-action queue, refill below24. RoboDojo:25Hz, synchronous whole-chunk. Real:24 trials/task, demonstration positions. |
| Scaling evidence | Fixed Volt-H/fine-tuning: scratch12.1%, EgoDex60.1%, full69.0%. Volt-S39M reaches52.3%. |
| Success scope | Banana: entirely inside basket. Insertion: peg remains inserted. RoboDojo-Precision average4.8%. |

来源：[论文方法§3](https://arxiv.org/html/2610.02840v1#S3)、[设置§4.1](https://arxiv.org/html/2610.02840v1#S4.SS1)、[缩放§4.3](https://arxiv.org/html/2610.02840v1#S4.SS3)、[处理B.1–B.2](https://arxiv.org/html/2610.02840v1#A2.SS1)、[训练/闭环C](https://arxiv.org/html/2610.02840v1#A3)。

[作者项目页](https://chrockey.github.io/PointWAM/)另可直接核对：参考消融网络 Volt-L12 的无 world supervision 为53.5%，3D scene supervision 为64.4%；10个语义手点为64.4%，10/512个手表面点为43.6/44.6%。这是作者设置内支持相关设计的干预证据。作者网页当前仍显示 **Code soon**，没有作者链接的训练仓库/权重可用于逐行确认；本项目的 [pointwam-repro README:3](../../src/task/pointwam-repro/README.md#L3) 也明确它是自行实现。不得把同名第三方仓库或本地复现视为已审计的官方实现。

未证实：作者完整数据清理、实际人类伪标签误差分布、所有接触标签精度、未公开训练代码的梯度/参数组细节，以及在本仓库本体、控制器、物体与 hold45 定义上的成功率。不能因为论文结果好，就推断其人类数据是无噪声真值。

## 当前本地实现应如何比较

1. **我们的 World 约束和作者的 scene 辅助预测不是同一个训练变量。** 当前 `FrozenEffectWAM` 将 scene/goal/观测手上下文置于 `no_grad`，只训练 trajectory；预测手轨迹经过 action encoder 再进入冻结、goal-blind 的 World。可检查 [effect_consistency.py:25](../../src/task/goal-point-wam/src/goal_point_wam/effect_consistency.py#L25) 和 [47](../../src/task/goal-point-wam/src/goal_point_wam/effect_consistency.py#L47)。因此，降低该 World 对预测手轨迹的 effect error，可能只说明轨迹满足代理模型；是否产生真实接触、闭合与持续持物必须另外测。论文的 world 消融无法提供本地 effect loss 的因果收益证据。

2. **不能再用旧版本缺少手坐标/只有终点描述当前 actor。** 现有 ref2_8 适配器实际加入48D q/dq/previous command，以及27D当前物体相对的三个参考姿态，广播到保留的 raw-hand query：[control_context.py:8](../../src/task/goal-point-wam/src/goal_point_wam/control_context.py#L8)、[29](../../src/task/goal-point-wam/src/goal_point_wam/control_context.py#L29)。仍可讨论接触状态信息是否足够，但“完全不看机器人状态”已被源码否定。原始 PTv3 手点采样稀疏的旧观察也不能代替新接口的消融结果。

3. **数据量应比较独立覆盖，不能拿窗口数与 episode 数直接相除。** main-three 规模卡已有4,378,478/715,689 train/val windows，见 [DDP卡:105](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ddp-scale.md#L105)；当前已不是六窗口拟合。另一方面，独立视频路线最新记录是303 train windows/18 sessions，新增下载/重建数据未进入这次拟合，且该训练没有 robot control supervision，见 [视频卡:1475](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-video-points.md#L1475) 和 [1523](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-video-points.md#L1523)。不能把下载容量、重复更新次数或相邻重叠窗口数量当作 actor 已获得广泛接触恢复经验的证据。

4. **“联合训练 R 就会好”已经受到本地新证据约束。** 当前新增 joint-anchor trainer 明确有 trajectory MSE、预测轨迹经R的动作损失、GT轨迹到R的动作锚定，没有 World loss：[train_visited_joint_anchor.py:238](../../src/task/goal-point-wam/tools/run/train_visited_joint_anchor.py#L238)。新卡记录3000步所有检查点未通过门；exact TRAIN predicted PD mean 1.335→2.489mm、exact DEV GT PD p95 227.806→288.004mm，未进入闭环：[joint卡:58](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-visited-joint-anchor.md#L58)。这不否定所有联合训练；它说明本地这个初始化、数据、冻结范围与目标组合没有解决问题，也说明只改成 action loss 联合反传不是已验证答案。

5. **本地控制合同与训练状态覆盖是必须独立通过的链路。** 当前R接收24步11点轨迹，输出8步12D动作，仅执行第一步后重规划：[chunk_retargeter.py:1](../../src/task/trajectory-policy/src/trajectory_policy/chunk_retargeter.py#L1)、[18](../../src/task/trajectory-policy/src/trajectory_policy/chunk_retargeter.py#L18)。新 joint 卡中 GT hand trajectory 仍可对应很大的 held-state action error，表明即使给 actor 完美手轨迹也不能保证该R在这些状态产生正确命令。这里的PD误差是动作空间诊断，并非真实手指接触误差，不能混称为同一指标。

6. **目标时间与实际进度错位合理，但已完成的进度目标Probe没有隔离其因果效果。** 当前工作树的 [progress_goal_indices:12](../../src/task/trajectory-policy/src/trajectory_policy/robot_context.py#L12) 用实际物体位置约束参考游标。最新 [progress卡:46](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-progress-reference.md#L46) 已跑完542步：actor hold0、clipping41%；teacher/GT-tau分别hold437/438。目标索引到tick50才改变，但与clocked realign的动作在tick6已分叉，因此不能将后续抛飞归因于进度目标，也不能据此证明目标相位无关。物体在接近阶段几乎不动，仅靠物体平移也不能区分“尚未接触”和“已闭合准备抬起”。

## 证据支持到哪一步

本轮另直接读取保存的 `trajectory.npz`，按原始高度、手物间隙和桌面支撑条件重算连续 held：C-final 的 teacher/GT-R/actor 为434/441/0帧；progress-reference为437/438/0帧。后者 actor 最高上升679.94mm，但没有任何合格held帧，不能把碰飞当抓起。这是已有轨迹的CPU数组复核，没有重新执行模型或仿真。

最新 realign 的实际 `outputs/goal-point-wam/wam-visited-realign-20261013-r1/result.json` 已是 COMPLETED、selected_update=3000、offline_gate=true，原卡顶部 RUNNING 尚未同步。精确TRAIN的first8手点EPE23.22mm、首PD位置目标误差均值1.34mm/p952.63mm；精确DEV分别为35.20mm、57.61mm/171.10mm。PD目标误差不是实际手点误差。相对旧基线的改善门不代表绝对接触控制精度已经足够；对应门实现见 `src/task/goal-point-wam/tools/run/train_visited_realign.py:125`。

独立只读审查还发现：保存的 nominal tick0/env24 与 live tick0 的q/dq/hand/obj/velocity相同，而离线batch16面板与live batch1的actor轨迹最大分量差约3.66mm。尚未逐项重建并对齐所有模型输入，故这只是数值一致性线索，不是已经确认的batch或BF16缺陷。它支持先做固定全部输入的batch/精度诊断；不能据此取代物理失败的因果分析。

- 作者的预训练和相关组件消融支持那些因素在其设置内有用；它们没有隔离出本地失败的唯一原因。
- 本地 GT轨迹到动作的held-state大误差、joint-anchor未通过门，是优先处理控制映射/分布覆盖的直接证据。低离线轨迹或World loss不能覆盖这一缺口。
- 模型容量、语言/目标接口、监督形式、闭环执行方式都有差异，但仅凭差异不能归因。尤其小模型在作者实验中已有非零且可观成功，因此不能将“加大网络”当成当前抓不起的必要解法。
- 原始人类数据本身跨本体，human geometry learning和本地机器人接触控制需要分别验收。当前失败也不能证明“人类轨迹预训练没有用”。

## 建议的下一步：每项只解决一个决策

**先测可执行性上限。** 用已保存的相同接触前后状态，分别比较 expert command、GT hand→R command、predicted hand→R command。先按接近、闭合、抬升、保持四阶段报告实际动作误差、clipping与持物结果；已做的资格证据应复用。GT→R仍明显失败时，继续扩大actor预训练无法消除这条下游误差链。这是对当前证据的优先级判断，不需要新开大训练。

**先解释现有进度比较中干预前的分叉。** 复用已保存状态，在同设备、同执行设置下比较完全相同输入的trajectory、R输出和实际命令；先查tick6的差异来源，再考虑同一状态上只改参考索引的最小比较。已有542步物理执行无需重复当作未做实验。未来进度条件需结合手与接触状态，单纯目标追得更近或World预测更好不能作为成功标准。

**把数据收集对准失败状态。** 用同一真实执行行上的专家恢复，增加不同闭合程度、手–物相对偏差、物体滑动与接触后的独立前缀；整条前缀留作dev，避免同一恢复的相邻窗口同时承担训练和泛化证据。此建议针对本地控制缺口；不是宣称作者用过同一种恢复策略。

**暂不以当前失败为由复制完整大模型预算。** 大规模场景/手共同预训练可以作为独立路线，但在本地R、参考时序与基本抓取通路未打通前，很难判定昂贵预训练为何有用/无用。若继续比较 frozen World effect，应保持actor、R、数据、初始化、采样和执行一致；现有 fixed-R effect-on 与 joint-R effect-off 同时改了两个因素，不能据此宣布effect loss是主因。

以上是诊断与最小判别实验建议，本轮没有启动它们。当前最稳妥的结论是：PointWAM提供了有证据支持的完整训练路线；我们的实现是不同的目标条件与控制系统，而当前直接证据首先指向接触状态上的轨迹→动作/闭环覆盖缺口。无法仅凭论文和现有Probe证明某单项改动就能抓起。
