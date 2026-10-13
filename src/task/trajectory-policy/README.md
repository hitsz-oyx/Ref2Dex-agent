# Robot control context rollout（2026-10-13）

Add `--control-checkpoint <selected robot adapter>` to the C/Chunk launcher
below. The peer receives its own q/dq/previous actually applied command and
planned object+8/+16/+24 relative poses; GT future hands remain calibration-only.
The full542-control candidate reaches contact but held0, with1.476% clipping;
teacher/GT hold434/441. Thirteen execution/input checks pass. Native and panel
audits are `tools/audit/audit_wam_effect_rollout.py` and
`tools/audit/audit_wam_control_context.py`.
See [ref2_8 card39](../goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref28-control-context.md).

# C final frozen-policy physical rollout（2026-10-13）

`tools/run/launch_wam_native_bridge.py --effect-checkpoint <C patch> --chunk-dir <qualified fit> --head-dir <exact warm policy>`
loads the C decoder over the exact warm scene/goal/hand-query and uses qualified
native Chunk-R with only first action execution. Default MLP-R behavior remains.
The542-control development rollout is actor held0 vs teacher434/GT-Chunk441;
independent native command/input/outcome auditing passes. The actor diverges
near contact despite initial good control; offline gains do not establish grasp.
See [C rollout card38](../goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref27-rollout.md).

# Trajectory policy

## PTv3 explicit hand readout（2026-10-12）

`tools/run/probe_wam_hand_readout.py --output <fresh-owned-dir>`比较同一warm
WAM在旧query接口与完整packed语义hand-token残差下的各200更新；固定合格
control-aware Chunk-R，保留32-point memory goal，action/world/统计buffer冻结。
`tools/audit/audit_wam_hand_readout.py --fit <comparison-dir>`独立重建标签、
指标、采样和门槛。当前短适配10/21非退化门通过，暂不启用world一致性或
actor物理比较；新R本身的GT资格保留。见
[手读出卡35](../goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref25-hand-readout.md)。

## Goal-Point-WAM ref2_5 control-aware Chunk-R（2026-10-12）

`tools/run/fit_chunk_retargeter.py --control-aware --output <fresh-owned-dir>`
保留原模型和采样，以4×首步native L1、后7步L1、首步物理PD误差训练；
按开发集首步PD XYZ p95选checkpoint，all8 L1仅作第二指标。PD尺度固定
XYZ10mm、腕角/手指各.1rad；训练用raw输出保留越界纠正梯度，执行仍裁剪。
新GT比较teacher/旧R/新R为73/75/74held，全零裁剪，局部GT门通过。
训练、物理与独立审计入口沿用下方工具；无flag时保持原all8目标。
见[首步控制目标卡34](../goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref25-control-aware-R.md)。

## Goal-Point-WAM ref2_4 Chunk-R（2026-10-12）

`tools/run/fit_chunk_retargeter.py --output <fresh-owned-dir>`训练小型
trajectory-conditioned Transformer，输出8×12实际native action；
`tools/run/execute_chunk_retargeter.py --fit <fit-dir> --output <fresh-owned-dir>`
在固定native runtime/GPU2比较teacher、GT旧PD-R、GT新Chunk-R。每步重规划，
只执行首步。`tools/audit/audit_chunk_retargeter.py --fit <fit-dir> --execution <trace-dir>`
独立核对动作标签、实测状态、GT坐标与native执行。

固定3000更新后的GT比较为73/75/0连续held、全部零裁剪，新Chunk-R第一关失败，
本轮按ref2_4暂停后续WAM hand anchoring/world consistency。旧PD-R在此入口
使用同查询batch2以复现其合格native推理；对应实际运行时回归为
`tools/audit/check_native_PD_batch.py --safe`。首轮无效对照记录保留。
结果仅适用于当前固定配方，详见
[Chunk-R卡33](../goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref24-chunk-R.md)。

## Goal-Point-WAM ref2_3 multi-takeover fit（2026-10-12）

`tools/run/fit_wam_multitakeover_anchored.py --output <fresh-owned-dir>`
从Probe30的GT-anchored WAM/R继续，以Probe31的259个实际专家future和
384个当前PD-only样本训练，保持nominal/startup回放和GT-only R梯度职责。
`tools/audit/audit_wam_multitakeover_fit.py --fit <fit-dir>`核对各实际源的
标签与报告指标。保留门通过后只运行一次新闭环，比较t8/12/16/24/32/48。
协议见[多接管训练卡32](../goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref23-multitakeover-fit.md)。

## Goal-Point-WAM ref2_2 continuation（2026-10-12）

当前robot执行线先做GT-tau R窗口诊断，再拆分WAM/R梯度职责，暂不扩大
WAM结构或启动RL。`tools/run/fit_wam_r_horizons.py`同初始化/同update stream
比较K1/4/8/16/24；省略的未来在标准化后屏蔽，旧checkpoint默认K24。
开发诊断未达到预设短窗口收益，暂保留24。

`tools/run/fit_wam_joint_retargeter.py --gt-anchored --output <fresh-owned-dir>`
让预测动作损失通过parameter-frozen R更新WAM，R只由真实expert tau训练。
纯student当前专家PD查询不更新R，也没有几何未来标签。匹配checkpoint
仍需要离线保留screen，并在新物理回放重新建立GT控制资格。
`tools/audit/compare_wam_short_rollouts.py`对已审计实际回放报告固定ticks的
PD/腕部偏离。`prepare_wam_multitakeover.py`聚合相同learned pair在8/16/32
实际接管的专家未来；不能用nominal或另一row的future替代student接管状态。
协议与研究判断见goal-point-wam的
[窗口卡29](../goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref22-r-horizon.md)
和[GT锚定卡30](../goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref22-gt-anchored.md)。

## 官方教师跨参考筛查（2026-10-11）

用户暂缓 WAM/PW，先看当前仓库官方 actor 的实际表现。
[corrected12 筛查](docs/experiments/probes/P-20261011-official-reference-screen.md)
用同一官方 Inspire checkpoint、seed191、每条8个frame0完整回放：11条均8/8持物，
手机仅2/8；受控放置合计68/96。s3/s9飞机、牙膏、水瓶均8/8放置，物体位置
RMSE约5–7.5cm，是下一轮教师采集候选；Retake飞机/鸭子放置1/8和2/8，
手机位置RMSE1.56m。抓持、放置与精准参考跟踪分别报告，整体质量UNCLEAR。
所有实测轨迹/PD/参考合同审计通过，评估包training_allowed=false；未启动WAM训练。
这是固定初始状态Probe，不能作为随机初始状态泛化或最终RL收益结论。

## ref3 当前进展（2026-10-11）

当前优先按 [ref3](docs/user/ref/ref3.md) 推进第一阶段：
`(H,G) -> 24步真实手轨迹 -> learned R(当前实测状态,轨迹) -> 实际PD目标`。
官方 actor 用于 teacher 数据和训练期 PD 查询；自主评估关闭教师干预。
用户已授权 G 使用 DExplore 式 t+1/t+16 人手和物体计划参考位姿，当前 H328/G408。
早先物体终目标 G12 是 agent 的限制假设，现已纠正。历史 H 仍来自真实测量，
未来实测 q/物体和接触标签不作为输入；冻结 R 的 PD loss 可更新高层 actor。

- [真实PD监督](docs/experiments/probes/P-20261011-competent-pd-retargeter.md)：32条新合格teacher，离线门槛失败仍 UNCLEAR。
- [R完整物理执行](docs/experiments/probes/P-20261011-competent-pd-physical.md)：teacher/真实PD/GT轨迹-R 各8/8持物和受控放置，独立审计通过，局部 PROMISING。
- [自主联合BC](docs/experiments/probes/P-20261011-competent-joint-bc.md)：离线启动误差0.922/0.693mm，梯度与预测审计通过；8步查询完整物理比较teacher/GT-R各4/4，自主两臂各0/8，执行链审计通过，局部 UNPROMISING。
- [每步查询冻结对照](docs/experiments/probes/P-20261011-competent-actor-cadence.md)：保持同模型/seed/判据完成并通过独立审计。联合监督4/8hold（311/309/427/427帧），纯轨迹0/8；完整任务仍0/8，整体 UNPROMISING。静态持物与后段失物都不能算完整参考任务成功。

- [参考条件与一轮PD DAgger](docs/experiments/probes/P-20261011-reference-goal-dagger.md)：G接入计划参考，并完成一轮真实学生状态PD查询。无教师干预评估中，PD DAgger8/8持物和放置、零裁剪、参考位置RMSE7.57cm；offline joint续训7/8hold、6/8放置、中位held92帧，对比DAgger376帧。完整独立审计通过，局部 PROMISING；先做冻结matched重复/Validation，尚未证明多motion泛化。

- [冻结模型双seed Validation](docs/experiments/validations/VAL-20261011-reference-goal-dagger-repeat.md)：两个预定holdout启动均8/8持物和放置、零裁剪，参考位置RMSE7.58/7.52cm，六项gate均通过，局部 SUPPORTED。离线续训对照放置5/8、6/8。两次真实初始状态exact相同，只支持当前frame0重复性；不证明扰动或多motion泛化。保留固定baseline，下一步Cm/PointWorld接入设计。

这些是同一 airplane reference 的 Probe及固定模型重复验证，尚未证明 G 泛化、精准目标位姿、完整参考跟踪
或 Cm 帮助 RL；先获得自主物理 baseline，再进入 PointWorld 动作条件 critic。
下面保留旧ref1/ref2实现与负结果，避免重复尝试。

用户 [ref1](docs/user/ref/ref1.md) 指定的新架构研究边界：纯测量历史H→独立高层
trajectory actor→latent c→24步11点手轨迹→每步读取实时状态的executor→native command。
旧behavior只提供初始化和可衰减训练先验，不作为永久在线动作基座；后续检验
PointWorld动作条件物理表征是否改善真实RL策略训练。

当前阶段：固定48维轨迹decoder首轮覆盖Probe已完成，局部UNPROMISING。
密集几何轨迹4/4长时终末抓持；四节点重建两组均0/4终末持有，输入/FF/命令审计通过。
最优四节点拟合仍有最差窗口32mm位置/31mm FF RMS，证实这种时间形状难以近似
当前可执行前缀。固定PCA D48把手点RMS降到5.61mm，但启动掌部max23.62mm/
FF RMS18.50mm未过几何screen，按协议未仿真；不作为新的抓持负证据。
最终pose/FF metric D48离线手点RMS0.919mm、掌部max4.386mm、FF RMS2.805mm通过，
但原生两组长时终末held为2/4、0/4；GT与dense各4/4。8行都先抓住，6行随后失持。
实际live q/object重编码下前缀手点RMS8.017mm；输入/命令/PD与独立实现审查通过。
按预声明停止48D重建搜索，改用保留24步独立腕姿/手指的直接D288，先核对既有dense
执行轨迹的接口等价，再进入H→c初始化及真实奖励RL。
`457c65f` D288工程回放通过68窗口x4条dense计划：native q max5.96e-7、
手点3.58e-7m、速度1.97e-5、腕FF2.35e-6；GPU4约0.975s、Torch峰值6.85MiB。
这是同一既有成功计划的编码/解码一致性，未新增物理rollout或学习证据。
已补充H328纯测量历史合同、独立Gaussian actor与有界监督初始化入口
`tools/run/fit_history_actor.py`：全部源轨迹保留，train-only normalization，按行留出；
首轮6504样本BC初始化2500updates完成14s；留出目标比常量低80%，但实际纯H两组
均0/4稳定抓持（GT4/4、dense3/4），启动手点RMS56.36mm。实际H/c/D/native链审计
通过，局部UNPROMISING；当时尚未PPO训练或接入WM。后续startup-balanced2500update/13s把启动手点RMS降到7.395mm，但掌部max15.61mm，
仍未过5/10mm初始化screen，按协议未仿真。停止BC权重搜索，下一步固定D/R做真实奖励
trajectory PPO；balanced500仅作为未验证warm start，不作为成功策略。
首轮真实任务PPO `4da53a9` 已完成24updates/48992交互/264s，训练采样曾有
486帧且终末持有，但最终冻结比较GT/dense各3/4，warm/PPO均0/4形成抓持。
全H、概率/价值/奖励/GAE、随机c物理解码及实际执行链审计通过，局部UNPROMISING；
尚未形成可用轨迹baseline或WM收益。已有rollout显示短credit导致长held样本启动
优势全负；固定快照lambda1回放把实际启动样本概率提升从1/7变5/7，KL受控。
lambda1固定预算Probe已完成265s/48992交互，9/12长held探索行启动Apositive，
但冻结warm/final仍各0/4，GT/dense各3/4；全链审计通过，局部UNPROMISING。
固定同H接触前XYZ mean改动<=.03893mm（per-coordinate RMS），探索std1mm。
`959cd74` 冻结随机64env比较85.72s完成并通过全链审计：GT4/4、dense3/4，
warm/final各2/28长时终末held，成功均同pair8/20，clip .9818%/1.0741%。
按固定screen局部UNCLEAR，仍没有可用baseline或稳定成功增量。
`eb3e96d` 真实Adam重建通过全部25个actor/value checkpoint bitwise核对。
单update恢复LR在19/24批次有>=2倍启动位移且更好surrogate/同KL上限，机制PROMISING；
位移中位仅.00591mm，尚未新训练或取得物理收益。用户随后给出ref1_1：已核对
PointWAM原文并建议先小型actual未来手轨迹→真实native action的learned retargeter
Probe，保留LR恢复为备选。CPU库存审计两轮PPO共有93,376合法窗，但均seed293、
完整稳定held仅1行；旧成功teacher packet禁止训练，不能复用其动作标签。
`653f884` learned R首次双臂1500updates/36.42s完成并独立审计PASS：τ val误差
比state-only低7.82%，同tick shuffle增加56.74%，但first-PD XYZ p9549.71mm，
未过5mm screen，局部UNCLEAR；无物理rollout或可用controller证明。
冻结train也有38.78mm p95，val L1只高6.95%；`1e9782c` 局部affine inverse
标定的hand/privileged next-q两臂均未过screen，hand几何恢复p95约.0016mm精确。
下一候选为global24-step context与first8 action query显式局部motion/帧差分条件，
固定数据/control/门槛小Probe；不直接扩PPO或接WM。
不把offline信用或训练reward改善当作抓取收益。consequence-evaluator后续实验按用户
要求暂停；保留其实现、修复和运行证据。继承根级Mission/Campaign/AGENTS，包括
旧ref1阶段无真实未来q/物体参考、phase/clock或触觉策略输入；当前ref3的人手/物体计划参考按用户后续授权使用。不新建分支、不push。

- [架构建议与最小验证路径](docs/research/20261010-ref1-architecture.md)
- [RLT/PPO/chunk原始来源核对](docs/research/20261010-trajectory-policy-source-check.md)
- [REGRIND源码级审计与迁移边界](docs/research/20261010-regrind-source-audit.md)
- [48维decoder覆盖Probe](docs/experiments/probes/P-20261010-trajectory-decoder-coverage.md)
- [固定D的前缀最优拟合诊断](docs/experiments/probes/P-20261010-decoder-prefix-fitting.md)
- [固定PCA时间基底覆盖](docs/experiments/probes/P-20261010-lowrank-trajectory-decoder.md)
- [pose/FF metric覆盖与停止48D搜索](docs/experiments/probes/P-20261010-metric-trajectory-decoder.md)
- [独立H→c初始化与完整执行](docs/experiments/probes/P-20261010-history-trajectory-actor.md)
- [启动平衡初始化与转入RL的决策](docs/experiments/probes/P-20261010-startup-balanced-actor.md)
- [首次真实任务PPO及冻结比较](docs/experiments/probes/P-20261010-trajectory-ppo.md)
- [只改信用时间尺度的lambda1任务Probe](docs/experiments/probes/P-20261010-trajectory-ppo-long-credit.md)

本Task成功仍需自训练操纵策略与matched Cm-on/off训练收益，不能由decoder重建、
单个成功视频或冻结控制器收益替代。具体Probe先写Task-local实验卡，再运行。

- [冻结同噪声随机部署对比](docs/experiments/probes/P-20261010-trajectory-stochastic-execution.md)
- [真实Adam重建与单update更新尺度诊断](docs/experiments/probes/P-20261010-trajectory-step-replay.md)

- [ref1_1 PointWAM原始来源核对](docs/research/20261010-ref1_1-pointwam-source-check.md)
- [ref1_1对本项目的结构取舍与最小验证](docs/research/20261010-ref1_1-retargeter-design-review.md)
- [真实hand/action监督的learned R及训练精度诊断](docs/experiments/probes/P-20261010-learned-retargeter-inverse.md)
- [腕部局部inverse与角度周期修正](docs/experiments/probes/P-20261010-retargeter-step-inverse.md)
- [共享goal-conditioned闭环执行器合同](docs/experiments/probes/P-20261010-goal-conditioned-executor-contract.md)
- [REGRIND启发的同执行器GT/生成τ物理对照](docs/experiments/probes/P-20261011-regrind-executor-gt-generated.md)

## ref2 Stage 1（2026-10-11）

已按 `docs/user/ref/ref2.md` 实现并审计 goal-conditioned low-level 接口：live state +
8-step hand/object goal → bounded native action。离线 BC 的 validation active-action L1
为 `0.00672288`，但 BC 冻结物理执行和 BC 初始化后的 24-update PPO 冻结执行均为
`0/16` long-held terminal；feature/command/PD audits 通过。当前 direct-action BC/PPO
variant 已关闭，不把 offline loss 或训练 reward 当作抓取成功证据。仓库已有 tau-only
nominal-goal + bounded-residual 物理路线；它恢复了短 hold，但严格 near-teacher/clipping
门槛仍未通过，因此后续应优先修复可行的 preload 参数化或取得明确
`training_allowed=true` 的 teacher rollout，而不是启动 REGRIND 原配置。

随后完成的固定 tau residual-basin 诊断在同一 executor 上找到一个局部低裁剪方向：冻结
`thumb_pitch_minus` 在单 motion 上为 `4/4` long-held terminal、`0.1845%` clipping，
但 `thumb_yaw_plus` 为 `0/4`，且完整 controlled-placement success 仍为 `0/16`。
这只授权一次 matched robustness Probe；它仍不是 goal-conditioned policy 已解决或完整
REGRIND 的结果。

seed287 的一次 matched repeat 仍满足门槛：baseline `3/4`、`thumb_pitch_minus` `3/4`
且 clipping `0.2306%`；yaw 从首轮 `0/4` 变为 `3/4`，显示 separate-live launch
variation。两轮完整 controlled-placement success 都是 `0/16`，因此只保留
`thumb_pitch_minus`，不把固定残差当作 Stage 1 已解决。

随后完成的 residual integration Probe 使用 24-step hand/object goals、前一步 bounded
residual 和共享 `GoalConditionedExecutor`，nominal q/velocity 来自 geometry cache 并明确
标为 privileged integration diagnostic。offline raw-latent L1 为 `0.0249544`，物理运行
10/16 long-held terminal、15/16 hold45、median held 483，但 clipping `6.6075%` 超过
`<1%` 门槛，故 `UNPROMISING`；独立 feature/latent/executor/PD audit 通过。这个结果关闭
当前 residual student 的 PPO continuation，不把失败归因于 executor wiring，也不启动
REGRIND 原始规模。

为检验冻结 tau basin 是否能迁移到 student，在同一执行器上加入
`thumb_pitch_minus` 的 raw residual coordinate 11 固定偏置 `-0.05`。该 matched Probe
达到 `12/16` hold45、clipping `0.8187%`，但 `0/16` long-held terminal，median held
仅 `79`，完整 controlled-placement success 仍为 `0/16`；独立 feature/latent/executor/PD
audit 通过。因此固定偏置没有通过预注册的 `12/16` physical gate，当前 residual student
路线关闭，不启动 PPO 或继续扫描 offset。

- [goal-conditioned BC](docs/experiments/probes/P-20261011-goal-conditioned-bc.md)
- [BC physical execution](docs/experiments/probes/P-20261011-goal-conditioned-execution.md)
- [goal-conditioned PPO](docs/experiments/probes/P-20261011-goal-conditioned-ppo.md)
- [PPO frozen execution](docs/experiments/probes/P-20261011-goal-conditioned-ppo-execution.md)
- [goal-conditioned residual integration](docs/experiments/probes/P-20261011-goal-conditioned-residual-integration.md)
- [goal-conditioned residual offset](docs/experiments/probes/P-20261011-goal-conditioned-residual-offset.md)
- [fixed tau residual basin](../consequence-evaluator/docs/experiments/probes/P-20261011-tau-residual-basin.md)
