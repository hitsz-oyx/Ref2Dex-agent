# Consequence evaluator

## 2026-10-10 生成τ执行状态

纯H四帧→十候选→选择→原生几何→冻结控制器的完整542步已完成（7840f92，107.71s）。
300step CUDA拟合保持原几何，解决了延迟；GTτ4/4长时终末held且无裁剪，生成三组各0/4
形成抓持，裁剪92--100%，当前冻结组合UNPROMISING。c8280e3全轨迹输入/命令/outcome
审计通过。此结果不能分别归因于proposal/scorer/τ路线，也未完成ref8或Mission。
下一步短诊断隔离GT在线投影/chunk接口与tick0启动覆盖；privileged GT对照不作为部署。
协议：[原生执行](docs/experiments/probes/P-20261010-generated-tau-native-execution.md)、
[接口诊断](docs/experiments/probes/P-20261010-generated-tau-interface-diagnosis.md)；
工具：[生成运行时](src/consequence_evaluator/generated_tau.py)、
[原生入口](tools/run/probe_reference_tracking.py)、
[执行审计](tools/audit/audit_generated_tau_execution.py)。

## 2026-10-10 ref8：H→候选τ 当前进展

按用户[ref8](docs/user/ref/ref8.md)优先解决proposal，冻结PointWorld和GT E/I必要性重复实验。
用户确认仅当前/过去状态，不用phase/clock。旧actor-observation H含未来参考/contact差，
保留旧结果但不再称其为纯历史输入；固定旧权重测试不能通过清零恒定列或简单clip修复。

已实现四测量state的300维历史及固定1200step位置/位移匹配对照，排除未来参考、接触力
和actor输入。完整测试位移预测RMSE212.66mm，优于persistence262.14mm，H24也改善，
通过原门槛`PROMISING`；位置预测463.30mm。K8训练集检索覆盖上界171.91mm，但top1
242.78mm未过门槛，未证明选择或接触执行。包含全部异常episode；target统计变化也属于干预。

工具：[测量历史输入](src/consequence_evaluator/proposal_history.py)、
[训练/检索Probe](tools/run/probe_measured_history_tau.py)、
[旧权重诊断](tools/audit/audit_history_tau_proposal.py)。
协议：[proposal诊断](docs/experiments/probes/P-20261010-history-tau-proposal-diagnosis.md)；
[ACT/CVAE/DP/PointWAM/DexWM方法核对](docs/research/20261010-candidate-tau-proposal-review.md)。
权重：outputs/consequence-evaluator/measured-history-tau-proposal-20261010-r1/displacement-best.pt，
必须同时加载checkpoint中的train-only statistics、输入clip10、当前手位移解码。
下一步检验生成τ的评分迁移与候选可执行性；当前没有在线selector/PW/τ→A→Z或Cm收益结论。

## 2026-10-10 τ-only 当前进展

按用户 [ref7_4](docs/user/ref/ref7_4.md)采用两层执行：11点未来τ先得到几何q_hat/
腕部速度，再由当前状态反馈学习残差控制。移除真实未来q_ref和未来物体输入/
奖励，不新增触觉输入；保留live q/dq、hand/object/velocity及previous residual。
几何参考只拟合τ、initial q和静态URDF，整体坐标RMSE2.47mm，拇指尖13.80mm。

最新手指命令重新拟合只更新六个手指输出，保护腕部/编码器，用旧τ控制器实际执行的
PD命令及学生实时状态覆盖作为训练信号，未用真实未来q/物体作标签。固定最终权重完整
评估12/16达到433帧且保持到结束，中位479帧，裁剪0.06919%，原门槛`PROMISING`。
仍有三行无支撑下落、一行近桌面分离；同轮冻结τ为15/16，不能宣称抓持稳定性更优。
随后64update仅手指PPO退化到0/16末帧持有；保护参数/输入/PD检查通过，失败记录保留，
保留预声明的重新拟合候选，不继续无约束PPO或seed sweep。

协议：[手指命令重新拟合](docs/experiments/probes/P-20261010-tau-finger-canonicalization.md)。
保留checkpoint：outputs/consequence-evaluator/ref7_4-tau-finger-fit-20261010-r3/final.pt；
配置/哈希/复现argv：outputs/consequence-evaluator/ref7_4-tau-finger-controller-20261010-r1/controller.json。
下一决策是在接触适配时保护有效的手指预载，仍未解决所有失持。

以下保留原τ训练的失败证据：冻结旧自训练tracker迁移后13/16近teacher，但裁剪41.01%。首次微调因奖励误把
手掌高度当物体高度而放弃抓持；真实模式回归测试和独立trace审查定位并修复。
修复后固定128update warmstart微调：最终16/16 hold45、中位292帧，4/16近teacher
且末帧仍持有，纯几何0/16；裁剪17.90%，强门槛`UNPROMISING`，尚未训练好稳定
τ执行器（此为重新拟合前的状态）。12条失持均为参考仍held时无支撑重力下落，非原始任务放置。
实际897输入/command/PD独立重构通过。本轮不关闭τ路线，也不增加参数/seed sweep。

协议：[τ几何闭环](docs/experiments/probes/P-20261010-tau-geometry-tracking.md)。
实验配置与复现argv：outputs/consequence-evaluator/
ref7_4-tau-tracker-controller-20261010-r2/controller.json（明确未通过强门槛）。
这是single-motion GT-τ、旧自训练oracle tracker warmstart上界；尚非新τ泛化/
从零训练/原始放置/Cm收益。旧oracle成功结果保留在下面，不能当成τ-only结果。

## 2026-10-10 reference-tracking 当前进展

按用户 [ref7_3](docs/user/ref/ref7_3.md)路线，已找到并修复腕部 position-only
PD 参考缺少速度阻尼补偿的问题。使用原生 Kd/Kp 与参考 q 的中心差分速度，
仅补偿 wrist6维，保持手指反馈与 native full-command adapter。

冻结旧策略、随机四角色完整542步对照中，保持中位数从285提高到479.5帧；
13/16达到近teacher门槛并保持到末帧，零裁剪，原始预声明screen为`PROMISING`。
随后固定128update微调、final完整评估：tracker保持中位483帧（同轮teacher484），
31/32达到433帧且末帧仍持有，nominal+FF中位92帧；裁剪0.438%，强门槛通过。
仍有1/32在tick451脱手自由落体。最终结果仍是单seed Probe，不是正式Validation。
固定checkpoint与复现argv见 outputs/consequence-evaluator/
ref7_3-tracker-feedforward-controller-20261010-r1/controller.json。
这是measured teacher holding上界。原始motion有放置，不能把成功放置的末帧松手
判为失败。尚有robot-q/object-future oracle，仍未完成11点τ→A，不宣称Cm收益。

协议与结果：[feedforward修复](docs/experiments/probes/P-20261010-reference-tracking-feedforward.md)、
[旧reference-tracking Probe](docs/experiments/probes/P-20261010-reference-tracking.md)。
工具：[训练与评估](tools/run/probe_reference_tracking.py)、
[行为与原生PD审计](tools/audit/audit_reference_tracking.py)、
[掉落与放置语义审计](tools/audit/audit_reference_tracking_drop.py)。
运行修复控制器必须显式传`--wrist-feedforward`；旧checkpoint的权重本身不包含
这个解析前馈，不能把单独加载旧权重等同于加载修复后的控制链。

## 2026-10-09 物体系GT诊断（历史状态）

按用户 [ref7_1](docs/user/ref/ref7_1.md)完成16组物体系 GT servo/接触诊断，
整体 **UNCLEAR**，完整几何控制gate未通过；旧R、ACT/G/H->V训练继续暂停。
分支保持 `main`。所有未来 geometry/q/actions 都是特权 oracle，尚不能部署或宣称 Cm 增益。
协议与逐轮证据见 [物体系 GT 上界 Probe](docs/experiments/probes/P-20261009-object-relative-gt-servo.md)。

有界物体系命令修正 held484/483，但保留了source finger PD预载；完整SE3反馈会
放大运动。11点解析腕逆解+train-only一步动态逆控制，在source finger命令下两次
held483、约3mm手误差，腕部信号PROMISING。完整11点控制一次held485但live
teacher未过行为门槛，重复仅238；实测next-finger-q及过去命令负载EMA两臂均0。
小世界系手误差仍不能保证接触。静态coupled逆解不能精确表示受载实际关节，
当前优先固定腕部、单独检验finger commanded PD target与接触状态解码。

16组执行/时序/矩阵/负载重建独立审计PASS，18项合同测试通过。相同目标流的
物理角色仍可在接触前后分叉，尚不具备strict same-state因果比较。
汇总CSV/图在 `outputs/consequence-evaluator/object-relative-gt-campaign-20261009-r1/`。
原 [GT-hand retargeter](docs/experiments/probes/P-20261009-gt-hand-retargeter.md)
held117/42/0历史失败保留，不将旧world replay称为geometry inverse。

## 暂停的手执行桥路线

之前按用户 [ref5](docs/user/ref/ref5.md)推进普通随机干预 rollout 的手执行桥：
`G(H,A) -> 24x11x3 hand -> frozen PointWorld -> frozen C1 -> 每8步选择7候选`。
不扩大 fork panel，旧 `U32=Y7+.25Y3-Y6` 仍只作为 evaluator teacher。
自训 e260、PW best46000 和 C1 step1000 全部冻结；在线选择只读当前/过去几何，
不接实测未来手轨迹。分支保持 `main`。

[手执行桥与在线 Probe](docs/experiments/probes/P-20261009-hand-execution-bridge.md)
完成96train/64val/64test和另一个64episode新test，全部完整542步；
原始96/64/64数据共3360窗口，另一个test960窗口。两次匹配HA/H-only训练
各2000步，保持同模型/采样/val选择，仅第二次改变固定native请求单位。
手轨迹预测明显优于persistence/nominal FK，但zero-A误差没有达到预设1%门槛，
因此不把桥称为可靠的candidate execution predictor。选模型仅用val误差，
保留第一版step200。同C1 waterfall已完成；初始seed重复已修复为匹配初始q扰动。
真实32episode/arm中baseline/planner完整受控放回均0，stable45为24/26，
稳定后失抓1/8，平均非零干预0/55.97。在线选择/过去几何/完整542步合同通过，
但控制收益UNCLEAR；不扩百级episode，不把stable45当完整任务或安全保持成功。
下一步须区分动作敏感性、C1跨actor迁移和baseline放回阶段不足，旧Y继续冻结。
工具：[采集/在线执行](tools/run/run_hand_bridge_rollout.py)、
[桥训练](tools/run/train_hand_execution.py)、
[三层误差审计](tools/audit/audit_hand_bridge_waterfall.py)、
[完整任务对照审计](tools/audit/audit_hand_planner_control.py)。

## 保留的旧 U evaluator 实验

之前按用户 [ref4_4](docs/user/ref/ref4_4.md)冻结旧 `U32=Y7+.25Y3-Y6` 为 teacher，
推进 C0(H+A)、C1(H+A+GT24)、C2(H+A+PW24) evaluator；future均只有24步几何，
force-pair/contact只生成32步标签。C2按用户选择使用实测未来手轨迹条件，替换物体
future，是离线oracle，尚不能作为可部署planner。分支仍为`main`。

[首轮旧U evaluator Probe](docs/experiments/probes/P-20261009-old-utility-evaluator.md)
完成192源episode的独立训练/验证/普通测试和保留25x7候选考试。
候选严格排序C0/C1/C2为60.26%/75.64%/69.23%；C1 shuffle降至51.28%，
GT信息screen为PROMISING，PW retention screen未过。真实one-shot Z90为
baseline20、C019、C120、C221（25锚点），C2 rescue1/harm0，控制收益UNCLEAR。
同C1权重换PW的离线诊断为73.08%，提示还需区分world-model误差和evaluator拟合。
当时建议冻结共同C1，在独立cohort比较GT/PW；ref5现在用普通rollout执行桥
取代继续扩fork数据。不再定义新Y，不直接进入RL。
旧teacher rolling23/25与本次one-shot分开保存，不能据单次选择关闭rolling路线。
[标签准备](tools/run/prepare_old_utility.py)、[PW推理](tools/run/predict_old_utility_future.py)、
[匹配训练](tools/run/train_old_utility.py)、[执行/权重复核](tools/audit/report_old_utility.py)。

## 保留的弱时序 evaluator 实验

之前按用户 [ref4_3](docs/user/ref/ref4_3.md)尝试 learned `Q(H,A,Zgt)`：
episode 最终抓稳并受控放回为 success，成功轨迹中的失抓/recovery 窗口 mask；
重新稳定45帧后恢复正监督。标签为 `±24/(T-t)`，时钟和 outcome 只作监督，
模型只输入原生 H、预知24步残差计划 A 和实测 future Z，不使用 TCC/S/P/M value。
开发分支仍为 `main`。

[首轮弱时序 Probe](docs/experiments/probes/P-20261009-weak-temporal-value.md)已完成：
192完整episode，HA/HAZ同初始化各1200步，GPU2用时26.66秒。
冻结测试排序为88.42%/89.67%，future打乱后84.32%；额外收益1.24pp未达3pp screen，
状态UNCLEAR，尚未进入完整任务GT候选控制/Gate1。训练窗口仅3.23%含非零计划，
没有成功recovery训练样例；下一步优先补决策时刻、同H候选及恢复覆盖，而非增加epoch。
复用旧25锚点的缓存仅做评分迁移审计，没有完整任务结局，不作成功率证据。
[标签准备](tools/run/prepare_temporal_value.py)、[训练](tools/run/train_temporal_value.py)和
[冻结审计](tools/audit/audit_temporal_value.py)使用独立 temporal-value schema。

## 保留的 reference-progress 路线

之前按用户 [ref4_2](docs/user/ref/ref4_2.md)修订 [ref4_1](docs/user/ref/ref4_1.md)：
保留`Y_t=P_(t+24)-P_t`，P改由成功真实机器人reference bank与实际3D因果历史匹配；
不用episode时钟或最终结局。开发分支为`main`。
[物理reference bank Probe](docs/experiments/probes/P-20261008-physical-reference-bank.md)
在既有source230前8条成功轨迹上固定1000步TCC，同source261四clean/四contact检查：
clean tick176 Y转正、负窗口全0、累计回退<=0.002428，既定开发门槛通过，PROMISING。
bank由[构建工具](tools/audit/build_physical_reference_bank.py)冻结实测11手点/物体姿态；
[标签工具](tools/run/label_reference_progress.py)对8个独立prior固定等权平均。
旧[原始参考几何Probe](docs/experiments/probes/P-20261008-reference-progress-labels.md)
和[原始R TCC Probe](docs/experiments/probes/P-20261008-tcc-phase-alignment.md)完整保留。
标签通过后按[完整链路](../../../docs/user/完整链路.md)先做直接GT-value的Gate1，
再考虑evaluator；[Gate1入口](docs/experiments/probes/P-20261008-gate1-gt-progress.md)
已修复单样本GPU actor前向，CPU-PhysX/GPU actor的72步零分支重放完全一致。
首轮1个配对episode已完成：严格滚动重放通过，任务成功0/1对0/1，结论UNCLEAR。
CPU后端baseline未抓起，GPU预检同策略hold483步；先恢复代表性GPU执行，
暂不扩大CPU样本或训练evaluator。
新schema独立于旧S/P/M和旧局部H匹配标签。

### ACT-like native proposal engineering route

`src/consequence_evaluator/action_chunk.py` defines a separate engineering
contract for `H_t -> action[t:t+24]`: the target is the 18-D control captured at
the native `pre_physics_step` boundary, while `residual_plan` and hand flow
remain excluded. `tools/run/train_action_chunk.py` enforces episode-separated
windows, clean `expert_success` provenance, train-only/frozen observation
normalization, and an explicit audit-only mode. Its outputs are marked
`engineering_only` and cannot be passed to evaluator fitting.

The native group behavior harness accepts the resulting checkpoint through
`--action-chunk-checkpoint`. `open_loop24` is the retained behavior screen;
`receding8` for the direct imitation checkpoint fails before lift. Candidate
mode additionally accepts a provenance-checked recorded chunk through
`--action-chunk-replay`, but its nominal behavior and zero-pair gates must pass
before any candidate effect is usable. The evidence and stopping decision are
recorded in [P-20261009-act-native-chunk](docs/experiments/probes/P-20261009-act-native-chunk.md).

The ref6 inference diagnosis additionally supports `overlap8` (query every8,
combine overlapping chunks) and `temporal1` (query and combine every step),
with absolute-tick expiry and official oldest-first exponential weighting.
Both reduce executed control changes but still fail to grasp with the frozen
historical checkpoint. The actual stride8 query-grid horizon errors do not
show a disproportionately weak first8-token average. See the
[temporal aggregation Probe](docs/experiments/probes/P-20261009-act-temporal-ensemble.md)
and [horizon/dispatch audit tool](tools/audit/audit_action_chunk_execution.py).

`receding1` queries every step and executes only the latest horizon0 control,
discarding previous chunks. It also fails to lift with this frozen checkpoint
(held0, while the teacher held484), despite bitwise latest-action/native
dispatch checks. See the [one-step Probe](docs/experiments/probes/P-20261009-act-receding1.md).

## 历史路线与保留证据

早期用户方案：[ref1](docs/user/ref/ref1.md)，连续采集按[ref2](docs/user/ref/ref2.md)；数据分级按[ref3](docs/user/ref/ref3.md)。历史实现分支：`consequence-evaluator`。
历史离线 oracle headroom 检验与当前弱时序 Probe 都不改变根级 Mission 的最终 Cm 策略收益要求。

2026-10-08：[官方DExplore生成器筛查](docs/experiments/probes/P-20261008-official-generator-screen.md)
已跑通归档的完整运行时，并完成官方/自训各64条同输入评估。官方45帧保持覆盖64/64，
但原reference末尾放回与全程保持S标签冲突。用户已选择完整reference，正常受控放回
计为Task成功；[新几何数据Pilot](docs/experiments/probes/P-20261008-full-reference-value-data.md)
先检查此语义，native诊断输出不直接接入训练。权重只用于明确标识来源的数据生成。

新Pilot已完成64条nominal完整episode：几何弱标签62成功/2失败，4160窗口、
256个自动S/P/M偏好；仅train组且负例不足，尚不可训练。独立新入口为
`collect_continuous.py --value-outcomes --official-generator --clean-only` 与
[value标签准备](tools/run/label_value_outcomes.py)，K/K_exec均为24；它们使用独立
value schema。以下旧H匹配局部偏好管线继续作为独立验证合同，不读取新schema。

[固定dose采集](docs/experiments/probes/P-20261008-value-dose-calibration.md)现已完成
192episode：train41/23、val38/26、test36/28（成功/失败），11904窗口/768偏好。
同H官方/自训策略误差只用train来源估计；新采样请求残差上限0.5需显式bank与
`--residual-bound0.5`匹配，实际原生控制仍[-1,1]、平移仍0.05。0.2默认和旧合同保留。
机械覆盖通过，尚未fit；原S恢复语义问题现由ref4_1的局部因果delta-progress方案替代，原标签不改。
[数值复核和训练样例图](tools/audit/audit_value_outcomes.py)供人工弱标签检查。

第一阶段固定 `K=24`、`K_exec=8`，H 沿用采集策略当前观测/历史的原始合同。
先采连续六专家 rollout，单次24步平滑扰动后让专家继续到 episode 结束；不 fork。
比较 `E0(H,δ)`、`Eoracle-E(H,δ,Z_object)` 和 `Eoracle-EI(H,δ,Z_object,Z_interaction)`。
δ为执行前已确定的24步18维请求残差计划，专家仍逐步反馈控制；PointWorld 适配与在线 proposal 延后。
主要衡量同任务、同当前阶段、跨 episode 的局部偏好排序；它不是同状态候选控制收益。

## 已实现的最小合同与训练入口

- [窗口准备](tools/run/prepare_windows.py)：episode/seed group 先 split，再切完整24步窗口。
- [连续采集](tools/run/collect_continuous.py)：复用原生六专家player，分阶段单次24步平滑扰动，完整episode导出。
- [数据合同](src/consequence_evaluator/data.py)：严格输入白名单、完整时钟、刚体效应、标签 mask 与 split 检查。
- [三臂模型](src/consequence_evaluator/model.py)：同容量三臂两层128维 Transformer，24步 progress 分布与独立标量分支评分。
- [训练](tools/run/train_matched.py)：相同初始化、抽样、优化器和更新预算；只在训练集拟合归一化，用 val 保存最佳权重。
- [独立评价](tools/run/evaluate_matched.py)：冻结val所选checkpoint后，报告test排序、任务/阶段分组及跨episode未来替换对照。
- [资产预检](tools/audit/preflight.py)：核对六专家、motion、旧 rollout 和当前 GPU 占用。
- [研究原文](docs/research/PRIMARY_SOURCES.md)与[代码/数据复用核对](docs/DATA_REUSE.md)。

三臂保留相同45维 future module 和初始化。归一化后，E0屏蔽全部future，
Eoracle-E仅保留12维物体effect，Eoracle-EI再保留11个测量手关键点的相对物体坐标（33维）。
模型输入只有H、执行前已知δ、物理future，不接收未来reward、success、drop/contact flag、
episode quality、扰动身份或时间到成功。物体effect为 `T_t^-1 @ T_(t+1:t+24)`，
手部future使用每个未来物体坐标系；具体顺序由 `contracts.HAND_LINKS` 固定。
请求残差区别于裁剪/噪声后的actual_residual；后者与真实闭环action只用于审计。
未来状态触发起点尚未知的窗口被排除，clean为已知零计划，触发后剩余计划已知。
18维残差接口与PointWorld的手部point-flow仍需适配；目前PointWorld只预测物体effect，
尚不能声称恢复interaction oracle的差距。

Robometer 原版使用联合两轨迹 preference head；独立 `s` 的 Bradley–Terry loss 是明确改编。
Progress 用10-bin soft CE，只监督可靠 `expert_success` 的绝对 episode 进度，失败/次优 mask。
不把每个窗口重新标成0→1，也不把 episode 最终失败直接继承为全部局部窗口失败。
首版要求可审计的局部偏好 annotation；不自动产生旧Y或从窗口外结局伪造局部排序。
[监督准备](tools/run/label_continuous.py)只对明确的保持/掉落和抬升/失抓窗口给出偏好，
模糊的miss/recovery比较不标。可靠clean轨迹的45帧保持且之后不掉落用于绝对progress锚点；
perturbed/失败/次优精确progress均mask。三臂另抽相同的train专家窗口，保证progress锚点不依赖偏好配对覆盖。

### 当前 fit gate

局部偏好必须同时匹配历史 `H`、当前物体位姿和11点手状态。`H` 的相对RMS上限为
25%，物体/手的几何阈值仍按上面的固定合同执行。每个 split 还必须有至少
8/4/4 个不同的无序 episode pair（train/val/test）；同一对 episode 的重复窗口不能
充数。collector 只把该要求写入来源 manifest；labeler、prepare、`Windows` 和 train
会实际核对配对数量以及当前规则、合同和来源 hash。六个专家全部通过固定资格并且
route 的 `training_allowed=true` 之前，连续采集和真实 fit 都保持 observational-only。
当前 collector 会拒绝不合格路由。已有原始连续数据可用 labeler 的
`--audit-only` 重算标签与覆盖率；审计输出始终 `training_allowed=false`，不能 prepare/fit。
collector 同样支持显式 `--audit-only`，仍验证全部六权重的自训练来源和输入hash，
并强制产物不可训练。审计模式可加 `--target-phase hold`（或其他已有阶段），
在每个实际motion内跨wave均衡分配clean/目标阶段；完整episode、单次24步残差、
专家后续反馈和不fork的合同不变。生产模式保留六阶段分配和全部六角色资格门槛。
用户已明确twin只作工程诊断；后续正式数据始终用ref2连续rollout。

同一 current state 的双分支由 `src/consequence_evaluator/twin.py` 定义
`ref2dex.consequence-evaluator.twin.v1` 合同：必须保存完整 native task/controller
buffer、Python/NumPy/Torch RNG、fresh simulator prefix replay provenance 和双分支第0帧
锚点。当前已提供不导入 Isaac 的 native capture adapter，可把初始化后的 task、controller
buffer 和完整 prefix trace 转为该合同；它会区分 30 Hz control tick 与原生 60 Hz
physics frame（按 `control_freq_inv` 核对帧数）、拒绝缺失字段、非 fresh replay、状态帧数不匹配
或超出 replay 误差门槛的记录，并冻结 task 的直接 tensor/scalar inventory、禁用随机化和
motion sampler。通用 snapshot 要求显式 `{'is_rnn': ..., 'state': ...}` sentinel，并校验
Python/NumPy/Torch RNG 的具体格式，并提供对应的 CPU/GPU Torch RNG restore helper。
native DExplore adapter 进一步固定 `control_freq_inv=2`、`sim_params.dt=1/60`；PhysX
`substeps` 必须由 caller 放入 physics provenance hash，但不计入 Gym frame count。
非RNN controller 也必须显式记录空 RNN sentinel。
[native 工程检查](tools/audit/native_twin_probe.py)现在独立执行三个 fresh simulator，
完成同前缀双残差及重复分支。2026-10-08 r4 精确状态/RNG及物体、手、done、完整
native future 重放均通过；内容哈希也通过保存/加载一致性检查。该输出始终不可训练，
不接入连续数据schema，不作为 twin coverage 或 evaluator 科学证据。

## 连续 episode 输入

源目录的 `manifest.json`：

```json
{
  "schema": "ref2dex.consequence-evaluator.episodes.v2",
  "status": "COMPLETED", "rollout_kind": "continuous", "training_allowed": true,
  "fps": 30, "units": "m",
  "action_semantics": "decision_known_requested_residual_plan",
  "history_contract": "采集策略的观测/历史 schema、维度与来源 SHA256",
  "episodes": [{
    "episode": "unique_episode_id", "split_group": "source_seed_group",
    "split": "train", "task": "airplane", "quality": "expert_success",
    "expert": "airplane_base", "motion": "s3_airplane_lift",
    "path": "unique_episode_id.npz", "sha256": "实际文件 SHA256"
  }]
}
```

每个 episode NPZ 使用 `allow_pickle=False`，仅含以下字段：

| 字段 | Shape / 含义 |
| --- | --- |
| history | `[T+1, ...]`，策略原始观测/历史，不新定义history length |
| action | `[T,18]`，真实执行的[-1,1] native control，仅作审计 |
| residual_plan | `[T,24,18]`，每次执行前已知的请求残差，模型action取此字段 |
| plan_known | `[T]`，bool；尚未知未来触发时刻的计划不可训练 |
| hand_keypoints | `[T+1,11,3]`，测量native刚体关键点世界位置 |
| object_pose | `[T+1,4,4]`，测得的 object-to-stationary-world，米 |
| timestamps | `[T+1]`，真实连续30Hz |
| phase | `[T]`，当前状态决定的采集阶段，不作为额外模型输入 |
| progress | `[T+1]`，可靠成功示范的绝对进度；未知可NaN |
| progress_mask | `[T+1]`，bool；失败/次优全部false |

其中 `action[t]` 把 `object_pose[t]` 变成 `object_pose[t+1]`。
episode 文件必须完整，不能把 reset 后的另一条轨迹拼进来。
采集驱动已经接入，使用CUDA PhysX tensor pipeline；真实六专家连续采集仍待新权重资格检查后执行。
CPU测试实际执行驱动中的采集循环，覆盖不同env先后done、原生in-place action转换、
零partial reset、T控制/T+1状态对齐和恢复专家控制；它不能证明真实物理响应或阶段质量。

采集模板（权重必须匹配路由配置SHA；本地native仿真资产和配置也须就绪）：

```bash
python src/task/consequence-evaluator/tools/run/collect_continuous.py \
  --route-config src/task/CmResidual/configs/multitrajectory_object_router_with_cup_probe.json \
  --asset-root <restored-checkpoint-root> --motions <native-motion-directory> \
  --cfg-env <native-environment.yaml> --cfg-train <native-player.yaml> \
  --output outputs/consequence-evaluator/<continuous-run> \
  --gpu <idle-gpu> --seed <registered-collection-seed> --split train
```

默认24env/2waves/每episode最多1200步（覆盖最长1062帧参考）/总900秒/产物2GiB。开始前核对输入hash、GPU PID与20GiB磁盘余量。
六专家始终用原生RMS和动作接口；可传入原观测路由模型及SHA，否则使用明确记录的固定object route。
每episode分配clean或approach/contact/grasp/lift/hold，阶段达到且剩余至少24步才启动一次残差。
平滑残差使用四结点插值和零边界taper，腕平移缩小至四分之一，native coupling覆盖的6个distal通道不加残差。
阶段定义是当前contact force proxy、3步contact、3cm lift与5步held的工程启发式，不能称为论文原版阶段。
目标阶段未达到的episode保留未触发状态，不计作扰动覆盖；完成一段扰动后完全恢复专家。
原生实现的`hybridInitProb=1`选择参考第0帧；运行时强制并核对完整起始帧。不同episode只在wave边界统一reset。
在sidecar记录真实q、hand root、contact及validity、实际残差/裁剪，供后续几何适配与标签审查；contact/force/gap、q/root与残差审计不进入Z；独立测量手关键点进入interaction future。
审计用执行动作从原生`pre_physics_step`入口捕获：wrapper裁剪及domain action noise已执行，Inspire手指/PD转换尚未执行。
不把请求的动作冒充实际指令；原生噪声若使真实归一化指令越过[-1,1]，合同检查直接拒绝该运行。
reset瞬间的contact force可能残留，标为无效，不用于阶段计数。采集progress保持未知且全mask，
不会根据最终结局自动生成局部preference。可靠专家progress与窗口内事件排序必须在独立监督步骤核验后提供。

偏好文件单独记录 `scope: local_window`、`label_provenance`，以及 `pairs`。
每对包含 `chosen` / `rejected` 的 `{episode, tick}` 与 `annotation`。
标签来源必须核对窗口内的具体事件；无明确局部顺序的比较不标。
程序验证 split/task/expert/motion/phase 相同、episode不同、annotation非空，
当前物体平移≤3cm、z差≤1cm、旋转≤15°，手11点RMS≤2cm；仍是observational matching。
事件标签要求净力proxy同时有≤1cm采样手物表面距离；稳定progress再要求抬升≥3cm、
持续45帧且之后不掉落。净力proxy及采样距离均不是精确collision-pair GT，
扩大采集前须检查真实轨迹的force/geometry一致率。

准备后运行的模板（真实源、偏好审计、空闲GPU和注册seed就绪后）：

```bash
python src/task/consequence-evaluator/tools/run/prepare_windows.py \
  --source <continuous-episode-directory> --preferences <local-preferences.json> \
  --output outputs/consequence-evaluator/<prepared-run>
python src/task/consequence-evaluator/tools/run/train_matched.py \
  --data outputs/consequence-evaluator/<prepared-run> \
  --output outputs/consequence-evaluator/<fit-run> \
  --gpu <idle-gpu> --seed <registered-probe-seed> --updates 1000 --batch 32 --seconds 1800
```

训练不使用 test pairs；训练入口只报告 val开发指标。保存三臂 initial/latest/best、训练/验证JSONL、
输入/源码hash、显存、吞吐与ETA。当前入口尚未在真实GPU数据上验证，CPU小模型检查只证明工程合同。

独立评价入口要求训练已经COMPLETED，核对原始输入/初始化/源码hash、train-only归一化和
验证日志中首次最佳的checkpoint。三臂采用相同选择规则，可以选中不同更新步；不会根据test重选。
第一次推理前在fit目录冻结`test_protocol.json`，之后不能通过换权重、诊断seed或batch重扫test。
保存每个参与test偏好对的唯一窗口分数/progress、偏好对索引和未来donor索引，便于独立复算。

```bash
python src/task/consequence-evaluator/tools/run/evaluate_matched.py \
  --data outputs/consequence-evaluator/<prepared-run> \
  --fit outputs/consequence-evaluator/<fit-run> \
  --output outputs/consequence-evaluator/<test-run> \
  --gpu <idle-gpu> --seed <fixed-diagnostic-seed>
```

主指标是test strict preference accuracy，tie计错；报告配对gain、三臂救回/损失的偏好对数量、
task/phase/quality分组和episode-pair macro accuracy。Progress MAE只统计明确可靠的masked帧，
同一窗口在多对出现不重复计算。另用相同oracle权重替换Z：同task/expert/motion/当前phase、相近当前物体/手状态、其他test episode
有放回抽取，H/A/labels均保持原样。这是未来对齐敏感性的诊断，不是严格置换检验或可执行候选。
窗口重叠，不把pair数量当独立样本数；当前汇总不提供独立pair CI，也不自动升级Validation。
默认最多128窗口/batch、300秒；GPU占用检查在加载模型前完成。31项微型CPU测试通过，
原因是当前GPU均占用，测试只验证数据合同/保存加载/评价逻辑；真实GPU采集、拟合与评价仍未运行。

## 当前判断与下一步

用户在2026-10-07明确允许重新训练专家并重新rollout。已找到外部read-only的canonical
几何motion和native资产，13条motion完成CPU合同检查。资产复制到
`outputs/consequence-evaluator/baseline-inputs-20261007-r2/`；原DExplore data失效链接已保留。
这只恢复训练输入，不恢复旧checkpoint或robot rollout。历史s3直接scratch曾失败，先新训s1 parent
并做完整frame0检查，再迁移s3、mixed12/train5/balanced5及duck/cup，最终生成新六专家route/hash。
[重建协议](docs/experiments/probes/P-20261007-consequence-baseline-rebuild.md)保留成本、资格检查和下一步。

2026-10-07：旧 oracle 原始 outputs 与六专家checkpoint未在当前仓库找到。
用户恢复本Task后，GPU0已启动baseline-rebuild-20261007-r4；2epoch原生GPU smoke通过，
64env/200epoch随机初始化s1母策略已完成，GPU1/2继续四源PointWorld混合拟合。
[母策略资格检查](tools/run/qualify_parent.py)固定64条frame0完整episode，核对45帧保持/之后不掉落，
此次0/64达到稳定抬升，先同输入回放参考动作以区分优化问题与参考/控制/几何问题；
通过后才扩展六专家与连续采集。当前44项CPU工程测试覆盖采集、标签、progress独立抽样、
三臂训练合同、独立评价和资格指标，不代表真实oracle headroom。
后续协议见[Oracle Probe](docs/experiments/probes/P-20261007-consequence-oracle-headroom.md)。

后续诊断：r4首步物体跳回创建原点，GPU reset违反setter/refresh时序。
task-local修复已接入三个native入口；同环境真实8env测试首步位移1.48m→1.43mm，
初始手FK和重复子集reset检查通过，51项工程测试通过。r4不作为有效训练负证据。
旧新均使用graspenv解释器/Dexplore_Inspire任务；已找回原s1 corrected tensor，
其q与接触合同区别于r4canonical。原输入真实reset/FK/三次子集检查已通过，缺失CmLite权重仍需重建，
不将临时随机Cm-off路线称为原母策略复现。
新一轮使用原s1tensor/原single-motion config和修复reset，GPU0完成200epoch Cm-off Probe，
独立64条frame0完整episode中50条通过45帧保持/无后续drop，满足预设8/64数据准备门槛。
该母策略权重仅属于新Probe，不继承旧Validation；缺失CmLite奖励尚未启用。
s3 corrected input已恢复并通过几何审计；下一步仅迁移20epoch并独立资格检查，
其他专家和真实evaluator拟合尚未完成。迁移和评估均核对权重祖先来自owned随机自训练。

旧ref1的reactive future A合同已被ref2替换，旧v1数据禁止进入新训练；未来action trace不再作为输入。
正向才值得进行EWM与后续在线规划；负向先查数据/监督/拟合，不能直接判世界模型无用。

## 2026-10-07 ref2/ref3 当前进展

新s1 parent固定资格50/64；s3从220有界续训至260后固定资格36/64（56.25%），
达到8/64运营准备门槛。其余十条corrected reference完成恢复/几何审计；
六专家中只有airplane_base完成，余五个专家、真实连续数据和三臂fit均未完成。
GPU0目前为他人进程，用户暂不能协调；先完成合同修订，不抢占其他任务。
74项Task工程测试通过，包括请求计划/实际动作分离、三臂屏蔽、interaction坐标、
几何接触约束和当前状态配对。GPU1的8env×128步真实Isaac geometry smoke通过（24.24s），712个forceproxy帧及568个抬升proxy帧均有≤1cm采样距离；
仍缺其他物体/桌面反例和完整采集，不能视为精确contact GT。
四源PointWorld已保存停止step14250；用户要求三源主监督（Oak/GRAB/ARCTIC），
ContactPose仅列为刚性运输辅助，EPIC保持candidate_only。三源已于23:20启动，GPU1/2每卡64，以latest14250权重初始化，
fresh优化器/调度，50000新更新、2026-10-08 10:00截止，另见混合预训练卡。

## 最新执行状态与 ref2 完成边界

2026-10-07晚：GPU0恢复可用，duck260→280短迁移完成，固定资格0/64；
5条曾保持45帧后出现drop。原生duck几何smoke1024帧/533forceproxy帧
均有<=1cm采样gap，reset/FK/子集检查通过；尚未获得桌面假阳性或
完整几何标签验证。六角色中airplane_base已达36/64，duck未过门槛，
其余四角色待训练，真实evaluator数据/三臂fit仍未开始。

修复collector读取native motion目录软链接和遗漏source依赖冻结，
按实际motion跨wave轮转clean及五阶段，记录assigned/triggered覆盖；
不能再用全局env序号分阶段并假设每对象都有clean。单元和fake-driver
测试只验证工程合同，不代替真实collect和标签审计。
三源PointWorld同panel验证会波动，完整状态以混合预训练卡为准。

### 重建后生成新的固定路由

六个实际角色训练并完成各自qualification后，用
`tools/run/prepare_expert_route.py --run airplane_base=<training-run> ...
--run cup=<training-run> --output outputs/consequence-evaluator/<run>/route.json`。
必须提供airplane_base/mixed12/train5/balanced5/duck/cup全部六角色；
脚本核对owned随机自训练祖先、每角色训练身份、六个不同checkpoint、
完整frame0资格及原始trace哈希；生产采集/fit要求六角色全部过8/64门槛。
其余弱角色的实际资格数显式保留，不能因路由就绪宣称六专家都可靠。
固定object route沿用既有映射，不从新的测试结果重选。
collector再次核对/冻结这些路由证据，真实progress和preference仍须
独立label检查；当前 route 实例的 `training_allowed=false`，而路由工具会按六个角色的
资格结果动态设置该字段。此工具不训练模型，
不会用一个actor替代六个角色，也不恢复旧Validation身份。

2026-10-08最新：六角色均已完成有界训练/资格检查，airplane/duck/cup分别
36/8/63条通过（各64条），mixed12/train5/balanced5为0/5/4，路由保持不可训练。
已有两轮连续采集共360条原始episode；第二轮216条在旧规则下只有2/1/5
train/val/test偏好，尚未拟合evaluator。新增H匹配审计得到窗口pair=1/1/2、唯一
episode-pair=1/1/1，低于8/4/4门槛，不沿用旧READY标记。
native twin r4已通过独立工程检查，但不进入连续数据schema。
