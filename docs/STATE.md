# Ref2Dex 当前研究状态

## 2026-10-10 主线 τ-conditioned evaluator Probe

为继续 `docs/user/完整链路.md` 的第 7–8 步，新增真实 hand-trajectory 接口的
离线 evaluator，而不是复用旧 `[24,18]` native-action branch。C0 输入 `H+τ`，
C1 输入 `H+τ+E_GT`，C2 输入 `H+τ+E_PW`；τ 使用 `pw_hand_future` 当前查询物体坐标系，
对象 effect 单独取 GT/PW 的前12维。GPU2 的 1200-step fit 在约31秒完成，输入、同-H
25x7 panel、train/val/test split 和 checkpoint hash 均记录在
`outputs/consequence-evaluator/trajectory-utility-fit-20261010-r1/`。

结果为 `UNCLEAR`：C0/C1/C2 严格 pair accuracy 为 .6282/.6410/.7692；C0 的 τ
shuffle 从 .6282 降至 .3718，说明新 τ 分支确实被使用；但 C1 增益仅1.28pp（gate
要求3pp），C1 τ-shuffle drop 也仅1.28pp。C2 的 τ-shuffle 为 .7821，高于未打乱的
.7692，不能把其 raw 分数写成 PW 保留能力。panel 仅5/25锚点有严格 label 差异、78
strict pairs，当前不扩 epoch、不改 gate、不进入可部署 planner；保留为主线第7–8步
的接口与证据，下一步需要非平凡同-H candidate bank/H→τ proposal 审计。

随后审计发现 r1 的 C0/C1/C2 是三个独立随机初始化，`initial.pt` 是未参与训练
的第四个实例，因此 r1 的 C1−C0、C1 shuffle 和 C1/C2 gate 不能作 matched-arm
归因。按 Decision Note 保留 r1、不覆盖产物，只修复初始化并以同数据/seed/步数重跑
`trajectory-utility-fit-20261010-r2`（GPU2，32.2s）。共享初始化后的 panel 为
C0/C1/C2 `.6282/.6538/.5897`，C1 增益 `.0256`、τ-shuffle drop `0`；GT
information 与 PW retention gates 均为 false。该实现修复后的结果仍只覆盖5个
informative anchors/78 strict pairs，故不扩 candidate bank、不启动 C2/PW 在线链路。
随后在 clean commit `15ca96f` 复播同一配置为 `trajectory-utility-fit-20261010-r3`；
初始与三臂 checkpoint hash 和 r2 byte-identical，确认 r2 数值不是脏工作树偶然产物。

随后对冻结 `H.pt`/`HA.pt` 做只读 candidate-bank audit（commit `e0b8e2f`）：H-only
七候选完全相同（candidate RMS 0），HA 虽有动作条件，但 panel 候选差异仅约
1.09 mm mean / 1.40 mm max，observed-τ point RMSE 约59.53 mm。将 HA 生成 τ
喂入冻结 evaluator 后 C0 严格 pair 仅29.49%，且选择集中在 candidate 2/4；这是
evaluator 训练于 observed τ、生成 τ 属于 OOD 的诊断，不是任务控制结果。它把下一
个低成本 blocker 具体化为“先获得有足够候选多样性且非 tie 的 H→τ bank”，而不是
继续加 evaluator epoch。

为区分 panel 太小与候选本身塌缩，随后只读重建 ref13 recovery 的单一路径
`initial + new-o8 + new-o16` bank（输出
`outputs/consequence-evaluator/trajectory-rolling-panel-audit-20261010-r7/`）：
75 个同-H panels、13 个 informative anchors、163 strict pairs；每个候选使用
自己的 post-query `trace.pt` geometry，H/prefix/valid32/zero-clipping 合同通过。
观测 τ 的 candidate spread 为 initial 9.37/61.35mm、new-o8 8.48/65.02mm、
new-o16 10.31/225.73mm（mean/max RMS），所以该 bank 并非 HA bridge 的1mm塌缩，
但三个 cohort 是同一 reconstructed e260 actor/motion 的 treatment-conditioned
时间视图，不是75个独立环境。

冻结 ordinary-data C0/C1 在该 bank 上为 .5706/.6319 strict pair accuracy，
C1 相对 C0 +6.13pp、mean regret .03528→.02958；然而 C1 τ-shuffle .6196，
只下降1.23pp，且 C1<.70，固定 GT-information screen 仍为 false。因而没有启动
C2/PW inference 或在线控制；该结果只说明扩大 observed bank 仍不足以证明稳健的
τ-conditioned ranking。parent run/offset 已写入 manifest，r1–r6 的构建工程失败保留，
未形成科学结果。

随后完成冻结离线 `H→τ→PointWorld→C2` planner-chain wiring audit（输出
`outputs/consequence-evaluator/trajectory-planner-panel-audit-20261010-r4/`，GPU2）。
每个同-H anchor 只取一份当前状态，再由 HA bridge 展开7个候选；PointWorld
重复推理 bitwise 稳定，shape/frame/hash 合同通过。生成 bank 的候选 spread 仍为
1.09/1.40 mm，和观测 τ 的 point/H24 RMSE 为34.37/44.97mm；OOD C2 在25×7
panel 的 strict pair accuracy 56.41%、top-1 agreement 20%、mean regret .01625，
且选择集中到 candidate5（10/25）。20/25 anchor 属于 teacher tie set（tie-set
coverage 92%），因此这些数不能写成排序或控制收益；它只证明完整离线适配器可执行，
并再次定位“独立且足够多样的 H→τ candidate bank”是当前 blocker。r1/r2/r3 的参数、
索引和断言工程失败均保留，未形成科学结果。

## 2026-10-10 ref7_2 full-action hand retarget follow-up

按 `src/task/consequence-evaluator/docs/user/ref/ref7_2.md` 完成路线 B 的最小
判别 Probe：GPU2 采集 96/64/64 条结构化 retarget rollout（每条 542 步），
覆盖 wrist 平移/旋转、finger 开合/预载、联合动作及 approach/contact/hold；
action capture、q/dq/hand 对齐、episode split、残差边界和零 clipping 全部通过。
第一轮离线 full-action fit 的 test L1 为 0.01637；加入三条 train、 一条 val 和
一条 held-out teacher motion anchor 后为 0.01596，仍仅是 engineering fit 证据。

GT future hand→R→native Gym 的首个对齐复核（teacher held481）中，R 三角色
held10/30/21、hand RMSE44.68/43.59/57.71mm，未达到 held≥90% 且 RMSE<40mm；
native requested/actual 完全一致且零 clipping。query_period=1 的 receding 诊断
同样失败（teacher482，R held0/0/8，RMSE177.12/226.69/207.27mm）。这排除了
“只要缩短 24 步 dispatch 就能通过”的充分解释，但不把结果升级为所有 hand
geometry 无效的正式结论。Probe 总状态 `UNCLEAR`，当前 retarget execution gate
`UNPROMISING`；不进入 H→hand、PointWorld、evaluator 或 Cm claim，返回
ref7_1 的 finger commanded-PD/contact-preload 诊断。证据卡：
[P-20261010-hand-action-retarget-data](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-retarget-data.md)，
产物位于 `outputs/consequence-evaluator/ref7_2-hand-action-*`。

随后做了 source/backend isolation：同 CPU tensor pipeline、同 4-env direct actor
layout 的 teacher packet 使 contextual R 在 env0 future broadcast 下 held=6/9/76、
hand RMSE=61.4/75.8/77.4mm（冻结 GPU source 为0/0/0、109–114mm），确认旧执行失败
混入 source/backend mismatch，但仍未过 gate。用每个 source env 自己 future 的单次
诊断反而 held=0/0/0、RMSE=117.2/101.2/81.3mm；同 CPU teacher env 间本身已有26–37mm
hand variation，不能把它解释为同-state验证。当前停止 query/model sweep，保留隔离
产物与失败 run，后续接触/preload 工作需 source-matched replay 和 contact-force proxy。

同一 source-matched CPU packet 上完成只读 contact-proxy audit
(`ref7_2-contact-proxy-audit-20261010-r14`)：R 三角色的 `surface_gap<=10 mm`
首段均从 tick43 开始，早于 teacher `pair` proxy 的 tick45；几何近段分别在
67/70/136 结束，随后 gap 为 15.87/14.07/11.76mm，而 teacher pair 仍为真。
全轨迹 hand RMSE 为61.43/75.83/77.40mm，近段退出时 wrist/finger误差为
5.56/17.01、6.03/24.72、30.99/51.09mm。source trajectory 与 source packet
字段逐项 bitwise 一致，执行 packet 没有 R-side pair、contact force 或 impulse，
故这些只是 gap/table-support 与 q/dq/action proxy，不能解释 preload 或宣称
contact-loss 机制；同-env q/dq/action 对比也不是 same-state 因果验证。
该审计只收紧了后续 source-matched replay 的观测需求，没有恢复 retarget gate，
不启动新的 R sweep、H-to-hand、PointWorld 或 evaluator 训练。

随后按该 Decision Note 完成了唯一一次 bounded source-matched native force
capture：`ref7_2-same-cpu-teacher-contact-20261010-r1` 与
`ref7_2-hand-action-context-contact-exec-20261010-r2`。两份数据均有限、543
帧状态/542 条命令、requested=applied 且零 clipping；source trajectory 与
packet 的 hand/object net-force、pair、state、action 字段逐项一致。新审计为
`outputs/consequence-evaluator/ref7_2-contact-force-audit-20261010-r2/`。

source teacher/env0 的阈值 pair 为真 498 帧（首个真帧 tick45）。三个 R 角色的
几何 near-gap 首段仍从 tick43 开始，但 R-side force-pair 只持续 23/27/93 帧，
分别在 tick66/70/136 变假；几何 near-gap 首段在 tick68/71/137 才采样退出，
而此时 R hand/object net-force 已为零、source pair 仍为真。这只说明 live R 与
teacher source 的观测时序不同，不能构成 same-state 因果比较。

接触字段是五个 configured hand bodies 的 net force 与可能包含非手部接触的
target net force，pair 只是 `hand norm>.1 any AND object norm>.1` 代理；reset
frame0 标为无效。它没有恢复碰撞 pair、法向力、冲量或 preload，也不改变
`ref7_2` Probe `UNCLEAR` / native retarget gate `UNPROMISING` 的判断。当前仍
不启动 H-to-hand、PointWorld、evaluator 或 Cm 集成；相关运行代码的 future
anchor 广播维度修复和布尔一致性审计修复已由测试覆盖，待提交为工程修复。

## 2026-10-09 ref7_1 物体系 GT servo 与接触控制诊断

按用户授权持续自主推进两小时，完成16组4env/64copies/seed282原生GPU Probe，
每组542步，源码/运行manifest/失败packet完整保留。分支main，旧Y与全局Cm claim
保持；旧R、ACT/G/H->V训练仍暂停。所有16组独立执行审计PASS，18项合同测试通过。

历史tick160..176世界GT手误差7.60mm、手物相对误差65mm，失抓后相对462mm。
但物体系每步完整SE3反馈会放大运动：命令臂held105、物体抬升峰6.63m。围绕
world nominal有界20mm/.15rad修正两次held484/483，无裁剪/中途失抓，仅说明
保留source PD预载的命令级控制PROMISING。固定世界命令fresh结果31..484，
角色即使执行同一绝对目标流，接触后也能held483对250；不存在严格same-state因果结论。

11点fixed-root解析腕逆解位置误差<1um。GPU审计coupled12 Jacobian满秩，但
实际18关节rank17、受载关节偏离理想耦合；静态coupled逆解thumb tip约5mm误差。
校准train3前40帧的两系数一步PD逆控制，只读当前q/dq与未来几何。保留source
finger命令时，冷启动腕控制两次held483，worldRMSE2.91/3.77mm，无裁剪；腕部信号
PROMISING。完整11点控制首轮held485却live teacher17未过行为门槛，重复238，
因此完整几何gate仍未通过。源未来q/命令污染的实际分支依赖测试通过，不是输入泄漏。

最终将finger目标换成特权实测next-q，未加/加因果过去PD负载EMA均held0，手误差
3.57/3.61mm、378/391次裁剪。该简单逆控制负信号有效，不能升级为几何表示无效。
离线接触阶段审计发现free-approach模型食指目标误差由.013rad升至.578rad，
它不是精确接触动力学逆解。当前完整路线UNCLEAR，关键缺口是finger几何/实际受载姿态与commanded PD
目标/接触预载的映射。下一最小问题：固定解析腕部，单独检验finger命令解码与
当前机械/手物状态；保留geometry-only和past-command对照。未重训旧R或推进H->V。

一次teacher被旧代理指标记为放回（held305/settled151），已核对桌面末态但未
迁移新控制器；不是完整任务可靠性证据。所有重复角色、不同布局及弱teacher运行
均保持原始gate，不计为独立seed或Cm增益。产物新增约0.35GB，GPU2已释放。
证据：[物体系GT上界Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-object-relative-gt-servo.md)，
汇总`outputs/consequence-evaluator/object-relative-gt-campaign-20261009-r1/`。

## 2026-10-09 ref7 GT-hand retarget/control upper-bound Probe

用户暂停 ACT 推理和 G(H,A)->hand 路线，改按 ref7 先做几何 action retargeter。
旧Y/U及全局Cm claim保持；尚未训练H->V或接PW/evaluator。main实现两层decoder，
输入GT24x11未来手位移+当前q18/dq18，预测相对查询状态的commanded PD targets。
显式intrinsic XYZ/原生DOF索引、6独立finger及逐horizon归一化；原生wrist为增量，
固定24步绝对目标每tick用机械q反馈转回控制，R仍仅每24步query。

复用完整身份的5条teacher持握轨迹（3train/1val/1test，390/130/130窗口）；
不是完整任务success数据。GPU2拟合2400步27.97秒，val选step1600，test相对位置
目标RMSE27.17mm，不能代替真实手跟踪。首次native preflight因描述性ring/pinky
名称颠倒中止，c28d9eb修复；数值索引/标签/权重不受影响，原checkpoint保留。

修复后4env/64copies/seed282完整542步81.48秒：teacher held484；直接GT PD目标
servo held117/手coordinate RMSE4.35mm；learned R held42/135.89mm，重复目标臂
held0/136.05mm。held>=90%teacher且RMSE<40mm门槛未过，所有完整任务success=false。
GT servo无裁剪，commanded target误差<=5.96e-8，仍在tick176后失抓；说明这次
固定source轨迹贴近不保证抓持。learned前120帧RMSE70.25mm且无裁剪，首次裁剪
tick265，不能把初期失败归因于后期349次finger裁剪。重复臂并非独立seed。

执行审计PASS：23次query，actual=requested bitwise；未来几何t+1、当前q/dq/手
对齐；只读几何和当前状态的checkpoint GPU重放误差6.11e-7；目标转换/重复流一致。
当前冻结learned inverse为UNPROMISING，整条几何retarget/control仍UNCLEAR，
GT command servo也未过持握gate，不能将全局负结果仅归因于R。下一优先级是直接
GT servo失抓窗口的接触/执行诊断及已有packet上的source/live-state inverse复核，
暂不扩大训练或推进H->V。13项相关测试通过，GPU2已释放，新输出<25MB。
证据：[GT-hand retargeter Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gt-hand-retargeter.md)。

## 2026-10-09 ACT newest-action receding1 follow-up

用户提出每次只执行一步。新增`receding1`每tick重新预测，执行最新chunk的horizon0，
不混合历史预测，区别于`temporal1`。冻结同一checkpoint/归一化、seed282和4env
native GPU合同，单次542步67.16秒：ACT最高抬升0m/held0、重复ACT held0；
teacher0.827m/held484，repeat held483。542次query、最多1个chunk，逐tick
`action[t]=proposal[t,0]` bitwise成立，native/requested差0，无early done。
14项相关测试通过，GPU2已释放。该冻结checkpoint的one-step remedy为UNPROMISING，
未解决ACT抓取失败，不据此确定唯一根因；下一步仍优先deployment-history诊断。
证据：[receding1 Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-receding1.md)，
产物`outputs/consequence-evaluator/act-receding1-20261009-r1/`。

## 2026-10-09 ref6 ACT inference diagnosis

冻结历史ACT checkpoint `ed26abd5`与其原归一化，不重训。`6751843`补齐按绝对tick
聚合的`overlap8`与`temporal1`，显式时间戳有效性、过期/reset和合法零动作保留。
四个seed282原生GPU group完整542步：open_loop24 held478/最高抬升0.595m，
receding8/overlap8/temporal1均held0；teacher分别held484/297/484/484。
前120步的8步边界wrist控制变化由33.56mm降到19.23/8.58mm，仍未恢复抓取。
四组actual-requested控制差均0，独立加权重建误差至多2.24e-8，query H因果对齐，
各自最多1/1/3/24个chunk、调用23/68/68/542次，重复ACT控制流一致。

真正stride8查询网格的四条val episode上，前8/后16 token MSE分别1.5994e-4/
1.6104e-4，没有前8平均误差特别差的证据；逐帧网格的7%差异不能用于解释receding8。
真实查询网格的同绝对时刻wrist预测差15.40mm，与teacher自然控制变化15.48mm
同量级。时间融合缺失已补齐，但当前checkpoint的抓取失败未解决：本次仅融合
remedy为UNPROMISING，根因仍UNCLEAR，不再默认把硬切换或部署分布说成唯一原因。
open_loop24仍只是抬升/持握基线，最终任务success=false，不能写成完整任务成功。

保留推理/审计工具与失败证据，下一优先级为deployment-history诊断，不扩融合
仿真、不在本轮重训或进入candidate/Cm/Gate1。严格same-state边界仍未改变。
13项相关测试通过，GPU2已释放。证据与命令见
[ACT时间融合Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-temporal-ensemble.md)，
产物`outputs/consequence-evaluator/act-temporal-ensemble-20261009-r1/`、
`act-temporal-offline-20261009-r4/`和`act-temporal-final-audit-20261009-r1/`。

## 2026-10-09 ref5 prospective hand execution bridge and rolling control

用户ref5取代继续扩fork-panel/C2a路线：普通e260随机干预rollout训练
`G(H,A)->24x11hand`，再用冻结PWbest46000和同C1step1000在线选7候选。
旧U32继续冻结，只作teacher；真实目标为任务成功和持续稳定，不是旧U排序。
本轮没有新fork、future-GT部署输入、actor/PW/C1训练或RL收益声明。

自训重建e260SHA8882fabd/s3/full-frame0采集96train293、64val294、64test295，
3360决策窗口；首次桥HA/H各2000步，val选step200，测试点RMSE63.05/63.10mm，
persistence116.73、nominalFK102.71mm。Zero-A活跃窗口MSE仅增加.2747%，
未过额外1%动作gate。仅改变固定native请求单位的匹配r2，用新64episode298测试，
HA/H67.45/67.29mm，zero-A变化-.0403%；量纲调整没有建立动作敏感性。
旧/新test不同，不能把两次RMSE作matched比较。总数据4320窗口，所有episode542步。

Root按ref5的实际任务目标缩小端到端Probe，保留未过动作gate，不用代理指标
宣布桥有效。只按val误差冻结原r1HA200(.008722vs.008818)，同C1三层普通单计划
waterfall teacherRMSE.34452/.39038/.38288，C2a/C2b与C1GT分数差.15058/.16290。
GT本身存在e260迁移误差，不能把控制结果全归因于G/PW。没有新same-H排序结果。

初始两seed296/297完整轨迹各臂bitwise重复，16env重复不能算32独立episode；
已保留并纠正记录。随后只修复评估支持：固定1mm/5mrad/10mrad初始q扰动，
匹配baseline/planner且不同seed实际轨迹不同。最终32episode/arm任务均0，
stable45baseline24/planner26，稳定后失抓1/8，any recovery9/14，干预0/55.97。
稳定抬升匹配IDrescue/harm7/5；完整任务0/0因全失败没有判别力，不是零收益证明。
baseline有23final-placement-not-settled、1中途丢失和8未稳定；planner18/8/6。

2d8c3ba实际路径通过初态actor/RMS/q/object匹配、评分argmax/请求计划、四帧
过去几何、PW重复性、零裁剪和完整542步检查。状态UNCLEAR，不扩百级episode。
代码首次实现无futureGT的G->PW->C1每8步真实replan，但任务收益未证明。
下一步是动作敏感/显式wrist rotation与相对手形预测、C1/source迁移和受控放回
能力边界的区分，避免继续改Y、同样加epoch或扩fork。未放弃核心Cm假设/claim。
14相关合同测试通过，GPU2已释放。实验卡：
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-hand-execution-bridge.md`；
最终结果：`outputs/consequence-evaluator/hand-execution-control-20261009-r2.json`。

## 2026-10-09 ref4_4 old-U evaluator and actual one-shot screen

用户明确冻结旧`U32=Y7+.25Y3-Y6`作为teacher，不再定义新Y，推进C0(H+A)、
C1(H+A+GT24)、C2(H+A+PW24)。contact仅生成32步标签，future只有24步几何。
C2按用户确认使用实测未来手轨迹作为PW条件，只替换物体未来；是离线oracle，
不能表述为可部署native-control planner。现存192完整扰动episode用于监督，
25x7恢复ref13候选整体不进入拟合/归一化/val选择。历史删除数据未被假装复用。

提交edccfe1完成三臂各1200步，val选800/1000/1200，耗时31.50秒。冻结same-H
候选排序C0/C1/C2为60.26%/75.64%/69.23%，C1 future shuffle为51.28%；
GT增益15.38pp、regret .07083→.05667，GT ranking screen PROMISING。
PW比GT低6.41pp，未过原retention screen。仅78严格pairs/5有效anchors，
不是正式支持或Gate1。PW best46000推理39.38秒；修复CUDA原子group-mean的
重复漂移，采用数学等价的确定性CSR求均值，冻结权重/源码不改；失败运行保留。

继续执行三条冻结one-shot真实90步混合路径（各约40秒）：baseline20/25，
C019/25（harm1），C120/25（rescue0/harm0），C221/25（rescue1/harm0）。
所有评分/执行H exact、几何prefix exact、原prefix合同、实际计划、零裁剪和完整Z90
检查通过；saved scores GPU重放误差0。C2 gain CI含0，控制收益UNCLEAR。
这不是旧teacher的三次replan23/25，也不是完整受控放回或RL学习收益。

同一C1权重直接换PW future的post-freeze离线诊断为73.08%，优于独立拟合C2的
69.23%；未用这个诊断改selector或补跑第四控制臂。因此不能把6.41pp差距全归因于
PW精度。下一步保留旧U和共同C1，在新独立候选/cohort上冻结比较GT/PW信息保留；
不自动加epoch、重训PW、定义新Y或进入可部署planner/RL。Rolling仍属后续证据。
GPU2已释放。实验卡：
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-old-utility-evaluator.md`；
产物：`outputs/consequence-evaluator/old-utility-evaluator-20261009-r1/`。

## 2026-10-09 ref4_3 learned temporal evaluator

按用户ref4_3授权恢复 learned `Q(H,A,Zgt)` 路线，先用最终任务结果和
`±24/(T-t)`作弱监督，不再直接规定value为TCC/reference delta-progress。
用户明确：最终重新抓稳并受控放回的episode为success；其局部失抓/下落/recovery
窗口mask，重新稳定抓持后恢复正监督。物理阈值只用于binary outcome/mask审计。

复用192完整episode，重新标注train41/23、val38/26、test37/27（成功/失败），
其中test一条恢复成功轨迹由failure改为success；成功recovery窗口mask共48个。
提交5d13cf1在GPU2完成匹配HA/HAZ各1200步，26.66秒、显存约561MiB，
val均选step400。冻结test同时间跨episode排序88.42%/89.67%，future打乱后84.32%；
future gain1.24pp未达预设3pp screen，保持UNCLEAR，不进入完整任务GT control/Gate1。

关键数据限制：训练3968窗口仅128含非零请求计划（3.23%），没有成功recovery训练样例。
模型利用future的信号存在，但这还是结局预测，未验证同H候选排序的任务收益。
旧ref13七候选缓存的25锚点评分审计只用于检查迁移：HA全选grip+，HAZ改变7个选择；
候选A有72/3024个计划元素越出训练坐标范围，涉及5/7个候选臂；分布覆盖不足，
不能解释为控制成功或失败。
下一步信息应来自决策窗口、同H候选和恢复样例，而非延长训练或提前接PointWorld。

提交d02f45b补齐确认失抓前5帧的mask边界，10项相关测试通过。重新准备r2后，
所有数组及windows SHA与r1完全一致，模型/目标代码未变；因此保留冻结权重和测试，
不重复相同训练。原始运行、修正规则与等价审计都保留。GPU2已释放。
实验卡：`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-weak-temporal-value.md`；
产物：`outputs/consequence-evaluator/weak-temporal-value-20261009-r1/`。

## 2026-10-09 ref13 whole-world replay recovery

按用户要求重新检查历史 ref13 方法。近期 host/group 诊断的 Torch2.0.1、强制
frame0/hybrid probability1 与跨 env broadcast 控制，不等价于原先 Torch2.4.1、
hybrid probability0.5、每个 env 自身反馈、跨 fresh world 对应 env 的重放。
因此下文的 host 关闭判断仅适用于那些具体实现，不能排除完整的旧 ref13 方法。

本次恢复 Torch2.4.1+cu121、96env、GPU PhysX/CPU tensor pipeline、单线程、
seed263 和原 r7 配置。原 checkpoint SHA16fd261b 与完整 panel 未找回；使用固定
的自训重建 e260 SHA8882fabd 和目前保留的三条转换 motion，不宣称原权重复现。
baseline 采到45锚点，fresh zero repeat 的45/45均通过旧筛选，记录的 before、
history、actor observation、control、PD target、height/pair 完全一致，Z90均33/45。
repeat未保存完整trace，不能把这些字段的一致升级为所有隐藏状态/几何的exact证明。
当前只恢复了旧执行筛选，不代表正式Gate1通过。

冻结的current-only同步s3组为tick71的25个env，重采baseline Z90为20/25；
reanchor的记录前缀误差为0。七候选与每条实际滚动路径都已完成：one-shot
baseline/原U/新P[t+24]-P[t]为20/25、21/25、22/25；三次真实8步决策后分别为
20/25、23/25、22/25。新Y对baseline救回2/harm0，原U滚动救回3/harm0；新Y
继续重规划没有增加成功，少于原U滚动一个救回。所有新增成功都在frame0组，
不是hybrid中途初始化造成：该17env组baseline12、原U滚动15、新Y14；其余8env
各臂均成功。首轮新旧选择21/25不同，原U选3次非baseline、新Y选22次。

35候选panel与6真实mixed执行均通过旧full-world前缀合同，候选评分所用的object/
11hand-point query前轨迹bitwise一致，初始动作无裁剪且实际进入运动。冻结bank/
TCC、actor、Y和Z判据没有调参。工程旧方法可继续使用；新Y one-shot有小幅局部
正向信号，但不能视为优于原U滚动或正式Gate1通过。结果保持UNCLEAR：单seed、
单motion、25锚点，收益CI下界为0，且只观察原Z90而非完整episode。本轮不扩跑、
不训练evaluator/PointWorld。下一步可在恢复后的合同中预先固定selector，做完整
episode的new-Y/baseline最小对照；更广reference bank和cohort是后续证据。
条件实验约2231秒、产物约1.37GiB，均在预算内；GPU2已释放，17个相关测试通过。
实验卡：`src/task/cm-interaction-oracle/docs/experiments/probes/P-20261009-ref13-progress-recovery.md`；
产物：`outputs/cm-interaction-oracle/ref13-progress-recovery-20261009-r3/`。

## 2026-10-09 GPU synchronous group Value-noise follow-up

在提交`4258adf`中修正了同步GPU group的角色合同：baseline、zero、positive、
negative现在由显式`role_index_map`绑定，`candidate_calibration_valid`不再在
reactive packet失败时无条件为true；runner和离线审计同时记录p99、峰值tick和首次
非零tick，避免接触脉冲被展平p95掩盖。旧r18的非默认zero pair没有角色元数据，新的
审计会拒绝它，之前按env0 baseline解释的r18 Value摘要不再采用。

按用户建议在原生`gpu_physx_gpu_pipeline`中做了两个4-env、seed282、query48、72-step
工程Probe。默认布局`zero=[0,1], candidate=[2,3]`的短窗baseline最高抬升0.2585m、
held15，物理bank Y为`[.043592,.043215,.022880,.024176]`，zero-pair标量中位噪声
`.000378`，正/负对baseline为`-.020713/-.019416`。交换布局
`zero=[2,3], candidate=[0,1]`的短窗baseline为0.1530m/held9，按角色映射后的Y为
baseline`.035543`、positive`.039070`、negative`.020899`，zero中位噪声`.002624`，
正/负对baseline为`+.003527/-.014644`。positive对比随角色布局变号且量级明显变化，
因此不能视为稳定的candidate effect；两包都保持engineering-only，没有进入Gate1。

两次zero pair的raw contact-force差异都在tick44首次出现；flattened p95可为0，但峰值
约8.1/26.2，说明p95-only gate不足。结论：关闭这条role-permutation scalar-Y
noise route，保留原生GPU group作为行为容器；不扩rolling Probe、不fit evaluator、
不进入PointWorld，不改Y、reference bank、TCC或policy weights。实验卡与产物：
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gpu-group-value-noise.md`、
`outputs/consequence-evaluator/gpu-group-value-noise-audit-20261009-r1/`和
`outputs/consequence-evaluator/gpu-group-noise-audit-20261009-r1/`。

随后做了一个仅改变`physx.num_threads=1`的原生GPU pipeline Probe（提交`98cec12`），
仍为GPU PhysX、GPU tensor、GPU actor、seed282、4-env、72步。baseline最高抬升
0.2740m/held13，object zero p95为`8.36e-4m`；raw contact差异仍在tick44首次
出现，contact p95为0但max约19.5。物理bank Y为
`[-.011617,.011970,-.016655,.015730]`，zero标量噪声`.023587`，噪声大于候选
对比并把negative误选为最高。单线程调度没有恢复执行合同，关闭该backend变体，不跑
full542确认；strict Gate1、evaluator、PointWorld和policy/reference/Y仍保持冻结。

## 2026-10-08 Gate1 首轮完成：工程通过，任务收益 UNCLEAR

64522f1在CPU-PhysX/GPU2模型执行1个paired episode（seed282），314.94秒完成。
12候选、重复零分支、4次chosen8step混合prefix及全542步rolling均通过严格
canonical/RNG/控制/观测重放。tick48选positive，Y0.031013对baseline0.018941；
其余3次维持baseline。独立成功baseline0/1、rolling0/1、rescue0、harm0；
最高抬升0.74cm/2.48cm、held0，未过Gate1，不训练evaluator。

关键限制：CPU baseline未能抓起，而同seed/同权重的GPU预检连续hold483步，
最高抬升81.51cm；首步控制几乎相同，首步物理/观测即开始偏离。CPU后端解决
严格重放，却改变策略表现，不能据此否定原GPU策略或新Y。保留原e260的
64env历史56/64 lift、24/64 hold45、18/64 hold后无laterdrop结果（不同指标）。
单样本GPU推理异常已由d603153修复；GPU接触force/history重放仍待解决，
不放宽容差。下一步恢复代表性GPU执行与同状态GT合同，暂不扩大CPU seed。
GPU2已释放。完整结果、实验动机与限制见
[Gate1卡](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-gate1-gt-progress.md)。

随后按历史 ref13 合同加入并验证 `gpu_physx_cpu_pipeline` backend：GPU PhysX、
CPU tensor pipeline、`num_threads=1`、GPU actor。seed282 的 fresh72-step baseline
与 fresh48+24 zero candidate 在73个状态、73个RNG、72个control及contact/history
字段上全部 exact；但单环境 host full baseline 只抬升15.49cm/held8，仍远低于
匹配 GPU pipeline 的81.51cm/held483，因此没有进入 Gate1，也不能把该差异归因于
Y。详见 host audit
`outputs/consequence-evaluator/gate1-host-engineering-20261008-r3/host-backend-audit.json`。
当前 Decision Note：停止单环境 host Gate1，下一步在相同 host 合同下做一次同步
多环境 group replay，检查历史 ref13 的 solver 排程是否是行为保持所需条件。

同步 group probe 已完成：4-env host full542 最高7.91cm/held2；按 ref13 主组规模
96-env 的 bounded72-step 最高7.38cm/held3，均未恢复原 GPU baseline 的81.51cm/483
held。两组都保持同一控制流，但不同 env 间初始/动力学字段并非 bitwise exact，
tick44 contact-force drift 约5.7e-5。结论是 host 单环境与同步 group 都不能承载
当前策略的 Gate1，同步路线关闭；不跑正式 Gate1、不扩 CPU seed、不改 Y/标签/权重。
完整 group packet 与 mismatch audit 保留在
`outputs/consequence-evaluator/gate1-host-group-engineering-20261009-r6/` 和
`gate1-host-group-engineering-20261009-r7/`。下一步只保留原 GPU 行为下的执行合同诊断
或新的可证明重放设计。

作为对照，同一4-env synchronous harness切回原 GPU PhysX/GPU tensor pipeline 后，
baseline 恢复到最高81.28cm、held480、first stable tick107，说明 group 排程本身
没有毁掉原策略；但初始 state/history 已不同，tick44 contact-stage 仍不满足严格
same-state contract。因此保留该行为对照，但不放宽 exact replay，也不进入 Gate1。
审计见 `outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r1/`。

随后在原生 GPU PhysX/GPU pipeline 内完成了4-env同步 zero-pair 噪声校准：env0
baseline、env1 zero repeat、env2/3在t=48 transition起的24个action tick分别加入
正/负固定残差，所有角色在query前共享env0控制。四次fresh full-baseline分别为81.28cm/480、82.39cm/484、
75.35cm/484、13.01cm/7，说明reactive policy的跨启动行为本身有大幅波动。最新
r4的24步候选效应相对同窗口zero displacement p95比值在object/hand/q位置约
1.34--2.87，但q速度、history和contact多为约0.36--1.18；没有形成一致的
candidate\(\gg\)solver-noise margin。该组只作为工程校准，不是正式Gate1或候选
价值证据。审计见
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r4/`
及实验卡的 Native GPU group noise calibration 小节。

当前不跑正式Gate1、不fit evaluator、不扩CPU/host seed，也不改Y、reference bank
或policy weights。runner已补上flattened native tensor按env reshape、q/dq/contact
字段、初始semantic role guard（不一致即停止校准）及query-relative noise ratio；
当前4-env只有一个zero pair，candidate env2/3各自的query前漂移仍未被独立配对。
下一步在新执行合同前
先评估冻结ACT-like 24-step proposal baseline，或提出新的native GPU replay contract。

随后补做原生GPU group完整行为 gate：4-env、542 steps、seed282、同一
`gpu_physx_gpu_pipeline`。env0 baseline达到0.799920m、first stable tick106、
连续held481帧并保持到终点，行为保持 gate 通过；env0/env1控制完全相同，初始
object/q/dq/hand/gap/history/contact semantic gap全为0。状态仍非exact twin：
hidden hash首帧不同，derived geometry从tick1漂移，tick44附近contact/support
差异放大。候选相对zero displacement的后查询p95比值仍不一致（object pose
0.55/1.49、history0.68/0.70正/负示例），因此该结果仅是engineering/noise
calibration，不能进入GT scoring。完整产物见
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r9/`。
原生GPU group保留为行为容器；exact same-state Gate1仍关闭，下一步落实冻结
ACT-like native-action chunk，再在该容器中重测。

## 2026-10-09 ACT-like native action chunk engineering

新增 `consequence_evaluator.action_chunk` 和
`tools/run/train_action_chunk.py`：输入 raw `history[t]`，一次输出未来24个
18-D native executed controls；目标取 `pre_physics_step` 捕获的 `action`，不取
`residual_plan`，不生成hand-flow。合同测试5项通过。labeled hold-audit中21条
同一`s3_airplane_lift`/`airplane_base` clean `expert_success`仅作engineering
数据（manifest仍`training_allowed=false`，显式`--allow-audit-only`，输出强制
`engineering_only`）；冻结当前checkpoint `running_mean_std`。17/4 episode holdout
的stride8 fit把chunk MSE从均值基线9.38e-4降到1.61e-4，first-action MAE0.01090，
只构成action-space PROMISING信号。

GPU group 542-step行为 screen（seed282）将reactive teacher、ACT、reactive repeat、
ACT repeat放入同一原生GPU进程。`open_loop24`中ACT最高0.7939m/held478，teacher
0.8134m/held483，均完成受控末端几何；23个chunk均一次生成后原样送入native
pre-physics。产物
`outputs/consequence-evaluator/act-native-chunk-engineering-20261009-r2/`。
但最终所需`receding8`在同一checkpoint下held0、未抬升；将21条全部用于fit把离线
MSE进一步降到4.53e-5仍然held0（r5），说明降低该离线拟合误差不足以恢复行为；
当时尚未区分chunk硬切换与部署分布误差。后续ref6推理诊断见
[时间融合Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-temporal-ensemble.md)。r3/r5均仅engineering，
不进入GT scoring。

因此当前只保留`open_loop24`作为ACT-like行为基线，关闭该直接模仿checkpoint的
4--8步receding route；Y、reference bank和policy weights保持不变。下一步可在
query处用带prefix provenance的记录nominal chunk测候选效应，或先采集专门的
reactive deployment distribution后再训练route-conditioned proposal。详见
[ACT chunk Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-native-chunk.md)。

随后补做了单环境 native GPU `open_loop24` 行为 screen（seed282、72步、tick48
一次生成24步chunk）。runner在`706f0a2`补齐了proposal manifest认证、实际/请求
control逐tick审计、prefix/RNG provenance和engineering-only边界。修正后的r2 packet
`outputs/consequence-evaluator/gate1-act-single-open-loop-20261009-r2/`控制合同完全通过
（actual-requested最大差`0.0`、proposal只调用1次、无未来observation feedback），
但对象72步最高抬升`0.0m`、held`0`，终点高度`-0.001388m`，未过预注册的`0.20m`
行为阈值。相同seed/backend的reactive-only 72-step fresh control
`outputs/consequence-evaluator/gate1-act-single-reactive-20261009-r1/`达到`0.2915m`、
held`13`；不过两次fresh process在tick44起contact/history和state/RNG hash已分叉，
不能把它们当作same-state ACT对照。首次r1因requested-control shape广播bug在运行时中止，
已保留并标为实现错误。当前单环境 reactive-prefix+frozen-chunk 执行合同关闭，不跑
serial ACT cluster；结果保持UNCLEAR engineering evidence，不升级为策略、GT value或
Gate1负结论，reference bank/TCC/Y和policy weights保持不变。详见
[单环境ACT Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-single-open-loop-behavior.md)。

在该 screen 之后做了只读的 Isaac Gym API feasibility audit（无 simulator、无 GPU）。
归档 Torch2.0.1 runtime 的 `gymapi.Gym` filtered surface 只有 actor root/DOF/rigid-body
state、net contact-force/force-sensor tensor、rigid-contact query 和 sim rigid-body
getter/setter；没有 contact-manifold、warm-start、solver-island、hidden-cache 或
serialize/snapshot/restore API。Task twin contract 的 `SOLVER_CONTRACT` 也固定为
`fresh_simulator_prefix_replay`，拒绝 warm PhysX restore；公开 tensor 加 Python/NumPy/
Torch RNG restore 不能冒充 hidden-state twin。审计产物在
`outputs/consequence-evaluator/hidden-physx-api-audit-20261009-r1/audit.json`，卡片见
[PhysX state API audit](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-hidden-physx-state-api-audit.md)。
因此当前没有值得重复的 synchronous group、serial noise、CUDA synchronisation 或
deployment-fit 仿真；严格 Gate1 blocker 已收窄为需要新的可验证执行合同，或在
Decision Checkpoint 明确改变 Gate1 claim。

后续 candidate engineering probe 已保持原生 GPU PhysX/GPU pipeline，并把四个
env的 tick0--47 controls 统一为同一已成功 packet，tick48--71 广播一个冻结 chunk，
再只给 env2/env3 加正/负残差。学习 proposal 的 r6--r8 在 reactive query state
上没有保住 nominal grasp（最高约0--0.012m），且 zero pair 在 query 前已经出现
约0.0786m object drift、20级 q velocity 和26--32级 history drift；这不是可解释的
candidate effect。修正原生 RMS `sqrt(var+1e-5)` 与 `[-5,5]` clipping 后的拟合和
542-step screen 保存在 `act-native-chunk-engineering-20261009-r10/r11/`，ACT
仍未抬升。用成功 r9 reactive prefix 的 recorded nominal chunk 重放（r15）最高
0.240m、held11，zero-pair object displacement p95仍约0.338m；显式
`zero_pair_gate` 和 nominal behavior gate 均失败。r6--r15 全部保持
`engineering_only`，不进入GT scoring；candidate effect不作价值证据。下一步先
解决 recorded prefix/chunk 的行为与zero-pair gate，再决定是否收集部署分布重训，
不跑正式Gate1、不fit evaluator、不改Y/标签/权重。

随后用空闲GPU2完成r8的72-step runner guard validation（18.2秒）：初始
object/q/dq/hand/gap/history/contact semantic gap全为0，env0/env1 control exact，
且state packet确实包含tick49--72的完整candidate窗口；但hidden world-frame hash在
tick0即不同，之后仍有contact/trajectory drift。短窗口baseline最高24.17cm、held10，
不能判断full-episode抓取行为。r6/r7仅是native worker相对路径保存错误，已修复且
不产生仿真结论；r8审计见
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r8/`。

随后做了16-env native GPU multi-zero follow-up。r17的72-step nominal group只达到
0.2536m/held10；r18用同一16-env prefix录制的reactive nominal chunk，在选定zero
pair env1/env4下达到0.2743m/12和0.2617m/10，positive/negative约0.253m/13与
0.259m/9。行为gate在72步没有评估为full542，`candidate_calibration_valid=false`，
所以这些数不能称baseline保持或candidate成功。选定pair的query前p95较小
（object1.83e-5、hand3.29e-6、q位置8.89e-6、q速度1.43e-3、history3.91e-5），
但max/contact仍有尖峰。runner随后改成同一个common pair的p95汇总，并要求至少
80%的zero pairs同时通过各字段阈值；离线重算r18为62/91=68.1%，multi-zero gate
失败，只作诊断。当前pairwise仍只含query前统计，尚未形成所有zero pair的post-query
displacement noise floor；因此不跑正式Gate1、不fit evaluator、不改Y/reference bank/
policy weights。完整packet见
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r17/`和
`...-r18/`，实现提交为`429d9ff`（common-pair p95为`e4a97d6`）。

按用户建议再做了当前代码下的4-env复核：r19原生GPU group完整542步恢复了行为，
env0达到0.8262m、held481，behavior gate通过；但env0/env1 query前p95在hand、
q位置、q速度和object velocity仍超过robust阈值。用r19 prefix和tick48--71
recorded nominal chunk做candidate r21时，fresh72-step nominal只达到0.2607m/held11，
behavior gate未评估且`candidate_calibration_valid=false`。positive/negative相对
selected zero displacement的p95比值在object pose为2.52/1.10、hand为2.06/1.33、
q位置为5.20/1.72，但q速度为1.07/0.79、history为1.14/0.69，contact也低于1，
effect margin gate失败。结论是4-env process可以保留full reactive behavior，但不能
把fresh candidate window当成稳定nominal或GT候选证据；下一步若继续，只补齐所有zero
role的post-query displacement分布，或改造replay contract。r19/r21 packet分别在
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r19/`和
`...-r21/`，仍不跑正式Gate1或evaluator fit。

随后新增只读审计入口`tools/audit/audit_gpu_group_gt_values.py`，用冻结的物理
reference bank/TCC encoder复算r19/r21的24步GT progress。两次progress-start
range都为4.90e-6，超过严格same-state的1e-9；r19四角色Y为
0.041369/0.042313/0.017920/0.041562，r21为0.042281/0.044033/0.040786/0.016099。
两次raw argmax都是zero-repeat，按固定0.01 deadzone都选择baseline。审计只作为
engineering contract evidence，不进入strict score worker或Gate1 utility claim；输出
在`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r22/`，工具提交
为`10ae238`。

随后补齐了同步GPU group的query-relative post-query noise审计。只读工具
`src/task/consequence-evaluator/tools/audit/audit_gpu_group_noise.py`（提交
`110cd13`）拒绝非candidate-role packet，检查所有role的prefix controls/done、状态
时间轴和zero-role集合，并对每个zero pair报告query-relative displacement。r18的14个
nominal role共有91个zero pair：object pose的post-query displacement p50/p90/p95为
0.2522/0.5375/0.5645m，q velocity为5.47/9.85/9.87，history为0.853/1.424/1.524。
候选增量相对所有zero pair median p95的描述性比值只有约1--2倍（object pose正/负
p95为1.96/1.74，history为1.55/1.81）；r21只有一个zero pair，object pose正/负
为2.02/0.86、history为1.21/0.87。该比值口径与runner selected-pair effect-margin
不同，不能称为candidate≫noise，也没有形成Gate1证据。审计输出在
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r24/`。

对保留完整hold的r19再做同一审计（r26）：zero-pair object pose displacement p95为
0.1195m，正/负candidate增量比为4.28/1.95；但history为1.42/0.81、q velocity为
1.27/0.93，且current state仍不exact。因此即使成功hold的单次launch也没有跨字段的
candidate≫noise分离，不能作为Gate1候选排序。

为隔离group-size actor GEMM的可能影响，曾在提交`949a163`临时把group actor改成
固定env0/64-row推理后广播；r25使用原生GPU PhysX/GPU pipeline、seed282、4 env、
542 steps，但baseline最高抬升只有0.0473m/held4，而旧r19为0.8262m/held481，
behavior gate失败。该改动已由`b0c4b33`/`ecc4def`回退；r25只作为负的工程Probe，
不能解释旧r17/r19差异，也不能作为新的执行backend。当前保持reference bank/TCC、Y、
policy weights不变，不跑正式Gate1、不fit evaluator、不进PointWorld。下一步需要真正
的同一隐藏状态fork，或有足够重复数的baseline/candidate统计设计；同步group仍只作
工程容器。

在旧group actor合同下又做了一个seed283、4-env、542-step复核（r27）。该次env0最高
抬升仅0.0891m/held5，negative role为0.5663m/398；zero pair的post-query object
pose displacement p95为0.8247m，正/负candidate-vs-zero incremental比值为0.82/1.29，
history为1.11/1.68。它与旧r19的0.8262m/held481形成鲜明差异，说明原生GPU接触执行
分布在fresh process/seed间本身不稳定，candidate effect没有超过这种运行级波动。r27
packet和noise audit在`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r27/`；
不把r27或r19升级为Gate1、GT utility、evaluator或PointWorld证据。

用冻结physical-reference/TCC bank复算r19、r21、r27的GT progress后，r27四角色Y为
0.016152/−0.000431/0.020753/0.036116，progress-start range为1.72e-6；raw argmax和
固定0.01 deadzone都选择negative residual（相对baseline +0.019965），而r19/r21都选
baseline。三组都超过1e-9 strict same-state容差，因此该choice flip只说明执行敏感性，
不是GT utility或candidate成功。审计输出在
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r28/`。

## 2026-10-08 ref4_2 成功机器人 reference bank

用户ref4_2明确替换原始运动作为唯一value时间轴的选择，先用已有source230成功
机器人rollout bank检验物理执行gap。340b847完成固定8参考、1000步TCC、同8条
source261开发标签对照，未调temperature/max_step/epsilon。四条clean tick176
Y由−0.0233..−0.0251变为+0.0432..+0.0442；负窗口5/6/6/8→全0，累计回退
0.249..0.368→0..0.002428，全部既定开发检查通过。Probe **PROMISING**，保留
物理reference路线；不是正式任务价值/策略收益结论。旧原始R结果保留。
失败e1末尾P0.0233；其tick51→75窗口仍在抬升，严重落下在之后，不能用later
failure污染局部Y；窗口内已开始分离，延迟风险和OOD仍待同状态候选排序验证。
GPU1完成训练10.12秒/标签27.38秒并释放，44测试通过。下一步按完整链路做
fresh simulator完整prefix replay的GT-value Gate1；当前未运行Gate1，未fit
evaluator，标签training_allowed=false。结果与限制见
[物理reference bank Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-physical-reference-bank.md)。

## 2026-10-08 Reference-conditioned delta-progress 监督

用户ref4_1替换airplane-specific S/P/M：Y=P[t+24]-P[t]，P来自原始成功参考
与实际3D历史的因果clip匹配；不是episode的t/T。用户确认reference使用原始
成功动作，按Inspire URDF重建同定义11机器人手点。复用固定Google XIRL的
距离/因果context helper，增加允许前进、停滞、回退的有界历史prior；不使用
全episode OT、反向cycle或窗口之后的恢复。旧数据/标签不改，S恢复问题不再
阻塞新路线。六项因果/重复阶段/停滞/回退工程测试通过，先做train-only8条
离线标签Probe再判断是否训练；未启动evaluator。当前在main开发，不push。
[标签Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-reference-progress-labels.md)。
几何标签r1/r2已完成8条/496窗口，因果截断误差0；修复早期prior支路丢弃后，
四条clean最终P仍仅0.262..0.437且有较多回退，UNCLEAR/不可训练。固定R不变，
转向[TCC对齐编码器Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-tcc-phase-alignment.md)，
只用source230train拟合geometry embedding，source261仍作已暴露开发检查。
TCC在ef9a770完成固定1000步（GPU1/10.84秒），8条496窗口标签重放9.69秒。
四条clean末尾P提高到0.932..0.994；self/static/因果检查通过，但累计回退
0.249..0.368均超过预设0.2门槛，局部Y仍UNCLEAR。再次抬升与末尾放回附近
出现负Y；固定checkpoint和标签保留、training_allowed=false，GPU1释放。
下一步核查已有局部阶段证据，不以低cycle loss或末尾P代替候选排序正确性。
用户完整链路要求标签过后先直接GT-value同状态滚动Gate1，不先训练evaluator；
允许fresh simulator完整prefix replay，禁止mid-state PhysX restore。Gate1成功
定义为完成稳定抓取并正常放回，允许中途恢复；中途失抓另统计。旧32anchor脚本
仍在但原输出目录缺失；旧23/32→27/32不能替代新Y验证。未运行新Gate1。

## 2026-10-08 DExplore 官方数据生成器筛查

用户授权尝试归档官方策略。官方 `inspire.pth` SHA8f6823db… 已核对并跑通；
可用完整运行时为 `.runtime_envs/dexplore_v120_train`（Torch2.0.1+cu118、
rl_games1.1.4），conda Torch2.2.2 环境存在依赖/扩展ABI缺口。采用进程内旧版
factory、单次外部RMS及严格compiled-key适配，不修改外部权重或环境。
同s3 airplane输入/seed239各64条完整episode：官方lift5/hold45为64/64、64/64，
自训为56/64、24/64；保持后直到episode结束不掉落为官方0/64、自训18/64。
原reference末帧回到初始高度+0.5mm，完整片段包含放回，与全程保持的S标签
存在任务终点冲突。源数据定义的放回前tick481仅作事后prefix诊断（63/64 vs18/64），
不能替代原门槛或新固定reference评估。用户明确选择完整reference并把受控正常放回
计为Task成功；新几何标签另行验证，不裁剪reference、不把落桌都计为成功。未启动fit，
官方权重仅为数据生成候选，不替代Mission的自训练最终策略。评估已结束且GPU释放。
记录：[官方生成器Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-official-generator-screen.md)。

按用户的完整reference/受控放回定义，9ced94d在GPU2完成64条nominal连续episode，
115.057秒、约140.53MiB；S/P/M几何弱标签为62成功/2失败、4160窗口、256偏好。
62条正例末尾均已离手且在桌面proxy上静止；两条负例在放回阶段失去几何控制/支撑。
这是自动数值核查，未人工确认视频或精确collision pair。独立value schema不放宽旧
H匹配验证合同；只有train seed230且负例不足，labeled `training_allowed=false`。
未开始evaluator fit，采集进程已退出/GPU2释放。下一步为保留clean的阶段定向扰动
Probe，再考虑独立held-out seed组，不能用nominal零计划声称动作收益。
记录：[完整参考value数据Pilot](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-full-reference-value-data.md)。

后续已完成经验placing、同H策略误差holding/contact及0.5dose采集。0.2contact
三组59/5、64/0、63/1，独立负例不足，原样保留且未fit。新的固定0.5dose（平移仍
0.05，原生实际控制仍[-1,1]）在98ceb76完成192条完整episode：train41/23、val38/26、
test36/28；11904窗口/768偏好、来源组261/262/263互斥，机械数据门槛通过。
官方/自训同H动作投影复现最大误差5.96e-7；所有96计划真实触发，实际命令符合
裁剪后请求，逐episode物理标签重算通过。输出约421MiB raw+69MiB窗口，GPU2释放。
仅生成数据，旧local-event trainer仍不接受新schema。样例出现“中途失抓后恢复放回”
的语义边界，已向用户提问，当前S会判失败；原标签保留，未启动evaluator fit。
历史下一步原为明确恢复成功定义再适配S/P/M训练；当前已被上述ref4_1路线替代。
记录：[value强度校准](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-value-dose-calibration.md)。

## 2026-10-08 HOCap 外部测试数据获取

用户指定 HOCap 为外部测试候选。官方 Box 的 calibration/models/poses 三包已下载，
SHA256 与第三方 HF 镜像对应 LFS 身份一致，ZIP CRC 校验/解压通过；通过 HTTP
Range/ZIP64 取得全部64条序列元数据及一相机连续32帧标签，不下载整套 RGB-D。
9主体/64序列/72944帧/64物体；shape/mesh引用/手侧/-1缺失值/四元数检查通过。
样本128次物体-帧比较中，相机标签转世界与原生pose平移最大差4.57e-8m，
仅证明发布数据内部坐标一致。新增磁盘约246MiB，全部获取任务已结束，未用GPU。
输出：`outputs/cm-pointflow-effect-pretrain/hocap-test-preflight-20261008-r1/`。
保留test-only：不加入训练、归一化或checkpoint选择；原始FPS/时钟仍未核实，
`test_ready=false`，尚未转换为30Hz H4/K24正式测试或运行模型评价。
[官方合同与待确认项](../src/task/cm-pointflow-effect-pretrain/docs/research/HOCAP_TEST_PREFLIGHT.md)。

按用户要求，8f857ae已在GPU0完成固定latest50000的HOCap帧序列Probe，49.425秒。
全部64序列/9主体，128运动窗口第24帧点EPE17.805mm vs静止63.704mm（降低72.05%）；
64序列均衡near-hand自然窗口20.346 vs54.985mm。但其中17近静止窗口模型22.160mm
vs基线0.356mm，存在明显虚假运动；不能用总体均值掩盖。源码/数据1318项哈希通过，
手21点与32帧发布标签通过10微米门槛；3项合同测试通过。原始FPS仍未核实，
仅以数组帧索引H4/K24和nominal30/900特征尺度进行无拟合评估，UNCLEAR/test_ready=false。
未调参、未换checkpoint；进程已退出/GPU0释放。
[协议与分层误差](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261008-hocap-frame-generalization.md)。

同一HOCap冻结192窗口补测Oak-only latest10000与混合latest50000：
GPU1/a3139fb的r3在40.990秒完成，1323输入哈希通过，混合版复现此前结果，
全部进程退出。128运动窗口h24点EPE22.421→17.805mm（降低20.59%）；
自然64窗口24.675→20.346mm（降低17.55%），但其中17近静止窗口
8.039→22.160mm（2.756倍），虚假运动加重。保留运动收益与近静止退化分别汇报。
两版相同模型/归一化；混合版额外64250更新及loss/batch变化，不能把差异单独
归因于混合数据。原始FPS仍未知，仅帧索引Probe/UNCLEAR，不证明Cm策略收益。
两次工程复现阈值失败原样保留；GPU亚微米数值波动修正为1微米/1微弧度复现检查，
没有改预测、拟合、调参或选择checkpoint；5项合同/回放测试及verify通过。

## 2026-10-08 迁入 main 开发

用户授权将 consequence-evaluator 的已提交历史和当前用户/并行 agent 修改一起
提交并合并到本地 main，后续直接在 main 开发；不推送远程。
合并前 main 是本分支的祖先，可快进保留完整历史；实验卡中的 branch/git_commit
继续表示实际运行身份，不随当前开发分支更名。

## 2026-10-08 用户修改审查与当前 blocker

consequence-evaluator 用户新增的H匹配、最小唯一配对覆盖和重放状态合同保留。
修复repeat验收遗漏未来手/done、生产门槛误封只读审计、native物理枚举/dtype
序列化，以及pickle对象共享导致内容哈希在save/load后变化的问题。
六角色airplane/duck/cup资格36/8/63（各64条），mixed12/train5/balanced5为0/5/4；
生产route仍不可训练。216条旧连续episode经H匹配重新审计，唯一pair=1/1/1，
低于8/4/4；没有启动evaluator fit。独立native twin r4（a7d5146/GPU0）通过：
三fresh进程同前缀状态/RNG精确一致、repeat全部future/action/done/native状态一致，
请求/实际control L2分别0.75865/1.14996。它是engineering_only且不可训练，
不改变ref2 continuous schema或科学结论。用户明确保留连续rollout、不fork；
twin只作工程诊断。对全部可比较窗口穷举后，物理匹配候选4/1/3、加H仍为1/1/1，
配对选择器未漏掉合格episode pair。随后完成不可训练的连续保持阶段定向诊断，
核对合格飞机专家的局部反例和匹配覆盖，生产资格和状态阈值未放宽。
连续hold审计7389d05已完成48条episode（24clean/24hold，23实际触发），
有11403可靠progress帧，但仅1个train唯一pair。93dde07逐帧审计17053事件窗口，
463事件可比较episode pair仍仅1个物理/H匹配；不是8步采样漏配。
12条带hold负事件episode的最近正样本中，11条超过手RMS、10条超过物体旋转阈值。
诊断产物均不可训练，不补采val/test、不启动fit，也不将此解释成Cm无效。
130项Task测试及verify通过；原始失败run、用户其他修改均保留。
GPU1/2三源PointWorld已按原冻结配方完成50000步，group/trainer均COMPLETED，
两rank参数哈希相同，全部自有进程退出，GPU0/1/2释放。
最终固定balanced moving-anchor h24：Oak13.778/GRAB27.795/ARCTIC39.804mm，
macro27.126mm（初始化34.480）；best46000为26.786mm。
latest/final50000及best46000均可加载，含优化器和双rank RNG；未开启新预算。
ref3的独立ContactPose auxiliary ablation另设300步matched短Probe，非主训练续跑。
两臂同latest50000模型/fresh优化器、三源main样本及CP辅助计算；仅loss系数0/0.05不同。
CP前向只允许history几何，future手/effect/label字段均不进入前向；小工程r2通过。
正式短Probe在681340a运行，control第112步后DataLoader worker因Aborted退出，
状态FAILED且进程已退出；无完整对照结果，不能据此判断auxiliary有效性。
原始产物保留，数据加载故障待修复；没有自动重启或扩展预算。
详见[配对覆盖卡](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-consequence-pair-coverage.md)。

## 2026-10-07 ref2/ref3 运行记录（历史）

新随机自训练s1 parent200资格50/64；s3 bounded220→260固定资格36/64，
恢复Probe PROMISING。十条剩余reference全部恢复；六角色airplane_base已过资格，duck340固定资格8/64，保留旧duck280的0/64记录；cup正训练，其余三角色待训练，
尚无真实evaluator数据/fit。GPU0已自然空闲，duck首轮smoke因缺mesh在PPO前失败；
owned资产补齐；duck真实1024帧geometry smoke通过，533forceproxy帧均有<=1cm gap。
短20epoch不足以复现历史80epoch specialist预算；GPU0重新空闲后，
1386d51从合格s3260完成新duck260→340，端点资格8/64；cup80epoch独立迁移已启动，完成后固定资格已排队；三角色待恢复。ref2改用执行前已知24×18
请求残差计划，三臂E0/object/EI，同容量、几何接触约束、同expert/motion/current
物体手状态配对；89工程测试通过（含motion软链、阶段覆盖、native driver与新路由回归）；GPU1真实8env×128步几何smoke通过，712forceproxy帧均有≤1cm采样gap，
但尚未验证其他物体/桌面反例或完整真实数据，不升级为精确contact GT。

按用户ref3，四源PointWorld正常保存停止step14250，moving-anchor h24验证
Oak9.24/GRAB38.86/ARCTIC51.24/ContactPose27.15mm；best/latest保留。
三源Oak/GRAB/ARCTIC主监督已于23:20启动，commit9019fd4/GPU1/2每卡64，
latest14250模型初始化/fresh优化器，50000新更新，
用户指定截止2026-10-08 10:00。新step0同panel EPE为Oak15.92/GRAB36.13/ARCTIC51.39mm，
不同于旧四源panel，不直接横比。首次250val宏34.48→37.16mm，早期退化，
step1500宏29.88mm曾改善，但step2500宏35.87mm再次高于初始化34.48；
短期波动，不认定稳定改善或平台；
每250步保存latest/best；含val/save实测.744s/update，50000预计09:41，硬截止不变。
ContactPose仅刚性运输辅助；EPIC仍candidate_only，
独立修订坐标约定、双手有效性、gap及OF overlap质量，未开放训练。
旧四源结果不升级为同质监督或Cm收益结论，最终matched trained-policy utility仍OPEN。

## 2026-10-07 用户指定 consequence-evaluator 路线

用户要求将当前PointWorld与数据处理代码合并到本地main，再创建`consequence-evaluator`。
数据扩充提交a620375已合并，实验索引跟进9c555d1；当前新分支继承oracle与PointWorld。
按Task ref1固定K24/Kexec8，先检验连续六专家rollout上的E0(H,A)与Eoracle(H,A,Z_GT)，
暂不推进WM适配或在线proposal。论文原文/源码已核对；独立scalar BT是Robometer改编。
窗口合同、标签mask、两臂模型、有界训练入口与六专家连续采集驱动已实现；CPU测试覆盖真实驱动循环，
但未在真实Isaac Gym上验证，也未采集或拟合真实数据。采集副产物保留measured q/root与扰动审计，标签仍需核验。
独立test评价入口已实现：冻结val选择的权重，报告配对排序与任务/阶段汇总、跨episode未来替换诊断；
31项工程测试通过，不代表真实oracle headroom，也不把重叠窗口当独立样本。
旧oracle outputs、六专家权重未找到，原motion输出缺失。用户已授权重训专家并重新rollout；
外部canonical几何motion与仿真资产可用，13条motion完成CPU合同检查，已复制/软链接到owned输出。
DExplore指向已删除baseline的失效data链接已保留并修复。先新训s1 parent，再检查frame0抓取后迁移s3与其余专家；
新权重不继承旧六专家Validation结论。PointWorld原三卡训练已正常完成10000新更新，源码hash未变；
固定balanced运动anchor h24 EPE14.0672→11.1342mm，best为step9500的11.1026mm，
预训练Probe PROMISING；末尾2500更新仍改善3.49%，仅能判断收益减慢，不能认定完全平台。
best/latest/final均保留，未开启test或新训练预算。专家重建r2的native smoke在首次PPO更新前
因NumPy别名兼容导入顺序失败，原日志保留；调整bootstrap导入顺序后重试原预算内预检。
随后用户授权OakInk2继续：GPU1/2、每卡64/global128，从原latest保留AdamW动量，
沿用末尾lr约1e-5追加最多10000更新，仍受原绝对deadline约束；这是显式两卡迁移，非逐位resume。
GPU0用于consequence专家/rollout。r3重建smoke在首次reset发现CPU/CUDA混用，未发生PPO更新；
task-local入口改用CUDA PhysX tensor pipeline后，按原输入/预算执行r4预检。
启动前用户暂停以上续训安排，优先混合OakInk2/GRAB/ARCTIC/ContactPose，原latest仅作模型初始化，
新优化器/训练从头开始。两卡续训与r4预检均未启动；当前无本会话GPU进程需要停止。
四源全量整合和双卡12步工程检查已完成；GRAB/ARCTIC官方split、原生类别映射，
ContactPose真实世界位姿/30Hz及gap屏蔽已接入，参与者train/val/test隔离。
2026-10-07 19:57:29开始混合训练，运行提交384f860，GPU1/2每卡64/global128，
OakInk2/GRAB/ARCTIC/ContactPose采样50/20/20/10；原latest只加载模型，优化器/调度重置。
训练窗口总4794229，固定64val窗口/source，step500运动anchor h24宏平均52.56→45.74mm；
OakInk2新固定panel9.79→10.49mm，较step250略恢复，其他三源改善，早期遗忘/适配取舍仍UNCLEAR。
沿用原deadline2026-10-08 09:58:37，最多40000新更新，不追加24h。用户随后恢复consequence-evaluator：GPU0启动r4重建，2epoch GPU smoke通过，
64env/200epoch s1 scratch母策略已完成（1114.91s），权重保留；最近接触/保持指标仍0。
固定64条frame0/45帧保持无后续drop资格检查完成（c0b10fb）：0/64合格，
不扩展六专家，先同初始状态参考动作回放诊断；这不构成evaluator/world-model负结论。新增局部物理事件标签、独立专家progress抽样和资格检查共44项工程测试通过；
尚未获得真实evaluator数据或oracle headroom结论。
后续真实物理诊断确认r4存在GPU reset提交/刷新缺陷：首步物体跳1.48m。
task-local修复后同环境降到1.43mm，重复子集reset/FK物理检查通过，51项工程测试通过；
原r4训练和0/64不能用于判断正确实现的学习效果。已找回原s1 corrected tensor，
与r4 canonical在q/坐标/contact上并不等价；旧新均用graspenv+Dexplore_Inspire。
原s1数据GPU物理预检已通过（15.78s，首步位移1.43mm，三次子集reset通过）；
原CmLite权重/转移与s3 corrected输入仍需重建。
当前推进：567c909启动原s1输入/修复reset/Cm-off新重建，GPU0、64env/h32/mb256、
LR2e-5/mini-epochs6、anneal40→80、200epoch；smoke36.18s通过，前80epoch有真实接触/抬升奖励，
训练吞吐约500FPS、显存18251MiB。缺失CmLite奖励未启用，不称原配方精确复现。
CPU已恢复s3 corrected reference（23.17s，几何相对误差<3e-7m，左contact0），迁移仍待母策略资格。
该新母策略200epoch已完成（1123.50s）；独立64条frame0完整episode中50条通过45帧保持/无后续drop，
超过预设8/64数据准备门槛，重建Probe PROMISING（不继承旧Validation或证明Cm收益）。
GPU0下一步从新自训练parent200迁移s3到220，先2epoch工程检查、再20epoch正式Probe部分，
最后固定端点资格检查；权重祖先/RMS/优化器/epoch均核对，LR显式1e-5，保留已结束anneal40→80。
s3首轮200→220已完成，但固定64frame0资格只有2/64通过（未达8），首步位移1.72mm内正常，
有16/64短抬升且训练接触上升。下一步同输入/配方有界续训220→260并再评固定端点；不扩展其他专家。
cup参考输入CPU恢复通过（934帧/相对误差<3e-7m），不是cup抓取证据。
其他专家和连续六专家数据仍未完成；尚未训练E0/Eoracle。
详见[混合预训练卡](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-multisource.md)。
见[重建Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261007-consequence-baseline-rebuild.md)。
见[任务入口](../src/task/consequence-evaluator/README.md)。不改变最终matched Cm-on/off策略utility要求。

### consequence-evaluator 2026-10-08 续跑

六个新 endpoint 已冻结为不同 hash：airplane_base36/64、duck8/64、cup63/64
通过固定资格门；mixed12/train5/balanced5 为0/5/4，保留为 observational
候选。route.json 明确 `all_experts_operationally_qualified=false` 和
`training_allowed=false`，不恢复旧六专家 Validation 结论。queue smoke 的
hard-object oversampling 与 minibatch divisibility 修复已提交
`3b4305b`、`e62dd08`。

首轮三 split 原始采集 144 episodes 后，严格 local-state label 为
train/val/test=0/0/2；独立 proxy/gap 几何审计通过。按 Decision Note 追加
296/297/298 三波次、216 episodes，仍使用 24 步 decision-known residual 和
2 cm hand RMS。追加标签在未改合同下变为 `READY`、pair=2/1/5，审计为
149,841 valid frames、49,388 native proxy、45,984 proxy-near、仅1 proxy-far。
但 2 train/1 val 不足以支撑 32-pair/update 的 evaluator fit；继续训练只会
重复采样并在一个 validation pair 上选模，故本轮不启动 `prepare_windows` 或
`train_matched`，也不放宽阈值。原始、label 和 coverage diagnostic 均保留；
下一步若继续，需设计同一 current state 的 twin residual branches，而不是
再堆 generic waves 或把 hand RMS 放宽后重判。North-star matched Cm-on/off
policy utility 仍 OPEN。

当前实现审查把这项 gate 固化为规则
`geometry-corroborated-H-matched-state-residual-events-v3`：偏好端点必须同时
匹配历史 `H`、当前物体位姿和11点手状态，历史相对RMS上限为0.25；train/val/test
还必须分别拥有至少8/4/4个不同的无序 episode pair。collector 只把该要求冻结进
source manifest；labeler、window 准备、`Windows` 和 `train_matched.py` 均拒绝不满足
的生产 fit 数据。上面的历史
2/1/5 产物因此只保留为 observational audit，不能作为 evaluator fit 输入；当前
route 仍是 `all_experts_operationally_qualified=false`、`training_allowed=false`。

新增的 `consequence_evaluator.twin` 是严格的 Isaac-free v1 合同测试：要求完整
native task/controller buffer、Python/NumPy/Torch RNG、fresh simulator prefix
replay provenance、刚体物体位姿和双分支第0帧锚点。它尚未连接 native collector，
也没有真实 twin branch 产物；这部分仍是下一项实现 blocker，不能把合同测试升级为
科学证据。Task tests 在本轮代码审查后继续作为工程回归，不改变 North-star claim。

2026-10-08 native twin 工程推进：`consequence_evaluator.twin` 新增不依赖 Isaac 导入的
native capture adapter、完整 prefix trace hash/帧数检查和 Python/NumPy/Torch RNG 捕获。
它现在能把初始化后的 native task/controller 边界转成 v1 snapshot，但尚未接入连续
collector 的 fresh-simulator 双分支执行，也没有真实 twin branch 产物；因此 blocker
从“缺少 adapter”收窄为“缺少 native branch runner 与真实覆盖”，不升级为科学证据。

更新：2026-10-06。本摘要整合已交付的主分支与本轮Cm研究事实，不产生正式科研结论，
不纳入其他独立会话尚未交付的结果。完整旧摘要见[状态快照](archive/research/STATE-20260930-before-workflow-simplification.md)。

| North-star | 当前判断 |
| --- | --- |
| Self-trained grasp | PARTIAL：限定 12-motion 的六专家初始观测路由已通过正式 C1 Validation；单一 actor 稳定结果仍未证。 |
| Cm one-step information | PARTIAL：物理效应可学，但依赖表示、分布与目标。 |
| Cm policy utility | OPEN：尚无跨训练 seed 的 matched Cm-on 优于 Cm-off 证据；effect-rank 正式 Validation 的正向主张已 REFUTED。 |
| Generalization | OPEN：未见物体/多轨迹上的 Cm 收益尚未建立，本阶段先聚焦固定任务分布。 |

## 关键事实与边界

- C1 的任务限定 SUPPORTED 不证明单一 actor、未见物体泛化或 Cm utility；冻结已验收的
  六专家 substrate。[正式验证](experiments/validations/VAL-20260926-observation-six-expert-c1.md)。
- HF01–HF05 的已失败局部路线保持冻结，不靠换 seed/门槛重置预算；这不是所有未来 Cm/GPU
  路线的全局禁止。历史路线边界保存在归档实验记录中。
- 用户已授权边界内自主选路线，以及六专家蒸馏与新的 Cm 探索；历史标签不因此升级。
- r6 support 的 teacher label 仅覆盖 source_e260，不能证明六专家蒸馏；其缺失轴字段不能回填。
- r7 轴合约通过，但 contact q10 与 delta 覆盖未过 calibration gate，不生成正式 Cm-on 标签。
  [校准证据](archive/2026-10-04-research-governance/handoffs/CM_SCRATCH_CPU_CALIBRATION_R2_AXIS_20260928.md)。

## 其他已有任务交付边界

主分支已记录的两个旧系统受控任务：一次 fit-only CPU 校准修复，以及独立六专家逐步轨迹蒸馏。
校准任务 2 CPU/15 分钟/1 GiB，蒸馏任务 1 GPU/60 分钟/5 GiB；均只形成 Probe 结论。
旧系统任务的实际终态以各自交付为准，不由新工作流猜测或重新启动。

先验收实际交付，停止失败的局部 calibration tuning；蒸馏独立推进。新 Cm 路线须服务于
真实策略因果增益，保留 matched Cm-off 对照。[实验索引](experiments/INDEX.md)按需检索。
运行细节、失效执行、数值和哈希留在原卡/manifest；资源授权见 [CAMPAIGN](CAMPAIGN.md)。


## 本轮 Cm 策略价值研究

用户授权先取得真实策略 matched Probe，正向则优先正式 Validation；期限为
2026-10-03 23:59（Asia/Shanghai）。完整交互与短期物理预测结合长期价值，
最终要求训练所得actor受益；瞬时物体位移不替代动作价值，跨数据集预训练可选。

HF06 teacher-envelope与HF07 BC物理预测输入实现均为UNPROMISING，保持关闭。
HF08 physical-value已完成：公共池1041599转移、1920完整episode，fit831811行，
开发holdout209788行；同一自训练source_e260、三条canonical airplane motion，
六臂均追加160epoch/327680交互，全部48点actor-only评价完成。原生gate为
UNPROMISING：终点Cm33/384，普通PPO41/384，直接Q40/384。这里只是Probe，
不否定Cm核心假设；North-star policy utility仍OPEN。

R2价值目标审计已由root验收：source主成功15/1920、holdout成功1/384，两个e420
V checkpoint均实际完成7680 optimizer updates。结论为UNCLEAR：V训练充分性和当前
策略校准仍未知；不据此宣称需要重训或已收敛。

在线适配后物体位置误差降至0.78/0.86cm，仍弱于恒定线速度基线；姿态误差
仍约1.34/1.36rad，明显弱于保持状态。模型不能称为已准确预测物理转移。
同source checkpoint的e0重复评价有差异，原生配对只覆盖env/motion/start/时长/
初始高度；初始完整state恢复审计此前已排除为当前阻塞，不以单次realized MC误差
宣称bias。暂不升级Validation，不调参重扫HF08。

collector与V诊断工程已验收；R1 nativecwd资产失败、R2 wrapper在GPU ownership前失败，
两者均无新数据、无V充分性结论。R3/r3b也已FAILED并完成进程清理，均0行；
CM真实输入守卫已验收、集成，main复验11项通过。
当前执行原生Gaussian、frame-0、冻结策略的完整episode诊断，尚无完整采集数据或新训练。
真实环境初始化暴露reward_shaper对象检查错误；修复后首个动作又因采集器误把logstd
当sigma退出。原生model还负责观测归一化，raw网络直调不能替代它。错误属于采集器，
不证明checkpoint标准差无效。root已派固定RL角色做CPU-only原生player合约修复；
原恢复轮截止不延长，未来采集按实际累计成本另记有界预算。单次MC误差不单独证明bias；
HF08 slot不重置。见[修复记录](archive/2026-10-04-research-governance/decisions/D-20261001-native-player-collector-repair.md)。
[HF08实验卡](experiments/probes/P-20260930-cm-physical-value.md)与
[完整结果](experiments/probes/P-20260930-cm-physical-value-results.json)保留边界与数值。
原始数据/checkpoint留在原研究工作树的research/output/P-20260930-cm-physical-value/r7，
未提交Git，不因本次合并移动或删除。

## Ref5：point-flow E → G 与 surface I

当前实现分支 `agent/cm-interaction-oracle`，新 Task 为
`src/task/cm-interaction-oracle/`。新工具/卡已放入 Task；旧工具和历史卡逐步迁移，
根级 Mission、Campaign、seed ledger 与状态保持唯一来源。

- 真正冻结 point-flow K1 E → G 已跑 matched Probe：同容量 H MAE23.3652、
  H+GT E23.2837、GT-trained bridge换预测E23.6477、预测E训练bridge22.8961。
  判定 UNCLEAR：GT仅0.35%改善，预测fit2.01%且CI跨零，收益受单个高RTG episode
  支配。去此episode剩余21个episode预测fit反而差5.82%。
- 同容量GT合同检查（短序列右对齐，避免GRU补零冲淡）：pose K1/K4/K8改善
  0.034%/1.067%/2.748%，full13 K8改善7.42%
  但去高RTG episode后负1.66%；没有合同通过预设门槛。保留K8 pose弱方向性信号，
  不据此扩大K4 predictor或进入policy teacher。
- 8patch×5D GT surface I比同容量零I改善10.83%，CI跨零且约97%收益来自同一
  高RTG episode；pooled I更好，未证明空间topology独立价值，不启动I head/K4 I。
- 工程审计发现fresh shards手root平移最高22.33mm，旧identity-root假设不能迁移。
  用当前measured body pose恢复共同root后，五body最大残差5.35微米；当前/未来root
  cache分开。首轮G失效、no-grad技术失败均留痕；修复后独立review未见其他致命bug。
  CM平移RMSE19.33mm，当前twist persistence4.92mm，不能称准确物理预测。
- 原始episode summary与assembled labels逐条一致：112 episodes仅2次达到45步
  held成功，两次随后都掉落；该source_e260数据的成功无后续掉落数为0。主导held-out
  RTG episode即使最长保持8.87s也随后drop，高RTG不能直接视为最终稳定抓取。
  这些统计不重判已验收六专家baseline，不据此直接训练稀疏success/drop二分类G。

Root选择：保留E物理主路线与修正的几何合同，暂缓局部I/predictor扩展；下一次
最便宜的决策应先核对持续hold/drop任务目标与action contrast。RTG回归不替代抓取、
保持、掉落评价，更不替代matched Cm-on/off训练所得策略的因果收益。
两路使用GPU6/7，均已结束、零新采集；本轮只给Probe判断，North-star Cm utility仍OPEN。

[E→G实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-pointflow-g.md)，
[surface I实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-ref5-surface-i-gt-value.md)。

## Task ref1：RECAP-style 相对任务 advantage

同一 Task/branch 已实现并执行 cross-fitted MC V → 32步 advantage 的有界
Decision Probe；task reward 固定 `held/10 + lift_progress/5 + stable`，排除
base/approach，所有模型显式控制 episode-fixed noise。历史90train/22test
episode split、两次V拟合×三fold，gamma0.99，固定16epochs，不改变核心 Mission。

- G0标签计数、episode支持、集中度均达标；held-out V MSE229.58低于zero324.80。
  但独立fit标签一致率 train34.72%/test39.15%，远低于预设75%，结论UNCLEAR。
  按门槛停止，G1 action critic、G2 predicted E/I、G3 selector均未执行。
- 连续advantage test相关性0.978，去最大MC episode仍0.952，保留方向性线索。
  真实32步task reward非零只覆盖16.3% test queries；该区标签一致78.0%，
  零reward区29.6%。后者fit差中位数0.255大于0.05幅度floor，不只是微小量化抖动。
  不能事后用未来reward active筛选测试样本或改一致率分母宣布通过。
- 工程review修复neutral样本污染binary AUC的问题（正式运行前，G0不受影响）；
  后续112 episode target/row/fold/threshold audit精确一致，无进一步致命问题。
  当前暂停原因是标签/value合同可靠性不足，不能写成RECAP或动作信息被否定。

Root Decision：保留相对任务后果方向，下一决策前提是可信标签/value；暂不投入
I/G容量扩展。任何改标签协议须重新明确假设和判定，不放松本轮门槛或追加seed搜索。
本轮GPU6 model执行含smoke<20s，零新采集，任务进程均结束；代码commit9dd503d。
North-star Cm训练策略因果utility仍OPEN，不把离线标签工作算作policy进步。

[相对任务advantage实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-recap-relative-action.md)。

## Task ref2：冻结连续 advantage 的 action ranking

ref2 明确授权新排序问题，未重跑V或放松旧三分类G0门槛。新 matched pairwise
Probe完整执行 H / HaK1当前动作 / HaK4已记录feedback；同96,705参数、同init、
同pairs/训练设置，既有90/22episode split、全部冻结advantage继续使用。

- 正式测试 macro同motion/phase/noise pair accuracy：H59.712%、HaK1 59.617%、
  K4 60.520%。K1增益−0.095pp；macroSpearman仅+0.0021；打乱当前动作仅降低
  0.133pp，两个独立V target的增益均负；8/19可配对episode改善。UNPROMISING
  针对当前fixed target/data/fit合同，不是物理动作信息或Cm被否定。
- 新审计纠正ref2的排序前提：Pearson0.978并不证明rank稳。两套冻结advantage
  的Spearman train0.363/test0.540，条件pair顺序一致约62.4%/66.8%。去三分类
  boundary并未形成强排序监督；训练93%–95%而测试≈60%，泛化仍弱。
- 配对实际覆盖19/22test episodes、64strata（2,432/2,816anchor queries）；
  macroSpearman使用88strata，包含单episode无cross-pair组，分母明确分开。
  去最大MC episode后K1+1.333pp是弱敏感性线索，未过整体门槛。
- 当前动作进入网络且打乱score RMS0.991；独立review及root SciPy/rawscore
  复算确认pair/input/normalization/target/hash/metrics正确，无进一步致命bug。
  第一smoke仅path-hash metadata失败，修复留痕；正式执行commit1a64eab。

Root按ref2负分支停止这份离线标签上的critic/predicted E/I扩展，未新采集或训练
teacher/policy。下一研究决策应先建立可信物理action/outcome contrast，不能仅
换V seed、width或threshold继续此数据拟合。保留E/I与relative action科学问题，
North-star causal Cm-on/off训练所得策略utility仍OPEN。
本轮GPU6 valid模型执行含smoke<49s，产物<8MB，全部任务结束。

[冻结相对advantage排序实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-relative-action-ranking.md)。

## Task ref3：真实随机动作干预

固定自训练source_e260、airplane三motion，移除episode噪声，七臂随机分配
四步feedback residual；336完整episode、320实际干预、320完整32步窗口。
每臂37–54次、有效dose100%；全batch reset屏障，pre/post/PD/GT标签经独立review。
全部decision为pre-lift（最高lift8.74mm），early-hold/drop-risk样本为0。

- 按环境留出66train/18test，预测器及scorerOOF也隔离环境。H E/I MSE0.5748，
  Ha0.5841；macro任务排序 H66.78%、direct66.14%、predicted65.17%、GT61.17%。
  七项预设gate全部失败，当前合同UNPROMISING，不启动selector/teacher/policy。
- GT保留contact-retention局部信息，但整体泛化弱；模型容量/OOF噪声差和少量
  history瞬态限制归因。PCA总test能量异常主要由单行支配，典型样本保留率中位
  86.96%，不是所有测试状态丢失93%；不删除outlier重判结果。
- 最便宜的既有数据Decision诊断控制当前状态后，物体短期旋转有弱随机臂响应
  （E/I家族探索性permutation tail0.045），任务窗口/完整episode证据仍不足。
  手实际收到不同扰动不等于已建立任务相关可预测中介链。
- Full summaries105/336达到45步hold、76随后drop；不同采样下的raw arm均值
  不当作策略增益，也不修改旧baseline结论。motion symlink漏预先hash已补明确
  post-run provenance，后续collector修复；原manifest/失败smoke保持可追溯。

Root停止本pre-lift E/I8→Y16/32 PCA/MLP扩展，保留随机干预数据和物理control
问题；不据此终止Cm核心路线。未来研究需要新的阶段/时域决策合同，不换seed/
阈值/encoder继续拟合本合同。单GPU6，正式仿真222s、smoke70s、模型4.86s，
统计1.12s，产物<13MB，进程均结束。North-star训练所得Cm策略utility仍OPEN。

[真实随机动作干预实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-randomized-action-intervention.md)。

## Task ref4：early-hold interaction retention control

按用户新增ref4，干预移到lift≥3cm且连续6步contact-proxy的early hold。
1,008完整episodes产生494随机干预，全部32步窗口完整、剂量实际执行。
后续214次失败中212有物理高度损失；98次早失败全部保留，不筛掉step8失败者。

- A UNPROMISING：短接触比例调整后臂间范围5.68pp，tail0.1335；I14 family
  tail0.5005，未建立预设action→retention-I可控链。
- B UNCLEAR：GT I令后9..32预测误差改善49.6%，failure AUC0.899→0.949；
  但两主目标均只有2充分支持分层，未达预设3层。原macro排序58.46→77.33%
  受1–2pair小层放大；充分支持层描述性87.76→96.32%，不事后改gate。
  未早失败的92个test trial也有预后改善，但这只是post-treatment描述性子集。
- 独立工程review及root逐项复算通过。未启动C、selector或policy训练。

Root关闭当前early-hold四步feedback residual→I8合同的继续拟合，保留GT I预后
价值。缺口转为可操纵的保持控制变量/动作时域，不能靠更多seed、网络或补B支持
绕过A失败；不否定Cm核心假设。单GPU6仿真约417s、模型2.31s、产物<18MiB，
全部结束。North-star训练所得matched Cm-on/off策略utility仍OPEN。
[early-hold实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-early-hold-intervention.md)。

## Task ref5：延长 feedback residual 的物理响应

仅做physics Decision，不训练Cm/S/PPO：在同一early-hold区随机分配7方向×K4/8/16，
2,016完整episodes、1,006完整干预窗口。非零cell最少32、两半最少13、pooled zero148；
全部剂量约1.0，native PD和独立标签/OLS/curve复算通过。

结果UNPROMISING：短接触tail0.1645，K16最大绝对效应6.62pp，未达10pp；I16 family
tail0.572，后17..32接触/高度失败tail0.581/0.244，未形成稳定duration→retention-I链。
动作实际改变了手：wristx+在同一step16调整位移随K为4.51/10.47/21.92mm。
部分反向baseline命令与补偿一致，但累计剂量/释放恢复时长同时变化，不能唯一识别
feedback cancellation。secondary all32高度失败有弱tail0.085，不能救援I控制gate。

Root关闭当前K≤16 feedback-residual合同，保留I预后信息及测得的反馈响应；不扩大网络、
时长/幅度/seed扫描或补旧B支持。不否定其他operator、I/Cm核心思想；固定绝对target
仍未测试，若以后推进须有能区分机制的新合同。GPU6仿真787s、CPU统计2.81s，产物
<37MiB，全部结束；25 Task tests与变更检查通过。训练所得matched Cm策略utility仍OPEN。
[duration实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-early-hold-duration.md)。

## Task ref6：固定 K8 的幅值与局部 interaction authority

alpha1/2/4×七方向联合随机：1,680 episodes、863完整干预窗口，剂量、support、
原生PD、输入哈希及独立/root复算全部通过。任务关联authority gate UNPROMISING：
短接触tail0.8895、alpha4最大效应3.68pp；后9..32接触/高度失败tail0.554/0.1965，
没有合格阈值候选，因此未启动条件性Stage2或神经模型。

局部interaction响应PROMISING：I8、切向投影、ratio family tail均0.0005，主要由
thumb_distal贡献；alpha4 finger−距离+22.95mm且力下降，finger+距离−8.44mm且力上升，
两半同向。wristx+同step8手位移12.95→23.53→47.61mm确认动作authority。
原force norm已捕捉信号，不能将缺少任务差异全部归因方向压缩；全手OR contact-proxy
可掩盖单thumb变化，物理高度结果也未明确分化。只证明此合同的局部proxy可控，
未识别抓持阈值、可用控制区间或Cm收益。

Root保留局部action→I证据，停止本次幅值合同，不扩大幅值/seed/网络以重复authority；
后续需区分可控局部接触是否实际参与承载。GPU6采集player累计671.48s，产物<40MiB，
均已结束；North-star Cm utility仍OPEN。
[幅值实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-amplitude-authority.md)。

## Task ref7：独立 finger DOF 与实际幅度审计

拆为六独立DOF±（含thumb-yaw）、zero、匹配旧synergy±共15臂，固定K8/20%driver range。
1,680episodes、854完整窗口，support/dose/nativePD/17input hashes及独立复算通过；
各指PD rad/deg、mimic、实际q和真实tip矩阵已写入Task实验卡，ref6旧卡也补PD幅度表。
同20%range并非同毫米：四指局部tip对照范数约15–29mm，thumb约9–14mm，未假定thumb
运动最大或已知冗余/承载身份。

预设own-I+late-task及useful gates均UNPROMISING；但后9..32contact family tail0.0005，
不能概括单指控制无效。middle−保持−10.94pp两半重复；thumb-yaw−保持+10.76pp、高度
失败−11.34pp，两半同向。后者height family tail0.1535及own-I幅度/半包一致性未达标，
保留为下一项针对真实高度安全的窄对照候选，不靠改gate宣称有效或启动Cm训练。
现有净力contact-proxy改善不等于可靠抓持；训练所得Cm-on/off utility仍OPEN。
GPU6累计691.45s（含初始化），均已结束，36tests通过。
[per-finger实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-per-finger-control.md)。


## Task ref8：GT consequence 对任务信息的保留

复用ref7的854随机窗口，不再采集；step8 E12/I14预测step9..32任务，
688train/166test、125/31环境隔离，六主模型及三预先声明signed-force扩展同容量/init。
GT预后PROMISING：HEI主误差0.73135→0.39311（46.25%改善，环境bootstrap95%
34.02–57.03%），物理高度失败误差改善32.52%（11.24–49.92%）；H+a也有18.99%
主误差改善。跨半包、留一臂PCA3/ridge1桥的EI接触/高度失败误差相对zero改善
21.65/16.14%，相关0.550/0.723；是共享zero的有噪声边际估计，不是因果中介识别。

完整充分性UNCLEAR：HEI再加a主/物理误差点gain−0.33/−0.93%，但one-sided95%
上界7.96/9.64%仍超过预设5%，不能排除有意义的剩余action信息。Signed扩展下a
又有8.68/12.91%增益，末50epochs loss仍降约40%，初始q/dq经PCA与泛化误差均限制判断。
无泄漏/关键工程bug；root和独立审查复算通过，逐指PD/q/真实tip幅度表附新卡。
下一项有决策价值的是独立冻结的conditional predictability（H+a→E/I，再与直接Ha
比较预测consequence的task增益），不以GT预后或点估计等同链闭合，不立即selector/PPO。
本轮未训练Cm；训练所得matched Cm-on/off utility仍OPEN。
GPU6固定拟合5.56s结束，首CUDA初始化前失败0.76s留存；新产物<4MiB、41tests通过。
[GT consequence实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-gt-consequence-sufficiency.md)。


## Task ref9：conditional consequence prediction 与 OOF 任务价值

冻结ref7数据及ref8的688/166环境隔离划分，E12/I14、intended one-hot保持；
21个固定1500epoch拟合，严格fold-only PCA/尺度/OOF，保存directHa和两个shuffle对照。
总体UNCLEAR，预设A/B/C均未过。Ha I误差比H改善8.50%但区间跨零，E反而恶化20.79%；
冻结Ha置换test动作使I误差增加11.96%(95%4.17–22.24%)，保留局部动作敏感性。
同H候选I差分corr.293、方向62.29%、幅度比.369，未稳定保留动作差异。

OOF后果P_Ha主MSE.6730，弱于directHa.6478；Ha+P_Ha .5995有7.45%改善点估计但
CI跨零。GT同预算oracle .4080仍强；预测后果保留31.18%oracle gain（CI2.90–56.56%），
不能等同超过directHa。跨half预测已做，83行子集GT contrast rank13/9记为NULL/UNCLEAR，
未用伪逆制造逐臂结论。全部14臂预测／GT向量及逐指PD/q/true-tip幅度表在新卡。

按用户要求独立review反常结果，root GPU全权重重放精确一致，无时域/OOF/尺度/shuffle bug。
预测器明显劣于train-mean和persistence，train接近零而test高；末期loss下降不支持追加epoch。
Shuffle较高原sign受共同zero偏移影响，中心化后corr−.076；不救援主结果。
结束当前固定拟合，不selector/PPO；下一Decision应区分受控泛化训练与缺少信息，非更多
memorization。没有关闭Cm核心假设，训练所得matched policy utility仍OPEN。
GPU6主运行59.99s，smoke3.93s，审计仅inference，均结束；新增<12MiB、46tests通过。
[conditional consequence实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-conditional-consequence.md)。

## Task ref10：几何动作表示与物理残差

按ref10将决策native q/PD指令经explicitroot×FK转成120surface对应点流，
共同提供baseline nominal geometry，对比State/Arm/Joint/Flow和shuffle；
fold-only PCA/尺度、严格environment OOF、固定α32ridge，无新仿真。
原PCA/bilinear残差screen UNPROMISING：Flow I12.440明显崩溃，testshuffle反而
降低误差。按用户要求独立review：无来源/标签/泄漏/单位/求解bug，root GPU全重放0误差。
单窗口/前五窗口贡献57.84%/85.13% Ierror，state×flow乘积超出训练支持并放大输出。

另预先冻结六fit尺度×乘法因子诊断（明确post-result、非独立确认）：移除products
I1.254，固定20mm additive1.138；仍弱于State1.074/Arm1.055/Joint1.041/mean.891。
当前PCA/ridge名义端点路线UNPROMISING，不能否定物理几何动作或OI-CmV2空间网络。
独立NumPy复算六fits/bootstrap一致。GT仍改善task31.50%(CI8.32–51.03)，
P_Flow .7149却输H .6540/P_State .6132，R−.2955；C=true仅胜崩溃directFlow，
root明确不认作uniqueCm收益。不selector/PPO、追加epoch或删outliers。

保留FK/continuous surface合同、OOF工具、14signedarm向量、逐指PD/实测q/true-tip
及名义surface幅度表；后续若继续几何路线，应检验localcontact/sharedspatial归纳
偏置与nominal→realized执行合同。全局matched trained-policy Cm utility仍OPEN。
GPU6两主run最终manifest2.88s/1.19s，首启动hash路径错误在模型计算前终止并留存，
审计只inference；新增<85MiB，50Task tests通过，无剩余本轮GPU进程。
[ref10主实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-geometric-innovation.md)，
[归因实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-geometric-support.md)。

## Task ref10 续接：局部 OI-CmV2 空间后果

续读原会话后补上尚未测试的空间路线：复用V13局部交互/token/fusion模块，
固定width32/4tokens/2cmKNN8，名义PD端点点流与persistence残差；全部854窗口，
原688/166环境划分，严格三折OOF，20后果+8同预算直接/GT/预测后果task模型。

本固定合同UNPROMISING：Flow I.8891，State.8837、Joint.8670、mean.8907；
动作shuffle仅增加I误差.565%，I contrast corr.239/sign53.14%/幅度比.066。
P_Flow主误差.5442与directFlow.5468接近且输P_State.5329，五gate均未过；
GT仍改善主误差30.22%(CI11.98–43.15%)。误差稳定，无旧bilinear爆炸。
全部28保存权重、OOF和15Flow候选重放精确一致；独立review与root核验通过。
99.06%窗口有空间边，动作改变邻居/特征且梯度非零；不是动作未接入。

Root停止这个固定fit，下一Decision需区分名义端点与可预测实际执行，不能靠更多
epoch/seed重复弱动作通路。端点/实际tip运动已有明显描述性差距，但不能唯一归因
反馈抵消或断言空间模型不可学。没有selector/PPO或正式科学结论；全局Cm策略
utility仍OPEN。GPU6主162.62s、smoke6.17s、重放3.44s、review GPU.74s；新增约154MiB、53tests通过。
[空间后果实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-spatial-consequence.md)。


## Task ref10 续接：真实／可预测执行几何

固定8个ridge执行拟合、2个实测oracle与3个可预测空间后果臂；854窗口、原环境
划分、执行训练输入严格source OOF，无新仿真。原全部18q裁剪发现实现错误：URDF
腕部3角为continuous，±pi仅PD尺度fallback。root与独立review确认，保留原run，
修复后重放8ridge、复用2oracle/4control，只重跑3个受影响300update空间臂。
错误投影的预测结果不作为原定连续执行路线负证据。

修正后Ha finger误差比H降低79.81%(CI75.70–83.48)，q18 .4793低于nominal .6090；
但surface XYZ-component RMSE34.09mm vsnominal32.30mm。无拟合FK分解发现真实
腕部+预测手指仅2.13mm，预测腕部+真实手指33.94mm，说明事实端点误差主要来自
腕部；前5窗口贡献83.45%误差，未删样本。名义／当前腕部的因果替代分别32.15／
50.92mm，没有取得oracle精度。不能据此宣称执行不可预测，也不能用oracle部署。

本固定端点空间合同UNPROMISING：PredSurface I.8972差于State.8837/PredJoint.8620，
I contrast corr.152/sign50.86%/幅度.0235、testshuffle penalty.301%，五gate均未过。
实测Surface I.8913未胜nominal；实测Joint.8415比State改善4.77%(CI1.17–8.06)，
仅post-treatment诊断，支持研究表示传递而非断言信息不存在。全部保存权重/候选
重放0、ridge残差≤6.66e−16，独立review重建metrics/bootstrap/OOF一致；动作通路有效
（99.30%窗口有边、test22.81active points），57tests通过，原+smoke+repair约154MiB。
Root停止反复拟合totalendpoint合同，保留可预测finger动作信号；后续区分共同腕部
演化和finger相对几何，或检视spatial瓶颈。没有selector/PPO；全局Cm utility仍OPEN。
[执行几何实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-execution-geometry.md)。

## Task ref10 续接：动作传递与按手指残差

冻结空间各阶段做24个source-only线性动作decoder：RawHandBase恢复已知输入
nominal≈1.000/forecast.9987，trainedFused仅.3306/.3436；当前有接触支持的
手指子集也只有.4122/.4275。该诊断只说明固定14arm线性恢复弱，不证明物理
泛化或2cm半径丢失：arm均值lookup自己已≈1/.996，LocalFlow均值代理省略
实际边的位置、法向、距离与个别flow。独立复算通过，root收紧结论范围。

据此执行七头匹配factorial：完整按手指mean/RMS相对点流／相同forecast关节，
普通／15候选均值中心化残差；共同H156，复用三环境OOF State，固定300update。
本固定合同UNPROMISING：PredFingerCentered I.9124差于State.8837（gain−3.24%，
CI−5.94..−.62%），也未优于匹配PredJointCentered.9077。测试动作打乱罚2.75%
但CI跨零，I contrast corr.1565/sign56.57%/zeroMSEgain1.58%，三gate均失败。
名义几何中心化相对plain改善3.91%但CI跨零；不把更大响应幅度当作方向正确。

完整H/action/source-norm、四State和七新head/候选重放0；中心化均值约1e−7，
独立review复算metrics/contrast0、bootstrap≤2.39e−7，未见影响负结果的实现错误；
source-OOF/full-test nuisance差及中心化不能修正共同偏差保留为限制，未唯一归因。
62Task tests通过。GPU6 fidelity主11.67s、factorial主9.30s，零新仿真或策略训练。
Root停止这份mean/RMS+中心化fit，不追加epoch/seed；下轮须区分conditional
response、state nuisance或详细contact信息，不能据此关闭全部空间Cm。
两项都不提供strict nested consequence OOF或matched trained-policy utility；
North-star Cm utility仍OPEN。
[动作传递诊断](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-spatial-action-fidelity.md)，
[按手指动作残差](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-relative-finger-innovation.md)。

## 暂停旧执行路线，在原分支推进 ref11

最后一个surface-relative/current-wrist物理basis × Physics/OOFState nuisance
matched ridge32 Probe为UNPROMISING：PhysicsContact I.98857与同配置State.98882
接近，输priorState.88370；OOFStateContact1.03884。动作shuffle罚2.512%
(CI+.279..+4.959%)，raw Icontrast corr.4047/sign62.86%，四gate未过。
Root重放0、独立saved-matrix/statistics复算通过，未见影响负结果的实现错误；
动作scale floor导致不同有效shrinkage，current-wrist/采样normal/有限basis
限制保留，不否定所有contact geometry。GPU6主3.34s，零新仿真或policy训练。
[收尾实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-contact-innovation.md)。

用户澄清“收尾”仅指暂停旧执行预测路线，继续在原分支
`agent/cm-interaction-oracle` 按Task ref11推进，不另建分支：先以真实hand point-flow为
oracle action，分离 `(H,F_hand)→E/I` 规划问题与以后 `joint→desired flow` 控制。
已完成State、GT endpoint、GT0→4/4→8 chunks及matched shuffled flow比较，
先只看E/I；不继续execution、候选nativearm或任务Y拟合。实际flow是
post-treatment信息，正向只能支持oracle表示Probe，不自动证明前瞻planning或
同状态候选因果contrast。最终matched Cm-on/off训练策略utility仍OPEN。


Ref11首个raw720oracle-flow Probe已完成（原分支，代码452d7e5）。同688/166
windows和125/31environment split，五臂同59066参数/300updates。Chunk E.48968
vsState.52895，改善7.42%（95%CI+.46..+14.04%），E门槛通过；I.55603
vsState.60486，改善8.07%但CI−1.25..+17.62%，预设联合E/I gate为UNPROMISING。
保留E的PROMISING探索信号与I不确定性，不据此关闭oracle flow路线。
trained/frozen shuffle显示action敏感性；Chunk-vsEndpoint CI均跨零，时序优越性
未建立。新H包含清晰current q/dq/geometry，旧State.8837与新State.6049不属
同合同，跨run提升不能归因于flow。test0同状态配对，candidate contrast为UNCLEAR。
完整GPU input/FK/weight/statistics重放误差0，69Task tests通过；GPU6主5.62s，
零新仿真或policy训练。只读检查确认已有restore仅支持coldstate，不能恢复warm
PhysX cache。用户明确暂不做配对数据及相关工程检查/采集，列入延后证据。
独立CPU权重/源统计/bootstrap复算通过（归一化误差≤1.34e-6），报告归档；
当前单seed预测Probe完成。不放松门槛、不追加seed/epoch，也不恢复旧execution
预测支线；尚未完成多seed泛化Validation或真正flow-space planning。
[Oracle flow实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-hand-flow.md)。


## Ref12：oracle flow → OOF E/I → Y 链路

按用户ref12在原分支完成现有数据的matched任务预测Probe，配对数据/相关工程
和execution predictor继续暂停。3环境fold的H/PCA/标签stats仅使用各fit环境；
688source每行仅用hold预测，166test用冻结ref11 full-source模型。六新Cm fits
300updates、六同容量Y heads500updates，E12/I14和ref8step9起Y8保持原义。

直接actualflow→Y误差.45788 vsH.63951，改善28.40%（CI12.23..41.48%）；
预测E/I→Y.54712，改善14.45%（CI−1.57..29.47%）；GT E/I.45440，改善
28.95%（CI10.61..42.74%）。R.4991（CI−.0639..1.2790），固定链路gate
UNPROMISING，保留直接flow与GT的PROMISING任务信息，不否定所有flow/Cm。
Hybrid.43694相对直接flow改善4.57%但CI跨零，unique Cm增益UNCLEAR。
OOF source Chunk I.61550 vsfull-source in-sample.39092，后者从未进入Y训练；
OOF/full-test分布与有限优化预算保留为限制，不唯一归因于任何机制。
GPU6主11.72s；全部input/foldnormalizer/FK/权重/OOF/统计重放误差0，零新仿真。
独立CPU核对所有8个Y标签、sourcefold统计/权重/OOF/shuffle/bootstrap/R，
归一化replay≤2.17e-6，未见影响结果的实现缺陷；72Task tests、仓库验证通过。
不追加epoch/seed或进入planner；未来改变模型须另立可判别的Decision实验。
全局matched训练策略Cm utility仍OPEN。
[Oracle flow任务链实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-flow-task.md)。

## ref13：Oracle-Y Utility Gate 已完成

用户 ref13 重新启用同前缀配对候选采集，在原分支完成逆向必要性 Probe。
先修复 GPU pipeline/异步前缀污染：最终用GPU PhysX+CPU tensor pipeline、
GPU6 frozen policy，四个同步fork组。32current-only前缀（29s3/3s7）在候选前
冻结；fresh baseline/repeat字节一致，24候选branch前缀差0，32×7结果完整。
原异步46cohort和失败候选保留为工程记录，不作GateA科学证据。

GateA固定合同UNPROMISING：baseline稳定Z23/32，GT-Y选择24/32，gain3.125pp
(anchor95%CI0..9.375)，未达≥5pp/lower95>0；GT-Z候选上限25/32。
仅两处潜在rescue，其中一处由候选顺序平局选中；另一处全部短Y相同，baseline
step89才越过掉落高度阈值，成功候选到step90也仅高于阈值1.838mm。
因此停止当前短Y/固定七候选selector，不把小正点估计当正式控制收益，也不
扩成全部Y、E/I或Cm无效。改变U系数不能拆开相同完整Y标签的候选。
GateB加噪、GateC表示比较和GateD执行/PPO均未启动；旧execution forecast和
进一步MSE调参继续暂停。未来需先冻结具有有意义长期机会的目标/候选合同。

GPU真实point-flow归档14panel FK/live最大tip误差1.714e−5m，Y/Z重建完全一致；
包含样点、代码/mesh和原始panel哈希。独立raw标签/组装/PD/selector/CI复算
全部一致，仓库验证通过。零新fit/policytraining。native累计1679.61s，
产物约420MiB，预算内；78Task/pairedsim+7coldcontract tests通过，1skip。
共享solver四组mosaic、强s3条件群体、forcepair代理及短Z horizon仍限制外推，
不把anchorCI当group数值独立或在线混合选择已验证。最终matched训练策略
Cm utility仍OPEN。[实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-y-utility.md)。


## Ref13_1 修正：滚动 GT-Y 可分性

完成纯CPU保存轨迹审计，零新模型或仿真：32×7候选的原Y/Z和tau0标签精确
重放，保持32步窗口、原utility、每8步cadence。124个Z不同的候选pair中50个
初始utility平局；46个在滚动后正确区分，涉及8/32原前缀，提前29–47steps，
记录路径可分性PROMISING。这些pair并非独立实验；8前缀中6个baseline本已
成功，env33已由one-shot救回，不能写成8个新增rescue或rolling策略收益。

env33 middle−/grip+在tau16分开、提前32steps；env36 baseline/thumb−在所有
已保存每8步查询tau0..56仍平局，dense诊断tau57/58才差1、提前32/31steps。
原90步记录的tau56窗口终点88，下一查询tau64需未来到96（缺6steps），不能
补齐或把dense诊断当主cadence成功。后续候选H已不同，无法拼接反事实切换。

因此ref13 one-shot no-go保留，明确不关闭rolling short-Y；后续优先新的共享
当前状态rolling GT-Y干预证据，而非直接改long-Y/扩大authority或继续predictor。
实际rolling收益/可执行上限仍UNCLEAR，原GT-Z25/32只是旧分支机会ceil。
主审计1.43s、产物不足1MiB、14合同回归测试通过；源码/数据哈希和边界冻结。
独立CPU全部rolling标签/risk/U、事件/提前量/统计复算一致；4pair缺后续查询，
不额外声称持续性。去掉这4pair后的42pair仍覆盖全部8信号前缀。
旧execution forecast/PPO和MSE调参继续暂停，最终matched训练策略Cm utility
仍OPEN。[滚动审计卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-rolling-gt-y.md)。

## Ref14 主链 / ref14_1 rolling oracle 已完成

用户更新ref14_1，当前只推进真正same-current-state RollingGT-Y Oracle Control。
保持32步Y/U、七个K8候选、每8步重规划与原稳定Z90；新的真实组合执行产生
下一轮状态，禁止候选世界状态拼接。每次fresh冷重放整条实际动作前缀，
PhysX solver历史由重放重建；旧25/32仅历史one-shot机会，不是rolling上限。
原32anchor/四组、已有暴露cohort复用，属于机制Probe，不提供独立Validation。

工程smoke163.16s完成：7anchor新baseline/repeat/旧状态历史动作90步结果
EXACT；4个非零组合动作之后，两次下一状态重放仍EXACT。代码与协议已本地
提交，主运行commit337d3a6，GPU6/7最多4进程（每卡2），<=7200s/4GiB。
固定utility<=1.25，所有baseline达到上界时，baseline优先平局可精确省去
六个候选，不改变选择，不编造其Y；其他情况完整同状态七候选。
主实验外部中断后已由本会话恢复（恢复入口a949b62），原progress保留，当前
进度见同run目录resume_progress.json；原短Y/候选/执行源码与协议哈希未变。
固定32-anchor gate 已完成：baseline 23/32，实际 same-current-state rolling
GT-Y 27/32，救回4、伤害0，净增益12.5pp；paired bootstrap 95%区间
3.125--25.0pp，lower95为3.125pp，达到卡片预先固定的 PROMISING gate。
四组全部完成，baseline repeat/prefix 和 actual mixed path 检查通过；这是暴露
cohort上的机制 Probe，不是独立 Validation，也不能替代最终 trained-policy
Cm-on/off。结果文件为
`outputs/cm-interaction-oracle/rolling-oracle-control-s263-s264/result.json`。
预测器、PPO尚未启动；离线 Y ranking/noise tolerance Probe 已完成：43个完整计划、
339个同状态面板；sigma .10 的 median pairwise accuracy=0.758，sigma .20=0.696，
离线筛选状态 PROMISING。它只提供预测器的排序门槛，不提供 noisy rolling Z
retention；下一步仍需冻结 predictor 输入/输出并做真实 rolling intervention。
原7200s/4GiB主实验预算不重置。
用户授权ref14_2并行工作已完成并本地提交912aa1a：只读完成组数据资产、固定Y
时序诊断、已有flow与tinyFK审计、ranking/noise评估合同；17新测试通过，无
新仿真或训练。[并行交付](../src/task/cm-interaction-oracle/docs/ref/ref14_2_progress.md)。
[固定协议与进度入口](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-rolling-oracle-control.md)。


## Ref14_3：actual-flow learned-Y ranking 已完成

按用户ref14_3进入预测器阶段：已有完整32anchor/fourgroup输出重新打包，
GPU FK重建120点两段actualflow，无新PhysX。四折anchor隔离，source内三折
E/I OOF，matched H/Direct/Bottleneck/Hybrid/GT_EI；保持原Y8和U。
339complete panels/2276strict pairs：Direct57.38%、Bottleneck57.82%、
Hybrid57.95%，均未过冻结70%离线gate，固定合同UNPROMISING。GT_EI67.36%。
三臂flowshuffle退化CI均正，保留动作敏感信号；Bottleneck/Hybrid相对Direct
仅+.44/+.57pp，CI跨零，unique E/I contribution UNCLEAR。
GPU0训练30.79s、savedweight重放3.95s且误差0；独立只读CPU/NumPy核查一致。
H约50%的微小偏差来自跨batch的1.19e−7舍入，仅5strictpairs，原始证据保留。
110Task tests/1skip与仓库changed验证通过。无新增rolling learned-Y、execution
或PPO；不将当前模型、预算、selectedoracle support的负结果升级为全局否定。
Source ranking也仅约.585–.711，下一步最小Decision可另行冻结source-panel
ranking objective比较以区分拟合/泛化问题；不直接扩epoch/seed或消耗仿真。
[Ref14_3协议与结果](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-actual-flow-y-ranking.md)。
最终trainedpolicy matched Cm-on/off utility仍OPEN。


## Ref14 baseline source_e260：冻结 critic 排序已完成

用户指定先比较critic与Y，使用同一source_e260 PPO checkpoint，无新模型训练。
原ref13的32anchor/七候选/fourgroup面板完整冷重放，28候选执行与原H、
动作、physical32以及整个height/pair/valid90 EXACT；独立critic replay误差0。
原PPO奖励含2approach/10held_lift/5lift_progress，已按源公式从保存状态离线
重建；最初遗漏shaping的评分保留为诊断，不作为原PPO评分证据。
同one-shot面板baseline23/32、GT-Y24/32；reward8+gamma^8V8为20/32，
救回1、伤害4；V-only19/32。原冻结gate UNPROMISING，暂不推进这个critic
评分的rolling执行或Cm→V，不改系数寻找正结果。独立组件/排名/统计复算通过，
112Task tests/1skip。重放RNG错误的旧运行保留并排除；修复后总主采集690.254s，
低于原1200s cap。这里只是one-shot潜在结果组合，不能与实际rolling GT-Y27/32
当作同协议对照，也不否定所有critic或全局Cm假设。
[协议、修复和完整结果](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-frozen-critic-ranking.md)。
最终trained-policy matched Cm-on/off utility仍OPEN。


## Ref15：training-only GT auxiliary 有界 Probe 已完成

用户授权以source_e260 PPO为起点，让GT真实h8交互loss直接塑造actor z，
推理仍仅actor；四臂均追加64epoch/131072交互/3072更新。完整96新frame0
episode持续45tick且以后不drop：PPO62、条件GT21、shuffle51、stopgrad62；
source本身3。条件GT较PPO差42.71pp，更多后续掉落。该cohort不同于旧32anchor
GT-Y/critic，不能跨协议比较。A/D整native权重/RMS与逐episode结果EXACT，
独立GT/action/mask/reward/MC/成功统计审计通过，未见实现缺陷导致负结果。

冻结gate为UNCLEAR：条件GT heldout loss仅改善2.24%，未达到5%学习门槛；
同z return readout较PPO改善2.52%，弱于shuffle。原生critic独立于actor z；
readout针对finite source-policy MC与MC-sourceV residual，不是当前策略advantage。
本轮无策略收益，但不能否定已充分学会GT监督的路线。停止固定h8/lambda.05
短配方，不扩epoch/seed/系数、不拟合Cm、不进入Validation。若重开先另冻最小
Decision核实aux可学性/独立覆盖；不把训练reward或loss升级为utility。
训练429.28s/1701.87GPU-worker秒，评价222.44s/552.89GPU-worker秒；117Task
测试通过/1skip及changed验证PASS，原始失败smoke和完整证据保留。
[协议与完整结果](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-gt-interaction-aux.md)。
最终trained-policy matched Cm-on/off utility仍OPEN。


## 新路线：cm-pointflow-effect-pretrain 数据预检

用户明确指定新分支`cm-pointflow-effect-pretrain`，先按Task ref1检查SPIDER
retarget_full是否可用于点流/effect预训练。四来源各一条Inspire，120文件checksum
通过；750帧qpos/qvel/ctrl/time及portable MuJoCo模型加载/FK点流工程检查PASS。
完整Inspire清单1946条，当前只下载4条及全部依赖。尚未开始大规模训练或PPO。

下载SSL EOF已用curl解决。Python3.12/MuJoCo3.7隔离环境可加载原始场景；
原Isaac环境未改。SPIDER18维ctrl不等于现有PPO action，实际轨迹违反当前native
耦合最大约1.017rad，因此native control replay仍需适配/漂移检查，不能假定直接
可用。数据支持样本级离线几何点流提取，不代表Isaac dynamics或策略utility。
[预检及后续数据合同](../src/task/cm-pointflow-effect-pretrain/docs/DATASET_PREFLIGHT.md)。


## 2026-10-06 主工作树删除后的恢复边界

用户确认删除了Ref2Dex-agent-baseline，当前工作树原.git和outputs均指向该目录。
从远程Ref2Dex-agent恢复至1afe075的历史，创建当前目录独立.git并保留
cm-pointflow-effect-pretrain分支；使用index-only read-tree，未checkout或覆盖工作文件。
当前Task未push的提交对象身份未找回，但代码/文档文件还在，保存为恢复快照。
旧.git指针和outputs链接在artifacts/git-recovery-20261006保留，新outputs为本地空目录。
被删除的checkpoint、rollout和结果数组未恢复；此前实验卡所指原始输出现不可访问，
不能把已保留的文档摘要当作原始证据仍可复算。Git恢复不是数据恢复。
本轮OakInk2产物使用Task声明的artifacts/cm-pointflow-effect-pretrain目录。


## 2026-10-06 OakInk2 annotation-only 100-sequence audit

按用户ref2，本Task当前只推进OakInk2，不混入SPIDER/其他数据。100条固定序列及
五个资产包已校验下载，6.29GB；GPU0完成双手11点/物体512点/SE(3)审计877.62s。
646565手帧、477序列物体对，无缺失mesh、无效rigid pose或帧号缺口。
4961958重叠窗口原始静止90.40%；program筛选731579窗口，运动384679，
静止47.42%，支持继续小规模预测Probe。120Hz的h8只有66.67ms，不能沿用30Hz解释。
发现少量相邻物体跳变，影响1344个program窗口；训练前检查并屏蔽跨异常窗口，
按sequence划分、运动/静止平衡，再决定大规模投入。尚未开始预训练/PPO；
数据工程可用不等于可学性、因果effect或策略utility。Git已独立恢复，但历史产物
删除仍影响6个旧测试；新OakInk2流程不依赖这些路径。
[结果与下一步](../src/task/cm-pointflow-effect-pretrain/docs/OAKINK2_DATA_RESULTS.md)。


## 2026-10-07 WM30/K24 架构实现与接口完成

用户指定架构并确认program锚点+0.5m局部物体、三组独立训练及B验证shuffle、
全627条标注/最多4GPU/整组24h。完成1cm稀疏卷积64/128/256、384维8层场景/
4层动作/6层dynamics Transformer、多物体24步SE(3)及解析点损失，37208777参数。
局部选择已核实使用当前mesh几何中心，避免部分标注原点偏离>20cm导致选错。
6项合同测试及12条现有数据三组6步GPU smoke通过，初始化hash一致，checkpoint
保存/恢复/独立推理接口通过。属于工程检查，不作模型质量或policy utility结论。
全量下载（HTTP range续传）与GPU3预处理运行中；627完整校验及split审计后自动
启动GPU0/1/2三组，每组40000更新、有效batch16，共享24h截止时间。不启动PPO。
[实现与运行入口](../src/task/cm-pointflow-effect-pretrain/docs/WM30_INTERFACE.md)；
[固定实验协议](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-oakink2-wm30-k24.md)。


## 2026-10-07 WM30 全量训练中间状态

627条全部下载/预处理完成，5060616窗口，501/70/56序列划分。实际运行提交ec4d918，
GPU1/2/3三组训练已运行约5h17m；H33310、H+A27796、shuffle28943/40000，
最慢组估计还需2.3–2.5h，仍在24h预算内。数值/进程正常，但最近运动锚点h24验证
点EPE三组均约28.17mm，与静止基线几乎一致，尚无A收益。不同更新数仅是中间观察。
四个验证窗口的checkpoint核对确认实际近零预测（GT最大90mm、预测最大约0.05mm），
标签非零、A梯度存在，FP32同样近零；不足以确定优化原因或否定方法。保持当前实验
到固定matched更新数，不改活跃配置/不追加预算；原始中间检查记录保留在Task产物。

## 2026-10-07 用户指定 ref3 PointWorld-small 替换

用户指定参考本地PointWorld、按Task ref3换架构并启动训练，随后明确授权停止旧训练。
旧三组均保存checkpoint/正常退出，H40000、H+A33464、shuffle34823，不能形成
matched最终比较；原始产物保留。当前分支不变，Mission/claim不变。

新实现直接使用PointWorld PTv3-small、128维/patch128/1cm，统一场景与24步双手
点输入、object pooling+SE(3)小head，逐时域train-only归一化和官方逐帧运动soft权重。
实测50,495,881参数，完整监督保持每物体512点。重复栅格的SparseConv不稳定问题
在正式运行前修复：唯一voxel聚合+inverse恢复，评估全部层固定序列化顺序。
7合同测试通过/1旧模型可选GPU测试skip；三组6步smoke初始化相同且checkpoint精确
保存恢复，独立val推理/恢复加载通过。80步重复运动batch loss3.016→0.225、
末端anchor EPE44.3→16.4mm，只证明学习链路/小batch拟合，不作泛化结论。

复用627条完整数据及原sequence split，GPU0/1/2三组独立40000更新、有效batch16、
共享24h上限。运行目录outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007，
启动前统计已冻结、接口已验收；实时进程/进度/运行提交见group_status.json。
实际运行提交f507f18，launcher3942111，worker3942116/3942117/3942118；三组均已
完成正式更新，loss/梯度有限、初始化/data/stats/config身份一致。早期每步约1.2–1.4s，
40000步粗估14–16h，共享24h上限不变。本次改动范围verify通过；全恢复分支仍有
6个依赖已删除baseline产物的历史测试失败（另102项通过），不影响新路线输入。
不启动PPO；最终trained-policy matched Cm-on/off utility仍OPEN。
[协议及证据](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-small-wm24.md)。

ref4只读诊断已完成：229训练序列/512窗口，528action点平均合为160.55个空间voxel，
同手同关键点跨时间碰撞影响83.70%的点，下一版优先保留时间身份。5mm逐帧selector
原始权重偏低，但归一化后moving-anchor相对均匀监督系数中位1.217，不能据此声称
所有moving监督仅剩2%–3%；部分旋转样本仍被相对降权。审计自身遗漏padding mask
的初次归一化份额已排除并修正，原训练mask正确；三组继续原配方/原预算，结论仍UNCLEAR。
[诊断协议、结果和修正边界](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-ref4-input-loss-audit.md)。

## 2026-10-08 consequence twin contract hardening

在独立工程复核后，twin v1 继续收紧：native prefix 按 30 Hz control tick 与 60 Hz
physics frame 分离计数；zero-step replay 只接受自然的空列表，显式空二维
action 仍核对 18 维；RNN 必须使用显式 `is_rnn/state` sentinel；Python/NumPy/Torch RNG
格式、非空 physics/history/controller provenance 均在入口拒绝缺失值；native adapter
自动冻结 direct tensor、scalar 和 Enum（包括 `_state_init`）inventory，并提供 CPU/GPU
Torch RNG restore helper。Task tests 为 114 passed，compileall 与 diff-check 通过。当前仍没有 native branch runner
和真实 twin branch 数据，不能把合同测试当作 twin coverage 或 evaluator 科学证据。

## 2026-10-09 native GPU serial reset/replay Probe

为区分“GPU fresh process 行为不稳定”和“同一 simulator 内固定控制能否配对”，新增
`engineering_serial_replay` worker（实现提交 `bf96546`；随后在 `11fd62d` 中强制该
模式只能使用 `gpu_physx_gpu_pipeline`）。在 seed282、原生 GPU PhysX/GPU pipeline、
单环境、同一进程内依次 reset 五个 arms：baseline、zero_repeat_1、zero_repeat_2、
positive、negative；baseline 的执行控制被完整记录，其他 arms 在 tick48--71 重放并
只给候选加固定 residual。实现通过 `tools/verify.py --changed`、py_compile 和
consequence-evaluator 171 tests。

r29 baseline 最高抬升 `0.8241 m`、held484；两个 zero replay 轨迹和 outcome 完全相同，
candidate positive/negative 分别为 `0.7971 m`/485 与 `0.6489 m`/310。每次 reset 前后
absolute frame 保持不变，reset RNG anchor 相同。object pose、hand points、gap、q/dq
和 object velocity 的 zero-pair query-relative p95 均为零；但 contact force p95 仍为
约15.82/17.15，history p95为2.0，首个 task-buffer divergence 是 reset 后 stale
`_curr_obs`。这些 trace 没有覆盖 PhysX warm-start/contact-manifold/island cache，不能
称 full-state twin。冻结 physical-bank/TCC 的只读审计给 baseline 与两个 zero 同一
`Y=0.0411695`，positive/negative 为 `-0.0145127`/`0.0044708`；该结果只说明本次
固定控制下 label 可复现，不是 Gate1 utility 证据。

因此 serial reset/replay 保留为 frozen-control 的工程噪声校准容器，确认它可以保持
GPU baseline 行为，但 strict same-state Gate1、reactive-policy candidate ranking、
evaluator/PointWorld 和正式 utility claim 仍关闭。raw packet、serial noise audit 和
serial GT-value audit 位于
`outputs/consequence-evaluator/gate1-gpu-serial-engineering-20261009-r29/`；不修改
reference bank、Y、policy weights。下一步需要 hidden PhysX state fork/restore，或预先
声明 field-specific noise 与重复数的 statistical paired design，再决定是否重启 Gate1。

为检查 serial arm 顺序，又在提交 `79cdbf3` 下做了 candidate-first fresh launch
（`baseline → positive → negative → zero_repeat_1 → zero_repeat_2`，r30）。与 r29
相同 seed/checkpoint/backend 的 baseline parity 失败：r29 为 `0.8241 m`/484 held，
r30 为 `0.8281 m`/483，action 在 tick47、物理轨迹在 tick48 已出现差异。因此不能
用 r29↔r30 的候选或 noise 差异归因于 warm-cache arm order。r30 内部两个固定控制
zero 仍保持机械字段 exact，而 contact/history divergent；其物理-bank Y 为
baseline/zero `0.0437091`、positive `0.0189845`、negative `-0.0122399`，仍是
engineering-only label diagnostic。该 order Probe 标记 `UNCLEAR`，serial 只保留为
单次 frozen-control field/noise calibration 容器；后续若没有冻结 baseline action stream，
不再追加 order-only launch。

## 2026-10-09 8-env GPU group decision note

当前决定问题：同步 GPU group 是否还能在不改变原生 GPU pipeline 行为的前提下，提供
可用于候选统计配对的短窗口执行容器。

关键证据：4-env group 的 r19 在一次 launch 中达到 `0.8262m/481`，但 r27 同合同
只有 `0.0891m/5`；因此把 env pair 当独立样本或直接进入 Gate1 都不成立。serial
r29/r30 的冻结 executed stream 两个 zero arm 在 object/q/dq 上可重复，但 contact
与 history 仍暴露隐藏 solver 状态。SDK 审查也未发现 GPU PhysX hidden-state clone/
restore API，不能用公开 setter 构造严格 twin。

root 选择先做一次低成本 8-env、72-step engineering Probe：env0 baseline，env2/3
候选，其他 env 为 zero roles；actor 总推理行数固定为 256（每 env 32 copies），
避免把 group size 与 actor GEMM 行数同时改变。先跑两次 fresh seed282 launch；若
两次 env0 都不能达到短窗口原生行为，停止 group 扩展并回到 serial frozen-control
统计设计；若两次通过，再决定是否跑四次完整542-step cluster Probe。所有结果仍为
`engineering_only`，不修改 Y/reference/policy，也不能替代 docs/user/完整链路.md
规定的 strict same-state Gate1。

该 Probe 已在 `01c085a` 下运行一次（r31），使用原生 GPU PhysX/GPU pipeline、seed282、
8 env、32 copies/env（总 actor rows=256）、72 steps。初始 semantic state exact，所有
角色 query 前 controls/done exact，但 env0 最高仅 `0.1439m`、held8；env1 zero 为
`0.1537m`、held8，候选角色几乎不抬升。相同 packet 的 zero-role query 前 object
p95 约 `9.37e-4m`，q velocity p95 约 `0.151`，history p95 `0.0231`；query 后
object displacement p95 约 `0.200m`，candidate effect 未形成稳定 margin。TCC 只读审计
的 progress-start range 为 `1.86e-4`，严格 contract 失败，deadzone 选择 baseline。
因为单次短窗口已低于预设 `0.20m` 筛查线，按停止条件不再跑第二个 8-env launch，关闭
group 扩展；r31 及其 audit 仅保留 engineering evidence，下一步回到 serial
frozen-control/launch-level statistical design 的决策，不进入 formal Gate1/evaluator。

group 关闭后的下一步 Decision Note：为区分“group solver 退化”和“固定控制下候选
标签本身无信号”，再做一个单进程 serial 72-step frozen-control Probe。该 Probe
只记录 baseline executed stream，随后重放同一 prefix 到 tick48 并运行两个 zero
repeat 与 ±candidate 到 tick72；它不评估完整 hold/place，不进入 strict scorer。
若 serial 的 zero Y 仍一致且候选选择稳定为 baseline，近期停止继续仿真，保留
strict Gate1 blocker；若选择不稳定，则只记录为 execution-noise evidence，不把它
升级成 utility 结论。

在 r32 结果出来后，新增一次且仅一次的 serial-cluster Decision Note。r32 的
reactive teacher 在 72 步内抬升约 `0.272m`，超过 `0.20m` 短窗口线；两个 frozen
zero 的 object/hand/q/dq/velocity 几何字段仍 exact，但 contact/history 继续漂移。
因此下一 Probe 使用 native `gpu_physx_gpu_pipeline` 的同进程单 env，先运行一个
`reactive_teacher` 保存 executed action stream，再按固定交错顺序重放
`frozen_zero_1..5`、`positive_1..2`、`negative_1..2` 到 tick72。候选只在
tick48--71 使用 `clip(teacher_action + residual)`，zero 臂全程重放 teacher
action；teacher 不作为 counterfactual zero。审计只报告 paired noise、机械字段与
contact/history 分层结果以及描述性 TCC/Y 对比，不计算置信区间、p 值或选择器结果。
若 teacher lift <`0.20m`、action/reset 合同失败、机械 zero noise 超预设阈值，或
两类 candidate 没有任一重复在主要机械字段达到约 `2x` zero floor，则关闭这条
统计工程路线。无论结果如何，packet 保持 `engineering_only`，不进入 strict
Gate1/evaluator/PointWorld，不修改 reference bank、Y 或 policy。

r33 已按该合同完成（代码 `e1c71b4`，packet 为
`outputs/consequence-evaluator/gate1-gpu-serial-cluster-20261009-r33/serial-cluster.pkl`）。
teacher tick72 最高抬升 `0.2706m`；action、done、reset frame/RNG、schedule 和
executed residual 合同审计全部通过。五个 frozen zero 的 object/hand/gap/q/dq/
object-velocity query-relative p95 均为 `0`；contact/history 仍单独出现隐藏状态
噪声。两次 positive 和两次 negative 的机械轨迹分别重复一致，candidate 相对 zero
的机械 effect 明显超过声明的 field floor；contact-force effect/noise 约为
`1.6--2.5x`。physical-bank/TCC 只读值为 teacher/zero `0.0051948`，positive
`-0.0117869`，negative `-0.0129771`，progress-start range 为 `0`；这只是
固定 executed stream 的影响诊断，不是 candidate utility 选择。

r33 因此通过 cluster 的短窗口继续条件，但不通过 strict Gate1。按预注册上限，
下一步最多再做一个完全相同的 fresh r34 launch，用于检查 launch-level teacher
行为和 paired effect 是否能重复；r34 若 teacher <`0.20m`、机械 zero 不再 exact，
或 candidate effect 不再重复，则关闭 serial-statistical route。无论 r34 结果如何，
不跑 evaluator/PointWorld、不改 Y/reference/policy，也不把十个 arm 当独立样本。

r34 已完成并通过同一 audit（packet 为
`outputs/consequence-evaluator/gate1-gpu-serial-cluster-20261009-r34/serial-cluster.pkl`，
read-only audit 为其 `gate1-gpu-serial-cluster-20261009-r34-audit/audit.json`）。
teacher tick72 最高抬升 `0.3001m`、held13，超过短窗口线但不构成抓取成功；action,
done, reset, schedule, residual 和 hash 合同均通过。五个 frozen zero 的主要机械
字段 query-relative p95 仍全为 `0`，contact hidden noise 继续存在；positive/negative
各两次重复的 mechanical effect 一致。TCC 只读值为 zero `-0.0094461`、positive
`-0.0117899`、negative `-0.0131493`，两类 candidate 都低于 zero，因此没有
candidate utility 信号。

按预注册上限关闭 serial-statistical cluster。它证明了同一 executed action stream
下可见机械 replay 在两个 fresh launch 中可重复，并量化了 contact/history hidden
noise；它没有恢复 PhysX hidden state，也没有把 strict same-state Gate1 变成统计
替代。Gate1、evaluator、PointWorld、Execution Bridge 和 MPC 仍未开始。下一步回到
可验证执行合同设计：优先评估 ACT/open-loop chunk 或原生 PhysX hidden-state
fork/restore 的可行性；在其通过前不改 Y/reference/policy，不fit evaluator。

### ACT deployment-conditioned chunk Decision Note

当前问题是 clean hold-audit 上的 ACT-like chunk 是否覆盖原生 GPU deployment
state。现有 checkpoint 在 clean holdout 上 chunk MSE 约 `1.6e-4`，但直接喂给
r33/r34 teacher query history 时 MSE 为约 `4.2e-4--6.3e-4`，说明存在
deployment-distribution gap。root 先做最便宜的离线判别实验：以 r33/r34 的
`reactive_teacher` history/action 为两个 launch-level episodes，leave-one-launch-
out 训练一个同结构 native 24-step proposal，冻结 train-only standardizer，并在
另一 launch 上评估 full chunk/first-action error。若跨 launch 仍优于现有 clean
checkpoint，下一步才值得收集更多 reactive deployment distribution；若不优于，
关闭该 route-conditioned proposal 路线。该 Probe 不运行仿真、不改变 Y/reference/
policy、不训练 evaluator，结果仍为 engineering-only。

该离线 Probe 已完成，代码提交 `94ef7df`，产物为
`outputs/consequence-evaluator/act-native-chunk-deployment-20261009-r3/`。使用 r33/r34
两个 serial-cluster 的 `reactive_teacher`，每个 launch 13 个 stride-4、非跨 episode
窗口；每折只在另一个 launch 上评估，standardizer 只由训练 launch 拟合，模型选择也只看
训练 launch（修复了初版用 held-out MSE 选 checkpoint 的 validation leakage）。r33→r34
held-out MSE `1.902e-4`、first-action MAE `0.00542`，r34→r33 为
`1.827e-4`/`0.00521`；均优于 clean checkpoint 的 `1.932--2.017e-3` 和
route-s3 checkpoint 的 `3.532--3.963e-4`。两次 launch 的 physics/controller/backend
identity 与 arm schedule 均显式相同，GPU2 运行约21秒、显存约361--369MiB。

因此 deployment-conditioned proposal 在动作空间标为 `PROMISING`，但每折只有一个
launch、13 个重叠窗口，不能形成泛化、行为、utility 或 Gate1 证据。该结果只保留“未来
若获得预算，先收集更多 native reactive-deployment episodes 再拟合 proposal”的工程
方向；不启动新的 ACT candidate/serial simulation，不fit evaluator，不进入 PointWorld。
严格 Gate1 仍等待可验证的 PhysX hidden-state fork/restore 或新的执行合同。

### Clean-only continuous source audit

上一条 offline Probe 暴露了 continuous loader 的来源漏洞：`unlabeled` episode 也可能
带有 intervention。新增的 `clean_only` 合同只保留 manifest 中
`assigned_phase=clean`、`perturbation_tick=-1` 且数组中 residual plan 全零、
`plan_known` 全真的 episode；默认 loader 行为保持不变。用
`continuous-20261008-{train,val,test}-r1` 和 `expert=airplane_base` 审计后，三个 split
各有 8 条 clean episode、1204 个 24-step windows；精确 `s3_airplane_lift` 每个 split
只有 1 条。所有入选数组通过零 residual/finite/known 检查，来源 manifest 中冻结的
`GRAB_00000260.pth` SHA 与当前 history RMS 一致。

这只支持一次 episode-held-out 的 offline chunk fit，不支持 route-specific 泛化或 native
行为结论。下一步最多使用 train split 内的 episode holdout，并在独立 val/test manifest
上报告动作空间误差；在行为 screen 通过前，不启动 GT ranking、evaluator 或 PointWorld。

该 bounded fit 已完成，产物为
`outputs/consequence-evaluator/act-native-chunk-clean-airplane-base-20261009-r1/`。
训练使用 6 条 train episode，2 条 train episode 做 holdout；独立 clean val/test 各 8 条
episode、1204 windows。新 chunk 的 val/test MSE 为 `2.405e-4`/`3.397e-4`，低于只用
train target mean 的 `5.855e-4`/`7.483e-4`，也低于旧 clean checkpoint 的
`7.905e-4`/`8.857e-4`；first-action MAE 为 `0.0118`/`0.0125`。因此该动作空间
Probe 标为 `PROMISING`，但不外推为 native 行为、utility 或泛化结论。下一步只做一次
`open_loop24` native behavior screen；不跑 `receding8`、candidate ranking、evaluator
或 PointWorld。

首次 behavior screen 的 4-env group 没有满足 baseline 前提：reactive teacher 只有
`0.129m/7`，reactive repeat 为 `0.637m/237`；同一提交下独立单环境 baseline 恢复到
`0.8165m/485`，因此该 group packet 标为 `INVALID_IMPLEMENTATION`，不能作为 chunk
负证据。进一步按 motion 分层后，当前 broad fit 在真正的 `s3_airplane_lift` clean
val/test 上 MSE 为 `5.823e-4`/`6.669e-4`，旧 clean checkpoint 为
`2.962e-4`/`2.054e-4`。这说明 aggregate windows 的改善没有覆盖当前抓取路线。

下一步 Decision：利用已有 `continuous-hold-audit-20261008-r1` 中 24 条同一
`airplane_base/s3_airplane_lift` clean episode，在显式 audit-only、冻结 expert RMS 的
合同下做一次 route-specific offline fit；若 episode-held-out 仍不优于 train mean/旧
checkpoint，则关闭该 proposal route；即使优于，也必须先解决 group baseline contract
才可做行为 screen。

route-specific fit 已在 audit-only source 上完成，产物为
`outputs/consequence-evaluator/act-native-chunk-route-s3-engineering-20261009-r1/`。
20 条 episode 训练、4 条 episode holdout；holdout MSE `1.783e-4`，旧 clean
checkpoint 为 `1.985e-4`，train-mean 为 `9.207e-4`，first-action MAE 为
`0.01136`。这是约 10% 的动作空间改善，样本来自同一 hold-audit wave，仍只标为
`PROMISING/UNCLEAR`，不构成行为证据。由于 group baseline contract 已失败，暂不再
启动 native chunk screen；ACT 只保留旧 r2 的 open-loop24 行为工程基线。

### Native GPU CUDA synchronisation probe

旧 ref13 的原始 binary panel 已确认不在当前工作区或历史 baseline worktree；当前只剩
`tmp/ref13` 的日志和 `raw_stat_replay.json`。历史 panel 虽可由源码确认有 32 步
trajectory/native-q，但没有完整 24 步 candidate controls、现行 `object_pose`/11 点手坐标/
timestamps，也属于 GPU PhysX + CPU tensor pipeline，不能重标为当前 Gate1 packet。

为区分 fresh native GPU 行为波动是否主要来自 host/kernel 异步排程，在不改变 backend、
seed、checkpoint、Y 或 reference bank 的情况下做了 `CUDA_LAUNCH_BLOCKING=1` Probe。
两个 fresh 单环境 baseline 都达到约 `0.79m/483 held`，行为 spread 明显比之前的
full-to-zero launch 小；但接触 buffer 仍在 tick44 分叉，完整 state 只在 `44/543` 个
hash 相同，不能声称 same-state replay。相同设置的 4-env group env0 只有
`0.186m/10 held`，zero-pair hand/q/velocity/object-velocity 噪声门也失败。结论是
同步设置可能改善单环境行为稳定性，却没有形成候选执行合同；关闭 synchronisation-only
route，不启动正式 Gate1/evaluator/PointWorld。审计见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gpu-cuda-sync.md`，
摘要产物在 `outputs/consequence-evaluator/gate1-gpu-sync-probe-20261009-r1/` 和
`gate1-gpu-sync-group-20261009-r1/`。

### Native GPU group role/layout probe

进一步审计发现，4/8-env 组把 env0 的世界原点/首行路径与非零 origin env
配对时，固定 controls 下 tick1 就出现约 `3.46 mm` hand、`0.0116` joint-position
漂移；这早于 tick44 的显著 contact-force 分叉。新增的工程入口支持 bounded
`--query-tick`、非默认 `--zero-env-pair`，并把 query 和 scene-layout hash 纳入
replay identity。

一次 tick32 pre-contact query 仍在 tick1 漂移，4-env baseline 只有 `0.245 m`；一次
小 spacing 诊断也没有恢复行为或 noise gate（且会改变 broadphase，只保留为诊断）。
8-env、256 actor rows 下把 zero pair 改为非零 origin 的 `[1,4]` 后，选中 pair 的
geometry p95 通过但 pairwise gate 和行为筛查失败；更稳定的 `[4,6]` pair geometry
接近 exact、72 步 baseline `0.234 m`，candidate effect 与 pairwise gate 仍失败。
预先限定的 full 542-step `[4,6]` confirmation 最终只有 `0.153 m/9 held`，行为、
pairwise 和 candidate-effect gates 全部失败。结论是 group role/layout route 关闭；这些
packet 不能进入 strict Gate1。证据见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gpu-group-contract.md`
及 `outputs/consequence-evaluator/gate1-gpu-precontact-group-20261009-r1/`、
`gate1-gpu-spacing-group-20261009-r1/` 和
`gate1-gpu-nonzero-pair-group-20261009-r1/`、`r2/`、`r3/`。

### Native PhysX binary hidden-state audit

在 Python surface 审计之后又做了归档 Isaac Gym/PhysX native binary 的只读符号检查。
`libcarb.gym.plugin.so` 确实包含 `PxCloneDynamic`、`PxCloneStatic`、`PxCloneShape`
以及 PhysX object/RepX serializer helper，但没有可调用的 `PxSerialization` scene
collection create/serialize entrypoint、`PxCollectionExt` 场景收集器，或 contact-manifold、
warm-start、solver-island、GPU cache 的 fork/restore boundary；`libPhysXGpu_64.so`
也只有内部 contact/solver/cache routines。actor clone 只复制公开 actor 属性，不能复制
scene membership、joints、sleep timer 或 hidden solver state；RepX/object serialization
也不是 live solver snapshot。原始 symbol/hash 清单在
`outputs/consequence-evaluator/hidden-physx-binary-api-audit-20261009-r1/audit.json`，
实验卡为
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-hidden-physx-binary-api-audit.md`。
因此 strict Gate1 blocker 已从“可能遗漏 Python wrapper”收窄为当前 runtime 没有可调用
的 hidden-state 执行合同；不重复 group/serial/CUDA/CPU/host，也不启动 evaluator 或
PointWorld，等待新 runtime/API 合同或明确的 claim Decision Checkpoint。
