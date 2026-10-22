# 2026-10-22 历史负结果逐卡 CPU 审计

本表只审计既有卡片、运行 JSON、保存的冻结源码和当前入口；没有启动训练、推理或仿真。
原始数值不改，`UNPROMISING` 不自动等于方法被否定。保存的旧 audit.json 只作为可读证据，
本次没有重跑其 GPU 审计。当前源码定位用于解释合同，不能自动证明旧版本已经包含该实现。

## 最重要的身份结论

`P-20261020-inspire-measured-official-ppo` 的 r3 **使用 WAM v4，完全不是 v6 PPO**。

- `outputs/wam-rl-policy-v3/inspire-measured-official-20261020-r3/rank-0/manifest.json`：
  `git_commit=3a0779f6981ee01ffb8e637abb9b7b3fea2437f3`，
  `wam=outputs/ptv3-shared-wam-v4/local-control-final-20261019-r1/step80000-model.pt`，
  `status=STOPPED`，iteration1575，completed_episodes53347，successes0，workspace_failures27869。
- 实际 argv 的入口位于
  `tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/tools/run/inspire_measured_official_ppo.py`。
  冻结文件第28行导入 `V4Runtime`，第48行读取 **BC checkpoint config 中的 wam**；
  第59行把该身份加入 resume identity。当前同名入口第28/48/59行仍对应这个实现。
- 冻结 `wam_v4_runtime.py:277–279` 读取 checkpoint 并严格要求 v4 schema；当前
  `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/wam_v4_runtime.py:280–282` 对应同一检查，
  行号因后来的scale修复移动。`wam-rl-policy-v3` 是执行器 Task 名字，不是 WAM 版本。
- r3 使用 Scheduled686 `inspire-scheduled686-bc-20261020-r4/best.pt`（step0 的共同 BC 起点），
  冻结 CAD encoder，actor/critic 是686维官方 UniDex PPO 适配，不是 v6 在线入口。
  policy 为 graspenv / Torch2.4.1+cu121；physics 为 dexplore_v120_train / Torch2.0.1+cu118。
- 53347 是同一训练运行中累计结束回合，且跨 rank 汇总、回合重复初始化；不是53347条独立
  测试序列，更不是“v6 在53347次独立评估中成功率为0”。0次完整任务完成不能替代抓取/抬升率。
  运行顶层 manifest 仍写 RUNNING，rank 的最终 manifest 已 STOPPED；顶层状态陈旧。

用户指定的两个版本现已由 root 使用 `git show` 补证：
`187690962bb338780ed2e479061afca14a283fc0` 与
`3a0779f6981ee01ffb8e637abb9b7b3fea2437f3`。本子审计仅CPU读取root导出的源码快照，
未调用Git。证据根为
`outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1/code-evidence/`，
其 `index.json` 记录完整commit、相对文件路径及命中行；两个子目录
`18769096/src/task/`、`3a0779f6/src/task/` 保存各自六份源码快照。
两版本的runner第28/48行均导入/实例化V4Runtime，runtime第277–279行均严格校验v4
schema、第304–305行均选择SharedWAMV4；v4 model第137行均reshape为HORIZON24×2×11×3。
这直接证实两版本的模型身份，**不是把main修复倒灌到旧版本**。当前runtime相应
SharedWAMV4选择位置已移到307–308行，冻结引用仍用304–305行。

另外两条旧路线的模型身份也不能写成v6：

- coordination/learned-short-goal 的继承文件
  `outputs/wam-rl-policy/task-readout-20261016-r1/inputs/v3-step80000.pt` 已用
  graspenv、`CUDA_VISIBLE_DEVICES=''`、`torch.load(map_location='cpu')` 独立读取：
  checkpoint schema及config schema均为 `ref2dex.ptv3-shared-wam.v3`，step80000。
  这不是只按文件名推定v3；其运行manifest的schema/step字段本身为null，需由checkpoint补证。
- r7残差PPO manifest展开的实际组成是：
  `goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt` 的decoder overlay，
  `wam-control-context-20261013-r3/stage2_step4500.pt` 的robot control context，
  以及 `trajectory-policy/wam-v2-control-fit-20261012-r1/chunk.pt` 的Chunk-R。
  这条旧GoalPointWAM/World/R链也不是后来PTv3 WAM v6。

## 判定词与来源缩写

- **实现问题**：实际入口违反它自己的目标/动作/时间合同；它产生的轨迹仍是真实轨迹，但不能
  否决原本未被正确实现的方法。
- **协议假象**：指标名、校准条件、初始化/干预、分母或训练/评估口径与所下结论不相符。
- **保留的真实现象**：在明确实现、场景、预算及代理指标下实际发生的行为，不外推到整个路线。
- TP=`src/task/trajectory-policy`；WR=`src/task/wam-rl-policy`；V3=`src/task/wam-rl-policy-v3`；
  V6=`src/task/ptv3-shared-wam-v6`。下表的 `run` 均在对应 Task 的 `outputs/<Task>/` 下，
  表中源文件相对各 Task。所有执行版次优先采用 **实际 manifest**，不以卡头最后一次提交覆盖。

## 逐卡审计表

| 实验卡 | 实际身份、入口与模型 | 独立读到的训练 / 评估证据 | 实现与 gate 判定 | 排除的推论 / 应保留的结论 |
| --- | --- | --- | --- | --- |
| P-20261017-gt-tau-new-sequence | TP；卡头886f481b，r4/gt-tau实际manifest **c0c859565f1a6c6aec0863b102c34fadd6bba503**；`collect_competent_pd.py`/`probe_official_generator.py`；冻结 tau/R 与官方 Inspire teacher；native Torch2.0.1+cu118 | r4/gt-tau/result：teacher hold45=8/8、placement8/8；source-PD hold45=8/8、placement5/8；GTτ/R hold45=0/8、placement0/8、median6.5帧、clip0.174% | r1固定542步跑279帧且自然终止reset、r2错误partial_filtered物体坐标均应排除；r4才是坐标有效运行。runner把source-PD placement≥6/8也作为校准，故原result是UNCLEAR，卡的UNPROMISING是另行限定的抓持判断 | 保留“该固定R在新序列不能保住hold45，而teacher/source-PD可保持”；不保留“完整placement方法失败已被合格正控制认证”，更不是learned executor/v6/PPO负证据 |
| P-20261017-direct-tau-executor | TP；卡头e8ea6019，BC/执行实际manifest **5fb0c65b17b89ea5bfa620f84cd286d2f2d206c7**；`fit_direct_tau_executor_bc.py` + `probe_direct_tau_execution.py`；DirectTauExecutorPolicy / OracleC，单s3 teacher，native Torch2.0.1 | BC环境划分val L1 .01550，unseen .11206；direct unseen0/8 hold45与success，oracle-C也0/8；teacher分别hold45 8/8、placement4/8或5/8 | **有实质合同问题**：将参考future减live手；encoder mean-pool时间，单运动32环境并非32序列。执行gate还要求teacher placement≥6（源码79–81），本次未满足。Oracle-C只是10维当前几何/速度代理 | 排除对“参考τ语义错误”“C无用”的方法否决；保留该有缺陷单序列BC实现的实际0/8及泛化差。r3修复不能倒算旧运行已正确 |
| P-20261018-direct-tau-executor-r3 | TP；卡3774db68；BC/执行manifest均为运行时HEAD **5f04a758ec46a2db01d507396d26cae7625b0c61**，卡说明文件随后提交3774db68；`fit_direct_tau_executor_bc_r3.py`/`probe_direct_tau_execution_r3.py`；TemporalTauExecutorPolicy，Torch2.0.1 | 40序列28/6/6；val/test/unseen L1=.04030/.04584/.04047；431控制，teacher hold45 8/8、placement5/8；direct hold45 0/8、success0/8、clip0、终点32.46mm | reference_tau与order-sensitive encoder修复可在当前代码定位，保存manifest绑定输入文件；没有完整冻结源码时不能仅凭卡的后提交号认证每行旧实现。**完整gate仍混入teacher placement≥6，正控制仅5/8** | 可保留修复后这个BC/一条s1序列仍无hold45；不能认为通过了完整placement资格、不能否决τ或证明缺C；40序列不是40独立物体/目标分布，单seed八环境不等于八训练seed |
| P-20261018-direct-tau-dagger | TP；卡40d5d4b5，rollin/fit/执行manifest **44c8b7ea9c2a795ed00318ebe66d0a552bb7778a**；`probe_direct_tau_dagger_rollin.py`、`fit_direct_tau_executor_dagger.py`、`probe_direct_tau_dagger_execution.py`；r3 temporal warm start；Torch2.0.1 | 431×8=3448 live-state teacher queries；query val action L1 .41245→.11906，original val .06503；独立无干预执行0/8 hold45、0/8 placement、终点831mm；teacher hold45 8/8、placement7/8 | r1 producer末尾保存时重复slice导致packet失败，排除；r2有效查询是当前动作/PD，**不是从learner状态重规划未来τ**。query fit5env/val3env属于同运动、相关tick；本次teacher placement校准7/8确过6/8 | 保留“一轮当前状态动作DAgger的离线改善没有转化本序列自主保持”；排除“完整future-trajectory DAgger无效”“τ路线被否决”；大离轨状态label可执行性/恢复域仍须另外区分 |
| P-20261018-direct-tau-dagger-pd | TP；fit manifest **0dc48974615aef5620ae4a9d92017093d15f31d3**；执行 **357e4a6dea6ed671e2f0b3ed7c75ea96793da5aa**；`fit_direct_tau_executor_dagger_pd.py`、共用dagger_execution；temporal executor，Torch2.0.1 | query PD RMSE .6475→.3388，original val .1359；执行direct0/8 hold45/placement，终点40.20mm；teacher hold45 8/8、placement4/8 | PD loss优化信号不同于真实动作执行资格；执行复用gate仍要求teacher placement≥6，4/8不合格。接近物体/终点40mm不能代替持握，query绝大多数来自严重离轨状态 | 保留PD空间拟合改善但没形成hold45；完整placement的否决需标校准不足；不能认为normalized PD目标已足以执行、不能据此加C或关τ路线 |
| P-20261018-absolute-plan-executor | TP；卡d066138a；旧query fit manifest **25844582ca920ef123b9108b4e1b7c30d75a562c**，warmup rollin **8b18605ead982d1b5bf124ce4ffbe8b32fd58f8d**，最终fit **4d21a1fb70de15f2acb0c75866c96a98fc8178b0**；`fit_absolute_plan_executor_pd.py`/`probe_absolute_plan_dagger_rollin.py`，AbsolutePlanTauExecutorPolicy；Torch2.0.1 | 旧query held xyz152.49mm/rot.482rad；新packet teacher先120步后β.25，实际intervention .46114，混合rollin hold45=8/8；最终query xyz210.52mm/rot.4335/finger.1816，normalized Huber .4926，source xyz41.66mm | **8/8有teacher干预，绝非自主executor成功**。最终没有自主物理评估。卡称native-unit offline gate，但真实脚本:203的PROMISING阈值仅normalized_huber<.2，未独立约束xyz/rot/finger原生阈值；原生误差确有保存 | 保留这一固定拟合/数据分布未过原normalized筛选；排除“自主8/8”“绝对锚点已证明/已否定”“物理原生精度gate已验证”。catastrophic占比是恢复域诊断，不是科学独立样本数 |
| P-20261013-wam-residual-ppo | TP；卡头ea2575aa不等于修正版；r7实际 **a2f85d4b7a28c26faf9d10378e3c2974a30d80a5**；`train_wam_residual_ppo.py`，旧冻结WAM+World+Chunk-R、Gaussian latent clip(base+.15latent)；不能归为v6 | r2–r5存在zero bootstrap、伪造reset H4/previous、reset-local lift、KL退避/Adam恢复问题；r6初始化前失败。r7四env20×96更新，4phase zero-residual最小gap均<30mm；eval frame0四env hold0、max rise≤1.53mm；audit22reset/20bootstrap最大差1.91e−6 | **r2–r5是实现污染负结果，不能关残差方法**。r7修后训练held .2346主要来自phase62已被抓住的reset，不能算自主起抓；r5→r7物理初始化也变了，非单变量改进 | 保留r7该短预算不能从frame0起抓，以及旧故障真实轨迹；排除残差RL/WAM普遍无效、phase62奖励证明起抓进步、修正版审计能洗白旧版本 |
| P-20261020-inspire-measured-official-ppo | V3；实际冻结 **3a0779f6981ee01ffb8e637abb9b7b3fea2437f3**；`inspire_measured_official_ppo.py`→V4Runtime→v4 step80000；Scheduled686 DEV-best step0起点，官方UniDex PPO；native Torch2.0.1/policy Torch2.4.1 | r3 rank manifests STOPPED1575；53347结束训练回合，完整TaskController成功0，workspace27869，source mismatch0、参数sync0；有GPU/CPU后端诊断和明确迁移记录；没有本run固定共同初态独立评估成功率 | 实际三rank是一个同步模型，不是三个独立训练seed。0成功可能来自task goal、可达性/动作/重置/反馈等执行链，日志不足以唯一归因。top RUNNING是陈旧状态。源码schema严格v4 | **禁止写成v6 PPO失败**、禁止0/53347冒充独立评估成功率、禁止从完整任务0推导抓取0；保留这个v4执行桥真实训练未观测完整任务完成与大量workspace退出 |
| P-20261011-goal-conditioned-bc | TP；**0b5466e9914d14d54f7a8b9e5c10eaf7371ed468**；`fit_goal_conditioned_bc.py`，456维MLP，live state+8future hand/object goal；graspenv/Torch2.4.1 | r3完成3000更新，13008/4336行，val active L1 .00672288<.05；tracker自有执行动作labels；工程packet只供几何目标，packet动作/q不作labels | 环境划分同一运动，不是序列/任务泛化；离线gate仅许物理Probe，不证明闭环；实际result elapsed13.159s与卡约12.6s细微不一致，采用manifest/result原值不改历史 | 保留低动作蒸馏误差；排除“低L1已证明executor合格/有WAM收益” |
| P-20261011-goal-conditioned-execution | TP；**0b5466e9914d14d54f7a8b9e5c10eaf7371ed468**；`probe_goal_conditioned_execution.py`，同BC，native/graspenv Torch2.4.1 | 16env×542control，hold45=0、long-held-terminal0、median0、clip0；保存独立audit PASS，requested/applied0，首action对label L1 .00208 | 同一运动、单seed；long-held需433帧且终点held，不能与一般hold45混称，但本次连hold45也0。此前tracker校准是另一波，不是本wave的matched teacher控制；audit只排除已验接线项 | 保留固定BC闭环失败；“covariate shift/hidden state”只获支持非唯一因果认证；排除一般goal-policy/REGRIND/WAM高层结论 |
| P-20261011-goal-conditioned-ppo | TP；**d07b739be20921cc533f469e9802ca510f03bf23**；`train_goal_conditioned_ppo.py`，same BC起点/实际原生动作，graspenv/Torch2.4.1 | 24更新×128control×16env=49152训练交互，actor change .05288，mean reward .10728；result UNCLEAR/engineering_pass | 训练return和更新非成功率；仅一短预算，后续冻结eval是不同卡；不能把训练曲线或本身UNCLEAR改成已认证负任务结果 | 保留确实更新且物理执行；方法效果须以冻结eval卡限定，不能据此否决从零RL/预测预训练 |
| P-20261011-goal-conditioned-ppo-execution | TP；卡run_id写r1，**有效产物r2**、manifest **90e78dd5e6c9c074330b87e0eda4982aea9c27b5**；same执行入口，PPO final.pt deterministic mean，Torch2.4.1 | r2 16×542，hold45/longheld均0、median0、clip0；audit PASS。r1缺provenance/final trace bookkeeping，不能替代r2可审计运行 | 区分原始r1设立失败与r2有效物理负结果；还是一运动一seed、短PPO预算，独立held-out任务/预训练效果未测 | 保留这个24更新策略的冻结闭环失败；排除PPO普遍失败和WAM解码瓶颈 |
| P-20261011-goal-conditioned-residual-integration | TP；BC实际 **eddab7526cf1ba42fd2b4fd61e364b8d587dfb27**，有效exec **1b56d4f7237e7962c2c5ef2b2db2720d964dafdd**；`fit_goal_conditioned_residual_bc.py`/`probe_goal_conditioned_residual_execution.py`，共享GoalConditionedExecutor，1170维/raw latent；Torch2.4.1 | BC6504/2168行，val .0249544；exec542controls，**long-held10/16、hold45 15/16、median483**，clip6.6075%，placement全false；audit字段逐项PASS | 注册gate≥12/16 long-held且clip<1%，两项均不满足，所以状态UNPROMISING；但绝不是“不能抓/保持”。nominal q/velocity来自特权geometry cache；r1–r3 setup失败均排除 | 保留较强实际持握代理和clipping不合格；排除“零抓取”“可部署无future-q planner”“整个残差方法失败” |
| P-20261011-goal-conditioned-residual-offset | TP；**028b5d41b20dbae274fec6f1df15bfab45133a03**；共享残差入口，在raw latent[11]加−.05，其他同上；Torch2.4.1 | longheld0/16、**hold45 12/16、median79**，clip.8187%通过clip半gate；placement全false，audit PASS | 失败是longheld≥12/16未满足，并非clip gate或完全无保持；这个原先thumb basin迁移不成功与普遍残差能力不同；仍依赖特权nominal | 保留offset减少clipping但没恢复长保持；排除“优化clip就能成功”“12/16算注册gate已过”及无先验/可部署结论 |
| P-20261011-goal-conditioned-finger-residual | TP；卡81b59f4，物理有效r2 manifest **b1f1883db6f0d15a259bc29898abe2b44f3b7310**；`fit_goal_conditioned_finger_residual_bc.py`/共享残差exec，wrist由nominal、finger student；Torch2.4.1 | val finger L1 .0208398；exec longheld0/16、hold45 7/16、median5，clip13.5148%，audit PASS；r1control384资源guard终止且无轨迹排除 | wrist/finger拆分未救本实现，不能反推出nominal普遍不对；单motion单seed。完整placement全部false与hold45须分开 | 保留7/16短持握代理、clip/longhold gate失败；排除“完全不能持握”“指头策略一般无效” |
| P-20261016-wam-learned-short-goal | WR；**c85a1837c48ab0ef297f2abccd84d62ac1a38a30**；`probe_learned_short_goal.py`，raw-query/wam-mean读出、旧final80k冻结WAM，shadow60Hz；不是v6执行 | 1792状态1344fit/448held、24/8整物体；两模型固定1000更新；held endpoint零66.73/raw73.32/wam67.61mm，WAM hand/scene32=42.54/44.58mm；audit保存模型重放PASS | gate同时要求endpoint≥10%和两域forecast收益，两臂不满足；是旧混合行为真实未来回归，不是任务最优/可达目标选择；448相关窗口不是448独立实验。没有闭环成功率 | 保留这个endpoint预测桥未过筛选与部分预测增益；排除“未来预训练无助PPO”“预测goal物理不可实现已测”“目标选择已被学习” |
| P-20261016-wam-coordination-bonus | WR；正式 **4b82bd84a635206c4a0fa7c9d95e7a0e66085d5d**，baseline360054d7；shadow_loop + coordination，冻结自训BC step50000+新reference-gaussian，旧v3-step80000 WAM副本；非v6、非零BC | none/geometry/wam各636800交互；fixed DEV128，指标28/26/23/25（BC/none/geom/WAM），nine initial fields exact，bonus重算2.98e−8；单训练seed。WAM目标改变约4.27mm、DEV bonus特异差均值.001295 | **held60_with_lift10cm是物体在目标5cm内且抬升10cm的连续位置代理，不是force持握**。gate WAM比两对照各+5且不退BC未过。参考+观测+bonus共同变，不能归因纯reward；没有发现明确接线bug不等于科学零效应证明 | 保留当前协调接法位置保持未优于BC/none以及弱特异奖励；排除“v6失败”“BC-free”“物理抓持认证”“WAM一概无效” |
| P-20261022-wam-v6-goal-chain-audit | V6；主推理 **c971bca689fa76845128cecb04c2dc43173f069c**，completion报告最终ded133758829a68849df6f8baae2221dff013ce6；`audit_goal_chain.py`/各audit helpers，healthy4371与FM8500；graspenv strictFP32 | completion独立读到DEV3356、TRAIN5241、local-prefix905，FM K8；DEV oracle H/O15.797/14.722→17.374/16.206mm；Planner21.122/25.550→22.786/27.604；并有mesh/GT参照产物 | 匹配shape≠匹配随机变量：训练goal是实现t+24 pose，planner是无保证到达时间的spatial waypoint。Planner重放有oracle录制参考、E_next固定录制标签，缺完整grasp/release/support反馈；checkpoint objectives/steps不同；K8是采样不是8训练seed | 保留语义错配、该checkpoint pair条件敏感性/DEV泛化差和误差成分；排除闭环v6失败、FM架构因果劣势、τ不可辨识定理、仅低相对距离即正确或无穿透 |
| P-20261022-wam-v6-samples-efficiency-3arm | V6；正式冻结 **fe81b6b535c677d87e154aeb232c75fce8aaf213**；outputs/runtime-code四入口；A noτ/B几何/C固定healthy4371，846维PPO/公共H4/dense2048，native2.0.1/policy2.4.1 | owner HOLD后A191 STOPPED139updates/284672已更新交互、B191 STOPPED72/147456；两者训练success0；**科学评估完成0/45**；顶层FAILED因为STOP后缺约定eval milestone（B128、A256） | 只有部分A/B训练，无完整C/三seed/45panels，更无AUC/阈值sample-efficiency结果；工程3env/256env smoke不能填补。实际物理成本另计native step，包括STOP前已采未更新rollout。任务成功可达性/动作重置验收优先于收益解释 | 保留已发生的工程与部分交互记录；**不得宣布三臂无收益或C失败**，也不能用缺checkpoint的queue FAILED说策略方法失败。此卡按root要求不修改，仅在本表审计 |

## 本次核到的具体实现 / 协议定位

1. 旧 direct τ 时间消失：`TP/src/trajectory_policy/tau_executor.py:164`、`:206`
   使用 `.mean(dim=1)`；r3 新类`:236–260`是 position+flatten temporal MLP。
   旧执行 `TP/tools/run/probe_direct_tau_execution.py:236–237` 把参考future传给
   live measured `features()`；r3 `fit_direct_tau_executor_bc_r3.py:87–91`明确调用reference_tau。
   这是不同接口，不能按相同“GTτ”字符串混成一个已验证方法。
2. direct/r3/PD-Dagger 完整gate混入teacher placement：
   `TP/tools/run/probe_direct_tau_execution.py:79`、
   `probe_direct_tau_execution_r3.py:77`、`probe_direct_tau_dagger_execution.py:77`。
   teacher hold45与placement校准分别报告；失败校准不能悄悄变成方法否决。
3. absolute-plan脚本实际selection：`TP/tools/run/fit_absolute_plan_executor_pd.py:203`。
   原生单位误差有报告，但唯一状态阈值是normalized Huber .2；不能重新命名成独立物理验收。
4. corrected residual bootstrap：`TP/tools/run/train_wam_residual_ppo.py:507–524`记录真实
   post-control observation/bootstrap，并保存当时value。这个是**当前/修正版r7**的证据，
   不是r2–r5已经正确的证据。`r7/audit.json`实际resets22、bootstrap_queries20、误差1.907e−6。
5. coordination指标：`WR/tools/audit/audit_coordination.py:60`仅取goal-hold长度≥60；
   `WR/src/wam_rl_policy/shadow_loop.py:146` manifest说明“position proxy”。指标命名不提供接触证据。
6. 历史PPO身份严格从冻结入口到checkpoint绑定；不能以当前V6工具存在证明历史PPO用过v6。
   同样，P1冻结fe81b6b5与较早capacity512点/小smoke版本应分开，不把后修复套回工程数据。

## 实际执行证据的缺口与影响面

- 历史多数卡只有当时HEAD、输入文件hash与JSON结果，没有本次可读的完整冻结源码目录。
  卡写“随后提交”与实际manifest HEAD不一致时，表中都保留两者，**不伪称已核到唯一运行代码**。
  没有检查到的 runtime细节不从当前入口倒推；18769096与3a0779f6的上述v4身份已由root
  导出的各自源码快照补证，其他后来main修复仍不能据此自动认证到旧版本。
- 原生请求/应用PD与policy重放的PASS，只排除它审查过的变换/接线项；不证明teacher恢复域、
  task-success可达性、完整物理无穿透、目标语义或方法收益。
- 单一motion的8/16个同seed环境、同packet高度相关tick、多个DDP rank、同模型K8抽样、
  不同run的累计训练回合分别是不同统计单位；没有一项可自动充作独立训练seed。
- 本次不新增方法实验；保留损坏/中断产物，实施错误的旧运行移出方法比较，协议未合格的
  指标只保留描述性现象。后续若重开实验，先执行同目录
  `negative-result-hygiene.md`的七类证据协议，且另立明确运行身份。
