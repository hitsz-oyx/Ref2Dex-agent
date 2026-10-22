# 2026-10-22 负结果审计（CPU，P1 HOLD）

审计状态：历史身份/501任务/评分/逻辑动作与重置已复核；**物理执行资格仍未通过**。
原始0成功、UNPROMISING与失败轨迹全部保留，修订的是它们能否否决相应命题。
本轮停止自己的P1队列，无新训练、新方法评估或仿真；GPU4–7与原flowmatching长训未动。
本报告依据本仓库源码、当次冻结入口、checkpoint、JSON及真实轨迹独立重算；外部审计仅提供问题。

## 1. 四个代码断言：结论与逐行证据

源码基准分别是 `187690962bb338780ed2e479061afca14a283fc0` 与实际PPO冻结
`3a0779f6981ee01ffb8e637abb9b7b3fea2437f3`。root用 `git show COMMIT:PATH`
逐个导出六个相关源文件，证据根：
`outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1/code-evidence/`，
`index.json`记录两版本、路径与行号；以下冻结行号在两提交对应源码中一致。
当前runtime的行号因本次scale防线增加3行，与冻结源码分开引用。

| 断言 | 核实判定 | 证据文件:行号与含义 | 否决范围 |
|---|---|---|---|
| a. PPO不是v6 | **确认** | `src/task/wam-rl-policy-v3/tools/run/inspire_measured_official_ppo.py:28`导入V4Runtime、`:48`从BC config实例化；冻结`wam_v4_runtime.py:277`载入checkpoint、`:278`严格检查v4 schema、`:304`导入SharedWAMV4；`src/task/ptv3-shared-wam-v4/src/ptv3_shared_wam_v4/model.py:137`reshape HORIZON×2×11×3，HORIZON=24。当前runtime相应为`:280–282`、`:307–308`，tau-only head`:364`也明确24×2×11×3。 | 53,347结束训练回合/0完整成功是真实v4运行现象，**不是v6的负证据**，不是独立评估成功率。 |
| b. UNKNOWN无成功路径但仍准入 | **确认** | `task_annotations.py:37`无有效接触区间标UNKNOWN；`measured_official_env.py:41`仅TRAIN/engineering PASS/prefix>0准入；`task_scheduler.py:172`与`:174`分别是RELEASE/HOLD，只有其调用的`:196–200`可赋complete=True。UNKNOWN无调用分支。上述文件均位于`src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/`；实际冻结副本同路径位于`tmp/inspire-measured-official-20261020-r3/frozen/`。 | UNKNOWN结构性不可能成功，污染完整任务分母；**不能据其占部分样本推断全部任务都不可能**。 |
| c. 1570时27792/53174是voxel gate | **确认** | r3 `rank-0/curve.jsonl:1339`保存53174/27792/0；另两rank相同行是已做全局reduce的重复值。冻结runtime`:329–342`拼scene+有效hand，float32 floor((xyz-origin)/voxel_m)，要求index∈[0,65535]；env`:102`设workspace，`:146`立刻done。当前runtime`:332–345`。 | **52.2661%**结束回合触发表示域终止；不是机械工作空间或物体越界判据。触发可能源于真实发散，不能仅由gate名字断言这些都是误杀。 |
| d. 没reward只是日志缺项 | **确认**；“PPO没收到reward”**否认** | 冻结env`:145`算r、`:159`有return_sum、`:169`返回reward；实际`outputs/wam-rl-policy-v3/inspire-measured-official-20261020-r3/rank-0/runtime-official/dexgrasp/algorithms/rl/ppo/ppo.py:181–184`把rews交给storage，`:214`compute_returns；同目录`storage.py:60`用reward算TD residual。runner`:98–109`覆盖ppo.log却没序列化return/phase/reward分项，三rank curve均缺这些字段。 | 可认定可观测性缺陷，不能认定奖励没有进入PPO，不能由缺日志反推“奖励全0”。 |

r3各rank最终manifest均STOPPED1575：全局53347回合、success0、workspace27869
（52.2410%）。顶层manifest仍RUNNING是陈旧汇总。三rank是一个同步模型，不能算3独立seed。

## 2. 实际501份task.json、采样权重与时间预算

入口：`tools/audit/audit_task_reachability.py`（本Task相对路径），按实际r3 source_manifest
逐个打开被准入的501份task.json和admission.npz；同时读取其原始task，不能只统计重建的任务。
详细逐任务路径、事件、前缀、duration、mode、source_fps与预算在
`outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1/task-reachability.json`。

| 统计单位 | HOLD | RELEASE | UNKNOWN | 结论 |
|---|---:|---:|---:|---|
| 准入后的501个任务 |416|21|64|UNKNOWN=12.77445%；不能假设按序列均匀采样。|
| 56449次实际motion初始化draw |44967|2168|9314|UNKNOWN=16.49985%；包括首次初始化及reset，**不是结束回合分母**。|
| 原始完整录制task（同501条） |11|490|0|安全前缀截断会将原本RELEASE改成HOLD或UNKNOWN，任务语义改变。|

按512环境/50物体分配及物体内motion轮转的名义UNKNOWN权重为16.95252%。实际权重
来自最终三份native manifest的501维motion_visits，逐序列分子与nominal_weights均已保存。
visits=rank indexed_resets+512，三个rank累计draw56449；不能把它当iteration1570的UNKNOWN
结束回合比例。历史没有每回合sequence结果表，该比例**读不到**。

- 前缀65/501没有≥3样本的有效接触区间，129/501未曾建立coupled反馈；不能把所有prefix
  都说成“只剩接近”，但其中确有失去接触/抓稳阶段者。
- 前缀抬升≥5cm至少1秒（按真实elapsed时间）251/501，持握代理245/501；旧30样本口径
  抬升254/501，三例差异来自30个状态仅覆盖29/30秒。
- TaskScheduler预算：acquire15s、release10s、episode60s、终态连续1s。已有teacher首次
  coupling晚于15s=0；参考终点时间+1s超过60s=0。**通过这一必要时间上限不等于实际任务可达**。
- UNKNOWN在任意时间预算下都不能成功；HOLD/RELEASE逻辑有合成正控制成功，但本批真实
  参考未提供完整TaskController正控制，不能认证这些501个任务均有已验证的成功路径。

## 3. 真实轨迹评分回放与正负控制

只读取原始真实录制，不新增物理仿真。501份记录采用冻结3a0779f6的真实TaskController
逐帧observe；pose/hand/velocity/force-valid/support及timestamp按记录对齐。另写独立几何
path/lift/hold/release/full评分器；耦合见证来自既有admission标注，不伪称独立测得的接触真值。

| 相同501条来源，评分范围 | 抬升≥5cm elapsed≥1s | force/距离/无支撑持握代理 elapsed≥1s | 独立几何完整条件 | 原TaskController完整成功 |
|---|---:|---:|---:|---:|
| 准入安全prefix |251|245|10|0；499录制结束删失，2 path_no_confirmed_milestone|
| 全部原record，prefix task |467|446|12|0；490删失，11 path_no_confirmed_milestone|
| 全record、原始raw task |见同一全record|见同一全record|未与prefix定义混算|0；492删失，9 path_no_confirmed_milestone|

全record超出了安全prefix，包含已知失稳段；446/501不能认证其为合格全任务示教。
独立path按真实路径顺序经过8mm/.12rad/8mm平均CAD误差的参考点，501/501可重放经过；
这不是TaskScheduler的候选接触/动作阶段/确认进度合同。独立10/12例全是RELEASE；HOLD416例
没有满足独立终态“在终点稳定且抓稳1s”。这些差异是待解释协议/反馈问题，不可把宽松评分
的10例当作原TaskController应成功的既定事实，更不能把两个不同评分器的0说成一致验收。

合成测量反馈控制（不冒充真实抓取）：HOLD在2.0667s、RELEASE在2.5s成功；UNKNOWN即使
path complete也不成功；插入一帧不稳定/force invalid、移除released/远离证据均会被拒绝。
终态计时真实elapsed≥1s；浮点边界可能延迟1tick，合成HOLD实际1.0333s。raw记录中成功持握、
掉落、仅接近、未释放的代表性情况及完整逐帧对齐补证见同目录产物与后文补充。

### 四类真实轨迹代表例（不是四个完整任务正控制）

| 类别 | 真实sequence | 秒制证据 | 解释 |
|---|---|---|---|
| 持握代理成功 |s10_apple_eat_1|tick56–114=1.8667–3.8s，elapsed1.9333s|满足持握代理；完整TaskController仍未成功，不能冒充全任务正例。|
| 耦合后掉落代理 |s10_airplane_fly_1|coupled tick84=2.8s，peak113，drop160=5.3333s，落差0.7491m|drop超安全prefix52；不写成安全prefix里的失败，也不认证“滑落”因果。|
| 仅接近/接触未抓稳 |s10_alarmclock_pass_1|首次有效接触tick41=1.3667s，prefix无coupling|接触不等于抓稳。|
| 未释放 |s10_cubemedium_inspect_1|tick41–71=1.3667–2.3667s无release witness|任务HOLD，不要求release；未释放本身不是失败。|

全套逐任务统计与四例细节在`task-reachability.json/representative_categories`；没有真实完整
成功档案可作为正控制就明确保留缺口，仅用合成反馈检验评分器逻辑可达性。

### 真实分歧的逐帧定位（CPU补证）

- `s6_bowl_drink_1`：计划RELEASE tick171，独立几何到达tick185；tick173失握退回ACQUIRE，
  最终里程碑35/41，下一里程碑源frame171。它没有被确认走完整条抓持路径。
- `s2_hammer_use_2`：计划RELEASE tick233，独立到达tick233；tick235失握退回ACQUIRE，
  最终101/103，下一里程碑源frame234。末段确认不足，RELEASE阶段未获准。

独立评分放松了连续持握路径、确认里程碑和释放前0.2s path settle；到达几何终点/后来稳定
释放不等于完整在线合同通过。这两例**不构成已定位TaskScheduler实现bug**。仍需以满足
整个合同的真实正控制验收，不能通过事后放松评分把它们当方法成功。
对齐trace及秒制事件详见CPU产物 `task-reachability.md` 与各 `aligned-*.json/.md`。

### P0历史持握秒制的边界修订

旧P0使用连续30个post-control状态的occupancy代理n/30，不等于首末观测elapsed。
同一已保存结果用max(0,n−1)/30重算：动作回放15/16、标准GT0/16、live-GT0/16、
healthy0/16、FM由旧1/16变为**0/16**；FM唯一30样本bout实际29/30=0.9667s。
最长持握elapsed均值原动作2.05625s、标准GT0.052083s、healthy0.039583s、FM0.060417s。
旧卡与raw summary原值保留为legacy，追加本修订；GT资格未通过的判断不变。
重算与故意将0.9667s报作1s的mutant拒绝证据在`invariants.json/legacy_p0_timing`。

## 4. voxel越界细节与不可恢复的历史缺口

UNKNOWN与workspace的统计分母不同且可重叠，不能相加为“解释了多少负结果”。
历史统计可核到，27792个结束回合的**首个越界点类别/轴/坐标没有保存，读不到**。
本次不从scalar倒造点云，也不为补日志启动新仿真。新增可复用`voxel_first_failure`诊断
严格沿用实际V4 float32 floor gate，返回scene索引或hand(slot,joint)、轴、world/anchor
坐标、origin与index。为验证工具，CPU回放另一批已存在P0 flow轨迹20个状态：与真实V4 gate
20/20一致，2个失败状态均可定位；这不是历史r3的失败分布。

| P0独立旧轨迹样例 | 首点/轴 | anchor xyz(m) | world xyz(m) | origin/index |
|---|---|---|---|---|
| s2_cup_pass_1，tick253=8.433s |hand(right,wrist)/y|[-.090843,-2.006290,-1.038306]|[1.287252,-2.641349,.024682]|[-2,-2,-2]；[190,-1,96]|
| s1_fryingpan_cook_2，tick291=9.7s |hand(right,wrist)/x|[-2.004243,-.500194,-.758553]|[1.634111,-2.122313,.022807]|[-2,-2,-2]；[-1,149,124]|

voxel_m=.01，允许65535上界；首个gate点index512表示512个scene点之后的首个有效手点。
两例都发生在各自准入prefix之后（104/141控制），不能倒写成P0主prefix指标失败。
产物 `voxel-diagnostics.json` 含各状态、前一帧、checkpoint schema/step与故意越界错误检测。

## 5. 动作生效与两环境交错reset：CPU能证明的范围

单一真实初态，12个active轴各±.001，完整保存 raw→clipped→canonical12→expanded18→
PD target。实际原生`third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py:38–50`
按AST原样提取执行：wrist为q+scale*a，finger为offset+scale*(1+a)/2及原生coupling。
**24/24扰动均在正确PD坐标产生预期变化**。另对16条原始真实轨迹2316个已准入control
逐帧applied/q重算PD，最大绝对误差0；实测12/12轴均有非零q变化/dq，数值逐轴保存在
`invariants.json/action`。这排除“动作完全没接到PD”说法，**不证明稳健执行能力**。
档案中wrist角速度峰值约360.4rad/s、单步旋转坐标变化约37.1rad；不把“非零”写成健康控制。
**同状态逐轴扰动后的实测q/dq未做**：没有启动新物理，档案观测不能代替反事实实验。

真实MeasuredOfficialEnv/TaskController/reward逻辑以两环境CPU replay transport运行5步；
第2步reset0、第4步reset1。检查episode归属、所有已记录native字段、未reset环境的
controller/tracker/history/previous/obs字节相等，reset目标的history长度1、previous0、
任务/事件游标0、reward unpaid、force-invalid、tau-invalid，均通过。故意清空未reset环境
history后明确FAIL。**这不是PhysX隐藏状态/原生reset物理隔离的认证**；历史reset审计也未
保存足够双环境全状态，P1继续HOLD。未做的两项必须先作为有界工程验收，不开新方法实验。

## 6. 可复用不变量检查器与scale回归

`src/ptv3_shared_wam_v6/negative_result_invariants.py`与
`tools/audit/run_negative_result_invariants.py`本轮19项PASS/19项故意错误被拒绝，另有voxel
首点检查PASS与一个越界mutant被拒绝，合计20个边界检查。覆盖τ[0]第一未来帧/插值τ(0)=0、
30/60Hz的8步时长、旋转、21→11语义索引/左右slot、native/canonical逆、action clipping、
measured686各字段位置及归一化往返、future-label隔离/mask、独立reward分项与求和、
event reset、共同评估身份及交错reset。只要正控制失败或mutant没被发现即输出FAIL。
v6 forward使用真实SharedWAMV6与CPU geometry测试seam，未认证GPU PTv3 backend；评估身份
一致检查不等于新跑过确定性物理评估。所有scope写入artifact，不能只看PASS标签。

`load_geometry`存在object而请求scale缺失时静默用rows[0]的bug确认。已改为拒绝不存在/
非有限scale（当前`wam_v4_runtime.py:145–152`），并保留已有scale正控制。
修前实际回归1 failed（DID NOT RAISE），修后通过；此更改不修改历史冻结源码、checkpoint
或正在跑的训练。没有证据表明r3请求过不存在scale，故此bug目前**不能解释历史全部0成功**。

## 7. 历史逐卡审计表

下表实际身份/入口/当时实现可认证度/协议gate分别判定。TP=`src/task/trajectory-policy`，
WR=`src/task/wam-rl-policy`，V3=`src/task/wam-rl-policy-v3`，V6=当前Task；run均在对应outputs。
完整附注见[逐卡证据底表](negative-result-card-audit-20261022.md)。


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


## 8. 应当排除的结论

1. “53347回合零成功证明v6/WAM未来预训练无效”。实际是v4桥，混合不可成功UNKNOWN，
   训练重复回合而非独立评估；未得到完整任务正控制。
2. “零完整任务成功就是零抓取/抬升”，“workspace就是物体越机械工作空间”，“PPO没收到reward”。
   均与实际代码/代理指标/训练入口不符。
3. 把r2–r5旧残差实现bug、teacher干预8/8、teacher placement未过的复合gate、normalized
   Huber筛选、位置held代理或中断P1缺checkpoint，作为整个方法被否决的证据。
4. “UNKNOWN解释了全部负结果”或“修掉voxel gate就会成功”。前者覆盖有限，后者缺历史
   终止点与因果证据；禁止直接扩大voxel范围或删除gate来制造成功。
5. 当前P1三臂无收益/C失败。仅A/B seed191部分训练，C与其他seed未启动，科学评估0/45。

## 9. 应当保留的结论与我的判断

保留历史v4训练确实0/53347完整成功与其频繁voxel退出；保留某些修正后的executor在
相应单序列hold45=0/8而teacher8/8的局部物理失败；保留旧残差integration的hold45=15/16，
不能因未过long-held+clip复合gate抹成全失败。保留goal-chain离线语义错配与该checkpoint
对照误差，不从它否定闭环或架构。P0的机器人动作回放15/16持握而标准GTτ0/16也保留，
但回放全任务没有正控制，适用当前v4训练执行器/v6适配桥，不认证新direct executor。

**最致命的归因错误是模型身份混淆：它使最大53k负证据对v6完全失效。**
若问“什么最可能解释大量实际零成功”，我优先怀疑**任务标注/终态与反馈确认协议没有经过
真实完整任务正控制**：原参考有245/501持握却TaskController0/501，416个HOLD连独立
终态稳定1s也0；它的影响面比UNKNOWN子集更广。这是有证据支持的研究判断，**还不是
已定位的单一实现bug或因果证明**。voxel退出占52.27%是最大直接终止类别，但可能是
动作/跟踪失稳的下游，不能自动称为表示gate误杀。UNKNOWN本身则是已证实的结构性bug。

修完后第一个应做的是**实际成功路径与动作/reset的有界工程资格验收**：选明确HOLD与
RELEASE、真实可执行参考且有完整终态停留的少量任务，冻结同一个真实执行器/入口，
同初态回放动作、机器人GTτ、预测τ；在训练前先让独立评分与TaskController对实际正负
轨迹逐帧一致，并做同状态12轴实测q/dq及两环境native reset隔离。身份实际为v6才登记v6。
这只是下一步建议，本轮未执行；通过前P1保持HOLD，不恢复RL或引入新方法。

## 10. P1中断与复核命令

owner STOP已生效，A191停止139更新，已更新284672/实际286720交互；B191停止72更新，
已更新147456/实际149504交互，合计实际436224。各多采一个未更新rollout，不能只报已更新
成本。C/其余seed/科学评估均0，阈值步数/AUC/成功率对比待定。supervisor顶层FAILED是
受控STOP后仍索取不存在milestone checkpoint的编排问题，不能当策略学习失败；产物保留。
本次不改runner或恢复任务。详情见P1卡、`p1-stop-request.json`与各`interaction-costs.json`。

执行环境：graspenv `/home2/wyy/miniconda3/envs/graspenv/bin/python`，Torch2.4.1+cu121、
NumPy1.24.4，CUDA_VISIBLE_DEVICES空，task回放4 CPU workers，约223秒。临时文件均项目tmp。

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/tmp/negative-result-audit-20261022-r1"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
AUDIT=outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1
TASK=src/task/ptv3-shared-wam-v6
"$PY" "$TASK/tools/audit/audit_task_reachability.py" --workers 4 --output "$AUDIT"
# 已有501结果时只补代表性对齐、checkpoint/stop核验：
"$PY" "$TASK/tools/audit/audit_task_reachability.py" --augment-existing --output "$AUDIT"
git show 3a0779f6:src/task/wam-rl-policy-v3/tools/run/inspire_measured_official_ppo.py
"$PY" "$TASK/tools/audit/run_negative_result_invariants.py" --output "$AUDIT/invariants.json"
"$PY" "$TASK/tools/audit/audit_voxel_failures_cpu.py" --output "$AUDIT/voxel-diagnostics.json"
PYTHONPATH="$PWD/$TASK/src:$PWD/src/task/wam-rl-policy-v3/src:$PWD/src/task/trajectory-policy/src" \
  "$PY" -m pytest -q -p no:cacheprovider --basetemp="$TMPDIR/pytest-review" \
  "$TASK/tests/test_negative_result_invariants.py" \
  src/task/wam-rl-policy-v3/tests/test_workspace_failure.py \
  src/task/wam-rl-policy-v3/tests/test_measured_official_reset.py \
  src/task/wam-rl-policy-v3/tests/test_measured_reward_bridge.py \
  src/task/wam-rl-policy-v3/tests/test_tau_contract.py \
  src/task/wam-rl-policy-v3/tests/test_inspire_contract.py \
  src/task/wam-rl-policy-v3/tests/test_batched_geometry.py
```

targeted tests：26 passed in6.78s。全库verify仍报6张既有旧卡metadata缺字段；
不为了审计重写无关历史卡。发布时遵循[负结果协议](negative-result-hygiene.md)七类证据。

## 11. 上一轮交付状态与环境限制（后续已解除）

26项针对测试通过（6.78s）；owned33文件的Python语法、Markdown链接及结构检查PASS；
`git diff --check`与自动INDEX一致检查PASS。全库verify仍受6张原有v2卡metadata缺字段阻断，
没有把这个旧问题伪报成全部通过。审计新增产物约5.1MiB，磁盘剩约146GiB。

已按机主§5尝试暂存本轮文件，实际失败：`.git/index.lock: Read-only file system`。
当前会话将.git设为只读且禁止提权，本轮**没有新commit，未push**，未改凭据。
当前本地HEAD=`ded133758829a68849df6f8baae2221dff013ce6`，这是已有HEAD，不是本轮提交。
`git ls-remote origin main`实际失败`Couldn't connect to server`，远端SHA**读不到**，
不能核对两端相同。原始错误与完整owned文件清单在CPU产物`git-delivery.json`；
代码、卡片、STATE和审计文档均保留在工作区，未覆盖无关`docs/PROJECT_MEMORY.md`。

机主12:3x以完整权限启动后，已开展修复与危害量化；后续以[修复报告](negative-result-fixes-20261022.md)为当前状态，上一轮只读错误记录保留作历史。
