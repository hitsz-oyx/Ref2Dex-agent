## 2026-10-23 新采刚体96条：来源绑定认证通过，零格61→51

source4/Shadow新采96×10s@30Hz，28,800输出ticks；96/96条及26,304/26,304窗口准入，
96/96条goal可认证。reset前显式指令、同clock时间/hash、reset_target_pose恢复及21点/part
均落盘，独立重算PASS。source4原10个零格全部补上；其余8源未冒充覆盖。
实测总文件约0.504GiB；始终保留≥30GiB。前36条GPU4、后60条GPU5；GPU4第二批有
预检后其他执行器启动的短时并存，发现后批次边界等待，未干预对方；详见资源偏差。
原6k/512面板身份不变，旧goal认证仍0/6,000；新增索引独立、0updates、预测NOT_RUN。
建议下一批N=384、4GiB硬上限，未执行；排后续同面板/同条件/同预算训练对照。
[结果解读表、证据、资源与下一批成本](audit/rigid-source-bound-pilot-20261023.md)。

## 2026-10-23 刚体P0：合法新增0，来源认证0，交回是否新采集

真实1,564个legacy候选逐窗重判0准入/1,564拒绝；coupled registry可绑定，但21点/
validity/contact-event缺证未补造，cm同parent跨view运动不一致另拒绝绑定。原6,000
通过同谓词、0/6,000可认证goal；保守内容去重6,000→5,999，原61.23%时间重叠大体仍在。
phase×数值几何类×source共108格、61格为0，去重前后不变。未扩源/大转换/训练，
新索引未接生产；交回机主是否新采集，并列自报成本。[结果与资源偏差](audit/rigid-dataset-20261023.md)。

## 2026-10-23 brief45修正：任务级最终目标与局部waypoint分离

撤回“把调度器waypoint变成goal通道”的方向。ResetWaypointChannel保留可认证落盘，
主条件改为reset声明的最终pose + task_completion阶段；局部waypoint仅单列诊断。
未来真实pose负控制、训推online/collate等价、幂等防重复及写入拒绝不推进通过。
旧6,000窗口严格join仍0可认证/6,000缺证，不能把缺证算非法；Shadow30条仍只是候选。
主对照改为合法任务级目标vs unknown；局部诊断沿用旧反例，无新机制理由则预算0。
新语义/落盘工程通过不放行旧reader、来源缓存或训练；本轮无训练/物理/推理，成功差异null。
任务级目标接回后若无执行提升只能说“该通道未通过执行落点”，不能说“任务信息无用”。
代码23a11ca6；定向23通过，全套249通过/1个未改动fixture缺字段失败。
[实验卡与复现](experiments/probes/P-20261023-reset-bound-goal-channel.md)。

## 2026-10-23 τ event contract also ununified; cluster CI primary; no training

当前 TRAIN207,579 窗口中 event raw-valid207,561，CONTACT19,740/RELEASE185,439，
130,740 条有效事件在 K24 之外。真实同前缀反例证明 type/delta/validity 依赖未来
teacher；实际 Steps→DataLoader→collate35字段逐位一致，32条批次26valid/6dropout。
部署 TaskScheduler 使用事前 TaskSpec 与测量反馈生成事件/倒计时，事件合同亦未统一，
不能因 goal 修好就放行训练。config status不变、0训练、0checkpoint。

固定512锚点/124母序列的2000次聚类bootstrap，overall/first8/contact-window的
B1−B0a中位差CI分别[-50.817,-35.526]/[-21.198,-13.684]/[-36.580,-26.002]mm。
8/8子集仍不跨0，但只是旧条件下TRAIN面板证据；主scorer已切聚类CI并保留帧级对照。
采样索引设计可把每批母序列55.68→288、同批时间近重复75.96%→0，但母序列/物体池
仍1064/61，source比例会大改，故只留机主选择项，不接入或重排。
[完整结果解读表、字段证据与复现](audit/tau-event-contract-ci-sampling-20261023.md)；
代码9eeb3fac，产物`tau-contract-20261023-r1/`，合计163,972bytes。

## 2026-10-23 τ r9 causal scheduler replay remains BLOCKED

6,000 curated 窗口、1,067 sequence/slot 组完成实际输入字段盘点：缺部署前已知任务意图
或其因果生成输入，可重放 0/6,000。同观测前缀的两种合法独立指令给出相差 14mm 的
scheduler 条件，不能从录制状态唯一恢复意图；另有 84 个窗口状态 mask 前缀含缺口。
helper 已修复 live/reference pose 混用与整段 joints 前视读取；tail 截断/NaN/guard 与
边界回归合计 10 passed。真实 6,000 条等价与语义差异仍未验收；无训练、无 checkpoint、
config 不变，门槛(2) BLOCKED、(3) NOT_RUN。最小合同改动和结果解读表见
[因果重放报告](audit/scheduler-causal-replay-20261023.md)，代码 `5939b405`，
receipt v2 位于 `outputs/ptv3-shared-wam-v6/tau-curated-20261023-r9/`。

## 2026-10-23 v6 online goal contract repaired; cache identity gate remains HOLD

真实部署路径 `online_runtime.py` 已接入 `GOAL_SEMANTICS="spatial_path_waypoint"`。
`online()` 和 `OnlineV6Runtime.predict()` 对缺失/非有限 goal 均逐 flow query 设置
`object_goal_valid=False` 并零填充；`predict()` 在 `collate()` 后、真正送入
`SharedWAMV6` 前断言 bool mask、有限 `goal_object_delta` 和 invalid query 零填充。
显式传入 t+24 endpoint 等其他语义会抛出 ValueError。v6 包不跨 Task 导入 v4，
本地 helper 与 v4 `_prepare_goal_inputs` 由 targeted test 逐位等价验证，语义来源
记录在 [goal contract repair](audit/v6-goal-contract-20261023.md)。

验证命令与结果：

```text
PYTHONPATH=src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src:src/task/wam-rl-policy-v3/src /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ptv3-shared-wam-v6/tests/test_v6_goal_contract.py
6 passed in 2.18s
PYTHONPATH=src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src:src/task/wam-rl-policy-v3/src /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ptv3-shared-wam-v6/tests
154 passed, 1 skipped, 7 warnings in 17.51s
```

门槛逐项：goal contract tests **达成**；scheduler-conditioned cache identities
**未达成**。现有 `outputs/ptv3-shared-wam-v6/tau-curated-20261023-r1/` manifest
仍声明 training goal 是 realized t+24 endpoint，并列出必须在训练前重算的 scheduler
goal、sidecar/cache identities；r2/r3 仍有 `object_goal_valid_hard_code: not changed`
和 `training_allowed: false`。因此 `configs/tau_curated_data_only_v1.json` 保持
`status=HOLD_UNTIL_GOAL_CONTRACT_REPAIR`，本轮不放行 τ 训练；40 面板成功率/τ
预测没有在本轮重跑，结果解释表见上述合同报告开头。

## 2026-10-22 S2c：阶段化诊断与有界 dense reward 已接入

[交付/公式/复现/反例](audit/s2c-staged-criteria-dense-reward-20261022.md)。仅修改v6，
物理和v3 executor未改；无训练/新仿真。环境输出各阶段current/achieved/first_time_s、
reward五项与累计和；lift必须在有效contact之后新增5cm，hold连续elapsed1s，release
必须授权+有效零力+真实分离。进度上界增量限制重复刷分，新reward/task identity拒绝旧resume。
合成6例阶段数6/4/3/2/1；固定历史六初态600controls档案为2/2/1/1/0，完整任务0/6。
此严格新口径且首次terminal截断，不能与旧窄5/6直接比较，也不代表生产reset修后新执行。
44项合同回归通过；dense reward学习收益与完整RELEASE新物理资格均未测，P1仍HOLD。

## 2026-10-22 S2b：参考时域补齐，物理 RELEASE 资格仍未通过

[S2b diff/复现/期望与协议](audit/s2b-tau-reference-transfer-20261022.md)：同source-PD连续
重放205状态/204controls/194有效GTτ，前81状态及原80动作与旧校准精确相等；终点166后
38个实测状态覆盖1s settle及8步前视。后缀保持最后PD target是新增假设，完整RELEASE
scorer仍未成功，不能称为完整阳性见证。未重训、未用新reference再跑校准R。
CPU合同明确τ[0]为第一未来、插值原点独立及30/60Hz差异；旧64/66是最后正持握tick，
首负为65/67，GT仍有效。冻结校准迁移预检PASS，但不同初态/异轨迹性能NOT_EVALUATED；
最小9rollouts/5400controls协议已记录，尚未执行。P1/RL仍HOLD。

## 2026-10-22 13:0x v2审计收口；正式RL仍HOLD

[一页结算](audit/negative-result-audit-one-page-20261022.md)；
[共同复核diff/命令/期望与未决项](audit/fix-review-20261022.md)。
补修三臂复制的voxel终止，三信号与ever/final分开；实际A/B/C各20controls故意outage
无额外终止，C WAM13/20覆盖。原始TaskSpec保留，新episode是独立+8cm位置HOLD任务，
source_full_task_success=null，旧训练不续作新协议。raw-invalid/dropout分离，501完整
录制重建全一致、64旧UNKNOWN逐例恢复62R/2H；计划意图尚非独立真值。

采纳单轨迹同τ校准，已见cylindersmall80条state/action、同架构R2000updates/5.376s，
actionRMSE0.103540→0.001849。独立5cm/force耦合/离桌elapsed1s窄任务source-PD1/1；
GTτ原R/校准R均0/2，WAMτ11000原R/校准R均0/1。source阳性不能替代GTτ executor，
也不能放行精确8cm/完整RELEASE；两份真实receipt被硬门拒绝。正式RL入口绑定任务/
executor/执行策略/初态序列范围的资格证据；P1 HOLD，大训练待Hermes另行确认。

v6独立adapter和C路径已存在并真实验证，UNWIRED仅指官方V4 PPO。单轨迹物理1488+
三臂接口两轮120=1608controls，GPU4/5/6均已释放。微拟合预算用尽，审计按机主F收口；
后续需要的新控制器/完整RELEASE/泛化/样本效率均为future evidence，不继续扩大本轮。

## 2026-10-22 12:4x 完整声明/释放时序/voxel协议已修，teacher上限资格FAIL，P1 HOLD

共同复核入口：[fix-review-20261022.md](audit/fix-review-20261022.md)。未启动新RL或v6大训练，
训练重启需Hermes确认。旧437/64筛选方案由完整声明修复取代：501原cohort全部保留，
473终态翻转恢复成490R/11H/0U，56418安全prefix控制、1221事件、81794有序source_frames
不变；491删失、709超prefix事件保留，不把不可观察事件当NONE。真UNKNOWN仍显式拒绝，
旧坏sidecar必须先迁移，不能静默过滤。初态voxel501/501通过。

固定原501真实轨迹/完整raw task，TaskScheduler v2同指标复评分：full成功0→77/501，
prefix0→9/501；RELEASE授权1→205、pathcomplete1→204，剩415删失/9pathstall。
bowl171帧授权、222帧成功；hammer233帧授权、269帧成功。修复按有序release位姿授权，
继续检查释放后剩余路径与终态1秒，未降低门槛。不是新物理执行/独立策略评估成功率。
491 unsafe/censored来源不能凭新scorer成功自动认证成高质量完整正示教。

voxel保持初态准入与每tick输入保护，在线坏点不送PTv3，τ invalid并单独记数，不再当机械
工作空间立即终止。注入错误点额外done1→0、invalid计数1；初态错误仍拒绝。
新增admission v2/scheduler v2/domain v1身份绑定resume，不能与旧训练混用。

真实GPU4工程24/24 PD/运动通过。已修continuous腕角reset代表，生产r4同协议24/24、
PD误差0，2/4步indexed reset旧13+palm_pose共14字段untouched差0；真实刚体首tick旋转
airplane0.002434/cubemedium0.000435rad。此前teacher/sourcePD各6条完整任务都0/6，
teacher5删失；sourcePD3acquire timeout、1path stall、2未完成。审计nearest-angle单例0/1，
平移Y动作饱和未解决。不能把这些旧初始化采集0/6套作生产reset修后完整链结果；修后尚未
重跑完整正控，上限资格仍未通过。含全部尝试共9366envcontrols、所有自有进程已退出。
GPU5真flow11000/PTv3→21/11→BC→PD smoke PASS，单次WAM0.491s；官方PPO仍v4/UNWIRED。

证据：`outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1/`；
`outputs/ptv3-shared-wam-v6/prefix-task-rescore-20261022-r1/`；
`teacher-task-scoring-20261022-r1/`；`fix-review-20261022-r1/`。
88回归、19不变量/错误注入、8production源码mutants通过；物理上限未过不等于方法无收益。
本节覆盖下方同日旧准入/CPU-only/待测描述，历史记录与旧结果不改写。

## 2026-10-22 12:3x 机主停止FM长训；确认缺陷已修、P1继续HOLD

Hermes12:29 SIGINT由机主转达；独立确认目标11个PID已退出，GPU4/5/6各6MiB/0%，未重启。
机主step11020观察后因后期收益趋平决定停止；实际最后日志11380，主阶段权重10500/11000
真实存在，最新11000；11020有日志但无对应payload，至少380更新/109440次样本曝光未保存。
8.711h step9000总loss.689010→10.713h step11000 .661275；±15min窗口总loss下降3.96%，
contact仅0.73%、geometry_contact2.48%，固定DEV未持续改善。不是严格全程flat或FM架构否决。
原process FAILED/SIGINT、progress残留RUNNING不回写；独立stop receipt记录实际已停。

UNKNOWN采用显式拒绝：501 eligible→437 admitted/64 rejected，完整序列清单保存、不改标签。
策略与native使用同一准入函数，握手检查旧native采样集合，TaskScheduler前置拒绝UNKNOWN；
新resume身份绑定准入policy，不能用旧分布恢复伪装同实验。官方PPO仍v4，v6明确UNWIRED；
仅完成1个真实H4经小随机v6/CPU geometry seam→实际BC→PD边界smoke，不冒充GPU/物理迁移。
scale不存在请求已拒绝；新增reward/phase/controls/return统计和8MiB/rank episode日志，奖励不变。

危害：1570 voxel27792/53174=52.2661%，最终27869/53347=52.2410%；更新预算16515072，
native实际16527360，尾部12288未更新。UNKNOWN ended集合界6212–9314；同时排UNKNOWN/
voxel后只能描述已发生训练子集0/N，N16164–25478，不是修复后的结果。精确时长/重叠缺日志，
关联步数只能给宽界，不能拿回合比例当浪费率。UNKNOWN安全prefix候选2150/56418=3.81084%。
P0旧状态4768个回放，两首越界均right wrist且距目标约2.2m；admission-only/每tick完整
保留4800/4744 controls，主prefix均2316、完整任务均0/16；不证明去gate的动态效果。

报告：`src/task/ptv3-shared-wam-v6/docs/audit/negative-result-fixes-20261022.md`；
`negative-result-harm-20261022.md`、`v6-ppo-wiring-feasibility-20261022.md`、
`flowmatching-stop-checkpoint-audit-20261022.md`。
回归与smoke均为CPU，无新方法训练/物理；P1保持HOLD。下一步最便宜可证伪的是两环境
HOLD/RELEASE真实原动作正控制（≤600交互）+逐帧评分/动作q/dq/native reset合同，未过不进RL。
56项针对回归通过（6.39s），19项不变量/故意错误检测与CPU接口smoke通过；整库verify仍受6张旧卡metadata缺口阻断。
本轮完整权限已解除上一轮git只读限制；审计与修复将统一提交推送，SHA以实际Git记录为准。
以下所有同日RUNNING/未停止/只读未提交段落均为历史记录。

## 2026-10-22 机主优先级更新：负结果审计；P1 HOLD（覆盖下方旧RUNNING状态）

已受控停止本线程P1队列，保留A191139/B19172更新，实际物理交互286720/149504；
C及其余seed未启动、科学评估0/45，禁止据此宣称三臂无收益。顶层FAILED源于STOP后
索取缺失milestone checkpoint的编排问题，不是新的方法负结果。原flowmatching长训与GPU4–7未动。

独立git show核实18769096与实际冻结3a0779f6的历史官方PPO均是V4Runtime/SharedWAMV4，
24×2×11×3；53347训练结束回合0完整成功不是v6负证据，也不是独立评估成功率。
501准入任务HOLD416/RELEASE21/UNKNOWN64（12.774%）；实际初始化UNKNOWN9314/56449
（16.500%，不是已结束episode分母）。UNKNOWN没有成功分支，却未被准入排除。
iteration1570 workspace27792/53174=52.266%，是PTv3 voxel index gate；旧首点信息未保存。
奖励实际进入PPO storage/GAE；缺失的是runner return/phase日志，不能说PPO没reward。

501真实档案CPU回放：安全prefix真实elapsed≥1s抬升251、持握代理245；TaskController
完整成功0（499录制删失、2 path failure）；独立几何完整条件10，差异尚需定位。
原参考不是已合格完整任务正控制，UNKNOWN子集也不能解释全部零成功。
秒制复核另修订P0：30个held状态只跨度29/30s，FM旧1/16按elapsed≥1s为0/16；动作回放15/16、GT/healthy0/16不变，原数字保留legacy。
19项CPU合同/19个故意错误检测均通过，另voxel首点1项含mutant；26项相关测试通过。
12轴动作可到PD，16轨迹2316control PD重算误差0；受控扰动后实测q/dq与native隐藏reset
仍未认证，P1保持HOLD。load_geometry缺请求scale静默fallback已修，未证实影响历史r3。

19张历史卡已逐项审计并标注适用范围，保留真实局部负现象，排除模型错认/实现污染/
正控制不足/统计单位错误产生的方法否决。主报告：
`src/task/ptv3-shared-wam-v6/docs/audit/negative-result-audit-20261022.md`；
协议`negative-result-hygiene.md`；CPU产物
`outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1/`。
本轮没有新方法实验；修后首项应是明确HOLD/RELEASE真实成功参考、评分一致与动作/native
reset工程资格验收，通过前不恢复三臂。下面同日RUNNING/继续supervisor等文字为历史启动记录。
交付受当前sandbox只读.git阻断（index.lock创建失败），没有本轮commit/push；远端ls-remote网络失败，SHA读不到。

## 2026-10-22 历史启动记录：GT-τ 执行资格未通过；三臂现已HOLD

本线程独立推进机器人闭环，不重复另线程的可辨识性/goal契约诊断。
固定已适配τ的Scheduled686 step50000，同16个机器人DEV初态（相对其TRAIN名称/源路径
交集0），原动作回放抓取抬升持握15/16=93.75%，标准机器人GT-τ0/16，健康4371预测0/16，
固定FM8500预测1/16。标准GT与预测均前3控制zero/invalid；额外live-centered GT也0/16，
但其旧startup不同，单列oracle。五臂初态7数组逐项exact；每臂native300步，PD误差/episode错配0。
平均最长连续持握分别2.090/0.0542/0.0438/0.0625s（原动作/标准GT/健康/FM）。完整原路径任务
均0/14未删失，另2例截断；不要当16例全任务失败。自然失持恢复原动作0/2、FM0/1，其余无
机会记UNKNOWN；强制扰动与mesh穿透未测。首FM整批越界失败已保留，修复逐环境边界后完成；
末FM完整300步workspace2/16均在各自准入前缀后，主指标0/16。

结论：当前执行桥连可执行机器人GT-τ都不能稳定执行，资格门未过，不能断言τ解码质量是
主瓶颈。范围仅这个v4-τ训练的执行器/v6适配桥，未训练的新direct executor仍不能伪报完成；
WAM祖先终身未见也无法认证。数字/失败类型/配对CI/精确命令见
`src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261022-wam-v6-executor-qualification.md`，
产物`outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/summary.json`。

P1已先写设计，单Inspire、cubemedium/cylindermedium/mug，统一真实CAD2048、相同BC、奖励、
显式position-only抬升8cm/HOLD1s与共同H4；A无τ、B几何优化参考、C健康4371预测。
3独立seed191/192/193，每臂每seed目标1,048,576在线控制；共9,437,184，eval共同seed194，
14字段初态exact校验。B是受限几何优化候选，未达到完整碰撞约束IK强基线，不能据C>B宣称
打败所有几何重定向。256环境真实工程容量A/B/C均2更新40optimizer/4096交互，错配/PD/
logprob0；C batch64保留24帧、跳独立未用C头后实测26.921s/更新，改统一6h/run硬界。
科学首批已于11:35前后由supervisor2708769脱离会话启动，frozen fe81b6b5，GPU1/2，
run=`outputs/ptv3-shared-wam-v6/samples-efficiency-3arm-20261022-r1/`；receipt在同名tmp/launch.json。
A/B已实际optimizer更新，首轮logprob误差0；尚无正式eval点，成功率/阈值步数/AUC均待定。
所有9run和eval均从runtime-code快照读取实现、有限队列/共享6h每run/8GiB产物/20GiB磁盘底线。
按统一dense2048正式A/B吞吐更新：9训练约22GPUh、双卡11–14h，含评估暂估16–22h wall；
C科学run尚排队，ETA须随progress更新。
首轮实际A约8.3s/update、B16.5s/update，GPU利用率偏低，采集/几何CPU路径主导；已如实报告。
科学完成0/9、正式eval0/45；工程/在线随机episode计数不能冒充固定面板成功率。结束自动生成sample-efficiency-summary.json与
learning-curves.png/svg；缺seed/点记INCOMPLETE。中断多收集未更新rollout用native.steps成本账，
不冒充完整预算。冻结smoke缺配置r1保留，r2已真实A/C checkpoint0评估及14字段配对/图表通过。
工程smoke不是学习曲线或成功率证据。P2仅公平训练设计，0运行。

本线程GPU2/3完成P0；原诊断后来使用0/2继而0/3，未干预。P1已用启动时实测空闲1/2；
GPU4/5/6长训、7、凭据、GitHub push均未触碰。仅本地提交。整库verify受6张既有旧卡
schema缺字段阻断；本线程3卡schema/seed检查PASS。

## 2026-10-22 大规模 WAM goal / 可辨识性诊断完成（本轮无新训练）

固定健康4371与FM8500，严格FP32；DEV3356窗口/113缓存序列/112原生录制，TRAIN5241/
1064/1063，本地未覆盖前缀905。TRAIN和DEV原生录制交集0；完整28帧支持排重，DEV另报
847不重叠核心集；祖先训练窗口未知，本地未覆盖前缀不宣称终身未见。FM K8种子172、181–187，
候选选择181/182/183，capture/group bootstrap188。未启动训练、改loss、扩模型/数据或push。

独立核到goal语义错配：调度器spatial_path_waypoint、endpoint_time=None；训练实际t24终点。
DEV约69.85% Oracle终点超调度器50mm/.35rad/50mm单步预算。原部署路径子模块逐序列回放
（已知录制路径作oracle reference；完整TaskScheduler缺实测抓稳/释放/支撑反馈，未伪造）。
全DEV前8手/物EPE：健康Oracle15.80/14.72→Planner21.12/25.55mm；FM17.37/16.21→22.79/27.60。
Oracle C头F1健康.686→FM.709，几何GT物接触F1却.507→.484，GT τ参照.971。
FM DEV手/物增量95%CI[1.449,1.701]/[1.334,1.658]mm；TRAIN前8改善10.56/7.55→9.95/6.43。

可辨识性仍待定：两种子TRAIN候选31619/31576、均覆盖1064序列；事件内候选19672/19747。
50mm事件分组14/11组、69/57录制，手8原始→相位对齐方差1610→709mm²（种子182）。
控制相位与相同窗口后，加接触区域手8下降−4.56%/＋4.53%，95%CI均跨0；宽邻域表面收益
受样本剔除影响。不能据此断言τ不可辨识或网络架构差。

自一致：Oracle预测手接触距离GT/自己物体健康14.71/10.54、FM15.05/10.57mm；Planner
FM23.53/11.62，绝对误差更大而相对距离仍较小。t8共同平移诊断N3354：FM Planner手/物
49.09/58.60→减去事后GT拟合共同平移后41.02/25.18mm，共同项44.97mm；非部署校正。
CAD复核覆盖2944窗口/92录制，每draw128手皮肤采样点；FM Oracle对GT物体>1mm穿透3.115%
（93893/3014656 vertex-draws），Planner1.818%但最小signed clearance均值12.62→31.54mm，
穿透少伴随分离增大，不能当质量更好。其他412窗口CAD资产未映射；物体为预测32点的affine
CAD扩展，手为观测皮肤运输代理，均已标明局限。

GPU2后被占，只迁移本轮诊断到3；另一worker用0。4/5/6长训与7及其他作业未动；本轮GPU
作业全部结束并释放GPU；随后其他用户占用0/3约3.1GiB/卡，未干预。新增诊断产物约2.405GiB，文件系统剩余约146.03GiB，未清理原数据。
9项针对性测试通过；全仓verify仍有6张既有trajectory-policy实验卡schema错误。
Executor三条件比较只完成设计，尚未执行，待机主确认；不新增self-consistency loss。
卡片：`src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261022-wam-v6-goal-chain-audit.md`。
数值/命令/日志：`outputs/ptv3-shared-wam-v6/goal-chain-audit-20261022-r1/audit-report.md`、
`large-summary.json`、`decision-metrics.csv`、`launch-*.json`。

## 2026-10-22 第一阶段现有checkpoint诊断完成，无新训练/扩容/加数据

固定健康4371与FM8500；GRAB三面板各96窗口：TRAIN95录制、序列DEV66、本地未覆盖
前缀93。种子178，统计180，flow噪声172；H4+K24完整28帧支持排重，原生TRAIN1040/
DEV92录制交集0。祖先远端窗口清单缺失，本地未覆盖前缀不能宣称终身未见。

可辨识性待定：4096TRAIN库，50mm严格组仅5组/30录制，匹配事件时间后1组。
组内手8方差1609→接触区域1927→手指2190→交互类型2663mm²，无可靠收缩。
配对原始3031→2034的约33%下降，被未来是否接触控制（188对2029）基本解释；
同区域66对2034未再降。100mm宽组全24手方差接触区域降约42%，只能算粗条件信号。

前8手/物EPE：TRAIN10.42/4.55→9.90/4.97mm；序列DEV20.28/9.36→21.78/10.97mm
（配对增量95%CI手[0.89,2.07]、物[1.08,2.12]）；本地未覆盖40.64/2.80→42.34/5.55。
DEV C头MAE0.05546→0.04934却几何GT接触MAE0.06826→0.06942，C改善未保证τ改善。

DEV前8预测手接触距离GT物/预测物：健康15.025/10.477、FM15.534/10.630mm，
差值−4.548/−4.904，未见大的正分离缺口；不能凭相对距离更小认证绝对预测正确。
16窗口/15录制的t8/t24闭合物体mesh复核，每锚点/时刻1024个手皮肤点均未检出>1mm
穿透；手仍是观测皮肤运输代理，不能外推全表面。物体刚体拟合残差t8=4.60→7.83、
t24=5.71→15.35mm，非刚体性上升；GT刚体控制最大0.0000313mm。

GPU0等待他人临时占用后续算，GPU1仅几何smoke；未信号其他作业或GPU4/5/6长训，
未碰7/凭据、未push。原长训仍RUNNING9700/max43200不变；本轮输出1.4GiB，余149.46GiB。
4测试、实数据平移/归一化/立方体/CAD内部控制通过。数字与完整60分层/命令见
[实验卡](experiments/probes/P-20261022-wam-v6-identifiability-audit.md)，
数据根`outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1/`，入口`summary.md/summary.json`。
物理可控性未由离线预测误差验证，等待机主对数字的下一步决策。

# 2026-10-22 FM trial complete; sustained three-rank training RUNNING

Frozen0917f47e;supervisor1926227 PPID1/SID1926227,torchrun1947491 separateSID.
Only GPUs4/5/6,local96/global288,strictFP32,latestGT-object2048 contact0.05,
goal/events retained. No other jobs/cards/credentials touched.47 tests and real
preflight/capacity8/DDP8+strict2/trial128/ODE audit PASS:617 finite Adam states,
RNG3,exact scales,equal rank signatures. Fixed four-TRAIN teacher MSE ratio0.883
to current zero field; fixed GRAB DEV prefix-object17.800→18.699mm and hand
55.222→56.571mm,no held-out benefit yet. Owner-directed longer exploration.

At01:33 main160 loss0.8639/norm1.2235/shared-context RMS3.77/5.84 finite,
0.310updates/s,GPUs89/93/98%,~20GiB/card,free153.79GiB. Bound30k or cumulative
12h;estimated~12k–14k/13:20–13:30+08,08:00~7000–7500,provisional. DEV/save500.
Local207579 TRAIN/3356 DEV,~99.8%GRAB;no new data. Output
`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/`:training.log,
training/,process.json boundedETA,startup-health.json. Launcher receipt and
supervisor.log in matching tmp/run directory. Disk20GiB floor;owned rootSTOP
requests saving/stopping. No rollout/effectiveness claim.
[Full record and exact commands](experiments/probes/P-20261022-wam-v6-flowmatching.md).

# 2026-10-22 Joint conditional residual flow trial RUNNING

Owner requests runnable FM trial then longer three-card training onGPU4/5/6.
Latest GT-object2048 contact retained,FP32,goal/events,healthy4371 plus new flow.
Bound30000 updates or12h cumulative training,detached,no credentials. Frozen0917f47e/supervisor1926227 PPID1.47 tests,real-input
gradients,capacity96/DDP8+2 PASS:617 finite Adam states,RNG3,exact normalizers
and equal rank signatures. Trial60 loss0.8784/norm0.923,all GPUs100%;
continue through128/sampling audit before automatic long stage. Local207579 TRAIN/3356 DEV,~99.8%GRAB.
08:00 is interim review, no efficacy claim.
[Experiment](experiments/probes/P-20261022-wam-v6-flowmatching.md).

# 2026-10-22 GT2048 contact arm saved/stopped3238; no held-out benefit

Execution STOPPED normally at the original4h boundary, final complete
training/step003238.pt (1,186,351,949bytes). CPU audit PASS:568 finite optimizer
states all3238, model finite, RNG3/world3/global288, all4 normalizers exact.
Supervisor1572421 and torchrun1583025 exited; no extension/restart.
Checkpoint cumulative elapsed14402.59s; stage-only result14286.84s excludes
readiness/recovery time. final-stop-audit.json records the evidence.

Latest fixed DEV is3000, not3238: GRAB hand52.588/object-prefix8=18.231/
object-h24=14.660mm, previous matched FP32=52.229/18.131/14.150mm.
GT geometry contact MAE0.14122→0.14177, no gain; macro hand32.647→33.383mm.
Do not extend this package based on these results. No statistical universal
harm, sole-cause or rolling-success claim. Retain20/3000/3238 and separate
readiness/recovery artifacts for inspection; original FP32 source3337 preserved.
[Final experiment record](experiments/probes/P-20261021-wam-v6-gt-object-contact.md).

# 2026-10-21 GT2048 contact probability training RUNNING

User-chosen GT2048 object plus predicted-hand contact-probability supervision
actually running, frozen39d66051/supervisor1572421, GPUs4/5/6/local96/global288.
Same healthy4371 initializer; original tau/scene/C/goals/events/strictFP32 retained.
GT future labels are loss-only. Original FP32 saved/stopped3337 with full audit.
44 tests and real preflight/capacity96/readiness8/strict2 recovery all PASS;
568 states, RNG3, exact scales and3 equal rank signatures. Calibrated weight0.05,
ramp100. New initializer physical DEV metrics match the prior arm (<0.0001mm).
Sustained startup at60, loss0.7292/norm0.916, finite activations;
warmups not yet complete. Bound10000 updates or4h new-objective time; likely
binds time first at initial~0.32updates/s. Free178.56GiB,20GiB floor.
No efficacy or rollout claim; main comparison is prefix1–8 hand/object errors,
new GT geometry contact metrics and unchanged predicted-both geometry diagnostic.
[Experiment](experiments/probes/P-20261021-wam-v6-gt-object-contact.md).

# 2026-10-21 Fixed TRAIN probe supports a generalization gap

32 fixed GRAB TRAIN examples improve from4371 to FP32 new3000: object prefix8
6.509→5.784mm, hand23.809→21.120mm, normalized scene loss0.0634→0.0465.
Fixed DEV object prefix8 worsens17.018→18.131mm. Optimization still progresses;
capacity saturation and gradient conflict are not established. Offline object
t+24 endpoint is directly conditioned; relative geometry permits common absolute
drift. Neither endpoint nor contact improvement certifies useful prefix gains.
GPU7 read-only probe20.1s/<0.453GiB; active three-card training unchanged under
original4h bound. Next work should separate held-out data slices and prefix
selection before extending training or expanding the FFN.
[Experiment evidence](experiments/probes/P-20261021-wam-v6-strict-fp32.md).

# 2026-10-21 FP32 stable through3000; short-horizon gain unresolved

Actual strict-FP32/backend arm has passed the BF16 instability interval.
At3040 gradient0.761 versus BF16=83.95; shared/context RMS3.63/5.64.
GRAB fixed DEV3000 hand52.229mm, object-h24=14.150mm versus BF16=39.462mm
and own init15.829mm. Object-h08=33.398mm still worse than init30.934mm.
Available-source macro hand32.647→33.244mm and DEV objective1.1253→1.1743
worsen; no overall accuracy or rollout-drift claim. Continue unchanged4h bound,
about50min remaining; GPUs99–100%, free184GiB. No new rolling video evaluation.
[Precision experiment](experiments/probes/P-20261021-wam-v6-strict-fp32.md).

# 2026-10-21 Healthy4371 strict FP32 training RUNNING

User-requested healthy-start full FP32 arm actually launched, frozen1b57a487,
supervisor1234454, GPUs4/5/6, local96/global288. Capacity8, real three-rank8-save
and strict2-recovery PASS:568 finite states/correct counts,3 RNG, exact scales,
equal signatures. Actual PTv3×16 and dense attention×13 use FP32 efficient SDPA;
AMP and matmul/cuDNN/spconv TF32 off.38 tests pass. No vendor/architecture edits.
Same source4371/geometry weight/goals/events/local207579 TRAIN/3356 DEV/seeds/LR;
fresh optimizer only at initialization, then strict recovery/continuation.

Sustained startup accepted at100, loss0.7583, norm0.851,
shared/context RMS3.87/6.06; LR and geometry warmups both complete.
Recent0.303updates/s, GPU average~96–97%, physical20–22GiB/card,
free187.08GiB.10000 updates or4h cumulative; at FP32 speed time bound likely
binds first. Root STOP saves/stops. No later-stability/drift benefit established.
Initial GRAB hand52.6872/object-h24=15.8287mm; step20=52.5363/15.0312mm.
Short prefix only; compare500/1000+ and frozen rollouts before any efficacy claim.

Previous BF16 geometry run COMPLETED10000, final GRAB hand56.4268/object-h24
30.3106mm, final checkpoint audit PASS, GPUs released before this arm. The
queued no-geometry control is not launched during this user-prioritized FP32 run.
[Precision experiment](experiments/probes/P-20261021-wam-v6-strict-fp32.md).
Live: outputs/ptv3-shared-wam-v6/strict-fp32-pretrain-20261021-r1/process.json
and training/progress.json; startup-acceptance.json records checks and telemetry.

# WAM v6 state — 2026-10-21

## 2026-10-21 Shared FFN scale drift localized; partial natural recovery at9000

Read-only four-TRAIN-example audits reproduce degraded scene gradients without
geometric loss. PTv3 token RMS1.62→1.86, but shared Transformer last-layer RMS
3.58→423.15 and scene_context5.58→3408.50 (source4371 versus geometry4500).
FP32/FP64 reduce numerical amplification but retain large scene-head gradients.
LayerNorm counterfactual lowers gradients yet worsens existing checkpoint loss,
so it is not a validated hot repair. Six disposable optimizer steps do not show
that resetting Adam resolves the issue. No source checkpoint or active run
changed; original trigger and matched-control causality remain open.
FFN-only4371 rollback in4500 copies lowers shared RMS423→14.45, while attention-
only rollback leaves355.62; partial hybrids worsen accuracy, so no hot repair.
All4 normalizer tensors unchanged;568 finite optimizer states all4500, no
negative second moments. Active run naturally recovers at9000: shared RMS5.12,
scene-head gradient6.40; fixed GRAB DEV hand55.45mm/object-h24=37.26mm versus
initial52.68/15.82mm. Not permanent monotone collapse, still no net object gain.
PointWorld source comparison: GELU/4x/pre-norm FFN core matches exactly in a
same-weight forward/backward smoke; release LayerScale disabled. Shared-stack
DropPath and layer initialization differ from PointWorld. Missing final norm
alone is not a demonstrated explanation. See paper/pointworld/ffn-comparison.md.
[Detailed diagnostic evidence](experiments/probes/P-20261021-wam-v6-geometry-consistency.md).

## 2026-10-21 Geometry rollout measured; matched control awaits resource plan

Frozen parent4371 versus geometry+4500 evaluated on GPU7 (436.53MiB allocated)
without touching active4/5/6 or unrelated users. Same four old GRAB TRAIN
anchors/H4/true t+24 goals, forecast24/consume8/predicted-state feedback, common11
scoring. Old v4/parent4371/geometry4500 means in mm: eyeglasses67.07/90.36/88.40,
bowl220.00/292.50/302.08, stamp52.19/179.84/195.66, bunny43.79/178.22/228.01.
Thus no consistent GRAB drift improvement; three worsen relative to parent.
All eight generated-history/raw-coordinate audits pass, no stop/truncated tail.
V6 native21/observed skin differs from v4 inputs; not a matched-loss causal test.

Apple same sparse11 initial observations (missing geometry explicitly masked):
geometry fixed replay improves old-v4 mean674.87→228.32mm, but endpoint1272.47mm.
TaskController unknown-event mean664.59→330.26mm, worse than parent260.58mm;
planned-event variant318.39mm, parent179.88mm. Both stop acquire_timeout at15s,
447 common frames, zero established grasp ticks. No physics/generalization claim.
Seven four-panel videos are complete under the evaluation output.
Additional recorded-plan GRAB arm: geometry181.38/286.39/188.94/259.43mm,
three cases worse than parent; planned instructions do not eliminate drift.
Four separately labelled planned videos are complete;13 MP4s including two
combined panels pass ffprobe and selected-frame inspection.16 raw audits pass.

Control entrypoint/test/queue prepared,31 v6 tests pass. Same source4371 and
fresh optimizer/draw streams/world3/local96/global288; only geometry training
graph disabled, DEV diagnostics retained. All new control checkpoints preserved.
Existing geometry500/4500 pinned independently before source rotation. Proposed
automatic three-card launch after current run naturally releases4/5/6 requires
the pending user resource-plan response; no watcher/control training started.
Current geometry training continued untouched to5620 at14:14 (loss9.44425).
[AB experiment](experiments/probes/P-20261021-wam-v6-geometry-ab.md).
Artifacts: outputs/ptv3-shared-wam-v6/geometry-ab-rollout-20261021-r1/evaluation/
(result.json/comparison.json/data-audit.json) and control-queue-plan.json at root.

## 2026-10-21 Geometry consistency training RUNNING from latest local4371

User-requested predicted hand/object geometry objective implemented and launched
from the baseline's normally saved final4371, frozen b852f39c. SharedWAMV6 goal
and event conditioning remain active. Observed skin follows predicted joint21;
predicted object flow moves surface queries. Positive contacts supervise the
predicted relative vector against its GT-proxy counterpart; local noncontact
clearance adds separation gradients. Approximate geometry, not exact mesh/SDF.

Old C-only loss has no direct coordinate-readout gradients; actual source4371
GPU preflight confirms new gradients to both readouts/decoded outputs and100%
eligible coverage on four TRAIN examples. TRAIN-only gradient calibration gives
weight0.02292055275, ramp100. All model weights/normalizers copied exactly;
fresh optimizer/RNG/count/selection for the changed objective.30 v6 tests pass.
Three-card8-update save→2 strict recovery passes full568-state/RNG/normalizer
checks and identical rank signatures. Sustained stage has reached100 with finite
loss and nonzero geometry terms. Capacity96 passes13.59GiB allocated/15.34GiB
reserved; training allocated peak14.12GiB, sampled utilization100/95/99%,
physical18–21GiB/card. Free197.71GiB. Initial compute~240samples/s; periodic
DEV/checkpoint overhead affects sustained ETA.10000 updates or new-objective
cumulative4h boundary,20GiB disk floor. No accuracy or drift benefit established.

GPU4/5/6, global288, existing TRAIN207579/DEV3356 recipe and normalizers retained.
Exact4371 baseline GRAB handEPE52.6763mm/geometry relative22.5379mm on fixed32
DEV examples; compare future metrics with this before point. Old expanded run
is STOPPED4371; older RUNNING sections below are historical. Live supervisor
943417, outputs/ptv3-shared-wam-v6/geometry-consistency-20261021-r1/process.json
and training/progress.json; root STOP saves/stops.
[Geometry experiment](experiments/probes/P-20261021-wam-v6-geometry-consistency.md).

## 2026-10-21 Historical three-GPU expanded training (now STOPPED4371)

Latest user requests multi-GPU, utilization and scale. Original single-GPU r2
normal STOP saved1269. Explicit local data/topology transition retains model,
568 optimizer states, normalizers, elapsed, selection and rank0 RNG;3-rank
readiness1269→1277 and strict resume→1279 passed finite checkpoint audits and
identical rank signatures. Full v6 suite23 passed.

Current local-expanded-ddp-20261021-r1, frozen b1821d9c, GPUs4/5/6, local96/
global288,8 workers/rank. This supersedes the previous two-idle-card arrangement
for this user-requested run; no unrelated GPU0–3/7 processes affected. Capacity128
adds only0.8% throughput versus96 but reserved21.90GiB versus17.02GiB, so96
selected. Readiness compute248samples/s; sustained1300→1360 interval242.37samples/s
versus the old single-GPU1020→1180 interval85.59samples/s, a measured2.83x
increase. Different window pool/batch: practical throughput, not identical-load
DDP scaling efficiency. Active at1360; peak allocated14.05GiB, physical~18GiB
per card, free206.70GiB. Recent four10s GPU samples average76/98/96%, with
startup/loading/synchronization fluctuations; do not equate NCCL spin with
productive throughput. Initial ETA~3h to10000, within remaining4h cumulative
budget. Acceptance snapshot contains the exact intervals and telemetry.

TRAIN33672→207579 (all inherited GRAB TRAIN windows, sixfold), DEV3356 unchanged,
1177 existing sequence caches reused, no new raw data copies. Still a GRAB-
dominant subset, not the remote nine-source frozen recipe. Target cumulative
10000 updates or14400s including earlier single-GPU training, LR1e-5/warmup50
unchanged, DEV/checkpoint every500. Source checkpoint/readiness retained.
Live outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/process.json and
training/progress.json. Root STOP saves/stops. [Experiment card](experiments/probes/P-20261021-wam-v6-local-subset.md).


## 2026-10-21 Independent local subset training RUNNING

User authorized completing preparation, saving/stopping PPO and starting WAM.
1177 local sequences/33672 TRAIN/3356 DEV windows admitted under an explicit
local-only subset scope, disjoint sequences; original engineering data unchanged.
Initialize model-only from local step23758, exact inherited normalizer buffers,
new optimizer/RNG/count. GRAB~98.81% TRAIN sampling; no new coupling loss or
claim of strict full-nine-source continuation. All original full-recipe gates
remain strict.21 v6 tests pass; real2351 reader endpoints pass.

PPO saved/stopped normally at1575, complete61,158,850byte checkpoint verified;
all3 ranks/six workers/supervisor exited. WAM8-step save→2-step strict resume
passed:568 optimizer states all at8 then10, finite and scales unchanged. Frozen
d20b565c; current local-subset-pretrain-20261021-r2 training step80 at startup,
GPU4 single/local64/global64, LR1e-5/warmup50, target2000 or cumulative4h.
Measured1.36updates/s, allocated peak9.54GiB/physical12–13GiB, GPU76–100%;
GPU5/6 idle, free216.92GiB. Initial ETA~30min including approximate overhead,
not a promised completion time. First r1 attempt had no update, failed on long
IPC socket path; fixed with project-local short tmp path, preserved evidence.

Latest sustained checkpoint/DEV200 published. GRAB hand EPE49.9303→51.5141mm,
macro loss1.077912→1.065935; no accuracy benefit established. Continue the bounded
pilot, preserving initial/readiness/best checkpoints and reporting metrics apart.
[Local training card](experiments/probes/P-20261021-wam-v6-local-subset.md).
Live: outputs/ptv3-shared-wam-v6/local-subset-pretrain-20261021-r2/process.json;
startup-acceptance.json and training/progress.json; supervisor806802/worker810310.
Root STOP requests saved termination. Existing remote strict resume remains
blocked on its recipe/stats/index inputs; local adaptation does not replace it.


## 2026-10-21 Local data subset prepared; no training

Fresh source-header inventory covers39526 sequences. Target /home2 (/dev/md0)
started228.10GiB free; current about222.35GiB. Prepared and independently verified
1177 complete sequences (GRAB1132 plus45 across the other8 sources),5.69GiB,
33672 TRAIN/3356 DEV bounded engineering windows. No deletion or training;
training_allowed=false. Original frozen recipe/window indices/strata/stats remain
unavailable, so strict step23758 resume is not ready. Details and artifacts:
[local preparation record](research/2026-10-21-local-data-preparation.md).


Post-reboot resume requested, currently blocked on SSH connectivity. Three
attempts to127.0.0.1:22023 timed out during SSH banner exchange; the local port
is listening but remote boot/NAS/GPU status cannot yet be verified. No training
has been launched. Explicit world3→4/local128/global512 restoration is prepared
in13612688 and8 topology/actual-entrypoint tests pass. Reconnect and inspect
four-GPU health before choosing it; otherwise retain strict world3 recovery.
The local full checkpoint backup is now VERIFIED (1186363152bytes,23758,568
optimizer states,3 RNG payloads). User was asked whether the rebooted node's
SSH tunnel/address has been restored; code snapshot13612688.tar is ready locally.

Current status: STOPPED normally at23758 for the user's planned remote-server
reboot. Complete step023758.pt (1186363152bytes) was CPU-loaded and checked:
all model tensors finite,568 optimizer states (564 at23758,4 event states at8638),
3 RNG payloads, world3/local64/global192, elapsed20075.13s. File fsync completed.
Supervisor/torchrun/new workers3114372/3114380/3114387–9 all exited. Old faulted
kernel-exit contexts still remain for the reboot to clear; no system reboot
was executed by the agent. NAS is an NFS mount on /mnt/ugreen_nas.

Original checkpoint is under the threegpu-r2 run, with an additional hard link
pinned at outputs/reboot-preparation-20261020-r1/step023758.pt (same NAS file,
not an independent storage backup). Checkpoint receipt and prepared strict
three-GPU post-reboot launch script are in that directory. The optional local
1.19GB download is slow and was still IN_PROGRESS at handoff; it must not be
claimed verified. Local backup-status.json records completion/interruption.
The verified/pinned NAS checkpoint is sufficient for this compute-node reboot.

After reboot: reconnect SSH, confirm NAS mounted and GPU health/free memory,
then resume into a fresh output. Prepared resume_three_gpu_after_reboot.py
keeps world3/global192 and ordinary free-GPU checks; no old retained-PID
exception is reused. Returning to four GPUs/bigger batch requires an explicit
new topology transition and capacity check, not ordinary resume. Resume has
not been launched before the reboot.

Historical three-GPU operation follows:

Historical status: RUNNING on three healthy remote GPUs0/1/2, as explicitly
requested by the user who cannot resetGPU3. New run:
pretrain-nine-v6-events-threegpu-20261020-r2, frozen d43b662d, supervisor3114372.
Explicit world4→3 transition from original complete17000 preserved model,
568 optimizer states, elapsed/warmup/selection, RNG ranks0–2 and unchanged
24-step normalizers. Batch changed local128→64/global512→192; LR remains1e-4.
Original80k/cumulative604800s ceiling remains. Lower per-update exposure means
this is not equivalent to the old global512 training schedule.

Single-GPU CUDA checks and3-rank NCCL all-reduce passed.8-update recovery
17001→17008 passed DEV, full saves and3 equal parameter signatures. CPU checkpoint
audit:564 old states at17008,4 event states at1888,3 RNG payloads, strict
world3/global192 binding and exact horizon-normalizer buffers. Long run resumed
that actual checkpoint; verified17120 at2.034updates/s (about0.49s/update),
nine sources seen, no new failure. Recovery allocated peak9.95GiB; long first
window peak10.14GiB. Physical memory about38–40GiB includes old25–27GiB retained
contexts; it is not all new-process memory. Selected GPUs83–100% utilization.
Remaining80k initially estimated10–12h including DEV/saves; step-only ETA8.6h
omits periodic overhead. No event-efficacy/80k reliability claim.

Original GPU3 still reports requires reset and is excluded from CUDA/NCCL and
supervisor health decisions. Guarded sharing verifies only the failed run's
recorded owned worker PIDs, UID/start time, orphan/empty exiting command lines,
pending termination, zero utilization and>=20GiB free capacity before launch;
unknown/live work is rejected. No reset/system operation performed.56 v5/v6
tests passed in17.96s. Frozen original checkpoints and data remain untouched.

Historical four-GPU failure: around00:35 Beijing after last logged17300,
supervisor failed parsing[N/A] and model reported nonfinite valid observation.
The stale old progress.json is not active training. Original17000 retained.

At17000 DEV objective1.281051, handEPE32.784425mm, trajectory score16.892473mm;
prior16129 values1.286467/33.167013/17.007955. Event-on minus unknown handEPE
is−0.051441mm; same-model diagnostic only. Replayed next20 draws17301–17320 on
all4 ranks:10240 CPU samples, checked observed scene/hand/history/dense/features
and goal inputs all finite. Actual failed rank/step is unlogged, so this does
not prove device-memory or every intermediate valid. Hardware fault is a leading
explanation, not an established sole cause of the nonfinite observation error.

Local supervisor now handles missing telemetry as None, records reset faults,
refuses unhealthy/unconfirmed-free GPUs, and requests controlled STOP on reset.
3 targeted regression tests pass; this patch cannot repair a failed GPU and
has not hot-edited the frozen failed run.

The following startup observations are historical:
User requested continuing v6. v5 after-v6-probe-r3 saved/stopped normally at
15723; its complete checkpoint remains available. Strict v6 continuation uses
the learned event model's own readiness checkpoint16136, frozen d37e2320,
not a new v5 migration. Current four-GPU run:
pretrain-nine-v6-events-20261020-r1, supervisor3083547/torchrun3083564.
Verified RUNNING at16520, global512, loss0.77588,1.114 updates/s, peak allocated
18.57GiB; GPUs99–100% and about23–27GiB physical memory. Remaining80k updates
estimated17–19h including scheduled DEV/saves; progress ETA itself omits some
periodic overhead. Neither full completion nor event efficacy is established.

Model architecture and data recipe remain unchanged:768/12heads/12layers,joint21/H4/K24/30Hz,
surface778/object2048,voxel5mm, original TRAIN-only horizon normalization,
AdamW LR1e-4, existing warmup history and optimizer/RNG/elapsed. Restore moved
568 fused AdamW step tensors to parameter devices; strict local128/global512
and world4 binding passed. Original80000/cumulative604800s bounds retained.
Event encoder50304 parameters/dropout15%, no KNN or new tokens.

## Target outlier diagnosis

Actual reader index3150491/sample[29384,0,35], Shadow s4-34534, produces8.92525m
maximum scene-point displacement. Original local rollout and remote compact
payload object/time/hand-root/reset/success arrays match exactly; compact object
translations also match the frozen cache exactly. At frame36 the object starts flying away; maximum speed14.087m/s,
h24 center displacement8.85587m, eventual y17.023m. Clock is regular30Hz, no
reset, rotation orthogonality error1.19e-7, query radius43.65mm. This is an
inherited simulator-runaway segment, not a v6 event/scale/frame-transform bug.
Actual current parent recipe lacks strict motion-admission metadata; historical
cleaning validates finite poses/clocks and observed workspace, not future
runaway motion. It admits ticks3,11,19,27,35. The last3 windows cross onset36.

Read-only audit7a817817 scanned all39526 sequences and actually admitted
TRAIN3218740/DEV545459 windows in67.28s. Object-center displacement>2m occurs
in29 TRAIN/5 DEV windows, all Shadow. A Shadow-only candidate prefix cutoff at
first known observed center speed>5m/s identifies60 TRAIN/10 DEV windows in25
sequences. Weighted TRAIN exposure is0.00275% of draws (about1.4% of global512
batches under independent draws), so rare samples can still cause visible spikes.
Candidate exclusions are prepared but NOT applied; no running recipe/cache was
modified. A clean data continuation needs a separately admitted recipe and
explicit checkpoint data-contract transition. No scratch retrain is implied by
these sparse findings. Large net displacement alone is not a bad human label.
HOT3D s6-00372 has2 additional TRAIN windows crossing known observed center
transitions12.10/12.54m/s; retained as tracking candidates, not classified as a
confirmed simulator defect. The scan covers center translation, not full hand,
angular or surface-point motion quality.

## Prior readiness evidence

The8+1000+8 probe completed15120→16136, checkpoints/recovery/signatures passed.
CPU audit verified564 original optimizer states at16136,4 event states at1016,
4 RNG payloads and exactly unchanged horizon-normalizer buffers. Latest probe
DEV handEPE33.167mm versus unknown-event33.285mm: a small same-model diagnostic,
not a matched-training causal gain. Combined v5/v6 suite48 tests passed;
2 new motion-audit tests cover future-only runaway, rigid frame invariance and
observed-frame clock/masks. No model/training code changed for this switch.

Remote root:/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v6-20261019-r1/.
Historical four-GPU run:outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-20261020-r1/.
Switch:outputs/switch-to-v6-20261020-r1/report.json.
Motion scan:outputs/target-motion-audit-20261020-r1/summary.json.
Forensics and candidate exclusions:outputs/target-motion-forensics-20261020-r1/.
Small report/plot copies:outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-20261020-r1/ in local repository.

[Current training and diagnosis](experiments/probes/P-20261020-wam-v6-mixed-pretrain.md)
· [completed readiness check](experiments/probes/P-20261019-wam-v6-event-continuation.md).


Current remote three-GPU artifacts:
outputs/three-gpu-continuation-controller-20261020-r1/report.json (transition and checkpoint audit),
outputs/ptv3-shared-wam-v6/three-gpu-recovery-20261020-r1/ (completed8updates),
outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-threegpu-20261020-r2/ (active).
Local small receipts:outputs/ptv3-shared-wam-v6/three-gpu-recovery-20261020-r1/.
## 2026-10-23 Local-data scale correction; training HOLD pending goal contract

The local WAM-v6 data is a fixed **1,177-sequence** cache (GRAB 1,132 plus 45
from the other eight sources), with **33,672 TRAIN / 3,356 DEV** windows. The
local-expanded recipe only reuses those caches and expands GRAB TRAIN windows
to **207,579**; it does not prepare the registered **38,349**-sequence full
recipe. The prepared cache is 6,031,884,656 bytes (5.62 GiB), about 4.89 MiB
per sequence. With 141 GiB free on `/home2`, a linear full-recipe estimate is
183.0 GiB total / 177.4 GiB additional, before temporary peak and the 20 GiB
free-space floor. The detailed table and assumptions are in
[the scale-correction note](research/2026-10-23-local-scale-correction.md).

The flow-matching receipt was bounded at 30,000 updates or 43,200 cumulative
seconds. Its log reached step 11,380 and its latest durable checkpoint is
step 11,000; `expected_final_step=12,205` is only the receipt forecast. The run
is stopped. No full-data retraining is promised. After the `object_goal_valid`
and train/inference goal-semantic fixes are verified, the only scale choices
are longer training on the same subset (24–48 h) or owner-approved expansion
after reclaiming disk space. Until then, training and new cache preparation are
HOLD.
## 2026-10-23 固定 TRAIN τ 面板与 step011000 基线已建立

完成 Decision Probe [P-20261023-tau-fixed-panel-metric](experiments/probes/P-20261023-tau-fixed-panel-metric.md)。查重确认阶段进度、执行器面板、单轨迹校准和六回合 staged panel 都不等价；唯一新增打分器是 `tools/run/evaluate_tau_fixed_panel.py`，复用冻结 `outputs/ptv3-shared-wam-v6/tau-curated-20261023-r2/train-anchor.json` 的512行、seed170、v6 H4/K24 reader/collate、事件dropout关闭和flow seed172。GPU7确定性模式下对 `flowmatching-overnight-20261022-r2/training/step011000.pt` 完成64批/17.5s，无训练/物理回放/优化器更新，确定性重复检查PASS，新增输出约12KiB。

基线 hand endpoint EPE：全板 **23.409973mm**（467208 valid values），前8帧 **9.357826mm**（156135），接触窗口 **19.834665mm**（288960；7416/12288 sample-frames）；phase approach/close/contact/hold/unknown 分别 **29.897741/21.873276/22.194105/19.982422/22.469606mm**。JSON receipt 在 `outputs/ptv3-shared-wam-v6/tau-fixed-panel-baseline-20261023-r1/baseline.json`，一行摘要在同目录 `summary.txt`。后续 checkpoint 只有全面板、前8帧、接触窗口同向改善才可判为 τ 整体改进；单 phase 变化按局部结果报告。训练门槛与 `tau_curated_data_only_v1.json` 的 HOLD 状态不变。
## 2026-10-23 Scheduler-conditioned cache identity gate remains blocked

The current r7 curated TRAIN manifest has 6,000 admitted rows but no planned
scheduler reference path or waypoint payload. The v5 reader still constructs
`goal_object_delta` from the observed `rel[4:]` trajectory endpoint
(`src/task/ptv3-shared-wam-v5/src/ptv3_shared_wam_v5/data.py:44-61`), so these
rows are realized t+24 endpoint conditions. A bounded CPU/IO receipt explicitly
excluded all 6,000 rows rather than treating observed poses as scheduler input:
`outputs/ptv3-shared-wam-v6/tau-curated-20261023-r8/scheduler-cache-identity-receipt.json`.

New identity/replay code is in `src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/scheduler_cache_identity.py`
and `tools/run/verify_scheduler_cache_identities.py`. It uses the inference
`OrderedGoalScheduler` + `update_schedulers` path, hashes exact float32 32x3
conditions, and keys identities by `(source_id, sample_id, window_start_frame,
goal_semantics, condition_sha256)`. The equivalence tests pass, but scheduler
coverage is 0% until a separate planned reference path is supplied. No training
or checkpoint write was started.

## 2026-10-23 τ 四格条件归因：旧B有益、新B有害

同一512锚点/124母序列复用A、仅新增B，init A/B的三项EPE分别为
16.169/59.222/11.252与9.358/19.835/4.816mm；trained A/B为
10.288/31.760/6.504与12.131/45.561/7.861mm。旧模型恢复B降误差
42.13/66.51/57.20%，新模型恢复B增误差17.91/43.46/20.86%，
配对cluster CI均排除0。条件依赖是关键解释，不能说去掉意图天然更强；
数据/训练与条件适应仍混杂，无法单独分摊端点或event贡献。提高合法且训推一致
goal通道优先级，不直接给新权重接回旧B。B来自未来观测，仅诊断，禁止部署。
本轮GPU7两格各约29s，无训练/新采样/物理，已释放，新增约21MiB。

详见[四格归因报告](audit/tau-condition-attribution-20261023.md)。
