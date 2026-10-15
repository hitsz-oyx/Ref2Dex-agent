## 2026-10-15 shared actor三臂中期仍在更新

已实时检查三臂supervisor与GPU worker存活，最新约3360–3540步。复核matched合同后，
同一步3000/new DEV256的pred_tau、shared、shared_tau XYZ mean依次65.64/68.69/69.59mm，
p95依次216.16/208.19/197.35mm；动作L1为0.42036/0.43999/0.44800，近500步均下降。
shared尚无mean和p95共同收益，不据此提前结束12k协议；forecast DEV也未同步下降。
中期快照已保留在shared-actor-interface-20261015-r1/interim-step3000.json，实验卡同步。
GPU0–5继续、6/7空闲；Shadow单卡小测试已经完成并释放GPU6。

## 2026-10-15 shared actor三臂已并行启动

先提交补采数据/严筛27bd9913，再实现参考ref1的state+预测τ、state+shared、
state+shared+预测τ三臂。固定代码51307935，三臂同80k父权重/严格补采TRAIN93680、
dev10376/新固定dev256/q_PD 8×15；WAM forecast继续，统一关闭GTτ anchor，无warmup。
新CAMPAIGN允许本地用6卡且留2卡空闲：三臂GPU0/1、2/3、4/5，同时各12k更新，
每卡256/global512；远端原四卡不受影响。supervisor1922935/1922938/1922941。
两档batch双卡smoke全部通过，29项相关单测通过；一次encode、无GTτ泄漏/动作梯度
/mask/物理指标/独立checkpoint选择已核对。shared-only DDP unused时间参数已修复。
正式首步loss完全相同，参数99269429，新panel/admission逐项一致。三臂已更新40步，
约1.8–1.9s/步、暂估6–7h完成；GPU均值94–98%，显存约11GiB，6/7空闲。
CPU collector1922942完成后自动汇总同一步12k和selected物理指标。当前RUNNING，
无shared收益/抓取结论；legacy物理执行入口明确拒绝新头，正信号后再加native桥接。
详情P-20261015-shared-actor-interface；共同输出shared-actor-interface-20261015-r1。

## 2026-10-15 新训练切换补采660与失败前50帧严筛

用户要求物体位置偏离当前参考>10cm时连同前50帧不训练。新robot入口统一默认
补采dexplore-current-rollout-20261015-r1，旧数据/manifest/checkpoint保留。
按H4历史至future24全部support排除交叠窗口；超过阈值与非有限均视为坏帧，等于阈值保留。
独立逐窗interval审计PASS：TRAIN93680（546/589条有窗），dev10376（68/71条有窗），
合计104056；原workspace后122299减少18243窗，约14.92%。startup TRAIN7205/dev971。
这是位置准入而非成功抓取标签；人工备注未自动应用。新旧dev不同，后续三臂共同使用
新dev，不能与旧实验绝对指标作单因素归因。报告rollout-admission-current-20261015-r1。

# PTv3-Shared-WAM v2 当前状态

## 2026-10-15 s1人工备注已与审计核对

读取8条已保存s1标注，备注异常均有数值支持；原字段均uncertain/other，不自动重标。
8条工程audit均PASS，说明工程一致性不等于抓取质量。位置筛选保留1282/3507窗口，
大多是失败前缀或回入片段，不能全当误放行。watch晚期掉落段已被24步位置门排除；
phone参考接触424/424帧无实测接触、torussmall_pass缺接触100/120帧。自动准入仍
遗漏手姿/旋转/持续保持语义，需要人审校准；不将s1其余82条未标注序列视为已通过。
固定annotation snapshot与逐条比较在s1-human-audit-20261015-r1，源与训练准入不变。


## 2026-10-15 teacher输入的优化对照完成：起步训练拟合精确，dev差距未解

用户人工审阅660并授权继续对照；固定缓存/面板，未动态使用标注。full/current输入×
固定LR/12k后退火4臂各24k，本地仅0/1分批，远端四卡仍保留，合计6卡。
完整12k前缀权重相同；退火使full TRAIN面板10.14→6.28mm，但dev43.80→43.22mm。
全量窗口重推：full TRAIN/dev11.33/34.90mm（窗口均匀权重，不能替换原面板指标）。
关键定位：full TRAIN起步9402帧mean0.466mm，dev起步1134帧47.213mm；current对应
0.832/53.193mm。可精确拟合见过起步，但未跨序列推广；两split都含相同50物体/10subjects。
优化/容量尚未排除，结论UNCLEAR；停止此固定预算，不继续同类长训。所有本地GPU
任务已结束；Viser8768与人工标注保持独立。详见P-20261015-teacher-fit-generalization。


## 2026-10-15 Full-teacher-input upper bound完成：尚未跨过20mm

无tau、fromscratch同MLP，实际1442(delta1+16)对照721(delta0)，同native660准入窗和
旧dev512面板。全660/122299准入tick的冻结teacher重放最大PDXYZ0.000795mm。
完整1442 relativePD selected mean/p95为45.70/134.35mm，对照49.77/149.58；
teacher RMS补充为42.80/135.03，对照48.97/144.02，仅约8–13%mean改善，未达到20mm。
509样本memory训练3.16/7.70mm；全量最终TRAIN14.11、dev44.01mm，拟合/跨序列
泛化需先定位，不能只归因WAM或坐实主要information mismatch。literal absolutePD
126.33mm涉及world坐标合同，不与relative混因果。6臂完成、独立指标/代码审查通过、
GPU已释放。详见P-20261015-full-teacher-input和full-teacher-input-20261015-r1/summary。


## 2026-10-15 九阶段实际抓取面板完成：K4下均未达held45

9个完整微调checkpoint × 飞机/杯子/小球3条dev序列，seed148，GPU0/1/2并行完成。
27组teacher均held45，预测τ→R全部held0；GTτ→R最好8k飞机27/杯子19帧，其他
仅9k飞机8/杯子14帧，其余0。局部UNPROMISING，不能用离线动作loss下降替代抓取。
全部初始状态角色间/阶段间相同；27组最终独立控制审计通过，实际命令重建max0。
旧资产目录与审计float32量化问题已修复，失败/诊断记录保留，物理不重复覆盖。
三卡已释放，完整81行指标/阶段表/曲线在v2-stage-grasp-20261015-r2/summary。
本轮未继续训练，不能把GTτ→R失败等同GT q_PD replay不可执行；下步先定位R闭环
偏离或K4消耗，而非仅选择loss最优checkpoint。详见实验卡P-20261015-v2-stage-grasp。


## 2026-10-15 四卡微调已完成，最终24次dev与九个完整阶段已归档

r5已完成12000/12000，exit0，四卡训练和阶段归档watcher均已退出。最终固定dev256
action0.367689、anchor0.338220、forecast合计4.900133、total5.606042。动作继续慢降，
总dev未改善，保持离线Probe级UNCLEAR；未做物理抓取验证，不自动延长。
完整恢复轨迹24次dev已补到实验卡，最终CSV/JSON位于wam-v2-finetune-final-metrics-20261015-r1。
完整阶段1000/3000/3500/4000/8000/9000/10000/11000/12000均已保护；11500仅有评估指标，
没有该步完整checkpoint。全轨迹best total为1000，现存最低action checkpoint为12000。


## 2026-10-15 完整预训练指标已补录，80k远端阶段已完成

用户指出实验卡只有本地1k结果。已只读重取远端原日志，与留存完成快照的81个dev点
逐字段一致：0…80000每1k、固定384窗/每域128、108个域×指标、共8748条物理分项，
以及4001条训练total/hand/scene日志。正式产物位于wam-v2-metrics-20261015-r1，
完整JSON/含count的CSV/全曲线均已保存；预训练卡包含全部81点主指标及best/final全分项。

远端manifest/result/process均COMPLETED，exit0，80000更新耗时15.44h；
best按总dev objective为59000/0.774678，final80000/0.844785。最终三域hand all24
为14.454/72.421/49.318mm，scene all24为2.155/20.957/14.928mm（OakInk2/GRAB/ARCTIC）。
延续离线Probe级PROMISING，不形成抓取/Cm结论；ARCTIC静态scene分项仍恶化。
远端完整权重实际保留20k/40k/59k/60k/80k；现有robot父权重为80k final，未替换为best。

微调卡另补齐本次快照至9500的19次固定dev256记录，r2≤3000后衔接r5，同面板已核对。
预训练与微调dev均没有分别保存归一化hand/scene损失，微调只能恢复forecast合计；
训练日志中的分项与dev、不同数据面板之间不混比。未记录项已明确列出，未补造数值。

## 2026-10-15 阶段权重已保护，后续保存点自动归档

用户要求保留各阶段以比较action/forecast对下游的帮助。完整joint checkpoint现存
1000/3000/3500/4000/8000，均已用硬链接固定到robot-finetune-stage-archive-20261015-r1；
不是每个历史保存点都仍存在，部分已按旧规则轮换删除，不能用R-only快照补称完整权重。
CPU watcher1672685每30s保护后续保存点，主run完成/失败或4h后退出，16GB逻辑体积上限、
20GiB余量，无新增GPU、无训练源码/配置修改。archive-manifest记录完整权重和各loss；
现存action最低8000，forecast/total最低1000。真实完整8000加载通过；source unlink后
硬链接仍保留的工程smoke通过。阶段下游比较留待后续qualification，不以loss直接定优劣。

## 2026-10-15 完整补采已完成；人工审阅与原生输入实验并行

`dexplore-current-rollout-20261015-r1` 已全量完成：660条、166337状态帧、165677控制帧，
耗时1193.60s；88783帧存在原生手部接触。全量raw audit PASS，独立q_PD重算最大误差0，
原TRAIN/dev不变，已显式admit，位置窗口122623。数据一致性不等于成功抓取示范。
采集退出后恢复全局6卡上限；四卡微调继续，其余最多2卡用于新输入matched实验。

用户要求人工审阅。Task-local Viser工具按实际保存body/object pose和URDF重建回放，
可叠加当前参考、接触点、误差与现有筛选标记，并独立保存帧段标签；CPU读文件、
浏览器渲染。源数据和训练准入不自动修改。入口与端口转发见Task README。
用户明确要求同步推进真正native当前721D与baseline的对照；GTτ R-only诊断与
既有predτ四卡联合微调需区分，输入收益和物理抓取结果仍待实验。

Viser已在127.0.0.1:8768启动，独立标注output为dexplore-review-20261015-r1；
2项几何/标注测试、50物体mesh读取、3条真实不同长度回放、HTTP/WebSocket客户端
拖帧检查通过。服务器无头Firefox缺WebGL，三维画面仍待用户本地浏览器确认。
全660描述性审阅优先级已保存；确认s8_spheresmall_inspect_1的tick255手误差7.93m、
q平移到[5,-5]限位，而物体future24误差仅3cm仍通过位置筛选。这一实例直接说明
当前筛选不保证示范质量；尚不将统计排序自动当人工标签或改动训练窗口。

## 2026-10-15 三卡完整当前观测补采与四卡隔离续训

用户明确要求GPU0/1/2全量重rollout660条，并临时允许与四卡微调合计7卡。新增原生
delta0/721D观测、25刚体完整状态、实测接触力、current reference及frame_tick，与
同次q_PD同步。supervisor1548597，run dexplore-current-rollout-20261015-r1，batch16，
42批，继承原TRAIN/dev，8GB/10800s/20GiB底线；不导出future actor分支作为新输入。
完整当前观测不再用可重建近邻特征替代。修复旧collector短序列末尾zero-padding导致
teacher future查询偏离end-clamp：仅进程内repeat-last；不覆盖原数据。9项测试与
真实native smoke通过。首48条/9622帧独立raw审查通过，实测contact5252帧，PD重算0误差；
raw body→native observation独立数值重建一致。CPU auditor1553774在全660完成后自动
全量核对payload/时序/接触/PD，PASS后才admit。详见对应rollout实验卡。

原四卡r2于10:57在step3340因外部并发修改src触发implementation guard退出，最后完整
checkpoint3000保留。按用户“保留四卡微调”要求，使用独立detached工作树恢复：
tmp/robot-finetune-fixed-code-20261015-r3，代码fd83aebd（模型数值代码沿用dccd77ee，
只调整entry/runtime root与provenance路径，另复制相同PointWorld/ptv3依赖）。
RUNNING r5，supervisor1544990，GPU4/5/6/7/global512；相同config/source/optimizer/
4-rank RNG/dev panel，从3000继续至12000，实际已到3300。当前工作树的native模型修改
没有覆盖/回滚，也不再影响固定训练代码。r3缺少gitlink依赖、r4隔离IPC路径过长的启动
尝试均保留，未完成科学更新，不作为方法证据。resume日志ETA按累计step计算有误，
实际应使用elapsed_s/(step-3000)，约1s/新增更新、剩余约2.4h；后续用修正观察文件。

## 用户纠正：此前增量实验不是原生DExplore当前观测对照

此前只加可重建的参考q/物体pose误差24D和11手点CAD位移33D；未加实测contact、
16-key-body姿态/速度/角速度、完整参考误差或原生IG。UNCLEAR结果仅适用于这个
子集，不能用于否定DExplore信息。真正的当前信息实验尚未执行：应复用原生721D
单分支在delta0构建（原actor1442D=delta1+delta16），同次执行补采contact/body/
native observation与q_PD，保留τ合同并做matched对照。不能用原始几何660字段
伪装实测contact，也不能把11点object-frame近邻位移称为原生IG。详见原卡末尾合同。

## 2026-10-15 10:21 CST 当前参考/表面几何信息 Probe

用户允许当前参考目标及误差，排除新增未来参考。GPU0启动四臂matched R-only
Probe：baseline/current_reference/current_geometry/both，各3000更新、batch128，
统一从主训练step500的R快照开始，GT τ、数据抽样和新增encoder容量相同。新增输入为
同tick参考q/物体相对误差24D，以及11手点到CAD最近点的当前位移33D；没有新增接触力
或16-body动力学测量。独立审查和3项已知值/未来不变性测试通过。此为oracle τ下的
离线预测诊断，不能用作抓取或部署性能结论；四卡主训练保持运行。
详见[实验卡](experiments/probes/P-20261015-R-current-information.md)。

首轮3000步已完成并独立复算。baseline/reference/geometry/both 的dev腕部XYZ
mean/p95分别59.43/219.86、60.85/199.81、58.65/209.09、59.67/187.94mm。
两输入合用尾部改善14.5%，平均不变，按预设门为UNCLEAR；四臂曲线仍在下降。
据此启动唯一一次预算延长：r3两卡并行、同初始R重新训练每臂12000步；不再自动延长。
GPU0 baseline/geometry，GPU1 reference/both，每worker900s/150MB，合计300MB。

最终12000步四臂已全部完成，停止延长。selected mean/p95分别为baseline49.06/168.37、
reference52.78/187.03、geometry49.99/162.11、both54.42/176.04mm；均无整体收益，
维持UNCLEAR和原生产输入。独立原始PD复算、前3000轨迹一致性、选点及输入敏感性
检查通过；早期both尾部优势未保持，不否定尚未采集的接触/刚体动力学信息。

## 2026-10-15 10:01 CST 远端80k权重的四卡q_PD微调已启动

用户明确要求无warmup、q_PD监督，并改用四卡/更大batch。远端科学step80000已取回，
严格加载parent与其原始数值统计；R/optimizer从头初始化，预测τ动作损失更新WAM+R，
GT-τ anchor仅更新R。科学run `ptv3-shared-v2-robot-finetune-20261015-r2` 已在本地
GPU4/5/6/7启动，128/rank/global512、12000更新，supervisor1436540每60秒监督，
代码dccd77ee。双卡r1按用户新要求正常停止，保留原始FAILED/signal15回执，不作为负证据。

大batch工程门发现原RL660物体跟踪准入仍包含手漂移超过6m的窗口，导致预训练固定
体素域越界。保留原corpus/voxel origin/坐标/标签，新增显式observed-fixed-grid.v1：
TRAIN110777→110469，308窗跨6序列排除；DEV11908不变。逐序列排除tick与旧新索引
写入run的encoder_workspace_admission.json，R归一化和sampler均绑定新TRAIN成员。
这只验证当前encoder输入域，不代表剩余窗口均为成功示范。17项测试、独立只读审查、
四卡global512真实四步/评估/保存通过；工程显存约6.5GiB/卡，不形成学习/抓取结论。
运行协议和首段资源观测见对应实验卡与run的startup_observation.json。

10:03:44 CST 已到step80/12000、loss有限。30秒GPU4/5/6/7平均利用率约87/90/88/91%，
物理显存峰值约6.6GiB/卡；稳定0.951秒/update、538样本/s，剩余纯训练约3.16h，
dev/保存开销另计。首段无资源异常，尚未到step500首次dev，不作收敛或抓取判断。

用户要求平台检查：step2220快照下仅4次固定dev；1000→2000总损失5.202→5.262，
pred-τ action0.4897→0.4361、GT anchor0.4838→0.4282，动作仍改善约11%。由总损失
减去两项动作得到的forecast4.229→4.398回升，抵消动作收益。rank0训练日志500步分组
均值4.241→3.568→3.000仍下降（不同抽样，不与dev绝对值直接比较）。因此是总dev
目标早期平台/回升，不是整体训练或动作平台；保持原训练，至更多dev点复核。best按
总损失仍为step1000，不能据best不更新断言动作无进展。图与JSON在主run的
plateau-step2220.{png,json}。尚无物理抓取平台结论。

## 2026-10-15 q 三维合同与远端 WAM 适配

RL660 reference-q raw replay 与 nearest-live branch follow-up 均通过独立
command/PD/clock 审计，但 q_ref 直接执行仍为 UNPROMISING 信号（全程手/物体
误差大、held=0；整体 audit 因 native 短回放门保持 UNCLEAR），因此没有把 q_ref
改成监督标签。新 v2 adapter 位于 `src/task/ptv3-shared-wam-v2`，每个窗口显式
保存 `q_ref_next`、`q_pd`、`q_real/q_real_next`，动作损失只使用 q_pd 的
continuous-lift 编码；本地 checkpoint 的 4-step engineering smoke 已通过，
13 项 v2 合同测试通过。远端 v2 80k checkpoint 尚未能读取：反向 SSH
`127.0.0.1:22023` connection refused，直连 `100.64.0.23:22` timeout；因此
两卡科学微调尚未启动，不将本地 smoke 冒充远端结果。GPU4/6 保留给远端父权重
到达后的两卡启动，GPU5/7 既有 geometry-lift 已自然完成 step19000/19000。
适配器实现 commit 为 `d4b011a4`，新增的
`tools/run/launch_robot_finetune.py` 只监督和终止自己创建的 torchrun 进程组，
每60秒记录 progress/GPU/磁盘，8GB本run产物预算与20GiB可用空间安全线均独立于
已由用户解除的旧400GB总上限。远端状态缓存仍停在 step33520，不能据此伪造
最新 parent；需等 SSH 恢复后按 registry、稳定 hash 和 checkpoint 元数据验收。
04:07 CST 复核显示 `tailscale whois 100.64.0.23` 为 `peer not found`，本机
Tailscale netcheck 亦无可用 UDP/DERP；因此当前阻塞是远端网络/Windows 转发状态，
不是本地微调代码或 GPU 资源问题。

2026-10-14：本地1000步Probe已完成，局部PROMISING。固定三域dev的hand_all24
从29.72/123.10/63.58降到23.83/87.16/55.14mm；scene在GRAB/ARCTIC改善，
OakInk2略差。best/latest为step1000，不能据此形成抓取或Cm增益结论。
模型98,591,142参数，geometry11,092,000、shared85,054,464。

远端GPU0/2已启动80k长训：`pretrain-main3-remote-80k-20261014-r1`，代码1650ffce。
每卡128/global256，从seed136重新初始化，不迁移工程权重或静默扩展旧1000步配置。
最多48h、run8GB，384窗固定dev（每域128）每1000步评估；每2000步恢复点，
保护best/latest与20k里程碑。监督PID1718152、torchrun1718160、worker1718169/1718170。
目录 `/home/wbcd/workspace/oyx_ws/wam-v2-20261014/repo`；graspenv为Torch2.0.1/cu118。
复用只读NAS输入与原horizon统计；已通过双卡四步及完整模型step4→5恢复更新。
启动固定step460快照约0.682秒/update、375.5样本/s，纯训练剩余约15.1h；dev/保存另计。
30秒GPU0/2平均利用率96.8%/96.3%，物理显存最大29107/23733MiB，尚无训练后dev。
本地预算watcher3838855在393GB主动停止本run，初始合计344.48GB，共享硬限400GB。

长训/恢复须使用修复代码d0288294，并在启动shell先执行 `ulimit -n 65536`。
这是进程软上限；hard1048576与系统配置不变。默认1024会在mmap缓存约1018FD时失败。
修复还处理Torch2.0 fused AdamW恢复的CPU step标量；远端最小GPU回归及完整模型
step4→5恢复更新均通过，模型/optimizer/RNG恢复、有限梯度和实际参数更新已核对。
本地Torch2.4 GPU回归与13项v2合同测试通过；原有61项新旧测试记录保持。
未改模型、数据、优化配方或旧Task的5/7微调，不操作远端1/3上的其他任务。

预检报告：`outputs/ptv3-shared-wam-v2/remote-preflight-20261014-r1/summary.json`。
短工程检查不保证长期吞吐/利用率；正式长训仍需固定预算、完整配置与持续资源观察。
详见[实验卡](experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md)。

## 2026-10-15 新补采数据的原生721D matched R-only诊断

在原始四卡R5保持运行的同时，GPU0/2并行启动baseline/native_current两臂，
run `R-native-current-20261015-r1`，supervisor1601232，worker1601234/1601235。
代码6e08ed66；独立审查和2项关键合同测试、真实新660数据双臂各1步smoke通过。
使用新采集的同次执行q_PD、继承sequence split，GTτ24合同保持；额外输入是原生
当前delta0完整721D（含实测contact），没有future16分支。两臂相同step500 R、
721→128→128容量/初始化/采样，baseline在TRAIN统计归一化后mask；各12000步，
1200s/150MB一臂，不自动延长，6卡总上限。新执行的dev512面板重新固定并保存
sequence/tick，不能沿用旧数据行号。此为GTτ离线动作预测诊断，未形成predτ联合
训练收益或抓取结论；结果自动汇总到run/result.json及native-current.png。
详见[P-20261015-R-native-current](experiments/probes/P-20261015-R-native-current.md)。


原生721D R-only诊断已完成：两臂各12000步、GPU0/2已释放，自动matched检查与
独立raw q/q_PD XYZ重算通过。固定dev512 selected baseline9500为mean64.08/
p95222.52mm，native11500为57.23/192.58mm，改善10.69%/13.45%，clipping0/0.1953%。
未满足预设两者均≥15%门，结论UNCLEAR，不延长或改生产输入；GTτ离线有利趋势
不能直接解释为predτ联合微调或物理抓取收益，也不能单独归因contact。耗时247.46s、
总产物82.7MB，主四卡R5未受干扰。卡与run/result.json保存selected/final完整证据。
