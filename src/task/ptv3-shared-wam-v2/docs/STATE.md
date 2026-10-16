## 2026-10-16 query控制头容量/goal五臂完成，Shadow BC→RL已同步试跑

冻结原shared7k WAM、同strict窗口和完整tokens，fresh query头6000步五臂全部完成。
DEV256 XYZ：128无goal47.151、128直接goal48.291、512无goal43.304、512直接goal48.808、
512直接goal+721直连45.662mm。同宽初始张量、所有首draw完全匹配。无goal扩宽mean/P95/
startup改善8.16%/16.53%/21.75%，未达预设mean10%门；直接goal和bypass无一致收益。
UNCLEAR，不据此关闭容量/goal假设或判定128瓶颈无关，也不与目标不同的腕部MLP直接归因。
下一步优先固定头区分单拍/8拍、腕部/全动作监督；不扩大joint WAM更新。
512无goal原生4-control smoke通过，仅工程证据，不是完整抓取评估；goal-on/bypass未物理验证。
详见P-20261016-query-capacity-goal及run的summary/results.json/comparison.png。

新wam-rl-policy Task完成Shadow冻结9k BC+新actor/critic的20更新、127360 env steps；
三个配对reset保持BC/PPO为5/8、4/3、2/4（总11/48→15/48、DEV1/24→3/24），
official均26/48。BC自身还存在同初态数值回放波动，故UNCLEAR，不能宣布RL收益。
全部冻结、canonical/force/torque/target、初态和raw复算通过。见
`src/task/wam-rl-policy/docs/STATE.md`与P-20261016-shadow-bc-ppo；learned RL-token未开始。
远端训练数据读取和小型GPU forward/backward已修复验证，未做远端native仿真。
本轮所有进程结束，GPU1–7释放，GPU0外部任务保留。

## 2026-10-16 新腕部MLP实际抓取评估完成：shared收益因对象而异

用户要求直接核对行为，已运行固定seed1966k的none/aligned腕部MLP，手指共用原
shared7k控制器，实时own-row shared，不回放离线缓存。GPU1–3三组native完成并全部
通过独立执行审计：最长保持none/aligned为飞机40/60、球75/56、杯58/23帧，teacher
63/79/98。两组各两条观察到hold45，不能外推为成功率；none飞机末段37帧仍在保持，
其40帧阈值读出不能等同完整失败。aligned杯明显落下；腕部均无clipping，手指裁剪增多
只是伴随现象。shared效用记UNCLEAR，不据单对象或离线误差统一替换原811D控制路径。

7项测试、4帧smoke、CPU命令重放和独立review通过；追加12tick GPU重建WAM/MLP回放，
腕部XYZ最大差0.069mm，详见[实验卡](experiments/probes/P-20261016-wrist-shared-grasp.md)。
本轮约78.7s含审计、24MiB输出；GPU1–3已释放。保留新MLP行为基线，下一步若比较
路线应固定更大行为面板并审查释放/恢复阶段，不能以offline mean替代抓取表现。

## 2026-10-16 同一腕部MLP±shared：14组有界对照完成

同811D直通、原腕部单拍目标、同优化与抽样下，none/aligned/shuffled三次配对6k的
TRAIN256均值16.46/14.37/15.13mm，DEV256为37.84/42.54/40.52mm。aligned每次均
比none差4.45–4.87mm，但TRAIN更准；首seed none精确复现历史MLP。固定原MLP后的
线性shared残差也未改善DEV（37.83→38.16mm）。本次均值pooling接法UNPROMISING，
不推广为WAM无效，不进native/joint长训。低LR单变量控制拖慢拟合，高WD未明显恶化；
原128D控制头/融合/八拍目标差异仍待匹配定位。独立只读语义review、TRAIN-only统计、
分片覆盖和三次配对初始身份通过；详见[实验卡](experiments/probes/P-20261016-wrist-shared-matched.md)。

本地四卡冻结提取108.6s，MLP每组约20s；本地1–5已释放。远端同形状单卡/双卡DDP
实测825/366 updates/s，故小MLP使用每卡独立对照。远端portable数据仍RECEIVING，
未以其宣称实际远端学习结果。输出outputs/ptv3-shared-wam-v2/wrist-shared-matched-20261016-r1/。

## 2026-10-16 721D shared首控定位：stored TRAIN仍未拟合精确

接续原会话后，用已通过native执行的固定代码在GPU1对有效fresh8两组7k做冻结推理，
54.88s完成。相同TRAIN balanced256上masked/current首控XYZ为58.08/56.05mm，
startup为65.50/63.31mm；同一DEV256为60.10/56.12mm。既有wrist-only current MLP
在同样评估身份上TRAIN/DEV为16.69/37.83mm，但其结构/目标/采样不同，不能将差异
因果归给WAM或某个接口。源码复核确认两版同时改变输入融合、控制头、动作目标与优化。
下一步优先同一腕部MLP的±shared配对，保持原811D输入直通和训练设置，先冻结WAM
隔离特征效用；joint vs冻结WAM仅训R留作后续独立问题。本轮未启动新训练/PPO。

现有fresh8 current飞机/球首拍PD误差46.49/56.95mm；tick16腕部相对同次teacher
偏122.3/134.0mm，早于teacher接近物体。完整轨迹held独立重算一致，球partial单列。
共同4k前缀评估实际已结束：纯单拍current杯子held92/91，其他对象没有一致收益，
不将其扩展为主线。Probe仍UNCLEAR；详见[实验卡](experiments/probes/P-20261016-shared-action-horizon.md)
及outputs/ptv3-shared-wam-v2/current-policy-fit-20261016-r1/。GPU1已释放。

## 2026-10-16 远端微调数据核验：预训练副本不足，独立补传已启动

用户要求最近UniDex/Shadow与DExplore/Inspire微调可在远端运行。现用远端九源scope只有运动预训练native数据，不含Shadow原生300D状态/8×24命令cache；现用远端repo也没有DExplore新补采660/current721副本。已定位实际依赖：Shadow full12,456条（TRAIN1,536,897/DEV354,761窗，2.185GiB），DExplore660含current721（0.849GiB），两套DExplore参考/成员数据0.743GiB，以及CAD、原固定面板、controls和实验起点权重。v2预训练80k远端已存在，直接在远端复制入独立包，避免回传。

Decision Note：在用户数据迁移授权内选择独立NAS目录 `/mnt/ugreen_nas/storage/oyx_storage/robot-finetune-portable-20261016-r1/repo`，不改现有训练目录/本地数据。迁移包共6.063GiB/18,007文件（其中2.237GiB为shared7k与旧对照起点权重），代码固定271ebda5；只重定位JSON路径、刷新原controls输入文件stamp，数值归一化/物理样本/窗口/标记不变。无新训练、无全UniDex raw重复传输（raw_episode只保留物理身份追溯，完整raw复审需另接）。

上传supervisor本地PID3511323，六连接64MiB分块/有限重试/最长3h，远端磁盘约47TiB可用、20GiB底线。最初单连接仅约1MB/s；最终ETA等待分块实际吞吐。完成后自动解包、文件闭包检查、原controls/stamp/DEV身份检查，以及Shadow TRAIN/DEV和V2RLRobotWindows严格93680/10376真实CPU批读取；只有reader-audit PASS才标DATA_READY。当前仍为上传中，不宣称远端微调已可运行。脚本与进度 `outputs/ptv3-shared-wam-v2/remote-finetune-data-20261016-r1/`，远端部署状态deployment.json。

运行环境另查：graspenv有现有PTv3训练栈但未登记Isaac Gym；dexplore-4090登记了现存Isaac Gym源码，Torch2.0.1+cu118与graspenv一致，但自身无spconv/flash-attn。已通过添加现有Gym路径使graspenv与Isaac Gym共同导入（exit0），远端scope/env.sh已保存环境设置；尚未运行仿真或GPU微调smoke，数据复制不能替代执行验收。保留所有现有进程/权重；不自动恢复训练。

## 2026-10-16 仓库复核：训练已结束，监督跨度结论需纠正

本地v2实验现均未占GPU。shared-current追加1k已完成，masked/current XYZ61.73/60.87mm，
三条验证学生均held0；其卡片原RUNNING状态已更正。夜间7k运行有新保持信号：有效8拍
current权重杯子held60，其余对象失败/越界。混合协议权重杯子held117不归因单拍训练。

关键语义审查更正：action输出/标签一直8×15，24是forecast；原continuation未因缺省配置
误跑24拍，撤销这项INVALID_IMPLEMENTATION，fresh8是额外重复。真正错误是single两组
稀疏恢复用了不读取action_supervision_steps的旧固定源码：masked尾2000步、delta0尾1000
步切成8拍监督。最终7k不能支撑完整1拍vs8拍对照。CPU实际函数复现和独立review一致，
原日志/权重/results_summary保留，更正见horizon-runtime-audit-20261016-r1/correction.json。

共同4k的纯协议checkpoint仍在，优先核验并补闭环，避免重复长训；本次未启动训练。
current721分块证据提示联合当前运动/同tick参考误差信息有用，非contact单块主导，仍是
单seed诊断。Residual只有离线零残差恒等检查，未完成server集成、未启动PPO。

以下为历史记录，不代表当前任务仍在运行。

## 2026-10-15 旧Inspire两臂12k已完成，不再延长

实际result.json均COMPLETED、完整step12000.pt存在；旧collector因用户停止C而提前退出，
其FAILED_ARM不是A/B训练仍在跑。固定DEV256，pred_tau/shared的action为0.38060/0.39818，
XYZ mean59.48/60.51mm、p95186.34/183.07mm；shared未满足mean/p95同时改善20%门。
10k→12k动作仅降约1.02%/0.91%，不延长原输入训练。shared按p95选点仍7k，当前新对照
从此7k起步；不把7k失败自动推广成未测试的12k抓取结果。C停7920、最新完整7k保留。
用户询问是否可停止，旧Inspire现已无需再停；Shadow9k仍慢降，另做原16env最新行为读出。

## 2026-10-15 原shared联合目标的当前信息配对已启动

用户要求继续对照，新增shared-current-continuation：从同一shared7k旧权重起步，
同原8×15 q_PD、forecast、WAM/R均更新、旧LR，两组fresh optimizer；同721→128→128
零末层支路，masked对照与delta0实际输入仅值不同。strict93680/10376及原DEV256，
追加1000步，固定终点三条val K1闭环/独立audit自动接续。不存在GTtau anchor/warmup。
相关测试与独立review通过，冻结e3333e76，supervisor2589005，GPU6/7配对smoke和
native4步审计均PASS（max<=3.28e-7），初始loss均0.2522353232、actor/stat/旧DEV一致。
监督器已进入正式train1000阶段，科学结果尚未产生。
正式产物shared-current-continuation-20261015-r2；r1因vendor缺失在GPU计算前退出，
证据保留，不计方法负结果。当前尚无该配对科学结果；详见P-20261015-shared-current-continuation。

## 2026-10-15 当前完整teacher信息腕部恢复局部抓取信号

current-wrist-mlp两臂固定6k离线后，替换原shared7k腕部6D、保留各自shared手指，
无teacher动作干预。TRAIN飞机teacher/A现有状态/B加当前721 held106/0/49，
VAL飞机62/0/41。B分别抬升0.605/0.769m；TRAIN随后掉落，VAL有两段保持（最长41、
末段36至评估终点），尚不能称稳定泛化成功。两组audit通过，max<=3.01e-7，初始q差0。
B手指clipping约23–30%，腕部clipping0。局部行为PROMISING，offline预设强信号仍UNCLEAR。
支持优先补回当前teacher输入bundle，不单归因contact/q_ref，也不直接证明数据量无影响。
归档runtime新6D MLP单样本CUDA偏差经逐层重放定位，仅在新入口重复batch2解决，复核
CPU replay约3e-8；不影响原shared在graspenv batch2的历史执行。训练f816b3d5、
native7b2576d5，详见P-20261015-current-wrist-mlp。6/7释放，Shadow4/5继续。

## 2026-10-15 腕部拟合诊断完成，当前信息MLP配对Probe

shared7k精确stored TRAIN256首拍XYZ66.42mm，startup76.42mm；DEV256为62.70mm，
两面板起步占比不同，不用总均值判泛化。四条起步状态/H4/goal与原native一致，batch
1/2/16所查差<0.1mm。闭式回归加入当前腕姿误差DEV82.13→82.53mm无改善。
当前优先区分信息和学习设置，不能只怪闭环、val或数据少。详见shared-execution-gap卡。

current-wrist-mlp配对已20s完成6000步：完全匹配的小MLP、first native wrist6，A输入
state87+已有object goal24，B再加captured current delta0 721D。DEV256均值42.60/37.83mm，
相同原shared7k为62.70mm；全DEV48.23/40.13mm，起步85.09/56.93mm。当前信息有增益，
但均值改善未达预设20%强信号线，UNCLEAR。两臂均优于旧shared是不同网络/损失/学习率/
goal显式入口的联合变化，不能只归因学习率或压缩。没有未来手参考，无抓取成功结论。
代码f816b3d5、独立review无blocker，6/7释放；详见current-wrist-mlp卡。Shadow4/5继续。

## 2026-10-15 混控定位到腕部瓶颈，动作表示2k未恢复抓取

动作表示driver2298258完成，同2000步native/q_PD的XYZ mean78.66/77.37mm、finger
0.07374/0.08945rad；手指改善17.6%，两臂三条val所有student仍held0，六组audit通过。
局部UNPROMISING，不再据此扩扫checkpoint；不refute native长训。

条件接续driver2365418也完成：固定shared7k、val飞机teacher held62，teacher腕部+
student手指held59，反向混控held0；独立audit通过。TRAIN飞机teacher held103，两个
student都held0。腕部是本例优先瓶颈，不能只归因val泛化或握不牢。
同状态teacher建议对比：teacher腕部引导行的student腕部预测差仍约57–71mm，自主
腕部行进一步放大。下一步优先精确TRAIN teacher-forcing拟合与参考任务条件合同，
区分预测拟合差和闭环放大，不继续凭动作loss增加训练。详见component-oracle卡。

## 2026-10-15 动作表示配对Probe启动记录

用户新目标定位Inspire/Shadow抓取差异，提出q_PD对比原生action并临时允许空卡实验。
独立CPU审计660序列首拍PD逆命令max5.96e-7，暂未见粗大label/clock/旋转branch错误；
两边均std归一化L1，且Inspire首拍XYZ标签/std与native实际相同。主要待检验变量是
finger相对编码、额外旋转/维度权重、未来8拍监督及其优化影响。

两臂GPU6/7，同80k/fresh actor/87D+shared/strict数据/dev256/seed195/LR/batch256，
native12 vs geometry15各2k后自动三条K1抓取；不混完整teacher输入。固定076b5c31，
driver2298258，run=inspire-action-representation-20261015-r1。smoke及matched初始化PASS，
15单测PASS，正式RUNNING，约1.8s/步、10.4GiB/卡。中途750步native/geometry的
DEV XYZ mean95.43/90.84mm、finger MAE0.08746/0.10449rad；原生手指稍好但位置略差，
尚无全面改善，不据中途改选点，终点2k及native抓取待完成。
详见P-20261015-inspire-action-representation及跨Task动作表示research笔记。

CPU回看shared7k三条已完成native：学生最小手物表面gap仍0.444/0.250/0.536m；
第3/4/4帧腕部已偏离teacher5cm，teacher第42/27/43帧才接近3cm。因此这三条首先失败
于接近阶段，不能只解释为握不牢。新增teacher腕部/student手指和相反混控诊断已实现，
8单测、旧日志回归和独立代码审查通过；尚未运行GPU，待现有配对释放卡后先4-step
smoke再单条teacher-positive定位。混控不计作学生自主成功；详见component-oracle卡。
条件接续监督器2365418已启动并确认存活，固定c1d2dca1，当前仅等待配对pipeline2298258，
不占GPU；上游完成且两臂未held45才在6/7空闲后执行混控及一条TRAIN自主资格对照。
否则跳过或退出，不重启上游，不停止其他训练。实际oracle/TRAIN结果尚未产生。

## 2026-10-15 shared原生抓取初筛完成，资源调整

用户要求shared_tau停止、GPU4/5转Shadow双卡扩大训练。supervisor1922941已SIGTERM退出；
停止进度7920，最近完整7000，1k/2k/4k/6k/7k保留。A=pred_tau/B=shared仍继续12k，
C没有12k结果，原三臂联合12k协议未完成，signal15不是数值失败。

新增明确shared bridge，GPU6/7并行测试当时control_best7k和forecast_best1k，各3条val
飞机/小球/杯子、seed148、30Hz K1。teacher held分别62/76/92和62/74/95；学生两点均
三条held0、最大抬升<=3.289mm，两student工程重复也相同。6组执行和独立PD/输入审计PASS。
局部UNPROMISING，不全局refute shared，不将Shadow成功率与此不同metric横比。
详见P-20261015-shared-native-grasp，输出shared-native-grasp-20261015-r1，6/7已释放。

## 2026-10-15 shared actor同步6000步：尾部改善，均值未改善

三臂实际进程仍存活，最新约6040–6340步。固定共同DEV256、同一步6000：
pred_tau/shared/shared_tau的action L1为0.39389/0.41533/0.41962；
first-control XYZ mean为62.357/62.536/63.266mm，p95为214.289/171.351/176.794mm。
shared的p95相对A降低20.04%，shared_tau降低17.50%，但mean分别高0.29%/1.46%；
仍未达到原定12k双指标各>=20%的门，不据中期尾部改善提前选路线。
5000→6000动作L1仍下降1.66%/1.50%/0.98%；DEV forecast继续上升，均值收益放缓。

三臂6000完整checkpoint已CPU实际加载审计：模型/optimizer有限、父forecast统计准确、
六个actor normalizer跨臂完全相同、两rank RNG存在，三组LR维持原值。
shared少一个optimizer state来自未使用的τ时间参数，符合原实现，不是缺权重。
1k/2k/4k/6k文件全部存在；其中本次只对6k做内容审计，不冒充已检查所有历史阶段。
快照/四项曲线与审计在shared-actor-interface-20261015-r1的
`interim-step6000.{json,png,svg}`及`checkpoint-audit-step6000.json`。
GPU0–5继续12k训练；Shadow临时单卡起步采样Probe占GPU6，GPU7空闲，详见Shadow Task状态。

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
