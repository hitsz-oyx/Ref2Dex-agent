## 2026-10-16 v3最终80k与v2同面板比较完成

复用原固定544抽样（541不同窗口，旧抽样重复权重保留）与v2 best59k/final80k逐窗物理读数，新评v3 best31k/低LR best41k/final80k共1632次前向；index/capture/有效计数逐项一致，各自TRAIN normalizer反归一化。相对v2best，v3final原三域手宏误差44.933→44.392mm（-1.20%），scene12.365→12.356mm（近持平）；原三域手配对区间都跨零，视作基本保持。新增五域手44.469→28.793mm（-35.25%）、scene17.662→11.410mm（-35.39%）。HOT手仅改善3.61%且CI跨零，目标物体first8恶化24.15%，不称全面改善；其余四新增域手CI低于零，仍仅32窗/源的小面板。v3final相对v2final原三域手/scene改善2.21%/2.55%，新增五域38.60%/37.13%。

重要修正：31k只是原256窗归一化综合loss最优，在更大的原三域物理面板上，手宏误差31k46.969、41k44.968、80k44.392mm。不能把31k作为所有下游的唯一最优；保留41k/80k为后续微调候选，最终取决于同配方下游评估。本轮PROMISING仅指新增域覆盖保留，原三域基本保持，不作数据源因果或机器人抓取结论。v3最终global512×80k的抽取曝光为v2final/global256的两倍。

新推理80.2秒、GPU峰值1.15GiB/3GiB上限，已释放；固定原读数函数013105b1。首个工程入口错误要求抽样唯一，推理前退出，r2移除此假设并保留原面板。完整报告/图/配对区间 `outputs/ptv3-shared-wam-v3/main3-v2-v3-final-comparison-20261016-r2/`；未新增训练、未读取TEST，DexYCB无旧recipe DEV。详见[实验卡](experiments/probes/P-20261015-nine-source-pretrain.md)。

## 2026-10-16 09:21 核验：80k训练已正常结束，后半程未刷新best

远端降LR续训约05:57 CST写完step80000完整checkpoint，process/progress均COMPLETED、exit_code0；当前四卡均0MiB、训练资源已释放。两组LR维持3e-5，输入仍是原strict冻结recipe，未部署Dex扩充或boundary修复。旧31k/37k、新41k/80k checkpoint均确认存在（各约1.18GB）。

相同固定256窗八源DEV：旧31k best1.03455；低LR41k best1.03699；最终80k1.06463。38–42k与76–80k五次均值比较，TRAIN0.23293→0.13257（-43.1%），DEV1.05458→1.07998（+2.41%），手all24宏误差33.527→34.281mm（+2.25%），scene12.953→12.927mm（-0.20%）。后半程未突破验证平台；继续相同配方增加步数缺乏正向信号，不据此正式判定过拟合或MSE无效。

本次仅状态/CPU统计，无新训练。建议后续以31k（综合DEV最优）和41k（低LR、部分物体/短期指标更好）作为候选，按下游用途评估，不默认使用最终80k。完成快照 `outputs/ptv3-shared-wam-v3/final-status-20261016-r1/{snapshot.json,summary.json}`；详见[实验卡](experiments/probes/P-20261015-nine-source-pretrain.md)。

## 2026-10-15 23:58 降LR约12k步：平均DEV改善，历史最佳未突破

37k两组LR由1e-4降至3e-5后，当前49220/80000，最新49k DEV1.06541；四卡99–100%、显存33.01/33.67/33.18/23.45GB，约1.44–1.47步/秒，剩余约6–6.5h含验证保存。相同256窗八源DEV上，33–37k、38–42k、45–49k五次评估均值依次1.10834→1.05458→1.07197，最近仍较降LR前改善3.28%，但较初期回升1.65%。手all24宏EPE34.318→33.527→33.959mm，scene13.736→12.953→13.337mm。低LR阶段best41k/1.03699仍未超过旧31k/1.03455；TRAIN均值0.31788→0.23293→0.22290，不能据其下降宣称平台突破。48k升高在49k回落，并非持续发散。

诊断为有限的平均改善、验证平台仍在；属于单条自适应轨迹描述，无恒定LR匹配对照，不能将差异全归因于LR。保持当前已授权运行与数据冻结；本次只读监控/CPU统计。证据与曲线 `outputs/ptv3-shared-wam-v3/lr-effect-20261015-2353/`，详见[预训练卡](experiments/probes/P-20261015-nine-source-pretrain.md)。

## 2026-10-15 22:30 同样本比较：原三域约持平，新增域预测改善

对v2 best59k/final80k与v3降LR后40k，在完全相同384旧三域DEV窗+160native窗重评，模型各自normalizer/同物理mm指标；旧日志重放偏差<1e-6mm。相对v2final，原三域手all24改善1.1–2.3%、scene改善2.6–7.2%，多数原域配对CI跨零，视作基本持平。相对旧best仍有ARCTIC手0.86%/GRABscene2.63%退步。五新增域手误差下降12.2–65.8%，但每源32窗，HOT目标物体first8还由3.58→4.00mm。诊断PROMISING支持继续广域预训练，不作九源因果/抓取收益结论。旧80k/global256与新40k/global512均20.48M抽取，但seed/曝光/训练过程不同。

代码013105b1，1632次窗口推理248.7s、显存峰值1.15GiB，独立审查/2测试通过；诊断显存已释放，主四卡41680步、3e-5/约1.446步每秒。报告与图 `outputs/ptv3-shared-wam-v3/main3-v2-v3-comparison-20261015-r1/`；详见[预训练卡](experiments/probes/P-20261015-nine-source-pretrain.md)。

## 2026-10-15 21:37 已按用户要求降LR并恢复四卡持续训练

当前活动run为 `pretrain-nine-strict-lr3e5-remote4-80k-20261015-r1`（390ac393），短目录 `/home/wbcd/workspace/oyx_ws/wam-v3-lr/repo`，supervisor1925134/torchrun1925207。两组LR1e-4→3e-5，旧strict在37k完整点停止；20步四卡恢复检查后，从完整37020模型/AdamW/四rank RNG继续，目标仍累计80k/48h。同recipe/normalizer/数据配比/DEV/global512；未切Dex扩充或boundary配方。旧31k best和37k checkpoint保留。

37060日志确认实际两LR3e-5、loss/grad有限，1.468步/秒；四卡99%util、约27.25/27.91/27.42/17.69GB，剩余约8.5–9h。四rank optimizer步数与RNG恢复通过；恢复前后37k固定DEV1.07435261完全一致，20步后1.05039101，未超过旧31k best1.03455，不作效果结论。7合同测试与独立审查通过，nullable-key gate补丁回归通过。新run best别名是阶段内选择，跨阶段须保留旧best参照。证据 `outputs/ptv3-shared-wam-v3/lr-drop-20261015-r1/startup-audit.json`；详见[预训练卡](experiments/probes/P-20261015-nine-source-pretrain.md)。21:37复查已到37260，最近四段1.440–1.468步/秒，四卡85–90%util、33.01/33.67/27.42/17.69GB，无failure；两LR仍3e-5。稳态回执同目录steady-state.json。后续观察2–3k低LR更新是否改善固定DEV与逐源物理误差。

## 2026-10-15 21:19 训练约36k：DEV平台/波动更明显

远端36420/80000（45.5%），best31000/1.03455、latest36000/1.11735；四卡100%，约1.466步/秒。29–32k与33–36k块比较，TRAIN0.3474→0.3216（-7.42%），DEV1.0959→1.1168（+1.91%），手/物体物理误差均略回升。31k比旧27k best仍改善3.06%，此后5次评估未刷新；当前验证平台迹象更明确，未正式证明过拟合。保持运行/LR不变，40k约40–45分钟后到，届时应复查降LR小对照；80k剩余约8.5–9h。新曲线与回执 `outputs/ptv3-shared-wam-v3/plateau-check-20261015-2119/`。

## 2026-10-15 19:41 训练约28k：训练loss下降，DEV接近平台

远端strict28040/80000，best27000，四卡正常、1.44–1.47步/秒，剩余约10–11h。21–24k与25–28k评估块相比，TRAIN均值降9.65%，DEV宏objective仅降1.11%，手all24 EPE34.554→34.418mm（0.39%）；best22k→27k仅改善0.87%。HOT/H2O部分手指标停滞或回升，TACO手/GRAB物体/ARCTIC短期手仍改善。判断为验证收益放缓/近平台，非整体训练收敛；固定DEV仅256窗。保持本次运行与LR不变，建议40k里程碑再检查、必要时做降LR小对照。证据 `outputs/ptv3-shared-wam-v3/plateau-check-20261015-1941/`，详见[预训练卡](experiments/probes/P-20261015-nine-source-pretrain.md)。

## 2026-10-15 MSE/FM现有权重诊断：衰减存在，多模态证据不足

冻结step16000完成1024 DEV/11216 TRAIN诊断，按原始录制去重570次。幅度中位数0.72–0.92xGT、移动速度0.46–0.82；各源ADE中位数均优于零运动/匀速。2cm下无>=8独立邻居的查询，5cm下仅ARCTIC2/H2O2，未见稳健双空间多模态，判断UNCLEAR，继续MSE基线；本轮未训练FM。H2O拆段去重已修正，UniDex跨batch相对名保持独立。最终分析56a1da34、artifact analysis-20261015-r4，4测试与独立审查通过。远端80k主训练持续，诊断GPU已释放。详见[MSE/FM报告](mse-flowmatching-audit-20261015.md)。

## 2026-10-15 复审后远端实时状态核验

19:17 CST只读SSH再次确认supervisor1892360与torchrun1892489均存活，strict run实际
step25980/80000；progress近期1.468步/秒、剩余ETA约10.22小时，loss0.39517，
hand0.10279/scene0.29238有限。GPU瞬时利用率0/100/100/100%，显存约29.4/42.8/41.6/38.1GB；
该单点不能当稳态GPU0故障，整体仍在正常更新。未停止或切换输入，仍有下面记录的8窗历史/
未来采样暴露限制，新boundary配方未部署。本次原始回执
`tmp/nine-remote-20261015-r1/runtime-20261015-1917.json`；此前18260步回执也保留。
该快照是运行核验，不是DEV或正式效果结论；后续实际步数以远端progress/进程为准。

## 2026-10-15 九源复审确认插值边界漏筛，修正版仅派生未部署

本轮只读复审40161条序列，确认HOI4D8个TRAIN窗口的未来23→24帧包含5.65–8.11m/s
插值半段，旧strict与expanded都有；原生转换两端AND候选预选导致遗漏。root独立复算
8个raw/valid标签通过，未见TEST/group泄漏或payload变更。详见2026-10-15-nine-source-reaudit.md。
修复代码ea5c78f6用full_capture选候选，再执行原inclusive窗口隔离；拒绝从已有quality
配方重复重建，防全局索引漂移。4回归通过。新strict-boundary与expanded-boundary两份
快照均额外排78TRAIN/8DEV并保留旧排除；8个真实异常均被隔离。统计数值沿用各自父参考。
**远端当前训练仍用旧strict，未在运行中替换输入**；新快照待后续显式数据切换。
数据修复与当前Inspire/Shadow任务互不修改。完整Decision Note见motion-boundary-fix-20261015.md。

# 数据更新：DexYCB全被试已补齐，本地与远端新九源配方验收通过

按用户要求已下载/转换DexYCB全部10被试1000段/72,746帧，官方s1原始700 TRAIN/100 DEV/200 TEST；一条TRAIN因手有效覆盖不足无准入窗，清洗后699 TRAIN/100 DEV、3,096 TRAIN窗/382 DEV窗。TEST200段独立保存。相较旧346个TRAIN窗约增加至8.95倍，未重复八相机。

新配方 `outputs/ptv3-shared-wam-v3/nine-source-expanded-dexycb-20261015-r1`：4,805,826 TRAIN/821,791 DEV，Dex概率1%（既有上限）。新native `outputs/3d-data-preparation/native3d-expanded-dexycb-20261015-r1` 物化此前所有native严格隔离；非Dex其余八源准入集合不变，main3全局排除索引不变，人工隔离row及其嵌套provenance已重映射。重新计算TRAIN-only24-horizon共同normalizer，未读取TEST拟合。

远端新输入在既有scope的 `data-dexycb-full-r1/recipe`；部署代码f2c2b35c、短目录 `/home/wbcd/workspace/oyx_ws/wam-v3-dex/repo`。本地/远端真实mixed-reader、端点重算、独立集合/统计审查及3项测试通过；共同mean/std数值完全一致。当前远端80k运行仍使用以下原严格冻结recipe，本次没有修改运行中的模型/normalizer/优化器/采样。新配方供后续显式训练阶段使用，DEV新增第九源，不能将其曲线与旧八源panel当匹配对照。

Viser http://127.0.0.1:8769 已切扩充版，PID1955156，默认source8 subject01 row37491/frame19，可选Dex DEV。标记独立保存到 `nine-source-expanded-dexycb-review-20261015-r1/annotations`。

完整记录见[DexYCB扩产](../../3d-data-preparation/docs/research/2026-10-15-dexycb-full-expansion.md)。以下运行历史保留；数据候选已更新，正在运行的训练配方保持冻结。

# 当前：九源严格运动准入已完成，远端已从step8000恢复

用户明确授权：模棱两可的高幅度数据直接隔离，阈值可以更紧。全量39,528条运动审计完成；严格规则为物体原点速度>5m/s、有效手部点速度>5m/s或物体SO(3)角速度>40rad/s，使用实际观测时间（HOI4D原生15Hz）。隔离所有覆盖异常转换的H4+K24窗口；UniDex隔离异常之后尾段；HOI4D37130另隔离用户确认的初始0..30帧。

在manual-r2基础上额外排除2,943 TRAIN/204 DEV窗，最终4,803,076 TRAIN/821,409 DEV。TRAIN逐源排除：OakInk2 1803、GRAB4、ARCTIC0、UniDex828、TACO79、HOT16、DexYCB5、H2O4、HOI4D204。原始payload/CAD/index保留。阈值是用户偏好下的保守质量准入，可能丢弃合理快动作，不能保证剩余数据没有异常。

本地recipe `outputs/ptv3-shared-wam-v3/nine-source-strict-quality-20261015-r1`，远端 `data-strict-r1/recipe`（共同scope不变）。固定代码3709c966，远端短目录 `/home/wbcd/workspace/oyx_ws/wam-v3-q1/repo`。6项独立合同测试、实际subset-resume复核、本地/远端mixed-reader和排除采样检查均通过。继续训练保持父模型24-horizon数值normalizer、来源概率、optimizer与各rank RNG；统计是此前TRAIN拟合参考，DEV面板明确变更。

Viser `http://127.0.0.1:8769`，PID1887923、冻结代码3709c966，默认HOI4D37130/frame26；全39,528条完整轨迹可选，被排除片段仍可回看。新标记目录 `nine-source-strict-review-20261015-r1/annotations`；旧标记均保留。九源真实reader和五条已知异常样本准入检查通过、HTTP200。

远端manual-r2已在完整step8000保存后停止own run。实际checkpoint四个normalizer buffer逐个一致、optimizer和四rank RNG恢复校验通过；新run `pretrain-nine-strict-remote4-80k-20261015-r1` 已恢复并稳定更新至step8080，supervisor1892360、torchrun1892489。九源gradient-audit/新八源初始DEV有限，四卡约82–99%利用率、显存17.7–36.1GiB；稳态1.465步/秒（global512），剩余纯更新约13.6h，含验证/保存暂估14–16h。无OOM/非有限量，预算仍总80k/48h/4卡。完整运行/切换回执见远端scope下 `strict-quality-{stop-receipt,checkpoint-audit,training-launch}.json`。CPU文件审计不占GPU；项目远端四卡计入全局6卡预算，其余任务最多2卡。本地v3不恢复。

以下为此前运行历史；当前准入以上述严格版本为准。

# 当前：远端四卡motion九源全量预训练已稳定更新

用户已授权直接开始。运行`pretrain-nine-motion-remote4-80k-20261015-r1`，远端supervisor1877848，固定代码6e535bc1、短目录`/home/wbcd/workspace/oyx_ws/wam-v3-20261015/repo`。使用下述motion清洗recipe，从头初始化、GPU0/1/2/3、每卡128/global512、80k更新/48h/8GB预算；本地GPU当前空闲。启动检查已通过：step120，四卡98–100%利用率、显存约22–32GiB，loss/grad有限；完整step1 checkpoint已保存。近期1.4–1.47更新/秒（global512），80k暂估16–18h；训练后首次固定DEV在step1000。当前具体步数/失败原因以远端process/progress.json及本地`tmp/nine-remote-20261015-r1/training-startup-monitor.jsonl`为准。

按用户“先筛查清洗”要求，全量扫描UniDex33,056条，版本化规则隔离25条中的136窗（TRAIN114/DEV22）。保留15条的可信完整前缀，10条无剩余窗；不按抓取成功过滤、不改原始payload，其余八源窗集合不变。

新native view：`outputs/3d-data-preparation/native3d-clean-motion-20261015-r1`；新九源recipe：`outputs/ptv3-shared-wam-v3/nine-source-motion-recipe-20261015-r1`，TRAIN4,806,030/DEV821,613。原三源4096/source统计抽样保持，扣除114个TRAIN窗的精确贡献后重新计算配比和24-horizon hand/scene mean/std；DEV不参与拟合。114窗独立reader重算并回加统计误差≤7.28e-12；已知row14909移除59/67/75、保留3..51。

规则针对高冲量、共同手物强冲量、最初两个转换已高速飞行、极端转速的保守疑点隔离；不宣称排除了所有物理异常。纯旋转接触与正常掉落要继续视觉复核。详见[清洗记录](../../3d-data-preparation/docs/research/2026-10-15-unidex-motion-cleaning.md)。原清洗view、raw数据和本地旧训练全部保留。

远端验收已完成：`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v3-nine-source-20261015-r1/data-motion-r1/recipe`。仅传约3.24MB元数据，复用37,591 payload/2,638 CAD；九源36 TRAIN/八源32 DEV真实mixed-reader通过，数值统计与本地完全一致，14909窗口准入和原始坏窗仍可读另检通过。以上为启动前CPU验收；证据`tmp/nine-remote-20261015-r1/motion-r1-evidence/`。

Viser已切换同8769新清洗recipe，PID1686665、固定代码494c2111；10条整段隔离仍可选择回看，真实UI检查通过，HTTP200。新标记目录`nine-source-motion-review-20261015-r1/annotations`；旧标记保留。

以下为先前运行与部署记录；当前状态以上述清洗版本为准。

# 九源统一预训练状态

用户已要求第一轮合并九源并尝试大规模训练。九源 TRAIN 共4,806,144窗，
统计 `nine-source-recipe-20261015-r1` 已完成；HOcap、H2O TEST 和encoder越界窗排除。

已通过5项合同测试、九源mixed collate审计及真实双卡满batch启动检查。
`startup-nine-ddp-20261015-r1` GPU1/3、每卡128/global256，完整模型98,591,142参数，
4次更新、step1/step4模型与optimizer和rank RNG保存成功。
最终峰值分配12.96GiB，reserved17.62GiB（rank0），loss/grad有限。

长训 `pretrain-nine-80k-20261015-r1` 从step4完整恢复，目标80000步，
固定代码67f7af78，独立工作树 `tmp/nine-source-code-20261015-r2`，supervisor1584518。
GPU1/3，与已有GPU4/5/6/7训练合计6卡。48小时/8GB预算，可用磁盘下限20GiB。
已确认step4→60持续更新，稳态0.918–0.933步/秒，约235–239窗口/秒；
预计80k约24小时。两卡利用率100%，rank0峰值分配13.64GiB，reserved21.15GiB，
物理显存约21.59GiB/15.58GiB，无OOM/非有限量。预算内继续，不形成模型效果或机器人收益结论。

用户追加远端九源数据准备，已并行检查远端已有数据及可迁移闭包；本地长训不等待传输。

详见 [实验卡](experiments/probes/P-20261015-nine-source-pretrain.md)。

质量观察：step140出现由少量大位移标签解释的孤立scene loss峰值。CPU回放已定位
UniDex未来24帧最大7.387m；后续loss恢复通常范围。暂未修改冻结数据或loss，
不把大位移直接视为坏标签，待固定DEV读出决定是否追加质量Probe。

## 当前：本地已停止，转入质量审阅和远端准备

用户要求停止本地v3。GPU1/3已释放，最后记录step440；未到1000步首次固定DEV，
模型有效性仍UNCLEAR。已有完整恢复点保留至启动step4；本地日志和停止回执保留。
运行同期资源guard因其他GPU任务出现而终止own进程组，原始FAILED回执不改写；
这不构成方法负结果。本地不再恢复；远端数据传输与CPU准入检查继续，后续训练使用四卡。

已加入九源viser完整序列审阅脚本，基于旧多域viewer的源数据读取/回放方式，
新增六源canonical/compact适配、固定检查帧坐标、速度/加速度、人工质量标记。
默认异常定位source4 row14909 tick75。精确raw独立复核支持原仿真高速飞离，
尚不能定位具体动力学/穿透校正原因。

远端四卡准备：已提供显式world4配置`pretrain_nine_remote4_80k.json`（每卡128/global512），
launcher允许1–4卡，保持单机合计6卡保护；只提升自己的soft nofile至hard内最多65536。
训练代码部署短路径以满足loader Unix socket108字节限制。当前没有启动远端GPU训练。

## Viser已部署

`http://127.0.0.1:8769`（转发8769），PID1622883，固定代码cfc45d24，
工作树`tmp/nine-source-viewer-code-20261015-r1`，默认row14909/frame75。
脚本`tools/audit/review_nine_sources.py`。覆盖全量39,528条TRAIN/DEV完整序列，
4,806,144个TRAIN窗/821,635个DEV窗，不是抽样面板。
九项Task测试通过；九源真实Viser控件切换、空DEV、坐标、播放、标记通过；
HTTP200、版本匹配的WebSocket下发九源控件与曲线并接受交互事件通过。
服务CPU运行、最长24小时；标记保存在独立`nine-source-review-20261015-r1/annotations/`。
