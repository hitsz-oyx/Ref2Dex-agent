---
schema: ref2dex.probe.v2
probe_id: P-20261014-unidexgrasp-local-admission
experiment_id: P-20261014-unidexgrasp-local-admission
date: 2026-10-14
task: video-data-preparation
branch: main
git_commit: 6141c34e592085f1f102205eab45dd643429e0ae
claim_id: C1
hypothesis_family: HF-unidexgrasp-rigid-admission
probe_index_in_family: 1
seed_pool: probe
seeds: [123]
decision_changed_if_positive: use native rigid geometry and explicit ShadowHand FK, then assess rollout acquisition
decision_changed_if_negative: resolve asset scale, pose or kinematic semantics before expansion
status: PROMISING
run_id: unidexgrasp-multi-replay-20261014-r2
---

# UniDexGrasp / ++ 刚体样例接入检查

## 续接会话后的当前状态与最小决策（2026-10-14）

20对象/320条有效回放已经完成，实际按需provider得到TRAIN2560/dev640窗口。
3200窗实际字段、同ID endpoint、20对象future intervention及spawn双worker检查已通过，
`provider-audit.json`耗时23.21秒；不是模型前向/反向或完整训练准入。
用户确认相机/碗两张预览手物位置正常，相机留在手边、碗拿起后脱落；只覆盖两份抽样，
不能推广为全部轨迹已审。

当前决策：完整CAD/ShadowHand数据源是否已有跨对象、持续接近目标的执行覆盖，值得
下一轮模型工程接线与扩对象。最便宜方法是只读现有320条原生30Hz轨迹，补报末0.5秒
每个记录状态均在5cm目标范围内的数量，区分累计命中、原始末帧距离和持续接近。
它只表示距离，不证明接触/稳定抓持，也不作为隐式过滤规则。若覆盖过少，先排查对象/初始化；
若覆盖够用，下一步真实Shared-WAM工程前向/反向，保留成功与失败供物理预测使用。

成本：CPU字段/统计与spawn回归≤2分钟，新报告≤2MiB；CPU用于文件、解析几何和统计，
无训练/神经网络推理。报告`provider-audit-r2.json`写新路径，不覆盖之前证据。
现场核验远端GPU0/2已运行v2训练，远端1/3旧重建与本地5/7训练合计六卡；
真实GPU smoke因此延后，不改变或停止既有任务，不启动新的回放/下载。本轮代码沿用main。

## 继续处理：20对象官方初始化与批回放 Decision Note

用户明确要求继续处理UniDexGrasp。沿用已验证的确定性++ state actor、原物理配置和
ShadowHand/CAD profile；本轮区分“单Car偶然可用”与“少量不同刚体可稳定批量采集”。
若多对象的真实连续执行、标签合同及成功示范覆盖可用，再扩大对象队列；若缺资产或初始化
异常，先修对应工程链路，不把失败实现当作策略负结果。

最便宜入口是作者提供的140,597,661字节 `datasetv4.1_posedata.npy`，加441,222,940字节
scaled mesh与26,844,762字节feature，共608,665,363字节；无需先下载2.72GiB静态标签包。
公共下载先检查精确HTTP Range、文件长度与来源身份；压缩输入缓存最多750MiB，解压仅选中
对象，prepared assets≤128MiB，扫描mesh总展开≤8GiB。官方TRAIN按排序取不同类别首个对象
共20个，预先固定每第五个mesh为dev（16/4），TEST不采集，不根据回放结果挑选split。

输入准备 run_id=`unidexgrasp-policy-assets-20261014-r1`；下载/解包CPU最多30分钟，6条并发
HTTP连接，可按完整range恢复。GPU回放run另记，最多20对象×8env×2轮=320条，新增GPU预算
10分钟、单次hard watchdog≤10分钟、输出/extension≤1GiB；优先空闲本地GPU4，不使用已为
用户训练腾出的远端GPU0/2。启动前复核共享六卡上限；若远端双卡训练已经启动且本地5/7
仍在运行，则先完成CPU准备并等待项目卡释放，不抢占现有任务。

成功/失败与实际timestamp、root/q/dq/action都保留；compact trajectory按需切H4/K24，不批量
保存重叠窗口/视频。完整CAD/current profile和11点映射仍显式登记；本轮产物在完成provider、
多对象/dev及实际视觉检查前保持`training_allowed=false`。同一资产路径供后续worker复用，
避免按run重复VHACD；只允许本任务新asset产生的小缓存，不修改HOME/其他任务cache。

### 20对象实际结果与provider续接审计

- 准备资产已完成，20不同mesh各选一个scale，所有对象都来自官方TRAIN且不在官方TEST
  mesh集合中。16/4对象划分在回放前固定，不按成功挑split。本次包含名为pillow/BeanBag等
  资产，但仿真里全部是刚体，不是软体数据。
- r1保留作诊断：上游按最后一个物体的碰撞形状数分配所有aggregate，复杂物体出现容量
  警告。采集器按批次最大实际碰撞容量修复后r2重跑；未调整policy、碰撞mesh或物理参数。
  有效运行collector提交`6141c34e`，upstream `6c9e710b`；r1不作为方法负证据。
- r2为160env×2轮，320条×200个60Hz状态。采集19.7756秒，含初始化45.2910秒，GPU4
  观察峰值2713MiB；无aggregate容量警告。GPU已释放，不在本次续接重新采集。
- 原始末帧官方累计距离命中119/320，末帧位于目标5cm内100/320；它们均不是稳定抓持率。
  新报告将原始200/60s末帧与compact199/60s末采样帧显式分开，前者100、后者99，
  不是统计重算冲突。末0.5秒持续距离仅使用真实30Hz状态，不插值、不看单次命中替代保持。
  后续读取实际timestamp修正文档的绝对时间：原始索引199/compact索引198均须计入
  初始化warm step，首状态为1/60s。报告一直读取真实时间，数据和上述计数不受影响。

| 距离描述（不等同抓持成功） | TRAIN256条 | dev64条 | 合计320条 |
| --- | ---: | ---: | ---: |
| 官方累计命中 | 95 | 24 | 119 |
| 官方原始末帧在5cm内 | 77 | 23 | 100 |
| 30Hz末采样帧在5cm内 | 76 | 23 | 99 |
| 30Hz末0.5秒每个采样状态均在5cm内 | 67 | 16 | 83 |
| 曾官方命中但30Hz末采样已在范围外 | 19 | 1 | 20 |

持续距离覆盖TRAIN13/16个对象、dev3/4个对象，足以保留本方向并安排下一次模型工程
接线；不自动批准成功示范，不为其余对象增加尝试次数。本数据源的U/V物理监督保留
成功和失败，二者都是真实执行；示范/控制用途须另定义质量，不把ShadowHand动作当Inspire。

provider提交`0576812f`，原审计提交`4e4b2b00`。CAD和normal按对象共享，320条compact
含30Hz手点、物体、q/dq、根状态和60Hz动作/实际命令，共15,721,464字节（约15MiB、
47.98KiB/条）；窗口不展开存储。scene18D当前normal/H4motion、query32及手轨迹均从
同执行构造，完整CAD明确包含背面。source_id4为临时几何profile，不生成控制batch。

全部3200实际窗口再次通过：TRAIN2560/dev640，20对象future intervention不改变当前
场景/手输入，endpoint与U末帧同点，缺失左手无标签；spawn双worker进入既有
`forecasting_collate`。新审计25.75秒，全相关回归76项通过，日志在
`tmp/unidexgrasp-continue-20261014-r1/regression.log`。

结果目录`outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2/`包含
`trajectory-manifest.json`、`trajectories/`、`geometry/`、原`provider-audit.json`、
新`provider-audit-r2.json`和具名抽样`user-visual-confirmation.json`。旧报告不覆盖。
修复审计spawn IPC使用项目tmp；统计与图片只在CPU，未运行神经网络。

限制与下一步：当前provider一mesh一scale、非空mesh-disjoint dev已具备；还未完成
真实GPU模型前向/反向、完整source准入、跨scale索引、混域normalization/sampler或收益
验证。下一张项目卡释放后先跑四步双头工程smoke，成功再考虑100对象有限回放及source
准入；扩量按mesh/scale与有限初始化次数，不做无界“补到成功”。六卡满额时不后台自动抢卡。

Result: PROMISING（刚体几何接入Probe）。20对象320条回放产生TRAIN2560/dev640窗；全部3200窗合同与未来隔离通过，末0.5秒持续接近目标83条；真实GPU模型接线及正式训练准入待验证。
Decision: 保留完整CAD/ShadowHand按需provider和成功/失败物理数据；六卡满额时不另启GPU，释放后先做真实Shared-WAM工程前向/反向，再扩对象；不将距离指标升级为稳定抓持或全库结论。

Class: Decision. 服务于Shared-WAM刚体手物监督来源的选择。
区分“原生刚体mesh/抓取标签可便宜解析”与“格式/尺度/手FK仍是主要blocker”，同时确认
发布文件是否提供真实连续手物轨迹。最便宜方法是一个官方示例物体、四个静态抓取，
读取原始数据与尺度约定，生成真实FK/mesh预览；静态标签不能通过复制或插值伪造future24。

## Decision Note / execution budget

- 用户明确要求先试UniDexGrasp和++；沿用main，不建分支，不修改既有训练和外部资产。
- 官方原仓库自带Car样例；使用固定上游提交`36c9bfcf7cedc987dd15e2384c548e51d5aadc21`。
  ++源码另在`third_party/UniDexGrasp2`，仅核验它复用的数据与策略接口。
- 有界获取≤64MiB必要样例/源码，总下载≤10分钟，每次网络请求≤30秒；
  本轮审计输出≤32MiB，静态FK/绘图CPU≤2分钟，无GPU训练/仿真。
  CPU用于少量参数几何与文件审计；如果需要策略推理/仿真，重新检查GPU预算，另记录运行。
- 输出`outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1/`，
  scratch `tmp/unidexgrasp-source-audit-20261014/`。两个第三方仓库先以blob:none/no-checkout克隆，
  不自动获取大模型或全量数据；当前GitHub API限流，改用固定SHA的raw文件。
- 格式/坐标不明、手与物体尺度严重不符、下载超限即停止扩量；保留失败记录。

## Limitations / future evidence

不以一个物体验证大规模质量/训练收益。已发布静态抓取、目标姿态和模型checkpoint与连续
rollout是不同产物；如无时钟和序列，本轮不生成完整Shared-WAM训练样本。
ShadowHand与当前Inspire的控制action不同，语义手点映射不意味着能混用Chunk-R action标签。
完整轨迹需实际策略仿真、记录hand/object/action/时间，再按30Hz构造H4+future24。

## 实际输入与CPU几何检查

- 使用官方原仓库固定SHA的小样例：一个Car、四个NPZ抓取、五种已缩放mesh/URDF和MJCF，
  必要输入/源码18文件573,016字节（约0.55MiB）；首次raw请求遇到SSL EOF，保留文件后有界重试成功。
  下载manifest及一次性fetch脚本在声明的output/scratch内。没有下载全量tar或修改外部模型。
- 每个抓取文件只有`qpos`（28个命名参数）、`scale`、`plane`；没有time、frame、action序列。
  四个文件是独立抓取而非相邻帧。实际尺度倒数12.5/8.333334/12.5/6.666667，对应
  .08/.12/.08/.15，URDF mesh scale=1；不再二次缩放。物体最长尺寸约15.0/22.5/15.0/28.1cm。
- 原始wrist-free MJCF包含22个finger DOF；保留body/joint/inertia/site原参数，去除仅供视觉/
  actuator用的asset/default引用，用已安装MuJoCo做静态FK（无物理step、无GPU）。
  手root的`sxyz`/`xyz`Euler与translation作用于同一物体坐标系，没有拟合额外对齐。
- 沿用11槽角色，但landmark显式为palm origin、thhub、四个knuckle和五个XML fingertip site；
  这是ShadowHand profile，不宣称等同人类解剖或Inspire link原点。XML tip offset与policy中
  20mm distal offset不同，后续不能悄悄混用。左手全invalid。
- 四个FK均有限，重建指尖到物体mesh的unsigned距离约5.5–12.8mm，只是几何接近关系，
  不能据此证明表面无穿透、真实接触或抓取稳定。三个抓取无关节越界；00003的THJ0为
  +0.001498rad，超原范围上限0，保留原值与告警，不自动clamp再声称原样验证通过。
- CPU audit1.22秒，生成`static-grasps.npz`与`review-static-grasps.jpg`；静态profile不含H4/future24，
  `training_allowed=false`，完整动态窗口数0。初步目视没有明显整体尺度错位，未请求用户批准
  抓取稳定性或将该图升级为数据质量Validation。

## ++策略回放准备与当前运行缺口

按用户提出的policy回放方向，另获取官方++固定SHA
`6c9e710b04769bf5d9765e3233820101cb3d718f`的state checkpoint（17,220,395字节）
及最小loader代码、五份官方`pc_feat_*.npy`，总输入仍小于64MiB。CPU仅反序列化/严格加载
参数及查验feature，不运行神经网络推理；GPU使用0。

- checkpoint为有限tensor的state dict，state actor为300→1024→1024→512→512→24，hidden ELU，
  actor参数严格加载成功，五份形状编码均64维/有限。没有用零向量代替官方形状编码。
- 应采用++的状态观察合同：236:300为0.1×官方pc_feat；其target相关207:236置零。
  保留官方obs/action clamp及24D动作解释（base力/力矩与18个actuator目标），不能当成22DOF角度序列。
  原项目goal-conditioned观察与++状态checkpoint不可混搭。详细来源见[研究笔记](../../research/2026-10-14-unidexgrasp-admission.md)。
- 只读检查远端现有pointwam-expansion环境为Python3.10且没有isaacgym、gym、transforms3d。
  本地graspenv3.8已有这些组件，但本次检查时本地六张GPU已有任务，未额外分配或停止进程。
  现有Isaac Gym native bindings为Python3.6–3.8，不能在远端3.10环境里仅加PYTHONPATH冒充兼容。
  尚未新建环境、安装包或运行policy，checkpoint结构加载不证明回放成功。

证据：`download-manifest.json`、`audit-summary.json`、`checkpoint-audit.json`、
`plus/manifest.json`、`remote-runtime-preflight.json`。下一最小采集为一个物体、少量env、短rollout，
用兼容环境与实际GPU资源另记录run_id/预算；必须采到实际timestamp、q/root、物体SE(3)、
action及reset边界后才能构造30Hz窗口。future query用当前选定的mesh局部点经真实object pose传播，
不得跨reset拼接，也不得用最终抓取姿态插值替代执行轨迹。

## 本地GPU短回放 Decision Note（run r2，用户授权）

- 用户要求在本地实际回放；服务同一数据来源决策：公开++ checkpoint是否能便宜产生真实
  连续手物轨迹，以及H4/future24是否能由同一执行切窗。
- 最新进程归属核验：本地GPU0–3为其他用户，本项目GPU5/7运行既有训练；远端EgoDex旧重建
  使用GPU1/2/3。因此本项目当前共5卡，使用空闲本地GPU4后为6卡，符合共享上限。
  上次仅按本地活动GPU数量判断预算已满过于保守；不停止或共享任何现有GPU进程。
- root选择单Car、scale .08、8个env、至多2个200-step episode的确定性actor推理，seed123；
  禁止训练/梯度更新，沿用原物理与300D观察，关闭random_time以完整采集从reset起步的轨迹。
  成功/失败均保留并分开标注；不由单物体成功率形成正式方法结论。
- 运行`unidexgrasp-local-replay-20261014-r2`：准备下载/源码≤128MiB、scratch/extension≤1GiB，
  输出≤128MiB；GPU4一次最多5分钟，含编译/初始化的外部owned watchdog最多10分钟。
  初始化失败先修明确兼容问题；物理异常、非有限tensor、观察不符、占用冲突或预算超限即停。
- 输出`outputs/video-data-preparation/unidexgrasp-local-replay-20261014-r2/`；
  scratch `tmp/unidexgrasp-local-replay-20261014-r2/`。复用本地已有Isaac Gym/graspenv，只读外部
  安装；源码/资产放本项目，extension/cache不写入外部项目。执行提交写launch manifest。
- 运行后按真实sim.dt每2步抽30Hz，审计不跨reset、future点身份与米制一致，并生成可视化。
  scene先用明确完整CAD geometry profile作工程检查，不冒充视频可见点；产物先禁止训练准入。
  不需要新分支、权限突破或停止现有进程。

r2初始化31.25秒/2695MiB观察峰值，GPU PhysX与原18-actuator手加载成功，但采集器查找
`robot0:wrist`失败：官方replay资产的wrist已注释，实际palm作为根；不是policy有效负结果。
改为明确palm origin，与前次静态profile一致，保留r2错误记录。Gym另生成本任务的13,808字节
VHACD默认用户缓存；当前API未发现重定向入口，不修改HOME、不替换碰撞几何。
r3复用r2准备好的资产路径和这份既有缓存，只读核对posedata，避免按run复制碰撞cache；
extension/日志仍在新scope内，不修改其他缓存。后续扩物体需重新明确缓存边界。

r3在初始化暴露上游`../assets`硬编码：仅设置assetRoot不覆盖posedata读取。修复collector工作
目录与已准备资产的相对关系后在r4重新执行，旧运行保留。r2/r3均未采到policy执行轨迹，
不作为策略负结果；后续不为路径兼容改变物理或模型观察。

r4采到两轮×8env×200frame，CUDA推理/同执行hand-object-action均有限，正式成功flag为0/16；
但关键链路复核发现collector漏读官方cfg.sim（solver iterations/contact offset/substeps等）。
这是实现缺陷，r4保留作采集工程证据，不能作为原policy负结果。按官方parse_sim_config
修复后以同seed/对象/scale在r5重跑，不调整checkpoint、成功门或初始化。

## 修复后的本地真实回放结果（r7）

执行代码`11b103f8`，上游++代码/资产固定`6c9e710b`，checkpoint沿用原官方文件。
r5/r6仅GPU预检拒绝，没有child或推理；随后只读查询空闲并记录raw预检后，r7在GPU4执行。
不将短暂显存读数推断为其他任务，也没有停止未知进程。

- 8env×2轮，每条200个真实60Hz状态帧（初帧是官方reset的零动作warm step之后），
  每轮199个policy动作及实际施加root force/torque/DOF target；不训练、不更新checkpoint。
- 官方cfg.sim已应用：2substeps、8position iterations、contact_offset .002、原其余物理参数。
  当前300D观察、64D shape feature、动作clamp、++ target置零均按原实现；snapshot与控制来自同一执行。
- 官方goal-distance成功累计13/16（第一轮6/8、第二轮7/8）；独立核对末帧也有13条处于
  5cm目标邻域，成功条末端物体z约.88–.93m。官方flag只表示目标接近，不能独立证明稳定抓持。
  单物体、单seed不构成论文成功率复现或全库性能估计。
- 策略/采集12.45秒；supervisor含编译/初始化36.27秒，观察显存峰值2707MiB。
  r2/r3/r4/r7外部运行总计130.01秒，实际GPU使用不超过该总墙钟，仍小于5分钟。
  GPU4运行结束已空闲；本轮没有改变本地训练或远端EgoDex进程。
- 全部q、hand、object、observation有限。terminal reset仅raw第199帧出现；每2帧选取真实30Hz
  得每条100帧，不插值。stride8扫描160个手物候选窗口，按预设tick27导出16份核心合同fixture，
  包含成功/失败，训练准入0。hand未来`[24,2,11,3]`、scene未来`[1,24,32,3]`；左手全invalid。
- 32个query来自同一已缩放CAD的固定局部点，由实际object SE(3)传播；history/future共用
  当前物体anchor，goal对应同一query的t+24。未来扰动检查：只改t之后的手/物体，所有当前
  输入字段逐值不变，U/V/G标签变化。未伪造相机、mask、置信度或Inspire控制量。
- 目前只导出核心手/物体窗口，没有生成含scene18D、source注册、分片、归一化和非空dev的
  正式Shared-WAM corpus。完整CAD包含背面，是显式仿真完整几何profile，不能当成当前可见表面。

最终输出：`outputs/video-data-preparation/unidexgrasp-local-replay-20261014-r7/`（约8.37MiB含extension），
包含`episode-00/01.npz`、`result.json`、`simulation-config.json`、`audit-summary.json`、
`causality-audit.json`、`run-accounting.json`、`windows/*.npz`。
audit工具实际提交`6e3afc8f`，CPU几何检查/绘图11.65秒。成功/失败各一份
`replay-success/failure.mp4`与`review-success/failure.jpg`，是记录状态的三维重建而非相机RGB。

剩余决策：少量不同mesh的成功示范覆盖与对象分组dev；准确landmark/profile注册；scene采用
完整CAD还是当前模拟depth可见观测；在同一执行上建立训练provider再做共享模型工程检查。
当前不直接全库下载/训练，不将r4错误物理的轨迹混入有效方法证据。
