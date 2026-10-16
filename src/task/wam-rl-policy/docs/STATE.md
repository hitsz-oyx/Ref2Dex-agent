# 当前状态

## 2026-10-16 critic资格诊断与v3直接特征（最新）

远端small/large/free/hold四组200episode已完成，没有方向稳定的10点保持增益。
独立检查发现free最终critic在2048个state上输出逐值相同、深层Tanh近100%饱和；hold亦明显饱和。
各组确实有约1.9万–2.8万次actor更新，不是未训练；full-action确定性偏离BC始终<.14，未真正挑战旧cap。
因此这些负结果不用于关闭大网络、自由动作、保持奖励或WAM路线。当前结论UNCLEAR。
下一项最便宜Decision：固定已有free TRAIN轨迹，GPU做Tanh/ELU × zero/remaining clock四臂
fresh critic MC-return回归，同540D/初值/1000步；按物体留出，只判断价值学习资格。已完成：留出MSE依次830/572/833/435；时钟PROMISING，尚非行为收益。

远端四卡现在实际运行v3-none/v2-aligned/v3-aligned/v3-shuffled各200episode，公共MLP50k BC，
独立冻结WAM encoder。后续v2/shuffled分别连续8/10轮全部13段EV近零，且末两Tanh100%饱和，现保留产物并用owned STOP停止；
不能以它们否定WAM。none/aligned仍有学习信号，继续200。释放GPU1/3做四臂25轮在线价值资格对照。
v3物理预测新五域约35%改善尚未证明机器人收益；Shadow388DEV与v3源4 TRAIN对象无交集，
但参与过预训练DEV，不称独立TEST。本地WAM PPO继续至200；100轮core held80→75/256，
lift124→143、到达117→130、终点101→116；2条实际初态不一致，只作描述性读数。

## 2026-10-16 扩张配方与直接WAM特征（进行中）

远端四卡分别运行small、large、full-action、full-action+hold四臂，每16步5epoch×4mini，
各32TRAIN/128env/200episode；先smoke审计再fresh训练。原full199和chunk16对照已完成，
actor800→10400步仍无稳定大收益（held202→197/210/768），不据此评价WAM有效性。
本地WAM主训继续；50轮同初态审计PASS，held80→83/256、lift124→148、到达117→139，
终点101→114；尚无明显持续保持改善。26个BC-only保持病例退步、29个PPO-only进步，不忽略退步。

用户要求WAM z和新v3后：新增独立冻结特征源，公共MLP BC不改，拟四臂v3-none/aligned/shuffled
及v2-aligned，同2631D(完整raw300+WAM2304+goal3+base24)、同大头/full Gaussian/hold/mini。
v3首选固定final80k，基于既有物理预测面板；不宣称预测改善已经转化为抓取收益。
完整组合GPU5两episode原生smoke已通过（26批、109actor步、maxKL.019892、重放3.05e-5、冻结PASS）。
当前实现还没有自定义失败terminal，不能冒称已补齐终止；下一项需alive/valid/逐env bootstrap
与全局有效样本加权的独立资格检查。所有阶段仍UNCLEAR。详见规模卡最新Decision Notes。

## 2026-10-16 用户授权大规模持续推进（进行中）

[规模Probe](experiments/probes/P-20261016-shadow-scale-continuous.md)：本地GPU1–4同步PPO已启动，
128个TRAIN物体、64env/rank、256个全局环境，目标200次更新/10188800环境步，最多12h。
保留50/100/200轮checkpoint；GPU5自动完成全部388DEV基线和固定64DEV阶段读出。
本地GPU0外部作业保护，6/7始终空闲。四卡实际显存约5.2GiB/卡，实测约95s/update，
初步预计总训练约5.3h；资源持续写入run的resources.jsonl。不能把启动训练当有效结果。

代码加入显式all-reduce梯度、全局advantage归一化/重放误差/KL、共同rollback，
两进程CPU合并batch等价测试通过；两卡原生2轮smoke的actor/critic跨rank逐值相同，
两卡全部597个冻结字段未变、无grad、原生action/force/torque误差0。
训练恢复包括actor/critic/Adam及各rank RNG，限定episode边界重新reset，非PhysX精确续跑。
WAM BC省略未被actor使用的forecast readout，批量H4输入各字段与旧路径测试一致。

并行单拍BC MLP为fresh current300→1024→1024→512→512→24；采用相同strict源窗口，
完整TRAIN1536897/DEV354761条标签，TRAIN-only归一化；50k步/batch2048约163s。
离线固定6208窗DEV动作MAE=.01939，root=.03302、finger=.01485，不能当闭环误差。
全部388DEV×4起点=1552条真实native rollout：official679/1552、final590/1552、
held60+lift10cm396/1552=25.52%。全DEV WAM9k为511/1552=32.93%，比MLP50k高7.41个百分点；九类实际初态全部逐值一致。
两基座训练历史/目标不同，不能将差值单独归因shared结构。
另同一32DEV×4起点面板，旧WAM9k与MLP20k均held37/128，final64/55、official70/61；
九类首帧状态逐值相同。不是MLP50k的配对全DEV比较，不能混合宣称改善。

本地WAM r1第32轮因rank03单env越出BC支持域中断，所有共同checkpoint为31；
已在r2显式采用原控制器的范围外零BC回退后恢复，非有限仍硬拒绝，记录fallback病例。
r2持续运行中，50/100/200核心DEV读出队列已切换r2；r1记录保留。

远端4×4090原生四卡smoke通过，128TRAIN的MLP PPO完成200轮：core held69→71/71/66
（50/100/200轮），没有明显收益。64TRAIN标准/强探索各两卡200轮完成，三reset合计
202→226/768和209→231/768，均约3个百分点；不足预定10点且有reset退步。
BC100k constant/cosine core held68/74，对照50k的69；离线误差下降不能当抓取改善。
目前转入539D归一化current vs 冻结BC512隐藏特征的等容量对照，见规模卡Decision Note。
远端有4/256环境在第一次策略动作之前初态不同；五个最小工程排查未定位，保留实际审计、
暂缓继续追查，小幅变化不作严格配对结论。不能混合本地/远端Torch版本宣称单变量效果。

关键产物：`outputs/wam-rl-policy/shadow-scale-panels-20261016-r1/`、
`shadow-scale-wam-128obj-4gpu-20261016-r1/`、`shadow-current-bc-seed241-20261016-r1/`、
`shadow-scale-mlp50k-alldev-20261016-r1/`、`shadow-scale-eval-queue-20261016-r1/`。
新实验状态仍为UNCLEAR；正式研究claim和learned RL-token效用尚未得到验证。

## 2026-10-16 用户转向Shadow：全量既有轨迹行为复查

只读复查三对reset全部48轨迹/策略，无新训练或仿真。逐物体曾达到held60计数BC/PPO：
瓶子TRAIN4/5、相机TRAIN6/7、手机DEV0/1、书DEV1/2，各12条。
两策略总抬升10cm均29/48、到过5cm目标均26/48；曾保持11→15，末段持续60帧直到
记录结束11→14。PPO仍15/24条DEV未抬到10cm，主要失败集中于DEV抓取/抬升。
相同起点曾保持的转换为both8、BC-only3、PPO-only7、neither30；不能只看净增4忽略退步。
相机233/env6虽曾保持114帧，终点抬升仍11.49cm但goal距离9.26cm，只能说明离开目标区，
不能据此声称掉落。全部仍为UNCLEAR，未新增正式RL效用结论。

新增离线HTML同步回放，覆盖所有48对起点，可切物体/seed/env、拖动旋转和逐帧查看；
显示真实手部11关键点与按记录object pose变换的同scale CAD表面点，不是仿真相机视频。
目录`outputs/wam-rl-policy/shadow-bc-ppo-20261016-r1/behavior-review-verified/`，
含`replay.html`、`objects.png`、`behavior.json`、`cases.csv`和`viewer-audit.json`。
生成code57151f23；48case×5帧的Node DOM/canvas-stub检查通过、坐标有限，已查看统计PNG；
没有真实浏览器视觉测试。纯文件统计/CAD变换在CPU执行，无神经推理，不占GPU。

## 先前Stage-0训练与评估

2026-10-16：Shadow BC→RL Stage-0完成，UNCLEAR。代码在本Task，路线来自用户ref1；
冻结Shadow9k BC/WAM，只训练新327D actor/critic。20×199×32=127360 env steps，
约19.2分钟；80个actor epoch均接受，jointKL<=.01762，latent logprob重放最大5.73e-6。
全部597个冻结state_dict字段逐值未变、无gradient；canonical/yaw/force/torque误差0，
手指目标最大1.2e-7。

三个同初态16env reset面板216/233/234，BC/PPO保持分别5/8、4/3、2/4；合计11/48→15/48，
DEV1/24→3/24，official sticky均26/48。方向不一致；同一BC本身还有9/16与6/16的数值
重复波动，不能据合计+4宣布有效。保持为60帧goal<=5cm且lift>=10cm位置代理，非接触认证。
一个训练seed、仅两TRAIN物体，不作正式结论。

完全同初始输入/冻结权重/fixed RNG，GPU reference重复推理仍有约6e-6–1.12e-5 command差；
zero same-packet精确等于BC，但独立闭环不能要求逐值相同。原完整轨迹门FAIL保留，
未定位具体算子，也未排除额外PhysX波动。独立审查已复算全部raw分split指标与初态。

下一步先诊断有效探索和reward/保持关系，检查饱和边界是否吞掉残差，再决定是否扩大
PPO预算；不要因合计增益直接放开WAM。learned RL-token、aligned/shuffled/none和独立
动作路径仍是后续路线，当前尚未实现/验证其效用。

产物`outputs/wam-rl-policy/shadow-bc-ppo-20261016-r1/`；模型`train/latest.pt`，
评估`evaluation-pairs.json`，汇总`summary/results.json/csv`与`comparison.png`。
当时checkpoint保存actor/critic/optimizer/RNG，CLI仅evaluate加载actor；本轮规模实现已补齐train-resume。
全部有界作业退出，未停止外部GPU0任务。协议与证据见
[P-20261016-shadow-bc-ppo](experiments/probes/P-20261016-shadow-bc-ppo.md)。
