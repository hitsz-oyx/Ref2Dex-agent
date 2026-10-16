# 当前状态

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
held60+lift10cm396/1552=25.52%。全DEV WAM BC基线仍在计算，不预判两者优劣。
另同一32DEV×4起点面板，旧WAM9k与MLP20k均held37/128，final64/55、official70/61；
九类首帧状态逐值相同。不是MLP50k的配对全DEV比较，不能混合宣称改善。

远端4×4090部署完成，128TRAIN/388DEV全部原生资产链接核对通过。首轮native smoke
在gymtorch尚未构建时失败，无训练结果；旧supervisor曾将快速全rank失败误写COMPLETED，
exit_codes全1为真实证据，已修复末尾退出码检查。单进程项目内新cache的gymtorch编译/加载
已通过，改用该cache且补齐conda/bin PATH，再做四卡真实native smoke。
不同Torch/CUDA版本须在各机器内部建立BC/RL配对，不作跨机器单变量比较。

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
