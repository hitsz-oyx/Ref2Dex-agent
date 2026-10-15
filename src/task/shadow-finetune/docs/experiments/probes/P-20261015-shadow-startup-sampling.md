---
schema: ref2dex.probe.v2
probe_id: P-20261015-shadow-startup-sampling
experiment_id: P-20261015-shadow-startup-sampling
date: 2026-10-15
task: shadow-finetune
branch: main
git_commit: fc441a2b
claim_id: C1
hypothesis_family: HF-shadow-startup-sampling
probe_index_in_family: 1
seed_pool: probe
seeds: [214, 215, 216, 218]
decision_changed_if_positive: prioritize startup coverage and paired native qualification before longer data scaling
decision_changed_if_negative: inspect actor capacity and optimization before adding data or tail supervision
status: RUNNING
run_id: shadow-startup-sampling-20261015-r1
---

# Shadow起步采样：同一全量数据的最小干预

Class: Decision，服务独立执行底座。数据量对照中full1000步的forecast变好但保持0/16；
已见small41个起点首拍MAE仍0.223、torque0.674，原采样tick0只占0.67%。
标签全量重放PASS；teacher状态条件下近目标/安全尾段误差远小于起步。
检验“起步曝光不足是当前动作拟合的可干预因素”，不假定它是全部闭环失败原因。

## 固定干预和最便宜对照

复用`shadow-scale-controls-20261015-r2/train-full/step1000.pt`作为uniform终点对照，
新增一个从同remote80k父模型/fresh actor初始化的1000步candidate，不从uniform权重续训。
仅改变TRAIN sampler：50%概率选有准入tick0..7的序列，再在该序列的早期准入query均匀选；
其余50%沿用原“序列均匀→全部准入query均匀”，不新准入任何窗口或修改labels。
0%分支与原draw及最终RNG逐值相同；旧数据函数/方法AST和forecast/model/execution源码必须相同。
新运行与旧control有代码身份差异（只新增sampler/记录/审计接口），不冒充同一提交或新seed复现。

同TRAIN1567物体/10118轨迹/1536897窗、相同小TRAIN统计、同seed214、batch64、
AdamW/三个LR/forecast+8拍action权重、同父step80k、原DEV64/seed215和native16env/seed216。
无GTτ warmup，不读teacher actor参数初始化学生。实际51个初始actor张量必须完全相同。
每250步保存完整模型/optimizer；只用1000步终点闭环，不挑有利阶段。

新增共享起步DEV64面板：从原DEV的准入tick0中seed218抽64条，固定raw/env/tick身份，
两个终点都重新推理该面板；不用于normalizer/训练或checkpoint选择。原随机DEV面板保持不变。
保存raw/clip首拍force/torque/finger误差；原随机DEV须复现保存的action<=1e-5、forecast<=1e-4。
采样重放最终RNG、真实startup/tick0计数必须与新checkpoint/manifest一致。
闭环核对全200状态teacher trace、teacher/student初态、命令时钟/范围和三种成功代理。

## 预设读出与停止

拟合PROMISING：startup DEV首拍MAE和torque MAE均下降>=30%，随机DEV首拍MAE不恶化>20%。
整体局部PROMISING还要求相同16起点同步抬升保持60帧增加至少2/16、无invalid；否则UNCLEAR。
若拟合改善而保持没有改善，转向早期闭环误差传播/分布移位，不把拟合当抓取资格。
若起步拟合仍差，先检验actor拟合/容量，不继续无界增训或优先加尾段BC。
单seed/小面板不能证实因果或归因Inspire；复用control的CUDA浮点与时间顺序局限保留。

## Decision Note / 资源

用户持续授权临时一张空卡做小规模Shadow测试，原Inspire三臂GPU0–5继续。
本轮GPU6单个1000步候选+16env+终点诊断，预计15–20min，总<=1h/8GB，GPU7空闲。
复用已有uniform避免重复1000步；不延长既有训练，不修改当前Inspire或远端输入。
固定提交/detached工作树、父checkpoint/manifest/panel size+mtime，20GiB空闲底线。
非有限、超时、预算、代码或输入漂移、外来GPU6进程或GPU7被占时只终止自己的进程组。
启动后记录真实GPU利用率/显存/ETA。完成后释放GPU6，不自动扩大预算。

## Limitations / future evidence

本轮只检验当前模型与固定训练预算下的采样干预；充分收敛、独立seed、更广物体起点、
接触稳定性、300D teacher-only学生与actor容量对照均延后到能改变下一步决策时。


## 实际启动 / 2026-10-15 19:06 CST

实现fc441a2b，detached工作树`tmp/shadow-startup-code-20261015-r1`；PointWorld/PTv3复用原
uniform运行相同只读源。真实准备通过：原model/forecast/execution源码逐文件相同，原data
函数/已有方法AST相同；共享controls、旧输入size/mtime和起步DEV64物理身份已固定。
10项Shadow合同测试和工作流验证PASS；supervisor2161329，GPU6，GPU7预留。
输出`outputs/shadow-finetune/shadow-startup-sampling-20261015-r1`。
当前只完成启动，不作拟合或抓取收益判断；进程与后续终点以process.json和summary为准。

启动后实际51个actor初始张量与uniform对照逐值相同；父权重/统计/配置/参数量/DEV IDs一致。
已完成160/1000，0.81秒/更新，预计余约11分钟训练，native+readout另计。
实际首10240样本中tick0..7为5343（52.18%），tick0为652（6.37%），符合50%混合加原采样早期支持。
35秒稳态GPU6平均61.1%/物理最大3701MiB，峰值allocated约3046MiB；GPU7全程0%/2MiB。
吞吐与原full约0.8秒/更新一致，scattered CPU加载间隙使GPU利用率波动；不为利用率改matched batch。
`startup-control-audit.json`、`startup-observations.json`、两段GPU采样均在run目录。
当前loss有限，仅工程与干预进入检查，不提前判定拟合或闭环收益。
