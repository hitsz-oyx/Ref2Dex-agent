---
schema: ref2dex.probe.v2
probe_id: P-20261015-shadow-data-scale
experiment_id: P-20261015-shadow-data-scale
date: 2026-10-15
task: shadow-finetune
branch: main
git_commit: db3d9d13
claim_id: C1
hypothesis_family: HF-shadow-data-scale
probe_index_in_family: 1
seed_pool: probe
seeds: [214, 215, 216]
decision_changed_if_positive: prioritize larger native Shadow dataset and broader paired success panel
decision_changed_if_negative: inspect fitting and closed-loop distribution shift before longer scaling
status: PREPARED
run_id: shadow-scale-controls-20261015-r1
---

# ShadowHand：固定计算预算的小数据/全量数据对照准备

Class: Decision。用户要先在ShadowHand自身环境微调并看成功率，以检验现有DExplore数据覆盖
不足的猜想。该Task服务独立策略资格，Shadow收益不能自动成为Inspire失败的数据量归因。
原pilot只用7TRAIN物体/41轨迹，学生TRAIN1/8、DEV0/8，不能检验数据量假设。
全量候选已通过12456条数据合同审计，故现在准备最小fixed-compute对照，隔离TRAIN支持集合。

## 设计与判别

- small：pilot cache的7TRAIN物体、41轨迹、6232窗；full：1567物体、10118轨迹、1536897窗。
  small保存状态、标签与完整tick集合必须是full的物理同一子集，按raw episode/env匹配，
  不能因metadata行顺序不同误配。两者均使用相同终点5cm/安全窗准入，不使用DEV误差选训练。
- 同80k父权重、fresh Shadow query actor、seed214、AdamW/各模块LR/forecast权重不变；
  两臂各1000更新、batch64。采样均为序列均匀，再在序列内均匀，故full不保证一个epoch或充分拟合。
- 两臂state/action mean/std均仅由small TRAIN拟合并冻结，避免normalizer改变成为额外干预，
  small不读取full TRAIN或DEV标签拟合统计。预训练WAM numerical normalizer保持父权重值。
- 两臂使用同一个full DEV reader、seed215预先抽取的64窗，保存索引和raw/env/tick身份；
  不重新抽各自面板。TRAIN/DEV按物体分开；不把小集原DEV64与这里的新DEV面板直接比较。
- 同16env固定2TRAIN+2DEV对象面板、seed216，teacher/student完整初始状态配对。
  闭环主读出用相同1000步终点，分别报告官方success、末帧5cm和同步抬升保持代理，
  TRAIN/DEV分开；原pilot seed213和step500结果不能作为该对照的一臂。
- 保存每250步权重、best action和final；离线全DEV面板动作L1/root/finger/首命令MAE及forecast。

最便宜先完成共享controls和真实数据合同核验；GPU执行按实际空闲和预算另行启动。
局部PROMISING信号：full终点相对small的共同DEV首命令MAE降低>=20%，且DEV闭环多成功
至少2/8个起点；这只是后续扩大面板的触发条件，不能当正式成功率结论。否则UNCLEAR，
先看TRAIN/DEV拟合与分布移位，不能以full尚未充分学习为数据量路线负证据。

## 资源与当前状态

本轮只准备代码、controls和CPU元数据/小集统计，不启动GPU，不延长已完成的临时单卡测试。
常规资源仍须本地留2卡；原Inspire三臂GPU0–5继续、6/7空闲。将来的两臂粗估各10–30分钟，
但真实full I/O吞吐尚未测量，不视作已批准长训。每训练入口2400秒/12GB checkpoint/20GiB空闲
保护继续生效；native每worker600秒。不得为了对照暂停既有任务或自动占用保留卡。

输入manifest、父checkpoint、native panel保存size/mtime身份；启动拒绝settings/panel/input漂移。
`tools/run/prepare_scale_comparison.py`只生成controls和命令，`train.py --controls`才消费同一份配置。
6项Task合同测试涵盖物理同一子集与标签变动拒绝、同DEV/同归一化，以及设置/输入漂移拒绝。
当前尚无本实验GPU训练、checkpoint或成功率结果。

## Limitations / future evidence

固定计算预算衡量更大支持集合在同预算下的收益，不能证明完整数据的收敛性能。只一个seed、
64个可能相关的DEV窗口、2个DEV物体闭环，统计功效弱。更广闭环面板、多seed、充分拟合、
teacher300-only MLP对照、接触稳定性，以及与Inspire的差异归因都留待有判别信号后执行。
