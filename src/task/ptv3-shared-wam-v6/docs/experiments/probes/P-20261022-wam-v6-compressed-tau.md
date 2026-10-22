---
schema: ref2dex.probe.v2
probe_id: P-20261022-wam-v6-compressed-tau
experiment_id: P-20261022-wam-v6-compressed-tau
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: PLANNED
claim_id: C1
hypothesis_family: HF-v6-runtime-trajectory-compression
probe_index_in_family: 1
seed_pool: probe
seeds: [to-be-assigned]
decision_changed_if_positive: replace dense runtime trajectory decoding with a fairly trained compact or latent policy interface when success and interaction efficiency are retained
decision_changed_if_negative: retain the full trajectory interface if its reproducible policy benefit outweighs decoding cost
status: PLANNED
run_id: not-run
---

# 压缩计划与完整 24 帧 τ：公平训练对照设计

**仅设计，未训练、未运行仿真、无成功率或效率数字。** 当前 HEAD
`fbdb82a74d3eec65026977243ccf97a669bac1bc` 仅为代码核查版本，不是本卡运行提交。
本卡不重复另一线程的可辨识性、离线误差或 checkpoint 诊断。

## 要区分的假设与决策

这是 Decision Probe：在同样预训练、机器人示教与交互预算下，完整时间展开的
24 帧计划是否带来压缩计划和预训练表示无法保留的抓取—抬升收益？

- 若压缩后策略成功率、样本效率等效且成本下降，采用压缩接口。
- 若只保留预训练表示也等效且更便宜，删掉推理期未来轨迹解码，只保留预训练表示。
- 若完整 τ 有可复现收益，保留解码，报告收益相对延迟、显存和预训练成本的代价。
- 若所有接口都失败，不能证明完整 τ 必要或无用；先返回 P0 的 executor 资格问题。

最便宜的可判别方案是复用 P1 冻结的单手、少量物体和共同初态面板，训练完整与
压缩两臂；有等效迹象后再训练 latent-only 第三臂。每臂必须真的训练，不能通过
在完整 τ 策略推理时置零、截断或改成常量来替代本实验。

## 独立核到的实现事实

- `src/ptv3_shared_wam_v6/model.py` 的 `SharedWAMV6.encode()` 生成观测/目标/计划事件
  条件下的共享 tokens；`forward()` 另外执行手、物体和 interaction 未来读出。
  `return_shared=True` 仍经过这些读出，不能直接算作“跳过解码”的计时入口。
- 原生手输出为 `B,24,2,21,3`，物体轨迹为 `B,1,24,32,3`；原生 τ 共
  `24 × (42+32) × 3 = 5,328` 个浮点数/窗口。输出有效性仍须使用当前观测 mask。
  单手任务不能凭空把另一只手的 padding 当作额外监督或可用机器人状态。
- `flow_matching.py` 的 `pack/unpack`、位置 embedding 和采样布局硬编码 24 帧，
  每步 222 维；压缩版需要独立、显式的训练/采样合同，现有 FM checkpoint 不能
  静默加载为公平压缩对照。
- `events.py` 定义 NONE/CONTACT/RELEASE、相对 30 Hz 时间与 validity；
  `data.py` 的离线训练条件来自录制事件标注。在线两臂事件都只能来自共同任务
  scheduler 或当前观测预测，不能读取实际未来接触时间。
- 当前 Task 的 `train_pretrain.py` 是 forecast 训练入口，未实现本卡 compact/latent
  策略对照。以下预算、阈值和接口均为待实现协议，不是已完成能力。

## 对照接口

| 臂 | 每次闭环推理时可用接口 | 如何训练 |
| --- | --- | --- |
| F：Full24 | 当前观测共享表示 + 完整未来 24 帧手/物体 τ | 标准完整 τ 监督 + 相同 BC/RL |
| K：Compact | 相同共享表示 + 终点 t24 + 三个锚点 t6/t12/t18 + 事件摘要 | 直接训练四时刻 compact head 和 compact 条件策略；从第一步训练起即使用此接口 |
| L：Latent-only | 相同观测共享表示 + 共同 goal/event 条件；跳过所有未来 τ/C 读出 | 具有相同预训练预算的 encoder + 从第一步训练起不接收解码 τ 的 BC/RL 策略 |

锚点先固定为 t6/12/18（0.2/0.4/0.6 s），终点为 t24（0.8 s）；这是时间锚点，
不是保证在该时刻发生接触的标签。这样不需要在线读取 GT 接触或选择“最优未来帧”。
每个锚点与终点沿用同一手/物点身份、坐标系、有效性和 TRAIN-only normalization。
若后续改成语义抓取锚点，锚点时间也必须由在线预测产生，并单列新配置，不能用
GT 实际接触时间作为隐藏 oracle。

为不改变原生两手输出合同，K 的空间数值为 `4 × 222 = 888`；事件摘要用
3 类概率、预测的相对事件时间和有效概率，共 5 个数值。相对于 F 的 5,328，
空间输出少 6 倍；含事件摘要时为 893 个数值。**这是接口尺寸计算，不能代替
实测 FLOPs、延迟、显存或成功率。** 单手策略只消费同一只活动手及同一物体。
已提供给两臂的 scheduler 事件不可在 K 中升级成 GT 事件。预测事件摘要只用
TRAIN 标签拟合，推理不读取未来标签。

所有策略使用相同宽度的公共共享表示、同容量的最终策略网络和同维的 plan adapter
输出。F/K 的输入投影因接口尺寸不同可以不同，但必须报告参数量；如需容量匹配，
只调整 adapter 隐藏宽度，不增加额外未来监督。L 从训练开始就使用无计划接口，
没有“完整策略临时少一个输入”的分布突变。

## 训练公平性和预算

1. 使用 P1 `P-20261022-wam-v6-samples-efficiency-3arm` 最终冻结的手、物体、
   机器人 TRAIN/DEV/TEST 序列清单、奖励、初始化、动作频率、观测、BC 数据量、
   PPO/其他 RL 算法及交互步数。P1 尚未冻结的字段在本卡仍未指定，不把它们
   当成已存在的运行配置。不得因压缩臂结果不佳而追加数据、奖励或 oracle 条件。
2. 主要因果对照从**同一个预训练 encoder 起点**分叉；F 和 K 各自拟合相应
   解码头，再分别训练 executor。共享 encoder 是否冻结必须两臂一致。不得把
   已长训 Full24 与只训几步的 compact 新头直接比较。L 先用同一起点训练其
   专属 executor；这回答能否删除该预训练模型的推理解码，而非压缩预训练是否充分。
3. 预训练适配主要预算固定数据 exposure：同一 draw 序列、global batch、更新数、
   optimizer/LR、精度、GPU 型号；BC 也固定 batch/更新数和同一 action 标签。
   保留同样的公共辅助损失；完整未来监督可作为 K/L **训练期**辅助头，推理时
   不执行它。辅助头是否存在逐臂记录，不把训练阶段节省成本冒充推理收益。
4. 主要结果另报告每臂实际 GPU-hours、FLOPs 估计与数据 exposure；若有效计算量
   不同，再做**总 GPU-hours 相同**的补充控制。补充控制不得增加 TRAIN 数据集合，
   也不混入主要等 exposure 对照。若两种预算给出不同判断，结论限定到预算口径。
5. 每臂至少 3 个独立策略种子；与 P1 同样的 seed pairing 和共同初态，启动前
   由主线程分配 probe seed 并记录全局 ledger。本设计尚未占用 seed。
   每种子交互步数至少与 P1 首批相同，在其完整预算内比较；不能用纯 smoke 结果
   宣布等效。主要学习曲线每 10% 预算评估一次，固定 checkpoint 选择规则。
6. 开跑前将上述继承字段展开到不可变 run manifest，记录 actual git_commit、
   conda/Python/Torch/CUDA/仿真器版本、物理 GPU 与 clock、数据和 checkpoint 路径。
   当 P1 不具备可运行且公平的基线时，本卡保持 PLANNED，先完成 P0/P1。

## 指标与判据（开跑前固定）

沿用 P1 的抓取—抬升成功定义与达到目标成功率阈值；若 P1 未定义则本卡不擅自
给出达到阈值的交互步数。每个种子至少 300 个共同 TEST 初态（每物体均衡），
各 checkpoint 使用同一初态集合。保留 episode 成对结果和物体分层；不同 seed
的结果不能只混成独立 episode 样本后忽略训练方差。

- 成功率、持握时长、达成率和失败分类；同一固定交互预算下的成功率差。
- 首次达到目标成功率的交互步数，要求连续两次评估达标；未达标记右删失
  `>N_max`，不能从未达标的短学习曲线外推。评估步骤不加入学习交互预算，但
  单独计入总仿真成本，所有臂相同。
- 在同一 `[0,N_max]` 区间的归一化成功率 AUC：梯形积分除以 `N_max`。
- 预训练/适配/BC/RL 分开计 GPU-hours、示教条数、样本曝光量、峰值显存、参数数。
  预训练基础成本注明共享且不可重复冒充某臂新增成本；另报从头总账。
- 单环境 batch1 和实际 rollout batch 的端到端决策 p50/p95、encoder/decoder/
  adapter/actor 分解时延；同 GPU 同 precision、100 次 warmup + 1,000 次计时、
  CUDA synchronize、关闭编译/缓存首次开销后计稳态，并另报首次启动耗时。
  包括 compact 插值/优化/retargeting 及 L 的特征处理成本，不能只计网络部分。

等效仅作 Probe 决策，不作为正式等效性证明：K−F 或 L−F 成功率的配对 95% CI
下界高于 −5 个百分点、归一化 AUC 差下界高于 −0.03、达到阈值步数不多于 F
的 1.10 倍，且至少 3 种子方向一致。区间按训练 seed 和共同初态层级计算；
样本太少/区间太宽则待定，追加在原预算内的 seed 或共同初态，不把无显著差异
写成等效。若双方均未达到成功阈值，不触发删除决策。

成本要求：端到端 p95 至少下降 20%，或在保持 p95 不恶化时推理峰值显存至少下降
20%；这只是本卡预设的工程收益门槛，必须报告绝对数值和误差。F 自身须通过
P0/P1 的可执行与任务效果要求。

- **K 等效、L 未测/更差**：可替换完整时间展开轨迹，但仍保留 compact 推理解码。
- **L 也等效且满足成本要求**：结论为“本任务删掉推理期解码、只保留预训练表示”。
- **F 稳定优于已公平训练的 K/L**：在本手/物体/预算范围保留完整 τ；不外推为
  所有任务都需要 24 帧。
- **区间宽、seed 不一致或所有臂失败**：待定，不能凭推理置零失败宣布必要性。

## 资源、输出与执行状态

优先级 P2；本轮仅设计，不消耗 GPU。未来运行前重新 `nvidia-smi`，仅可用机主
许可且当时空闲的 GPU0–3，优先2/3；绝不触碰 GPU4/5/6 的 FM 长训或 GPU7。
不停止/重启任何已有训练、不 push、不读凭据。若授权设备被占用，保存自己的
运行状态并记录暂停点，不迁移到受保护设备。

输出规划 `outputs/ptv3-shared-wam-v6/<run_id>/`，包含 manifest、每 seed 学习曲线、
共同初态 ID、逐 episode 指标、延迟样本和成本账。设置时间/磁盘/checkpoint 上限，
从 P1 共享剩余资源预算，不另领取全局上限。未形成实际实现前没有可执行训练命令；
不得把下列核查命令冒充实验运行：

```sh
git rev-parse HEAD
sed -n '135,220p' src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/model.py
sed -n '1,145p' src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/flow_matching.py
sed -n '1,95p' src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/events.py
```

当前结果表：训练种子 **0**，交互步数 **0**，评估 episodes **0**，实测 GPU-hours
**0**；F/K/L 成功率、AUC、持握、恢复、端到端延迟均**未测**。接口大小
5,328 vs 893 为代码形状推导；压缩是否提高效率仍为假设。
