审查了当前 `main@03eba5919b`（北京时间 2026年10月20日 04:52）。这次有一个比之前的任务奖励、网络容量问题更明确的发现：

Scheduled686 BC 的初始化代码存在确定性错误：它声称新增 τ 权重零初始化，实际却保留了随机初始化值。与此同时，离线 τ 缓存和在线 τ 推理使用旧调度器，而 BC 的 task10 使用新调度器。

这两个问题都发生在 WAM→BC 的关键接口上，应该先修复，再判断 WAM 是否有帮助。

## 一、当前 BC 实际表现

根据最新提交的实验记录：

| 指标                 | Current421         | Scheduled686                |
| ------------------ | ------------------ | --------------------------- |
| 最优 DEV 动作 MAE      | 0.038258（step3000） | 0.036683（step1000）          |
| 最终 step50000 MAE   | 0.055811           | 0.052845                    |
| 相同5693条 DEV 样本 MAE | 0.036321           | 0.036683                    |
| 300拍闭环             | 两个环境均停留在ACQUIRE    | 飞机停留ACQUIRE；闹钟进入TRACK但进度仍为0 |

因此，目前不能认为 Scheduled686 优于 Current421。反过来，也不能据此断言 WAM 无效，因为当前条件接入存在需要修复的问题。

## 二、最高优先级：Scheduled BC 的零扩展实际上没有实现

位置：

[`train_measured_scheduled_bc.py`](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/03eba591/src/task/wam-rl-policy-v2/tools/run/train_measured_scheduled_bc.py?utm_source=chatgpt.com)

当前代码的关键部分是：

```
model = MeasuredInspirePolicy(with_tau=True).to(device)state = model.state_dict()for key, value in pstate.items():    if key == 'actor.net.0.weight':        state[key][:, :421].copy_(value)    ...
```

问题是 `MeasuredInspirePolicy(with_tau=True)` 创建时，第一层 Linear 默认是随机初始化的。

这里仅复制了前421列，没有执行：

```
state['actor.net.0.weight'][:, 421:].zero_()
```

因此新增的264维τ和1维valid，其第一层权重仍是随机值。

这违背了你们原来明确要求的：

\\[ \pi\_{\mathrm{scheduled}}^{(0)} = \pi\_{\mathrm{current}} \\]

实际情况是：

\\[ \pi\_{\mathrm{scheduled}}^{(0)} = \pi\_{\mathrm{current}} +\text{随机新增条件的影响} \\]

这里的加法只是表示初始网络行为受到随机条件扰动，并非严格的输出线性相加。

影响：

- Scheduled BC 不是严格从成功复制的 Current BC 行为开始训练。
- τ的输入使用0.05米归一化，随机第一层权重直接作用于归一化后的轨迹。
- 即使原本 Current BC 已经学到一定控制能力，Scheduled BC 起始行为也可能改变。
- 后续无法把二者训练结果解释成干净的“只增加WAM条件”的对照。

尤其你们 Scheduled 最优发生在step1000，且两阶段 DEV 差距很小，这个初始化问题不能忽略。

修复后必须增加真实checkpoint回归：

同一批有效τ样本，分别用 Current421 和刚扩展、尚未更新的 Scheduled686 推理，要求输出在浮点误差内一致；然后确认新增265列为零，再检查第一轮优化后这些列确实获得梯度。

这是明确的代码缺陷，不是猜测。

## 三、第二个严重问题：BC 的 task goal 与 WAM 的 goal 不是同一个

这是我认为比 MAE 本身更值得调查的系统性问题。

### 1. Current421 的 task10 来自新 TaskController

数据读取位于：

[`teacher_dataset.py`](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/03eba591/src/task/wam-rl-policy-v2/src/wam_rl_policy_v2/teacher_dataset.py?utm_source=chatgpt.com)

每个示教状态都会调用：

```
result = controller.observe(...)task[t] = result['task']
```

因此，task10 由新的实测阶段状态机产生，包含：

\\[ C_t^{task}= [\Delta SE(3)\_{\mathrm{local}},progress,phase] \\]

在 ACQUIRE 阶段，调度器通常让物体保持原位，或者在抓取候选时发出小幅探测目标；只有抓取建立后才允许路径进度推进。

### 2. Scheduled686 的 τ 却来自旧 OrderedGoalScheduler

位置：

[`generate_measured_tau_cache.py`](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/03eba591/src/task/wam-rl-policy-v2/tools/run/generate_measured_tau_cache.py?utm_source=chatgpt.com)

当前代码仍然是：

```
scheduler = OrderedGoalScheduler(    spec.object_path,    spec.source_frames / spec.source_fps,    geometry[2],    interaction_points=geometry[3],    goal_type=spec.goal_type)decision = scheduler.update(pose, t / spec.source_fps, hand_points=hand)goal = rigid_goal_delta(pose, decision["goal_pose"], geometry[2])
```

这里没有调用生成task10的同一个TaskController。

因此可能产生：

同一帧：机器人还没有建立抓取

实测状态：ACQUIRE，物体位置接近起始位置

BC 的 task10

新状态机给出保持/探测目标，progress = 0

WAM 的 goal

旧空间调度器可能给出沿参考路径的前视位移

同一个 BC 同时收到可能不一致的条件

在线评估又重复了同样的问题：

[`inspire_scheduled_bc_eval.py`](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/03eba591/src/task/wam-rl-policy-v2/tools/run/inspire_scheduled_bc_eval.py?utm_source=chatgpt.com)

它分别创建了 `TaskController` 和 `tau_schedulers`，然后使用后者构建WAM目标。

这不一定导致每一帧的goal不同，但代码没有保证它们一致。 目前也没有看到对所有训练样本的逐帧目标差异统计。

我建议直接把 τ 生成流程改成：

\\[ \text{同一个 TaskController} \longrightarrow \begin{cases} task10\\\ goal\_{\mathrm{WAM}} \end{cases} \\]

再检查 WAM goal 的局部点位移是否与 task10 的6维目标在同一物体坐标系下对应。

这是明确的双调度源接口问题。在重训 Scheduled686 前必须解决。

## 四、还有三个会让 BC 实际效果下降的问题

### 1. 训练后期明显过拟合，不应使用最后一步

Current421 的最佳DEV出现在3000步，Scheduled686的最佳点甚至在1000步。

到50000步：

- Current421 的MAE从0.038258升至0.055811，增幅约46%。
- Scheduled686从0.036683升至0.052845，增幅约44%。

这说明现有50k更新预算远超这批数据达到最优DEV所需的更新数。

代码正确保留了best.pt，所以这不是导致当前best闭环失败的直接bug。 但它说明现在的数据规模、网络容量、学习率或训练配方没有很好匹配。

特别是新语料筛选后只有约5.4万条TRAIN命令，网络仍然是1024/1024/512/512。这种情况下反复随机抽取2048条训练50k次，可能导致记忆训练轨迹而不是学会闭环纠偏。

我不建议马上扩大网络或换优化器。先看不同checkpoint的训练/DEV差距、逐阶段误差和闭环表现。

### 2. 训练状态来自 teacher，闭环状态来自自己，缺少纠偏样本

Current BC 训练学习：

\\[ \pi\_{\mathrm{BC}}(s_t^{teacher},C_t^{teacher}) \rightarrow a_t^{teacher} \\]

闭环时，第一步动作有误差，后续就可能进入：

\\[ s_t^{student}\notin\mathcal D\_{\mathrm{teacher}} \\]

此时不仅手部状态不同，连 `TaskController` 根据实测反馈计算的 phase、goal 都可能不同。

这不是普通代码错误，而是BC数据分布的结构性问题。

现在的现象正好支持这个方向：离线动作误差比较小，但300拍后几乎没有路径进度。

然而，不能仅凭这一点就认定是分布偏移。应该进行逐拍反事实检查，比较示教状态下的动作误差，以及学生开始偏离后的动作误差。

### 3. 物体形状编码器的归一化统计被冻结在随机初始化时

位置：

[`train_measured_bc.py`](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/03eba591/src/task/wam-rl-policy-v2/tools/run/train_measured_bc.py?utm_source=chatgpt.com)

初始归一化统计这样生成：

```
shape0 = model.shape(train_t["points"])model.actor.x_mean[345:409].copy_(shape0.mean(0))model.actor.x_scale[345:409].copy_(shape0.std(0).clamp_min(.05))
```

但训练中 `model.shape` 继续更新，`x_mean/x_scale` 作为固定buffer不再更新。

这不一定错误，固定输入归一化本来就很常见；但这里归一化的是一个持续训练、输出分布会变化的中间特征，而不是固定原始观测。

如果 shape encoder 的输出发生明显漂移，后面的MLP收到的标准化特征分布也会变化。

需要增加shape特征均值、标准差、激活范围和梯度的训练前后审计。它的优先级低于前两个明确问题。

## 五、动作映射和实测状态还有没有明显错误？

目前没有找到新的确定性动作映射错误。

已有审计显示：

- `physical248`与DExplore原生状态最大差约 \\(2.98\times10^{-7}\\)；
- canonical action→PD target误差为0；
- 没有跨episode动作错配；
- Current和Scheduled的平均动作绝对值分别约0.185和0.188，接近teacher的0.191。

因此，不能再把当前失败简单解释成“模型输出全零”“12维动作接错了”或“physical248完全不对应”。

但平均动作幅值正常，并不代表动作方向正确。例如六个腕部维度正负抵消、拇指与其他手指的相对命令错误，整体平均值仍然可能接近teacher。

建议把动作比较拆成腕部6维、各独立手指命令及接触前后阶段，并计算逐维有符号误差、动作方向相似度、PD目标误差和后续手点偏移。

## 六、我建议现在按以下顺序处理

| 顺序 | 修改或实验                                   | 目的              |
| -- | --------------------------------------- | --------------- |
| P0 | 修复 Scheduled686 新增265列未清零               | 恢复严格的BC零扩展初始化   |
| P0 | 统一 task10 与 WAM 的局部goal来源               | 避免同一帧出现不同的任务意图  |
| P1 | 固定当前两个best checkpoint，做逐阶段动作误差和闭环误差传播审计 | 区分动作模仿不足和状态分布偏移 |
| P1 | 在示教原始状态上运行BC，比较命令及下一拍PD/手点变化            | 找到闭环第一次明显偏离的位置  |
| P2 | 检查shape特征分布变化、训练/DEV过拟合与物体划分            | 判断数据规模与网络是否匹配   |
| P2 | 必要时增加纠偏数据或进入经过验证的PPO                    | 解决纯BC本身的分布偏移    |

特别提醒：Scheduled686修复后，应重新生成使用统一goal的τ缓存，再进行一次匹配BC训练。只修改权重初始化、不改旧τ缓存，仍无法排除任务条件不一致的问题。

我不建议现在直接开始大规模PPO，也不建议用当前Scheduled686相对Current421的微小DEV差异证明WAM没有价值。

### 最终判断

目前至少有：

1. 一个确认的代码bug：Scheduled686新增τ权重没有真正零初始化。
2. 一个确认的接口不一致：task10与τ使用两套不同的goal调度器。
3. 一个尚未隔离的学习问题：离线示教拟合与在线闭环控制之间的分布偏移。

从现有证据看，BC失败不完全像原生控制接口接错，更可能是条件输入不一致、初始化不严格，以及纯BC面对实际闭环状态时缺乏纠偏能力共同作用。

先修前两项，再做一次严格配对BC实验。这比继续把训练轮数加大、切换WAM版本或增加新的网络模块更有价值。