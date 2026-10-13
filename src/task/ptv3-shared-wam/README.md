# PTv3-Shared-WAM

实现依据是 [ref1.md](docs/user/ref/ref1.md)，goal编码按[ref1_1.md](docs/user/ref/ref1_1.md)，共享条件按
[ref1_2.md](docs/user/ref/ref1_2.md)更新。
实际模型、张量合同、损失与执行语义见
[架构.md](docs/user/架构.md)。本 Task 单独实现共享预测器，不修改其他 Task 的模型或训练入口。

## 入口

使用 `/home2/wyy/miniconda3/envs/graspenv/bin/python`。已有三域数据、TRAIN 统计和
PW step50000 的默认路径在 `src/ptv3_shared_wam/training.py`；CLI 可以显式覆盖。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/run/train_pretrain.py \
  --smoke --gpu 0 --output outputs/ptv3-shared-wam/<fresh-pretrain-run>

/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/run/train_robot.py \
  --smoke --gpu 0 --init-shared outputs/ptv3-shared-wam/<pretrain-run>/step4.pt \
  --output outputs/ptv3-shared-wam/<fresh-robot-run>

/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/run/launch_native.py \
  --checkpoint outputs/ptv3-shared-wam/<robot-run>/step4.pt --controls 4 --execution-steps 4 \
  --gpu 0 --output outputs/ptv3-shared-wam/<fresh-native-run>

/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/audit/audit_native.py \
  --native outputs/ptv3-shared-wam/<native-run>/native
```

非 smoke 运行须先建立 Task-local experiment card，分配全局 seed、说明问题/决策/预算，
通过 `--experiment-card` 指定；训练 seed 来自卡片第一个 seed，rollout seed 必须列在卡片内。
engineering checkpoint 不会静默进入科学微调或完整物理评估。

预训练和机器人微调均支持 `torchrun`。显式设置空闲设备，例如
`CUDA_VISIBLE_DEVICES=0,1 .../torchrun --standalone --nproc_per_node=2 .../train_pretrain.py ...`。
`batch_size` 是每 rank 数量。`--resume` 恢复模型、优化器、步数和每 rank 随机状态；
保持 stage/config/数据路径和 world size 一致。检查点采用唯一 `stepN.pt`，不覆盖历史文件。

机器人初阶段默认无 human replay；通过 robot config 的 `replay_fraction` 开启后续混合阶段。
人类样本只计算手/场景损失，不生成或使用机器人动作标签。

当前goal是32个 `[当前点, t+24目标点−当前点]` 6D token，加可训练target embedding，
与256个scene、22个hand token共同进入共享trunk，合计310个token。U/V双头都读取
任务条件共享表示，不再有hand分支单独的goal cross-attention。三域/机器人/在线入口共用 `shared_collate` 的endpoint→delta转换，
不使用+8/+16 waypoint或整任务最终位置。config明确标记
`goal_encoding=current-point-horizon-delta.v1` 和
`task_conditioning=shared-scene-hand-goal.v1`；旧absolute-goal和hand-only delta checkpoint默认拒绝加载/续训，
不提供自动迁移。新goal模型可从原PW场景权重重新初始化。

## 验证

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/ptv3-shared-wam/tests -q
```

CPU 合同测试使用可微场景接口 fixture；真实稀疏 PTv3、权重迁移和机器人梯度由
GPU 工程 smoke 验证。工程 smoke 不能说明预训练已有效、策略能抓取或 Cm 有收益。

ref1_2当前20个合同测试通过，覆盖goal影响共享表示及双头、310-token布局和hand读取区间、
scene损失进入GoalTokenEncoder但不进入hand head、时间/目标分组指标、goal-only基线与旧路由拒绝。
评估每个source自动输出scene h1/first8/middle（9–16）/h24/all24，并分别统计target/non-target。
`goal_only/scene_*`基线只把终点delta按j/24插值，非目标静止；终点指标不独立代表动态学习。
新路由的GPU工程smoke已完成，运行代码固定为 `5453d03a`；使用空闲GPU1。
GPU0启动前被外部任务占用，资源检查退出后换卡，没有停止外部进程。

| ref1_2共享goal工程检查 | run_id |
|---|---|
| 真实三域四步预训练 | `engineering-shared-goal-pretrain-20261013-r1` |
| 机器人四步联合微调 | `engineering-shared-goal-robot-20261013-r1` |
| 共享goal四步原生执行＋独立审计 | `engineering-shared-goal-native-20261013-r1` |

预测器51,407,878参数，机器人pair51,991,314参数。三域smoke的hand/scene到goal梯度
范数分别为0.00660/0.01492；机器人对应为0.01391/0.01333，双方均到达真实PTv3和
interaction。scene→hand head无梯度；predicted-action更新forecaster与R，
GT-anchor→forecaster为0、→R非零。单样本训练显存约1.4GB；各4步更新与验证约4.5秒，
不含初始化。原生组约33秒完成，峰值观测显存约18.7GB，两个runtime退出码均为0。
独立数组审计 `pass=true`，PD/chunk/自身H4/R-state/计划goal核对误差均为0。

每个source的新指标与goal-only基线均已落入预训练 `curve.json`；无non-target样本的
类别保留 `mean_mm=null,count=0`。三域target h24的goal-only误差接近数值零，这是
输入终点的代数结果；其first8/middle仍可能有误差，不将短smoke分数解释为学习效果。
本轮没有启动正式长训或完整抓取评估。

历史ref1_1更新后17个合同测试通过，覆盖delta的平移/旋转、有效性掩码、精确6D编码、
hand-only goal路由及旧checkpoint拒绝加载。GPU工程记录如下；只验证接口/梯度，不证明抓取收益。

| Delta-goal工程检查 | run_id |
|---|---|
| 真实三域四步预训练 | `engineering-delta-goal-pretrain-20261013-r1` |
| 机器人四步联合微调 | `engineering-delta-goal-robot-20261013-r1` |
| 在线delta-goal四步原生接线 | `engineering-delta-goal-native-20261013-r1` |

三域/机器人各4步完成，hand与scene损失仍进入PTv3/共享trunk，GT锚定对forecaster梯度为0。
原生两runtime正常退出；保存轨迹的PD/chunk/自身历史/计划goal由独立数组审计复核。

历史absolute-goal版本：14个合同测试通过；双卡预训练与双卡robot+25%human replay各4步通过；
从step2恢复至step4的参数最大差为 `3.47e-7`。
原生工程入口默认4步；`--smoke --controls 16`允许验证完整8步buffer的两次查询，
仍不评价抓取效果。

| 历史absolute-goal工程检查 | run_id |
|---|---|
| 双卡共享预训练 | `engineering-ddp-pretrain-20261013-r2` |
| 双卡机器人＋human replay | `engineering-ddp-robot-replay-20261013-r2` |
| 连续更新 / checkpoint恢复 | `engineering-resume-original-20261013-r1` / `engineering-resume-restored-20261013-r1` |
| 原生K=1 / 4 / 8 | `engineering-native-k1-20261013-r1` / `engineering-native-k4-20261013-r2` / `engineering-native-k8-20261013-r1` |

上述产物位于仓库 `outputs/ptv3-shared-wam/`，各原生run的 `native/independent_audit.json`
复算命令、PD、自己的H4/R-state、计划goal和chunk索引。
早期r1预训练/机器人checkpoint属于scene读出定稿前的工程接线，保留原日志，
不作为最终模型的学习效果证据。第一次DDP robot replay被GPU0的外部任务触发资源停止；
已换GPU3/4完成r2，没有停止外部进程。

Residual PPO、旧 counterfactual World 和旧架构 matched 科学比较是后续实验，
本次没有启动，也没有为目录完整性创建空实验卡。
