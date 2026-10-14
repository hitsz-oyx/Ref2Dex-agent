# PTv3-Shared-WAM

实现依据是 [ref1.md](docs/user/ref/ref1.md)，goal编码按[ref1_1.md](docs/user/ref/ref1_1.md)，共享条件按
[ref1_2.md](docs/user/ref/ref1_2.md)更新。
实际模型、张量合同、损失与执行语义见
[架构.md](docs/user/架构.md)。本 Task 单独实现共享预测器，不修改其他 Task 的模型或训练入口。

## 入口

当前机器人适配使用[660条补录/联合微调实验卡](docs/experiments/probes/P-20261014-ptv3-shared-rl660-joint.md)。
原660个RL tensor缺实际命令，不能直接传给旧airplane loader。新入口先补录真实机器人
状态/手点/物体/实际命令，再冻结完整序列划分和10cm明显失跟准入；全部660均采集，
每条的实际准入窗口数在corpus manifest报告。该门不代表成功抓取。

```bash
python3 src/task/ptv3-shared-wam/tools/run/collect_rl660.py --detach --gpus 4,5 \
  --experiment-card src/task/ptv3-shared-wam/docs/experiments/probes/P-20261014-ptv3-shared-rl660-joint.md \
  --output outputs/ptv3-shared-wam/<fresh-rl660-corpus>

/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam/tools/run/launch_pretrain.py --detach --stage robot --gpus 4,5 \
  --config src/task/ptv3-shared-wam/configs/robot_660.json \
  --experiment-card src/task/ptv3-shared-wam/docs/experiments/probes/P-20261014-ptv3-shared-rl660-joint.md \
  --source outputs/ptv3-shared-wam/<completed-rl660-corpus> \
  --init-shared outputs/ptv3-shared-wam/pretrain-main3-ddp4-80k-20261013-r2/latest.pt \
  --output outputs/ptv3-shared-wam/<fresh-rl660-joint> \
  --loader-workers 4 --loader-prefetch 2 --sequence-cache 128 --pin-memory
```

新训练独立保存dev best/latest，清理恢复点时保护best。原三域预训练已由用户停止，
最后日志39100、完整latest38000；其best别名也是38k，仅代表现存权重中的最佳，
历史dev最低22k的权重因旧保留缺陷已丢失。不得称38k为历史真正best。

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

## 四卡80k预训练

用户授权的长期预训练配置为[pretrain_80k.json](configs/pretrain_80k.json)，协议与运行身份见
[实验卡](docs/experiments/probes/P-20261013-ptv3-shared-main3-80k.md)。GPU4/5/6/7，
每卡128、global batch512；80,000次optimizer updates。保持原学习率，不自动按batch放大。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/run/launch_pretrain.py \
  --detach --gpus 4,5,6,7 \
  --config src/task/ptv3-shared-wam/configs/pretrain_80k.json \
  --experiment-card src/task/ptv3-shared-wam/docs/experiments/probes/P-20261013-ptv3-shared-main3-80k.md \
  --output outputs/ptv3-shared-wam/<fresh-run-id>
```

数据管线可通过 `--loader-workers 4 --loader-prefetch 2 --sequence-cache 512 --pin-memory`
启用。workers使用spawn并长期复用；每source/worker的序列缓存有界，不重写数据。
每步抽样仍只由seed/step/rank确定，worker顺序和预取不消耗模型主RNG。
运行参数记录在provenance.execution，模型config保持不变，旧checkpoint可按相同recipe续训。

每1,000步验证，每2,000步保存恢复点；保留step1、每20k里程碑和最近一个恢复点，
只移除当前run登记表中被替代的非里程碑文件。默认旧配置仍保留全部checkpoint。
源码固定，wall-time硬上限7天，单run4GB/共享400GB预算不变。
监督进程独立于当前终端，保存PID、退出码、GPU快照和30分钟无进展检查；
只向它自己创建的torchrun进程组发送停止信号。
训练目录的 `progress.json`、`process.json`、`launcher.log`、`gpu.jsonl` 可查看进度；
`checkpoint_files.json`登记当前保留的恢复文件。直接torchrun入口仍支持 `--resume`。优化版可从旧运行完整恢复点继续：

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/run/launch_pretrain.py \
  --detach --gpus 4,5,6,7 \
  --config src/task/ptv3-shared-wam/configs/pretrain_80k.json \
  --experiment-card src/task/ptv3-shared-wam/docs/experiments/probes/P-20261013-ptv3-shared-main3-80k.md \
  --resume outputs/ptv3-shared-wam/pretrain-main3-ddp4-80k-20261013-r1/step2000.pt \
  --loader-workers 4 --loader-prefetch 2 --sequence-cache 512 --pin-memory \
  --output outputs/ptv3-shared-wam/<fresh-resumed-run>
```

本次已由 `switch_prefetch_at_checkpoint.py` 在step2000边界完成切换，原产物保留。
当前run为 `pretrain-main3-ddp4-80k-20261013-r2`，代码8510ec3a，目标仍总step80000。
四卡稳定约0.97–1.01秒/update、510–530样本/秒，取batch等待约1.2ms；
30秒平均GPU利用率约90%，原同步版本约40–43%。数据tensor/RNG一致性和恢复检查见实验卡。
主训练代码的延后补丁仅在旧进程组退出后应用；切换状态见原run scratch的
`prefetch_migration.json`。worker IPC目录使用短的项目 `tmp/lw-<pid>/` 路径，
避免长run/rank路径超过Linux Unix-socket限制。

大batch工程验证可使用 `--smoke --smoke-batch-size 128`，四步且不用于科学初始化。
单卡64和128、四卡128都已通过；四卡稳定约2.1–2.25秒/步、约230–245样本/秒，
实测显存约14–16GiB/卡。初步80k预计约49小时，后续以实际长跑ETA为准。

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
新路由的GPU工程smoke已完成，本Task实现提交为 `5453d03a`；使用空闲GPU1。
预训练manifest的运行commit为 `5453d03a`，机器人为 `84968e49`；期间其他Task
提交未改变本Task实现，具体运行身份以各manifest为准。
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

完整660补录支持 `collect_rl660.py --resume`：只复用完整且属于本corpus的batch回执，
失败批次和日志不覆盖；恢复保持固定成员、分批seed和控制参数，累计计入原时间预算。
`tools/run/finetune_after_collection.py` 可有界等待完整corpus、逐序列审核标签和准入索引，
然后交接到独立训练监督器；未完整采齐或GPU有冲突时不会启动微调。

容量扩展用 `configs/robot_660_large.json` 与 `--expand-shared`：从完整robot checkpoint
保留PTv3/观察编码器/预测头/已学R，显式扩宽加深共享trunk并重新初始化optimizer。
这与 `--resume` 的精确恢复不同。整数复制与恒等新层、token/padding向量化、
激活重计算、融合优化器及TF32均在当前Task内实现。性能工具：
`tools/audit/benchmark_robot_scale.py`、`tools/audit/audit_scale_transfer.py`。

## ref2：第一控制目标与RL660执行接口修复

修复保留旧配置的 `all8-standardized-l1.v1` 语义；显式设置
`action_objective=control-aware-first.v1` 才启用历史4×first+tail+PD0配方，训练raw输出，
指标/执行clamp。新模板 `configs/robot_660_control_first.json` 明确K1语义、run8GB预算，
旧4GB配置与运行不被静默改写。改变配方须新run/显式warm start，不能精确resume旧配置。
该模板尚未启动新一轮联合训练，先通过R-only和GT执行门。

新robot训练另存 `forecast_best.pt`（hand_all24+scene_all24物理mm误差之和）和
`control_best.pt`（GT first XYZ p95、predicted first XYZ p95、GT rotation/finger p95
依次打破平局）；兼容保留总objective的 `best.pt` 和恢复用 `latest.pt`。任一选择改善
立即保存完整权重，清理保护四个别名；resume只从仍有权重的历史点初始化选择，不能
把已删除/未保存的dev最优分数当成可恢复候选。多份完整大模型约需5GB，故新模板
为8GB，仍须在全局400GB内预留；不能用旧4GB运行假装容纳四份完整恢复点。

RL660执行必须显式 `--sequence <manifest中的完整序列名>`。server/native共用同record
的CAD、reference_obj、真实hand oracle和几何motion；原生场景检查实际motion/object身份，
关闭旧的重复采样/左手接触过滤仅限该进程的motion membership，不改变物理控制。
独立audit从原始corpus重建目标；旧单文件/airplane路径仍兼容。不得把加载成功当作抓取验证。

可用 `--retargeter <R-only/B_control_best.pt>` 在不复制完整WAM的情况下测试修复R，
要求source、TRAIN normalizers、PD units和8-query结构一致，server/native必须加载同一
候选；此时强制 `--execution-steps 1`。one-query C和256窗过拟合权重不能冒充合格执行R。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam/tools/run/launch_native.py \
  --checkpoint outputs/ptv3-shared-wam/<robot-run>/latest.pt \
  --sequence s10_airplane_pass_1 --execution-steps 1 \
  --retargeter outputs/ptv3-shared-wam/rl660-R-control-repair-20261014-r1/B_control_best.pt \
  --controls 4 --smoke --preflight-only \
  --output outputs/ptv3-shared-wam/<fresh-native-run>
```

`--preflight-only`只做CPU文件/权重合同，不启动GPU或仿真。移除该参数后才是物理
工程检查；完整Probe仍需card、seed、空闲GPU和预算。R-only配方比较与256窗检查见
主工作树的 `P-20261014-rl660-R-control-repair` 实验卡。活跃旧训练有源码保护，修复先在
隔离工作树验证，待旧进程退出后再应用主工作树，禁止热改正在执行的旧训练源码。

### ref2_1：统一查询时刻的物理目标 chunk

新模板 `configs/robot_660_physical_targets.json` 显式使用
`action_representation=pd-relative-current.v1`，all8 standardized L1等权；旧配置和
native权重仍保留旧单位。每个未来PD目标（含手指）减同一个查询q[t]，按TRAIN的
每horizon/坐标标准化。执行查询时恢复并冻结绝对目标，消费时以live q反算native
command，K4/K8不反复重加query delta。几何旋转向量表示是独立诊断臂，Euler分支
roundtrip未通过时拒绝加载到native执行。

`tools/run/probe_R_representation.py` 在相同小R、GT-τ、split/sampler/预算下比较
native、relative-PD与relative-geometry，不更新forecaster；协议与结果见
`docs/experiments/probes/P-20261014-rl660-R-physical-targets.md`。
`train_robot.py --init-forecaster-only` 可显式从保留的robot checkpoint只继承forecaster，
重新初始化R/optimizer；本轮尚未重启大规模微调。更换表示不保证消除输入条件歧义，
离线误差改善也不等于抓取通过。

### 连续分支几何目标与大规模运行

`configs/robot_660_geometry_lift.json`保持原大规模微调的全部训练超参，使用
`wrist-geometry-continuous-lift.v1`：每步原12维几何目标加3维未折叠腕部关节角差。
执行时冻结姿态和连续角参考，用参考恢复Euler分支，再按live q转换native command。
它保留±π方向、多圈和奇异姿态的控制器目标信息；不把同姿态误认为同servo动作。
R输出头增加387参数，trunk/层数/head数不变。标签全量回放和实际执行接线通过后，
才启动独立19k联合run；旧native/12维权重须按原合同加载，不能静默解释成15维。
新run参考实验卡 `docs/experiments/probes/P-20261014-rl660-geometry-lift-joint.md`。
