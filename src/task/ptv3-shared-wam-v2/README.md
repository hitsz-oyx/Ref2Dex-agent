# PTv3-Shared-WAM v2

用户已确认的新预训练路线：[架构与数值合同](docs/user/架构.md)。复用三域数据与
加载工具；独立联合PTv3、hand双路编码、768D×12 shared和horizon标准化。
旧Task仍承担历史 v1 运行；本Task现在提供显式 q 三维合同的 robot/Chunk-R
适配入口。`q_ref`（几何源）、`q_pd`（RL660 feedback-compensated target）和
`q_real`（执行状态）在窗口中分开保存，动作损失只使用 q_pd 的
`wrist-geometry-continuous-lift.v1` 编码。远端 WAM 只需提供兼容的 v2
pretrain parent checkpoint，RL660 与几何源仍由本机只读加载。

```bash
# 所有命令使用graspenv，TMPDIR指向项目tmp，先确认GPU空闲。
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v2/tools/audit/fit_horizon_stats.py --samples 4096 --seed 18 \
  --output outputs/ptv3-shared-wam-v2/horizon-stats-main3-20261014-r1/stats.json
```

全部命令从仓库根目录运行。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v2/tools/run/train_pretrain.py --smoke --gpu 4 \
  --stats outputs/ptv3-shared-wam-v2/horizon-stats-main3-20261014-r1/stats.json \
  --output outputs/ptv3-shared-wam-v2/<fresh-engineering-run>

/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/ptv3-shared-wam-v2/tests -q
```

训练支持torchrun与Task-local实验卡；预训练有界监督器为
`tools/run/launch_pretrain.py`，机器人适配使用
`tools/run/launch_robot_finetune.py`，两者都必须指定config、stats、experiment-card、
gpus和新output。机器人监督器每分钟记录 progress、GPU、磁盘和仅自己创建的进程组，
默认在无进展30分钟或剩余磁盘低于20GiB时失败退出。
resume显式指向v2 checkpoint，在新目录恢复相同config/world size/optimizer/RNG；
工程checkpoint不能进入科学运行，不能把v1权重隐式迁移到v2。

当前实现已通过61项新旧Task合同测试。最终模型98,591,142参数：geometry11,092,000，
shared85,054,464；其余为输入融合、goal与双预测头。
单卡batch64工程负载峰值分配约7.1GiB，物理显存约8.6GiB；工程检查不代表预测有效。
当前研究运行与后续结果见[实验卡](docs/experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md)。

机器人适配工程 smoke / Probe 入口：

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v2/tools/run/train_robot.py \
  --config src/task/ptv3-shared-wam-v2/configs/robot_remote_finetune.json \
  --stats outputs/ptv3-shared-wam-v2/horizon-stats-main3-20261014-r1/stats.json \
  --source outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1 \
  --init-shared outputs/ptv3-shared-wam-v2/<remote-parent>.pt \
  --output outputs/ptv3-shared-wam-v2/<fresh-run> \
  --experiment-card src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-ptv3-shared-v2-robot-finetune.md
```

远端 parent 取回并通过四步 smoke 后，正式两卡 Probe 建议交给 owned supervisor：

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v2/tools/run/launch_robot_finetune.py --detach \
  --gpus 4,6 --config src/task/ptv3-shared-wam-v2/configs/robot_remote_finetune.json \
  --stats outputs/ptv3-shared-wam-v2/horizon-stats-main3-20261014-r1/stats.json \
  --source outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1 \
  --init-shared outputs/ptv3-shared-wam-v2/<remote-parent>.pt \
  --output outputs/ptv3-shared-wam-v2/<fresh-run> \
  --experiment-card src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-ptv3-shared-v2-robot-finetune.md \
  --loader-workers 2 --loader-prefetch 2 --sequence-cache 16 --pin-memory
```

先用 `--smoke --gpu 4` 验证 wiring；科学运行需使用远端 parent，并以
`torchrun --nproc_per_node=2` 占用 GPU4/6。不得用本地工程 checkpoint 冒充远端结果。

## DExplore rollout 人工审阅（Viser）

直接从新采集的 `body_state`、物体姿态及原始URDF视觉网格重建真实执行状态，
不重新仿真。蓝色手、橙色物体；绿色为当前参考物体线框和16关键点骨架，
红色点为原生5个接触body的实测接触（灰色表示无接触）。桌子使用原几何文件中
与native loader相同的固定pose。参考骨架只连接真实保存的关键点，不补造关节。
可旋转/缩放、切换660条序列、暂停逐帧、调整速度、选择片段并标注可用/不可用/待定。
物体位置窗口筛选通过不等于抓取成功；接触力也可能来自桌子等碰撞，不能自动视为手物抓持。

```bash
TMPDIR="$PWD/tmp" /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v2/tools/audit/review_dexplore_rollouts.py \
  --corpus outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1 \
  --output outputs/ptv3-shared-wam-v2/dexplore-review-20261015-r1 \
  --sequence s1_mug_lift --port 8768 --hours 8
```

仅监听127.0.0.1。远程使用时在VS Code转发8768端口，再打开 `http://localhost:8768`；
也可SSH `-L 8768:127.0.0.1:8768`。多浏览器共享播放和标注表单，适合单人审阅。
CPU读取文件，浏览器绘制，不使用训练GPU；播放默认30Hz且可逐帧。
8小时后自动退出，可用同一output再次启动；保存的标注会保留。

标注只新增到review output的 `annotations/*.json`，不改变源corpus或TRAIN/dev。
GUI首尾帧均包含；落盘统一半开区间 `[start_frame,end_frame_exclusive)`，记录source run、
sequence、split、30Hz、判定、原因及备注。重复审阅保留每次记录，后续冲突合并和
训练窗口准入需另行显式处理；当前不会把标签自动应用到训练。新标注上限10000条、
每条备注2000字符；训练窗口的未来24帧与历史上下文也应受坏片段影响，不能只查起始帧。

## 完整训练指标

[预训练实验卡](docs/experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md)已补齐远端
80k完整81次固定dev主指标、59k best与80k final的所有物理分组、本地1k五次评估，
并链接JSON、含有效点计数的CSV与曲线。[微调实验卡](docs/experiments/probes/P-20261015-ptv3-shared-v2-robot-finetune.md)
包含恢复轨迹的固定dev全表，本次快照至9500。正式产物在
`outputs/ptv3-shared-wam-v2/wam-v2-metrics-20261015-r1/`；原始日志仍保留，不依赖tmp快照。

CPU-only重导出既有指标，不做模型计算，也不改训练：

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v2/tools/audit/export_training_metrics.py \
  --pretrain-snapshot outputs/ptv3-shared-wam-v2/wam-v2-metrics-20261015-r1/remote-completion-current.json \
  --local-probe outputs/ptv3-shared-wam-v2/pretrain-main3-probe-20261014-r1 \
  --robot-source outputs/ptv3-shared-wam-v2/ptv3-shared-v2-robot-finetune-20261015-r2 \
  --robot-source outputs/ptv3-shared-wam-v2/ptv3-shared-v2-robot-finetune-20261015-r5 \
  --output outputs/ptv3-shared-wam-v2/<fresh-metrics-output>
```

导出校验评估step完整、指标集合/计数稳定、空分组为null/count0、goal-only基线不随训练
变化，以及恢复前后微调dev面板相同。预训练与robot dev没有单独保存归一化hand/scene
loss；训练分项只作为训练记录，不代填缺失dev指标，也不从离线loss推断抓取成功率。

## 微调阶段权重保留

用户要求保留阶段权重，以便比较action与forecast趋势对下游表现的帮助。
当前完整权重归档为 `outputs/ptv3-shared-wam-v2/robot-finetune-stage-archive-20261015-r1/`，
`archive-manifest.json` 记录step、原run、dev指标和完整checkpoint路径。
使用同文件系统硬链接保护完整模型WAM+R、normalizer、optimizer和四rank RNG，
原run轮换删除目录项后，归档文件仍可读取。不修改活跃训练的源码、配置或清理逻辑。

```bash
TMPDIR="$PWD/tmp" /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v2/tools/audit/archive_robot_stages.py \
  --source outputs/ptv3-shared-wam-v2/ptv3-shared-v2-robot-finetune-20261015-r2 \
  --source outputs/ptv3-shared-wam-v2/ptv3-shared-v2-robot-finetune-20261015-r5 \
  --active-run outputs/ptv3-shared-wam-v2/ptv3-shared-v2-robot-finetune-20261015-r5 \
  --output outputs/ptv3-shared-wam-v2/robot-finetune-stage-archive-20261015-r1
```

CPU文件元数据操作，无GPU；每30秒检查，最多4小时/16GB逻辑checkpoint体积，
文件系统保留20GiB余量。只接收原子rename完成的step*.pt，不归档未写完的.part；
源run完成/失败后最后扫描并退出。启动时仍在的完整阶段为1000/3000/3500/4000/8000，
后续保存点自动保留；早期已轮换删除的完整权重无法从loss日志或R-only快照恢复。
stage选择和下游实验之后再做，当前没有改变任何训练输入或提前停止主训练。

## 远端运行条件

远端已部署在 `/home/wbcd/workspace/oyx_ws/wam-v2-20261014/repo`；使用
`/home/wbcd/miniconda3/envs/graspenv/bin/python`，GPU0/2。默认三域输入路径已软链接
到既有NAS data-remote；统计位于本repo的
`outputs/ptv3-shared-wam-v2/horizon-stats-main3-20261014-r1/stats.json`。
只迁移统计的data路径，TRAIN数值、划分与抽样权重保持不变。

该主机默认文件句柄软上限1024不足以容纳每source128项mmap缓存。启动监督器前，
在当前shell使用 `ulimit -n 65536`；这是进程级设置，不修改系统配置或其他任务。
训练与spawn workers会继承它。硬上限1048576无需改变。

远端Torch2.0.1的fused AdamW从CPU映射checkpoint恢复时，会把step标量留在CPU。
版本d0288294在v2的restore_optimizer中显式将fused/capturable step移到参数设备；
恢复必须使用该修复。GPU回归工具为 `tools/audit/audit_optimizer_resume.py`，
`--legacy`复现旧路径，默认路径验证加载后继续更新。
大规模训练前重新确认GPU空闲、固定预算和配置；工程预检不证明长期预测效果。

## 微调阶段物理抓取评估

入口复用旧Task三路native仿真，必须显式指定`--model-version v2`；v2完整checkpoint
严格加载WAM+R。批量入口以GPU0/1/2并行测试九个保留阶段与固定三个dev序列，
详情见[实验卡](docs/experiments/probes/P-20261015-v2-stage-grasp.md)。

```bash
TMPDIR="$PWD/tmp" PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v2/tools/run/probe_stage_grasp.py \
  --archive outputs/ptv3-shared-wam-v2/robot-finetune-stage-archive-20261015-r1 \
  --output outputs/ptv3-shared-wam-v2/<fresh-stage-grasp-run> \
  --card src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-v2-stage-grasp.md \
  --gpus 0,1,2
```

`result.json`保存每阶段/序列/角色的held45、最长保持、抬升、手物误差和饱和；
各子目录保留轨迹、运行回执与独立控制审计。GTτ→R是oracle轨迹经过学得R的诊断，
并非直接GT q_PD重放；teacher校准不足的序列不能作为模型抓取失败的独立证据。

## 完整teacher输入监督上限

[Full-teacher-input实验卡](docs/experiments/probes/P-20261015-full-teacher-input.md)
记录无tau的1442→单步q_PD拟合及matched721对照。数据从原660同次状态/参考重建，
先经原teacher确定性重放检查；缓存准备入口`tools/run/prepare_full_teacher_membership.py`
与`prepare_full_teacher_data.py`，训练入口`tools/run/probe_full_teacher.py`。
结果、曲线和完整误差表在`outputs/ptv3-shared-wam-v2/full-teacher-input-20261015-r1/summary/`。

## 新微调统一数据与严格位置准入

新robot训练默认使用补采`dexplore-current-rollout-20261015-r1`，保留其原TRAIN/dev序列划分。
实际物体位置相对当前参考偏差严格大于0.1m或非有限时，将该帧及此前50帧标为不可信；
任何H4历史至future24标签覆盖不可信帧的窗口都不参与采样、归一化和评估。
例如坏帧100会隔离50…100，query26…103随窗口排除。阈值恰好10cm不排除。
这只是位置质量准入，不保证接触、旋转、手姿和持续抓取成功。人工备注仍独立保存。
原始660文件、原manifest和历史checkpoint不改写；旧实验复现可显式构造无新准入的reader。
报告入口`tools/audit/audit_rollout_admission.py`，训练run保存独立`rollout_admission.json`。

## Shared actor接口对照

[匹配三臂协议](docs/experiments/probes/P-20261015-shared-actor-interface.md)：
`robot_shared_actor_{pred_tau,shared,shared_tau}.json`分别选择state+预测τ、state+shared、
state+shared+预测τ；同一80k父模型/严格补采数据/双卡global256/12k更新。保留forecast损失，
三臂均关闭GTτ anchor。run独立保存物理first-control指标与forecast/action全分项，
每2k完整阶段及按total/control/forecast分别选择的权重。legacy native执行入口拒绝新头，
不能直接用旧τ接口评估shared checkpoint。
