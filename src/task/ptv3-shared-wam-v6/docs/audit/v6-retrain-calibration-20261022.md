# v6 最小重训与单轨迹校准交接（只设计，未训练）

本轮任务 A/B 是配置与接口工程修复，不改变权重、损失或任务标注，**不要求重训 v6**。
完整任务声明与安全 prefix 分离则改变了 executor 的条件语义；旧缓存即使 shape 相同也不能
直接当作修复后的训练数据。校准和大规模重训均未启动；真实微拟合等待 Hermes 确认。

## 最小改动清单与决策顺序

| 层 | 最小必要改动 | 明确保留的边界 |
|---|---|---|
| 数据准入 | 对接 `outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1/manifest.json`；按完整 task + 独立 availability 取安全控制。原 cohort 501 条、56,418 controls，不能按旧 UNKNOWN 64 条直接删除 | 491 条删失不能作为完整成功正示教；保留 unsafe/censored 属性；不补造 prefix 外动作 |
| 任务与事件 | 用完整路径、原 source_frames、原 CONTACT/RELEASE 重建 task10；释放后路径和终态继续评估。原事件在 prefix 外时保留声明并标删失，未知事件用 valid=false | 490R/11H 是恢复的历史声明，非经认证的计划意图；raw force/proximity event 不能自动升级为任务标签 |
| 条件缓存 | 重建 executor 当前目标/task10、H4、τ、validity；旧 Current421/Scheduled686 缓存不能仅按维度复用。记录 scheduler/admission/task schema 和生成代码提交 | H4=[t−3,t−2,t−1,t]；GTτ是固定参考未来减固定参考当前，非在线偷看机器人未来；前3帧无 H4 时 invalid |
| τ 与动作 | v6 full21→canonical11，右手 slot0，anchor→world 旋转一次，Inspire 原生顺序重排一次；30Hz 取未来1..8，60Hz 插值且含隐式 τ(0)=0 | 不加物体平移；不把第一未来点当当前点；invalid 的264维归零且 valid=false；canonical12 实际执行动作作为监督 |
| executor 重训 | 首先复制原 Scheduled686 同架构、权重和 normalizer，冻结 CAD shape，只微拟合 actor；后续新条件 BC 如需全量拟合则独立 run/optimizer | 保留原 checkpoint；同条轨迹是 TRAIN 已见，不能当 DEV 或泛化证据；物理回放另验 |
| v6 条件适配（有决策需求才做） | 目前训练 goal 是 t+24 终点，线上是 scheduler waypoint。若要解决此错配，需在 `V6Windows` 加 scheduler-style goal/event 条件生成，并版本化 recipe/验证器与条件缓存 | 保持 GT K24 target、TRAIN-only normalizer 和原 DEV/source split；不能只换在线 goal 后宣称训练条件已对齐 |
| 优化与恢复 | 标注/条件分布变化建立新 experiment/run；可显式 weights-only 初始化，重建 optimizer/selection 记录 | 禁止用旧 strict-resume 伪装同分布续训；现有 `freeze_event_recipe.py` 只绑定原 proximity-v2 event，不能自动承认新的计划意图标签 |

只读依据：本 Task 的 `docs/audit/prefix-task-semantics-fix-20261022.md` 与
`docs/audit/flowmatching-stop-checkpoint-audit-20261022.md`。此次没有修改 v3 reader、PPO 或标注器。

1. **Blocker / 当前最便宜方法**：单条参考动作先在修复后的 reset/PD/评分链给出物理正控，
   再比较同架构 executor 微拟合前后同 GTτ 通道的执行。区别“条件/执行链不可实现”和
   “局部适配可实现”；若正控失败，继续修执行链，不能拒绝 WAM 方法。
2. **Decision / 后续候选**：仅在上述边界通过后比较 scheduler-style 条件适配前后，同一固定
   DEV 面板的预测误差与同一 executor 的执行。结果将决定是否投入完整 v6 条件重训。
3. **Evidence / 延后**：完整多 seed PPO、跨物体泛化、τ 必要性、长期任务成功。当前不执行。

## 预算与预计时间（待授权计划，不是运行结果）

| 阶段 | 硬上限 | 预计时长与依据 |
|---|---|---|
| 本轮接口/dry-run | 零 optimizer updates，CPU 单次 forward | 本机约2秒；已运行 |
| 单轨迹 actor 校准 | **2000 updates 或1800秒，先到停**；batch64/Adam1e-4/seed196；1条≤301行；仅最终权重，≤2GiB，空闲盘≥20GiB | 计划1–10分钟，但未实测 GPU update 速度；每10秒或100步采样 GPU/显存与 ETA，30分钟内停止更新 |
| 数据/条件重建 | 固定原501条/56,418可用控制；只重建派生条件，不改原数据 | 计划10–30分钟 CPU/IO，取决于是否需要新τ推理；批量 WAM 推理另行按空闲 GPU 实测估时 |
| 条件适配 Decision Probe | 候选300 updates 或30分钟先到；3卡只能用空闲4/5/6；local96/global288 仅作为原FM配置参考 | 旧FM step10500→11000权重mtime间隔1952.66秒，即约3.91秒/update；300步约19.5分钟，加初始化/DEV预计25–30分钟。不同损失/精度/数据需重新测量 |
| 大规模 v6/PPO | **本轮无预算、无启动** | 历史11000步累计约10.72小时，不作为新语义训练的完成保证。正式预算需根据上述 Probe 结果再定 |

脚本的30分钟限制是梯度更新阶段墙钟，逐 update 前检查、结束后同步 CUDA；最多存在一个
update 的停止延迟，保存/汇总不计入更新阶段。它不启动额外进程，不做 DDP，不回放 PhysX。
GPU 选择严格限定4/5/6，启动前 `nvidia-smi -i <gpu>`：memory≤128MiB、util≤5%，否则拒绝；
不自动抢占或杀进程。全局仍须保留两张空闲卡。

## 可执行校准脚手架

入口：`tools/audit/v6_single_trajectory_calibration.py`。独立于工作树原先已有的
`run_single_tau_calibration.py`，本轮未修改、调用该旧入口，也未启动它的仿真/训练。

复制实际 `MeasuredInspirePolicy(with_tau=True)` 的完整 Scheduled686 权重/normalizer，
MLP为686→1024→1024→512→512→12。`deepcopy` 检查无参数共享；冻结 shape，优化 actor。
dry-run 从现有 step50000 BC 权重严格加载，做一次 CPU forward；原模型/文件不变。

```bash
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/v6_single_trajectory_calibration.py --dry-run
```

期望 JSON：`status=DRY_RUN_PASS`、`baseline_copy_max_error=0.0`、`fit_updates=0`、
`optimizer_created=false`、`simulator_started=false`、`gpu_allocated=false`。
默认输入明确标记 `SYNTHETIC_DRY_RUN_ONLY`；不代表已验证真实轨迹采集、拟合或物理可实现性。
可加 `--data <record.npz>`，此时先审计真实数据合同，再做一次 forward，仍无 optimizer。

真实输入 NPZ（另行采集/打包，不是本轮输出）：

| 字段 | 合同 |
|---|---|
| `observation` | float32 `[N,686]`，N≤301；按生产 measured347/CAD/task10 构造的411D robot +10D task +264D native11 worldτ +1D validity |
| `action` | float32 `[N,12]`，该帧实际 canonical12 执行命令，范围[-1,1]，不能用未裁剪PD目标充当动作 |
| `tick` | 整数 `[N]`，从0连续递增，不得跨episode/reset拼接 |
| `metadata_json` | 单个 JSON 字符串：下列schema、sequence、任务/来源与τ合同 |

metadata 必需示例（具体路径由真实采集填入）：

```json
{
  "schema": "ref2dex.v6-single-trajectory-calibration.v1",
  "source_sequence": "s1_cylindersmall_pass_1",
  "source_recording": "<owned source-PD trajectory.npz>",
  "task_semantics": "<complete task/scheduler contract identity>",
  "native_hz": 30,
  "tau_frame": "world",
  "tau_representation": "displacement_from_current_hand",
  "hand_order": "native_inspire11",
  "tau_kind": "gt",
  "tau_times_s": [0.03333333333333333,0.06666666666666667,0.1,0.13333333333333333,0.16666666666666666,0.2,0.23333333333333334,0.26666666666666666]
}
```

GT 通道最初3帧和最后8帧 invalid；WAM通道同样前三帧 invalid，另须 `wam_checkpoint`。
两种通道分别建记录，禁止拟合 GTτ 后用 WAMτ 结果替代同通道检验。真实数据的τ推导、
任务意图和 source-PD 正控需通过采集审计；shape/数值检查不会认证这些科学前提。

Hermes 确认后才可运行的命令模板（**本轮没有运行**）：

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/v6_single_trajectory_calibration.py \
  --execute-training --data <single-record.npz> \
  --output outputs/ptv3-shared-wam-v6/<fresh-run-id> --gpu 5 --updates 2000 --seconds 1800
```

输出包括环境/代码/来源信息、拟合曲线、初末 action RMSE、GPU telemetry/ETA、停止原因及一个
`calibrated.pt`。不按训练成功挑多个 checkpoint。随后基线/校准模型用共同初态、同一固定τ
参考做真实物理回放，记录完整任务成功、掉落与安全abort；action RMSE下降不能替代此项。
