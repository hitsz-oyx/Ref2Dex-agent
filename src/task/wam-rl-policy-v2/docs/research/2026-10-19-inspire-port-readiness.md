# WAMv4 scheduled BC → PPO 在 DExplore Inspire 上的接入准备

日期：2026-10-19。范围：本地源码和现存产物的只读核查；未训练、未运行 GPU 仿真。用户要求在当前 Shadow PPO 评估后，尝试**同一方法**：冻结 WAMv4、在线局部 goal scheduler、自训练 BC warmstart、原生动作 PPO。旧 WAMv2/shared-R 实验只用于复用数据和原生接口，不是该流程的已完成复现。

## 判断

可以推进一次小规模 TRAIN lift 工程 Probe，不需要重新采集 660 条或生成全量预测缓存。但现有 Shadow runner/BC 不能直接换手加载；没有找到已经实现 WAMv4 + scheduler + Inspire BC/PPO 的完整 runner。

已有两种可用起点：重新用一个 TRAIN lift 的已记录原生命令拟合小型执行器，或把现存 current-only scratch BC 的主干转成新的 τ 条件执行器初始化。后一条仍需明确权重转换及再拟合，不能直接把 current-only checkpoint 称为 WAM 条件 BC。

## 已存在的数据和原生入口

已完成、允许训练的同执行数据根：`outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1/`。根 `manifest.json` 记录 660 条、30 Hz、继承 TRAIN/val 分割及 current-observation audit。每条 `trajectory.npz` 同时保存实测 `q/dq`、11 手点、物体 SE(3)、原生命令、实际 PD target 和 current721，避免把不同时执行的接触与动作拼接。[采集实现](../../../ptv3-shared-wam/tools/run/collect_rl660_batch.py)、[采集实验卡](../../../ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-dexplore-current-rollout.md)。工程审计 PASS 不等于全部示教抓取成功；已有人工异常记录，不能把全量窗口默认为优质示教。

小规模首选现存 TRAIN 序列：

| 序列 | 物体 | 状态帧/控制拍 | 同执行数据目录 |
|---|---|---:|---|
| `s1_spheresmall_lift` | spheresmall | 301 / 300 | `batches/batch008/sequences/s1_spheresmall_lift/` |
| `s1_airplane_lift` | airplane | 432 / 431 | `batches/batch004/sequences/s1_airplane_lift/` |

每条有 `canonical.npz`：`points[512,3]`、`normals[512,3]`、center/radius；以及 `trajectory.npz`：`q/dq[T,18]`、`hand[T,11,3]`、`obj/reference_obj[T,4,4]`、`native_current_obs[T,721]`、`applied/action[T-1,18]`、`pd_targets[T-1,18]` 等。参考是决策已知轨迹；**未来实测 hand 不能作为部署输入**。球的参考高度范围约 0.8473–1.0700 m，飞机约 0.9181–1.1364 m，这仅是所存参考描述，不是新成功率。

使用 `ptv3_shared_wam.native_inputs.resolve_native_inputs()`、`load_native_arrays()`、`install_selected_motion_contract()` 可以解析单条成员、匹配几何/计划及约束原生 motion loader。[源码](../../../ptv3-shared-wam/src/ptv3_shared_wam/native_inputs.py)。原始几何计划在 `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric_dexplore/<sequence>/interaction_hand_inspire.pt`，外部目录只读。

原生代码是 `third_party/DExplore/dexplore`，完整资产视图使用 `/home2/wyy/oyx_ws/dexplore/dexplore/data/assets`。既有 collector 在输出下建立只读资产软链接并以该输出为 cwd，避免相对资源路径误读。既有原生 reset/兼容 helper：`consequence_evaluator.native_reset.install_reset_patch()`、`probe_official_generator.install_legacy_player_compat()`。不能直接照搬旧 `execute_wam_native_bridge.py`：它绑定旧 goal-point-WAM 和 R 的特定身份，且不是本次 WAMv4 流程。

## 观测、动作和时钟必须新建明确合同

原生 actor 的 1442 维是 delta1 + delta16 各 721 拼接；部署 current-only 输入可调用 `_compute_observations_iter(all_ids, 0)`，而不是对 1442 取前半（前半仍是 delta1）。721 包含同刻 reference 相对特征，不能称为纯本体观测；原采集已验证修改未来参考不改变 delta0。[原生观测构造](../../../../../third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py)、[current 采集](../../../ptv3-shared-wam/tools/run/collect_rl660_batch.py)。

Inspire 是 18 维原生命令，但独立控制索引为 `(0,1,2,3,4,5,6,8,10,12,14,15)`：前六腕部 current-relative，手指由 `(-1,1)→(0,1)` 后映射 PD；7/9/11/13/16/17 在原生控制器被 coupling 覆盖。可训练 12 维独立命令，再显式扩成 18 维供环境，保留 canonical action 副本。[原生控制器](../../../../../third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py)、[ACTIVE 定义](../../../trajectory-policy/src/trajectory_policy/retargeter_data.py)、[PD helper](../../../trajectory-policy/src/trajectory_policy/pd_retargeter.py)。不要把 12 维 PD target 当归一化 command，也不要让 PPO 为无效六维优化独立 likelihood。

`inspire.yaml` 的 `controlFrequencyInv:2` 是 30 Hz 控制，60 Hz 物理；WAM H4 应直接取最近四个控制观测 `[-3,-2,-1,0]`。现 Shadow `scheduled_conditioning_contract()`、`MeasuredShadowState`、`tau_for_student()` 等硬编码 current300/565 维、60 Hz、history `[-6,-4,-2,0]`，不可直接复用这些合同。可为最小试跑选择 8×30 Hz τ（与 WAM 前八行直接对齐）、current721+264+valid1=986 维；这是**建议的新 Inspire 合同**，尚未实现，跨度约 0.267 s，不冒充 Shadow 的 8×60 Hz/0.133 s。若需相同时间跨度，应显式选 4×30 Hz，并同步 BC/PPO 维度。[配置](../../../../../third_party/DExplore/dexplore/data/cfg/inspire.yaml)、[当前 Shadow 合同](../../src/wam_rl_policy_v2/control_contracts.py)、[当前 τ 转换](../../src/wam_rl_policy_v2/tau_contract.py)。

原生手点顺序是 base、thumb-base、thumb-tip、index-base、index-tip…；WAM 顺序是 wrist、五个 bases、五个 tips。现成 `trajectory_policy.wam_shadow.NATIVE_TO_WAM=[0,1,3,5,7,9,2,4,6,8,10]` 与 `WAM_TO_NATIVE=argsort(...)` 已存在，接入前重排，送执行器前按自己的 BC 合同重排。它配对语义角色，不保证人手与机器人 landmark 物理位置完全相同。[映射源码](../../../trajectory-policy/src/trajectory_policy/wam_shadow.py)。旧 `robot_goal_sample()` 是旧模型字段，不能代替 WAMv4 sample。

## WAMv4 和几何的复用范围

可直接复用 `wam_rl_policy_v2.wam_v4_runtime.V4Runtime`、`v4_online_samples()`、`collate_v4()` 和 `OrderedGoalScheduler`。前者接收 H4 单手几何与物体 pose，负责当前物体 anchor、H4 dynamics、法线、mask、32 个同 ID flow goals、128 个 interaction query。冻结最终权重已有 `outputs/ptv3-shared-wam-v4/local-control-final-20261019-r1/step80000-model.pt`。[runtime](../../src/wam_rl_policy_v2/wam_v4_runtime.py)、[scheduler](../../src/wam_rl_policy_v2/goal_scheduler.py)。

`canonical.npz` 的 key 是 points/normals，而 runtime cache loader 要求 manifest records 中 `object_code` 和 geometry 文件，geometry key 为 surface/normals/**query**。可以只为这一个或两个物体生成几 KiB 的小几何包和 manifest：将点/法线复制，按 `prepare_v4_geometry()` 的默认 32 点选择建立并冻结 query 身份；再用它生成 128 interaction 身份。不得每 tick 重新抽样，也不能把两次不同的几何查询误当同 ID。[v4 geometry helper](../../../3d-data-preparation/src/pose_mesh_data/v4_windows.py)。

最终 WAM checkpoint 绑定远端 dense-surface 身份；本地 final-swap 评估已使用显式 `allow_geometry_replacement=True` + `dense_manifest`。Inspire 几何新增不能假装匹配原 UniDex 本地 surface manifest。需要新的小 manifest 与替换记录，并将实际几何、normalizer、WAM、scheduler 绑定到新的 BC/PPO checkpoint。`V4Runtime` 的 loader 对新物体 overlay 会跳过不匹配 key，故新 manifest 必须包含实际 Inspire key；不能只传旧 UniDex manifest 让开关看似通过。[runtime 的 load_geometry/apply_dense_overlay/构造函数](../../src/wam_rl_policy_v2/wam_v4_runtime.py)。

## 已存在 BC 与官方 PPO 的边界

现存 scratch current-only BC：

`outputs/ptv3-shared-wam-v2/teacher-fit-generalization-20261015-r1/current_relative-cosine_tail/step24000.pt`

该文件约 16 MiB，`ref2dex.full-teacher-MLP.v1`；delta0 721 **pad 至 1442**，MLP 1024/1024/1024/512，输出 12 维 PD 编码（前六减 q，手指为绝对 PD），TRAIN 统计，未初始化官方 actor 权重，**没有 τ 输入**。离线 val wrist mean 约 49.5 mm，不代表抓取已成功。[训练源码](../../../ptv3-shared-wam-v2/tools/run/probe_full_teacher.py)、对应目录 manifest/result。旧 `TeacherInputStudent` 只接 raw1442、固定 24000，并总是 clamp 到 [-5,5]，不能直接当 current-only checkpoint 的正确部署 adapter；必须按 packet 的 input_normalization/arm 原样重建。[部署类](../../../ptv3-shared-wam-v2/src/ptv3_shared_wam_v2/teacher_native.py)。

因此最小可评估方案仍需一个新的 Inspire BC adapter，或直接对小 TRAIN panel 拟合新的 native-command12 BC。τ 条件首层可新增权重并用零初始化保持旧 current-only 初始行为，但必须如实记录这是迁移初始化；之后再用**冻结 WAMv4 预测 τ**补拟合，才能称为本次方法的 warmstart。

DExplore 官方 PPO 入口为 `third_party/DExplore/dexplore/run.py` → `build_alg_runner()` → `learning.dexplore_agent.DexploreAgent`/rl_games，默认配置 `data/cfg/train/rlg/inspire.yaml`：MLP [1024,1024,1024,512]、horizon64、mini_epochs6、LR2e-5、gamma.99、GAE.95、clip.2。可借用 `CmResidual/tools/dexplore_ddp_rank_bootstrap.py:main(agent_class=...)` 注入小 adapter/subclass，复用上游 update/storage；无需用 UniDex PPO 强行控制 DExplore。环境 obs/action spaces、normalization、BC均值接口要随新合同适配。默认官方 reward 包含参考跟踪和能量/交互项，不是仅抓到最终位置；第一 Probe 应保留并注明该协议。[官方入口](../../../../../third_party/DExplore/dexplore/run.py)、[官方配置](../../../../../third_party/DExplore/dexplore/data/cfg/train/rlg/inspire.yaml)、[已用 bootstrap](../../../CmResidual/tools/dexplore_ddp_rank_bootstrap.py)。旧 `trajectory-policy/tools/run/train_trajectory_ppo.py` 是另一种 history→trajectory PPO/冻结 R，**不是**本次可直接续用的 native-action PPO。

## 最小可执行推进顺序

1. 只取上述一个 TRAIN lift 的已存 raw data 和已知 reference path，准备一个物体的小几何 manifest。先离线构造实测 H4、scheduler goal、v4 sample，验证点序/anchor/query/mask/30Hz；不创建全量预测缓存。
2. 用 graspenv Torch2.4.1 的独立 WAMv4 worker 做真实预测，native DExplore 继续固定 `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`（Torch2.0.1+cu118，Isaac 必须先 import）。复用 `trajectory_policy.wam_ipc` 的有界 JSON/Unix socket 帧（4MiB）和 peer PID 校验，新增 v4 schema，不调用旧 server 的旧 head。
3. 新建 native-command12 的小 BC 执行器，输入 current721+约定 τ+valid；用已存训练 action 和 WAM 预测的条件拟合，或由上述 current-only BC 做显式迁移再拟合。先测同一初态 BC 闭环，不把 teacher takeover 当成功。
4. 两个进程在一张可用卡做 16–32 拍工程 smoke，检查 frame/episode、实际 applied action/PD、workspace、fresh history。通过后才扩到一个完整 lift（球300/飞机431控制）；记录 object goal tracking、ever/final hold、drop、termination reason，给出终末物体和手部可视化供用户审阅。
5. 再接官方 DExplore PPO，先有限 env/2–4 个 update 验证 BC均值初始化、likelihood、reset 与参数更新，再决定小规模 warmstart-PPO 能否改善。不启动六卡大训练，不改变 GPU0–5 主任务。

上述 986 维合同、adapter、v4 IPC server 和 PPO 注入目前尚未存在，是当前工程工作；不是已经能执行的现成命令。现有 collector 的可运行命令可验证原生 runtime/data，但它只重放官方 teacher，不满足本次方法，故不以 teacher 重放替代新方法的 smoke。

## 后续原生 teacher 首动作差异：只读核查

主代理随后新增 `tools/run/inspire_native_peer.py`，以原 collector 的源码适配复用初始化、记录和 PD 审计，同时加入 WAM IPC/replay/teacher 三角色及 PhysicalGeometry/TableSupport 测量。主代理报告：单环境 `s3_airplane_lift` 新 teacher 首动作与旧 batch013 的同序列差异很大，但初始 q/hand/object/current721、参考及权重身份匹配；旧 canonical command replay 在当前单环境有保持，teacher 没有。这里仅审查调用链，未另跑 GPU，未独立复现该行为数值。

核查结果：**尚未定位确定的新 wrapper 错误**。teacher 分支 `super().get_action(obs,is_deterministic)` 正常到 DexplorePlayerContinuous → CommonPlayer（取 `obs_dict['obs']`）→ legacy PpoPlayerContinuous；确定性布尔参数位置正确，没有递归或重复包装。`restore()` 仅在 socket/replay 时跳过，teacher 仍 strict 加载 model、external RMS、AMP RMS。原 collector 和包装都调用 `install_legacy_player_compat()`，将 CommonPlayer 的 `_preproc_obs` 替换成当前固定 legacy BasePlayer 版本；该版本实际做一次 external RMS。RMS 在构造时 eval，load_state_dict 不改变 eval 状态。源码未显示此次新包装造成重复或遗漏归一化。

`PhysicalGeometry`/`TableSupport` 构造和 measure 中没有发现写 actor observation、模型权重或刚体 tensor 的操作；表面采样用局部 NumPy RNG。此静态检查不能排除底层运行时副作用。`git diff 362e18ac..HEAD` 对原 collector、兼容 helper、CommonPlayer、DexplorePlayer、base task 的相关文件为空，即这些调用链没有在旧 batch013 采集后发生可见源码改动。

当前关键证据缺口是**实际送 teacher 的 raw1442 和 normalized1442**。匹配 current721 不能单独替代这一对照，因为 teacher 使用 delta1+delta16；旧16行与新1行推理也应明确记录，在输入完全相同前不把首动作差异归因物理噪声或批大小。旧真实1442 已在 `outputs/ptv3-shared-wam-v2/full-teacher-cache-20261015-r1/train-full.npy`，可根据 manifest、train-sequence.npy、train-tick.npy 定位该序列 tick0。`ptv3-shared-wam-v2/tools/run/prepare_full_teacher_data.py` 有从同执行 body_state/reference_current 重建 delta0/1/16 的纯函数路径，旧缓存生产曾对 teacher action 做 <=2e-4 的重放门控。最小下一步应保存新 teacher 首次 raw/norm1442 对比旧行，再比较相同输入下 batch1/16/64 的 model 输出；在此之前不宣称已找到原因。

