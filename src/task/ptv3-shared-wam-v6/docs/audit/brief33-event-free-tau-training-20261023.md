| 结果解读表：读数 / 可能结果 | 含义 | 决定 |
|---|---|---|
| 固定 τ 面板前 8 帧 vs 旧 B1（9.357826 mm），母序列聚类 paired CI | 短期预测变化；B1 是旧配方参考，多项条件来源同时变化，不能归因为事件消融 | 改善且 contact / overall 无明确退化才进入执行器验证；跨 0 则不宣称优势 |
| contact window vs 旧 B1（19.834665 mm），同一固定面板 / 聚类 CI | 接触附近预测是否改善，不能单独代表 approach 或任务成功 | 与前 8 帧、整体联合判断，不以 contact 单项放行 |
| 整体 vs 旧 B1（23.409973 mm），同一固定面板 / 聚类 CI | 完整预测时域的可比读数；旧 B1 仅作旧配方参考 | 不新增基线；保留分阶段结果，尤其 approach（B1 29.897742 mm） |
| 执行器内实际任务进展 / 完成率改善、持平或退化 | 模型误差是否转化为任务进展，τ 读数本身不保证成功 | 改善才支持下一阶段；持平或退化不宣称部署收益；本轮不新开执行器实验 |
| 训练发生 OOM / 非有限 loss 或 gradient / 磁盘告警 | 运行或基础设施失败 | 停止并报告，保留现场；不静默重启、不换参数、不消耗第二次训练 |
| 本次：生产事件接线 FAIL、curated reader FAIL、继承 goal cache gate FAIL、GPU 分配 FAIL、磁盘 PASS | 配方政策 / 原型冻结完成，生产启动条件未齐；没有新 τ 面板或执行器结果 | **BLOCKED_PREFLIGHT，训练 0 updates，无启动进程**；保留全部 6,000 窗口，不以旧 endpoint 或 GT 事件填坑 |

## 冻结配方（先于代码与启动落盘）

Experiment: `P-20261023-tau-event-free`；run_id: `tau-event-free-20261023-r1`。
承接 brief31 / brief32 的冻结结论，不重复信号普查。Gate A 已失败，默认选择训练与部署统一 unknown/invalid 事件。该选择不把 unknown 当作已知 NONE，不删除或伪造调度器计划、反馈状态与执行器控制信号。

| 事件概念 | 真正模型张量与改动 | 不变项 |
|---|---|---|
| type | `event_type`: int64，占位 0，必须同时 `event_valid=False` | 原三类编码与参数形状保留；0 不表示已知 NONE |
| delta30 | `event_delta_steps`: float32，统一 0 | 30 Hz 时间基准与 K24 不变 |
| valid | `event_valid`: bool，统一 False；模型五维 event features / event residual 全零 | 沿用 `event_features` → `EventGoalResidual` 的 invalid mask |
| known-NONE | `valid & (type == 0)` 恒 False | 不增加一个假“无事件”的有效输入 |
| recovery | 不是独立模型输入列；不能通过事件包恢复有效状态 | 调度器 recovery 控制逻辑不变，不用 GT 补 runtime 状态 |
| release | 不再有有效 RELEASE 输入；type 的 2 编码保留 | 释放控制与接触预测监督不变，不移除未来真值标签 |

训练与推理必须使用同一 unknown 包常量及同一模型 invalid mask；仅配置声称 unknown、仅训练随机 dropout、或仅运行时默认缺省事件，均不构成合同通过。历史 checkpoint 未声明此策略时保持历史配方身份。新配方须在真实 collate 后与部署入口证明事件包一致。

完整保留 curated r7 的 **6,000** 窗口与其有序 `(source_id, sample_id, window_start_frame)` 清单；母序列、物体、阶段组成及 sampler 不作为本轮研究变量。保留手物轨迹监督、contact geometry loss 与接触预测目标、模型架构及执行器接口。goal 保持 `spatial_path_waypoint`，不能将观测未来 endpoint 作为 waypoint 冒名输入。原 `tau_curated_data_only_v1.json` 的 status 不变。

旧 B1 只作旧配方参考。新配方改变事件条件来源，不能把新旧模型误差变化解释为“事件消融的因果效应”。source-ceiling A0 不消费事件包，其测量语义不因本选择改变。

## Preflight 与启动记录

状态：**`BLOCKED_PREFLIGHT`，训练 0 updates**。本文件的配方和判读表先于代码写入，随代码冻结于 `a63be3dda80ed9f5b50768a3c561cbba1a532fdc`。该提交已推送；随后只补入本报告结果与小型审计 receipt。

GPU 7 属执行器，4/5/6 属 mutant 审计；不能利用占用间隙抢卡。必须核实空闲且可用资源；全满则排队。磁盘必须实测，不删除历史产物。训练时若触发预先冻结的内存、磁盘或数值阈值，停止并报告，不自动恢复。

### 已冻结的身份

新政策文件为 [tau_event_free_v1.json](../../configs/tau_event_free_v1.json)。它是 **FROZEN 政策**，不是伪装成旧 recipe schema 的可执行训练配置；生产 preflight 未通过，所以 `training_allowed=false`。没有修改旧 config 的 status。

| 身份 | SHA-256 |
|---|---|
| curated r7 manifest | `5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f` |
| 有序 6,000 窗口元组清单 | `56b9fac09e35e41c3922041e8794232fdeb4056b1e23838928afcd31fc49aaa3` |
| 新事件自由政策文件 | `21402da989dc32a9c0c9aa506e9c4da551b4ed964a7f3509922a67da40918801` |

清单是 [window-tuples.json](brief33-event-free-20261023-r1/window-tuples.json)，元组口径 `[source_id, sample_id=[sequence_id,slot,tick], window_start_frame=tick]`，严格沿用 r7 `selection.train_window_rows` 顺序，**6,000 条且无重复**；不重新抽样或删除窗口。此处 tick 是做出预测的当前帧，不是 H4 第一帧。配方未改变 1,064 母序列、61 几何实例或阶段组成。其余上游身份与旧 config 原状态均在 [preflight.json](brief33-event-free-20261023-r1/preflight.json) 中。

### 同一事件包 / mask 的证据与生产接线缺口

新增审计工具 `tools/audit/freeze_event_free_tau.py:31–53` 只有一个 `unknown_packet()`，调用现有 `events.event_packet('NONE',0.,False)`。训练原型的 `event_free_collate()` 和部署原型的 `event_free_predict()` 共用它；后者直接调用真实 `OnlineV6Runtime.predict()`，没有第二套编码器。两侧最终共用 `events.py:20–55` 的 `event_features` / `EventGoalResidual` invalid 路径。

CPU 探针对真实 `V6Windows` 的九个来源各取一个 curated 窗口，经原来的 collate 后记录如下；完整 sample_id、reader index 和 dtype 已写入 receipt。所有行新事件均为 `(type=0, delta30=0, valid=False, hz=30)`，不是 known-NONE。

| source | 旧 type | 旧 delta30 | 旧 valid | 新 valid |
|---|---:|---:|---|---|
| 0 | 1 | 87 | True | False |
| 1 | 2 | 0 | True | False |
| 2 | 0 | 0 | True | False |
| 4 | 1 | 10 | True | False |
| 5 | 1 | 12 | True | False |
| 6 | 1 | 249 | True | False |
| 8 | 1 | 7 | True | False |
| 9 | 2 | 17 | True | False |
| 10 | 1 | 77 | True | False |

实测四个事件张量 shape 均 `[9]`，dtype 分别 int64 / float32 / bool / int64；模型 event features `[9,5]`，非零元素 **0**，valid **0/9**，known-NONE **0/9**。训练原型与真实 `online()` 产生的事件包逐位相同，全部非事件字段（包括 goal、手物监督、dense geometry GT、contact / part target）与原 sample 的 collate 结果逐位相同。

新增 **13 tests passed in 2.17s**。部署测试通过真实 `predict()` / collate / SharedWAMV6 的 CPU geometry seam 取模型入口；特意把事件 encoder 的输出 bias 设为 7，排除零初始化让检查失去判别力。直接调用历史 runtime 时有效事件仍会到达模型；加原型适配后 residual 精确为零，goal 不变。这证明原型可实现所选语义，也证明 **仅冻结配方元数据不会自动修好生产入口**。

生产缺口有明确代码证据：

- `src/ptv3_shared_wam_v6/data.py:41–70,72–77` 仍按 sidecar 和随机 dropout 返回事件，未引用新政策。
- `src/ptv3_shared_wam_v6/online_runtime.py:160–162,249–252` 允许传入 event 覆盖默认 unknown；未按 checkpoint 的新政策强制 mask。因此不能把“默认缺省事件是 unknown”写成所有部署调用已统一。
- `tools/run/train_pretrain.py:125–126,207–208,288–301` 仍用原 V6Windows、Steps / 原 collate 到模型，未接入 curated sampler 或本原型；旧 recipe reader 实测长度 **207,579**，不是 6,000。

本 brief 的文件范围明确只列“配方/文档/审计工具与其 tests”。因此仅在该范围内提供可复用的共同适配原型，**未越界修改包内生产 reader / runtime 或训练入口，也未声称完成生产统一**。这是实现 preflight 的失败项，而不是要求重新裁决事件自由路线。

### 继承 goal gate 的方式

部署 goal sanitizer 已修复及测试通过的事实保持不变。本轮不重跑 brief31 / brief32 或 scheduler 因果普查；只读取冻结 r9 receipt，记录 `scheduler_eligible_windows=0/6000`、`scheduler_receipt_replayed=false`。`v6-goal-contract-20261023.md` 自身将部署 helper 修复与 scheduler-conditioned cache gate 分开。

九条真实 reader 样例也都满足 `goal_object_delta == scene_target[0,-1]`，与 `v5/data.py:60–61` 的现行读取路径一致。这里仅核验启动将消费的输入，**不推翻部署 sanitizer 的已通过结果，也不将 endpoint 重命名为 waypoint**。事件 mask 原型不触碰任何 goal 张量。要保持本 brief 要求的 goal 语义，训练仍需绑定同 ID 的合法 waypoint 条件；不能通过调 status、删除窗口或默默清空 goal mask 绕过。

### 实测资源与写盘预算

测量时间：**2026-10-23 05:04:00 +08:00**。环境：`dexplore_repro_py38_torch222_cu121`，Python 3.8、torch 2.4.1+cu121、numpy 1.24.4；审计 CPU-only，无 GPU 作业启动。

| 实测 / 预算 | 数值与解释 |
|---|---|
| `/home2` 可用空间 | **147,695,742,976 B（137.55 GiB）**，不是用用户提供的 139G 代替实测 |
| GPU 0 / 1 / 2 / 3 | 显存分别 15,621 / 15,697 / 15,569 / 15,539 MiB，有计算进程 |
| GPU 4 / 5 / 6 | 当次各 2 MiB / 0%，无计算进程；仍处于机主指定的他线卡组，不将空档自行视为资源移交 |
| GPU 7 | 4,884 MiB，2 个计算进程，执行器占用 |
| 可分配未预留 GPU | **0**；资源状态等待，未挂自动开训器。并且本地需保留两张空闲卡，旧三卡 DDP 入口不能直接抢走全部空闲卡 |
| 历史实测完整 checkpoint | 两个各 **1,217,116,001 B**；新 run checkpoint 尚不存在 |
| 历史留存文件增长 proxy | **2,243,923,985 B/h（2.09 GiB/h）**，按两次文件 mtime / 大小测得；它不是新 run 吞吐测量，也不包含已被旧任务轮转的文件 |
| 保守拟定上限 | **≤6,000 updates 或 14,400s，先到为止**；沿用 global batch 288 / local 96 / world 3 作预算，不暗中缩预算或改数据组成；这是未执行的预算上限，不复用旧 30,000-update 配置 |
| 最大保存次数 | 16（所有计划 checkpoint、每次 validation 都成为新 best 的并集，加 1 次紧急保存）；保留全部，不删历史产物 |
| 训练写盘预测 | **24,476,537,760 B（22.80 GiB）**，`ceil(1.25 × 实测最大 checkpoint) × 16 + 128MiB`；预留 payload 增长，预测后仍约 114.76 GiB，可用空间 gate PASS |
| 告警 / 停止 | 本 run 产物达到 32 GiB 停止；磁盘 <30 GiB 告警、<20 GiB 停止；任何 OOM / 非有限 loss 或 gradient 停止并报告，不静默重启 |

新 run 的实际 checkpoint 增长率、显存峰值、单步吞吐和 ETA 均 **null / 未测**，因为没有启动。不可拿历史 proxy 充作新的实测 preflight。未来输入与资源同时就绪后必须重新查 GPU，并在同一次训练初段实测吞吐 / 显存 / 保存增长；估计上限 4h 不应写成已经测得的预计完成时长。

### 启动裁决与最小修复

**本轮未启动唯一一次训练，0 optimizer updates，未创建 checkpoint、未新增 GPU 长任务，未改旧 config status。** Gate A 的失败和事件自由政策不再重新论证。本次失败项是启动所需实现 / 输入 / 资源未齐：

1. 将共同 unknown 包策略正式接入 checkpoint-config 驱动的训练与推理模型边界（包括 forecast-only），接入 CuratedTauSampler / 6,000 精确元组 reader。需要的包内生产文件 / 训练入口不在本 brief 所列修改类别内，本轮未修改。
2. 提供并绑定覆盖这 6,000 元组的 scheduler waypoint 条件缓存与 identity。现有冻结 receipt 是 0；若无法补足，不能在“goal 语义不变”的约束下启动。这与事件 unknown 的决定是两个不同合同层。
3. 资源移交后重新实测 allocation；旧入口要求三卡，若改成单卡梯度累积以保留 global batch，则需相应实现与恢复检查，不能仅换 `CUDA_VISIBLE_DEVICES` 冒充同一预算。

由于存在非资源阻塞，没有设置可能在 GPU 空闲时自动启动的不安全队列。训练启动 HEAD / GPU / PID / 时间 / ETA 均不存在；代码冻结 HEAD 与数据 / 政策身份已完整记录。

## 复现与写盘

```bash
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q -p no:cacheprovider \
  src/task/ptv3-shared-wam-v6/tests/test_event_free_preflight.py
# 13 passed in 2.17s

# 必须使用不存在的新目录；该工具只作 CPU 探针与 preflight，不会启动训练。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/freeze_event_free_tau.py \
  --output src/task/ptv3-shared-wam-v6/docs/audit/brief33-event-free-20261023-r1-reproduce
```

审计先声明写盘预算 **2 MiB**，实际 receipt + 6,000 元组清单共 **128,277 B**；无 checkpoint / 视频 / cache 副本。本轮 6 个文件合计 **165,695 B**（逻辑大小），磁盘分配 **180,224 B**；Git object 与他线写入不计入本任务产物。
