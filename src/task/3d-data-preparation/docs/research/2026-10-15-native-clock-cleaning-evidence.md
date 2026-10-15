# 新来源原生时钟与清洗准入证据

日期：2026-10-15。用途：给新清洗 manifest 的 `source_fps`、`stored_fps`、`resampling`、`clock_status`
提供可执行依据。只读本地已有数据/metadata、实际转换代码和原作者论文/源码；没有训练、仿真或下载新数据。
固定采样频率 + 保留原生 frame id 足以认证固定帧钟，无须把所有来源都升级为纳秒时间戳。

## 1. 可立即使用的分来源判定

| Source ID | 来源 | 可认证的时钟 | 新 manifest 建议 | 当前处理决策 |
|---|---|---|---|---|
| 4 | UniDexGrasp 实际执行轨迹 | 本地 simulator dt=1/60，原始记录200帧；审计后每2帧选择，compact已是100帧30Hz | `source_fps=60, stored_fps=30, selected_stride=2, clock_status=verified_execution_clock` | 可继续；不要对 compact 再做 stride2 |
| 5 | TACO | 原论文 §3.1 明确 cameras 与 mocap 30Hz；当前2317序列保留连续 native frame ids | `source_fps=30, stored_fps=30, clock_status=verified_fixed_rate_frame_ids` | 可继续；保留原始 ID offset，不要求不存在的纳秒标签 |
| 6 | HOT3D | 原论文图像30fps；本地有 native timestamp、目标grid及residual | `source_fps=30, stored_fps=30, clock_status=verified_timestamp_resampling` | 可继续；须过滤 QA失败、重复/倒退native timestamp、超残差窗口 |
| 7 | HO-Cap | 原作者 v4 §8.2 说明 ROS bag 按timestamp同步；当前native meta和公开loader未给原始fps或数组→bag时间映射 | `source_fps=null, stored_fps=30, clock_status=unverified_nominal` | 暂保留已有DEV隔离；不阻塞其他源 |
| 8 | DexYCB | 原论文 §3.1 明确8views同步30fps；本地meta num_frames+labels帧号生成完整数组 | `source_fps=30, stored_fps=30, clock_status=verified_fixed_rate_frame_ids` | 可继续；缺失label保留mask，不压缩时间 |
| 9 | H2O | 原论文 §4.1 明确30fps；本地分段保留连续 native frame ids | `source_fps=30, stored_fps=30, clock_status=verified_fixed_rate_frame_ids` | 可继续；跨缺口/episode不得拼窗 |
| 10 | HOI4D | 原论文 §4 明确15fps×20s，官方decode同为15fps；本地1251条300帧且duration20s，另75条冲突/缺证据 | 一致子集 `source_fps=15, clock_status=verified_fixed_rate_consistent_subset`；其他 `conflicting/unresolved` | 可先认证1251条native15；30Hz接口要显式插值和插值mask；另75条隔离 |

这是工程时钟认证范围，不是几何精度、标签质量或预训练收益认证。字段名是建议，由清洗实现统一落地。

## 2. 一手频率依据与本地实际合同

### 2.1 TACO：30Hz 已有直接原论文证据

[TACO 原论文 CVPR2024 §3.1，PDF第3页](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_TACO_Benchmarking_Generalizable_Bimanual_Tool-ACtion-Object_Understanding_CVPR_2024_paper.pdf)
写明其工业相机、L515与光学mocap采集系统，明确相机与mocap频率为30Hz。
这比 viewer 导出帧率或其他联合训练仓库的配置注释强，足以把固定采样合同从 nominal 升到 documented fixed-rate。

本地 `convert_taco_bulk.py` 按 native hand PKL keys 保存 `native_frame_id`，要求两侧一致、连续且长度与 object pose 相同；
现有 `taco-posemesh-full-20261014-r1` 全部2317序列、363,198帧的 frame ids 复算均连续。
两个样例 ID 分别为 `1..262`、`1..209`。当前 time 从0开始是局部时间原点，等价于 `(native_frame_id-first_id)/30`，
应将该映射写进manifest，避免把1-based ID擅自当0-based原始标签。
[本地 converter](../../tools/run/convert_taco_bulk.py)

### 2.2 H2O：30fps 是作者采集参数

[H2O 原论文 §4.1](https://arxiv.org/html/2104.11181v2#S4.SS1)
明确五台同步 Azure Kinect、RGB/depth 1280×720、30fps，并说明多相机同步延迟。

当前 `h2o-posemesh-full-20261014-r1` 的524段、67,358帧全部 `diff(native_frame_id)==1`。
分段可以从非零 ID 开始，例如一段 `0..61`、另一段 `64..127`；因此每段局部 `arange(n)/30` 可接受，
但不得将两段相邻拼接并声称原生没有缺口。官方TEST manifest继续隔离；这个时钟认证不改变split归属。
[本地 converter](../../tools/run/convert_h2o_bulk.py)

### 2.3 DexYCB：30fps + 完整 frame index，而非按有效标签重新计数

[DexYCB 原论文 §3.1](https://arxiv.org/html/2104.04631v1#S3.SS1)
明确同时录制8个同步view，color/depth均640×480、30fps。
本地复用converter读取 native meta.yml `num_frames`，逐 `labels_%06d.npz` 填数组，缺label留invalid；
100段、7,282帧的 `native_frame_id` 均连续，从0开始。可用固定30fps认证，不应要求新下载timestamp。
[复用 converter](../../tools/run/reuse_native_geometry.py)

### 2.4 HOT3D：优先原始timestamp，固定30仅描述采集频率

[HOT3D 原论文 CVPR2025 §3，PDF第3页](https://openaccess.thecvf.com/content/CVPR2025/papers/Banerjee_HOT3D_Hand_and_Object_Tracking_in_3D_from_Egocentric_Multi-View_CVPR_2025_paper.pdf)
明确 image streams 为30fps。官方 toolkit 的 masks/poses 同样以 timestamp[ns] 为键。
[官方 timestamp/mask API](https://github.com/facebookresearch/hot3d#using-mask-files)

本地 converter 保留 `native_timestamp_ns/grid_timestamp_ns/alignment_residual_ns`，目标grid从首个QA通过时间点起构造30Hz，
nearest匹配且无pose插值；重复 native timestamp 置invalid。
[本地对齐代码](../../tools/run/convert_hot3d_pilot.py)

新清洗在这个实际合同上认证：`time`表示grid时间；native时间代表被选择的真实观测；QA失败、availability失败、
重复/非单调native时间、超声明残差阈值都进入invalid或拒绝窗口。不能把被nearest重复选择的native帧当成新的观测。

### 2.5 UniDexGrasp：compact已经30Hz，不能二次下采样

本地 [replay_unidexgrasp.py](../../../video-data-preparation/tools/run/replay_unidexgrasp.py) 设置sim dt=1/60，
每控制步实际simulate `control_freq_inv` 次，保存 `dt=task.dt*task.control_freq_inv`。
[审计脚本 L95–112](../../../video-data-preparation/tools/audit/audit_unidexgrasp_replay.py)
先检查原始200帧相邻time为1/60、无内部reset，再 `indices=arange(0,frames,2)` 选择，检查相邻time为1/30。

本轮直接读取 index第一条指向的compact样本：`episode-00-env-000.npz`，100帧，首time约0.0166666675，
相邻time约0.0333333351。它已经是30Hz；若再 `[::2]` 会错误变成15Hz。
应保留现有compact time或减首time重设局部原点，均不可改变间隔；manifest记录原生60Hz和实际保留stride2即可。
[本地 index](../../tools/run/index_unidexgrasp.py)

## 3. HO-Cap：明确未确认什么，不扩大阻断范围

核验官方论文 [v4 §8.2](https://arxiv.org/html/2406.06843v4#S8.SS2)：八台RealSense与HoloLens由ROS统一采集，
图像/pose写单一ROS bag，按timestamp同步。作者提供了同步方法，但本轮没有找到在该版本中声明原始采集fps的文字。
同文 §11 的10FPS是**评估帧的子采样速率**，不能直接认证整段pose数组为10Hz或30Hz。

本地原生meta样例 `hocap-test-preflight-20261008-r1/dataset/subject_7/20231022_193809/meta.yaml`
含 num_frames=1043、相机serial与尺寸、object/side/task/subject，未含fps/timestamp。
[官方 `SequenceLoader` metadata读取](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/loaders/sequence_loader.py#L59)
同样读取num_frames等，但没有恢复数组到采集时间的映射。

当前64段、72,944帧虽都 `diff(native_frame_id)==1`，这些ID来自array index，单靠连续不能认证频率。
本轮未找到已有ROS bag/逐帧timestamp。故 source_fps 应保持null、保留 nominal30 的工程时间，不升级训练时钟准入。
若以后已有原始stream metadata/ROS timestamps可读，或作者明确对应release的采集fps，再单独补证；无需阻塞TACO/H2O/DexYCB。

## 4. HOI4D：15Hz有强原文，10秒metadata冲突需隔离

[HOI4D 原论文 §4](https://arxiv.org/html/2203.01577v2#S4)
直接说明每视频15fps、20秒；[官方 decode.py L13/L21](https://github.com/leolyliu/HOI4D-Instructions/blob/master/utils/decode.py#L13)
也将RGB/depth解码为15fps。其他仓库的“30→15”注释不能推翻作者定义。

直接汇总已有 `hoi4d-native-pose-full-20261014-r1/manifest.json`：1326条均为300个camera frame id。
其中1251条action duration为20秒，与上述声明一致；69条为10秒，与15Hz×300不一致；6条缺duration。
本轮不凭N/duration反向认证30，也不修改原始action标签去抹平冲突。

清洗可采用：

- 1251条一致片段保留native15和原生ID，记录论文+解码合同；若接入强制30Hz窗口，translation/points线性插值、rotation用SO(3)插值，保留native与interpolated mask，禁止跨缺失大gap插值。
- 69条冲突与6条缺duration先隔离。未来有原RGB视频duration/PTS和label对齐时再解决；不要求为了完整性现在下载视频。
- 15Hz双倍插值只生成中间目标，不等于获得额外真实观测；模型对H4/K24的观测与目标定义需显式声明。

以下为本地manifest可直接匹配的75条隔离 `source_sequence`，不用凭文件位置猜测：

### 69条10秒冲突

```text
ZY20210800001/H1/C1/N19/S100/s02/T1
ZY20210800001/H1/C1/N19/S100/s02/T2
ZY20210800001/H1/C1/N20/S108/s03/T1
ZY20210800001/H1/C1/N21/S99/s04/T1
ZY20210800001/H1/C1/N21/S99/s04/T2
ZY20210800001/H1/C1/N23/S104/s05/T1
ZY20210800001/H1/C1/N24/S100/s01/T1
ZY20210800001/H1/C1/N24/S100/s01/T2
ZY20210800001/H1/C1/N26/S104/s05/T1
ZY20210800001/H1/C1/N26/S104/s05/T2
ZY20210800001/H1/C1/N27/S99/s05/T1
ZY20210800001/H1/C1/N27/S99/s05/T2
ZY20210800001/H1/C1/N28/S100/s03/T1
ZY20210800001/H1/C1/N28/S100/s03/T2
ZY20210800001/H1/C1/N29/S99/s03/T1
ZY20210800001/H1/C1/N29/S99/s03/T2
ZY20210800001/H1/C1/N30/S104/s04/T2
ZY20210800001/H1/C1/N32/S108/s01/T1
ZY20210800001/H1/C1/N32/S108/s01/T2
ZY20210800001/H1/C1/N33/S100/s05/T1
ZY20210800001/H1/C1/N33/S100/s05/T2
ZY20210800001/H1/C1/N34/S108/s05/T1
ZY20210800001/H1/C1/N35/S104/s01/T2
ZY20210800001/H1/C1/N36/S100/s02/T1
ZY20210800001/H1/C1/N36/S100/s02/T2
ZY20210800001/H1/C1/N37/S104/s02/T1
ZY20210800001/H1/C1/N37/S104/s02/T2
ZY20210800001/H1/C1/N38/S100/s05/T1
ZY20210800001/H1/C1/N38/S100/s05/T2
ZY20210800001/H1/C1/N39/S99/s01/T1
ZY20210800001/H1/C1/N39/S99/s01/T2
ZY20210800001/H1/C1/N41/S104/s03/T1
ZY20210800001/H1/C1/N44/S99/s02/T1
ZY20210800001/H1/C1/N44/S99/s02/T2
ZY20210800001/H1/C1/N45/S106/s03/T1
ZY20210800001/H1/C1/N47/S104/s04/T2
ZY20210800001/H1/C1/N48/S100/s03/T1
ZY20210800001/H1/C1/N48/S100/s03/T2
ZY20210800001/H1/C1/N49/S106/s05/T1
ZY20210800001/H1/C1/N49/S106/s05/T2
ZY20210800001/H1/C5/N15/S57/s03/T1
ZY20210800001/H1/C5/N15/S57/s03/T2
ZY20210800001/H1/C5/N46/S57/s05/T1
ZY20210800001/H1/C5/N46/S57/s05/T2
ZY20210800001/H1/C5/N47/S57/s05/T1
ZY20210800001/H1/C5/N47/S57/s05/T2
ZY20210800002/H2/C5/N11/S121/s01/T3
ZY20210800002/H2/C5/N12/S121/s01/T3
ZY20210800002/H2/C5/N13/S121/s02/T1
ZY20210800002/H2/C5/N14/S121/s02/T3
ZY20210800002/H2/C5/N15/S121/s03/T1
ZY20210800002/H2/C5/N15/S121/s03/T3
ZY20210800002/H2/C5/N31/S121/s03/T1
ZY20210800004/H4/C2/N47/S16/s02/T1
ZY20210800004/H4/C2/N48/S16/s02/T1
ZY20210800004/H4/C2/N48/S16/s02/T5
ZY20210800004/H4/C2/N49/S16/s03/T1
ZY20210800004/H4/C2/N49/S16/s03/T5
ZY20210800004/H4/C5/N16/S109/s01/T1
ZY20210800004/H4/C5/N16/S109/s01/T5
ZY20210800004/H4/C5/N17/S109/s01/T1
ZY20210800004/H4/C5/N18/S109/s02/T1
ZY20210800004/H4/C5/N20/S109/s03/T5
ZY20210800004/H4/C5/N23/S109/s04/T5
ZY20210800004/H4/C5/N26/S113/s01/T1
ZY20210800004/H4/C5/N27/S113/s01/T5
ZY20210800004/H4/C5/N28/S113/s02/T5
ZY20210800004/H4/C5/N29/S113/s02/T1
ZY20210800004/H4/C5/N29/S113/s02/T5
```

### 6条缺duration

```text
ZY20210800003/H3/C13/N35/S140/s03/T1
ZY20210800003/H3/C13/N39/S140/s04/T1
ZY20210800003/H3/C13/N41/S140/s04/T1
ZY20210800003/H3/C13/N42/S140/s05/T1
ZY20210800003/H3/C13/N46/S144/s01/T1
ZY20210800003/H3/C13/N50/S144/s02/T1
```

## 5. 核验限度

固定采集fps不保证每帧都具备正确标签，因此仍须按native ID连续性、validity与episode边界筛窗。上述全量frame-id复算仅读取已有compact数组，不重复运行模型。没有把QA或坐标检查替代时钟证据，也没有要求不存在的纳秒timestamp。
