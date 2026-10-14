# 视频与仿真数据处理状态、审计入口及耗时报告

核验时间：**2026-10-14 16:25 北京时间**。这是现场状态快照，后续进程状态以实时日志为准。
本报告保留本文件原有审查建议，放在末尾；下面先给出当前已执行、可追溯的结果。

## 1. 现在是否持续产出

| 数据/路线 | 当前状态 | 当前规模 | 是否是新Shared-WAM已准入数据 |
| --- | --- | --- | --- |
| 本地UniDexGrasp++回放 | **没有持续生产**；一次有界批次完成，无活跃采集进程 | 20对象、320条真实轨迹，按需3200窗口 | 暂未正式准入；只有工程provider与审计 |
| 远端EgoDex下载 | **继续运行，且进度增长** | 完成210,487个下载配对；part4下载中 | 原始输入，尚未处理/准入 |
| 远端EgoDex旧重建 | **继续运行，且进度增长**；GPU1辅助、GPU3主重建 | 主状态completed13,107、failed17,722、旧admitted_windows85,213 | 旧10Hz重建/插值路线，不能当新管线准入 |
| 远端EgoDex新Shared-WAM处理 | **没有继续消费**；r4/r5两批已完成退出 | 100个不重叠episode、922次尝试、196个数值候选 | 全部自动候选待审 |
| EgoDex批准语料 | r4独立批准集保留 | TRAIN5 / VAL0 | 缺非空VAL，不能启动正常训练；旧r1 TRAIN7不叠加 |
| Open-AoE | 已按用户要求暂停扩展 | 本地2段90秒样本，手/相机审计完成 | 未生成完整物体U/G或训练样本 |

**因此，“后台还在处理EgoDex”目前指下载和旧重建；新的EgoDex管线及UniDex都没有源源不断生产。**
新consumer使用固定cohort及`--once`，完成后退出；没有自动循环扩cohort的生产调度。
UniDex当前也没有全对象队列/supervisor连续调度，不是已开启全库采集。

### 现场核验依据

- 本地`ps`未发现`replay_unidexgrasp.py`或资产准备进程，回放supervisor为COMPLETE/CHILD_EXIT。
- 远端未发现`prepare_egodex.py`/`supervise_consumer.py`活跃进程，r4/r5均COMPLETED/CHILD_EXIT。
- 下载器PID399615，旧主重建PID360908，辅助GPU1 PID671018仍存在。
- 约5分钟两次现场采样：下载210148→210487；主重建completed13090→13107；
  辅助GPU1 published5864→5876。不能只靠PID存在或一次GPU利用率判断进程健康。
- GPU0/2当前运行v2训练，本地GPU5/7运行已有训练，连同远端1/3共六张项目卡。
  本报告及审计没有新增GPU任务、停止任务、重启生产或更改数据准入。

原始证据：[远端进程与批次状态](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/remote-status-r1.json)、
[第一次进度](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/remote-progress-r1.json)、[第二次进度](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/remote-progress-r2.json)。
远端aux与主计数可能交叠，不把各worker计数直接相加为唯一处理规模。

## 2. UniDex数据怎么看、审哪些东西

本地目录：`outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2/`。

| 文件 | 用途 |
| --- | --- |
| [trajectory-manifest.json](../../../../../../outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2/trajectory-manifest.json) | 每条轨迹的对象、scale、train/dev和目标距离标签 |
| [provider-audit-r2.json](../../../../../../outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2/provider-audit-r2.json) | 全部3200窗检查、未来隔离、spawn读取及持续距离 |
| [result.json](../../../../../../outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2/result.json)、[supervisor.json](../../../../../../outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2/supervisor.json) | 实际执行与资源、进程退出原因 |
| `episode-00/01.npz` | 60Hz原始状态、policy动作、实际动作/力/力矩/关节目标 |
| `trajectories/*.npz` | 30Hz紧凑手物状态与原60Hz动作；窗口按需生成 |
| `geometry/*.npz` | 当前完整CAD采样、normal和固定query |

首先看[相机回放](../../../../../../outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2/replay-object-002-success.mp4)与
[碗回放](../../../../../../outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2/replay-object-001-success.mp4)。相机观察到抬起后留在手边，碗观察到抬起后脱落。
视频是**Gym实际执行状态的事后三维重建**，不是Gym相机录屏；11点骨架不展示完整手网格和碰撞接触。
因此它能审大致位置和运动，不能独立证明无穿透或稳定接触。完整Gym渲染仍待补。

你需要判断：对象尺度/手物位置是否正常；是否有瞬移、明显错位；是保持、脱落、抛起还是未拿起。
程序负责检查真实30Hz抽帧、reset、固定点身份、当前anchor、缺失左手mask和对象分组。

目前官方累计距离命中119/320、官方原始末帧5cm内100/320，末0.5秒持续距离83/320；
后者TRAIN67/dev16，**都不是稳定抓持率**。成功与失败可用于真实物理预测；成功行为示范须另外筛选。
ShadowHand的22关节/24维动作不能直接成为现有Inspire动作标签。完整CAD也不能冒充相机可见表面。

重新检查provider可在仓库根目录执行；报告输出必须是新文件名：

```bash
TMPDIR="$PWD/tmp" PYTHONPATH="$PWD/tmp/video-data-preparation-deps" \
  /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/video-data-preparation/tools/audit/audit_unidexgrasp_provider.py \
  --run outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2 \
  --output outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2/provider-audit-NEW.json
```

当前provider约26秒完成CPU文件/统计审计，不运行神经网络。原始状态在warm step后开始，
首状态1/60s、原始末状态200/60s、30Hz末采样199/60s。实际报告读取真实时间戳。

## 3. EgoDex新管线怎么看、审哪些东西

远端新管线根目录：
`/mnt/ugreen_nas/storage/oyx_storage/egodex-shared-batch-20261014-r1/`。

| 批次 | episode | 尝试窗口 | 数值候选TRAIN/VAL | 实际处理时间 | 含初始化墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| batch-r4 | 50 | 441 | 72/2 | 18.79分钟 | 19.11分钟 |
| batch-r5 | 50 | 481 | 119/3 | 20.68分钟 | 21.25分钟 |

两批的配置/代码profile不同，不合并为训练快照。r1是同cohort更早profile的部分运行，
不再加到上表；r2/r3为初始化权限失败，不是额外有效数据。

每个批次先看`supervisor.json`的status/reason/returncode、`corpus/progress.json`的attempted/
numeric_candidates/training_admitted，再看`corpus/shards/*.json`和`*.h5`。
`accepted`在这里是数值候选数，**不等于训练批准数**。r4/r5训练准入均0。
`progress.json`里的GPU字段是最后运行时采样，任务完成后会保留旧PID/显存，不能当实时占用。

### 实时查看命令

在本地服务器的仓库根目录，下面命令能重新查询远端当前状态，脚本只读既有状态文件：

```bash
python3 tmp/wam-v2-remote-preflight-20261014/remote.py \
  --script tmp/video-data-status-review-20261014-r1/remote_progress.py
```

该命令使用当前已建立的SSH通道；若换机器或本项目tmp脚本不在，可登录远端后直接执行：

```bash
python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/ugreen_nas/storage/oyx_storage')
paths = [base/'egodex-full-20261012-r1/acquisition_status.json',
         base/'egodex-full-20261012-r1/reconstruction_status.json',
         base/'egodex-shared-batch-20261014-r1/batch-r5/supervisor.json',
         base/'egodex-shared-batch-20261014-r1/batch-r5/corpus/progress.json']
for path in paths:
    data = json.loads(path.read_text())
    fields = ('status', 'reason', 'pid', 'returncode', 'completed_packets',
              'completed', 'failed', 'attempted', 'numeric_candidates', 'training_admitted')
    print(path.name, {key: data[key] for key in fields if key in data})
PY
nvidia-smi
```

同一`numeric_candidates`长时间不变、supervisor为COMPLETED且没有consumer进程，说明该新批次已结束；
下载/旧重建的completed与更新时间仍增长，则它们仍在推进。不同路线的状态需要分别查看。

### 已给你生成的审查包

打开[r5审查入口：23张图](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/review-r5/README.md)。
固定抽样20个不同TRAIN episode；另列出全部3个VAL窗口，VAL仅来自1个episode/session。
每张图依次为当前、+8、+16、+24帧；数字和颜色对应同一query ID。每行保留shard、row、
episode、tick、target、原始描述，方便追溯和明确批准，不改原候选。

你重点回答三件事：

1. 点是否落在**实际被操作的物体/部件**上，而不是相邻积木、容器、桌面或手？
2. 同一ID的点在物体移动时是否仍跟着相同表面；遮挡后是否漂到其他物体？
3. 描述与实际视频是否一致，颜色/部件是否合理；描述本身错误时不要强行相信语言。

反馈用“编号｜目标正确/错误/不确定｜轨迹正常/异常/不确定｜原因”即可。
四格是初筛，中间24步未逐帧人工审完；遇到遮挡或滑点再看原始视频与全horizon。

**本次AI初筛的重要疑点**：优先查看[20 VAL](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/review-r5/20-val.jpg)、
[21 VAL](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/review-r5/21-val.jpg)、[22 VAL](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/review-r5/22-val.jpg)。
点主要留在另一片静止表面，右手却搬运旁边的积木，目标身份有明显疑点；语言描述方向也需核对。
三张均未批准。初筛记录在[ai-prescreen.json](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/review-r5/ai-prescreen.json)，
没有据此宣称整批准确率。r4两个VAL此前已被审为错目标。

因此现在拿不出20个独立VAL窗口供你看，也不能把5个VAL数值候选解释成5个可训练样本。
当前首先需要审出真实独立VAL或修目标选择；多产数字不能代替这个问题。

### 正确的准入顺序与审计工具

**待审candidate → RGB审查 → 具名逐行批准 → 复制到新approved corpus → 冻结snapshot → reader/模型检查。**
本文件原有建议的“先冻结candidate再审查”不适用于当前工具：普通freeze与reader会拒绝
`training_allowed=false`的自动候选。不得为了运行审计直接改旧shard或绕过准入。

尚未批准的语料使用`review_candidates.py --corpus ... --output <新目录>`，不是先使用
`audit_snapshot.py`。其现有默认渲染按文件/行顺序取前N，不是随机抽样；本报告的23图另做
固定seed20261014抽样及episode去重，该seed仅用于审查选择。

批准后的snapshot才使用：

```bash
python src/task/video-data-preparation/tools/audit/audit_snapshot.py \
  --snapshot <批准语料>/snapshot.json \
  --output <新的审计目录> --split val --max-windows 12 --render
```

远端运行须从包含本Task的已部署repo执行，使用已有h5py/OpenCV环境；本地纯数值读取不用原视频，
RGB渲染需要metadata指向的远端原始MP4，不能只复制HDF5 shard就要求本地工具找到远端raw路径。

## 已执行的候选数值QC

范围是r4的74窗与r5的122窗，共196个数值候选；不是所有922个尝试窗口，也不是训练准入数据。两个profile分别统计，不合并为训练快照。

### 时间、坐标与划分

| 项目 | r4 | r5 |
| --- | ---: | ---: |
| 数值候选 | 74 | 122 |
| 重复episode/tick | 0 | 0 |
| train 窗口 | 72 | 119 |
| train session | 23 | 22 |
| train episode | 28 | 31 |
| val 窗口 | 2 | 3 |
| val session | 2 | 1 |
| val episode | 2 | 1 |
| TRAIN/VAL session重叠 | 0 | 0 |
| 合同错误 | 0 | 0 |
| 原生手/当前anchor标签复算最大差 | 0 | 0 |

涉及62段原始视频：全部30Hz，视频/HDF5帧数差为0。对62段逐帧视频PTS另做检查：
dt p01/median/p99约0.033333/0.033333/0.033334秒；最大相邻dt偏差0.67微秒，无非正步长或>1.5dt间隙。
metadata中的H4及未来25帧均为连续帧号。HDF5无顶层独立timestamp字段，本次没有用不存在的硬件时间戳宣称验证录制设备无丢帧。

U的current/query/endpoint合同通过；V和H4已独立从原始手点及固定当前anchor精确复算。U没有独立物体3D真值，本次没有重新运行depth/tracker来证明绝对米制准确。

### 每horizon标签有效率

手的分母为双手22槽：单手完整有效只有50%，因此下面约67–70%不能简单解释为30%追踪失败。物体分母为固定32槽。

| horizon | r4手 | r4物体 | r5手 | r5物体 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 70.45% | 98.23% | 69.45% | 99.49% |
| 4 | 69.78% | 96.96% | 69.34% | 98.57% |
| 8 | 69.23% | 97.13% | 68.41% | 97.87% |
| 12 | 69.04% | 97.38% | 68.29% | 96.95% |
| 16 | 68.43% | 98.06% | 67.73% | 97.13% |
| 20 | 67.87% | 98.10% | 67.62% | 97.93% |
| 24 | 67.57% | 98.14% | 67.32% | 99.03% |

![逐horizon有效率](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/validity.png)

r5最差平均物体有效率约96.8%，不代表语义可靠，也会掩盖个别遮挡窗口的低谷。查看逐窗曲线与RGB仍有必要。

### 位移与scale

位移仅统计valid点，单位mm；r4/r5分别统计，不把相异窗口作为matched算法比较。

| profile / horizon | 手median/q90/q99 | 物体median/q90/q99 |
| --- | --- | --- |
| batch-r4 / h1 | 0.21/8.25/21.56 | 8.06/19.82/28.04 |
| batch-r4 / h8 | 2.01/90.68/190.21 | 8.38/20.78/98.97 |
| batch-r4 / h16 | 5.50/172.46/369.85 | 9.48/26.05/190.83 |
| batch-r4 / h24 | 10.83/213.93/436.13 | 10.58/54.72/235.24 |
| batch-r5 / h1 | 0.27/7.89/18.72 | 7.80/22.14/29.05 |
| batch-r5 / h8 | 2.32/71.09/134.49 | 9.02/25.37/46.15 |
| batch-r5 / h16 | 8.56/139.76/259.21 | 9.86/26.84/90.43 |
| batch-r5 / h24 | 18.59/181.59/338.30 | 9.98/28.18/126.68 |

![位移分布](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/displacement.png)

| profile | 观测scale min/median/max | 观测IQR min/median/max | 当前观测/标签误差median（mm） | 最大刚体median/p95残差（mm） |
| --- | --- | --- | ---: | --- |
| batch-r4 | 0.401/0.478/0.660 | 0.063/0.139/0.240 | 8.34 | 4.65/13.92 |
| batch-r5 | 0.331/0.510/0.687 | 0.019/0.132/0.246 | 7.68 | 4.83/14.57 |

![尺度及当前深度一致性](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/calibration.png)

所有候选满足现有IQR≤0.25及刚体median≤5mm、p95≤15mm等门；这是条件筛选的结果，不能估算全下载集质量。
h1物体位移中位数约8mm，需继续区分真实运动与两次depth推理的对齐偏移，不能直接把该数值视为真实一步物体运动。
完整24-horizon及逐窗scale/support/残差见[QC JSON](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/egodex-candidate-qc.json)，视频时钟见[PTS检查](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/egodex-video-pts.json)。


## 4. 按目前方案需要处理多久

这些是**当前实现的耗时**，不包括研发时间、等待GPU、人工审查或训练收益验证。
当前EgoDex为geometry-only：scene提供当前几何/normal，scene的v/a/H4位移特征仍为零；
原生手H4已实现。ref1设想的scene H4稀疏运动尚未实现，增加它后需重新测吞吐。

### 4.1 UniDexGrasp：已实测和扩量估算

20对象各16次执行：160env×2轮，320条、每条200个60Hz状态。
纯采集19.78秒；含初始化45.29秒；完整3200窗provider审计25.75秒；compact约15MiB。
此前三个官方压缩输入共608.7MB，已经下载；一次复用缓存/扫描组装20对象耗时37.14秒，
**不代表首次联网下载只要37秒**，第一次下载含重试更慢。

下面按同样20对象一批、每对象16条线性外推；单seed、每mesh一个scale，不承诺成功示范数量。

| 规模 | 轨迹数 | 纯GPU采集 | 含批初始化回放 | 全窗口CPU审计 | compact估算 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20对象（实测） | 320 | 19.78秒 | 45.29秒 | 25.75秒 | 15MiB |
| 100对象（外推） | 1600 | 1.65分钟 | 3.77分钟 | 2.15分钟 | 79MB |
| 2066 TRAIN mesh，每mesh一个scale（外推） | 33056 | 34.0分钟 | 78.0分钟 | 44.3分钟 | 1.62GB |
| 3084 TRAIN mesh-scale（外推） | 49344 | 50.8分钟 | 116.4分钟 | 66.2分钟 | 2.42GB |

2066/3084来自实际解析并排除官方TEST mesh的清单，不是把3200条目当3200个不同mesh。
如每对象100条，上表轨迹、采集和数据容量近似乘6.25，初始化不应机械乘相同倍数。

全对象先按**数小时级的采集/审计预算**考虑更合理，但上表不计全量资产解包、冷VHACD、
复杂对象重试和NAS I/O；它是规划值，不是当前排队完成时间。全库成功率未知。
现有入口最多256env、资产选择5–50个不同类别；全2066 mesh队列、每mesh多scale索引与
断点调度尚未实现，不能直接把现在的命令放后台就得到全库。所有成功/失败轨迹保留，
不无界重复采到成功。当前六卡满额，尚未启动这个扩量。

### 4.2 EgoDex新管线：每window、每视频时长与全量成本

r4/r5共922次尝试、100段视频，共9736原始帧（约5.41视频分钟）。
处理计时2368.24秒，含初始化2421.69秒：**平均约2.63秒/尝试窗口、24.2秒/这批短episode**。
50段短片实测约19–21分钟。数值拒绝也计入这个均值，不能换成“2.63秒生成一个批准样本”。
大量拒绝很早发生；若大量窗口进入完整深度/分割/跟踪，平均成本可能更高。

2026-10-14补查已有分阶段计时：r5的122个数值合格候选平均3.48秒/窗，
中位数3.38秒；上面的2.63秒是包括提前拒绝及批次开销的所有尝试均值。

| 数值合格窗口的阶段 | r5平均秒/窗 |
| --- | ---: |
| 当前目标定位与mask | 0.236 |
| H4观测深度、标定及当前几何 | 0.353 |
| SAM2未来25帧身份mask传播 | 1.115 |
| CoTracker目标点跟踪 | 0.125 |
| 未来25帧深度、3D lifting及几何检查 | 1.654 |

最后两类较重时段是未来mask传播和未来深度/几何，合计约80%；不是纯GPU kernel计时，
其中包括相应PNG/JPEG临时写入、预处理和CPU检查。r4计时也近似，完整证据见
[阶段计时](../../../../../../outputs/video-data-preparation/data-status-review-20261014-r1/egodex-stage-timing.json)。

当前anchor每8帧滑动，但每个窗口仍分别重跑H4 depth与未来25帧depth、SAM2及CoTracker；
29个depth输入帧/8帧步长带来约3.6倍的帧级输入重复，模型多帧上下文使它不是可直接兑现的
3.6倍加速。减少query到64并没有消除dense depth/分割成本。当前backend还用临时图像文件
桥接模型入口，没有跨anchor复用depth或SAM状态，属于优先让合同跑通的首版。

优化应先验证目标身份，再评估episode级复用解码/预处理和标签侧推理、因果观测缓存、
更稀疏anchor及有限batch。不能将使用未来上下文生成的depth拿来构造当前输入。
例如stride8→24只减少anchor数、窗口内部30Hz/H4/K24保持不变；约3倍减少样本量，
不是同数据量无代价的加速。本轮只核查已有计时，没有修改处理代码或启动新实验。

保持原生30Hz、H4、future24、anchor stride8时：

`N_window = len(range(3, N_frame - 24, 8))`。

stride8只控制训练窗口密度，窗口内部仍连续30Hz；不能降到10Hz再插值来冒充同合同。

| 新管线输入规模 | 尝试窗口 | 单GPU当前均值外推 |
| --- | ---: | ---: |
| 5秒视频 | 16 | 约42秒 |
| 10秒视频 | 35 | 约92秒 |
| 1分钟视频 | 222 | 约9.7分钟 |
| 1小时视频 | 13497 | 约9.85 GPU小时 |
| 1000段与试批同分布的短episode | 约9220 | 约6.7小时 |
| 当前已下载210487段，假设全部与试批同分布 | 约194万 | 约59个单GPU连续日 |

最后一行是**假设场景，不是全量ETA**：已下载包含长片、组装、透明/软物体等，当前parser与
刚体profile不能处理全部类型；我们未逐段统计全下载集可接入时长。2/4卡理想线性分摊
约29.5/14.7天，但GPU空闲、I/O和多worker扩展尚未测，不能作为实际完成承诺。

在当前数值门下，r4候选率16.8%、r5为25.4%；正确目标和批准率还要再审。
不能据此保证59天后能得到全下载集的可训练数据。实际优先级是**修目标身份/独立VAL，
再减少重叠窗口的重复depth/segmentation计算**，之后重测全量时间。

### 4.3 EgoDex下载与旧重建：单独计时

16:25快照：part1、part3完整；part2还有121个失败配对；part4为15470/44129，余28659。
两次5分钟快照下载339个配对，约1.13配对/秒、网络7.4MB/s。
若维持这段速率，**仅剩part4约7.1小时**；视频大小、网络、121个补下载会改变它。
part5/test/extra尚未在当前已展开archive状态表中完成计量，不能把7.1小时当整个下载完成ETA。

旧主重建约5分钟completed增加17（短时约195 episode/小时），辅助published增加12
（约147/小时）；计数会交叠、输入长度不同，不相加成全局产能或外推完整旧重建结束时间。
主状态当前processed_bytes约464GB，仍是旧资产；其旧admitted_windows标签不代表本Task新准入。

### 4.4 Open-AoE及当前三域

Open-AoE实测只覆盖下载与手/相机：两段共440.55MiB下载36.93秒，CPU手/相机审计7.42秒。
没有完成depth/target/tracks，因此**没有实测完整处理ETA**。现有EgoDex parser对18条描述
接受0条，须先适配atomic action与目标选择，不能直接认为它已跑完整管线。

若以后能复用同成本视觉处理，90秒视频约335窗，机械外推约14.7分钟/段；
100视频小时约41单GPU日、规划2000视频小时约821单GPU日。这只是明确的成本假设，
不是已验证Open-AoE吞吐或“2000小时都已下载可用”的声明；扩量前需要episode缓存/批处理优化。

OakInk2/GRAB/ARCTIC当前训练输入已经准备好，本轮未重跑，重用不需要再处理一遍。
它们的训练剩余时间应查训练progress/ETA，不计入这里的视频/仿真预处理耗时。

## 5. 建议先看哪些材料、下一步是什么

1. 先看r5的三个VAL疑点和20个TRAIN抽样，给出目标/轨迹判断；不要先看训练loss。
2. UniDex先看相机/碗对照，确认完整CAD与ShadowHand作为独立数据源是否符合预期；
   成功动作示范另筛，不把距离命中当抓持。
3. 自动数值检查已覆盖196候选和3200 UniDex窗；缺口是语义、独立3D真值及完整Gym接触渲染。
4. 仍无合格EgoDex非空VAL，也没完成UniDex真实模型GPU双头检查；当前没有新数据科学微调启动。
5. 本报告没有恢复新consumer或启动持续采集。下一次扩量要有队列/profile/停止预算，并重新检查六卡限制。

研究回溯：[EgoDex批次卡](../../experiments/probes/P-20261014-egodex-shared-batch-ingestion.md)、[UniDex卡](../../experiments/probes/P-20261014-unidexgrasp-local-admission.md)。


---

# 附录：原有审查方案（原文保留）

以下是此前已有的方案建议，示例统计与待执行检查不等于本次实际结果；与当前准入工具不一致处以上面的报告为准。

现在这条 **EgoDex → Shared-WAM 新管线，时间合同应该严格保持 30 Hz**。不建议为了微调再特殊改成 10 Hz、15 Hz 或做插值。

当前仓库的合同实际上已经写死了：

\[
fps=30,\qquad H=4,\qquad K=24
\]

所以一个样本是：

\[
H=[t-3,t-2,t-1,t]
\]

以及：

\[
V/U=[t+1,\dots,t+24].
\]

对应真实时间：

\[
H4\text{ 覆盖约 }0.1s
\]

\[
K24\text{ 预测未来 }0.8s.
\]

现在预处理里所谓：

\[
\text{anchor stride}=8
\]

只是说**每隔 8 个原生帧取一个训练窗口**，即约：

\[
8/30=0.267s
\]

取一个 anchor。它**不改变窗口内部的采样率**。

也就是：

```text
原始 EgoDex 30 Hz
0 1 2 3 4 5 ... 27 28 ...

anchor t=8:
  H = 5,6,7,8
  future = 9...32

下一个 anchor t=16:
  H = 13,14,15,16
  future = 17...40
```

而不是：

```text
0,3,6,9,... → 10 Hz
再插值回 30 Hz
```

仓库文档也明确写了：

> `stride` 只控制窗口密度，不改变窗口内部 30 Hz；**不先降到 10 Hz 再插回去。**

这一点很重要，因为**旧的 EgoDex VGGT/Omega reconstruction 路线确实有 10 Hz 重建/插值**。那些旧 processed 数据不能直接当成现在 Shared-WAM 的训练数据。新 `egodex-shared-batch` 管线才是按原生 30 Hz 合同做的。

---

# 你现在真正需要核查 EgoDex 的 5 层东西

我建议不要随机打开几个 `.h5` 看数值就算检查完成，而是按下面顺序。

## 1. 第一层先查时间轴

这是最先应该自动化的。

每个 episode 检查：

\[
N_{video}\approx N_{HDF5}
\]

以及窗口：

\[
[t-3,\ldots,t+24]
\]

必须是连续的原生时间步。

也就是说要检查：

\[
\Delta frame =1
\]

或者如果存的是 timestamp：

\[
\Delta t\approx\frac1{30}s.
\]

尤其不能出现：

```text
t-3 = frame 30
t-2 = frame 33
t-1 = frame 36
t   = frame 39
```

然后还宣称自己是 H4。

你可以让 AI 对整个 processed corpus 输出：

```text
episode数
视频fps分布
HDF5长度 - video帧数 分布
连续28帧window比例
异常timestamp数量
实际dt median/p01/p99
```

我希望看到类似：

```text
fps: 30
dt median: 0.03333 s
dt p01/p99: ~0.0333 / ~0.0333
cross-gap windows: 0
```

如果某些 episode 真不是 30 Hz：

> **不要在 reader 里默默插值。**

要么 preprocessing 阶段显式 resample 到 30 Hz 并保存 provenance，要么拒绝。

---

# 2. 第二层：检查 3D 坐标和 scale

这个比 FPS 其实更危险。

必须确认这几个东西都在**同一个当前时刻 anchor**：

\[
H_{t-3:t},\quad
S_t,\quad
V_{t+1:t+24},\quad
U_{t+1:t+24},\quad
G_{t+24}.
\]

尤其不能出现：

\[
p_{t+k}^{A_{t+k}}
-
p_t^{A_t}
\]

这种每一帧重新居中的伪 displacement。

正确的是固定：

\[
A_t
\]

然后所有未来都：

\[
p_{t+k}^{A_t}.
\]

当前代码设计就是这么要求的。

你要随机抽几十个窗口，检查：

\[
V_k=h_{t+k}-h_t
\]

以及：

\[
U_{k,i}=p_{t+k,i}-p_{t,i}
\]

是否数值上能精确回算。

---

## scale 必须单独审

之前 EgoDex 已经暴露过这个问题。

例如那三个真实窗口：

- 芥末瓶：scale calibration IQR 0.337 → reject
- 蓝色块：0.455 → reject
- 白色积木：通过

所以对远端大批数据至少输出：

```text
calibration support count
scale
scale IQR
current observation/label agreement median
agreement p95
rigid residual median
rigid residual p95
```

不要只看：

> “模型能生成 32 个点。”

**scale 不对，32 个点全齐也没意义。**

---

# 3. 第三层：检查每个 horizon 的标签完整率

这个我认为你现在特别容易漏。

之前白色积木那个窗口：

\[
endpoint=32/32
\]

看起来很好。

但中间：

\[
t+18,\dots,t+23
\]

valid point 数曾经掉成：

```text
13 / 8 / 10 / 12 / 19 / 25
```

所以不能只统计：

\[
\text{current validity}
\]

和：

\[
\text{h24 validity}.
\]

必须画：

\[
\boxed{
validity(k),\quad k=1,\dots,24
}
\]

例如整个数据集：

```text
          h1    h4    h8    h12   h16   h20   h24

hand      98%   97%   96%   94%   91%   87%   84%
object    96%   94%   91%   86%   80%   68%   75%
```

如果出现：

```text
object validity:
95% → 90% → 40% → 90%
```

这种很怪的曲线，就要查：

- 遮挡；
- mask propagation；
- depth hole；
- tracker；
- query 是否漂到另一个物体。

---

# 4. 第四层其实是目前最大的 blocker：目标语义

现在远端这一批**还不能直接拿去微调**。

最新 batch 情况实际上是：

### r4

441 个尝试窗口：

\[
74
\]

个数值候选。

人工审 16 个：

- 5 个批准
- 10 个拒绝
- 1 个延期

而且两个 VAL candidate 都错了。

### r5

481 尝试：

\[
122
\]

个数值候选。

但是：

\[
\boxed{training\_admitted=0}
\]

目前只是待审。

所以现在真正的问题不是：

> “远端已经处理这么多，可以直接训练了吗？”

而是：

\[
\boxed{
\text{数值合格}\neq\text{目标物体选对}
}
\]

已经实际看到过：

- 白色块指令，却跟了绿色块；
- 跟了旁边蓝色物体；
- 跟到 drying rack；
- 透明瓶子时选到邻近物体。

所以你核查时我建议**优先看可视化，而不是 loss/statistics**。

---

## 最实用的抽查方式

仓库已经有：

```bash
python src/task/video-data-preparation/tools/audit/audit_snapshot.py \
  --snapshot .../snapshot-r1.json \
  --output .../egodex-audit-r1 \
  --split val \
  --max-windows 12 \
  --render
```

它会画：

\[
t,\quad t+8,\quad t+16,\quad t+24
\]

的 **same-ID query overlay**。

你看这四张图，主要回答：

1. 点是不是落在真正操作的物体上？
2. 有没有跑到桌面/邻近物体？
3. t→t+24 的点 ID 看起来是否还是同一块表面？
4. 手和目标运动关系是不是合理？

我甚至建议你先不看模型训练。

先随机：

- TRAIN 20 个；
- VAL 20 个；
- 不同 session；
- 不同 object/task；

一共看 40 个。

如果：

\[
>90\%
\]

肉眼目标/轨迹都合理，我们再讨论扩大自动准入。

如果只有：

\[
60\%-70\%
\]

正确，直接训练会把 WAM 搞脏。

---

# 5. 最后一层：一定要用“训练 reader”再读一次

不能只验证 preprocessing 输出。

最终必须经过：

```python
EgoDexSharedWindows
    ↓
forecasting_collate
    ↓
Shared-WAM
```

重新检查。

仓库里的 `ShardDataset` 已经会强制：

\[
(fps,history,horizon,object\_points)
=
(30,4,24,32)
\]

否则 reader 直接拒绝。

并且 `freeze_snapshot()` 还会检查：

- TRAIN/VAL session 不重叠；
- 同一 profile；
- 无重复 episode/tick；
- `training_allowed=true`；
- `causal_observation=true`；
- shard 大小没变。

这一步我很赞成。

正式训练前我会要求至少得到这样的审计：

```text
snapshot:
  fps = 30
  history = 4
  horizon = 24
  object_points = 32

TRAIN:
  sessions = ...
  episodes = ...
  windows = ...

VAL:
  sessions = ...
  episodes = ...
  windows = ...

session_overlap = 0
duplicate_episode_tick = 0
nonfinite = 0
```

然后 batch 再确认：

```text
scene        [B,N,3]
features     [B,N,18]
hand target  [B,24,2,11,3]
scene target [B,1,24,32,3]
```

以及：

```text
source_id = 3
native_state 不存在
native_action 不存在
```

因为 EgoDex 只是 human forecasting supervision，不能伪造 robot control 字段。

---

# 所以微调时间上不用特殊处理

当前正确方案就是：

\[
\boxed{
\text{EgoDex 30 Hz}
=
\text{OakInk2/GRAB/ARCTIC 30 Hz}
=
\text{DExplore 30 Hz}
}
\]

这其实是件好事。

这样我们的：

\[
H4
\]

和：

\[
K24
\]

在所有域里的物理时间意义都是一致的：

\[
H4 \approx 0.1s\text{ history},
\]

\[
K24=0.8s\text{ future}.
\]

不要按数据集改变 horizon。

---

## 唯一可以稀疏的是 anchor

比如：

\[
stride=8
\]

没问题。

因为它只是降低重叠窗口数量：

```text
anchor = 27
anchor = 35
anchor = 43
...
```

每一个窗口内部仍然：

```text
24,25,26,27 | 28 ... 51
```

而不是降低物理采样率。

---

# 我建议你现在让 AI 给你生成一份 EgoDex 数据 QC 报告

在正式 finetune 前，固定输出这 6 张表/图：

1. **时间完整性**
   \[
   fps,\ dt,\ frame-count,\ gap
   \]

2. **每 horizon hand/object validity**
   \[
   k=1\dots24
   \]

3. **每 horizon displacement distribution**
   \[
   median/q90/q99
   \]

4. **scale/calibration**
   \[
   scale,\ IQR,\ observation-label agreement
   \]

5. **session/episode/split 数量**
   确认无 leakage

6. **随机 40 个 t/+8/+16/+24 overlay**
   这是目标语义最终人工 gate。

其中第 6 项现在最重要。

因为**当前远端 EgoDex 新管线最大的风险已经不是 30 Hz，而是自动 target grounding 还不够可靠。**

所以我的建议是：

> **时间上完全保持 30 Hz；现在不要做任何特殊 temporal resampling。先把远端 candidate 做成一个冻结 snapshot，然后用现有 `audit_snapshot.py --render` 抽查语义和中间 horizon，再决定哪些 shard 能置 `training_allowed=true`。**

而且就当前 r4/r5 的状态，**我不会直接拿 numeric candidates 去微调**。先把独立 VAL session 也审出一批真正批准的窗口，这比再处理几千个 candidate 更重要。
