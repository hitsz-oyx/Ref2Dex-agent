# EgoDex shared forecasting 数据准备

本 Task 实现 [架构方案](docs/user/架构.md)的 geometry-only 首版：原生手与 H4、当前非手稀疏场景、固定 32 个目标 query、连续 30 Hz 的 24 步物体伪标签及 endpoint goal。与 Shared-WAM 的对接是独立入口，不修改正在运行的三域主训练路径。

补充数据来源见 [Shared-WAM 数据集调研（2026-10-14）](docs/research/2026-10-14-shared-wam-dataset-survey.md)：优先核查 H2O / HO-Cap 的原生 RGB-D 小集，用 DexYCB 作单手对照，并列出 HOT3D、TACO、HoloAssist、EgoVerse 等的字段、接入方式和限制。本轮调研未启动新增下载或数据接入。

## 当前能执行的流程

1. 从既有下载器 `ready.jsonl` 增量索引完整 official TRAIN 配对。
2. 读取原生 2×11 手点、confidence、相机和语言；校验 MP4/HDF5 原生时钟。
3. 当前目标 mask 内选 64 个候选点，在跟踪前固定前 32 个 query 身份。
4. 已观测 H4 单独运行 depth 与尺度标定；当前几何和未来 25 帧标签分别推理。
5. CoTracker 稀疏跟踪，逐步按 visibility、depth 和目标身份 mask 检查；相机补偿后生成标签。
6. 数值准入、HDF5 分片原子发布、SQLite 恢复，以及 immutable train/dev snapshot。
7. 用新增的 `EgoDexSharedWindows` 和训练入口进入原 Shared-WAM 双头；可选与三域混合。

内置视觉后端是已有本地资产的 VGGT-Omega 和 CoTracker。它们不自动下载、安装或修改第三方仓库。目标分割首版使用明确的逐帧 mask，也可接已有分割 backend；尚未实现自动语言 grounding 或 scene H4 motion。未提供目标身份 mask 的未来步不能伪装成有效标签。

本地已有 `/home2/wyy/miniconda3/envs/graspenv/bin/python` 能运行 PTv3，但缺 h5py。本轮将 `h5py==3.11.0` 单独安装在项目 `tmp/video-data-preparation-deps/`，没有修改正在训练的环境。运行示例需设置：

```bash
export PYTHONPATH="$PWD/tmp/video-data-preparation-deps${PYTHONPATH:+:$PYTHONPATH}"
```

远端需使用同时具备 PyTorch/CUDA、h5py、NumPy、OpenCV、ffprobe、相应 vendor 依赖的工作环境；训练还需要当前 PTv3 的 spconv/flash attention 等依赖。远端环境与本地环境路径不必相同。

## 下载队列与预处理入口

先只索引，不占 GPU：

```bash
python src/task/video-data-preparation/tools/run/prepare_egodex.py \
  --ready /path/to/acquisition/ready.jsonl \
  --output outputs/video-data-preparation/egodex-geometry-run \
  --index-only
```

正式运行时传入已有资产与 mask；所有路径须换成远端实际位置。默认预算配置是 [geometry_only.json](configs/geometry_only.json)：最多 1 小时、1000 个尝试窗口、2 GiB 新产物和 5 GiB scratch，每 episode 20 分钟；stride=8、目标 64 queries、场景 2048 点。为新增预算与远端 scope 建立 Task-local 实验卡后再做非工程规模处理，遵守 Campaign 中的共享上限。

```bash
python src/task/video-data-preparation/tools/run/prepare_egodex.py \
  --ready /path/to/acquisition/ready.jsonl \
  --output outputs/video-data-preparation/egodex-geometry-run \
  --annotations /path/to/annotations.json \
  --depth-vendor /path/to/vendor/vggt-omega \
  --depth-checkpoint /path/to/omega-checkpoint.pt \
  --tracker-vendor /path/to/vendor/co-tracker \
  --tracker-checkpoint /path/to/scaled_offline.pth \
  --gpu 0 --once
```

`--once` 处理当前队列后退出；不带它则继续等待下载器的新 READY 条目，仍受时间上限约束。停止只在安全边界退出；已完成窗口发布为分片，下一次相同命令跳过已发布 tick。正式处理开始后，配置、source 和 annotation revision 变化时需新的输出目录。`--index-only` 使用独立 `raw_index.sqlite`，不冻结视觉配置；之后可在同一输出目录补齐资产和 annotations 启动真实 consumer。

每次重启对 `shards/` 做一次恢复扫描，再按 SQLite 索引查询每个 episode 的完成窗口；避免每处理一个 recording 都遍历整个 corpus。`.partial` 不进入训练，rename 后尚未来得及发布 receipt 的完整 shard 会经过逐行合同检查后恢复。失败日志保留；最多两次 transient 尝试，长期或明确质量失败进入延期/拒绝记录。

`progress.json` 和控制台记录 GPU、计时、合格窗口吞吐、队列与容量；ETA 包含初始化/队列影响，不代表稳定的大规模 throughput。默认入口是同步消费进程，**尚无脱离终端的 supervisor 或外部硬 watchdog**。数值时间检查在推理返回后执行，模型调用本身卡死时需既有长任务监督器介入；远端扩规模应将此入口交给已核验的 owned process supervisor，不以普通终端长跑替代监督。

## 目标 mask 合同

`annotations.json` 的 episode key 必须与 READY 条目的 `key` 一致，不能凭 task/member 名猜。mask 路径相对该 JSON 所在目录，非零表示前景；`hand` 包含手与前臂。示意：

```json
{
  "episodes": {
    "download-record-key": {
      "target_name": "drawer-front",
      "current_verified": true,
      "frames": {
        "3": {"target": "masks/target-000003.png", "hand": "masks/hand-000003.png"},
        "4": {"target": "masks/target-000004.png", "hand": "masks/hand-000004.png"}
      }
    }
  }
}
```

示例只展示两帧，实际 t=3 的窗口需当前及 t…t+24 每帧的目标身份 mask。它们必须描述同一部件；移动物体不能把第一帧 mask 重复当成未来分割。当前选择不读取 future masks。语言只留作 selector 的提示与 provenance，不直接进模型。

为了接入已有模型，配置可指定 `mask_backend`、`depth_backend`、`tracker_backend` 为 `module:factory`；factory 接收完整 config，实例需提供：

```text
mask.select_current(key, tick, current_rgb, observed_native)
    → {target: H×W bool, hand: H×W bool, name, current_verified, method}
    observed_native 仅包含 H4，不含任何 future hand/camera。
mask.label_masks(key, native_frame_ids, image_shape)
    → T×H×W bool，目标可见表面扣除手；无法核验时返回 None。
depth.predict(rgb, native_frame_ids, role)
    → T×H×W float32 positive optical Z，已与输入像素对应；role 为 observation/label。
    观察调用仅给 H4，标签调用给 25 帧；pipeline 分别做原生手近似尺度校准。
tracker.track(rgb_25, current_uv)
    → (25×Q×2 tracks, 25×Q bool visibility)，query frame=0，ID 顺序固定。
```

内置 Omega 当前限定 landscape、vendor preprocessing 为纯 resize；检测到不支持的 crop/pad 几何时退出，不能把错位 depth 直接 resize 后宣称正确。不同后端的轴约定可在 config 的 `depth_to_native` 指定一个已核验 4×4 proper transform，默认 identity 沿用既有 EgoDex lifting 约定。近似手点/表面标定与 rigid residual 门仍需在真实视频上审查，不能当作 metric ground truth。

## 冻结与审计

`snapshot.json` 应直接位于 corpus 根目录，以便整体迁移后保持相对 shard 路径。freeze 要求独立 train/dev sessions、相同处理 profile、没有重复 episode/tick，且所有纳入 shard 合格。

```bash
python src/task/video-data-preparation/tools/run/freeze_snapshot.py \
  --corpus outputs/video-data-preparation/egodex-geometry-run \
  --name snapshot-r1.json

python src/task/video-data-preparation/tools/audit/audit_snapshot.py \
  --snapshot outputs/video-data-preparation/egodex-geometry-run/snapshot-r1.json \
  --output outputs/video-data-preparation/egodex-audit-r1 \
  --split val --max-windows 12 --render
```

审计生成 `summary.json`，`--render` 只读原始 MP4，保存 t/+8/+16/+24 同 ID overlay。数据迁移到别的机器后，metadata 的原始 HDF5 路径可能需重新映射；纯 tensor 读取不需要 raw 文件。训练快照不会因新 shard 发布而自动增长。

## Shared-WAM 训练入口

```bash
python src/task/ptv3-shared-wam/tools/run/train_egodex.py \
  --snapshot outputs/video-data-preparation/egodex-geometry-run/snapshot-r1.json \
  --stats /path/to/train-only-norm_stats.json \
  --output outputs/ptv3-shared-wam/egodex-engineering-r1 \
  --smoke --gpu 0
```

非 smoke 需要 `ptv3-shared-wam` Task-local experiment card，以及匹配架构/统计的 `--init-shared` 完整 pretrain checkpoint；设置 `--resume` 才能复用输出，恢复必须保持 snapshot/config/world size。支持显式空闲设备上的 torchrun，多 rank 最多 6。日志记录 loss、GPU 与 ETA，保存同一 shared schema 的完整 checkpoint，工程 checkpoint 不进入科研适配初始化。

要混合三域，传 `--main3-root` 并在自己的 config 中将 `egodex_fraction` 改为例如 0.25；每 batch 至少一个 EgoDex 与一个三域样本，EgoDex 按 session→episode→window 分层抽样。三域仍使用原 category/source sampler 与原 cap，新增 collate 的共同 tensor 已用真实三域样本核对一致。source_id=3 明确标识 EgoDex；所有背景 context 不生成静止 future 标签。

首版 sampler 关注跨 session/episode 多样性，尚未实现 SSD 热 shard 下载缓存和 shard 内成批局部读取。HDF5 的有界打开句柄、spawn workers、pinned memory 与预取已实现；在 NAS 实测后再决定局部 sampler，避免先破坏抽样分布。当前 shard 为简单直接的监督数组，重复存储窗口手部标签；episode-native 数组去重属于后续存储优化。

## 已完成验证

用 graspenv 加项目独立 h5py 目录运行：

```bash
python -m pytest src/task/video-data-preparation/tests src/task/ptv3-shared-wam/tests -q
```

工程 GPU run：`outputs/ptv3-shared-wam/engineering-egodex-adapter-20261013-r1/`。合成 geometry-only 数据经真实 HDF5→双 worker→真实 PTv3→U/V 完成 4 次更新，13.27 秒，GPU1 物理显存最高观察值约 1383 MiB；两项 loss 的 PTv3/interaction/goal 梯度非零，scene→hand head 无梯度。GPU 已释放。

上述合成工程验证只检查接口、掩码、数据读取、梯度与模型执行，**不验证真实 EgoDex 的目标选择、深度、tracker 质量或大规模吞吐**。首版实现时没有连接远端；后续真实小样本试跑见下一节，仍未启动科学训练。

## 已下载真实数据的小批量试跑

2026-10-13开始、14日补齐审计：[实验卡](docs/experiments/probes/P-20261013-egodex-shared-real-processing.md)。已只读复制远端三个完整basic_pick_place配对，在本机GPU1跑真实Omega和CoTracker。三窗53.21秒，峰值分配约8GiB；一窗积木产生了真实2048 scene/32 query/H4/K24样本并通过新Shared-WAM读取，两窗观察尺度标定被拒绝。

真实小批量入口 [process_selected_windows.py](tools/run/process_selected_windows.py) 接收固定tick输入清单和annotations，逐阶段保存原始depth/track缓存及拒绝原因，适用于这类最多三窗的Probe。默认不自动扩量；通过数值门的刚性窗口才发布分片，软物体只作diagnostic。Omega的preprocess现在沿用其合法balanced/max_size默认接口。

实际scope：`outputs/video-data-preparation/egodex-shared-real-processing-20261013-r1/`；四帧轨迹预览在 `visual-audit-r1/window-000000.jpg`，全时序覆盖在 `all-track-steps-10.jpg`。该快照只有TRAIN1/VAL0，尚不足启动训练；中间遮挡/插值mask覆盖和独立米制几何仍需审查，整体结论UNCLEAR。没有在远端启动额外GPU任务。

## 远端有界批量接入（2026-10-14）

用户已授权让一个远端 GPU 转入新线路。[批量接入实验卡](docs/experiments/probes/P-20261014-egodex-shared-batch-ingestion.md)
记录 GPU0 辅助重建进程的正常停止；下载、GPU3 主重建和 GPU1/2 辅助重建继续。

[select_ready_cohort.py](tools/run/select_ready_cohort.py) 从 READY 中冻结最多50个简单、命名
刚体 pick/place episode，每session最多3段；不复制原始数据、不使用未来物体运动挑选。
[automatic_masks.py](src/video_data_preparation/automatic_masks.py) 用本地显式 Grounding DINO
tiny 和 SAM2.1 tiny 资产，在当前 RGB 定位/分割目标并扣除手与前臂；候选面积、手邻近性、
SAM质量及同分候选歧义门失败时拒绝。语言对象识别是保守小词表，暂不覆盖所有任务。
SAM2未来传播只在当前query固定后收到25帧RGB，以当前target和hand mask初始化身份。
尺度及原有几何门保持不变。

自动窗口即使数值合格，也写入 `training_allowed=false` 的待审分片；`accepted`/`numeric_candidates`
表示数值合格数量，`training_admitted` 单独计数。普通freeze和训练reader继续拒绝待审分片。
[review_candidates.py](tools/audit/review_candidates.py) 生成同ID轨迹预览；具名审阅记录明确列出
`shard,row,decision,reason` 后，可将批准窗口复制到新corpus并冻结快照，不修改原候选。

[supervise_consumer.py](tools/run/supervise_consumer.py) 在独立scope启动一个owned consumer，
每30秒记录GPU、显存、吞吐、拒绝原因、队列及ETA；外部硬限时含模型初始化。`STOP`文件
或SIGTERM触发受控退出，90秒后仍未退出只终止该supervisor创建的child session。
`fetch_mask_assets.py` 只向独立asset scope下载固定版本的模型和SAM2源码，不改生产环境。
缺少的依赖使用scope-local packages；运行时HF离线，禁止隐式补下载。

远端scope：`/mnt/ugreen_nas/storage/oyx_storage/egodex-shared-batch-20261014-r1/`。
首次预算为一个GPU、最多1小时处理/1000尝试窗口、2GiB输出和5GiB scratch。
批量启动和审查结果以实验卡为准；本节不表示全量下载集已处理，也不自动启动训练。

首个cohort已完成50段/441窗：74数值候选，1127.44秒；其中两个VAL候选经RGB审查均
误选目标，未批准。r4抽查16窗批准5，独立 `approved-r4/` 为TRAIN5/VAL0。
这与旧profile的r1 TRAIN7不相加、不合并重复窗口。第二个不重叠50段cohort已在GPU0
运行，代码30b1e9a2，scope `batch-r5/`；当前输入解析保留全部目标修饰词、延期透明物体、
使用修订颜色门。所有自动候选仍需视觉审查；不能把74或后续候选数量称为训练准入数量。
NAS曾把新资产和split清单暴露为mode000；只恢复本任务这些确切输入，并补显式0644及
启动前metadata检查。失败r2/r3保留，不影响原始数据或既有教师资产。
