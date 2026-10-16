# 多源手部 3D 数据清洗、时钟与 Shared-WAM v2 混合准入

日期：2026-10-15。范围：只读核对数据集作者提供的 README、数据文档和 loader，再与本仓库当前
`pose-mesh` / Shared-WAM v2 合同对照。没有修改生产代码、下载新数据、启动训练或把任何新来源
升级为训练准入。结论是工程研究建议，不是数据集质量或模型收益的正式科学结论。

## 先给决策结论

1. **可以统一的是几何语义，不是原始参数格式。** 各来源先保留原始帧键/时间戳、坐标系、单位、
   MANO/关键点 profile 和 validity mask；只在 adapter 中生成当前 anchor 坐标系下的米制 hand
   points、object points 和 24 步位移。不要把 PCA MANO、axis-angle MANO、UmeTrack angles 或
   retargeted robot q 当作同一种 action label。
2. **统一 30 Hz 必须是显式转换结果。** HOT3D 有纳秒时间戳，ARCTIC 有 `misc.json` 的同步偏移，
   GRAB 明确固定 120 fps；相反，TACO viewer 的 30 fps 是导出/播放路径，HOI4D 官方解码器明确以
   `fps=15` 解码，[官方 decode.py](https://github.com/leolyliu/HOI4D-Instructions/blob/master/utils/decode.py)，
   H2O 的公开 README 给出帧文件结构但没有一个可替代所有来源的全局时钟。未核实来源保留
   `clock_status=pending/nominal`，不得伪装成 native-30Hz。
3. **混合训练的最小准入单位是序列/对象/主体分组，而不是窗口。** 先按来源官方 split 和
   object/subject/episode lineage 做 group-disjoint TRAIN/dev，再由同一 sampler 产生窗口。不能
   先随机切窗再声称跨对象或跨主体泛化。
4. **当前 TACO/HOT3D/HO-Cap 等工程产物继续保持 `training_allowed=false`。** 只有原生 clock、目标
   语义、几何 overlay、validity、split 和独立复算全部通过，才值得做一个小的 matched mixed probe；
   这不等于马上启动大规模混合训练。

## 1. 官方来源怎样处理数据

下面只列作者仓库/官方数据文档直接写明的行为。不同来源的“官方做法”不自动成为本项目的合同，
因为采样率、坐标和任务目标不同。

| 来源 | 官方接口/清洗事实 | 可借鉴部分 | 不应直接照搬 |
|---|---|---|---|
| HOT3D | hand/object/mask 以 `timestamp[ns]` 为键；官方导出的 `mask_*` 区分 hand/object availability、visibility、exposure 和人工 `mask_qa_pass`，并可按逻辑 AND/OR 合并。[官方 toolkit README](https://github.com/facebookresearch/hot3d#step6-mastering-the-dataset-and-its-api)；hand loader 保留 timestamp，并分别读取左右手 wrist SE(3) 与 MANO PCA/UMETRACK angles。[官方 hand loader](https://github.com/facebookresearch/hot3d/blob/main/hot3d/data_loaders/loader_hand_poses.py) | 原生时间戳、质量 mask、对齐残差和 source-specific validity 一起保存；用 nearest timestamp 前先过 QA mask。 | 不能把最近邻后的重复 native frame 当成新观测，也不能用某一 camera stream 的帧数推断统一 30 Hz；MANO PCA 维度不能直接当 45/48 axis-angle。 |
| ARCTIC | 官方处理链明确分为 `raw_seqs → processed_seqs → splits`：raw 中的 MANO/SMPL-X/object 是 world 坐标，processed 才转到各 camera 坐标；`meta/misc.json` 含 camera 参数和同步图像/GT 的 offset，且 split 由 sequence 名称定义。[官方 data doc](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md) | 保留 raw 与 processed 两层 provenance；同步偏移必须是 manifest 字段而非代码常数；保留 `is_valid/left_valid/right_valid`。 | ARCTIC 的 object 是 articulated，官方 7D 参数含 articulation + rotation + translation；不能压成一个刚体 SE(3) 后混入当前 rigid-object 标签。[同一官方文档](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md#documentation-on-file-formats) |
| GRAB | 每序列记录 `framerate: 120 fps - fixed for all sequences`，手是 MANO axis-angle + shape + translation，object 是 rigid mesh 的 rotation/translation，并另外提供 contact map；官方有 preprocessing 和 vertex extraction 脚本。[官方 README](https://github.com/otaheri/GRAB#contents-of-each-sequence) | 以显式 `source_fps=120` 做可逆重采样；contact 作为质量/辅助标签而不是替代运动目标；同一 object mesh identity 贯穿序列。 | 不可把 120 Hz 直接下采样成 30 Hz 而丢掉 stride、原始索引和滤波规则；contact map 是几何/接触标注，不是 WAM action，不能自动替代控制目标。 |
| OakInk / OakInk2 | OakInk 官方文档记录 frame/camera 元数据和 OpenPose hand-joint order，并对缺少图像边界关节的帧做过滤；OakInk2 toolkit README 警告 annotation 更新会改变文件数量，不能依赖旧 offset 索引。[OakInk 数据文档](https://github.com/oakink/OakInk/blob/main/docs/datasets.md#data-folder-structure)、[joint order](https://github.com/oakink/OakInk/blob/main/docs/datasets.md#hand-joint-order)；[OakInk2 toolkit](https://github.com/oakink/OakInk2/blob/dev/README.md#updates) | 任何缓存都绑定 annotation version/manifest；用稳定 sequence key 而非 tar 偏移；保留官方 subject/object split。 | 不应把单视图“关节在图像内比例”过滤规则当成 3D 几何质量门；多视角可用帧不能因单视角规则误删。 |
| TACO | 官方文件分离 `Hand_Poses`、`Object_Poses`、`Object_Models`，数据列表含 2317 序列以及 S1–S4（无/几何/交互/compound）测试定义；viewer/ffmpeg 示例以 30 fps 输出。[官方 README/data lists](https://github.com/leolyliu/TACO-Instructions#data-lists)；[官方 hand loader](https://github.com/leolyliu/TACO-Instructions/blob/master/dataset_utils/hand_pose_loader.py) | tool、target 和每个 CAD identity 显式记录；按作者 split lineage 做 group split；保存 viewer 30 fps 只是 `export_fps`。 | viewer 的 fps 不是 native timestamp 证明；固定 CAD + SE(3) 不适用于软物体或 articulated target；不能按“annotated tool first”推断所有任务的学习目标。 |
| H2O | 官方仓库提供 `pose` 模式（hand/object/egocentric pose，不含 RGB-D）；文件包含 `cam_to_world`、双手 21 点、MANO 48 pose + 10 shape + translation、`cam_to_obj` 21 个中心/角点/边中点，并公开 subject-level train/val/test 列表。[官方 H2O README](https://github.com/taeinkwon/h2odataset#download-script) | 只提取 pose-only 成员；将双手 presence bit、object class 和 camera transform 保留到 manifest；按 subject 隔离。 | camera pose 是该来源的坐标链，不能假设和 TACO/HOT3D world frame 同向；21 点不能不经 profile 映射直接截成 11 点。 |
| DexYCB | 官方 toolkit 的每帧 sample 同时给 label path、相机内参、YCB ids、被抓 object index、hand side 和 MANO betas；label 中 `pose_y` 是 object 3×4、`pose_m` 是 PCA MANO、`joint_3d` 是 camera 坐标 21 点，缺失 hand 用全 0 / joint 用全 -1 表示。[官方 toolkit README](https://github.com/NVlabs/dex-ycb-toolkit#loading-dataset-and-visualizing-samples) | 把 sentinel 转成显式 validity mask；保存 `ycb_grasp_ind` 作为 target slot；相机内参/外参只用于复核，不混进 3D target。 | 不应把 image sample 数当独立 motion 数；多 camera/重复 frame 要按 sequence lineage 去重；PCA 参数不可直接与 axis-angle 做数值拼接。 |
| HO-Cap | 官方 toolkit 目录把每 session 的 `poses_m.npy`、`poses_o.npy`、`poses_pv.npy`、`meta.yaml` 和 calibration/models 分开，标签包含 3D/2D hand keypoints、hand side、MANO pose、object 6D pose。[官方 HO-Cap README](https://github.com/IRVLUTD/HO-Cap#download-the-hocap-dataset) | reader 先核 `meta.yaml` 的序列长度、side、object id 与 pose 数组，再构造 compact 几何；保留 author-native profile。 | 官方 downloader 还会拉 RGB-D/labels；不能因为几何接口需要小就运行全量下载器；未核原始 fps 时不得写成 native-30Hz。 |
| InterHand2.6M | 官方实现明确存在 5 fps 与 30 fps 版本，并提供 world→camera 的 MANO 转换工具；加入自有数据需转换到统一 COCO annotation format。[官方 README](https://github.com/facebookresearch/InterHand2.6M#interhand26m-dataset) | 将“版本/采样率/坐标转换脚本”写入 source profile；用 world/camera round-trip 做独立审计。 | 不能把 COCO 统一格式误解成 3D 时钟或 MANO 参数统一；5 fps 和 30 fps 样本不能只靠文件名混合。 |

### MANO profile 的额外底线

官方 `manopth` layer 的构造器把 `side`（left/right）、`use_pca`、`ncomps`、`flat_hand_mean` 和
`center_idx` 都作为显式选项；在默认 axis-angle 路径关闭 PCA 时手部 pose 宽度为 45，代码还为左右手使用不同
fingertip vertex index 并将结果重排为 21 点。[`ManoLayer` 源码](https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py)
因此 adapter 必须记录完整 MANO profile（模型文件版本、side、PCA/axis-angle、center/translation
语义、单位），使用真正的 left/right layer 和固定的 21→11 semantic map；“镜像 right”或只保留一个
匿名 11 点数组都不足以证明跨源语义一致。

### 1.1 对清洗最有价值的共同模式

- **保留缺失，而不是重排时间轴。** HOT3D 用 mask 表示“该 timestamp 是否可用”；DexYCB 用显式
  sentinel；ARCTIC processed split 也带 `is_valid/left_valid/right_valid`。因此我们的 `valid` 应和
  `time/frame_id` 等长，缺帧留下 false，不能把后续帧前移填洞。[HOT3D masks](https://github.com/facebookresearch/hot3d#using-mask-files)、[DexYCB label fields](https://github.com/NVlabs/dex-ycb-toolkit#loading-dataset-and-visualizing-samples)、[ARCTIC split fields](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md#documentation-on-file-formats)
- **先用官方语义做 split，再做窗口。** TACO 提供 interaction/geometry/compound split；H2O 按 subject
  列出 train/val/test；OakInk 明确有 subject/object split；ARCTIC split JSON 按 sequence name 聚合。
  这些做法都支持“episode/主体/物体是 split 原子”，而不是随机窗口。[TACO splits](https://github.com/leolyliu/TACO-Instructions#data-lists)、[H2O splits](https://github.com/taeinkwon/h2odataset#dataset-structure)、[OakInk splits](https://github.com/oakink/OakInk/blob/main/docs/datasets.md#data-splitting)、[ARCTIC pipeline](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md#overview)
- **几何版本和坐标转换可追溯。** ARCTIC 明确 raw/processed 两阶段，GRAB 明确 preprocessing/vertex
  extraction，HOT3D 同时提供 MANO 与 UmeTrack。我们应保存 `raw_source_version`、`coordinate_chain`、
  `mano_profile` 和可逆 transform，而不是只保存最终点云。[ARCTIC pipeline](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md#overview)、[GRAB preprocessing](https://github.com/otaheri/GRAB#processing-the-data)、[HOT3D hand types](https://github.com/facebookresearch/hot3d#using-hand-annotations)

## 2. 与当前 Shared-WAM v2 的接口对齐

当前本地 v2 不是接收原始 MANO 参数的接口，而是接收“当前帧 anchor 下的 metric points + 24 步
displacement”。实现证据：

- [当前 `pose_mesh_sample`](../../../3d-data-preparation/src/pose_mesh_data/windows.py) 固定米制、30 Hz、
  H4/K24 窗口；hand 形状为 `[T,2,11,3]`，scene target 为当前 object frame 中的未来点位移。
- [当前 `shared_collate`](../../../ptv3-shared-wam/src/ptv3_shared_wam/data.py) 将 endpoint/current
  点转换为同一 frame 的 delta，并要求每对象 point count 一致。
- [v2 normalizer](../../../ptv3-shared-wam-v2/src/ptv3_shared_wam_v2/normalization.py) 只接受 TRAIN
  统计，按 24 个 horizon × XYZ 保存 mean/std；[v2 objective](../../../ptv3-shared-wam-v2/src/ptv3_shared_wam_v2/objectives.py)
  以 validity mask 计算 hand/scene loss。
- [v2 training loader](../../../ptv3-shared-wam-v2/src/ptv3_shared_wam_v2/training.py) 当前冻结来源名为
  `oakink2/grab/arctic`、source-balanced sampler 和 TRAIN manifest provenance；新 native3D 来源不能
  静默替换这份统计或 checkpoint。

### 2.1 建议的 source adapter 合同

每个来源在进入 `PoseMeshWindows` 前先产出一份**不可变 native record**，至少包含：

```text
source_id, source_sequence, episode_id, subject_id, object_instance_id, target_semantics
native_frame_id[] / native_timestamp_ns[]
source_fps, clock_status, alignment_method, alignment_residual
coordinate_profile, units, hand_profile, side_order, joint_order
hand[time,2,21(or source-native),3], hand_valid[...]
object_pose[time,object,4,4], object_pose_valid[...]
segment/reset, official_split, lineage, geometry_key, provenance
```

然后按以下顺序转为 v2-compatible sample：

1. **时钟阶段（不可逆前的最后一步）**：若有 native timestamps，建立显式目标网格；采用 nearest
   native sample 仅在残差不超过 source-specific bound 时有效，并保留 `native_timestamp`、目标网格
   时间和 residual。重复 native sample 要打 `repeated_native=true`，不能算新观测。没有可信 clock 的
   来源只生成 engineering record，不生成正式 30 Hz windows。
2. **坐标阶段**：先用 source 官方链把 hand/object 放到同一世界/相机 frame；做一次 transform
   inverse round-trip 和 SO(3) 检查，再以目标 object 在 anchor tick 的 inverse 作为 sample frame。
   不对每帧 mesh 做隐式 scale/centering；若必须 canonicalize，记录 fixed transform 并同步 pose。
3. **手部阶段**：用官方 MANO/FK 或作者 3D joints 生成 21 点，再按一个版本化 semantic profile
   映射到当前 11 点。左右手槽位固定为 `[left,right]`（即便一侧缺失也保留槽位），保留原 21 点、
   `hand_profile` 和 per-point mask；不要把 left mirror 或 PCA coefficients 当成点序转换。
4. **几何阶段**：每个唯一 rigid CAD 保存一次 metric surface/normals/query identity；每个窗口只保存
   当前 object-frame 点和未来 24 步位移。articulated object（如 ARCTIC）必须按 part/joint schema
   单独建 adapter，或在本轮明确排除，不能用单一 rigid SE(3) 伪造。
5. **窗口阶段**：只接受同一 `segment` 内的 H4/K24、anchor target 在历史和 endpoint 均 valid、且
   hand history 有足够观测的窗口。任何缺失或 clock residual 超门限都保留为 invalid/被拒窗口，并在
   audit 中计数。

### 2.2 归一化和混合采样

建议继续沿用 v2 的“**TRAIN-only、按 horizon/XYZ 的米制 moments**”，但在新增来源真正准入后重新
拟合一份版本化 stats：

- stats 输入必须是最终准入 TRAIN manifest 和固定 sampler 的重放，不读 dev/test，也不按窗口数让长
  序列或重复 camera 自动支配统计；source-balanced 的权重/计数写入 stats provenance。
- 训练 batch 中保留 `source_id`，每个 validation report 同时给 source-wise hand/scene mean/p95、
  valid-window count、clock residual 和 clipping rate；总 objective 不能掩盖某一来源全 invalid。
- 任何来源换单位、hand profile、采样率、CAD point sampler 或 target semantics，都应产生新的 stats
  identity 和 checkpoint schema，而不是复用旧 v2 stats。
- 不建议做 per-source 输出头或静默 source-specific scale；若必须做 domain adapter，应先做 matched
  ablation，且仍保留共同米制评估。否则“混合训练改善”无法区分数据量、单位和架构收益。

## 3. 训练准入前的最小审计与可视化

以下是可由现有 Task-local tools 扩展的**准入门**，不是要求现在启动训练。每个 source 先生成 JSON
和一张静态/时间图，再由独立 reader 复算关键数值。

### A. 文件与 provenance

- manifest schema、source version、license/access scope、native member list、geometry key 唯一；
  raw path 只读，compact output 不覆盖原件。
- `sequence_id + episode + subject + object_instance + official_split` 可回溯；同一 lineage 不得
  同时出现在 TRAIN/dev。
- hand/object/geometry 数量、shape、dtype、finite、quaternion/SO(3) 和 homogeneous row 全部检查；
  每个失败成员记录原因而不是静默丢弃。

### B. 时间和有效性

- 画 native timestamp/frame-id 间隔直方图、目标网格 residual、重复采样段和 valid-mask transitions；
  把 HOT3D 的 QA/availability mask 与本地 hand/object intersection 分开显示。
- 对 GRAB 记录 120→30 的显式 stride/filter；对 ARCTIC 记录 `ioi_offset`；对 TACO/HOI4D/H2O/HO-Cap
  在 clock 未证实前图上标 `nominal/pending`，不能只显示“30 Hz”。
- 计算速度/加速度的 robust percentile，标出孤立跳变、长 gap、姿态 quaternion sign flip；异常只触发
  review 或 invalid，不自动平滑覆盖原标签。

### C. 空间和语义

- 每个 source 至少渲染 3 个 current/future overlay：hand 21/11 点、CAD surface/query、object pose
  轴；再做一张 endpoint residual/hand-object distance 曲线。若采用 camera/world 转换，显示两次
  inverse 后的 max error。
- 统计单位/尺度（hand span、object extent、anchor translation）和 CAD bbox-to-mesh ratio；发现
  数量级异常时先查 unit/pose chain，不用 normalization 掩盖。
- 检查 left/right occupancy、11-point semantic mapping、target slot 与官方 object id；tool/target、
  multi-object 和 articulated part 必须在图例中可见。

### D. Shared-WAM v2 接口

- `shared_collate` round-trip：current + target → delta → reconstruct，误差按 float tolerance 报告；
  hand/scene masks 不得在 collate 中被广播成全 valid。
- 独立 NumPy reader 与训练 reader 对同一 8 个窗口复算 anchor transform、future point delta、valid
  masks；任何差异先修实现，不能把负结果归因于模型。
- smoke 只验证 tensor/gradient/finite 和 source-wise metric，不把单 seed loss 或一段 overlay 说成
  科研收益；正式 mixed validation 另行建立 matched control、多 seed 和预定义门。

## 4. 可借鉴做法与明确不借鉴做法

### 直接借鉴（在本项目语义下仍需适配）

1. HOT3D 的 timestamp-keyed mask 与 QA mask 组合：把“有标注”“可见”“人工 QA 通过”拆成不同位，
   使过滤原因可解释。[HOT3D mask API](https://github.com/facebookresearch/hot3d#using-mask-files)
2. ARCTIC 的 raw/processed/split 分层和显式同步 offset：把坐标转换、时钟校准和 split 作为三个可审计
   阶段，而不是一个黑箱 converter。[ARCTIC data pipeline](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md#overview)
3. H2O 的 pose-only 下载模式：几何/pose pilot 先不带 RGB-D，符合当前节省存储和尽快发现接口错误的
   目标。[H2O download modes](https://github.com/taeinkwon/h2odataset#download-script)
4. DexYCB 的 `target object index`、camera metadata 和缺失 sentinel：转换成明确 `target_slot` 与
   `object_goal_valid`，而不是从未来位移反推目标。[DexYCB labels](https://github.com/NVlabs/dex-ycb-toolkit#loading-dataset-and-visualizing-samples)
5. TACO 的官方 sequence lists 与多种 generalization split：把 interaction/geometry/compound
   分组保留到本地 manifest，而不是只做随机 train/dev。[TACO data lists](https://github.com/leolyliu/TACO-Instructions#data-lists)
6. GRAB 的固定 native framerate 与 contact annotation：作为 source profile/辅助审计输入，不将其
   直接改写为机器人动作。[GRAB sequence fields](https://github.com/otaheri/GRAB#contents-of-each-sequence)

### 本项目明确不借鉴

- 不把“作者 viewer/ffmpeg 输出 fps”当作原生采集时钟（TACO/HOI4D 尤其危险）。
- 不把不同 MANO parameterization 的数值拼接、平均或共享 normalization；统一到 points 后才进 v2。
- 不把 articulated object 用单个 rigid transform 近似后宣称同一 scene target；ARCTIC 的官方 7D
  articulated field 是反例。[ARCTIC object format](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md#documentation-on-file-formats)
- 不把 retargeted kinematic q、contact proximity 或单次 replay 当真实执行 action/抓持成功；它们只能
  作为独立 profile 或辅助证据，直到 native controller round-trip 通过。
- 不把多视角图片/窗口计数当独立 motion 数；应按 episode/sequence lineage 去重，并在 source-wise
  metrics 中报告有效窗口。
- 不为追求统一而删除原始 frame id、时间戳、invalid 区间、坐标 transform 或 license provenance。

## 5. 对当前路线的下一步建议（不启动训练）

1. 远端 Shared-WAM v2 结束后，先对每个已转换 source 运行同一套 **8-window engineering replay**：
   独立 reader、shared collate round-trip、静态 overlay、clock residual 和 source-wise validity；
   该 probe 只回答“接口是否忠实”，不回答模型优劣。
2. 对 clock 已证实且目标语义明确的 source，生成一份新的 mixed TRAIN manifest 草案，但仍保持
   `training_allowed=false`；先 dry-run source-balanced stats，检查每个 horizon 都有有效观测且 dev
   完全未进入 moments。
3. 若任何 source 的 clock/target/geometry 门失败，保留工程产物并修 reader；不要用插值、随机切窗或
   per-source normalization 让门“通过”。若连续多个 probe 没有改变路线，再按根级 Decision Checkpoint
   记录停止/转向决定。
4. 只有上述审计通过后，才设计 matched mixed probe：同一 v2 架构、同一 seed/预算、source-balanced
   sampler、source-wise metrics 和预先定义的 improvement gate；远端长训仍需单独资源/状态审查。

## 6. 证据边界

- 这些官方 README/loader 证明的是**字段含义、作者提供的 split/处理路径和可复现接口**，不证明每个
  标注都无误，也不证明混合训练一定提升。
- 当前已有本地 native3D 审计已经指出 TACO 原生 clock、部分来源目标语义和跨源准入仍待核实；本笔记
  不追溯改变那些状态。[Task README](../../README.md)、[Task STATE](../STATE.md)、[已有原生来源调查](2026-10-14-geometry-storage-survey.md)
- ARCTIC/H2O/GRAB 等数据的 license 和 MANO 访问条款必须随实际授权复核；“官方代码公开”不等于可以
  把数据或模型复制到仓库/对外发布。[ARCTIC license notice](https://github.com/zc-alexfan/arctic#license)、[GRAB license](https://github.com/otaheri/GRAB#license)、[HOT3D license/download instructions](https://github.com/facebookresearch/hot3d#step-2-sign-up-and-get-the-download-links-file)

**研究判断（截至本笔记日期）**：最稳妥的共同接口是“source-profiled native record → 可审计的
30 Hz/nominal adapter → 当前 object-frame 的米制 11-point/geometry 24-step displacement”，而
不是统一原始参数或强行统一采样率。当前可以继续做本地清洗、可视化和独立复算；在准入门全部满足前，
不应把不同来源混进 Shared-WAM v2 的正式训练或把工程通过写成科学支持结论。
