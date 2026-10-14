# HOI4D 原生坐标、mesh 与时钟独立审查

日期：2026-10-14；调查始于 14:01 UTC，限时 15 分钟。只读官方源码及已有原生 ZIP 的少量真实成员，没有训练、采集或修改共享代码。结论属于 reader 语义核验，不代表数据训练收益或正式准入。

本轮实际输入：`outputs/3d-data-preparation/hoi4d-native-full-20261014-r1/archives/` 中四个官方 ZIP。主要样例：

- A：`ZY20210800001/H1/C1/N19/S100/s02/T1`，ToyCar/019。
- B：`ZY20210800001/H1/C7/N31/S60/s04/T2`，Bowl/031。

## 先给实现结论

1. **直接读 output.log 的原始 4×4 就是 camera_to_world；不再 inverse。** 若用 Open3D `read_pinhole_camera_trajectory`，所得 `.extrinsic` 已是其 inverse（world_to_camera），转换手/物体到 world 才需要再 inverse。
2. 两个真实 rigid mesh 均已米制、原始 AABB 中心恰为零；原点是提供的 canonical frame。**不要减 vertex mean，不要乘 0.01，不要直接乘 JSON dimensions。** `R = Rotation.from_euler('XYZ', [rx, ry, rz])`，`p_camera = R @ p_mesh + center` 是当前有源码依据的基础映射。
3. JSON `dimensions` 可数值精确确认是局部包围盒的**全边长（米）**，不是给米制 CAD 直接相乘的无量纲 scale。官方没有提供“实际 CAD 必须 anisotropic scale 到这个 bbox”的明确映射；保留该差异，不擅自随帧缩放。
4. 帧配对用文件名转 int。JSON 内部 `frameId` 不是统一偏移：关键帧与插值帧可能混用不同编号。
5. **30 Hz 没有统一证实。官方 decode.py 明确输出 15 fps。** 两个真实样例都有 300 个 pose/LOG，但 action duration 分别 10 与 20 秒。`N/duration=30` 不能自动覆盖它与官方解码约定的冲突。
6. MANO 使用 `use_pca=False, flat_hand_mean=True, center_idx=None`；左右均返回 wrist/thumb/index/middle/ring/little 的标准 21 点顺序，left 需真正的 left layer。
7. 工程 TRAIN/dev 可以先按 **(category C, instance N)** 分组；仅用 sequence、H、camera、room 或 anchor 划分都不能防止同一 CAD instance 跨集合。

## 1. output.log 的两次 inverse：最容易抄错的链路

官方 HOI4D `prepare_4Dseg/scene2frame.py`：

```text
extr = o3d.io.read_pinhole_camera_trajectory(ex_path)
extrinsics[i] = extr.parameters[i].extrinsic
PointCloud.create_from_rgbd_image(..., extrinsic=extrinsics[int(frame_filename)])
```

但 Open3D 的 LOG reader 在读完原始矩阵 `trans` 后执行 **`param.extrinsic_ = trans.inverse()`**。随后 PointCloud factory 先计算 **`camera_pose = extrinsic.inverse()`**，再乘 camera 中的深度点。组合后是原始 LOG 的 `trans @ p_camera`。因此官方 scene2frame 最终把 foreground/background 放进与重建 static scene 一致的 world frame，不能把它传入 API 的 `.extrinsic` 直接抄成我们的 camera 点乘矩阵。

一手源码：[HOI4D scene2frame.py](https://github.com/leolyliu/HOI4D-Instructions/blob/master/prepare_4Dseg/scene2frame.py)、[Open3D FileLOG.cpp](https://github.com/isl-org/Open3D/blob/main/cpp/open3d/io/file_format/FileLOG.cpp)、[Open3D PointCloudFactory.cpp](https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloudFactory.cpp)。本次核对的是公开 current 源码；若实现换其他版本/自定义 parser，应保留一次小矩阵等价检查。

用列向量记号，设 `L_t` 为原始 LOG，`O_t` 为 JSON 中 mesh→camera 的 rigid pose：

```text
hand_world(t)   = L_t @ hand_camera(t)
object_world(t) = L_t @ O_t
query_world(t)  = L_t @ O_t @ query_mesh
```

当前对象 anchor 为 `A = inverse(L_a @ O_a)` 时，未来点为 `A @ L_t @ O_t @ query_mesh`。hand 与 object 必须都补这条相机运动链，不能只补一边。原始 LOG 不是直接的 object world pose，也不是时间戳。

真实样例 A/B 的 LOG 均有 300 个 block，header 从 `[0,0,1]` 到 `[299,299,300]`，第一个矩阵 identity；所有旋转的 `max(abs(R.T@R-I))` 分别约 `1.04e-8 / 1.07e-8`。这些能检查 parser/数值合理性，不能单独证明标注精度。

## 2. rigid CAD 原点、米制范围与 dimensions 的实际关系

官方 README 写 mesh 自身定义 pose 的 canonical frame；示范 `read_rtd` 按 center 的 x/y/z 读取 translation，按 length/width/height 读取 dimensions，rotation 使用 SciPy **大写 `XYZ`** intrinsic Euler（默认弧度）。不能把该函数返回的 rotvec 再当成 Euler 重算，也不能改用小写 extrinsic xyz。[官方 README 的 CAD / Category-level Object Pose](https://github.com/leolyliu/HOI4D-Instructions)

以下直接从已有 CAD OBJ 的 `v` 行统计，没有重新居中或归一化：

| 样例 | CAD min xyz | CAD max xyz | CAD extent xyz |
|---|---|---|---|
| ToyCar/019.obj | [-0.0494925, -0.036404, -0.0327735] | [0.0494925, 0.036404, 0.0327735] | [0.098985, 0.072808, 0.065547] |
| Bowl/031.obj | [-0.070084, -0.071103, -0.037261] | [0.070084, 0.071103, 0.037261] | [0.140168, 0.142206, 0.074522] |

来源成员分别是 `HOI4D_CAD_Model_for_release/rigid/ToyCar/019.obj` 与 `.../rigid/Bowl/031.obj`，可从上述本地 ZIP 重算。两者 AABB center 都是 `[0,0,0]`。另抽的 Mug/010.obj 范围约 13×10×9 cm、AABB center 也为零，但 vertex mean 约 `[-0.010004,0.001136,-0.000981]`；**减 vertex mean 会引入约 10 mm 的 canonical 偏移**，不能为了采样中心化而这么做后仍沿用原 pose。

相应 JSON 的尺寸：

| 样例 JSON | dimensions [length,width,height]（米） | dimensions / CAD extent |
|---|---|---|
| A `objpose/00000.json` | [0.09649472717,0.06947999058,0.07018338389] | 约 [0.97484,0.95429,1.07073] |
| B `objpose/34.json` | [0.14124340460,0.14116740776,0.07653122449] | 约 [1.00767,0.99270,1.02696] |

最小数值核验：构造 8 个局部角点 `p = signs * dimensions/2`，按 `p_camera = p @ R_XYZ.T + center`，用该 camera 的原生 K 投影，对投影 min/max 与 JSON 的 `2dBox.camera1` 比较。A 的 0/10/299、B 的 0/10/299 共六帧，最大逐边误差都在约 **1e-6–8e-6 px**。因此可以明确确认：

- dimensions 是以 canonical 三轴定义的全尺寸；center/XYZ rotation 定义该 box 的 camera pose。
- `2dBox.camera1` 在这些帧是该 3D box 的投影，不能把两者一致当成独立 RGB 对齐成功。
- 原始 mesh extent 与 box dims 有数个百分点差异；raw CAD 已物理大小，不是 unit cube。

仍未知：该差异应视作手工 bbox 与 scan 的差异，还是作者预期按 `dimensions/extent` 修正 CAD。README 没有给应用此修正的代码；公开 issue #25/#28 有其他使用者的 mesh 对齐问题，但没有足以改变这一结论的作者回复。本轮**不声称已核实唯一的 dimensions→CAD scaling 公式**。[issue #25](https://github.com/leolyliu/HOI4D-Instructions/issues/25)、[issue #28](https://github.com/leolyliu/HOI4D-Instructions/issues/28)

最小可执行选择：工程第一批保留 `mesh_metric_original + SE3` 与 box dims/extent ratio；不 per-frame scale。若需要归一化 mesh 来喂网络，保存显式、可逆的 fixed canonical transform，并把它同步补进 pose；不要悄悄换原点或 scale。真正判断 CAD 是否应修正，需要一两个真实帧的 mesh/hand overlay 或已有深度/原生 mask 的独立几何对照，box 自重投影不能决定这个问题。

## 3. 文件名是配对主键，frameId 不存在全局固定偏移

样例 A：

| filename int | JSON frameId |
|---:|---:|
| 0 | 1 |
| 1 | 1 |
| 9 | 9 |
| 10 | 2 |
| 28 | 28 |
| 34 | 34 |
| 270 | 28 |
| 299 | 31 |

这些是实际 JSON，不再只是 issue 风险线索。规律与“人工 keyframe 保留较稀疏序号、插值文件用展开帧号”相符，但这是解释性推断，不作为可跨序列推广的修复规则。每个文件保留 `native_filename_frame` 和 `internal_frameId`，按前者与 hand pickle、LOG frame block join。还需检查 padded/unpadded 名称映射到同一 int 时是否重复、同一 object ID 的时序连续性、`isEffective` 与对象缺失。

A 的 hand `0.pickle..299.pickle` 共 300 个；B 的 hand `0..298` 共 299 个、objpose 与 LOG 仍各 300。**hand 缺失不能压缩时间轴**：B 最后一帧应 invalid/拆段，不能把前面手点重排成等长 300 步或按样本数量反推 fps。左/右手用各自目录与 side，不能把两个目录直接累加成独立帧数。

## 4. 30 Hz 证据的边界

官方 `utils/decode.py` 对 RGB 与 depth 都使用：

```text
ffmpeg ... -start_number 0 -vf fps=fps=15 .../%05d.jpg/png
```

这是一手 **15 fps 文件命名约定**，比“论文 action feature 用 15 fps”更直接。官方 scene2frame 又按 `int(filename)` 取轨迹，而不是按 JSON frameId。[官方 decode.py](https://github.com/leolyliu/HOI4D-Instructions/blob/master/utils/decode.py)

真实样例的冲突：

| 样例 | LOG/objpose 数 | action.info.duration | N/duration |
|---|---:|---:|---:|
| A ToyCar | 300 / 300 | 10.0 秒 | 30 |
| B Bowl | 300 / 300 | 20.0 秒 | 15 |

两者 LOG header 都不是秒级 timestamp，pickle 也没有原生采集时间。由此可以说 B 与官方 15 fps 解码相容；A 的 action duration 与它冲突。**尚不能据 A 的比值直接断言 A 是原生 30 fps**，也可能存在 duration 标注、截取或发布版本差异。

推荐：先缓存 camera-frame hand/pose/frameID 与 LOG，不写固定 fps；按官方 decode 与 action duration 建每 sequence 的一致性标志。合同只接 30 Hz 时，没有足够证据的记录保留为 clock-pending；15 Hz 若保留在 native profile 则不冒充 30 Hz，不隐式插值。选择 `N/duration` 作为工程推断 clock 也要显式标 inferred，而非 verified。改变这项判断最便宜的独立证据是实际原视频 ffprobe/frame count 和 duration 对齐（只需已有文件或小范围 metadata，不需恢复 RGB→3D 管线）。

## 5. MANO 配置与左右 21 点

官方 HOI4D README：`poseCoeff` 是 3 global axis-angle + 45 finger axis-angle，`beta` 10 维，`trans` 在 camera frame，`kps2D` 是 21 点投影。官方示例 `ManoLayer(use_pca=False,ncomps=45,flat_hand_mean=True,side='right')`；其链接的 manopth 实现构造器默认 **`center_idx=None`**，不能换成 0。结果单位由 layer 乘 1000 输出为毫米，HOI4D 再 `/1000 + trans` 得到 camera 米制位置。[HOI4D hand 格式](https://github.com/leolyliu/HOI4D-Instructions#human-hand-pose)、[manopth ManoLayer 源码](https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py)

```text
layer = ManoLayer(use_pca=False, flat_hand_mean=True, center_idx=None, side=side)
verts_mm, joints_mm = layer(theta, beta)  # 不在这里重复传米制 trans
hand_camera_m = joints_mm / 1000 + trans
```

manopth 输出在加 fingertips 后重排为：0 wrist；1–4 thumb；5–8 index；9–12 middle；13–16 ring；17–20 little，每个 finger 为 base→tip。左右输出都采用该顺序。right fingertip 顶点 `[745,317,444,556,673]`，left `[745,317,445,556,673]`；因此用 right layer 简单镜像替代 left 不符合官方实现。

本轮真实 hand sample A/150、B/149 的 fields 为 `poseCoeff(48,float32)`、`beta(10,float32)`、`trans(3,float32)`、`kps2D(21,2,int64)`，没有 timestamp。11 点映射须继承当前 Task 的语义约定；如果使用 `[0,2,4,6,8,10,12,14,16,18,20]`，需注意 thumb 的第 2 点与其余手指第 2 点的解剖学名称不同，保留 profile 名与实际 indices，不只写笼统“11 joints”。

## 6. object-instance 分组与最小验证

工程 TRAIN/dev 建议 `group_id = hoi4d:<C>/<N>`，并映射到唯一 CAD path。C 是类别、N 是该类别 instance；`N19` 跨 C 不是同一个对象。H 是 subject、S/s 是 room/layout、T 是 task：将它们放进 group key 会把同一物体跨人/房间拆开，产生 instance leakage。不要按 anchor/window 随机划分。

还应保留官方 release/test membership；该工程分组不替代作者 benchmark protocol，也不宣称 unseen subject 或 unseen room 泛化。当前 rigid 第一批仅选实际 rigid mesh 类（原包有 Bottle/Bowl/Chair/Kettle/Knife/Mug/ToyCar），articulated 需要 part association 另核。`dataList[0]` 不是通用 target 规则：对象 ID/label 必须跟 sequence category 和 CAD 对应，多候选或 ID 跳变的段可先跳过。

最小验证顺序：

1. 取 A/B 上述各 3 帧：验证 LOG parser 与 Open3D 的两次 inverse 等价、`L@O` 组合、手/物体同一帧系；检查 finite / SO(3)。
2. 固定 mesh 和 query identity；记录原始 bbox、JSON dimensions、ratio，不做未知的 scale 修正；只对一两帧做独立 mesh/hand overlay。
3. 加一个 JSON frameId 与 filename 冲突样例，验证 join 使用 filename；一个 hand 缺最后帧样例，验证 invalid 保留时间轴。
4. A/B 的 clock 差异必须实际触发不同 pending/native profile；不能用 round 到 30Hz 让检查通过。
5. 同一 `<C>/<N>` 的所有 H/S/s/T 与所有未来窗口共享 split，保留源 frame index 与官方原始 membership。

通过这些检查可支撑“reader 忠实按明确合同生成几何”的工程判断；它不能解决仍待独立确认的 clock/CAD 对齐精度，且不构成 HOI4D 已准入正式训练或有效性的科研结论。
