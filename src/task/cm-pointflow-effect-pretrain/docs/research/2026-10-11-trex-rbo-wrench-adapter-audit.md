# T-Rex 与 RBO wrench/触觉时序适配审计

日期：2026-10-11。Task：`cm-pointflow-effect-pretrain`。目的：沿
`docs/user/ref/ref10.md` 主线核实 T-Rex 与 RBO 是否能提供真实的时序触觉、力矩和
物体动力学监督，并给出一个不加载官方权重、不下载全量数据的最小适配 Probe。

本次检查使用了本地只读克隆：T-Rex `ba4047ce3e9f3d2013d60de600e136897e90ccb8`，
RBO `b191d0cd3a076757fc60b8e1feed8bf5d0bd1103`。T-Rex 只从数据集
`zekaiwang/trex_dataset` 拉取了 `meta/info.json`、`meta/stats.json` 和
`meta/tasks.parquet`；数据集 API revision 为
`8b478400a9167d59ede0fdbfea616cb5e9791885`，没有拉取 episode 视频、checkpoint 或官方
模型权重。RBO 只取了索引、一个带 FT 的 42 MB 交互包和 66 KB 的 FT 传感器模型，均放在
`tmp/repo-audit-20261011/`，不进入 Git。审计对应代码提交为 `64ca4cf`。

## T-Rex：可用合同和真实缺口

T-Rex 的 metadata 给出 5,464 episodes、5,473,459 frames、5,370 tasks、30 Hz，全部
在 `train` split；data chunk 约 100 MB，video chunk 约 200 MB。官方 quickstart 的
`fetch_episode(..., max_gb=...)` 会先根据 episode 指针估算大小，因此可以按 episode
做有上限的拉取，而不必同步整个约 1.5 TB 发布集。

每帧的数值合同如下：

| 字段 | 形状/频率 | 实际含义 | 适配备注 |
|---|---:|---|---|
| `observation.state` | float32 `(58,)` | `[L arm 7, L hand 22, R arm 7, R hand 22]` 关节位置 | 关节空间，不是物体点或末端 SE(3) 目标 |
| `action` | float32 `(58,)` | 同一布局的目标关节位置 | 可作为动作条件；不要误读成力控制量 |
| `observation.tactile_force` | float32 `(60,)` | 两手、每手五指、每指 `(Fx,Fy,Fz,Mx,My,Mz)`，顺序 `left/right × thumb/index/middle/ring/pinky` | 可拆成 `(T,2,5,6)`，是每指局部 F6 信号 |
| `timestamp` / `frame_index` | float32 / int64 | 30 Hz 时间和帧号 | 以 timestamp 建窗并检查间隔，不只按行号对齐 |
| `observation.images.head_left`、`left_wrist`、`right_wrist` | RGB 640×360，30 Hz | 头部和双腕视角 | 可作为视觉条件或 RGB-to-tactile 辅助输入 |
| `tactile_*_raw_*`、`tactile_*_deform_*` | 灰度 tactile/deformation 视频，30 Hz | 每个指尖的原始图和形变图 | 视频为无损编码；可延后，只先用 F6 |

metadata 的 `observation.tactile_force` 统计显示各分量为有符号浮点数，力分量量级约
十几到四十，矩分量量级约小于一；但是 `info.json`、`schema.py` 和 quickstart 没有
声明 SI 单位或传感器到物体坐标的变换。README 将它称为 six-axis wrench，不能据此把
数值自动标成 N/Nm，也不能把每指局部 wrench 相加后称为物体 COM 合力。适配器必须保存
`force_unit_status: unspecified`，在没有额外标定时只报告 raw-F6 或 train-only normalized
误差；若后续取得单位说明，应在 manifest 中显式记录 `force_unit`、`torque_unit` 和
坐标系。

T-Rex 的几何链路比原先预期更完整：`hardware_code/camera/camera_params.yaml` 给出
640×360 的相机内参、OpenCV optical-frame 约定，以及 wrist `T_ee_cam` 和 head
`T_base_cam`（平移以米表示）；`dataset_quickstart/robot.py` 和 vendored URDF 可由
58D state 做 FK。文件中同时明确 wrist CAD 外参只有厘米级精度，且 Sharpa 手 mesh
投影存在已知错位。因此它可支持“相机 + 机器人关节 + 指尖触觉”的时序表征，却不能
直接生成经过物体标定的 CHORD object-COM wrench。它没有当前 PointWorld 的 object
point/scene-flow 标签。

### T-Rex 最小 adapter/manifest

第一版只拉取一个或少数 episode 的 parquet 和必要视频，使用 metadata revision
`8b478400a9167d59ede0fdbfea616cb5e9791885`，不使用 T-Rex checkpoint。建议输出每个
window 一个不可变 manifest 和 NPZ/Parquet：

```text
source: trex_dataset
revision: 8b478400a9167d59ede0fdbfea616cb5e9791885
episode_index, task_index, object/task split
fps: 30
frame_index[T], timestamp_s[T]
state[T,58], action[T,58]
tactile_f6[T,2,5,6], tactile_valid[T,2,5,6]
rgb_refs/head_left,left_wrist,right_wrist
camera_K[T or static,3,3], T_ee_cam, T_base_cam
force_unit_status: unspecified
force_frame_status: fingertip-local / object-frame-unavailable
source_file_hashes, selection_reason
```

最小 Probe 先固定 `H=4,K=8`（约 0.27 s），以 task/object 分组隔离 fit/held，比较
persistence、state/action history 和 state/action+F6 history 对未来 F6 的 masked MAE；
可另外报告每指 force magnitude 与 contact proxy，但不能称为 CHORD wrench。若它改善
的是共享 tactile encoder，而不是物体点流，应将它记录为 auxiliary representation
结果，不能直接升级为 PointWorld 动力学结论。T-Rex 路线**值得下一步实现**，但应在
本地 pressure 合同之外单独建 arm，避免把 F6 单位和 EgoTouch `[0,1]` 压力混在一起。

## RBO：实际带 wrench 的对象/视觉时序合同

RBO 官方仓库和数据索引确认总计 358 段 RGB-D 交互，78 段有力/力矩传感器；索引字段
还给出 object、背景、光照、相机运动、是否只含内部交互、是否有 FT 传感器和录制日期。
这不是只有网页描述：本次按索引选取 `cardboardbox26_o.tar.gz`（Zenodo record
1036660，大小 41,813,416 bytes，MD5 `0d069d0fb8e04c9b73a3729eb283b1c7`）并解压审计。

该序列的实际文件合同为：

| 文件/流 | 实测内容 | 对 adapter 的意义 |
|---|---|---|
| `camera_rgb/*.png` | 295 帧，640×480，文件名带 epoch timestamp，约 30 Hz | 视觉 anchor；必须按 timestamp 与 100 Hz 流近邻匹配 |
| `camera_depth_registered/*.txt` | 299 个 480×640 深度阵列，有限值约 0.452–2.68；约 30 Hz | 数值与场景尺度一致，看起来是米，但 CSV 没有独立 `depth_unit` 字段，manifest 应标为 `inferred_meters` 并保留审计状态 |
| `camera_*_camera_info.csv` | ROS CameraInfo，640×480、`plumb_bob` 畸变、K/P 矩阵、frame id | 可反投影 RGB-D；不能丢弃 distortion 或 frame id |
| `cardboardbox_joint_states.csv` | 1,041 行，约 100 Hz，`j_0_1` 关节位置 | 物体内部 revolute 状态；URDF 给出关节轴和米制几何 |
| `rb_poses_array.csv` / `cortex_marker_array.csv` | 1,041 行，四个刚体/marker pose，frame `/map` | 刚体姿态、marker 到物体模型的链接 |
| `tf.csv` | `/map` 到 `TransparentBoxWith4Markers`、`ftSensor`、相机 frame 的时间序列 TF | 提供世界、物体基座、传感器和相机间的变换链 |
| `ft_sensor_netft_data.csv` | 1,044 行、约 100 Hz；`field.wrench.force.{x,y,z}` 和 `torque.{x,y,z}` 六列 | 真实时序 FT 目标，原始 `frame_id` 是 `myframe` |
| `ftSensor.urdf` | `myframe` 是 FT measurement reference；`ft_frame`/`ft_frame2` 给出固定几何变换 | 可从 `myframe` 旋转/平移到 `ftSensor`/`/map`，但仍需处理工具负载 |

RBO visualizer 的坐标轴标签是 `F [N]` 和 `T [Nm]`，ROS 消息字段也是 wrench；这使
N/Nm 成为合理的来源解释，但数据合同没有单独的单位元数据，第一版仍应在 manifest
中写 `force_unit_source: visualizer+ROS_Wrench`，而不是把它当作已经完成的标定证明。
更重要的是，`ft_sensor_netft_data.csv` 的 frame id `myframe` 与 TF 中的 child
`ftSensor` 名称不同。`ftSensor.urdf` 可以解释这两个 frame 的固定关系，但 adapter
必须明确应用该 URDF，而不能把六个数直接当作 `/map` 或 object frame。

`cardboardbox` 模型配置 `configuration_2017-08-02` 给出 base marker
`/TransparentBoxWith4Markers`、link IDs `{1: rb0, 4: rb1}`、mesh、一个 `j_0_1`
revolute joint、joint origin/axis 和 marker link。该 URDF 没有质量、COM 或 inertia
标签。RBO README 也明确说，要把 FT sensor 的交互 wrench 提供给新数据，必须同时知道
传感器与物体之间工具的质量、COM、inertia 和 pose。因此从 sensor wrench 变成“物体
COM 净 wrench”仍缺 payload/dynamic compensation；没有这一步不能宣称是 CHORD 的
object-COM net wrench。可以安全使用的标签层级是：

1. `wrench_sensor_myframe`：原始六维传感器序列；
2. `wrench_ftsensor`：使用 `ftSensor.urdf` 固定关系后的传感器模型 frame；
3. `wrench_object_geometric`：使用 `/map` 与物体基座 TF 做旋转/lever-arm 变换的几何
   结果，必须同时带 `payload_compensation: false`；
4. `wrench_object_net`：只有取得工具质量/COM/inertia 和传感器 pose 后才允许生成。

### RBO 最小 adapter/manifest

建议先实现一条 42 MB 级别的 bounded adapter，不遍历全部 78 段：

```text
source: rbo_articulated_objects
zenodo_record: 1036660
interaction: cardboardbox26_o
archive_md5: 0d069d0fb8e04c9b73a3729eb283b1c7
object: cardboardbox
configuration: 2017-08-02
anchor_hz: 30
state_hz: ~100
streams: rgb, depth_registered, camera_info, tf, rb_poses, joint_states, ft_wrench
world_frame: /map
object_base_frame: /TransparentBoxWith4Markers
ft_sensor_frame: ftSensor
wrench_measurement_frame: myframe
position_unit_status: inferred_meters
joint_unit_status: radians (URDF/ROS convention)
force_unit_source: visualizer+ROS_Wrench; preserve raw values
payload_compensation: false
source_file_hashes, timestamp_fields, nearest_match_tolerance_s
```

实现步骤应是：

1. 读取所有 CSV 的 `%time` 与 `field.header.stamp`（纳秒）并检查单调性、重复和跨度；
   以 RGB/depth timestamp 做 30 Hz anchor，对 FT、joint、TF 使用最近邻且记录误差，不能
   按数组下标硬拼。
2. 用 CameraInfo、depth finite mask 和 TF 生成相机坐标点；用 object URDF、刚体 pose
   和关节状态把 `rb0/rb1` mesh 采样成固定 `N=512` 点，输出 object/world 点集及
   `flow[t,k,3]`。这一步才与当前 PointWorld 的点动力学合同相接。
3. 用 `ftSensor.urdf` 把 `myframe` wrench 变换到 `ftSensor`，再用 `/map` 和
   object-base TF 做几何旋转/lever-arm 变换；保存三个层级的 wrench 和 valid flags，
   禁止静默生成 net wrench。
4. 输出每序列 `points[T,N,3]`、`point_valid[T,N]`、`flow[T-1,N,3]`、
   `wrench_sensor[T,6]`、`wrench_object_geometric[T,6]`、`wrench_valid[T]`、
   `timestamps/frame_ids`、相机 K/D/T、URDF/config revision 和所有源文件 hash。
5. 以 object/interaction 分组做 fit/held，不按相邻帧随机切分。最小 H4/K8 或 H4/K10
   Probe 比较 point-flow persistence、point-flow+sensor-wrench head、
   point-flow+geometric-wrench head；先只报告 sensor/geometric wrench auxiliary
   可学性和点流 masked error。只有 payload compensation 已验证，才可以再做 CHORD
   风格 object-COM wrench 目标。

RBO 这条路线**值得优先于盲目拒绝**：它是目前找到的最小公开数据合同，同时给出物体
模型、RGB-D、姿态、关节和实测 FT。限制是单个解压包会占约 1.7 GB（深度文本冗余很大），
而总数据成本和老式 ROS CSV 解析仍需受限；adapter 应先只处理一个或两个短序列，不得
一次下载全部 78 个 FT 交互。

## 与 ref10 主线的决策

这次核查改变了“其他仓库都不能解决”的判断：

* T-Rex 已经解决了高频 per-fingertip F6、无损 tactile 视频、机器人状态/动作和相机
  标定的公开数据合同；它适合 tactile temporal auxiliary，不提供物体点流或 CHORD net
  wrench。
* RBO 已经解决了小规模 object-aware RGB-D + articulated pose + measured FT 的公开
  合同；它适合先验证“点云动力学 + wrench auxiliary”是否可学，但必须保留
  `payload_compensation=false` 边界。
* 两者都不需要加载官方视频/模型权重。它们只提供数据和公开解析代码；最终 claim 仍需
  当前 Task 的 matched `wrench-on`/`wrench-off` 训练和评估。
* 最便宜的下一步是实现 RBO 单序列 adapter 的文件级合同和 30 Hz 对齐 smoke，随后做
  一个短 GPU Probe；并行保留 T-Rex metadata-only/F6 adapter。若 RBO 对齐或单位验证失败，
  失败应归因于 adapter/证据不足，不能写成 wrench 方法被 ref10 否定。

## Primary sources

* [T-Rex repository](https://github.com/ZhuoyangLiu2005/T-Rex)
* [T-Rex dataset card](https://huggingface.co/datasets/zekaiwang/trex_dataset)
* [T-Rex paper](https://arxiv.org/abs/2606.17055)
* [T-Rex camera parameters](https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/hardware_code/camera/camera_params.yaml)
* [RBO repository](https://github.com/tu-rbo/articulated-objects)
* [RBO dataset page](https://tu-rbo.github.io/articulated-objects/)
* [RBO Zenodo record](https://zenodo.org/records/1036660)
* [RBO downloader](https://github.com/tu-rbo/articulated-objects/blob/master/scripts/rbo_downloader.py)
* [RBO visualizer](https://github.com/tu-rbo/articulated-objects/blob/master/scripts/rbo_visualizer.py)
