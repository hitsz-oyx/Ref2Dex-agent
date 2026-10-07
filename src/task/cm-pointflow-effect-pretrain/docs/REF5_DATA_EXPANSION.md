# Ref5 数据扩充工程试跑

本轮按 [ref5](user/ref/ref5.md) 优先接入已有直接 3D 数据，再验证 EgoDex
的原生手轨迹与免费 ObjectForesight 物体流程。执行记录见
[实验卡](experiments/probes/P-20261007-ref5-data-expansion.md)。这不构成数据增量改善模型或策略的科研结论。

产物根目录：`outputs/cm-pointflow-effect-pretrain/ref5-data-expansion-20261007/`。
所有新工具在 `tools/run/ref5_data_expansion/`，现有模型、训练配置及另一个 agent
的代码保持原状；没有切换分支、提交或推送。外部数据及已有 Python 环境仅作读取。

## 已有 GRAB / ARCTIC

当前可用产物为 `native3d-r6/`；`native3d-r5/` 的运动分类存在错误，保留用于追溯。
20 条 GRAB 加 20 条 ARCTIC，共 26,755 个重叠 anchor 窗口：

| 来源 / split | 窗口数 |
| --- | ---: |
| GRAB | 3,650 |
| ARCTIC | 23,105 |
| train | 17,998 |
| val | 3,667 |
| test | 5,090 |

窗口包括 4 帧历史和 24 帧未来，保持 30 Hz、米单位、静止坐标系，以及
左右手 wrist + 5 knuckle/MCP + 5 fingertip 的 11 点表示。GRAB 120 Hz 每四帧采样；
ARCTIC 原始物体网格和 root translation 从 mm 转成 m。ARCTIC 的可动物体
拆为 top/bottom 两个刚体，使用官方 raw0=top、raw1=bottom 和负 z 轴 articulation。
每个 part 固定 512 个面积采样点及法线；记录原始 triangle 和 barycentric 以核验来源。

`native3d-r6/audit.json` 为 `ENGINEERING_PASS`。核验覆盖所有 40 条序列的
抽样 pose/geometry、每个 canonical cloud 的全部采样点、输入 SHA256、各 split
的 loader/collate 与解析刚体未来对应。最大刚体对应误差约 `7.30e-8 m`。
按来源内 subject 划分 train/val/test，subject 无交叠；不声称 object 无交叠。

审查确认旧版仅比较当前与第24帧，漏掉中途移动后返回的窗口：2,596 个旧静止窗口中
546 个实际超过运动阈值（train391、val99、test56）。修复后检查全部24个未来帧，
任一帧相对当前位移>2mm 或旋转>0.02rad即为运动；分类为 moving24,705 / static2,050。
总窗口数、身份、SE(3)、手轨迹和几何均保留。r6 用新索引及序列 metadata 引用 r5
的不可变数组和 canonical geometry，因此需保留 r5。`category_repair.json` 记录每个
改动和原始数组/索引校验值。增强后的审计检查全部26,755个分类；旧审计未覆盖这一项，
其通过结果不能证明分类正确。已有 EgoDex 接受窗口复查未发现同类误标。

`NativeWindows(root, split)` 复用已有 WM tensor 接口。它核验真实 30 Hz timestamps，
仅在内存中为父类 OakInk 专用 frame-ID 断言提供四 tick 网格；磁盘上的 frame IDs
和 source frame IDs 保留真实身份。使用方式：

```python
import sys
sys.path.insert(0, 'src/task/cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion')
from native_data import NativeWindows
data = NativeWindows('outputs/cm-pointflow-effect-pretrain/ref5-data-expansion-20261007/native3d-r6', 'train')
sample = data[0]
```

没有捏造 program 标签，`program_available=False`。索引只有 near-hand moving/static
两类（1/2），类别 0 为空。现有 `balanced_indices` 要求三类都存在，因此不能直接
把这些产物接到现有训练启动器；后续需明确 native source sampler 或多源组合方案。
本轮只交付数据和相同 tensor 接口，不修改 live training，也不声称已完成多源训练集成。

重跑转换需选择新 output 目录，不能覆盖已有产物：

```bash
CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion/prepare_native.py \
  --output outputs/cm-pointflow-effect-pretrain/ref5-data-expansion-new/native3d
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion/audit_native.py \
  --input outputs/cm-pointflow-effect-pretrain/ref5-data-expansion-new/native3d
```

## EgoDex

国内镜像探测超时后，通过用户指定代理读取官方 test ZIP 的 byte ranges，
首批下载 20 对 MP4/HDF5（40 文件），传输约 60 MiB；随后用 30 个 HDF5 做
低成本 metadata 筛选，再取 12 对候选视频/标签，新增传输约 36.4 MB。
随后又取 20 个 metadata 和红色积木/蓝色骰子 2 对片段作刚体候选替换，约 13 MB。
当前共 34 对原始片段；没有下载整个 17.3 GB archive。
ZIP inventory、ETag、member CRC 和本地 SHA256 保存在 `egodex/download_manifest.json`。
这些是候选片段，含毛绒玩具、可动夹具等，后续物体 pilot 需筛选明确刚体，
不能把 20 条候选都算作通过验证的刚体轨迹。

`egodex/hands/` 已保留 20 条、2,834 帧、2,294 个手窗口的原生手点、confidence、
camera-to-origin 和 intrinsics；`review_grid.jpg` 提供手点投影人工检查。
缺失 confidence 不替换成 1；无效点保留 mask，并以零值填充坐标。
手目录本身仍为 `HAND_ONLY_OBJECT_PENDING`，不包含 object-effect 窗口；
后面的物体试跑另存于 `object-pilot/`，不覆盖这份原生手产物。
官方 test 仅用于工程验证，`training_allowed=False`，不得用于训练或 checkpoint 选择。

原生手点已在 stationary ARKit origin 中。后续物体 camera-space pose 的转换为
`T_origin_object = T_origin_camera @ T_camera_object`；不能与 SpaTracker 自己估计
的 world frame 混用，也不能直接从每帧 moving camera 下的差分构造 effect。
保留原生 30 Hz。SpaTracker 的 `fps` 参数实际用于字符串 video 输入的帧步长，
不表示采样率；本轮传入逐帧 tensor，以 source IDs/timestamps 核验真实时钟。

新增 12 条的原生手/相机已转换到 `egodex-rigid-candidates-r1/hands/`，有 896 个
手窗口。另两条替代片段的手转换在 `egodex-rigid-replacements-r1/hands/`。
`upstream/pilot20-candidates.json` 记录 20 个物体候选、目标名称、来源 SHA256 和
审核图；蛋形玩具、塑料奶精杯已用清楚的红色积木、蓝色骰子替换，保留排除记录。
这份候选表不等于 20 条接受的物体轨迹。

### 原版 ObjectForesight 后续流程（当前路线）

组件已在独立 `.venv-object` 部署，模型和 CUDA 算子检查通过。使用原仓库提交
`d1dc6b04daf6429e9c81f7238d689abac8d1f2a9` 的 step8/step10 核心函数，
手工选择物体/干净帧，并保留 EgoDex 原生手和相机；这些是输入适配。
step8 已实际执行手机、卷尺的多图重建、Gaussian 解码、网格处理和 2048px UV
纹理烘焙，分别约 102 秒和 106 秒。FLUX 权重未部署，按上游可选增强的默认
fallback 继续。不能将此描述为原始十阶段完整自动流水线。

运行控制审查已修复截止异常被原版 `except Exception` 吞掉的问题：截止时间现在抛出
独立的 `BaseException` 子类，直接退出整个运行并记录 `TIMED_OUT`，不会继续候选循环。
审计入口 `audit_native.py` 在任何第三方导入前主动禁用字节码缓存，调用者不需要依赖
环境变量才能保护只读外部源码。新增回归覆盖平移/旋转后返回、真实原版候选循环和
外层运行状态、外部源码缓存写入拦截；共16项测试通过，另覆盖子进程退出状态优先于残留 manifest 的情况。

| 试跑 | 输入 / 核心 | 结果 |
| --- | --- | --- |
| 手机 `upstream-r3` | 较早网格 + 原版 step10，640×360 | 49/64 有效帧，17 个窗口，工程审计通过 |
| 手机 `upstream-r4` | 原版 step8 完整纹理网格 + step10，640×360 | 初始化最佳 IoU 0.337 < 原版要求 0.4 |
| 手机 `upstream-native-r2` | 完整纹理网格 + 原版 step10，1920×1080 | 无执行异常，但最佳 IoU 0.382 < 0.4；无接受轨迹 |
| 卷尺 `upstream-r2` | 较早网格 + 原版 step10 | 最佳 IoU 约 0.33；未初始化 |
| 卷尺 `upstream-native-r2` | 完整纹理网格、原始分辨率输入 | 已准备，待 GPU 空闲后运行 |

原版初始化、尺度选择、评分、跟踪和重初始化逻辑保留。兼容修复只在自己的进程内
执行：PyOpenGL 版本及 GLdispatch 加载、sm86 编译、精确分块网格直径计算、
预算日志多传一个参数、调试视频帧宽不一致、候选图像变换分批以限制显存。
1080p 下原版把 888 个候选一起展开，曾申请单个 20.58 GiB 张量而显存不足。
分批版本保留所有候选及顺序；33 个候选的逐字段对比最大误差为零，见
`upstream/native_batch_parity.json`。这不证明估计位姿准确。

GPU3 空闲后已继续蓝色方块 `cube6/original-r3`，完整纹理网格约153秒完成，
深度估计和精化完成，原版初始化选择 IoU0.454 的候选（门槛0.4），跟踪正在运行。
此前默认简化后约48.7万面导致 XAtlas UV 展开过慢；使用上游 `to_glb` 的原有
简化参数控制为约5万面后，UV25秒、纹理烘焙26秒。没有替换重建/UV/跟踪算法。
CPU采样几何比较显示主体尺寸相近，但简化后不完全封闭且有局部偏差；该比较不是
真实几何精度证据。保留失败运行和 `simplification_geometry_review.json`。
当前仍未完成连续窗口审计，完整纹理路线尚无接受的 clip；20–50条 pilot仍未完成。
已有环境、系统驱动和另一个 agent 的训练均未修改。

复跑原版跟踪需先 source 本目录工具中的 `runtime_env.sh`，然后用
`upstream_object.py stage` 将 native-calibrated depth/masks/mesh 适配为上游目录，
再用 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0` 启动
`upstream_object.py run`。成功后用 `export_upstream_object.py` 转回 native schema，
再调用 `audit_object.py` 和 `replay_object.py`。所有目录需为新目录；阶段 manifest
记录原版源码、适配工具 SHA256 和实际参数。各工具提供 `--help`。

蓝色方块的串行执行入口（GPU 占用时返回77，保留 `RESOURCE_WAIT` 并在模型加载前
退出；空闲后可用同一命令续跑）：

```bash
source src/task/cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion/runtime_env.sh
"$REF5_OUTPUT_ROOT/.venv-object/bin/python" \
  src/task/cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion/run_original_clip.py \
  --inputs "$REF5_OUTPUT_ROOT/upstream/next-cube-inputs.json" \
  --output "$REF5_OUTPUT_ROOT/object-pilot/cube6/original-new" --seconds 1800 \
  --mesh-diagnostics --mesh-face-budget 50000
```

### 较早轻量诊断（保留，不作为完整流程结果）

较早轻量路线只生成顶点颜色网格，并自写了尺度/跟踪循环。后续已切换到上述原版
核心调用，下面的产物和命令仅保留用于复查；不能把两版重叠窗口合并计数。

实际完成两条物体工程试跑，未完成 ref5 建议的 20–50 条刚体批量转换：

| 片段 | 位姿检查 | 接受窗口 | 当前判断 |
| --- | --- | ---: | --- |
| 卷尺 `basic_pick_place/101` | 最后一次 64 帧中 23 帧通过；遮挡导致断轨 | 0 | 暂不接受 |
| 手机 `basic_pick_place/1` | 64 帧中 57 帧通过，后段失效帧排除 | 24 | 数据合同工程检查通过，估计精度未验证 |

手机最终产物为 `object-pilot/phone/pose-r1/`，原始帧48..111，接受窗口的
anchor 原始帧53..76。每个窗口保留4帧历史和24帧未来；24个窗口互相重叠，
不是24个独立 episode，全部为 near-hand moving 类。物体可见尺度约0.161m，
native-hand 深度标定比例0.50824，首8帧176个标定样本的 relative IQR0.143。
接受帧 silhouette IoU 中位数0.662、深度 RMSE中位数0.00277m、最近原生手点
距离中位数0.01665m。这些与估计深度/分割的一致性不能替代真实6DoF误差。

`object_audit.json` 核验 source clock、原生手点与 confidence、原生相机组合、
canonical triangle/barycentric、所有24个窗口的 loader/collate 及未来刚体点对应，
最大点对应误差约2.98e-8m；`motion_audit.json` 核验窗口内跳变，记录类别及运动分布。
`pose_review.jpg` 和 `native_world_replay.mp4` 提供轮廓、原生手投影及 stationary
origin 三维回放。后段有失效帧，本轮结果不足以判断跨物体转换成功率或重建精度。
当前先保留这条工程样例，继续扩到20–50条前需完善清楚帧初始化、遮挡和质量筛选。
全部 EgoDex 产物仍禁止训练/模型选择，不能附加到 `native3d-r5/train`。

FoundationPose 初始化原先对10000网格点做约1亿对距离，导致数分钟 CPU 开销。
本轮 wrapper 仅在自己的进程内替换为凸包精确直径和512行分块距离；同一网格约
0.05s，保留失败日志/调用栈。尺度归一化同时改用真实网格直径，避免包围盒对角线
把卷尺缩小约四分之一。最终6项语义/几何测试通过。此前失败目录不用于训练。

复跑手机样例时，每个 `--output` 必须选择新的目录，并逐段检查 GPU3 是否空闲。
先从仓库根目录 source 运行环境；下载独立在线完成，模型阶段只使用本地权重。

```bash
source src/task/cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion/runtime_env.sh
ref5_tools=src/task/cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion
ref5_raw="$REF5_OUTPUT_ROOT/egodex/raw/test/basic_pick_place"
ref5_pilot="$REF5_OUTPUT_ROOT/object-pilot/phone"
# 已有 mask/mesh/depth 的来源与 SHA256 在各阶段 manifest 中；逐段工具均提供 --help。
"$REF5_OUTPUT_ROOT/.venv-object/bin/python" "$ref5_tools/track_object.py" \
  --root "$REF5_OUTPUT_ROOT" --video "$ref5_raw/1.mp4" --hdf5 "$ref5_raw/1.hdf5" \
  --mesh "$ref5_pilot/mesh-r1/mesh.ply" --masks "$ref5_pilot/masks-r1/masks.npz" \
  --depth "$ref5_pilot/depth-refined-r1/depth.npz" \
  --sequence egodex_test_basic_pick_place_1 --object-key egodex_phone \
  --output "$ref5_pilot/pose-new"
"$REF5_OUTPUT_ROOT/.venv-object/bin/python" "$ref5_tools/audit_object.py" \
  --input "$ref5_pilot/pose-new" --hdf5 "$ref5_raw/1.hdf5" \
  --depth "$ref5_pilot/depth-refined-r1/depth.npz"
```

完整重建所用命令参数：`track_masks.py --point 904 1000 --box 768 879 1018 1080`；
`reconstruct_mesh.py --frame 40 --steps 25`；`estimate_depth.py --start-frame 48 --frames 64`；
再运行 `refine_depth.py`。默认 FoundationPose register5/track2 iterations，最多8次
registration。接受门槛 IoU≥0.25、depth RMSE≤max(0.025m,0.15×物体直径)，
当前手物距离<0.05m、窗口内单步位移≤0.15m/旋转≤0.6rad及连续有效时钟/手/物体。
它们是工程筛选阈值，不是正式精度标准。
