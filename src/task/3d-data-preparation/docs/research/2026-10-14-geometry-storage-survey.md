# 原生 pose + mesh 数据的下载与空间核验

日期：2026-10-14。范围：为 `docs/user/ref/ref1.md` 的路线切换核对公开下载源、最小几何输入和空间；不下载全量数据，不注册账号或替用户接受协议，不运行训练。

**已核验的主要数字：TACO V1 的纯几何三个 ZIP 共 15.69 GB，全部解压后 26.99 GB；HO-Cap 的三个几何 ZIP 共 79.63 MB，解压后 175.11 MB，但还缺每段 `meta.yaml`。** 用户随后提供HOT3D三个授权CDN JSON，纯标注和Assets合计2,701,783,954B；HOI4D的OneDrive元数据本次返回401，不能把未知大小写成已确认的小数据。以上都是原始输入，不是紧凑训练集大小。

GB/MB 使用十进制字节；GiB/MiB 使用二进制。精确字节是本次公开服务元数据观测；估算均单独注明。

## 1. 存储结论与证据强度

| 来源 | 最小几何输入 | 压缩/原生文件大小 | 解压文件逻辑大小 | 当前证据与缺口 |
|---|---|---:|---:|---|
| TACO V1 | Hand_Poses、Object_Poses、Object_Models；MANO 模型另需已有授权 | **15,690,340,641 B / 15.69 GB** | **26,993,255,576 B / 26.99 GB** | HTTP 206 `Content-Range` 实测 + ZIP central directory；无需 RGB/D |
| HOT3D full | 手 pose/profile、dynamic_objects、metadata、质量/可用性 masks、CAD | **2,701,783,954 B / 2.70 GB** | 全量未知 | 用户授权JSON精确求和；首批两段+完整Assets下载177.09MB、解压236.32MB；不取main_vrs或MPS |
| HOT3D-Clips CAD | 33 个 GLB + models_info | **174,905,353 B / 174.91 MB** 原生 GLB | 不适用 | 官方推荐 HF 仓库 tree 元数据求和；simplified eval 版本是另一套几何，不应无审计替换 |
| HOI4D | CAD、Hand Pose、Annotations 中 objpose/action、Camera Parameters | **未知** | **未知** | 官网分别给包，但本次 OneDrive 元数据 401；Annotations 可能混装 segmentation/scene 文件 |
| HO-Cap | poses.zip、models.zip、calibration.zip + 每段 meta.yaml | **79,630,165 B / 79.63 MB**，未含 meta | **175,112,756 B / 175.11 MB**，未含 meta | HTTP 206 + ZIP directory 实测；meta 可从 subject ZIP Range 提取，不需下载 RGB-D |
| H2O | subject{1..4}_pose_v1_1.tar.gz、object.zip、label_split.zip | **本轮未测** | **未知** | 官方访问需个人信息、同意条款和 7 天 HTTP basic-auth；约 0.32 GB 只是下游实现的近似报告 |
| DexYCB | 每帧 labels NPZ、sequence meta、calibration、models | 官方整体 **119G**，每个 subject **12G**，models **1.4G** | **未知** | 官方主下载按 subject 混装 RGB-D，没有公开独立 hand/object pose 包；标签训练集可很小但首次获取成本不能略去 |
| DexTrack | GRAB/TACO retargeted references + 对应 assets | **未知** | **未知** | 官方 OneDrive 原生数据与三组 assets，未发布可核验总大小；不能把 references 当 simulator executed actions |

TACO 来源：[官方 README 与文件结构](https://github.com/leolyliu/TACO-Instructions)、[官方 V1 backup](https://www.dropbox.com/scl/fo/6wux06w26exuqt004eg1a/AM4Ia7pK_b0DURAVyxpHLuY?rlkey=e76q06hyj9yqbahhipmf5ij1o&dl=0)。HOT3D 来源：[官方下载流程](https://github.com/facebookresearch/hot3d#step-2-sign-up-and-get-the-download-links-file)、[官方所链 Clips CAD tree 元数据](https://huggingface.co/api/datasets/bop-benchmark/hot3d/tree/main/object_models?recursive=true&limit=1000)。HOI4D 来源：[官方下载页](https://hoi4d.github.io/)。HO-Cap 来源：[官方 package 列表](https://irvlutd.github.io/HOCap/)、[官方 downloader manifest](https://github.com/IRVLUTD/HO-Cap/blob/main/config/hocap_recordings.yaml)。DexYCB 来源：[官方下载表](https://dex-ycb.github.io/)。DexTrack 来源：[官方 Data 说明](https://github.com/Meowuu7/DexTrack#data)。H2O 近似值来源：[NVIDIA adapter 文档](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/docs/H2O_SETUP.md)，它不是本次对 ETH 下载服务的实测，不能视为精确数据总量。

TACO 同时保留全部 ZIP 和解压目录，峰值 **42,683,596,217 B / 42.68 GB / 39.75 GiB**，还没有计入训练 shards、模型环境、缓存及视频审查产物。文件系统 block 分配和大量小文件可能进一步增加物理占用。这个峰值应与当前磁盘/项目预算相比较；无需为接入一个 pilot 先占用这 42.68 GB。

## 2. TACO：可只取一段，避免下载 15 GB 手标注

官方 V1 是 2317 条 motion、206 个对象模型。ZIP 元数据本次实际发现 **2317 个 hand sequence、4634 个 object pose NPY、206 个 OBJ**，与 V1 文件清单相符；论文/预发布版本的约 2.5K/244 条等数字不可混用。[官方版本与 split 说明](https://github.com/leolyliu/TACO-Instructions#data-lists)

HTTP Range 已核验的三个直接链接：

- [Hand_Poses.zip](https://www.dropbox.com/scl/fo/6wux06w26exuqt004eg1a/ABNVTswzgHl6iczNcZW033Y/Hand_Poses.zip?rlkey=e76q06hyj9yqbahhipmf5ij1o&dl=1)：15,231,790,511 B；解压 25,701,586,224 B；central directory 1,620,948 B。
- [Object_Poses.zip](https://www.dropbox.com/scl/fo/6wux06w26exuqt004eg1a/AMtatHMOYfCbwEYQbu-Vfl4/Object_Poses.zip?rlkey=e76q06hyj9yqbahhipmf5ij1o&dl=1)：34,340,303 B；解压 47,082,496 B；central directory 903,362 B。
- [Object_Models.zip](https://www.dropbox.com/scl/fo/6wux06w26exuqt004eg1a/AJLl79Ae4czOVbdaOlISXhU/Object_Models.zip?rlkey=e76q06hyj9yqbahhipmf5ij1o&dl=1)：424,209,827 B；解压 1,244,586,856 B；central directory 21,311 B。

一个有明确 tool/target、双手与匹配 CAD 的 pilot 是 `(dust, roller, pan)/20230927_032`。下表 offset 指 ZIP local header，**不是 compressed payload 起点**：需先读 30 字节 header，再加文件名与 extra 的长度；全部成员压缩方式为 DEFLATE（method 8）。字节位置只适用于上面这次实际核验的 ZIP，服务端更换文件时应重新读取目录。

| ZIP | member | local header offset | 压缩字节 | 解压字节 |
|---|---|---:|---:|---:|
| Hand_Poses | Hand_Poses/(dust, roller, pan)/20230927_032/left_hand.pkl | 260 | 9,543,900 | 10,091,010 |
| Hand_Poses | 同目录/right_hand.pkl | 9,544,275 | 9,521,699 | 10,091,010 |
| Hand_Poses | 同目录/right_hand_shape.pkl | 19,066,090 | 325 | 445 |
| Hand_Poses | 同目录/left_hand_shape.pkl | 19,066,537 | 326 | 445 |
| Object_Poses | Object_Poses/(dust, roller, pan)/20230927_032/tool_164.npy | 266 | 10,040 | 14,272 |
| Object_Poses | 同目录/target_057.npy | 10,422 | 9,356 | 14,272 |
| Object_Models | object_models_released/164_cm.obj | 305,349,901 | 2,395,985 | 7,049,105 |
| Object_Models | object_models_released/057_cm.obj | 370,463,868 | 2,546,786 | 7,092,197 |

这段约 **24.03 MB 网络 payload / 34.35 MB 解压产物**，另加目录和 HTTP header。本次调查只读目录，没有取该段完整 pickle/CAD。

官方 loader 从 hand pickle 的逐帧 dictionary 读取 48 维 axis-angle `hand_pose` 和 3 维 `hand_trans`，左右各有独立 10 维 shape。FK 产生 778 顶点和 21 joints；这只能确定 reader 实际用的字段，**不能在没有解析原包之前认定 pickle 不含其他较大字段或冗余存储**。[hand_pose_loader.py](https://github.com/leolyliu/TACO-Instructions/blob/master/dataset_utils/hand_pose_loader.py)

单位链路由官方 viewer 代码明确：CAD `*_cm.obj` 乘 0.01 变米，MANO 的毫米输出除 1000，随后加 hand translation；物体 pose 直接作用到米制 CAD。viewer 的动画输出使用 30 Hz，但实际 pilot 仍应核帧键、手/物体长度、缺失标注以及采集时间说明；动画 fps 不能单独替代真实时间戳审计。[visualization.py](https://github.com/leolyliu/TACO-Instructions/blob/master/dataset_utils/visualization.py)

TACO 同时包含 tool 与 target。首批应选择一个明确目标，按不同对象记录 canonical mesh identity。对于软物体/非刚体任务，现有固定 CAD + 一个 SE(3) 不能表达真实形变；此类目标要排除或注明仅刚体近似，不因数据标称高质量直接通过。项目列明 CC BY 4.0，MANO 模型权限另行继承现有授权。[项目 License](https://github.com/leolyliu/TACO-Instructions#license)

## 3. HOT3D：full metadata 需用户动作，公开样例足够做工程接线

用户只需通过[官方 full-data 入口](https://www.projectaria.com/datasets/hot3D/)审阅相关条款，页底填写 email 并选择 `Access the Datasets`；保持刷新后的标签页打开，分别点击 Aria、Quest、Assets 下载按钮。得到的是 **JSON 清单**，无需用户手工下载大量 VRS。清单链接有效 14 天，过期后重新输入 email 获取。[官方 Step 2](https://github.com/facebookresearch/hot3d#step-2-sign-up-and-get-the-download-links-file)

三个清单惯用名为 `Hot3DAria_download_urls.json`、`Hot3DQuest_download_urls.json`、`Hot3DAssets_download_urls.json`。下载器按清单的 `sequence_config/data_groups` 选择类型；`main_vrs` 是录像，MPS points 是另一个大类，pose+mesh 路线通常可跳过。具体 annotation 组名称与打包方式必须拿到 JSON 后读取，当前不臆造。[下载器分组选项实现](https://github.com/facebookresearch/hot3d/blob/main/hot3d/data_downloader/utils.py)

官网总规模是 833 分钟、超过 1.5M multi-view frames / 3.7M images、19 subjects / 33 刚体；**不是 1.5M 条独立动作**。[官方主页](https://facebookresearch.github.io/hot3d/) 测试参与者 P0004/P0005/P0006/P0008/P0016/P0020 的 full release 无 GT；训练可用规模必须按 `gt_available_status` 和 annotation availability 统计，不能把整个 833 分钟都承诺为可监督训练。

公开 reader fixture 根：[官方 tests/data_sample](https://github.com/facebookresearch/hot3d/tree/main/hot3d/data_loaders/tests/data_sample)。可直接使用 `raw.githubusercontent.com/facebookresearch/hot3d/main/hot3d/data_loaders/tests/data_sample/` 作为 base：

- `Aria/P0003_c701bd11/mano_hand_pose_trajectory.jsonl`：103,323 B；每手 15 维 PCA pose + 10 betas + wrist `t_xyz/q_wxyz`，0 左/1 右。**不能当 45 维 axis-angle 直接输入普通 MANO。**[官方 parser](https://github.com/facebookresearch/hot3d/blob/main/hot3d/data_loaders/loader_hand_poses.py)
- `Aria/P0003_c701bd11/dynamic_objects.csv`：68,220 B；`t_wo_[m]`、`q_wo_*` 和 nanosecond timestamp。
- `Aria/P0003_c701bd11/metadata.json`：238 B；可用性 masks 同目录 `masks/`，`headset_trajectory.csv` 可用于帧系核对。
- 替代 MANO：`umetrack_hand_pose_trajectory.jsonl` 92,905 B + `umetrack_hand_user_profile.json` 592,333 B；模型/点定义仍须按官方 FK 审计。
- `object_library/instance.json` 1,444 B 映射 uid 与名字；同目录六个 `<uid>.glb` 共 22,102,404 B。最小 cellphone `5462893327580.glb` 126,936 B，mouse `265826671143948.glb` 807,492 B，mug_white `223371871635142.glb` 12,213,696 B。[官方 instance mapping](https://github.com/facebookresearch/hot3d/blob/main/hot3d/data_loaders/tests/data_sample/object_library/instance.json)

Aria fixture 本次直接解析为 68 条手标注但只跨 **1.100110919 秒**：存在约 0.09 ms / 33.24 ms 交错时间间隔，是多相机时间集合，不能把它当 68 个 30 Hz 时间步。选择一个合法时间流/对齐规则再生成 H4/K24。Quest3 fixture 只有 15 条、0.466666666 秒，适合 parser smoke，时长不足完整 H4/K24。[Aria 原始 JSONL](https://github.com/facebookresearch/hot3d/blob/main/hot3d/data_loaders/tests/data_sample/Aria/P0003_c701bd11/mano_hand_pose_trajectory.jsonl)、[Quest3 原始 JSONL](https://github.com/facebookresearch/hot3d/blob/main/hot3d/data_loaders/tests/data_sample/Quest3/P0002_273c2819/mano_hand_pose_trajectory.jsonl)

33 个完整 Clips GLB + info 合计 174,905,353 B；eval simplified GLB + info 是 3,071,909 B。这是 **Clips/BOP library** 的盘点，不是 full Assets 包实测；若借用它，要审计 BOP ID 与 full uid、canonical transform、几何简化是否改变目标。公开托管无需新账号不意味着豁免 dataset 或 MANO 条款。[full GLB tree](https://huggingface.co/api/datasets/bop-benchmark/hot3d/tree/main/object_models?recursive=true&limit=1000)、[eval GLB tree](https://huggingface.co/api/datasets/bop-benchmark/hot3d/tree/main/object_models_eval?recursive=true&limit=1000)

## 4. HOI4D：下载体积未知，原生时间与 articulated 接口必须核实

官方分别提供 CAD、Annotations、Camera Parameters、Hand Pose 的 OneDrive 链接以及 Baidu 备份（官网给密码 7gfs）。本次未见强制用户注册条款入口，但 OneDrive `shares/.../root` 返回 HTTP 401，web 跟随链接也失败。**这是当前读取链路障碍，不是已经证明用户没有数据权限**；可由用户浏览器确认分享是否可访问，或提供下载清单/现有本地几何包路径。不要因此要求用户把全部 RGB-D 下载一遍。[官方页面](https://hoi4d.github.io/)

论文总规模 4000 sequence / 2.4M frames / 800 instances。当前官方 instructions 的 `release.txt` 本次实数 **2971 行**，`testset.txt` **933 行**；尚未核对最新下载包清单的差异，暂不能承诺全部 4000 条有公开可用 pose。[release.txt](https://github.com/leolyliu/HOI4D-Instructions/blob/master/release.txt)、[testset.txt](https://github.com/leolyliu/HOI4D-Instructions/blob/master/testset.txt)

首批只需 objpose、hand annotations、canonical meshes，以及将移动相机系转换至共同坐标系所需的 camera trajectory。官方 annotations 包还可能含 2Dseg 和 3Dseg 点云；下载/解压时应只选择关键成员。[官方 data organization](https://github.com/leolyliu/HOI4D-Instructions)

HOI4D 包含 rigid 和 articulated（例如 laptop/scissors/storage furniture）。官方为 articulated 提供 part meshes、aligned part canonical frames、hierarchy 和 joint limits；不能以一个整物体 SE(3) 生成全部未来点。工程第一批可先选 rigid 类，后续按固定 part identity + 每部分 SE(3) 接入。[官方 CAD 格式](https://github.com/leolyliu/HOI4D-Instructions#object-cad-models)

原论文 action segmentation 实验使用 **15 fps**，但这句话描述的是实验特征时间分辨率，并没有充分证明所有原生 pose 包都统一为 15 Hz。必须核实际帧号、camera记录及手/objpose cadence，才决定是否从 15 重采样到 30。插值增加时间点不增加真实观测带宽，且必须记录 synthetic/interpolated validity；不能把插值后的 30 Hz 标成原生 30 Hz。论文也说明采集/标注主要是单手，而非系统覆盖双手协作。[原论文 Sections 6.3 / 7](https://hoi4d.github.io/HOI4D_cvpr2022.pdf)

## 5. HO-Cap：几何非常小，官方原 downloader 会多下载图像

官方当前分享页重定向到 `utdallas.app.box.com`。旧 downloader manifest 用 `utdallas.box.com/shared/static/...` 本次得到 404；使用当前 app host 的同一路径则 HTTP 206 成功：

- [poses.zip](https://utdallas.app.box.com/shared/static/2lofbp2yd005d8o213ns77mdrtxg8eep.zip)：24,712,636 B；解压 39,997,888 B。
- [models.zip](https://utdallas.app.box.com/shared/static/con44iqej33weg9f3rpxof61eh3x2x21.zip)：54,906,991 B；解压 135,102,932 B。
- [calibration.zip](https://utdallas.app.box.com/shared/static/nlp4c6vtd0n8o0entxlh1vxdpcdeh0h8.zip)：10,538 B；解压 11,936 B。

ZIP directory 本次确认 poses 内是 `subject_N/<sequence>/poses_m.npy`、`poses_o.npy`、`poses_pv.npy`，**没有 meta.yaml**；calibration 含九名 subject 的 MANO betas。reader 还需 meta 的 `object_ids`、`mano_sides`、`num_frames`、`subject_id` 等；可从对应 subject ZIP 单独读取 metadata，再按一个 sequence 做 pilot。[官方 SequenceLoader](https://github.com/IRVLUTD/HO-Cap/blob/main/hocap_toolkit/loaders/sequence_loader.py)

不要直接运行官方 downloader 的 `--subject_id all`：它还会下载 labels 和全部 subject 图像包。官网 subject 包总计约 **100.5 GB**（rounded published sizes）；这与 79.63 MB 的几何输入相差三个数量级。[官方下载器实际行为](https://github.com/IRVLUTD/HO-Cap/blob/main/tools/hocap_downloader.py)

数据许可 CC BY 4.0；工具仓库 GPL v3 是另一项许可。无需重新执行它的 SAM2/pose estimation，因为已有 3D 标签正是本路线要使用的内容。[官方数据许可与下载说明](https://irvlutd.github.io/HOCap/)

## 6. 辅助来源与 UniDex 名称边界

H2O 原站需要 First/Last Name、email、affiliation，点击 `I Agree` 后邮件发账号密码，7 天有效；条款限定学术用途和不转交第三方。现有已授权数据可直接使用，不应为了公开调查新建长期身份。[ETH 官方注册与使用条款](https://h2odataset.ethz.ch/) 最小 pose archive / object.zip 方案可参照 adapter，但本轮没有实际认证包大小；4 subjects、8 objects、30 Hz 多视角的去重应按同一动作而不是 camera copy 完成。[H2O 原论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Kwon_H2O_Two_Hands_Manipulating_Objects_for_First_Person_Interaction_Recognition_ICCV_2021_paper.pdf)

DexYCB 原始官方发布 119G 混合包。即使只要 3D 标签，也不能在没有找到官方标签分包时承诺只下载几十 MB；可在支持 Range 的 ZIP 或现有授权镜像做选择读取，但 `.tar.gz` 通常不能像 ZIP 随机访问任意成员。若下载 subject pilot，要按 12G 临时 input 计预算，提取标签后转 compact，而不是原地解码图片。[官方 Data 表](https://dex-ycb.github.io/) 582K 是多视角图像规模，训练动作不应以镜头数扩张独立样本数。

DexTrack 的 GRAB/TACO references、Allegro/LEAP+Franka embodiment 及 object assets 符合后续 embodiment augmentation 候选，但与原 GRAB/TACO 共用来源划分；retargeted kinematic qpos/pose 不能自动充当物理 rollout 或 action supervision。[官方参考数据与控制设置](https://github.com/Meowuu7/DexTrack#kinematic-retargeting)

`unidex-ai/UniDex` 的 **2026 预训练/HandAdapter** 与 `PKU-EPIC/UniDexGrasp2` 的 **2023 ShadowHand policy rollout** 是两个不同项目。前者列 H2O/HOI4D/HOT3D/TACO 和 Inspire 等多手 retargeting；后者提供 ShadowHand 抓取策略和 simulator 任务。前者文档写处理后可达 **80 TB**，其步骤明确下载/解码 RGB-D，并生成 `retarget_RGBD/*.h5`；它不是我们的 pose+mesh 需求，不能照该脚本全跑，也不能据此认定本路线要 80 TB。[UniDex 官方 DATASET.md](https://github.com/unidex-ai/UniDex/blob/main/doc/DATASET.md)、[UniDexGrasp++ 官方 README](https://github.com/PKU-EPIC/UniDexGrasp2)

## 7. 紧凑训练存储的工程估算

以下是我们拟采用序列存储的数值推导，不是供应方下载包大小：

```text
每 sequence 保存一次：
  hand semantic points [T, 22, 3] float32         264 B/step
  object pose [T, M, 7] float32                  28 M B/step
  timestamp float64                              8 B/step
  hand/object validity、映射、必要原生状态          按实际 schema 计
每对象/mesh 保存一次：
  canonical CAD / sampled scene + fixed query identity
每 anchor 保存轻量索引；H4/K24 的 U/V/G 训练时从 poses 生成
```

双手 22 semantic points、1–6 个对象，带 masks 与必要原生参数时可按 **约 0.4–1.0 KB/真实时间步** 预估未压缩训练序列，再加共享 CAD 和索引。1M 时间步约 **0.4–1.0 GB**；HOT3D 1.5M 这个公开总规模下约 **0.6–1.5 GB** 的 pose/hand 数组，尚未扣测试无标签、补质量过滤，也未计额外重复镜头。HOI4D 如果实际 2.4M 有效原生步则约 **1–2.4 GB**，假如确实 15→30 全重采样则数组约翻倍；这是明确条件下的算式，不是已处理数据的实测。

mesh 采样不应逐帧复制。全 CAD surface points、逐帧 hand mesh、K24 future query、anchor scene 的反复物化会把上述小规模放大几十到几百倍。紧凑 provider 只在 batch 内生成旋转/平移后的点与未来位移，源 pose 与唯一 mesh 常驻/缓存即可。

因此地点决策应分三层：**原始 ZIP/大资产归档空间、pilot/转换暂存峰值、GPU 训练所读的 compact shards**。TACO 原始全几何需要十几 GB 下载、约 43 GB archive+unpack 峰值，但首个真实 pilot 只约 24 MB；HO-Cap 原始全几何不到 80 MB（另加很小 metadata），完全无需远端大存储才能开始。训练地点最终还要看当前实际可用磁盘、授权预算、GPU 显存与占用，而非候选数据集总 RGB-D 大小。

## 8. 本轮测量方法与后续边界

- TACO：官方 README 所链 backup 页面公开 metadata 定位具体 ZIP；三个 URL 以 `Range: bytes=0-0` 验证 Content-Range；再读取 ZIP 尾部 EOCD/ZIP64 EOCD 与 central directory。只求每成员压缩/解压字节与路径，未下载 15 GB 内容。
- HO-Cap：官网与官方 manifest 确認分享 token；公开 Box 页 metadata 和修正域名后的 Range 字段互相核验，再读 ZIP directory。未下载整个约 80 MB 内容。
- HOT3D CAD：官方所链 BOP/HF tree API 枚举 33 GLB + info 求和；没有下载 GLB 全量。官方小 fixture 只读取文本标签与 repo file-size 元数据进行 cadence/FK 格式调查。
- HOI4D/H2O/DexTrack：未知处明确保留，不把链接存在等同可下载，也不把一篇论文总规模等同当前公开 annotation coverage。

后续最便宜的改变决策信息是：TACO 一段真实 pose+mesh reader + 几何抽审；HO-Cap 一段 metadata+pose reader；HOT3D 用户授权后盘点纯 annotation groups；HOI4D 获取包清单后核原生 cadence 与 rigid 子集。多源训练是否有收益仍由 Probe/Validation 决定，来源完整或工程 smoke 通过不能替代有效性结论。
