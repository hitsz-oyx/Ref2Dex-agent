# HOCap 外部测试集预检查

检查日期：2026-10-08。范围：官方资料和工具源码只读核验；下载由主会话独立执行，本记录不代表已经完成转换或模型评估。官方工具仓库检查时 `main` 为 `576c63ebf3b84dfec8744ba0f021234213bf0dab`。

HOCap 有独立拍摄的连续手物交互、刚体 mesh 和逐帧姿态，适合作为当前 PointWorld 的外部测试候选。当前应保持 TEST_ONLY，不加入训练、统计量拟合、学习率选择或 checkpoint 选择。下载成功不等于通过我们的数据合同。

## 官方入口与最小下载

[项目主页](https://irvlutd.github.io/HOCap/)链接的官方工具仓库是 [IRVLUTD/HO-Cap](https://github.com/IRVLUTD/HO-Cap)，原始包由 UT Dallas Box 托管。优先尝试独立的 `calibration.zip`（19.6 KB）、`models.zip`（52.5 MB）和 `poses.zip`（23.8 MB），总压缩体积约 76 MB；包大小是官网声明，需以实际响应核验。其内容是否足够独立恢复每条序列的 `meta.yaml`、手侧和对象编号，仍需检查 archive。若缺少 metadata，应先寻找官方 metadata 小包，不默认下载全部 RGB-D。

直链定义来自官方 [hocap_recordings.yaml](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/config/hocap_recordings.yaml)：

| 包 | 官方 Box 地址 |
| --- | --- |
| calibration | https://utdallas.box.com/shared/static/nlp4c6vtd0n8o0entxlh1vxdpcdeh0h8.zip |
| models | https://utdallas.box.com/shared/static/con44iqej33weg9f3rpxof61eh3x2x21.zip |
| poses | https://utdallas.box.com/shared/static/2lofbp2yd005d8o213ns77mdrtxg8eep.zip |

主会话已实际取得上述三包，大小分别为 10,538 / 54,906,991 / 24,712,636 bytes；其 SHA256 与第三方 [pablovela5620/hocap-original](https://huggingface.co/datasets/pablovela5620/hocap-original) 对应 LFS 哈希一致。第三方镜像不等同官方发布，匹配证据仅覆盖已校验的文件。实际 `poses.zip` 有265个条目、包含逐序列 `poses_m/o/pv.npy`，**没有 `meta.yaml`**。官方 Git tree 也未发现独立 sequence metadata 小包；因此还需从 subject archive 获取 meta.yaml，优先 HTTP Range 定位 ZIP directory 后只取 metadata（大于4GB的 subject ZIP需正确处理ZIP64）。本地下载与网络证据入口为 `outputs/cm-pointflow-effect-pretrain/hocap-test-preflight-20261008-r1/`。

官方 [下载脚本](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/tools/hocap_downloader.py)没有登录或 token 参数，但即使指定单个 subject，也会额外下载 `labels.zip` 和整 subject。因此最小预检查不直接执行该脚本。公开直链并不保证当前网络可达、链接有效或 Range 支持；需要实际响应和 zip 完整性证据。脚本的断点续传遇到服务器返回完整 `200` 时仍会 append，已有 partial 不能盲目沿用。

## 许可与版本

存在**官方许可描述冲突**：项目主页写 CC BY 4.0，而 [NeurIPS 2025 正式论文 Appendix K](https://proceedings.neurips.cc/paper_files/paper/2025/file/f92532079ab50a5db4109ca8f0cb0849-Paper-Datasets_and_Benchmarks_Track.pdf)写 CC BY-NC 4.0。当前非商业科研测试可继续；在许可澄清前不宣称已获得商业使用或再分发许可。工具仓库另为 GPL-3.0，不能把工具许可直接当成数据许可。MANO 模型来自单独的官方注册下载流程，也不能根据 HOCap 数据许可推断 MANO 授权。

镜像必须保存来源、revision、获取时间和文件 SHA256，并与官方包目录、对象表和数据语义比对。镜像 repo 名相同不是版本一致证据。官方源码的 commit 也不能代替 Box 包版本身份；官网包未见不可变发布编号。

## 逐字段合同

| 字段 | 官方证据 | 我们的接入要求 |
| --- | --- | --- |
| object identity / mesh | [SequenceLoader](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/loaders/sequence_loader.py)从 `meta.yaml.object_ids` 选择 `models/<id>/textured_mesh.obj` 和 `cleaned_mesh_10000.obj` | 固定每个对象的 512 个 canonical 点，所有时刻用同一个点对应；不得逐帧重采样，也不能仅按语义名称关联 mesh |
| object SE(3) | [SequenceRenderer](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/renderers/sequence_renderer.py)读取 `poses_o.npy`，经 `quat_to_mat` 后直接作为世界节点下的 mesh pose | object→world；检查对象维与 `object_ids` 一致、有效 SO(3)、缺失 sentinel、单位和有限性 |
| quaternion | [transforms.py](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/utils/transforms.py)取前四维传给 SciPy `Rotation.from_quat`、后三维为平移 | 标量在后的 `(qx,qy,qz,qw,tx,ty,tz)`；不能套用 ContactPose 的标量在前转换 |
| hands / side | renderer 明确按 `poses_m[0]` 取 right、`poses_m[1]` 取 left，并按 `mano_sides` 选存在手 | 输出固定 `[right,left]`；缺失手 mask=false，不填复制手或默认有效零坐标 |
| MANO decode | [MANOGroupLayer](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/layers/mano_group_layer.py)每手 51 维：48 pose + 3 translation；[MANOLayer](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/layers/mano_layer.py)使用 PCA45、`flat_hand_mean=False`，v/j 除以1000 | 不能把其48维直接当我们其他数据源的非 PCA axis-angle；复制官方解码的数学语义，独立验证输出米制 |
| 11 semantic points | [mano_info.py](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/utils/mano_info.py)给出21点的 wrist、拇指 CMC、各指 MCP 和 fingertip | 取 `[0,1,5,9,13,17,4,8,12,16,20]`；与当前 `contactpose.py` legacy `thumb_mcp` 实际指第一拇指关节 CMC 一致；不得按名称误取拇指 index2 |
| world / camera | loader 使用 `tag_1^-1 @ camera_to_master`，并明确用逆矩阵投影 | 手物必须处于同一世界系；米制 mesh+poses 直接一致时不追加相机 transform。相机标签应经相机到世界转换后再比较 |
| clock / FPS | 已读官方 loader/renderer 不提供原始 timestamp/FPS 合同；论文附录的 HPE 10 FPS 是评估抽样 | **待核验**原始连续帧时钟；不能以 HPE/OPE 稀疏样本冒充30Hz连续动态。若真实30Hz且无gap，H4+K24需至少28连续帧，未来覆盖0.8秒 |
| validity | [dataset_factory.py](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/factory/dataset_factory.py)对全 `-1` 的 MANO pose 原样保留 | `-1` 是有限数，不能仅用 isfinite 判有效；也不能因有效 quaternion 而把 sentinel 姿态纳入测试 |

特别注意 MANO translation 应直接以米传入官方解码：完整追踪 [manopth 的 ManoLayer](https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py)可见底层先加 `th_trans`，末尾才将 v/j 整体乘1000，随后 HOCap wrapper 除1000，得到米制世界点。只看到 wrapper 的除1000而推断原始 t 是毫米会造成错误。实际下载的 `subject_1/20231025_165502` 中 `poses_m.shape=(2,676,51)`，有效 t 各轴范围为 `[-0.262303,-0.311378,0.019229]` 至 `[0.164720,0.012798,0.289620]`，与同帧物体 t 米尺度一致；右手首帧是全 `-1`。这些证据支持米制解码，但最终仍需固定实际 MANO/manopth 依赖 revision，并核验 hand/object 同帧位置和投影。mesh 单位应以实际尺寸及配套 pose 校验。

官方 [arXiv v3 注释方法](https://arxiv.org/html/2406.06843v3)描述了多视角手关键点过滤、缺失帧插值和轨迹平滑。因此该测试是与发布的优化/重建标注比较，不能自动称为独立 mocap 真实值；缺少逐帧置信度时不能伪造高置信 mask。

### 原始采集时钟追查结果

进一步核验仍**不支持“原始连续流已经证实为30Hz”**：

- [NeurIPS 2025 论文 §3](https://proceedings.neurips.cc/paper_files/paper/2025/file/f92532079ab50a5db4109ca8f0cb0849-Paper-Datasets_and_Benchmarks_Track.pdf)明确写八台 D455、多视角标定和同步视频，但没有给出原始 FPS/Hz 或逐帧 timestamp 定义。其附录 E 的 10 FPS 指 HPE 评估抽样，不能推出原始流30FPS。
- 官方 [HO-Cap-Annotation README](https://github.com/IRVLUTD/HO-Cap-Annotation/blob/main/README.md)公开的是分割、检测和姿态优化入口；已读 [sequence_loader.py](https://github.com/IRVLUTD/HO-Cap-Annotation/blob/main/hocap_annotation/loaders/sequence_loader.py) 与 [hocap_loader.py](https://github.com/IRVLUTD/HO-Cap-Annotation/blob/main/hocap_annotation/loaders/hocap_loader.py)没有 FPS、timestamp 或原始 RealSense stream 配置。这并非对所有尚未取得配置的否定证明；进一步 Git tree 查询遭 GitHub API rate limit，未据此虚构 capture 配置结论。
- 主会话 Range 提取的 `subject_1/20231025_165502/meta.yaml`有676帧、4物体 `G18_1..4`、`mano_sides=[left]`，但没有 FPS/time 字段。与 `poses_m/o.npy` 帧数一致可以证明索引对应，不能证明每个相邻索引严格间隔1/30秒。

第三方整理资料可能标称30FPS，但不是原始采集者的时钟证据。本轮建议保持 `source_fps_verified=false`、`timestamp_provenance=UNVERIFIED`，不给数据制造 `timestamps=np.arange(T)/30` 作为已测量时钟。如果后续从官方 subject archive 找到录制配置或采集 timestamp，可再审核；视频渲染文件的30FPS播放参数也只能证明播放节奏。未解决前可以完成 schema/几何检查，但不将其报告为与训练域严格相同的0.8秒物理预测测试。

## 与已有训练域的隔离

官方对象表 [hocap_info.yaml](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/config/hocap_info.yaml)提供 Gxx_x 对象编号、名称和独立序列编号，显示其为自己的 capture collection。**这是不同采集域的证据，不是与 OakInk2 / GRAB / ARCTIC / ContactPose 零重叠的证据**。相似日用品、同品牌对象或 canonical mesh 重用仍需检查。

接入前检查训练及历史初始化来源中的 sequence/source identity、实际 mesh SHA256、对象名称/型号和几何近重复；subject_1 与其他域同名 subject_1 不能认定为同一个人。没有跨数据集身份对应证据时报告未知，不宣称 subject 完全独立。父权重曾用过 ContactPose，隔离审计必须覆盖父权重的数据链，不能只检查最后一次主三域 manifest。

HOCap 官方 HPE/OPE/ODET split 是感知任务的 frame/camera 列表，不能直接作为连续动力学划分。建议整条 HOCap 序列和所有相机都归外部 test-only；可留固定少量序列做 schema/坐标工程校验，但不得用其预测误差挑 hyperparameter、best checkpoint 或主模型归一化。最终用预先固定的训练域 checkpoint 和训练域 stats，报告与现有指标一致的刚体点 EPE，分别列 moving / near-static、任务、对象和主体，不把八相机重复同一世界动作算八条独立样本。

## 本次实际下载与检查结果

运行 `hocap-test-preflight-20261008-r1`，获取时代码为 `c594f6a`，约637秒完成，
新磁盘占用约246MiB。输出入口
`outputs/cm-pointflow-effect-pretrain/hocap-test-preflight-20261008-r1/acquisition_manifest.json`。
ModelScope检索未找到可用条目；国内HF镜像两次连接超时后，使用本地代理读取官方
Box公开直链。Hugging Face第三方包身份固定为
`d9a562638bc4eda48451ce1211385371bbd17be0`；未使用账号认证。

- 三个完整校准/mesh/pose ZIP 下载后核验SHA256与CRC，安全解压；包含全部9主体、
  64序列、72944帧和64个物体模型。
- 从九个subject ZIP通过严格HTTP Range/ZIP64取出全部64份 `meta.yaml`，保留CRC
  和已提取内容哈希。未下载subject RGB-D包；部分读取不能宣称原subject整包SHA校验。
- 从 `labels.zip` 取 `subject_1/20231025_165502/105322251564` 的连续帧0..31，
  NPZ含双手 `(2,21,3)`、物体 `(4,4,4)`、相机内参和分割。仅32帧标签样本，
  不宣称全72944帧的逐帧关节标签已下载；完整手MANO/物体pose数组已下载。
- 元数据与手/物体数组维度、对象mesh引用、缺失手全-1及单位四元数检查通过；
  最大四元数范数偏差约1.19e-7。64个mesh最大轴尺寸约0.100..0.358米，未缩放。
- 32帧×4物体的相机标签转世界与原生pose对比，平移最大差4.57e-8米、旋转矩阵
  元素最大差8.88e-8。这是同一发布内部的坐标一致性，不是独立标注精度评价。
- 全部64份meta都无原始FPS/timestamp。统计有284864个几何有效28帧物体窗口，
  仅按连续数组帧计数，不等于已确认30Hz/0.8秒的正式测试窗口。

Range首轮因labels ZIP的77.63MB中央目录超过预设64MiB单次读取上限而停止，
保留原始请求证据；随后在原4GiB下载/10GiB解压/1800秒整组预算内使用128MiB
单次Range上限完成。后续批次Range总传输约238.52MB，预算未重置。
最终 `status=COMPLETED` 仅表示上述获取与几何预检查完成；
`training_allowed=false`、`test_ready=false`、`source_fps_verified=false`。
本次不运行神经网络、不下载全RGB-D、不生成预测误差。

## 仍需实际数据确认

1. 完整pose/model/calibration包与定向元数据获取已验证；扩大标签覆盖时继续保留源身份，
   不把部分Range获取描述为原始subject/labels整包已下载。
2. 真正的采集 FPS、连续帧时钟、gap 和原始有效性字段；至少一条完整 H4+K24 窗口。
3. MANO 参数解码、平移单位、手点和 mesh+SE(3) 世界系一致性。
4. 最终持有测试 manifest、跨域 mesh/序列重叠审计及固定 checkpoint/stats 哈希。

上述问题通过前，状态只能是外部测试候选；本次工作不运行模型、不改变主训练数据、不输出泛化性能结论。
