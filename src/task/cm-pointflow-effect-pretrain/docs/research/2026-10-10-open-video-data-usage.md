# 开源视频动力学：作者实际怎样用数据，以及当前应先复用什么

核查日期：2026-10-10。问题来自用户：先找开源，尽量不自行重写；目标是判断视频是否帮助现有模型。使用 research 技能并行核查论文与官方源码。以下调查没有训练模型；权重获取/兼容性检查单独记录。全局目标仍是动作条件 Cm 改善自训练机器人策略。

## 选择

优先核查官方 **PointWorld-small 的 DROID 预训练编码器迁移**，复用当前训练器，比较视频初始化与随机初始化；不继续自写 RGB/LK 视频入口或扩大未来传感器 MLP。原因：官方代码、视频预训练权重都可获得，而本项目已经使用其 PTv3-small；这一对照直接问视频资产是否帮助现有模型。它首先检验机器人视频预训练，不自动回答人类视频或触觉是否有效。

离线全视频重建可以用于制造训练监督。历史重建输入也受未来影响时，不能宣称严格历史预测，但**这不自动禁止将该离线任务用于表征预训练**：是否值得复用，要在独立、合格的下游输入上做有无预训练的匹配比较。不能以旧的少量 LK 学习失败关闭整个视频路线。

## 三种实际使用方式

| 工作 | 实际输入和目标 | 公开可用性 | 与当前模型的关系 |
| --- | --- | --- | --- |
| PointWorld | 当前 RGB-D 场景点与 URDF/FK 生成的机器人未来几何动作，预测场景点流；DROID 标签来自视觉跟踪/深度，BEHAVIOR 来自模拟刚体几何 | 官方 main/data 代码、处理后 H5 包、small/large 权重已发布 | 动作条件接口最接近；先查 small 空间编码器权重，不直接套用机器人动作 head |
| ObjectForesight | EPIC 的 FoundationPose 刚体物体姿态历史与场景几何，预测未来物体 SE(3)；默认不加载手动作 | 官方处理和训练代码、EPIC/HOT3D 模型入口 | 对已有 EPIC 的使用，应先看刚体 pose supervision；它不靠最初100条 SpaTracker查询的物体覆盖，也不等于我们自写的稀疏 LK 点模型 |
| PointWAM | EgoDex/VITRA 人类视频构造共同坐标 scene/hand 轨迹，当前彩色点云/手和语言联合预测未来 scene/hand，之后机器人示范微调 | 作者项目仍标 Code soon，未取得官方训练代码/处理后 corpus/权重 | 方法非常相关，但当前不复刻或把第三方同名仓库当官方 |

主来源：[PointWorld 论文](https://arxiv.org/html/2601.03782v1)、[ObjectForesight 论文](https://arxiv.org/html/2601.05237v1)、[PointWAM 专项核查](2026-10-10-pointwam-human-video-usage.md)。触觉作者的实际任务见[触觉专项核查](2026-10-10-open-tactile-data-usage.md)。

## PointWorld：已有代码与 checkpoint 比重建完整视频语料更便宜

本地官方 submodule 固定 `05484826dfef74cbe278a3974179a5a16705d35d`，其 README 明确区分 main 训练和 data 标注处理分支；small 权重是 DROID-only，large 有 DROID-only 与 DROID+BEHAVIOR。模型是当前场景加候选机器人动作到场景点流，不是在推理时追踪整个未来视频。[固定训练 README](https://github.com/NVlabs/PointWorld/blob/05484826dfef74cbe278a3974179a5a16705d35d/README.md)

data 分支固定 `3872ec6ee73146aa671192ef79b5dfbedc0246e3`。作者用 CoTracker 产生图像轨迹，再用逐帧 FoundationStereo 双目深度提升到3D；机器人动作来自已知 URDF 与 joint/gripper states。处理流程与我们的单目 EPIC 重建输入不同，不能把它的物理尺度资格直接转给 EPIC。[2D入口](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/real/compute_2d_flows.py#L34)、[深度入口](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/real/compute_depth.py#L18)、[3D提升](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/real/convert_2d_flows_to_3d.py#L265)

发布 main 合同是1当前帧、10预测步；论文步长0.1s。训练加载当前 RGB-D，机器人未来点由已知几何/FK生成。与本任务 H4/K24/约30Hz、11 semantic hand points、18维无RGB几何特征和刚体效果 head 不同。[固定 constants](https://github.com/NVlabs/PointWorld/blob/05484826dfef74cbe278a3974179a5a16705d35d/dataset_components/constants.py#L37)、[模型结构](https://github.com/NVlabs/PointWorld/blob/05484826dfef74cbe278a3974179a5a16705d35d/pointworld/base.py)

本次实际 Hub API 元数据确认（未以 card 的 TBD 字段推断未发布）：

- 模型 revision `b9e2e19a4f2bd65922e1f6d70aa953fe70aa9dba`，public/ungated，`small-droid/model-best.pt` **1,826,853,514 bytes**，LFS SHA256 `ccb9ed93dff5eea976010c57dd0cb5634db61c68b732c4437cbf54c8da9de8fe`。两份 large 各约13.04GB。[固定模型文件树](https://huggingface.co/nvidia/PointWorld_models/tree/b9e2e19a4f2bd65922e1f6d70aa953fe70aa9dba)
- DROID revision `dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f`，1030文件。发布的是分片压缩的 H5/JSON 包，不是已经训练就绪的 WDS；卡报告下载约3.91TB、展开约4.65TB。不能取任一分片便宣称能独立恢复任一episode。[DROID 数据卡](https://huggingface.co/datasets/nvidia/PointWorld-DROID)
- BEHAVIOR revision `5579a0c584caa182db7bda66764eb5c9b52c1c55`，172文件，允许按 task 包恢复；task-0000 整包1,660,733,672bytes。卡报告全包约718GB、展开约1.55TB。它是模拟数据，单独使用不能证明真实视频收益。[BEHAVIOR 数据卡](https://huggingface.co/datasets/nvidia/PointWorld-BEHAVIOR)、[作者按任务恢复示例](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/README.md)

当前 native 模型与官方 small 都按同一 PTv3 blueprint 创建空间 backbone，native latent width128；这只是结构候选，**尚未证明 checkpoint 所有键/shape可加载或其输入分布相容**。RGB/DINO 和机器人 head 不可冒充 native feature/action/effect head。完整 official 推理还需要作者依赖和 DINO 权重，而 backbone-only 迁移可先不启动这些额外部分。所有已存在 native checkpoint 原样保留。

## ObjectForesight：与我们的 LK 试验不同，不能因为 LK 失败否定作者数据

训练仓库固定 `1a2fa7e8b41caec3c2c9930bcb077ddef22cf319`；配置 `context_len=3, H=8, frame_skips=0`，`n_points=20000`，voxel下采样目标4096，`load_hand_poses=false/use_hand_context=false`。它消费的是 FoundationPose 对象刚体轨迹和单帧场景深度，不把 SpaTracker `coords` 的100条查询作为物体未来监督。[固定配置](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/conf/epic.yaml#L12)、[pose读取](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/data/fpose_io.py#L96)

loader 将窗限制在同一 `init_from_frame` 注册段，按 IoU drop 过滤。`post_train` 另有meanIoU/运动筛选，默认关闭。相机变换后统一 anchor 坐标；固定mesh不是网络预测点流的前提。源码当前在 pose 行列表上切窗，并未证明原始视频frame gaps为零；时钟和相机约定仍需核验。[窗口构造](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/data/datasets/dataset_epic.py#L593)

具体时间合同也不能只看 H=8：`target_future` 从索引 `P_ctx` 起，`context_T_cam_anchor_obj` 则取 `P_ctx+1` 行，两者共享 anchor 行；8个输出并不自动等于我们8个严格未来步。缺首帧mask时，fallback会遍历包含未来的整个窗口。离线几何也来自全片段。因此照用作者代码时可测试表征迁移或复现作者离线任务，但不能无检查宣称严格前缀预测。[getitem/anchor](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/data/datasets/dataset_epic.py#L712)、[targets/mask/context](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/data/datasets/dataset_epic.py#L810)

root 只读列出已有 P03_03/P03_13 tar 的 `poses.npz/track_log.csv/run_summary.json`：分别27/20文件，约245,641/125,185bytes。证明本地存在作者刚体轨迹资产；**未加载其窗口或验证质量**。可在官方加载器路径上继续，而不是因原始稀疏查询没16点就断言这些shard不能学物体动力学。缺密集未来手动作仍阻止把它伪装成完整 native hand-conditioned 数据。

## 下一步决策与范围

root 在现有用户授权内选择：先获取固定SHA的官方 small checkpoint，只作 tensor/key/shape 兼容检查；新产物<=3GiB、获取<=15min、CPU检查<=2min，无GPU训练，无新branch，无完整corpus或DINO下载。不覆盖任何已有文件；source/checksum漂移、资源越界或不可加载即停止并保留记录。权重下载依次尝试ModelScope可发现性、HF国内镜像、现有代理官方源。

成功后才固定一个下游matched Probe：同native数据/统计/动作接口/初始化后的非backbone参数/seed/batch/optimizer/更新预算/固定末步评价，只改 backbone 视频初始化。现有随机初始化50000步结果可作背景，不能直接拿它与一个短迁移运行比较并归因预训练。至少报告moving/static object误差和actual/shuffled-action敏感性；是否改善最终RL必须另作Mission要求的Cm-on/off比较。当前不承诺收益，也不启动大规模训练。

失败后优先核查官方 ObjectForesight checkpoint/loader 对既有 EPIC 的复用，或选择已公开的共同场景/手处理产物；不默认自己复刻待发布 PointWAM。研究范围不缩减成人类动作识别、2D像素预测或视觉触觉同期估计。

## 证据文件

固定小源码及其逐文件URL/SHA在 `tmp/open-video-data-usage-20261010/{RustinS-ObjectForesight-1a2fa7e8b41caec3c2c9930bcb077ddef22cf319,NVlabs-PointWorld-3872ec6ee73146aa671192ef79b5dfbedc0246e3}/source-manifest.json`。Hub API快照为同目录 `nvidia-PointWorld*-hub.json`，模型卡为 `PointWorld-models-README.md`。GitHub API一度限流，commit改用只读 `git ls-remote` 获取，再读固定raw文件；未修改本地submodule。
