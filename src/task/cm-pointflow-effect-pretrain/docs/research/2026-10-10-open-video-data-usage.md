# 开源视频动力学：作者实际怎样用数据，以及当前应先复用什么

核查日期：2026-10-10。问题来自用户：先找开源，尽量不自行重写；目标是判断视频是否帮助现有模型。使用 research 技能并行核查论文与官方源码。以下调查没有训练模型；权重获取/兼容性检查单独记录。全局目标仍是动作条件 Cm 改善自训练机器人策略。

## 选择（已纠正）

此前错误地优先执行了官方 **PointWorld-small 的 DROID 预训练编码器迁移**，复用当前训练器比较视频初始化与随机初始化；这实际测试的是官方权重，不是官方视频数据。该运行已保留，但相对于用户要问的“官方视频数据能否帮助现有模型”，应标为 `INVALID_IMPLEMENTATION / INCONCLUSIVE`，不能作为数据路线的负结果。

用户真正要求的是优先复用公开视频数据，训练或适配原生模型后再与原生数据控制臂比较。下一步应先做不下载大语料的官方数据合同和可用子集审计，再决定是否在 NAS 上取得一个有界子集；不再把 checkpoint transfer 当作数据实验。

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

当前 native 模型与官方 small 都按同一 PTv3 blueprint 创建空间 backbone，native latent width128。后续工程核查已完成：固定文件的字节数和完整SHA256正确；空间backbone448/448张量、50,417,280entries，键、shape、dtype全部匹配且有限。两组3-update GPU检查通过。**这证明权重能装入，不证明输入分布相容或视频迁移有收益**。native保留patch128和time-aware pooling，作者small使用patch256；非backbone参数和归一化统计保持配对相同。RGB/DINO 和机器人 head 不可冒充 native feature/action/effect head。完整 official 推理还需要作者依赖和 DINO 权重，而 backbone-only 迁移可先不启动这些额外部分。所有已存在 native checkpoint 原样保留。

## ObjectForesight：与我们的 LK 试验不同，不能因为 LK 失败否定作者数据

训练仓库固定 `1a2fa7e8b41caec3c2c9930bcb077ddef22cf319`；配置 `context_len=3, H=8, frame_skips=0`，`n_points=20000`，voxel下采样目标4096，`load_hand_poses=false/use_hand_context=false`。它消费的是 FoundationPose 对象刚体轨迹和单帧场景深度，不把 SpaTracker `coords` 的100条查询作为物体未来监督。[固定配置](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/conf/epic.yaml#L12)、[pose读取](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/data/fpose_io.py#L96)

loader 将窗限制在同一 `init_from_frame` 注册段，按 IoU drop 过滤。`post_train` 另有meanIoU/运动筛选，默认关闭。相机变换后统一 anchor 坐标；固定mesh不是网络预测点流的前提。源码当前在 pose 行列表上切窗，并未证明原始视频frame gaps为零；时钟和相机约定仍需核验。[窗口构造](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/data/datasets/dataset_epic.py#L593)

具体时间合同也不能只看 H=8：`target_future` 从索引 `P_ctx` 起，`context_T_cam_anchor_obj` 则取 `P_ctx+1` 行，两者共享 anchor 行；8个输出并不自动等于我们8个严格未来步。缺首帧mask时，fallback会遍历包含未来的整个窗口。离线几何也来自全片段。因此照用作者代码时可测试表征迁移或复现作者离线任务，但不能无检查宣称严格前缀预测。[getitem/anchor](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/data/datasets/dataset_epic.py#L712)、[targets/mask/context](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/data/datasets/dataset_epic.py#L810)

root 只读列出已有 P03_03/P03_13 tar 的 `poses.npz/track_log.csv/run_summary.json`：分别27/20文件，约245,641/125,185bytes。证明本地存在作者刚体轨迹资产；**未加载其窗口或验证质量**。可在官方加载器路径上继续，而不是因原始稀疏查询没16点就断言这些shard不能学物体动力学。缺密集未来手动作仍阻止把它伪装成完整 native hand-conditioned 数据。

补充资产核查：作者[项目页](https://objectforesight.github.io/)直接链接
[EPIC 模型](https://huggingface.co/raivn/ObjectForesight-EPIC-DiT)和
[HOT3D 模型](https://huggingface.co/raivn/ObjectForesight-HOT3D-DiT)，均公开且非gated；
四个权重文件匿名HEAD均返回200。EPIC revision
`08084845391b0db49966c8da58c453d47430316c`，best.pt733,255,992bytes，
model.safetensors733,124,554bytes；HOT3D revision
`808f3361b6bce3e9e1ceb422b14df21d084f8403`。数据审批与模型可下载性不同。
仅读取EPIC safetensors的94,464-byte JSON header，没有下载完整权重。
实际包含655个encoder tensors、50,564,431entries，不是只发布DiT。

当前官方代码直接创建PTv3、可导调用encoder、AdamW接收全部model.parameters()；
config名main_ptv3_fresh，没有冻结/加载通用Sonata的代码证据。因此不能把它描述成
“仅冻结Sonata训练DiT”，但也不能单凭当前源码证明历史每个encoder参数实际更新。
[构造](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/encoders/ptv3_adapter.py#L52-L55)、
[优化器](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/train_main.py#L419-L423)、
[完整保存](https://github.com/RustinS/ObjectForesight/blob/1a2fa7e8b41caec3c2c9930bcb077ddef22cf319/src/models/poser_v1/io.py#L12-L18)。

不能整体导入现有模型：其stem输入6、encoder宽度(32,64,128,256,512)、decoder64、
末端投影768；现有backbone输入及前两层128、decoder128。条件PDNorm/姿态上下文
和池化也不同。后三级部分尺寸相同只表示局部候选，不能当作已兼容的视频迁移。
核查证据 `tmp/objectforesight-assets-20261010/source-manifest.json`，包含固定来源、
API与header；没有改代码或训练。若复用应优先官方完整模块/入口，另冻匹配协议，
不默认写部分权重映射来凑兼容。

## DROID-100：确认拿到的是官方视频数据，并与官方 3-D 标注配对

为回答“官方视频数据能否进入训练”而不是“官方权重能否加载”，先审计了
官方 DROID-100 RLDS 子集。官方文档将它列为约 2 GB、100 episode 的调试子集，
而原始 MP4/SVO 发布物是多 TB 规模（见[The DROID Dataset
docs](https://droid-dataset.github.io/droid/the-droid-dataset)）。只下载了
一个 TFRecord shard（27,488,816 bytes）和两个元数据文件，没有下载全量语料。
该 shard 含两个 `SequenceExample`，166 和 238 个 step；三路字段
`wrist_image_left`、`exterior_image_1_left`、`exterior_image_2_left` 的抽样帧均
能解码为 320x180 RGB JPEG，并同时带有 7-D action、joint state、语言和源路径。
因此它确实是官方视频帧/动作记录，不是 PointWorld 的派生标签包。

随后按 PointWorld-DROID 固定 revision 的 manifest，只取得 flow shard `000409`
（压缩包 3,499,128,679 bytes，SHA256
`38fc5fc032f422a34d62a5b7a6d554189f68fafdab0f568660651467de2b23e9`）。官方恢复
脚本得到 68 个 H5、722 个 clip、1,444 个 camera group；全部可打开，抽样 scene
flow 没有非有限值。DROID-100 首条 episode
`RAIL+80edfcb1+2023-04-17-14h-48m-05s` 在其中出现十个 clip，两个 camera serial
`20521388`、`24259877` 与 raw metadata 对上；每个匹配 clip 同时有 320x180 初始
RGB、11-step 3-D scene flow、7-D gripper pose 和 7-D joint positions。两份发布物
的初始 RGB 尺寸与相机身份一致，但像素 MAE 约 4.46，说明它们经过不同 JPEG/处理
路径，不能声称字节级相同。

这一步的结论是 `PROMISING`，范围仅限“官方视频样本和官方 3-D 标注可以有界地取得
并配对”。尚未把它们送进 native trainer，也没有任何视频收益结论。下一步最便宜
且可判别的动作是先审计该输入合同。结果显示所有匹配 clip 只有 11 个时间点，
而当前 native learner 要求 4 个 history 加 24 个 future（共 28 点），所以直接
native 覆盖为零；同时没有已验证的 object/background `point_kind`、跨 clip 点身份
或 2-hand×11-keypoint×9-D action 张量。这个结论记录在
[native adapter Probe](../experiments/probes/P-20261010-droid100-native-adapter-contract.md)。
因此不能把 H5 clip 静默填充成 `H=4,K=24` 后训练。后续必须明确选择原始长序列重处理、
显式缩短时域的 PointWorld-style 模型，或独立 RGB/robot-state 前端，并为所选路线
重新冻结随机初始化控制臂。

## 下一步决策与范围

此前的资源决策只批准获取固定 SHA 的官方 small checkpoint 做兼容检查，
没有批准获取官方视频数据。该检查和后续 checkpoint-transfer 运行已经完成，
但它不回答视频数据问题；对应结果按 `INVALID_IMPLEMENTATION / INCONCLUSIVE`
保留在[实验卡](../experiments/probes/P-20261010-open-video-backbone-transfer.md)。

当前 Decision 已从“能否取得数据”推进到“选择哪一种显式适配合同”；已配对子集不能
直接接入现有 native 时域。
官方数据卡报告 PointWorld-DROID 全量下载约 3.91 TB、展开约4.65 TB，而本
Campaign 本地产物上限为300 GB；DROID 原始 MP4/SVO 规模更大。因此保持有界
episode/shard 策略，不下载全量数据，不启动新的权重迁移训练。只有在 adapter
合同通过后，才评估一个仍在本地上限内的训练子集；远程 NAS 或大规模 staging
仍需要单独的资源决策。

有效的数据 Probe 必须使用官方视频样本作为训练输入，并保留同一原生模型随机
初始化控制臂、固定 native 数据/统计/评价接口和明确的数据适配层；不能把官方
checkpoint 当作“视频数据臂”。是否改善最终 RL 仍需独立的 Mission 级 Cm-on/off
比较。

失败后优先核查官方 ObjectForesight checkpoint/loader 对既有 EPIC 的复用，或选择已公开的共同场景/手处理产物；不默认自己复刻待发布 PointWAM。研究范围不缩减成人类动作识别、2D像素预测或视觉触觉同期估计。

## 证据文件

固定小源码及其逐文件URL/SHA在 `tmp/open-video-data-usage-20261010/{RustinS-ObjectForesight-1a2fa7e8b41caec3c2c9930bcb077ddef22cf319,NVlabs-PointWorld-3872ec6ee73146aa671192ef79b5dfbedc0246e3}/source-manifest.json`。Hub API快照为同目录 `nvidia-PointWorld*-hub.json`，模型卡为 `PointWorld-models-README.md`。GitHub API一度限流，commit改用只读 `git ls-remote` 获取，再读固定raw文件；未修改本地submodule。

## Mismatched checkpoint-transfer result (2026-10-10; retained, not data evidence)

The pinned small-DROID backbone was tested in the existing native action trainer,
with the same seed, draw, data, statistics, optimizer and 2000-update budget as a
random-backbone control. Both arms completed normally. The balanced source-macro h24
moving-object EPE was 0.080660 for random initialization and 0.082070 for the
released video initialization (+1.75%); static-object EPE was 0.022597 and
0.029279 (+29.57%). The predeclared moving-at-most-90% and static-at-most-120%
gates both failed. The released-backbone-only transfer is therefore
`UNPROMISING` as a checkpoint-transfer recipe only. The run is
`INVALID_IMPLEMENTATION / INCONCLUSIVE` for the official-video-data question:
neither arm consumed official DROID video samples. Final evaluation and the full source/horizon
breakdown are preserved in
`outputs/cm-pointflow-effect-pretrain/open-video-backbone-transfer-20261010-r1/result_analysis_r1.json`.

The action shuffle diagnostic still increases moving h24 error by 42.84% for the
random arm and 39.92% for the video arm, confirming that the native action path
is being used. It does not establish a video benefit, and the static shuffle
change goes in the opposite direction in both arms. Stop this checkpoint-transfer
recipe before longer training. This result says nothing about training on the
released video data and does not refute other video or human-video pretraining
routes.
ObjectForesight's released encoder remains structurally incompatible with the
native backbone, so no second open-source transfer run is started automatically.
