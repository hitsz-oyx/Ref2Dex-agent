# 官方 DROID 视频数据的可复用路线（不加载官方模型权重）

核查日期：2026-10-10。目的：核实是否必须安装 ZED/`pyzed` 或加载 PointWorld 官方 checkpoint，才能让官方 DROID 视频数据参与当前 PointWorld/PointFlow 研究。结论基于官方作者仓库、官方数据说明和本地固定提交的源码；本次没有下载新的大模型，也没有训练。

## 结论

存在可以执行的路线，之前把“当前环境没有 `pyzed`”说成“官方视频路线无法继续”是不准确的。需要区分三种数据层：

1. **官方 DROID RLDS/`droid_100`**：直接提供视频帧、动作、关节/机器人状态和语言。官方 DROID policy-learning loader 直接从 RLDS 读这些字段，训练依赖 TensorFlow/Octo/robomimic，不要求 ZED SDK。它可用于 RGB/机器人状态的辅助预训练或数据合同验证，但不能原样提供当前 PointWorld 所需的三维点云。
2. **官方 PointWorld-DROID 处理后 H5**：PointWorld 作者已经把 DROID 视频经过跟踪、深度和标定处理成 H5，并发布恢复脚本。恢复后的 H5 可直接运行官方 `data_integrity_check.py`、`make_wds_manifest.py`、`convert_wds.py` 生成 WDS；这一条路径不再读取 SVO，也不调用 `pyzed`。H5 中已有 RGB、初始深度、内外参和 scene-flow 监督，可以作为独立的 scene-flow 辅助预训练数据。此路线不需要加载 `nvidia/PointWorld_models` 的任何权重，模型可从随机初始化开始。
3. **DROID raw MP4/SVO**：MP4 可以用普通视频解码器读取，但相机内参/立体深度在官方说明中来自 SVO；PointWorld 的 raw 处理代码在 `gather_data_dict()` 入口无条件调用 `_require_pyzed()`。所以“只用 MP4 做 RGB 任务”可以绕过 ZED，“从 MP4 恢复 PointWorld 3-D 标签”不能猜内参，仍需要 SVO/ZED 或一个有来源、可核验的深度/标定替代品。

因此最便宜、证据链最短的实验是：使用已恢复的官方 PointWorld H5/WDS，随机初始化当前共享点云 backbone，增加一个显式短时域的 flow auxiliary head；用官方 H5 的 scene flow 训练该 auxiliary task，再在完全相同的 native downstream 训练和评估中比较有无预训练。该实验研究的是“官方视频派生数据是否能帮助现有表示”，不是官方 checkpoint transfer。

## 官方代码核查

### PointWorld H5/WDS 不依赖 ZED

官方 PointWorld `data` 分支固定在 `3872ec6ee73146aa671192ef79b5dfbedc0246e3`。其 README 的 “Build Train/Eval Datasets from Generated H5” 给出了完整的 DROID 流程：先运行 `data_integrity_check.py`，再生成本地 manifest，最后用 `convert_wds.py` 转换 H5；输入目录是已生成的 `flows-fs-optimized`，不是 raw SVO。见 [data branch README（固定提交）](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/README.md)。该 README 同时把 ZED Python API 放在 raw 数据生成环境的安装步骤中，说明 ZED 是 raw 重建依赖，不是 H5/WDS 训练读取依赖。

官方 main README 进一步明确：`main` 是训练/评估分支，`data` 是数据准备分支；发布的 DROID 是生成后的 H5 包，恢复后可做小子集的 integrity check 和 WDS conversion。见 [PointWorld main README](https://github.com/NVlabs/PointWorld/blob/05484826dfef74cbe278a3974179a5a16705d35d/README.md#datasets-and-checkpoints)。所以已有 H5 包不是“只能拿来展示的标签”，而是作者公开的训练输入格式。

### raw 分支确实要求 `pyzed`

在同一个固定 data 提交中，`real/droid_utils.py` 的 `gather_data_dict()` 首行调用 `_require_pyzed()`，随后从 metadata 找 SVO 路径并通过 ZED API 读取内参、baseline、RGB、深度和时间戳。见 [droid_utils.py](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/real/droid_utils.py#L346-L390)。因此不能把“未安装 ZED”通过伪造内参或把 RGB 当深度来绕过；但这只封锁 raw 3-D 重建分支，不封锁已经发布的 H5/WDS 分支。

作者的 2-D flow 代码还把视频切成固定 `frames_per_clip`，对不足长度的窗口复制最后一帧并记录 mask/有效长度，默认入口 `compute_2d_flows.py` 的 `frames_per_clip` 可改。见 [flow_2d.py](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/real/flow_2d.py#L112-L175) 和 [compute_2d_flows.py](https://github.com/NVlabs/PointWorld/blob/3872ec6ee73146aa671192ef79b5dfbedc0246e3/real/compute_2d_flows.py#L300-L312)。这证明公开实现支持短窗口，但当前任务仍需对有效时间步使用显式 mask，不能把补帧后的时间步误当成真实未来。

### DROID 官方视频/轨迹可以独立读取

DROID 官方文档把完整 RLDS、100 条调试集和 raw MP4/SVO 分开发布：RLDS 的 `droid_100` 约 2 GB；raw 版本才包含 stereo HD MP4 和 SVO。见 [DROID 数据说明](https://droid-dataset.github.io/droid/the-droid-dataset)。官方 [droid_policy_learning](https://github.com/droid-dataset/droid_policy_learning) 的 README 说明 RLDS loader 直接读取 `droid_100`，并列出 `rlds_utils.py`、`dataset.py` 和 stand-alone dataloader；示例读取 `exterior_image_1_left`、`exterior_image_2_left`、`wrist_image_left`、action 和语言字段，见 [官方 dataloader](https://github.com/droid-dataset/droid_policy_learning/blob/master/examples/droid_dataloader.py)。该训练路径没有 ZED 读取步骤。

官方 raw 数据说明也明确 MP4 与 SVO 的职责：MP4 是高分辨率视频，SVO 是包含 ZED 录制信息的原始文件。见 [DROID raw 格式说明](https://github.com/droid-dataset/droid-dataset.github.io/blob/main/droid/the-droid-dataset.md)。作者的 `svo_to_mp4.py` 头部注释明确写着该脚本应在带 ZED SDK 的机器上运行，见 [官方转换脚本](https://github.com/droid-dataset/droid/blob/main/scripts/convert/svo_to_mp4.py)。所以 MP4-only RGB 辅助任务可以跑，MP4-only 3-D PointWorld 标签恢复不应被宣称已经解决。

## 对当前模型的实际接法

当前模型的 native 输入合同包含 `H=4` history、`K=24` future、点级几何特征、point-kind 和动作条件。官方 H5 clip 通常是 11 个采样时刻，且没有当前 native 所需的双手 11 keypoint 动作张量、跨 clip 点身份和现有 `point_kind` 合同。因此不能把 11 帧静默填成 28 帧后直接送入 native trainer。

这不等于数据不能训练。PointWorld 自身发布的 scene-flow任务就是“当前场景 + 未来机器人几何动作 → 未来点流”，其公开 H5 已包含 scene-flow、机器人状态、初始深度和相机几何。可采用以下保持语义的辅助任务：

- 随机初始化共享 PTv3/PointFlow 空间 backbone；
- 读取 H5 的当前点、法向/颜色/深度和 scene-flow；
- 采用 `K_aux=10` 或 H5 实际有效步数的短时域 flow head，并用有效时间 mask；
- 只把该 auxiliary head 的 backbone 表征迁移到 native 任务，native action/effect head 仍按当前数据随机初始化；
- 用同一 native downstream 数据、seed、优化器预算和 Cm-on/off 评价做 matched comparison。

这复用的是官方**视频数据产生的标签**和公开 PointWorld 数据管线，不读取 `small-droid/model-best.pt` 等官方模型参数。若选择 RLDS，则可以先做 RGB/动作辅助前端或机器人状态时序预训练，再通过明确的投影/点云适配进入共享表示；它不能直接冒充 H5 的 3-D scene-flow 监督。

## 本地证据与边界

本 worktree 已经恢复并审计了一个官方 PointWorld-DROID flow shard，以及与同一 DROID episode 配对的 RLDS/raw 文件：H5 clip 可打开，含 11-step scene flow、初始 RGB、深度、内外参和机器人状态；RLDS 含官方视频帧与 action；raw MP4 可用普通解码器读取。对应实验卡是：

- [DROID-100 视频数据审计](../experiments/probes/P-20261010-droid100-video-data-audit.md)
- [PointWorld H5 与 DROID episode 配对](../experiments/probes/P-20261010-droid100-pointworld-match.md)
- [DROID raw 输入依赖审计](../experiments/probes/P-20261010-droid100-raw-input-package-audit.md)
- [native 适配合同审计](../experiments/probes/P-20261010-droid100-native-adapter-contract.md)

这些审计证明“官方数据可取得、可配对、可从 H5 读取”，尚未证明预训练一定提高最终 Cm。官方 checkpoint transfer 的既有运行没有消费任何官方 DROID 视频样本，不能作为视频数据路线的反证。

本次继续做了不含模型的官方管线 smoke：配对 episode 的 10/10 clips 通过
`data_integrity_check.py`；`make_wds_manifest.py` 生成 8 个 train、2 个 test
clips；`convert_wds.py` 写出官方 WDS tar。随后用官方 `decode_data` 和 DROID
scene decoder 读取 train tar，恢复了两个外部相机，每个相机都有 11-step
`scene_flows`、`scene_colors`、`scene_normals`、visibility、depth-valid mask、
RGB、depth、intrinsic 和 extrinsic。这个 smoke 在隔离的项目 `tmp` 依赖目录中
完成，未加载 PointWorld checkpoint，也未调用 ZED。

还做了当前仓库模型的 GPU wiring smoke：从同一 `0:11` clip 选取 128 个全时刻
有效点，按 `K_aux=10` 构造官方绝对轨迹目标并屏蔽其余 14 个输出步，随机初始化
当前 `VideoPointWorldWM` 前向得到 `(1,128,24,3)`，预测和 masked Huber loss
均为有限值（RTX 3090，50,456,707 个当前模型参数，未加载官方 checkpoint）。
这只证明“官方 H5 样本经过显式短时域 adapter 可以进入现有 backbone”；输入特征
在该 smoke 中只填了当前法向量和零占位通道，不能当作预训练收益或最终特征合同。

## 最终判断

“其他仓库各方面都解决不了”这个说法不成立。公开仓库已经提供了三条可复用的基础设施：DROID RLDS loader（无 ZED 的视频/轨迹读取）、PointWorld H5→WDS converter（无 ZED 的 3-D 标签训练输入）和 PointWorld raw flow 管线（需要 ZED，但只在重新从 SVO 生成标签时需要）。当前真正需要实现的是一个显式的短时域 auxiliary adapter 和匹配对照实验，而不是安装官方权重或重新发明整套数据处理。
