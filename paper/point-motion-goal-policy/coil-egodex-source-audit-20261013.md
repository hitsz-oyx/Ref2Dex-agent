# COIL：对现有 EgoDex 的源码兼容性审计

日期：2026-10-13。范围：训练数据合同、动作/本体状态、坐标与时间、mask、可抽取组件及现有环境的极小 CPU 执行。没有安装依赖、修改第三方源码、训练或使用 GPU。

## 结论

**不能把现有 EgoDex 或我们的 video `clip.npz` 直接交给公开 COIL 训练入口。**公开仓库是 Panda/夹爪的仿真采集、行为克隆和仿真评估管线；监督标签是已执行机器人动作。EgoDex 人手关键点不是这个动作空间，补齐 Python 包也不能解决此语义缺口。

**可以复用部分 goal 编码代码，且已用我们真实样本完成有限工程 smoke。**原版 `FlowPositionalEncoder`、`SpatialTemporalTransformer` 和小型 `PointNet2Encoder` 在本地 CPU 实际运行并输出有限 tensor。这个证据只支持组件能处理转换后的 tensor，不支持原版策略能在 EgoDex 训练、控制我们的机器人或提升任务表现。

前一轮“值得借鉴轨迹接口和时序编码”的判断仍成立；如果被理解成“开源 COIL 可以直接跑 EgoDex”，则需要纠正。以下均以本地代码为证据。

## 源码身份与发布范围

- 上游：[realquantumcookie/COIL](https://github.com/realquantumcookie/COIL)。
- 本地：[third_party/coil-egodex-audit-20261013](../../third_party/coil-egodex-audit-20261013)。
- 审计提交：`d9947346c072e4c2c00fcbeed5d7d881393ae8e5`。
- 稀疏 checkout：文本/Python 源码；未下载模型或大资产。资产未 checkout 不等于上游未发布资产。
- [README](../../third_party/coil-egodex-audit-20261013/README.md) 直接声明发布的是 simulation-side pipeline，数据由 scripted reference policy 收集，评估使用 ManiSkill 固定仿真任务。
- [pyproject.toml](../../third_party/coil-egodex-audit-20261013/pyproject.toml) 要求 Python ≥3.10，声明 UniEnv、PyTorch3D、diffusers、tensordict 等依赖；README 另外要求兼容的 ManiSkill/SAPIEN。
- [LICENSE](../../third_party/coil-egodex-audit-20261013/LICENSE) 是 MIT；以后抽取代码须保留原版权与许可。

## 输入及监督合同核对

| 项目 | COIL 源码事实 | 现有 EgoDex/video 合同 | 影响 |
|---|---|---|---|
| 数据 loader | `FlowPolicyDataset` 从 `FlowDataCollectionWrapper.load_replay_buffer()` 建立 `FrameStackedBatch`；需要 `step_data/`、`episode_context/` | 当前是分段 `clip.npz` 加 admission manifest；原始 EgoDex 是人类录像/跟踪数据 | 原版 loader 无对应路径，必须写 adapter 或新 dataset |
| 训练标签 | `act = batch_slice['act'][:, obs_horizon-1:]`；ActorConfig 设置 7 或 10 维动作 | 真实样本有 `hand[28,2,11,3]`、`hand_valid`，没有机器人 `act` | 无直接机器人 BC 标签。不能伪造夹爪或把人手坐标改名成 Panda action |
| 当前本体状态 | feature extractor 使用末端 `pos[3]`、`6d_rotation[6]`、`gripper_pos[1]` | 双手关键点与有效 mask，既非机械末端本体状态，也非夹爪 | 人手预测要替换本体编码与输出头；机器人执行还要真实机器人数据或可信 retargeting/控制闭环 |
| goal | `context['flows']` 为 `[B,T,M,4]`，最后一维是 `xyz, valid` | 我们已有 `[H,32,3]` desired points、mask、current points、offsets/times | 可以转换 shape，但目标实例、对应点身份与未来质量仍须审核 |
| 当前目标跟踪 | `tracked_flow[B,L,M,5]` 是 `xyz, visibility/occlusion channel, valid`，M 与目标点一致 | compact clip 给当前 query、未来点和未来 valid；没有对应点的过去场景轨迹和独立可见性 channel | 完整 feature extractor 不能直接接。仅 L=1 当前目标点可做不同合同的工程适配，不能称原协议复现 |
| 当前场景 | `pointcloud['xyzw']` 当前点云（xyz+valid）；PointNet++ 聚合 | `query_xyz[N,3]`、`query_valid` | 能组成合法 tensor；坐标、采点范围、尺度和 mask 排列要明确 |
| 训练 action loss | diffusion 对 action noise 做普通全 tensor MSE，没有人手点/时间有效 mask | 双手或部分点可能无有效标签 | 改做人手轨迹预测后必须实现 masked loss，不能将缺失手标签零填后直接训练 |

对应证据：

- [train_utils.py](../../third_party/coil-egodex-audit-20261013/flow_manipulation/train_utils.py)：`ActorConfig.construct` 35–57 行硬编码动作维数；`fetch_and_transform_data` 85–115 行读取/转换机器人动作；140–225 行读取 episodic achieved flows、相机上下文；205/263–280 行要求对应点的 tracked history；`FlowPolicyDataset` 313–378 行加载 UniEnv replay buffer。
- [policy_feature_extractors.py](../../third_party/coil-egodex-audit-20261013/flow_manipulation/networks/policy_feature_extractors.py)：57–74 行本体维数固定为10；102 行要求 `use_tracked_flow`；181–259 行实际读取本体、目标点历史、场景点云与 flow；304–307 行只在辅助未来 flow loss 使用 flow valid。
- [common/wrappers.py](../../third_party/coil-egodex-audit-20261013/flow_manipulation/common/wrappers.py)：758–770 行要求 replay 的两个目录；1062–1081 行保存 achieved flows/夹爪轨迹；1177–1203 行保存实际执行 `act`。
- [diffusion.py](../../third_party/coil-egodex-audit-20261013/flow_manipulation/policy/diffusion/diffusion.py)：126–147 行将机器人 `ref_action` 加噪并监督 noise，没有有效标签 mask 参数。

## 坐标、时序及跟踪的关键限制

1. **原版在 robot frame 工作。**[envs/wrapper.py](../../third_party/coil-egodex-audit-20261013/flow_manipulation/envs/wrapper.py) 175–210 行用 robot pose 把相机外参转换到机器人坐标；767–787 行把初始 goal 点关联到有 segmentation 的仿真点云并建立 `ManiSkillPointcloudRef`。EgoDex moving-camera geometry 不是机器人 base 标定。我们 compact 点和未来目标同在当前相机坐标，这足以搭建人手预测组件 smoke，却不证明能直接送入 Panda 控制。
2. **发布的 tracker 依赖仿真真值。**上述 wrapper 767–787 行要求 pointcloud segmentation，初始点离 goal 起点 <5cm，并建立仿真对象点引用；后续831–849行调用该 tracker。它不能直接替换 EgoDex 的 CoTracker/depth pipeline。点轨迹质量问题仍在我们的数据端。
3. **flow 没有物理秒字段。**[flow_encoder.py](../../third_party/coil-egodex-audit-20261013/flow_manipulation/networks/flow_encoder.py) 479–534 行用“有至少一个有效点的时间槽”的累计计数构造 `flows_times`，再归一化位置编码；不读取 source frame IDs、`goal_times_s` 或8/16/24。时间是顺序/相对位置。整槽失效会压缩时间位置；缺点不是静止点，适配时要保留区别。
4. **原版训练也不保留抽样的真实时间间隔。**[pointcloud/flow.py](../../third_party/coil-egodex-audit-20261013/flow_manipulation/pointcloud/flow.py) 98–140 行随机抽起点/中间/终点；`train_utils.py` 193–225 行把点坐标放进 ctx，却不把所选 frame IDs 传入 encoder。若我们要求“0.27/0.53/0.8秒分别到哪里”，要补显式 time embedding；直接拷贝只实现有序几何路径。
5. **PointNet++ 的 mask 要是前缀有效布局。**`flow_encoder.py` 297–334 行说明此前提；实际仅用 mask.sum 传 `lengths` 给 FPS，不根据任意 bool mask 压缩点。EgoDex adapter 必须先筛选/打包有效当前点。此次 r03 所有1792当前点有效，满足此前提；其他数据不能默认满足。
6. **不能直接搬 action 尺度或 horizon。**[env_utils.py](../../third_party/coil-egodex-audit-20261013/flow_manipulation/env_utils.py) 30–42 行默认 `MyPandaDroid`、`pd_ee_pose`、15Hz；[policy/base.py](../../third_party/coil-egodex-audit-20261013/flow_manipulation/policy/base.py) 49–55 行默认 history8/pred16/execute8。`train_utils.py` 91–115 行与 [MyPanda.py](../../third_party/coil-egodex-audit-20261013/flow_manipulation/envs/agents/MyPanda.py) 768–784 行还使用 Panda 速度/步数归一化绝对末端目标。我们是30Hz、history4、future24，人手米制轨迹不能沿用这些动作归一化。
7. **作者的未来 flow 辅助头不等于我们的 Cm。**`policy_feature_extractors.py` 221–263 行辅助预测依赖读取 desired flows 的 feature extractor，没有我们 World 的“当前状态+候选已执行手动作→实际物体后果”合同。复用该头不满足 goal-blind Cm，也不能代替 Cm-on/off 验证。

## 实际执行记录

解释器：`/home2/wyy/miniconda3/envs/graspenv/bin/python`，Python3.8.20、torch2.4.1+cu121。尽管部分叶模块在该 Python 下可导入，整包支持的 Python 下限仍是3.10；不能据此宣称 full install 兼容。

所有执行是 CPU、单线程、无随机权重训练；随机小编码器只用于 shape/finite 合同。测试真实文件：

- [inputs/r03/clip.npz](../../outputs/goal-point-wam/video-object-goal-audit-20261013-r1/inputs/r03/clip.npz)
- [goals/r03.npz](../../outputs/goal-point-wam/video-object-goal-audit-20261013-r1/goals/r03.npz)

| 检查 | 实际输出 |
|---|---|
| 原版 `pointcloud.flow` 导入 | 成功 |
| 原版 `networks.flow_encoder` 导入 | 成功，环境已有 PyTorch3D |
| `train_utils` / `env_utils` 导入 | 失败：`ModuleNotFoundError: No module named 'mani_skill'` |
| 原版 policy feature extractor 导入 | 失败：缺 `tensordict` |
| 原版 diffusion policy 导入 | 失败：缺 `diffusers` |
| `pick_flow_T_index` / `pick_flow_M_index` / `pad_flow` 接真实 clip | 成功：原flow `[25,4,4]`；tids `[0,9,20,24]`；mids `[3,2]`；padding `[6,3,4]`，finite=true，只有2个valid point-time（说明包装成功不表示高质量目标） |
| FlowPositionalEncoder + SpatialTemporalTransformer 接 r03 已注释 goal | 原goal `[1,4,32,4]`，96个valid point-time；输出 `[1,16]`、`[1,32,16]`、`[1,4,32,16]`，均finite |
| 把第2个中间 goal 槽全部置 invalid | 上游 `flows_times` 从 `[0,1,2,3]` 变为 `[0,1,1,2]`，实证时间压缩 |
| 小型原版 PointNet2Encoder 接真实当前点云 | 输入 `[1,1792,3]`，输出 `[1,9,16]`、`[1,9,3]`、`[1,9]`，均finite，FPS/ball-query CPU ops实际运行 |
| clip 的机器人监督字段 | 无 `act`、无 `end_effector_pos` / `end_effector_quaternion` / `gripper_pos` |

核心编码器 smoke 可从仓库根目录复跑：

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python - <<'PY'
import sys, numpy as np, torch
sys.path.insert(0, 'third_party/coil-egodex-audit-20261013')
torch.set_num_threads(1)
from flow_manipulation.networks.flow_encoder import (
    FlowPositionalEncoder, ZeroToOnePosEmbedding, SpatialTemporalTransformer)
from flow_manipulation.networks.transformer import TransformerBlock
z = np.load('outputs/goal-point-wam/video-object-goal-audit-20261013-r1/goals/r03.npz')
xyz = np.concatenate([z['start_object_points'][None], z['desired_object_points']])
valid = np.concatenate([z['current_object_valid'][None], z['desired_object_valid']])
f = torch.from_numpy(np.concatenate([xyz, valid[..., None]], -1).astype('float32'))[None]
pe = FlowPositionalEncoder(ZeroToOnePosEmbedding(16)).eval()
net = SpatialTemporalTransformer(TransformerBlock(16, 2), None,
    TransformerBlock(16, 2), None, num_layers=1, proprioception_input=False).eval()
with torch.no_grad():
    pos, mask, times, *_ = pe(f)
    feature = torch.nn.Linear(3, 16)(pos) + pe(f)[5].unsqueeze(2)
    out = net(feature, mask)
    print(times.tolist(), [list(x.shape) for x in out],
          [bool(torch.isfinite(x).all()) for x in out])
PY
```

这个调用使用 `proprioception_input=False` 的通用 Transformer 组件，未冒充作者完整 feature extractor；没有编造机器人本体状态或 action。

## 分阶段可用性与建议

| 我们想做的阶段 | 审计后的判断 |
|---|---|
| 原版 COIL loader 直接读 EgoDex | 不可直接运行，文件格式与监督语义都不同 |
| 用 EgoDex 直接训练作者 Panda 控制策略 | 不成立；缺已执行 Panda 动作和本体状态 |
| 用 EgoDex 的可信物体 goal 预测未来人手 | 需要新 dataset、手状态编码、手输出头/有效 mask loss、显式时间及坐标合同；这是改编路线，不是作者代码开箱训练 |
| 抽取有对应点身份的时空 goal encoder | 有限工程可行：真实样本 CPU smoke 已通过；需单独验证训练收益 |
| 用已有本地机器人数据训练我们自己的执行头 | 方法上可改编，须用我们的实际机器人 action/state 合同；本审计没有做该训练 |
| 作者策略 checkpoint 直接部署我们的机器人 | 未验证且机器人/动作合同不匹配，不推荐 |
| 使用作者仿真真值 tracker 清洗 EgoDex | 不适用；需独立真实视频跟踪/深度流程 |

最便宜的下一步是在我们的现有模型中只借鉴“同一 keypoint 沿 T 轴保持身份”的 encoder，把物理时间单独输入，仍预测已定义的人手轨迹；用可信的现有合成/原生物体变换先比较成对 endpoint 与短序列 goal。视频可靠目标点不足的问题继续单独处理。不要为填补 `act` 缺口先安装整套 ManiSkill，或仅因叶模块能前向就启动长训练。

尚未证明：作者完整仿真可复现性、预训练权重可用性、EgoDex 改编训练收益、跨 embodiment 控制迁移、真实任务成功率、Cm收益。上述工程 smoke 不提供这些证据。
