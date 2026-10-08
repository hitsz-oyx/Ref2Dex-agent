# ObjectForesight-EPIC：公开数据与接入合同核查

日期：2026-10-07。范围：只读官方数据卡、Hub API 与小文件/HEAD 请求；不启动 GPU，不恢复已暂停的 EgoDex 重建，不下载大文件，不修改 ref 文档或训练代码。

## 结论

ObjectForesight-EPIC 已经发布处理后的物体轨迹，值得作为候选数据源优先核查。它提供重建 mesh、相机/深度及 FoundationPose 物体姿态，可省去逐片段执行完整物体重建。**目前尚不能说可直接加入当前 hand-conditioned PointWorld 训练**：真实数据访问需要账号授权；发布合同没有原生 3D 手动作；mesh 与姿态的尺度、时钟和重初始化段必须用实际样本验证。

这不是要求继续修复或重跑 EgoDex→ObjectForesight，而是复用作者已经产生的提取结果。数据是估计得到的伪标签，不能等同于真值运动捕获。[官方数据卡](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC)

## 已验证：发布与下载门槛

- 官方 API 返回 `private=false`、`gated="auto"`，当前 revision 为 `0473697d241ccc670b8413b1de93034b695944eb`。公开页面要求登录接受访问条件，填写姓名、邮箱、机构、用途，并确认非商业研究和不识别视频中个人。`auto` 是访问配置，不表示匿名可下载。[官方 API](https://huggingface.co/api/datasets/raivn/ObjectForesight-EPIC)
- 匿名实际请求：`README.md`、`LICENSE` 为 HTTP 200；`trajectories.parquet`、`extrinsics_conv.json`、示例、loader 源文件 GET，以及最小 tar HEAD，均为 **401 `GatedRepo`**。没有尝试绕过访问控制，也没有提交用户身份或替用户接受条款。请求记录在 `tmp/ref6-epic-check/access-check.json`。[版本固定的 metadata 文件](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC/resolve/0473697d241ccc670b8413b1de93034b695944eb/data/metadata/trajectories.parquet)
- 先探测 `hf-mirror.com`：README、API、tar 路径均返回 **308 跳转到 Hugging Face 官方**，因此没有证据表明该镜像可独立提供本数据。随后按用户指定使用 `127.0.0.1:7897` HTTP(S) 代理，官方元数据访问正常；代理不会替代账号授权。记录在 `tmp/ref6-epic-check/mirror-check.json`。
- 官方 `LICENSE` 为 CC BY-NC 4.0；要求非商业使用、署名 ObjectForesight 与 EPIC-KITCHENS，并遵守 EPIC 条款。具体用途仍需与许可匹配。[官方 LICENSE](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC/blob/main/LICENSE)

## 已验证：规模与最小获取方式

数据卡报告 **29,006 clips、34,286 object trajectories**，对象级 train/val 为 **29,499 / 4,787**，来自 34 人、633 源视频。原始提取结果没有提前划成训练窗口；论文的大量窗口是在 loader 中生成的。已有 train/val 划分应保留，不把 val 当新增训练来源。[官方数据卡](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC)

实际 API 文件树枚举 **659 个文件、624 个 tar shard**；tar 总大小 **942,162,032,640 bytes ≈ 877.46 GiB ≈ 0.857 TiB**，metadata parquet 仅 **572,662 bytes**。数据卡约 0.84 TiB 是近似描述；633 源视频与 624 个 shard 的差异原因尚未核查，不据此判定数据缺失。[官方文件树 API](https://huggingface.co/api/datasets/raivn/ObjectForesight-EPIC/tree/main?recursive=true&expand=false)

最小 shard 示例（真实 API 尺寸，尚未下载内容）：

| Shard | Bytes | MiB |
| --- | ---: | ---: |
| `data/shards/P07_106.tar` | 10,936,320 | 10.43 |
| `data/shards/P26_25.tar` | 14,530,560 | 13.86 |
| `data/shards/P08_12.tar` | 14,970,880 | 14.28 |

因此小规模尝试不需要下载整个 0.86 TiB。获得授权后，先取 parquet、split 和相机约定 JSON，再筛选 train 中满足长度和刚体对象要求的一个小 shard；单纯最小 tar 不保证包含符合当前 4+24 帧合同的对象。不要直接运行官方无 `--include` 的全量下载命令。可限定文件，例如：

```bash
hf download raivn/ObjectForesight-EPIC --repo-type dataset \
  --revision 0473697d241ccc670b8413b1de93034b695944eb \
  --include 'data/metadata/*' 'data/metadata/splits/*' \
  --local-dir outputs/cm-pointflow-effect-pretrain/ref6-epic-probe
```

这是待授权后执行的方案，**本次未执行数据下载或账号操作**。具体 shard 再单独用 `--include 'data/shards/PXX_YY.tar'` 获取；按视频打包意味着同 shard 可能跨 train/val 对象，解包后仍按对象 split 过滤。HTTP Range 是否可用于 tar 内选择、是否有成员索引，目前没有验证，不承诺只下载单 clip。

## 数据合同：明确字段与未知项

官方声明每个 action clip 包含 RGB、SpaTracker 相机/深度/稀疏轨迹、EgoHOS masks；每个对象有 SAM2/amodal masks、TRELLIS GLB、FoundationPose 姿态与质量日志、手接触标志。具体关键路径为 `trellis/model.glb`、`foundationpose10/poses.npz`、`run_summary.json` 和 `track_log.csv`。深度以 float16 存储，量化精度不等于估计误差。[官方数据卡](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC)

`poses.npz` 明确包含 `frame_ids (T,)`、`T_c_o (T,4,4)` 和 `init_from_frame`，其中 `T_c_o` 是 object→camera。`frame_ids` **可能不连续**。每 clip 的 extrinsics 是 c2w 或 w2c，作者提供 `extrinsics_conv.json`；数据卡称大多数 c2w、782 个 trajectories 为 w2c，必须逐 clip 读取而非全部统一假定。[官方相机与姿态约定](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC#camera--pose-conventions)

在符合相机约定且同步的条件下，c2w 可用 `T_w_o = T_w_c @ T_c_o`；w2c 先求逆。此公式只解决变换方向，**不自动解决 mesh 尺度、坐标原点、时钟或重初始化问题**。

| 核查项 | 当前结论 |
| --- | --- |
| 物体 mesh + 6DoF 是否已经提供 | 官方发布合同明确提供；样本内容因 gated 尚未直接验证 |
| 固定 30 Hz / 固定 24 步对应 0.8 s | 数据卡没有保证；需读取 MP4 FPS/时戳并审计 frame gaps |
| 原生左右手 3D semantic joints / 未来手动作 | 发布合同没有列出；只有手 mask、检测和接触标志，不能替代 3D action |
| 原始 TRELLIS GLB 与姿态是否同尺度同 object frame | **未验证，存在具体风险**，见下节 |
| 质量日志能否表示真实标注精度 | 只能用于筛选、自一致性审计；没有独立真值，不能宣称实际厘米精度 |
| 是否可把不同 source/reinitialization 段拼成连续轨迹 | 不可默认；须按原始帧和 `init_from_frame` 检查与切段 |

## 必须先查的尺度风险

本仓库已经下载的原始 ObjectForesight 核心会在 tracking registration 中估计 mesh scale，并在内存调整 mesh；`run_summary.mesh_path` 仍可指向未经该调整的 TRELLIS GLB。[固定版本原始 step10 源码](https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/step10_fpose.py) 中 scale apply 约在第408、542、1067行。root 对本地 cube6 产物实际复算：原始 GLB 三轴 extent 约 **1.001**，最终供 pose 使用的 `mesh_metric.ply` 三轴约 **0.02552**，比例约 **0.02550138**。证据在 `tmp/ref6-epic-check/mesh_scale_reproduction.json`。

这证明**本地原始流程确实有尺度转换**，并不证明 Hugging Face 已发布的 GLB 有错；作者可能在发布阶段另作处理，当前未能读取 shard 验证。ref6 的“固定采样 canonical GLB，再直接乘 `T_w_o`”需要先确认 scale/centering 的来源与一致性，否则会得到约几十倍大小偏差而 SE(3) 矩阵仍合法。当前 512 canonical points 的对应身份、单位和姿态必须一同审计。

## 训练仓库合同补充

root 独立取得作者模型/训练仓库的 `dataset_epic.py`、`fpose_io.py`、`spatrack_io.py`、配置与 loader，小文件副本在 `tmp/ref6-epic-check/`。这是 **2026-10-07 查询到的 main，未固定 Git commit**，本地字节 SHA-256 记录在 `tmp/ref6-epic-check/metadata-sha256.json`，其中 `dataset_epic.py` 为 `2c58ee66cbc79fb7211225c21fe3402122cd5954135cde5ffbddfc993adf6980`。

- [`dataset_epic.py`](https://raw.githubusercontent.com/RustinS/ObjectForesight/main/src/data/datasets/dataset_epic.py) 第866行起 sample 含 scene point cloud、物体 past/future poses、mesh 路径、source frame IDs；没有当前任务需要的 future semantic hand action。第600行按已有 pose 行索引选窗口，并在602–605行要求同一 `init_from_frame` 组；这不是固定 FPS 或 source frame 连续性的证明。
- [`fpose_io.py`](https://raw.githubusercontent.com/RustinS/ObjectForesight/main/src/data/fpose_io.py) 从 `poses.npz` 读取 `frame_ids` 与 `T_c_o`；[`spatrack_io.py`](https://raw.githubusercontent.com/RustinS/ObjectForesight/main/src/data/spatrack_io.py) 负责相机/深度输入。它们没有证明 GLB 与 pose 同尺度。
- [`epic.yaml`](https://raw.githubusercontent.com/RustinS/ObjectForesight/main/conf/epic.yaml) 配置 `H: 8`、scene `n_points: 20000`，并关闭 `load_hand_poses` 和 `use_hand_context`。作者这一训练设置不能直接代替当前 4+24、512 canonical object points、左右手动作输入合同。

数据集 gated loader 尚未实际读取，不能声称模型仓库 loader 与 dataset 打包 loader 完全一致。任何模型 ADE/FDE 都是预测相对于其评价标签的误差；没有独立标注真值时，不能拿它作为伪标签本身的厘米精度保证。

## 建议的最小决策 Probe（未执行）

问题：作者已处理的 train shard，能否转换为尺度/时钟/刚体身份正确的物体未来轨迹？结果决定是否先采用其 object-only 辅助预训练，再评估补手动作的成本。

最低成本步骤：获得官方访问后取 metadata 和一个小 train shard；检查 GLB vs pose scale/centering、RGB FPS、source frame gaps、重初始化切段；选一段连续 4+24 窗口，固定采样512点并用姿态传播，在源 RGB/depth 上重投影核对。先区分纯数据工程失败与伪标签质量问题，不启动完整重建或大训练。

如仅 object-only 数据合同成立，应显式选择支持该监督的训练入口；不得把缺失的 hand action 填零伪装成完整 hand-conditioned 样本。补手 3D 动作或引入混合监督会改变实现与评价，须另作有边界的 Probe。**不依据当前材料断言补手成本低，也不认为动作噪声必然无害。**

本次结果：数据发布与授权门槛已核实；实际样本尺度/时钟、可用窗口和与当前训练输入的适配仍未验证。
