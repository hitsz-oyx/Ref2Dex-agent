# UniDexGrasp / UniDexGrasp++ 的刚体数据接入核验

日期：2026-10-14。所属 Task：`video-data-preparation`；目标消费者：`ptv3-shared-wam`。
用户暂停 Open-AoE 扩展，转而要求试用这两个开源项目。本笔记只核验第一方论文、源码和
官方下载服务；本地子集实测由对应实验卡记录。没有把论文成功率当作本项目的训练收益。

## 当前判断

值得先试：它们提供可直接使用的刚体几何、明确缩放和机器人抓取姿态，能减少视频线路的
目标误选与单目物体几何重建成本。但发布的抓取标签主要是**静态抓取终态**，不能直接
替代我们的 H4 观察与未来 24 步手物轨迹。UniDexGrasp++ 复用原数据，主要新增策略学习
代码和 state-based checkpoint；没有查到独立发布的大规模连续示范轨迹库。
[原始项目](https://pku-epic.github.io/UniDexGrasp/)、
[++ 官方 README](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/README.md)

上述“没有查到”限定于当前官方发布入口、静态 loader 与仓库资产，不推断作者内部没有
轨迹。后续可运行官方策略采集真实仿真 rollout，届时才会产生有物理时钟的动态训练数据。

## 规模、对象和机器人

原论文的 proposal 数据为 112 万有效抓取、5519 个对象、133 类；对象归一化到单位球后
按 `0.06/0.08/0.10/0.12/0.15` 五种因子缩放，仿真中按刚体处理。生成数据的稳定性与
穿透筛选是作者原始生成协议，尚未独立复验下载子集。
[论文 §4.1、附录 A](https://arxiv.org/html/2303.00938v2)

policy README 的 split 是 3200 个 train、141 个 seen-category test、100 个 unseen-category
test。这个抓取执行集合与上述 proposal 全集是不同口径。仓库附带一个训练对象示例，
可以先验证读取与几何，不必立即下载全量。
[policy 数据说明](https://github.com/PKU-EPIC/UniDexGrasp/blob/36c9bfcf7cedc987dd15e2384c548e51d5aadc21/dexgrasp_policy/README.md#L37-L41)

机器人为单侧 Shadow Hand。生成代码通过 MJCF 和 forward kinematics 构造手网格，
不是 MANO，也不是我们的 Inspire 手；不能直接复用其关节角当作我们的机器人控制量。
可以定义 wrist、5 个指根、5 个指尖的 FK landmark，映射到 11 点合同；未提供的另一手
必须 invalid。finger link 原点与指尖不是同一点，需采用明确的局部偏移或 mesh landmark。
[官方手模型构造](https://github.com/PKU-EPIC/UniDexGrasp/blob/36c9bfcf7cedc987dd15e2384c548e51d5aadc21/dexgrasp_generation/datasets/shadow_hand_builder.py)、
[++ 运行时指尖偏移](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/tasks/shadow_hand_grasp.py#L543-L570)

## 实际字段与两个容易踩错的尺度语义

| 发布/读取路径 | 源码中的字段或行为 | 对本流程的含义 |
| --- | --- | --- |
| 原始 `datasetv4.1/<object>/*.npz` | `qpos` 字典、`scale`、`plane` | 一个抓取姿态与一个桌面支撑关系；文件编号不能当作连续帧 |
| generation `DFCDataset` | 腕 translation/Euler、finger qpos、`pcs_table.npy`、`poses.npy` | 对象/桌面点云与静态 grasp，可审计坐标变换；没有 future24 |
| ++ `datasetv4.1_posedata.npy` | `target_qpos`、`target_hand_pos`、`target_hand_rot`、`object_euler_xy`、`object_init_z` | 按对象/scale 保存初始化与终态信息；其 state-based task 主动将三个 target 手字段置零 |
| policy 运行时状态 | root pose、DOF、rigid-body states、object pose/vel | rollout 时可导出真实手物序列；这些是运行时 tensor，不是已发布标签序列 |

原始 policy 明确把 npz 的 `scale` 解释为**尺度倒数**：实际因子为 `round(1 / scale, 2)`。
不要把标签内约 `16.67` 直接当作 16.67 倍物体。`plane` 经 `plane2euler` 重建桌面上的
初始姿态。不要脱离官方坐标关系单独套用手的 root transform。
[原始静态标签读取](https://github.com/PKU-EPIC/UniDexGrasp/blob/36c9bfcf7cedc987dd15e2384c548e51d5aadc21/dexgrasp_policy/dexgrasp/tasks/shadow_hand_grasp.py#L179-L206)、
[generation loader](https://github.com/PKU-EPIC/UniDexGrasp/blob/36c9bfcf7cedc987dd15e2384c548e51d5aadc21/dexgrasp_generation/datasets/dex_dataset.py#L104-L125)

`meshdatav3_scaled/.../coacd_006.urdf` 等是**已缩放**几何；++ 加载这些文件后将 actor scale
设为 1.0。不能再乘一次 `0.06`。++ 不使用 grasp-pose goal 的行为也意味着它的 checkpoint
不自动提供“按任意给定终态抓取”的能力。
[++ 数据读取及 target 置零](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/tasks/shadow_hand_grasp.py#L171-L189)、
[++ 已缩放对象加载](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/tasks/shadow_hand_grasp.py#L293-L314)、
[++ actor scale](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/tasks/shadow_hand_grasp.py#L444-L456)

## 获取路径与本次真实访问情况

官方推荐[北大镜像入口](https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023/)。网页 URL
返回 Vue 页面，不能由 HTTP 200 判断已经获得数据；实际目录/文件 API 是同站的
`/files/dl-release/UniDexGrasp_CVPR2023/`。2026-10-14 实际 GET 目录、HEAD 文件成功：

| 官方文件 | 当前服务报告字节 | 约下载量 |
| --- | ---: | ---: |
| `DFCData/datasetv4.1.tar.gz` | 2,920,161,280 | 2.72 GiB |
| `DFCData/meshdatav3.tar.gz` | 51,685,765,120 | 48.14 GiB |
| `dexgrasp_policy/assets/meshdatav3_scaled.tar.xz` | 441,222,940 | 420.78 MiB |
| `dexgrasp_policy/assets/meshdatav3_pc_fps.zip` | 619,907,863 | 591.19 MiB |
| `dexgrasp_policy/assets/meshdatav3_pc_feat.zip` | 26,844,762 | 25.60 MiB |

这些是压缩文件大小，不是解压总占用。本次只核验文件服务，没有下载以上全包。
数据目录没有展开的单标签入口；最小检查优先用 GitHub 仓库自带示例。
[原始标签/mesh 目录 API](https://mirrors.pku.edu.cn/files/dl-release/UniDexGrasp_CVPR2023/DFCData/)、
[policy 资产目录 API](https://mirrors.pku.edu.cn/files/dl-release/UniDexGrasp_CVPR2023/dexgrasp_policy/assets/)

`mjcf` 目录提供可逐文件获取的 XML、mesh、contact/penetration points。++ 另列
`datasetv4.1_posedata.npy` 的 [Google Drive 文件](https://drive.google.com/file/d/1DajtOFyTPC5YhsO-Fd3Gv17x7eAysI1b/view)；
其当前下载大小和匿名可用性未独立核验。没有发现必须先用 HF 才能取原始数据的要求。
[官方 MJCF 目录](https://mirrors.pku.edu.cn/files/dl-release/UniDexGrasp_CVPR2023/mjcf/)、
[++ 官方下载说明](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/README.md#L67-L69)

## 适配到 Shared-WAM 的最小路线

1. 静态几何 smoke：少量 grasp + 对应已缩放 mesh/MJCF，核验米制坐标、支撑平面、
   手 FK、手物接近关系和可视化。通过只说明静态读取可行。
2. 如做终态/接触辅助训练，使用独立静态 profile/loss，保留缺失动态标签的 mask；
   不能将终态复制 24 次或线性插值冒充抓取执行。
3. 如要进入现有动态预训练，运行可用策略的短 rollout，导出 time、当前 scene、手 11 点、
   物体 SE(3)；当前选择固定 mesh 表面 query，未来仅用物体 pose 传播同一批点。
   按对象实例分 train/val，避免同一 mesh 不同 scale 的泄漏。

第 2、3 步是接入建议，尚未验证训练收益。++ 配置标注控制为 60Hz；实际 rollout 必须
记录 `sim.dt × controlFrequencyInv`，再对齐我们 30Hz 的 H4/K24，不能直接把 24 个
60Hz step 当作 0.8 秒目标。
[++ 控制配置](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/cfg/shadow_hand_grasp.yaml#L18-L24)

策略采集需要 Isaac Gym 及兼容 checkpoint，真实 scene 可从模拟 RGB/depth/segmentation
生成；官方 vision task 包含 camera tensors、view/projection matrices，但完整 CAD scene
属于仿真几何输入 profile，不能伪装成真实视频的局部观测。
[vision 数据链路](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/tasks/shadow_hand_random_load_vision.py#L83-L121)

原始项目 README 标注数据/工作 CC BY-NC 4.0；++ 根 LICENSE 和继承组件可能不同，
后续复用代码时逐组件保留原许可。此处不将仓库代码许可替代数据许可。
[原项目许可说明](https://github.com/PKU-EPIC/UniDexGrasp/blob/36c9bfcf7cedc987dd15e2384c548e51d5aadc21/README.md)

## 同日追加：用开源策略采集轨迹的准备条件

用户建议先运行策略采集再适配训练。这条路线在数据语义上合理：得到真实仿真动态，
而且对象身份/尺度/pose 已知。当前只完成 checkpoint 元数据检查，**尚未执行 rollout**。
下载的官方 `dexgrasp/example_model/state_based_model.pt` 为 17,220,395 字节；它是裸
state dict，actor 权重形状为 `300→1024→1024→512→512→24`。隐藏层均 ELU、输出层
不加激活，官方 deterministic inference 返回 actor mean，无需采样噪声和 critic。
[实际 checkpoint 审计](../../../../../outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1/checkpoint-audit.json)、
[官方 actor 实现](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/algorithms/rl/ppo/module.py)、
[官方启动脚本](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/script/run_train_ppo_state.sh)

最小独立 replay loader 可以严格加载 `actor.*`，按上述层结构运行，不需要 PPO 训练器、
PointNet 或整个 DAgger 依赖。上游 `module.py` 无条件导入 PointNet，且构造器先读取
`backbone_type/freeze_backbone`，而公开 state PPO config 未包含这两个键；属于可绕过
的入口依赖问题，不能通过 `strict=False` 忽略 actor 权重。`algo/pn_utils` 实际存在于
上游仓库，只是当前下载子集不完整，不能称为上游缺失代码。
[上游 PointNet 文件](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/algo/pn_utils/maniskill_learn/networks/backbones/pointnet.py)、
[state PPO 配置](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/cfg/ppo/config.yaml)

独立 loader 仍必须保持官方输入和动作语义：观察裁剪 `[-5,5]`、动作裁剪 `[-1,1]`；
`obs[207:236]` 为零，`obs[236:300]` 为 `0.1 ×` 对象的已发布 64 维 `pc_feat`，
不能用随机或全零 feature 替代。24 维动作是 root force/torque 的 6 维，加 18 个
actuator target，而不是直接执行 22 个 Shadow joint qpos。
[官方 wrapper](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/tasks/hand_base/vec_task.py#L125-L151)、
[观察及控制实现](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/tasks/shadow_hand_grasp.py#L655-L706)

该对象各 scale 的 feature 可单文件下载：++ 的
`assets/meshdatav3_pc_feat/sem/Car-669043a8ce40d9d78781f76a6db4ab62/pc_feat_{006,008,010,012,015}.npy`。
原始仓库相同文件位于 `dexgrasp_policy/assets/` 前缀下；不要误用 `after_relu_pc_feat`
版本。++ 仓库没有捆绑完整 `datasetv4.1_posedata.npy`，原仓库有
`dexgrasp_policy/assets/posedata_small.npy`，其字段/对象范围必须先检查；也可按原
label loader 的规则从静态标签生成当前单对象的初始化数据。
[单对象 feature 示例](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/assets/meshdatav3_pc_feat/sem/Car-669043a8ce40d9d78781f76a6db4ab62/pc_feat_008.npy)、
[原始 small posedata](https://github.com/PKU-EPIC/UniDexGrasp/blob/36c9bfcf7cedc987dd15e2384c548e51d5aadc21/dexgrasp_policy/assets/posedata_small.npy)

实际资源 blocker：远端现有 Python 3.10 环境没有 Isaac Gym/gym/transforms3d；本地
已有 graspenv Python 3.8 能找到这些包，但当前 6 个活动 GPU 达到共享预算，未额外
分配 GPU。本地既有 Gym 仅提供 Python 3.6/3.7/3.8 native bindings，不能直接搬到
Python 3.10。正式启动须先导入 Isaac Gym 再导入 torch，并将 Gym 的 C++ extension
cache 定向到项目输出目录。以上是本机源码与环境检查，未运行模型或模拟；不能由
“包存在”推断 runtime 兼容。
[远端环境预检](../../../../../outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1/remote-runtime-preflight.json)

下一步只需一张可用 GPU、单对象少量环境的短 replay，先看抓取/抬升是否正常及是否能
导出完整手物帧。通过后再扩对象与生成训练 shard。当前 example checkpoint 的实际
对象覆盖和此环境成功率仍待 replay；不由文件名推断它能稳定处理全库。
