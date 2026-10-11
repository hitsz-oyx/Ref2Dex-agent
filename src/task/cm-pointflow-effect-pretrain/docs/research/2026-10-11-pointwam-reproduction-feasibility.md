# PointWAM Pipeline 可复现性核查

核查日期：2026-10-11。本文只使用 PointWAM、EgoDex、VITRA 和作者/项目维护者的一手资料，不下载大数据、不修改外部仓库。这里区分“严格复现论文结果”和“复现核心 pipeline”；二者不是同一个目标。

## 结论

**严格复现当前做不到，核心 pipeline 可以做，但需要重新建立 PointWAM 的数据合同和一条较重的预处理链。**

严格复现要求同时具备作者训练代码、作者 checkpoint、完全相同的 1.15M 人类 episode 处理结果、机器人示范处理、训练配置和评估环境。PointWAM 作者项目页目前仍把代码标为 **Code soon**，按钮没有链接到训练代码或 checkpoint；因此公开材料足以重建方法，但不足以逐字节重放作者实验。[PointWAM 作者项目页](https://chrockey.github.io/PointWAM/)

核心 pipeline 的实现是可行的：官方 EgoDex 提供逐帧视频、相机内参、手部 SE(3) 标注和语言；官方 VITRA 提供人手视频的元数据、相机/手部重建和训练代码；PointWAM 论文明确给出了共同场景/手关键点、窗口、损失和两阶段训练合同。代价是我们必须自己实现论文没有公开的 PointWAM 预处理、模型和 retargeter，且只能把结果标成 **paper-faithful reimplementation**，不能称为官方复现。[EgoDex 官方仓库](https://github.com/apple-aiml-research/ml-egodex)、[VITRA 官方仓库](https://github.com/microsoft/VITRA)、[PointWAM 论文](https://arxiv.org/html/2610.02840v1)

## 官方 pipeline 的实际合同

PointWAM 不是“把视频直接喂给 native effect head”。它的接口分成两阶段：

```text
当前 RGB-D scene points + 当前 semantic hand points + instruction
    -> future scene displacement + future hand displacement
    -> robot stage: forecast hand trajectory + current robot state
       -> action chunk
```

论文将场景点和手点放在同一 3D 坐标系中，以当前时刻为原点预测位移；场景 head 预测每个观测点，手 head 预测每手 10 个语义关键点。scene trajectory 只监督共享 trunk，retargeter 在机器人示范上从手轨迹和当前 robot state 预测末端姿态/手指关节 action。[PointWAM §3.1–§3.4](https://arxiv.org/html/2610.02840v1#S3.SS1)

人类预处理的关键细节是：

* EgoDex 每手 25 个关节的 SE(3)，VITRA 每个可见手重建 21 个关节；二者都映射到每手 10 点（6 个 palm offset + 4 个拇指/食指/中指/无名指指尖，小指省略），缺手标 invalid，不补零。
* 每个 episode 切成 1 秒、10 Hz、11 帧的 clip，步长 0.5 秒；腕位移总和小于 2 cm 的 clip 丢弃；线性重采样为 31 帧。
* 每个 clip 的首帧在 640×360 图像上隔 2 像素放置 CoTracker3 query，离线跟踪；VGGT-Ω 预测深度后，用手标注恢复米尺度，再用内参和相机位姿把 tracks 抬到世界坐标。
* 首帧可见且有有效深度的点保留，按 1.5 cm 去重，最多 12,000 点；query 每个 clip 重新开始，所以 point identity 只在 clip 内成立。
* 场景和手先旋到机器人使用的 z-up frame，再进 1 cm voxel、5³ voxel patch 和 Volt transformer；冻结 Mosaic3D 场景编码器与 Recap-CLIP 文字编码器。

这些细节都不是我们之前的 H5 `H=4,K=24` 或 raw sensor 目标。它们要求 metric depth、可投影手标尺、clip 内点轨迹和语义手点同时存在。[PointWAM Appendix B.1/B.3](https://arxiv.org/html/2610.02840v1#A2.SS1)、[PointWAM §3.2](https://arxiv.org/html/2610.02840v1#S3.SS2)

## 数据和代码是否公开

| 组成 | 官方公开状态 | 对复现的含义 |
|---|---|---|
| PointWAM 训练/预处理代码 | 作者项目页为 `Code soon`；未列官方训练仓库、PointWAM checkpoint 或处理后 scene/hand trajectory pack | 严格复现目前缺少最关键的实现和产物 |
| EgoDex | 官方仓库公开 schema 和下载入口；训练约 725 小时，分成 5 个约 300 GB 压缩包；test 约 16 GB，additional 约 200 GB | 可先用 test 做 loader/坐标/手点审计；完整预训练需要很大的原始数据和自建 scene-track 处理 |
| EgoDex 标签 | 每个 MP4 配一个 HDF5；包含 camera intrinsic、camera/hand transforms、可选 confidence、语言属性；同一 episode 内是 ARKit stationary origin，episode 间 world origin 不保证一致 | 手部和相机标注可复用，但官方没有发布 PointWAM 的 12,000 点 scene-flow pack |
| VITRA 代码和 metadata | 官方代码、模型和 VITRA-1M metadata 已发布；README 列出 1,222,918 个 episode，metadata 约 100 GB | VITRA 代码可复用，但 PointWAM 使用的是自己筛选后的 824,157 shorter episodes；不能把 README 总数直接当 PointWAM 输入 |
| VITRA 原始视频 | 处理后视频因许可不能直接提供；需从 Ego4D、Epic-Kitchens、EgoExo4D、Something-Something V2 官方来源申请/下载，再按官方脚本去畸变 | 没有原始视频或合适许可时，VITRA 分支无法严格重建 |
| Mosaic3D | NVIDIA 官方仓库和 Hugging Face 模型公开 | 可复用冻结场景编码器，但仍需核对 PointWAM 使用的具体 encoder/checkpoint 与输入适配 |

EgoDex 官方仓库明确说明数据是 829 小时、30 Hz、1080p，配有头部、上身和双手 3D pose；官方代码只是加载、可视化和 metrics 的示例，不是大规模 PointWAM 训练代码。[EgoDex README 数据说明](https://github.com/apple-aiml-research/ml-egodex#dataset-access-and-download)、[EgoDex README schema/code 边界](https://github.com/apple-aiml-research/ml-egodex#dataset-structure)

VITRA 官方文档说明 metadata 总量约 100 GB，并明确指出处理后视频不能直接发布；原始视频需要用户从四个源数据集取得并执行去畸变。官方 metadata 约 90% annotation accuracy 是人工检查估计，仍需要在 PointWAM 的 depth/hand/track 过滤链中做质量审计。[VITRA data preparation](https://github.com/microsoft/VITRA/blob/main/data/data.md#2-data-download)、[VITRA video preprocessing](https://github.com/microsoft/VITRA/blob/main/data/data.md#3-video-preprocessing)

同名但非作者确认的第三方 `xiaooai/PointWAM` 不能当官方实现：它的 README 描述 DINOv3/SigLIP2、PointWorld PTv3、scene planner 和 flow DiT，与 PointWAM 论文描述的 Mosaic3D、Recap-CLIP、Volt 和 learned retargeter 不一致；作者项目页也没有指向它。它最多是另一个实现路线，不是严格复现依据。[第三方 README](https://github.com/xiaooai/PointWAM/blob/30850013f09896851d5ec66882dcb07f829970c7/README.md)、[作者项目页](https://chrockey.github.io/PointWAM/)

## 三个复现等级

| 等级 | 可行性 | 需要完成的工作 | 可以声称什么 |
|---|---|---|---|
| L0：数据合同 smoke | 现在可做 | 只下载 EgoDex test，解析 HDF5/MP4、把手映射为 10 点、检查 z-up、clip、invalid mask、深度/相机字段 | 官方数据能否进入 PointWAM 格式 |
| L1：核心人类 forecaster | 可做，成本中高 | 自己实现 CoTracker3 + VGGT-Ω 轨迹生成、手尺度恢复、scene/hand 联合 31-frame 训练；先 EgoDex 子集，再接 VITRA | PointWAM-style 共同 3D scene/hand 表征是否能训练；不能复现 69.0% |
| L2：机器人迁移 | 可做，需 DexJoCo 环境 | 用 FK 生成 10 个 robot hand points，回放 DexJoCo 获得 scene trajectories，按论文的 action normalization 训练 retargeter，并运行 ten-task protocol | paper-faithful 两阶段 pipeline 和有限规模 transfer |
| L3：严格论文结果 | 当前不可审计 | 需要官方 PointWAM code/checkpoint/处理后轨迹或完整重建等价物，加完整 EgoDex/VITRA、4×B200 级训练预算和所有 benchmark 细节 | 才能有资格比较论文的 69.0%/56.9pp claim |

论文给出的主设置是人类预训练 60k steps、global batch 192、4×B200、约 34 小时；DexJoCo 微调再 60k steps、约 19 小时，主模型用 1,148,324 个 human episodes。人类窗口 31 帧约 1 s，DexJoCo 窗口也是 31 帧但对应 0.6 s；PointWAM 的主结果不是 500-step、单 episode probe 的同等实验。[PointWAM Appendix C.1/C.2](https://arxiv.org/html/2610.02840v1#A3.SS1)

## 为什么之前的尝试没有复现它

之前的失败诊断是正确的，但现在可以更精确地说：

1. DROID/H5 有 scene flow 或 state，却没有 PointWAM 要求的共同 metric hand keypoints、future hand target 和 clip 内 scene/hand coordinate contract；把 state 广播到 scene point 不能代替 hand branch。
2. EPIC/ObjectForesight 的稀疏/弱 point windows 没有达到 31-frame、metric depth、10 semantic hand points、最多 12,000 scene points 的完整合同。
3. 我们的 scene-only、raw tactile、native effect 和 CHORD probes测的是不同目标；PointWAM human stage 预测 future scene + future hand，action retargeter只在 robot stage 学习。
4. 一次 checkpoint-transfer 还把官方 PointWorld权重加载到了 OakInk2/GRAB/ARCTIC 数据；官方 PointWAM 视频数据没有进入那次训练，不能把它当成 PointWAM 失败。
5. 规模、窗口和训练预算差距很大：本地 probe 主要是少量 clip/数百步，论文使用约 1.15M episodes、60k steps，并在 DexJoCo 做闭环 ten-task 评估。

详细的本地失败归因见 [ref11 失败诊断](2026-10-11-ref11-failure-diagnosis.md)。这些负结果只能关闭具体数据合同或配方，不能关闭 PointWAM 的核心假设。

## 建议的最小可行路线

如果目标是“复现 PointWAM pipeline”，而不是立即复现论文数字，最稳妥的顺序是：

1. **先做 L0 数据 gate。** 只用 EgoDex test 16 GB 中一个任务，检查 HDF5 hand/camera transforms、语言属性、30 Hz frame alignment 和 10-point conversion；不下载 VITRA 全量、不训练大模型。
2. **再做 EgoDex-only L1。** 先生成少量 31-frame clips；用官方 Mosaic3D encoder，暂时固定一个明确的 depth/track backend；同时监督 scene displacement 和 hand displacement。若 hand/scene mask、米尺度和 point identity 不通过，停止扩展数据。
3. **然后接 DexJoCo L2。** 复用机器人 replay/FK 合同，先跑一个 single-arm task，验证 retargeter 的 action chunk、relative action normalization 和 closed-loop evaluator，再扩展十任务。
4. **最后才接 VITRA。** 只有 raw source videos、metadata、intrinsics 和 undistorted output 都在本地并通过审计，才把 VITRA 计入 human pre-training；否则报告应明确为 EgoDex-only core reproduction。

这个路线能给我们可审计的中间结果，也能直接解释此前失败来自哪一层。它不会凭少量数据或单个成功 rollout 宣称 PointWAM 论文结果，也不会把官方 VITRA/PointWAM 权重误称为视频数据本身。

## 最终判断

* **Exact reproduction：现在不能保证。** 阻塞点是 PointWAM 官方代码、checkpoint 和处理后 trajectory release 缺失，加上 VITRA 原始视频许可/下载与大规模预处理成本。
* **Core pipeline reproduction：可以。** 需要从 EgoDex test 开始，自建 10-point hand + metric scene track + 31-frame + scene/hand joint supervision，再接 DexJoCo retargeter；这是一项新的、可审计的复现工程。
* **对本项目当前 claim：不能直接把它当成现有 Cm/native effect 路线的替代证据。** 它最适合作为独立的 shared 3D scene/hand representation transfer 对照；若要证明视频帮助当前模型，还要在固定 action/effect 接口下做 matched transfer。
