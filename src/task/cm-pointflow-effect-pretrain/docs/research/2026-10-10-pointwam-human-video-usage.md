# PointWAM：人类视频怎样变成共同场景/手轨迹，能借鉴到哪里

核查日期：2026-10-10。使用 research 技能，在独立背景代理中检查作者论文、作者项目页及小文件源码；没有下载数据/权重、运行模型或修改训练代码。论文身份：[arXiv:2610.02840v1](https://arxiv.org/abs/2610.02840)，2026-10-02。本文区分论文描述、公开可执行实现、以及针对本项目的推断。

## 当前公开程度

作者项目页仍显示 **Code soon**，代码按钮实际指向本页 citation，且标记 `aria-disabled=true`。本次未找到经作者确认的 PointWAM 训练、预处理、checkpoint 或处理后轨迹数据发布入口。因此下文的方法与收益属于 paper-described，不能称为已经可下载复现的官方流水线。[作者项目页](https://chrockey.github.io/PointWAM/)

搜索结果中的 `xiaooai/PointWAM` **不能作为该论文的官方实现**：作者项目页没有指向它，其 README 描述 DINOv3/SigLIP2、PointWorld PTv3、scene planner 和 flow DiT；论文实际描述 Mosaic3D、Recap-CLIP、Volt 及 learned retargeter。仅凭同名不能建立作者归属；这次不运行该代码。[固定 README，commit 30850013f09896851d5ec66882dcb07f829970c7，行3–12](https://github.com/xiaooai/PointWAM/blob/30850013f09896851d5ec66882dcb07f829970c7/README.md#L3)

小文件证据缓存在仓库 `tmp/pointwam-primary-20261010/`，逐URL状态/字节/SHA见 `source-manifest.json`。作者项目 `project.html` 行86–87保存 disabled/soon 按钮，SHA256 `9ab39994e5c07c6108ca3583a52c3d2d7787332b96a907d8544ca171be709fbd`；论文HTML SHA256 `f31530f51174452ef01e3802be7c62783a30f4cfa84048a22693d0177c41b8f8`；固定第三方README SHA256 `b1f8fde0264cd413fbe7e48f4b4feb77054837b480d710165158227bc898d4eb`。官方训练源码 Git SHA/行号为 **不可得**，不以网页仓库或第三方SHA替代。GitHub API部分请求返回403，不能由它们推断所有作者仓库不存在；公开程度判断主要依据作者项目按钮及没有获得作者代码链接。

## 数据：不是 EPIC，也不是触觉视频

人类预训练来自 EgoDex 的324K episodes，加 VITRA 的824,157较短 episodes，合计1,148,324。EgoDex提供每手25关节的SE(3)标注；VITRA重建可见手的21关节。后续机器人数据来自DexJoCo、RoboDojo-Precision和作者采集的OpenArm demonstrations。该论文没有把EPIC/ObjectForesight或EgoTouch触觉数据列为其预训练来源。[论文§3.4、§4.1、B.1、C.1](https://arxiv.org/html/2610.02840v1#S3.SS4)

注意 episodes、提取clips、训练windows是不同数量：EgoDex的324K episodes形成约5.2M clips，不能把它们当5.2M独立人类任务。[论文B.1](https://arxiv.org/html/2610.02840v1#A2.SS1)

**EgoDex原始3D标注已公开，但不是PointWAM处理后scene trajectories。** 官方仓库现为 `apple-aiml-research/ml-egodex`（原 `apple/ml-egodex` 会跳转）：MP4与HDF5逐帧配对，HDF5有camera intrinsic、手/相机SE(3)及可选confidence；这些transforms在同一记录的stationary ARKit origin中，episode间world origin不保证相同。官方列出的schema没有dense scene depth/point-flow字段；其代码明确是loader和可视化等示例，不是完整large-scale training code。可以复用现成手与相机标注，不能把它说成已发布的12000点共同scene/hand轨迹pack，也不能据此承诺无需scene预处理。[EgoDex作者仓库](https://github.com/apple-aiml-research/ml-egodex)

本次固定EgoDex作者仓库commit `718d9ebd8ec17187f72a74a08c66deea80178cb6`，缓存README、`simple_dataset.py`和`visualize_2d.py`；schema在README行63–85，sample-code边界在行91–95。原始投影还存在Vision Pro多相机合成RGB的perspective mismatch，官方明确提示不能把小的reprojection差自动当作3D joint错标。[固定README](https://github.com/apple-aiml-research/ml-egodex/blob/718d9ebd8ec17187f72a74a08c66deea80178cb6/README.md#L63)

## 一起落到3D空间的处理链

| 环节 | 论文描述 | 对我们重要的边界 |
|---|---|---|
| 手表示 | 每手10点：6个固定Allegro掌部offset按人手palm frame放置，4个真实指尖；缺失手标invalid | 不要求人手与机器人全部关节同构；不能把缺手补零当真实点 |
| 时间 | 每0.5s开启1s clip，10Hz的11帧；腕位移总量不足2cm的clip被排除；线性重采样31帧 | 这是插值和运动筛选，不能宣称31帧独立精确观测 |
| 深度尺度 | VGGT-Ω预测depth；投影已标注/重建手关节，计算hand depth/predicted depth ratios，剔除偏离中位数超过2倍的ratio，再以剩余中位数恢复scale | 有共同可投影的手几何标尺，不能直接继承我们SpaTracker名义米尺度 |
| 场景点 | 640×360，每clip首帧隔2像素查询CoTracker3 offline；按track像素读取scaled depth，经intrinsics/camera poses抬到world | 点身份只承诺clip内持续；离线tracker使用整个片段 |
| 点筛选 | 首帧visible且depth有效，1.5cm体素去重，上限12000点；人手与scene共同转到z-up | 不按未来visible fraction挑输入点；不是只选任务物体上少数corner |

以上来自[论文B.1](https://arxiv.org/html/2610.02840v1#A2.SS1)。论文给出了scale与共同坐标原则，但未发布可核对的相机/手坐标字段映射、每条sequence对齐变换和生成源码；VITRA的手也是重建量，不是由这篇论文独立验证的真值。因而不能说该公开论文已经替我们解决所有metric/world/visibility问题。

**Observation 与 teacher 要分开理解。** Forecaster接口是当前colored point cloud、当前手点、language；不是未来真实手作为条件，不是完整未来视频作为推理输入，也没有本地Probe的H4速度/加速度合同。人类视频的offline轨迹/深度用于准备监督，机器人推理从当下RGBD与状态获得点。论文没有证明人类阶段的“当前”重建字段只访问当前/prefix，深度scale使用chunk中的手，CoTracker3明确offline；所以这条人类预处理不能直接被我们称为严格因果 observation frontend。[论文§3.1、B.1、B.2](https://arxiv.org/html/2610.02840v1#S3.SS1)

这与本地[ObjectForesight/SpaTracker合同核查](2026-10-10-objectforesight-spatracker-track-contract.md)的关键教训相容：全视频teacher labels可以存在，但复用其历史字段作为在线输入，需要另行验证prefix合同。该判断是我们的实施边界，不是在声称PointWAM论文的实际部署访问了未来视频。

## 训练的东西与实际机器人接法

1. 冻结Mosaic3D场景编码与Recap-CLIP文字编码；场景/当前手落在1cm共同voxel网格，经stem、5³ voxel patch、Volt transformer融合，两个head分别预测未来scene与hand位移。[论文§3.2、B.3](https://arxiv.org/html/2610.02840v1#S3.SS2)
2. 人类阶段随机初始化forecaster，联合训练scene和hand轨迹；scene有效目标有运动软权重，hand按有效点步平均，位移按horizon/坐标分别标准化。它没有训练触觉目标。[论文§3.4、B.5](https://arxiv.org/html/2610.02840v1#S3.SS4)
3. 机器人阶段微调forecaster，同时从头训练retargeter。后者读取预测的未来hand trajectories与current robot state，输出末端位姿及手指关节action chunk；scene预测只监督共享backbone，不直接进入retargeter。Retargeter来自robot demonstrations，不是IK求解器，也没有显式collision/joint-limit约束。[论文§3.3、B.4](https://arxiv.org/html/2610.02840v1#A2.SS4)
4. DexJoCo replay产生rigid scene真轨迹，FK产生robot hand轨迹。RoboDojo没有可恢复scene轨迹时，冻结scene head，fine-tune仅手轨迹和action；缺监督不会被伪装成完整scene标签。[论文B.2](https://arxiv.org/html/2610.02840v1#A2.SS2)

训练主模型的预算是4×B200、global batch192、人类60K步约34h、机器人60K步约19h；不是我们500步少clip Probe的等价设置。人类31帧跨度1s，DexJoCo31帧跨度0.6s，两阶段物理时钟也不相同。[论文C.1–C.2](https://arxiv.org/html/2610.02840v1#A3.SS1)

## 收益对照究竟支持什么

- 主结果平均DexJoCo success为69.0%，最强比较方法57.3%；十任务，每任务50episodes×3 evaluation seeds。[作者项目results](https://chrockey.github.io/PointWAM/#results)
- 从scratch12.1%到EgoDex-only60.1%是约48pp；加全VITRA到69.0%后才是摘要的56.9pp。不能把56.9pp都写成EgoDex单源收益。[论文§4.3](https://arxiv.org/html/2610.02840v1#S4.SS3)
- 控制ablation中，去scene trajectory supervision而保留depth/backbone/hand head：53.5%；3D scene supervision：64.4%，增10.9pp。该reference是较小Volt-L12+EgoDex，不是主结果Volt-H+全corpus。Mosaic3D64.4%对lifted DINOv3 59.0%；semantic hand points64.4%对512 surface points44.6%。[作者项目component results](https://chrockey.github.io/PointWAM/#results)、[论文§4.4](https://arxiv.org/html/2610.02840v1#S4.SS4)

它支持作者设置中“共享3D scene/hand轨迹预训练后，机器人示范微调可获益”。它没有执行我们的action-conditioned Cm-on/Cm-off、自训练RL在线样本预算比较；没有证明human未来手是可控机器人干预，也没有证明我们的raw sensor teacher应有效。论文还说明主模型checkpoint在第一evaluation seed选20K/40K/60K、reference预训练按held-out hand error选snapshot；这些与本项目固定final-checkpoint协议不同。[论文C.1–C.2](https://arxiv.org/html/2610.02840v1#A3.SS1)

## 对当前路线的具体建议（推断，不是已经实施）

当前[Mission](../../../../../docs/MISSION.md)要求短期action-conditioned Cm促进自训练机器人策略；PointWAM的联合future-hand/action生成不是替代目标。它提供共同坐标scene/hand表示与迁移对照的方法参考，但目前缺官方可执行实现，**不建议自行复刻PointWAM来回答“视频是否帮助当前模型”**。

用户最新约束是“尽量先找开源，不要自己写，目的是看视频能否帮助我们的模型”。因此下一步优先挑选**已有官方代码/权重/processed data且能保留当前动作条件接口**的参考路线，核对真实输入与预训练来源后做最小matched transfer比较；不要将joint-action imitation policy的收益直接迁移成Cm收益。PointWorld这类已有动作条件world-model入口比待发布PointWAM更适合先核查，具体公开代码/processed data合同由本次并行的PointWorld调查负责，本文不替它作未经核实的可执行承诺。

EgoDex官方手/相机标注可以成为未来复用入口；只有找到对应已开源处理工具或已处理scene+hand产物后，再决定是否进入人类视频适配。届时对照应固定当前Cm架构/动作接口/训练预算，比较有无视频初始化或表征迁移，保留actual/shuffled action与held-source effect评价；先问“是否帮助现有模型”，不先新写一个视频模型。材料不足就停在数据gate，不能把EPIC稀疏Contact+少数RGB corner改名为论文的12000点合同。本次没有启动下载、模型运行或自写新模型。

PointWAM没有触觉预训练证据。触觉作为可选teacher应继续遵循本地未知硬件角色/不可由manual false推断物理无接触的边界；可借鉴的主要是共享交互表示和迁移验证，而不是把姿态到sensor的拟合直接当作物理动力学证明。
