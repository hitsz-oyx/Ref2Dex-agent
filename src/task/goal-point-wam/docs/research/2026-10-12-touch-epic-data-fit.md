# Touch / EPIC 对 Goal-Point-WAM 的数据适配判断

日期：2026-10-12。Task：`goal-point-wam`。这是等待主线结果期间的主源调研，不是新训练实验。检查官方论文、数据卡、源码说明，以及本仓库已完成的相关数据合同与 Probe；未下载数据、使用 GPU 或修改模型。服务器连接由 root 单独核查，本笔记不替代连接结果。

## 当前建议

**Touch 可以作为接触表征的独立辅助数据，当前优先核查 T-Rex 的少量、多对象 episode；原先 EPIC 不能直接加入现有联合手/物体轨迹训练。** 这是合同适配判断，不是“触觉无用”或“EPIC 无用”的结论。多卡和较大 batch 可以提高训练覆盖与稳定性，但不会补齐缺失标签；已有极小训练的负结果也不足以关闭一个数据类型。

现行模型需要：同一个当前 anchor 坐标系中的 3-D 场景、4 帧已观测手历史、未来24步双手22语义点轨迹、物体32配对 start/end points，以及逐物体 rigid effects。未来手是 trajectory 标签/world 分支的执行动作，不是 goal 分支的观测输入。参考 [Task 入口](../../README.md)。数据只有 RGB、归一化触觉或估计 object pose 时，不能当作拥有全部这些监督。

| 来源 | 可补充的信号 | 直接接入主 loader 的缺口 | 最便宜的接入方式 |
| --- | --- | --- | --- |
| T-Rex | 真实机器人关节状态/目标动作、指尖触觉图像与估计6维 wrench | 无已发布逐物体 mesh/metric pose/point-flow；机器人手拓扑不是当前人手22点 | 独立 state/action + F6 表征辅助，先按 episode 审计，不混写 native world 标签 |
| EgoTouch / TouchAnything | 实测手套传感信号与 processed 手面 pressure grid、估计手姿 | 当前本地 hand 没有经审计的世界尺度/外参，缺物体轨迹；pressure/bend 与 Newton wrench 不同 | 重用已取得10条 recording 的 masked grid；新的研究问题才开新 Probe |
| EgoTac-SC / Aether | RGB + 左右 MANO778点的 tactile/contact 监督 | processed 接口没有现行物体效应/执行手轨迹合同 | 共享手面表示和缺监督 mask 的设计参考；独立视觉触觉辅助 |
| ObjectForesight-EPIC | 视频重建 object mesh/depth/camera/6DoF，弱物体动力学 | 无原生 dense 手动作；尺度/时钟/相机方向/点身份需审计 | object-only 弱监督侧路；若仍要联合手轨迹，先补 dense hands |
| EPIC-Contact | 拟合手物几何与人工 contact 先验 | 非独立 MoCap 动力学；clip pose 使用 WiLoR 手运动传播 | 单帧 contact/相对几何辅助，不能当实测触觉或干预动力学 |
| RBO | RGB-D、物体/关节姿态、实测工具 FT | 无灵巧手执行点轨迹；工具负载补偿未完成 | 已有 adapter 的 object/wrench 合同检查，独立多交互辅助 |

## Touch：可复用什么，不能夸大什么

T-Rex 官方发布5,464 episodes、约50小时、30 Hz、207对象；state/action 均为58维关节位置，F6为左右五指各六分量，另有3路 RGB 与20路 raw/deform tactile video。官方提供按 episode 拉取与 URDF/FK replay；公开数据卡标注 MIT，不需要为了审计同步约1.53 TB全量数据。[官方数据卡](https://huggingface.co/datasets/zekaiwang/trex_dataset)、[quickstart](https://github.com/ZhuoyangLiu2005/T-Rex/tree/main/dataset_quickstart)

这里的 F6 是从触觉传感图像估计的指尖 wrench，不能写成直接测量的物体 COM 净 wrench。数据卡的2026-10修订说明 single-frame blackout 以 previous-frame hold 填充，每传感器用最初15个 clean frames的中位数 tare；当前 README显示提交 `8b47840`，与仓库旧审计固定的数据 revision 前缀一致。接入时要记录处理版本与修复来源，不把重复值自动当作静止接触 GT。单位/指尖轴与物体坐标变换仍按旧审计的未确认项保留。[数据更新说明](https://huggingface.co/datasets/zekaiwang/trex_dataset#tactile-data-update-2026-10)、[已有T-Rex/RBO审计](../../../cm-pointflow-effect-pretrain/docs/research/2026-10-11-trex-rbo-wrench-adapter-audit.md)

相机参数和 URDF 能支持机器人状态/视觉表征，但官方 wrist CAD 外参注明厘米级精度并有 Sharpa mesh projection 偏移，因此 FK/投影不能自动成为毫米级物体标签。[相机参数源码](https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/hardware_code/camera/camera_params.yaml) T-Rex 的方法启发是把过去的一段触觉序列编码成 temporal token，并独立快速更新触觉分支；它的 VLA 成功率不等于我们的 goal 条件轨迹已有收益。[论文](https://arxiv.org/abs/2606.17055)、[官方实现](https://github.com/ZhuoyangLiu2005/T-Rex)

EgoTac 的共享 MANO 手面、contact/force 分头、缺失标签 mask 可以借鉴。processed Zarr明确包含 RGB、左右778点 tactile/contact与 episode 边界；raw HDF5/SVO及 glove-to-MANO转换不在公开源码中。不能把 EgoTouch21×21 grid直接 reshape 为1556维标签，也不能把 EgoTac 模型预测当实测力。[EgoTac官方入口](https://github.com/Mr-Zwkid/EgoTac)、[processed合同](https://github.com/Mr-Zwkid/EgoTac/blob/87ba7059304bd581050315f3a9af3029710ed690/docs/process_data_aether.md)

EgoTouch 本地已有10条记录的 processed targets，本次确认目录及 manifest 仍存在。WiLoR 的 `aligned_to_vive=false`、稀疏217 finite grid cells、混入 bend、全记录 max 归一化及 nominal row 时钟等问题已审计，不重新把公开字段名当世界标定证明。[已有工程审计](../../../cm-pointflow-effect-pretrain/docs/research/2026-10-11-ref10-engineering-audit.md)、[TouchAnything源码](https://github.com/Jianyi2004/TouchAnything)

## EPIC：具体是哪套，已有失败能说明什么

本仓库原先研究的是 **ObjectForesight-EPIC + EPIC-Contact 的配对**，不是只有原始 EPIC-KITCHENS视频。ObjectForesight公开的是29,006 clips/34,286对象轨迹；论文“2M+”是 loader 划出的短窗口，不是独立采集episode。它使用 segmentation、估计depth/camera、重建mesh和FoundationPose得到弱3-D标签；`T_c_o`、逐clip c2w/w2c和可能不连续的 `frame_ids` 必须显式处理。[官方卡](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC)、[官方生成链](https://github.com/RustinS/ObjectForesight-Data)

EPIC-Contact官方说明 annotation/fitting而非MoCap，clip-level pose使用WiLoR手运动传播。推论：从同一传播过程生成的手与物体标签学习相关性，不能证明独立接触力/真实动作的因果效应；单帧稳定抓持接触先验仍可利用。[项目标注流程](https://sid2697.github.io/epic-contact/#dataset)

已有严格本地审计在34个clips里找到95个候选starts，**0个合格H4+K24手窗口**。这不是训练太短，而是保守 semantic-point validity、source quality、clock/bracket及split条件下没有训练资格。plate/bottle source membership不同，不能用 ObjectForesight train覆盖Contact test。[readiness实验卡](../../../cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-video-data-readiness.md)

放宽到 point-only后已有h8训练：78 TRAIN/71 dev重叠窗口，只有2个dev clips满足预注册h8物体支持；model object macro24.808 mm，static9.464 mm。这个小配方UNPROMISING，样本和domain shift使总体路线仍UNCLEAR；它没有检验ObjectForesight作者刚体pose目标，也不足以关闭video预训练。不能仅增加batch后把旧 LK/depth 标签视为可信3-D GT。[point-only实验卡](../../../cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-video-point-dynamics.md)

## 可以做的下一次 Decision screen（本次未执行）

1. **先查已有文件，而非先下载。** 远端只读 inventory定位 T-Rex parquet/episode metadata、EgoTouch processed/标定、ObjectForesight shards/新dense hand。记录路径、revision、bytes、字段；没有文件时按 ModelScope→HF国内镜像→代理顺序找 bounded sample，不同步全量数据。
2. **T-Rex优先的文件审计：** 一到两个 episode用于 parser smoke；拟开展表征实验时预先按object/primitive划分约20–40个独立episodes（具体下载byte预算需按指针估算）。检查时间gap、连续H4/K24窗口数、修复段与F6 finite/active coverage、指序/关节序和单位状态。单episode不能判断跨对象泛化。
3. **训练要服务现行输出，而非重复未来触觉MLP。** 合同通过后，比较同预算的原生点encoder与加入独立触觉辅助预训练的encoder；native hand/world验证指标、moving/static分层及goal ablation共同决定是否继续。外部数据不提供的loss置missing mask，不补造object效果。使用可装下的较大batch、多卡并行对照；batch、步数、见过的独立episode和样本数分别记录，不将更大batch本身视为规模证据。
4. **EPIC重新打开的gate：** 远端若有更密的手标注，先固定source/time/object identity、split交集、semantic mask和尺度，统计多个独立clips的合格H4+K24覆盖。只对新增数据运行readiness，保持旧失败审计；没有dense hand时object-only auxiliary可以另试，但不作为完整WAM数据。弱标签生成使用整段视频时保留retrospective provenance，部署输入必须单独证明截至当前。

这些是推荐，不是已授权的远端写入/大规模下载或已完成的训练。新增auxiliary需要在root批准的资源内登记experiment card及matched control；本次不改变主线goal条件实验。

## 用户配合与证据边界

T-Rex公开数据适配与本地EgoTouch审计无需用户提供额外标注。ObjectForesight官方Hub需要账户接受EPIC条件；只有现有credentials确实不具备访问权时，才需要用户在自己的账户接受条款/配置访问，不把token写入文档。远端登录是否需要用户处理以root实际连接报告为准。

如果希望touch作为最终机器人实时输入，还需明确部署端是否真的有触觉硬件、坐标与标定；目前可以先作为训练auxiliary，不因此阻塞主线。pressure、估计F6、实测工具FT、几何contact和object COM net wrench必须保留source/type/mask，不能统一命名为真实contact force。

已完成EgoTouch processed history与future-hand oracle小Probe都输给persistence；它们只关闭相应窄配方，不关闭更多数据的接触表征辅助或T-Rex。RBO单序列未来wrenchMLP也只测试那段chronological tail，没有检验多交互point-flow辅助。[pressure history卡](../../../cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-egotouch-processed-history.md)、[future-hand卡](../../../cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-egotouch-processed-action.md)、[RBO卡](../../../cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-rbo-wrench-auxiliary.md)
