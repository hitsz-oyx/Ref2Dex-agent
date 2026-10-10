# 别人实际怎样使用触觉数据：TouchAnything 与 EgoTac

日期：2026-10-10。问题：是否有可参考的开源实现，怎样迁移到本 Task 的视频/触觉预训练？
范围：primary-source 阅读与小源码缓存；未下载数据、weights，未训练、未修改现有代码。
承接 [原始通道合同](2026-10-10-egotouch-tactile-channel-contract.md)，不重复 raw/grid 重建。

## 用户建议

**有开源代码。按用户“先用开源、检验视频是否帮助现有模型”的要求，优先复用已经发布的模型，暂不新写触觉预测模型。**
TouchAnything 是理解 EgoTouch 作者真实数据消费方式的首要参考：RGB＋WiLoR 手姿 → 同时刻双手传感器图。
但本次检查的官方 README、pinned 源码 tree 和推理入口中，没有找到可直接下载的官方 TouchAnything trained checkpoint；
这是本次检索范围内的结果，不是断言权重不存在。原任务也不同，可先搁置其重训。
EgoTac 已发布 h1/h4 weights，可以作为现成视频触觉表示候选；它把信号/几何 contact 统一到 MANO 手面，分别 mask force/contact supervision。
两者都不能原样解决“未来手轨迹条件下的物体 3D 点动力学”：公开目标主要是当前触觉/接触，缺少这个任务的物体运动监督。

近期重点是开源视频表示是否给现有 Cm 增益；触觉只是候选表示之一，不是必须先解决的主线blocker。
不要因为作者称作 pressure/contact 就把 mixed grid 或模型预测当 Isaac 实测force。
现有 Cm 是point encoder，没有RGB入口；不能直接把DINOv2/EgoTac权重当作兼容初始化。
需要先核对现有接口能否使用冻结特征或teacher auxiliary，再决定matched小Probe，不能把跑通作者demo称为Cm受益。
早期两样本用于grid合同重建；当前已核对10个官方TRAIN raw records与10个chest RGB，
并完成4fit/5held任务的330/535窗及六臂RGB Probe；当前raw/RGB筛查没有通过增益gate，尚未进行native辅助收益验证。
这些当前结果不能升级为视频或触觉整体无效的结论，详见[Task STATE](../STATE.md)。

| 发布状态（2026-10-10只查manifest） | 可直接复用到哪一步 | 未验证 |
|---|---|---|
| TouchAnything training/converter/inference源码已公开 | 数据处理/同帧估计方法参考 | 本次未发现官方trained checkpoint入口；论文run完整复现未知 |
| EgoTouch raw已公开，本地10个TRAIN records/10个chest RGB已核对并完成六臂Probe | 两样本历史schema证据＋当前多任务数据/模型筛查 | 全release完整性/真实设备精确同步和processed HDF5训练包未知 |
| EgoTac h1/h4 `.pt`＋`.json`已发布，HF gated=false | 官方checkpoint inference，不必自训sensor模型 | 尚未下载/加载，不能称当前服务器已跑通 |
| EgoTac-SC processed ZIP分片已发布，HF gated=false | 官方Zarr训练链路候选 | 未下载/解压，数据完整性与磁盘成本尚未核验 |

EgoTac model revision `40be21c77f4e2eb1e33742b5c0dace385cd98e57`，
dataset revision `3506c3e11e3134e77d4328edbaf9fd219b0ee0a3`；只读HF API manifest确认实际文件存在。
h1和h4每个checkpoint为 `933,773,086 bytes`（约934MB）；不应默认同时下载两份，processed ZIP也不应整库先拉取。
[官方权重文件](https://huggingface.co/wkzhang/EgoTac/tree/40be21c77f4e2eb1e33742b5c0dace385cd98e57)、
[官方processed数据文件](https://huggingface.co/datasets/wkzhang/EgoTac-SC/tree/3506c3e11e3134e77d4328edbaf9fd219b0ee0a3)

## 固定版本与可查看源码

- TouchAnything 官方：`Jianyi2004/TouchAnything@d74f9ef5c189a957b7ff72781a0c998e41b45a56`。
- EgoTouch 原始发布：`zhouzhoujy/EgoTouch@cfdbb0ac31cc2af4247943820aa250575e7e6637`，沿用已有样本证据。
- 唯一对比工作 EgoTac：`Mr-Zwkid/EgoTac@87ba7059304bd581050315f3a9af3029710ed690`。
- 缓存：`tmp/video-tactile-primary-sources/touchanything-d74f9ef5c189a957b7ff72781a0c998e41b45a56/`
  和 `tmp/video-tactile-primary-sources/egotac-87ba7059304bd581050315f3a9af3029710ed690/`。
  converter 使用此前缓存 `TouchAnything-scripts__core__convert_to_hdf5.py`。
  缓存为对应 immutable raw URL 的原字节；API tree rate limit 后直接读取 pinned GitHub tree/源码。

下文“源码事实”只适用于该固定版本，不等于论文实际运行 checkout/checkpoint 的保证。

## TouchAnything：真实训练链路

| 环节 | 代码实际行为 | 固定源码 |
|---|---|---|
| 默认 launcher | 指向 `touchanything_with_glove_aug_wilor.yaml` | [launcher L12–15][launcher] |
| 数据集选择 | `dataset=touchanything` → `TouchAnythingDataset` | [build_dataset L37–55][build] |
| 输入 | 同索引 ego/双腕 RGB、WiLoR 左右各21 joints；pressure不作为模型输入 | [loader L279–348][loader]、[train L258–264][train] |
| 目标 | 同一8帧的双手21×21 grids，预测输出与目标直接比较，无future offset | [loader L350–367][loader]、[forward L160–197][model] |
| clip | 8帧、frame_interval=2；覆盖15源帧，训练窗口步长15，不重叠 | [config L89–90][config]、[loader L143–150][loader] |
| 网络 | frozen DINOv2 ViT-B/14，三视角融合、temporal transformer、pose cross-attention、joint decoder | [config L11–60][config]、[forward][model] |
| 训练目标 | NaN sensor mask下 weighted MSE＋L1，另加全图TV；target>0.1权重3 | [loss L63–91][loss]、[config L148–153][config] |

**时间结论：它是同帧传感器图估计，不是未来触觉预测。**
TemporalTransformer 的 temporal attention L135–141 没有 causal mask；一个 clip 内早期输出可看到后面RGB。
推理会遍历所有完整窗口、给同一帧的重复输出取平均，也不是直接在线单向预测。[temporal][temporal]、[inference L325–409][inference]
若改为只预测窗口末帧，它可以使用截至当前的8帧；这属于我们重新定义的任务，原论文结果不能直接作其验证。

姿态来源默认 `wilor` 且无fallback；invalid frame的整手xyz置为-10、坐标clip，默认 `invalid_pose_policy=keep`。
PoseEncoder 又按固定 offset/scale 缩放；它没有从本 Task 的metric手轨迹继承几何合同。
loader中“z通常10–50米”等注释不能替代标定；迁移时需重新审计坐标、尺度和validity。
[config L68–72/L133–135][config]、[loader L312–334][loader]、[pose encoder L153–164][pose]

每个腕视角训练时独立按0.3 dropout，ego总保留；推理允许ego-only，仍要求pose输入。
这直接支持“缺腕相机的辅助表征”设计，但没有训练 tactile-history缺失或robot joint-domain适配。
[view dropout L107–131][views]、[model输入约束][model]

## raw、HDF5、归一化和空间映射

源码 converter 读取 `chest/left/right.mp4`，使用三视频最短长度；timestamps按30FPS重新合成。
帧数一致和合成timestamps不能证明真实多设备瞬时同步。
[converter L674–720][converter]

converter 的有效pressure路径是直接读 `pressure_grids.npz`；不在这里从256raw实现mapping/baseline/bad-column。
它保存预生成左右grid和少量max metadata；不会因为类/docstring写“preprocessed in converter”就补全缺失producer。
[converter L591–626/L842–867][converter]

默认训练的 `TouchAnythingDataset` 消费 grid，mapping路径只是兼容参数；
`src/data/hdf5_dataset.py` 是另一套generic读取raw/可选baseline的utility，**不是当前默认 train 入口**。
不能用generic loader支持raw的事实推断作者本实验训练raw256。
[build_dataset][build]、[grid读取][loader]、[generic loader L209–224][generic]

训练grid helper：NaN→0用于tensor，`~isnan`单独作sensor mask；需要resize时grid线性插值、mask最近邻。
already-normalized时vmax=1，否则用当前clip左右grid最大值、至少1再缩放。
当前 converter 没写 `normalized` attr；但已有发布grid在[0,1]时fallback vmax仍为1，不能因此声称再次改了数值。
对未知未归一化release，这个fallback可能引入clip未来值，不能当future forecasting通用normalizer。
[loader L166–192/L350–364][loader]、[converter保存属性][converter]

原始record maxima和首帧baseline限制沿用前note：未来pressure进入history时应raw/255或TRAIN-only统计，
不能逐record用未来峰值缩放。mixed grid、复制raw来源和坏列imputation也应保留provenance。

**新增作者软件语义证据**：`configs/hand_joint_positions.json` 明示 `display_grid_21x21`；
metrics把该文件(row,col)作为bend区域排除。
CPU比对：左/右各66 positions，均与前note候选bend raw分组投到display坐标后的66格完全一致。
因此可称为“作者评价实现明确排除的bend区域”；这是软件合同证据，不是硬件force/contact校准。
JSON右手 `sensor` 数字仍是208..222，不能照抄为右手raw通道；实际metric只读取row/col。
[positions][positions]、[metrics L6–71][metrics]

## 切分和评价：不要只看指标名称

论文写episode80/10/10及seen/unseen objects；当前 split generator 按 Scene/Task 挑 `test_unseen` tasks，
其余任务的episodes汇总shuffle后切train/val/test_seen，没有使用object-instance ID。
所以该generator保证的是held-out task key，不自动保证真正 unseen object、subject或session。
实际发布split是否与论文物体划分相同仍 UNKNOWN；要沿用split文件且独立查overlap。
[split L221–296][split]、[论文实验协议](https://arxiv.org/html/2605.13083v1#S4.SS1)

没有split_file时loader仅按排序后的HDF5列表ratio切分，不能当作task-balanced random split。
config写100 epochs，论文写25；公开默认文件和实际论文run不可混为一份复现manifest。
[loader L114–130][loader]、[config L130][config]、[论文训练细节](https://arxiv.org/html/2605.13083v1#A2.SS6)

验证训练循环只累计tactile val_loss并据此选best；Contact IoU等在单独inference阶段算。
推理的eval mask按“有预测帧”决定，没有把pose_frame_valid纳入该mask。
[train L369–403/L713–716][train]、[inference L723–743][inference]

| 指标/监督 | 当前代码有效区域与规则 | 含义边界 |
|---|---|---|
| training MSE/L1 | 仅NaN sensor mask；**未排除bend区域** | mixed-grid reconstruction |
| training TV | 对整个pred图求空间TV，未应用sensor mask | 手形外/复制/插值也可影响正则 |
| TemporalAcc/F1/onset-offset | 排NaN及bend；inference threshold0.1且接触比例至少5% | 派生软件contact，不是manual_contact |
| ContactIoU | 排GT NaN，**未排bend** | threshold后的mixed-grid IoU |
| VolIoU / MAE | 排GT NaN，**未排bend** | normalized grid magnitude误差 |

源码 `compute_tactile_metrics` 把 `exclude_bend_sensors` 只传temporal系列，
不是统一过滤后再算所有指标。MAE docstring写physical units，但这里输入是归一化grid，不能报N/Pa误差。
IoU union=0返回NaN，汇总按trajectory nanmean；无contactepisode不能静默当完美IoU。
[metrics L244–311/L515–651][metrics]、[inference L951–956][inference]

## 唯一对比：EgoTac怎样混用实测触觉和几何contact

EgoTac有完整公开loader、policy和dataset转换入口。与本问题最有关的是同一MANO表面支持不同监督类型，
不是其宣传中的“force”一词。

- pinned `train_h4`：4帧ego RGB → 1帧1556维（左右778 MANO vertices）目标；
  dataset `tactile_offset=-1`默认取 `obs_horizon-1`，即最后观测帧，不是未来帧。
  手姿字段config `ignore_by_policy=true`，loader实际pose读取被注释。
  [config L13–14/L24–88][egocfg]、[loader L197–199/L1751–1764][egodata]
- 主6source：EgoTac-SC/Aether、HOT3D、HOI4D、TACO、H2O、ARCTIC；没有EgoTouch。
  config允许missing tactile；loader用0值＋**0 force mask**区分无监督，另读contact target/mask。
  [config L94/L131–141][egocfg]、[loader L1597–1607/L1764–1789][egodata]
- policy分别做masked force MSE、独立contact BCE head及低权重contact-force consistency。
  没有force标签时不应把contact当force真值；这一结构适合我们的source-specific auxiliary loss。
  [policy L94–107/L187–274][egopolicy]
- tactile先log1p，可zscore；normalizer可缓存而非逐episode峰值。
  本次没有完整追到stats helper，**不认证其所有split的统计隔离**，只能借鉴冻结normalizer设计。
  [loader L930–1088/L1633–1635][egodata]
- 其EgoTac-SC/Aether有独立采集协议；guide指向processed Zarr，原始HDF5/SVO和glove→MANO转换不在source release。
  不存在已验证的EgoTouch256raw→该MANO顺序/force尺度桥接。
  [官方Aether guide L3–24][egoguide]、[前note中的硬件/mapper边界](2026-10-10-egotouch-tactile-channel-contract.md)

EgoTac可以参考“共享手面表示＋缺失监督mask＋分开的contact/force目标”。
不能直接把EgoTouch grid喂给其1556维目标，也不能将EgoTac推断force当真实测力补充我们Isaac训练。

## 下一步：优先开源复用，直接服务视频增益判断

1. 暂停自写future sensor MLP和TouchAnything全量重训；上述合同问题记录为复用限制，不为完善触觉资料持续扩大任务。
2. 若主代理判断触觉视频表示最可能改变Cm决策，先核查EgoTac官方h4推理所需checkpoint大小、依赖及数值输出接口，
   在资源允许后复用作者入口对少量已有RGB做冻结推理；保留模型预测与实测触觉的区别。
   这项候选优先于继续扩写当前raw-sensor/RGB自建路线；本次开源调研仅查主源，尚未执行EgoTac推理。
3. 只有接口审计通过，再使用现有训练/评估入口构造“原Cm vs同初始化＋冻结开源视频特征/teacher”的matched Probe，
   不改变原生物体运动label、split、训练预算和选择规则；以现有物体point误差/移动与静止分层指标判断增益。
   若RGB/point空间接口接不上或必须重建大路线，则先选其他更匹配的开源视频路线，不把触觉变成阻塞项。
4. 准备同帧EgoTac信号作历史输入时，核查官方推理窗口截至当前；TouchAnything原双向同帧估计不能静默当future-free输入。
   本报告不证明video→Cm迁移收益，两者官方实验也没有验证本Task的手轨迹条件物体动力学。

[launcher]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/run_train_ddp.sh#L12-L15
[config]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/configs/touchanything_with_glove_aug_wilor.yaml
[build]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/src/data/__init__.py#L37-L55
[loader]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/src/data/touchanything_dataset.py
[generic]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/src/data/hdf5_dataset.py#L209-L224
[model]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/src/models/touch_anything.py#L121-L197
[temporal]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/src/models/temporal_transformer.py#L125-L151
[pose]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/src/models/pose_encoder.py#L142-L164
[views]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/src/models/multi_view_encoder.py#L107-L131
[loss]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/src/losses/tactile_loss.py#L63-L91
[train]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/core/train.py
[converter]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/core/convert_to_hdf5.py
[positions]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/configs/hand_joint_positions.json
[metrics]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/src/utils/metrics.py
[inference]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/core/inference_tactile_parallel.py
[split]: https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/core/create_dataset_split.py#L221-L296
[egocfg]: https://github.com/Mr-Zwkid/EgoTac/blob/87ba7059304bd581050315f3a9af3029710ed690/configs/train_h4.yaml
[egodata]: https://github.com/Mr-Zwkid/EgoTac/blob/87ba7059304bd581050315f3a9af3029710ed690/tactile_prediction/dataset/vt_dataset.py
[egopolicy]: https://github.com/Mr-Zwkid/EgoTac/blob/87ba7059304bd581050315f3a9af3029710ed690/tactile_prediction/policy/direct_adaln_mano_policy.py
[egoguide]: https://github.com/Mr-Zwkid/EgoTac/blob/87ba7059304bd581050315f3a9af3029710ed690/docs/process_data_aether.md
