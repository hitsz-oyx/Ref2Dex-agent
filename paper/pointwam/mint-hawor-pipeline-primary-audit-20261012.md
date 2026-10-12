# MINT / HaWoR 对 EgoDex 视频轨迹 pipeline 的借鉴

日期：2026-10-12。范围：官方代码和论文的只读审查，未运行模型、未下载权重/数据、未修改第三方源码。结论是工程候选与实现边界，不是精度或训练收益验证。

审查代码固定在：

| 项目 | 官方代码提交 | 本地只读目录 | 论文 |
| --- | --- | --- | --- |
| MINT | `1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b` | `third_party/mint-primary-audit-20261012` | [arXiv 2609.04958](https://arxiv.org/abs/2609.04958) |
| HaWoR | `66c7d4108d58a716deccd192cb7645170cdc7bd7` | `third_party/hawor-primary-audit-20261012` | [arXiv 2501.02973](https://arxiv.org/abs/2501.02973) |

本轮上下文由主会话提供：EgoDex 已有 native camera、metric hand 与 confidence；最长已获取录像 40.17 s，以 10 Hz 的 402 帧做整段单上下文 VGGT Omega 深度重建、统一尺度，工程运行成功。现存问题是少量初始 scene 点在几十秒内遭遇遮挡和覆盖下降。本文不独立复算该运行，具体数值以 [视频实验卡](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-video-points.md) 为准。

## 直接影响当前决策的结论

值得借鉴的是**有限窗口的模型计算、统一坐标下组织产物、明确缺失/来源 mask、分阶段缓存恢复和质量统计**。当前保留 EgoDex 原生相机/手和整段统一深度缓存，再按短窗口重新选 scene queries、跟踪、保留 visibility，是优先路线。MINT / HaWoR 的手重建与手 infilling 没有提供 scene 表面点跨遮挡恢复功能。

两套工程都不能直接按输出 `valid` 理解为观测可信度：HaWoR 学习式补全和 MINT 短 gap 插值会把缺失部分改为有效。若借鉴清理，必须另存 native/observed mask、imputed mask 和处理来源，不能覆盖原始监督。

## 1. MINT 的长视频组织：窗口接起来，并不等于全局几何优化

论文 IV-B 描述固定 32 帧的独立重叠窗口、SE(3) camera chaining、共享帧的 hand/presence 融合。[论文 IV-B](https://arxiv.org/html/2609.04958v1#S4.SS2)

源码需要区分两个层次：

- **EgoPipeline 数据生成分块**：超过 15 s 默认切 15 s clips，重叠 1 s；元数据显式保存原视频 `start_frame/end_frame/clip_idx`。这是上游 pipeline 分块，不是 MINT 网络的 32 帧窗口。[`ego_pipeline/backends/long_video.py:13–15,51–79`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/ego_pipeline/backends/long_video.py#L13-L79)
- **MINT 推理窗口**：blend/smooth 模式 overlap 为窗口的 25%，最多 8 帧，32 帧得到 8；tail 左移形成完整窗口；fade 权重由实际邻窗 overlap 计算，避免用固定 overlap 错算尾段。[`mint/inference/engine.py:115–168`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/mint/inference/engine.py#L115-L168)
- **camera stitch 的具体实现**：取本窗 w2c 矩阵 `E_l`，计算 `E_rel = E_l @ inv(E_l[0])`，用已有 `global_extr[s]` 为起点，`E_g = E_rel @ G`。已经写出的 overlap camera frames 保留，追加新帧。这里没有对多帧 overlap 做拟合，没有尺度修正，没有 loop closure。它通过首帧坐标接续消除 gauge reset；无法凭这一操作消除窗口内预测误差与长期漂移。[`engine.py:927–951`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/mint/inference/engine.py#L927-L951)
- hand 参数和 presence **logits** 在共享帧按 fade 权重相加并除以总权重，presence 再 sigmoid；这不是把未知表面轨迹拼成确定对应。[`engine.py:955–985,1005–1023`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/mint/inference/engine.py#L955-L1023)
- window batch OOM 会缩小 batch 重试，单窗口 OOM 则抛出；元数据记录实际 batch、窗口数、时间和 hand postprocess。[`engine.py:900–919,1025–1041`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/mint/inference/engine.py#L900-L1041)

**对我们的建议**：立即借鉴全局 frame ID、完整 tail 和资源降级记录。当前整段 depth 已能运行，且 native camera 已提供统一坐标，暂时没有理由改成各窗独立 depth 或 learned camera stitching。将来遇到超长录像 OOM 才比较 depth 分块；必须单独测 overlap depth/scale residual，不能宣称接好 SE(3) 就解决尺度一致性。不同窗口重新选的 scene queries 是不同身份，不应按同一 grid 下标跨窗合并或平均。

## 2. MINT 的监督 mask 与清理

独立 presence head 用 `hand_kept` 做 BCE；pose 的 mask 在此基础上再乘来源 validity，例如 `mano_gt_valid` / `kpt21_gt_valid`；velocity 的 pair mask 是相邻两帧 mask 的乘积。[`model_train/losses/hand_presence_loss.py:23–49`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/model_train/losses/hand_presence_loss.py#L23-L49)，[`mano_joint_loss.py:63–75,90–124`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/model_train/losses/mano_joint_loss.py#L63-L124)

这部分适合借鉴为数据合同，但 EgoDex 的低 confidence 是“标签不可靠/缺失”，不能不加检查就当作“物理上没有手”。当前不需要增加 learned presence head。scene target mask 应组合 tracker visibility、finite depth、坐标合法性和原始时间支持；速度/相邻运动必须两端有效，插值不应跨无效观测洞。

公开清理顺序是短 outlier block rejection → bounded gap interpolation → camera-aware UKF/RTS。[`ego_pipeline/data_cleaning/final_clean.py:31–78`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/ego_pipeline/data_cleaning/final_clean.py#L31-L78)

- 插值仅填两端有有效观测的内部 gap，默认 `max_gap=30` **帧**；平移、关节轴角参数和 shape 做线性插值，wrist rotation 用 SLERP；之后 `valid=True`。它不是所有旋转关节都用 SLERP，也不是缺失实测值被恢复。[`cleaning_modules/slerp_rot.py:6–51`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/ego_pipeline/data_cleaning/cleaning_modules/slerp_rot.py#L6-L51)
- UKF/RTS 对有效时间索引处理，先把 world wrist position/orientation 转到 camera frame，滤波后转回 world；保留传入的 validity。RTS backward 使用未来信息，限于离线处理。[`ukf_cam_smoothing.py:124–157,191–241`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/ego_pipeline/data_cleaning/cleaning_modules/ukf_cam_smoothing.py#L124-L241)

**边界**：降低 jitter 不等于改善真实动作精度；平滑可能消掉 contact、快速运动和 cloth 非刚体形变。在 10 Hz 数据沿用 30-frame gap 等于填 3 s，不能照搬 30 Hz 清理参数。原生 hand 标签保持原始 bytes；只允许另存候选 teacher/smoothed 产物并标注 `offline/noncausal/imputed`，需匹配比较才能准入训练。

## 3. HaWoR：camera 与 hand 解耦、动态掩码和尺度聚合

HaWoR demo 从检测到的手段估计 camera-frame MANO，使用 SLAM camera 转 world，再填缺手段。[`scripts/scripts_test_video/hawor_video.py:149–191,251–286`](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/scripts/scripts_test_video/hawor_video.py#L149-L286)

手模型推理每 16 帧一次，末尾重复最后帧补齐并去除 padding；这一路没有 overlap blend。[`lib/models/hawor.py:379–437`](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/lib/models/hawor.py#L379-L437)

camera 路线渲染手 mesh mask，SLAM 把 mask 区域从输入图像置零且作为 confidence mask 传入 DROID；Metric3D 在 SLAM keyframes 估深度，尺度拟合排除 masked hand 区域，最后对 keyframe scales 取一个 median 保存。[`hawor_video.py:194–219`](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/scripts/scripts_test_video/hawor_video.py#L194-L219)，[`lib/pipeline/masked_droid_slam.py:130–156`](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/lib/pipeline/masked_droid_slam.py#L130-L156)，[`hawor_slam.py:96–139`](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/scripts/scripts_test_video/hawor_slam.py#L96-L139)，[`lib/pipeline/est_scale.py:74–110`](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/lib/pipeline/est_scale.py#L74-L110)

**对我们的建议**：保留 camera/scale 与动态效果监督的分工。在静态几何审计时识别 hand/动态物体区域有价值，但 scene motion 的目标就是动态物体，不能把所有动态点删掉。我们已有 native camera 与 metric hand anchors，不必新装 SLAM、Metric3D 或 HaWoR 重算这些已知量。全轨迹只确定一个尺度的组织方式可以借鉴，median 本身不能证明所有帧深度准确；应同时报时段尺度分布与置信度。

源码尺度失败路径含 `while math.isnan(scale)`，逐步放宽近远阈值而无显式次数上限。[`hawor_slam.py:123–127`](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/scripts/scripts_test_video/hawor_slam.py#L123-L127) 不照搬这类无界 retry；我们记录失败并有停止条件。

## 4. HaWoR infiller：补手运动，不是补 scene correspondences

infiller 是 120 帧 horizon 的双手 Transformer，用平移、MANO shape、wrist/15 joints rot6d 表示；先在 world 坐标组织两只手，用两手都有效的时间构造 data/attention mask，寻找缺失段前可用双手上下文，模型补全后只替换原 missing 参数。[`hawor_video.py:222–234,289–356`](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/scripts/scripts_test_video/hawor_video.py#L222-L356)

随后整个处理区间 `pred_valid=1`，输出五元组只带完成后的 valid，未在此返回值保留 observed/imputed 来源。[`hawor_video.py:358–365`](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/scripts/scripts_test_video/hawor_video.py#L358-L365)

**适用性**：只有未来处理没有 native hand 的 RGB 数据时，才考虑 hand reconstruction/infilling 的独立实验。它使用手运动先验，不能推断被 cloth/hand 遮挡的任意表面点真实位置，更不能让原无 visibility 的 CoTracker label 变成观测真值。当前先不引入。

## 5. MINT 的 resumability 值得借鉴，但 stage DONE 仍要验产物

manifest 保存 video 状态、错误、attempts、长视频 clips 及 clip stage；重启把 RUNNING 回滚 PENDING；默认跳过 DONE，失败只有 retry_failed 才重试；写文件先同目录临时文件、flush/fsync 再 `os.replace`。[`manifest/state.py:31–46,110–137`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/ego_pipeline/manifest/state.py#L31-L137)，[`manifest/reconciler.py:42–75`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/ego_pipeline/manifest/reconciler.py#L42-L75)。clip stage 查询按状态字符串 done 判断。[`manifest/__init__.py:207–228`](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/ego_pipeline/manifest/__init__.py#L207-L228)

立即借鉴为我们 `whole-depth → per-window-tracks → window-extract → quality-admit` 的缓存层次。恢复时除了 DONE，还检查 raw/source/model/config/cache hashes 和 source frame IDs；工程完成与 training_allowed、native_GT / pseudo-label / imputed 的可信等级分开。该可信等级是我们的设计建议，不是声称上游已实现同样合同。

## 6. 最小后续实验与延后项

| 优先级 | 假设/决策 | 最低成本检查 | 结果如何改变行动 |
| --- | --- | --- | --- |
| 立即，工程 smoke | 分阶段缓存与来源 mask 能避免补全混入 native GT | CPU 查 loader、manifest、两端 mask，模拟某 tracking stage 中断后恢复；不重建深度 | 合同一致后继续按窗追踪；发现错误则修合同 |
| 当前 Decision Probe | scene 点覆盖下降主要来自 queries 存活时间太长 | 同一 40 s depth/cache，取原低覆盖时段与末尾，1 s 窗口首帧新 queries；比较当前有效点数、未来有效比例、几何残差和耗时 | 覆盖改善且质量过关则扩至该轨迹；仍差再定位遮挡/低纹理，不直接增训练 |
| 延后，独立 Decision Probe | smoothing 减少 teacher 抖动而不损害真实运动 | 只对另存 teacher 副本比较 held native anchors、快运动/contact 区间和静态残差；保 observed/imputed mask | 只在误差和运动保真均改善时准入；仅 jitter 降低不准入 |
| 延后，资源 blocker 才做 | 超长输入需分块 depth | 真实超过当前单 pass 限制的样本，比较 overlap depth/scale residual、bounded stitch drift 与资源 | 单 pass 不可行且误差满足原 gate 才采用分块 |
| 当前不做 | MINT / HaWoR 取代 native camera/hand 或填遮挡 scene 标签 | 已有 native 信息且无 scene completion 合同 | 不启动新权重下载或 GPU 路线 |

资源/语义边界：MINT 公共数据 camera trajectory 被作者明确标为 scale-enlarged，不能当 metric GT；数据生成公开代码也缺少部分生产集成和最终打包，不能当一键可复现 pipeline。[MINT 官方 README](https://github.com/wuji-technology/wuji-ego-mint/blob/1bd21e4edd6d5d2d59789ce64ee5450de24a7c2b/README.md#L378-L437) HaWoR / MANO 有独立许可；本轮仅阅读，未导入模型或重新分发修改。[HaWoR README](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/README.md#license)

未验证：两模型在当前 EgoDex subset 的精度、memory、吞吐；任一种清理对 WAM 训练/robot policy 的收益；超 40 s 实际录像的几何表现。论文与当前 README 的 MINT 手指标存在更新差异，本 note 不引用比较表数值，避免将不同 manifest/版本合并成同一结果。
