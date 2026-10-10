# ObjectForesight 原始 SpaTracker 轨迹合同审计

日期：2026-10-10。Task：`cm-pointflow-effect-pretrain`。研究时本地代码 `d7f4fff`。
范围：只读小源码、已有 NPZ 和 CPU 统计；未下载模型/视频、运行转换器或训练。
研究问题：官方 `spatracker.npz` 的 100 条轨迹是否值得替换独立 LK＋逐帧深度回投弱标签？

## 当前决策

**可以做低成本覆盖与几何一致性审计，暂不能直接替换训练标签。**
官方联合跟踪器提供持续列身份和几何联合优化，可能减少独立回投的不一致；这是待验证推论。
它仍是单目模型生成的 pseudo trajectory，不是实测 3D GT。
关键限制是稀疏、重复、物体覆盖未知，以及全视频 teacher 的未来信息进入历史几何。
近期先检查实际投影点的物体覆盖；不足现有点数 gate 就停止，不补点、不扩大训练。

原 h8 结果已有横向投影错误，不能把失败全归为深度误差；换标签也不能证明原方法有效。
本次只形成数据合同判断，视频学习整体路线仍 `UNCLEAR`。

## 固定主来源与复现边界

- ObjectForesight-Data：`d1dc6b04daf6429e9c81f7238d689abac8d1f2a9`。
  首要证据为这个仓库实际导入的 bundled `SpaTrackerV2/`，不是任意上游版本。
- SpaTrackerV2 上游：`7e12274c52077860cebfe007a6290777db43b63c`。
  本次逐字节比对 `predictor.py`、`SpaTrack.py`、`TrackRefiner.py`，三个文件与 bundled 相同。
  比对不等于确认整个仓库、checkpoint 或历史发布运行相同。
- [SpatialTrackerV2 论文 v2](https://arxiv.org/html/2507.12462v2)：联合估计视频几何、camera motion 和点轨迹。
  论文性能不能替代当前 EPIC 原始片段的质量审计。
- [官方数据 README][readme] 说明 bundled 副本可能修改；生成程序从 HF 按名称加载前端和 Offline 模型，
  **没有固定 HF revision**。默认 CLI 参数也没有被保存在本地 NPZ 中。[生成程序][step9]

本地三个 NPZ 只有 `coords/extrinsics/intrinsics/depths/visibs`，没有当前生成程序另存的 `unc_metric`。
因此能确认公开源码合同与本地数值相容，不能确认这些文件由该 SHA 和默认参数生成。
缺失的逐点 confidence、depth confidence、query lineage、checkpoint revision 不能从列数反推。

## 坐标、单位和投影

**源码事实**：[step9][step9] 把 tracker 返回的 camera-space 3D 点变换后存盘：

```text
coords[t,n] = c2w[t,:3,:3] @ track3d_camera[t,n,:3] + c2w[t,:3,3]
extrinsics[t] = inverse(c2w[t])
```

所以 `coords` 已经是该片段重建的 world 坐标；`extrinsics` 是 world-to-camera。
对它再次按 camera 坐标套 c2w，会重复变换。正确投影为：

```text
X_camera = E[t,:3,:3] @ coords[t,n] + E[t,:3,3]
uv = (K[t] @ X_camera)[:2] / (K[t] @ X_camera)[2]
```

`depths` 是 refined point-map 的 camera z 分量；保存前把 `conf_depth<0.5` 置零。
它和 coords 的 z 不是同一个参考系中的量。[生成程序][step9]、[tracker camera-space 返回][refiner]

本地样本 `extrinsics[0]` 严格为 I；首帧 world 与首帧 camera 重合。
公开 stream 从 I 初始化、累积窗口 camera poses；该 world 是片段局部重建坐标，不能跨 clip 当共享世界。
[窗口拼接 L210、L347–348][stream]

**单位边界**：代码没有把 coords 换成像素、mm 或单位球；点和 camera translation 继承预测深度的尺度。
官方 README 把输出称为 metric depth；但公开生成流程使用 RGB 前端，没有当前片段的实测深度标定输入。
因此可以记录为“teacher depth-scale 3D coordinates”；将其称为经验证的 meter GT、与原生数据精确同尺度，仍 UNKNOWN。
数量级约 1 不构成标定证据。[README][readme]、[VGGT 前端输出][vggt]

## `visibs` 数值合同

**源码事实**：[TrackRefiner L1167–1168][refiner] 最终保存前使用
`vis_pred_out = sigmoid(vis_logits) + 0.2`，注释称 invisible points 存在 bias、尚待修复。
该值范围是约 `[0.2,1.2]`，不是 `[0,1]` 校准可见概率，更不是人工遮挡标签。

| 消费规则 | 等价规则 | 可支持的表述 |
|---|---|---|
| `stored_vis>0.5` | 原 sigmoid `>0.3` | 带 +0.2 偏置的 teacher 可见 gate |
| `stored_vis>0.7` | 原 sigmoid `>0.5` | 未偏移 sigmoid 中点 gate |
| `clip(stored_vis,0,1)` | 饱和截断 | 丢失原始偏移信息，不建议 |

阈值是工程选择；更高阈值不能自动保证物理身份、遮挡、深度和物体归属正确。
生成脚本没有保存 tracker 的 `conf_pred`。不可将 visibs 同时当三维置信度或 LK forward-backward check。
[stream 返回字段][stream]、[生成保存字段][step9]

## 100 条列身份、抽样和 lost recovery

默认查询是首帧 `t=0` 的 10×10 全图网格，不是 100 个物体关键点，也不是 mask 内采样。
518×294 输出时 margin=`518//64=8`；x/y 范围分别 8→510、8→286。
`grid-size=10` 决定外部查询数；`vo-points=100` 设置内部 support track 预算，二者不同。
[step9 query 创建][step9]、[网格函数 L1090–1115][grid]

**容易遗漏的源码事实**：[SpaTrack L616–625][stream] 在 `full_point=False` 且首窗口无 overlap depth 时，
过滤 query 的 `unc_metric<=0.5` 部分，并随机复制保留 query 补足原长度。
输出没有保存被过滤/复制之前的网格 index。因此原始第 n 列不能无条件解读为原始第 n 个格点；
100 列也不等于 100 个独立采样位置。tracker 初始位置还可能被 refinement 移动。

stream 把 queries 按时间排序，最后按 inverse permutation 恢复列顺序；后续窗口携带所有此前 queries。
窗口 overlap 中按 `vis*conf` 选 anchor 接续，并不因为某帧 invisible 永久删除该列。
不存在当前 LK 方案的永久 lost-ID 状态；因此低 vis 后高 vis 可再次出现。
这能提供“同一模型列恢复”，不能仅凭可见分数确认恢复到同一真实表面点。
[stream L221–257、L350–354][stream]

## 历史输入中的未来信息

**不是严格在线/causal 轨迹。** 以下三条源码证据足以改变使用方式：

1. step9 先把整个 `action.mp4` 读成 tensor，一次调用 VGGT 前端。
   前端 aggregator 把所有帧的 patch 合并成 `S*P` tokens 做 global attention，attention 没有 causal mask。
   因此 depth、K、poses 的历史帧本身可利用 clip 未来 RGB。[step9][step9]、[global attention][aggregate]、[attention][attention]
2. tracker 每窗口的 2D/3D temporal attention 对完整窗口 T 做双向信息混合；没有传 temporal causal mask。
   还显式包含 `coords[t]-coords[t+1]` 的未来差分，以及 dynamic probability 的时间均值。
   这不等于显式“反向再跑一遍视频”，但确实允许窗口未来影响早期估计。
   [refiner L818–822、L876][refiner]、[temporal blocks L449–455][temporal]
3. support queries 的 mixed mode 包括窗口最后一帧和随机时刻，
   `forward_stream` 重叠写回此前时刻的轨迹、depth、intrinsics 和 poses。
   名称 `forward_stream` 不能据此推断 online contract。[网格 support 生成][grid]、[stream L297、L306–348][stream]

**使用推论**：未来目标可以来自 offline teacher，但 `coords[:history_end]` 不能宣称只看历史观测。
未来 vis 可以作 target-side label mask；不能用于选择当前输入点、决定当前点 birth 或历史有效性。
以首帧 coords 去重仅是“只读取 t0 数值”的去重，仍承受 t0 teacher 的全 clip 依赖。

已有 LK 的 2D tracking 即使只用已观测 RGB，搭配公开 SpaTracker depth/K/E 时也共享前端未来依赖。
换用原始 coords 不是首次引入这一条件；两种方案都应明确标为 offline pseudo-geometry。
具体连结是 step9 L72 前端读取整 clip；L74–76 提取前端 depth/poses/intrinsics，L84–88 整段交给后端；
L104–114 又把后端 world coords、inverse camera poses、intrinsics 和 refined depths 存到同一 NPZ。
所以用这些字段回投出的 LK xyz、history displacement/velocity，不能称为严格 causal perception。
本地 future-label-mutation 测试只能检查 pack 的显式 learner 输入接口隔离；
它没有重新生成上游 NPZ，不能证明历史几何在未来 RGB 改变时保持不变。
旧运行应保留为“full-clip offline teacher 弱标签可学习性 Probe”；
不能用来形成部署 history-only forecasting 或 future-free 科学结论。
若要形成可部署 history-only 结论，必须独立历史几何，或只在截断前缀重新提取 teacher 再比较；
本次不下载/运行模型。上游 pinned README 的 online release 仍在 TODO。[上游 README][upstream]

root 请求后的四份小源码缓存保存在
`tmp/video-tactile-primary-sources/objectforesight-d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/`：
`step9_spatracker.py` L61–72/L74–88/L104–114；
`SpaTrackerV2/models/SpaTrackV2/models/vggt4track/models/vggt_moe.py` L64–87/L92–115；
同目录 `aggregator.py` L291–308，以及 `../layers/attention.py` L50–72。
缓存来自上述 immutable raw URLs；未修改缓存内容或现有源码。

## 已有文件的最小 CPU 复算

根目录：`outputs/cm-pointflow-effect-pretrain/video-point-dynamics-20261010-r1/data-r2/raw/`。
三样本均首帧 E=I，图像几何 294×518，只有五个上述 keys。

| clip | T | 首帧 exact unique/100 | stored>0.5 有效比例 | 0→1 次数 | stored>0.7 有效比例 | 0→1 次数 |
|---|---:|---:|---:|---:|---:|---:|
| P03_13_12 | 158 | 81 | 79.9241% | 18 | 78.2468% | 31 |
| P03_13_19 | 70 | 93 | 48.1571% | 10 | 47.4571% | 12 |
| P03_03_12 | 159 | 88 | 84.9560% | 82 | 81.1069% | 101 |

P03_13_12 vis 最小/最大 `0.20000021/1.1999979`，与偏移源码一致。
首帧投影到最近初始网格的距离 median/max：1.237/5.090、2.069/5.334、1.645/4.684 px。
这是 query 初始化/refinement 相容性检查，不是 RGB 真值误差或物体覆盖验证。
0→1 次数按逐帧阈值计数，含同一列多次恢复，不能称为可靠重识别次数。

文件 SHA-256（完整原始 NPZ）：

```text
P03_13_12 e0566fdbce69746990e891321a49a36d9636fa5c3130ec052ccc1ab9b8714168
P03_13_19 54c8b3329d926c47cf23127f58a09f8be2f71a4137b69dce2297f98f5bcbf3da
P03_03_12 2cf065d3ef4f510e0da64c7f6ca7af581473e72329027ce0ec6850ba3d0ac49f
```

时间合同：step9 只在 `stride>0` 时显式抽帧；传入 tensor 时 `fps=1` 不执行 predictor 字符串路径的 subsampling，
更不代表物理 1 Hz。[predictor L48–54][predictor]
NPZ 无 timestamps/source_frame_id；本地 meta 三样本 FPS 均 `59.94005994005994`，
start/stop 为 1929/2087、3018/3088、1202/1361，T 与 stop-start 相符。
这支持“当前样本未额外 stride”的相容性推断，不证明全部发布文件、decode failure 或其他生成命令的时钟。
官方切片遇解码失败会复制上一帧，故 row identity 与真实采集帧仍须核对。[切片实现 L176–187][split]

## 最便宜的下一步与停止条件

**Decision probe，不训练**：在当前已有 clip 做实际 coords 的首帧投影→物体 mask 覆盖 census，
同时报告原始列数、t0 exact unique 数、正深度/图内数和每个物体 unique 点数。
只使用当前帧 mask/数据做输入资格，不用整条未来轨迹重复判定过滤输入。
保持现有至少 16 个物体点等资格 gate；不满足就停止原始100点替换，不为达到 gate 填充重复点。

如果覆盖通过，再做少量 RGB overlay 和 overlap-boundary/静止背景漂移审计，
比较 LK 与原始 tracker 的像素 identity、重复与恢复错误，并分开 teacher label quality 和预测模型错误。
这仍只能决定小 Probe 是否值得，不足以确认 physical GT、原生监督迁移或 causal action effect。

延后证据：真实 checkpoint/生成 revision、window_len、metric尺度标定、recovery物理身份、prefix-vs-full误差、
跨 clip shared world、每轨迹物体身份均 UNKNOWN。当前覆盖若不通过，不立即为这些问题占用模型预算。

[readme]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/README.md
[step9]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/step9_spatracker.py#L58-L115
[predictor]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/SpaTrackerV2/models/SpaTrackV2/models/predictor.py#L34-L86
[stream]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/SpaTrackerV2/models/SpaTrackV2/models/SpaTrack.py
[refiner]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/SpaTrackerV2/models/SpaTrackV2/models/tracker3D/TrackRefiner.py
[grid]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/SpaTrackerV2/models/SpaTrackV2/models/utils.py#L1090-L1221
[vggt]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/SpaTrackerV2/models/SpaTrackV2/models/vggt4track/models/vggt_moe.py#L64-L116
[aggregate]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/SpaTrackerV2/models/SpaTrackV2/models/vggt4track/models/aggregator.py#L291-L312
[attention]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/SpaTrackerV2/models/SpaTrackV2/models/vggt4track/layers/attention.py#L50-L77
[temporal]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/SpaTrackerV2/models/SpaTrackV2/models/tracker3D/spatrack_modules/utils.py#L414-L458
[upstream]: https://github.com/henry123-boy/SpaTrackerV2/blob/7e12274c52077860cebfe007a6290777db43b63c/README.md#L39-L45
[split]: https://github.com/RustinS/ObjectForesight-Data/blob/d1dc6b04daf6429e9c81f7238d689abac8d1f2a9/step1_split.py#L176-L187


## Root 后续核验与覆盖结果

root 已阅读上述四份固定源码缓存，核对整 clip tensor→前端→后端→NPZ保存，
以及所有帧token的无causal mask global attention，确认这项公开流程的未来依赖；
本地历史生成版本仍UNKNOWN，不能反过来声称当前释放文件已证实因果输入。

原始mask/track SHA校验的初始查询审计已在53142b2执行，CPU0.37s，
产物 `outputs/cm-pointflow-effect-pretrain/video-point-dynamics-20261010-r1/original-track-initial-coverage.json`。
实际初始投影mask内独立查询：train_12/13/14/15=4/0/1/1，
dev_10/12/19/8=1/1/2/0。原始100列的t0 exact unique为72..95；整条轨迹
exact unique数逐片与t0一致，仅作复制诊断，不参与初始去重选择。
没有clip达到16个初始物体点gate，停止原始100点直接替换计划。
这不限制另做目标物体查询或其他label-only teacher，但不为这些新方法先扩训。
当前记录与下一步决策以[Task状态](../STATE.md)及原video Probe卡为准。
