---
schema: ref2dex.probe.v2
probe_id: P-20261014-openaoe-local-admission
experiment_id: P-20261014-openaoe-local-admission
date: 2026-10-14
task: video-data-preparation
branch: main
git_commit: 84b441991ee255ec5961f3859df8d08f395aea8d
claim_id: C1
hypothesis_family: HF-openaoe-shared-track-admission
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: implement a bounded Open-AoE provider using published hand and camera estimates
decision_changed_if_negative: resolve the specific input coordinate or timing blocker before expansion
status: PROMISING
run_id: openaoe-local-subset-20261014-r1
---

# Open-AoE 本地子集接入探针

Result: PROMISING（仅手/相机接入）。两段真实30Hz视频完成MANO解码、H4/K24合同检查，用户确认抽样投影基本贴合；完整手物训练准入仍UNCLEAR。
Decision: 保留有界reader/audit，下一步用刚性容器/瓶子做少量完整手物窗口；先适配当前目标语义与坐标，暂不扩大下载量。

Class: Blocker/Decision. 服务于Shared-WAM的大规模人类演示预训练子目标。
区分“发布MANO/相机估计可通过薄reader进入现有手历史合同”与“时间、坐标或投影使其不可直接接入”。
最便宜的方法是2个90秒左右的不同recording，核验视频/参数/相机时钟与矩阵关系，解码手点并可视化。
数值自洽只能作为Probe证据，不能证明独立几何精度或机器人训练收益。

## Decision Note / budget

- 用户明确授权下载子集到本地并检查困难；沿用已有main，不建分支。
- ModelScope接口可用，直接下载明确sample路径；不装SDK或修改共享环境。
- 初始按目录顺序检查三段轻量标注，完整下载选择两个较短的不同录制。
  保留一个包括刚性容器/瓶子、也有走动与软包的混合片段；另一个软物体片段作为范围边界检查。
  选择是数据类型与文件大小，不根据未来物体位移挑目标。
- 原始输入总预算512MiB；审计输出≤64MiB；下载最长20分钟、并发2；GPU解码最多1张、累计60秒。
  大多数文件、帧时钟、统计和绘图使用CPU；批量MANO优先GPU，使用前重新检查资源。
- 只读已有ARCTIC MANO模型，不改外部项目，不动本地/远端其他进程、不下载teacher权重。
- 输入不齐、视频时钟不匹配、无法解释坐标/投影、预算超限即停止扩量并保留审计结果。

## Outputs / provenance

输出：`outputs/video-data-preparation/openaoe-local-subset-20261014-r1/`。
scratch：`tmp/openaoe-local-subset-20261014-r1/`。
具体文件路径与字节数写入download-manifest，实际工具提交在启动前记录。
来源：[ModelScope](https://www.modelscope.cn/datasets/inclusionAI/OpenAoE-2000h)、
[官方格式](https://github.com/ant-research/Open-AoE/tree/main/open-aoe-2000h)。

## 实际执行与结果

- 下载代码：`d5dc7a46c52294a9c478fabccc3765093a6eb51c`；审计代码：front matter中的`84b44199`。
  完整样本为`aoe_20260131_193655_p002`和`aoe_20260131_204056_p001`，共14个必要文件、
  461,948,113字节（440.55MiB），ModelScope下载36.93秒。另一个recording仅取轻量标注，不计完整样本。
- 两段去畸变视频均1920×1080、90秒、2700帧、30Hz；视频、手参数、相机帧数一致，K一致。
  第一段原始视频相邻PTS相对30Hz的最大偏差约10.02ms；去畸变视频误差小于1微秒。
  后续配套使用去畸变RGB、其K和帧索引，不把原始视频强当恒定30Hz。
- MANO参数为绝对axis-angle：`flat_hand_mean=True`，对现有受影响左手模型使用官方shapedirs修正；
  先世界系解码再将关节点刚性变换到相机系，不能直接以相机根参数重跑MANO代替。
  输出沿用当前provider的right/left、wrist+5个base+5个tip顺序。
  旧pickle/chumpy的NumPy兼容问题用进程内别名解决，没有安装包或修改共享环境。
- 手根世界→相机关系最大偏差低于0.5微米，只证明重复字段数值自洽。
  右/左手帧有效率分别91.19%/92.37%与100%/97.15%；每手有效性来自原始binary mask，
  没有伪造逐关節置信度或将遮挡判为可见。
- 以stride8扫描各335个anchor，H4+future24全28帧至少一手有效共659/670；
  这是手部候选数，不是完整训练窗口数。每段实际通过`NativeEpisode.window`生成一份手部fixture：
  history为`[2,11,9]`、future为`[24,2,11,3]`，scene/U/G尚未生成。
- 两段CPU MANO解码共0.89秒，整个审计7.42秒。设备重新评估后选择CPU：本次仅少量参数几何与
  文件审计，未运行神经网络teacher，GPU启动/占用不划算；GPU使用为0，未触及共享进程。
  原始输入与审计总计约445.49MiB，审计增量约4.94MiB，均在预算内。
- 现有EgoDex `target_phrase`对18条原始动作描述接受0条：材质拒绝5、解析格式拒绝3、
  名词不明确10。此结果说明现有保守parser不能原样复用，不证明Open-AoE目标不能提取。
  新adapter应利用atomic action的object/verb及当前RGB/H4确定操作对象，区分源物体与目的容器；
  不通过放宽质量门或未来运动挑选目标。标注bbox不是逐帧分割/轨迹真值，本轮未使用。
- 用户查看两段各四帧投影图并回答“基本贴合”。该确认仅覆盖抽样手部投影；没有批准物体标签
  或米制精度。软物体片段不进入当前刚体目标profile。

审计回溯：输出根目录下`download-manifest.json`、`audit-summary.json`、`target-parser-audit.json`、
`visual-review.json`，以及每段`audit/<sample>/audit.json`、`decoded-hands.npz`、`hand-only-window.npz`。
查看图：`review-container.jpg`和`review-soft-object.jpg`。所有产物均`training_allowed=false`。

## Limitations / future evidence

数据是视觉重建伪标签而非原生设备GT。没有发布目标物体轨迹/密集depth；本探针先检查手/相机，
不将通过此检查的窗口称为已具备完整U/V/G的可训练样本。完整物体teacher接入、
观察因果边界、独立尺度精度、大样本覆盖率与训练收益均待后续明确设计。
相机SLAM世界轴/重力没有独立验证，fixture暂用当前相机anchor；不能直接沿用EgoDex的ARKit→Z-up
常量。MANO thumb base与EgoDex ThumbKnuckle的解剖等价仍需核对；离线全序列优化的手/相机
估计也不能未经审计就称严格因果观察。容器/瓶子可能有透明或反光部分，下一步仍按现有材质门
筛选明确表面。当前没有非空完整训练/dev快照，没有训练结果，也没有证据证明整库筛选效率。
