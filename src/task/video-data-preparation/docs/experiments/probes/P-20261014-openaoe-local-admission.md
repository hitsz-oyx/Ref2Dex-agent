---
schema: ref2dex.probe.v2
probe_id: P-20261014-openaoe-local-admission
experiment_id: P-20261014-openaoe-local-admission
date: 2026-10-14
task: video-data-preparation
branch: main
git_commit: 8ef5e766a25f34b50027d759af768b937b5d6b43
claim_id: C1
hypothesis_family: HF-openaoe-shared-track-admission
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: implement a bounded Open-AoE provider using published hand and camera estimates
decision_changed_if_negative: resolve the specific input coordinate or timing blocker before expansion
status: RUNNING
run_id: openaoe-local-subset-20261014-r1
---

# Open-AoE 本地子集接入探针

Result: RUNNING. ModelScope轻量目录和标注可读，完整小样本下载及几何审查待完成。
Decision: 按用户授权先下载小份完整样本，在真实输入上核验Shared-WAM手/相机接入难度；不训练、不扩全量。

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

## Limitations / future evidence

数据是视觉重建伪标签而非原生设备GT。没有发布目标物体轨迹/密集depth；本探针先检查手/相机，
不将通过此检查的窗口称为已具备完整U/V/G的可训练样本。完整物体teacher接入、
观察因果边界、独立尺度精度、大样本覆盖率与训练收益均待后续明确设计。
