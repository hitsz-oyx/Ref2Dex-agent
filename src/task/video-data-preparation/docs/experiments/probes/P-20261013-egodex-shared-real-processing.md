---
schema: ref2dex.probe.v2
probe_id: P-20261013-egodex-shared-real-processing
experiment_id: P-20261013-egodex-shared-real-processing
date: 2026-10-13
task: video-data-preparation
branch: main
git_commit: recorded in run manifest before execution
claim_id: C1
hypothesis_family: HF-egodex-shared-track-admission
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: expand a bounded cohort of sparse goal-object windows using the same recipe
decision_changed_if_negative: fix the specific mask, coordinate, scale or tracking blocker before expanding preprocessing
status: PLANNED
run_id: egodex-shared-real-processing-20261013-r1
---

# 已下载 EgoDex 的 Shared-WAM 真实处理小探针

Result: PLANNED. 先核对完整原始配对、当前目标与后端，再运行最多三个窗口。
Decision: 沿用 geometry-only、原生 30Hz/H4/K24、当前固定 32 个 query；不启动训练。

Class: Blocker/Decision. 服务于 Shared-WAM 人类手物轨迹预训练子目标。问题是：已有
真实 EgoDex 上，当前观察 depth 与未来 label depth 能否落在相同米制坐标中，同时提供
正确目标表面的 32 点 correspondence？合格支持扩一个小 cohort；失败则定位实际阻塞，
不以合成 smoke 通过或单个失败关闭整个方法。

最便宜的判别：按下载 READY 中的原始 recording 顺序选择最多三个 basic_pick_place
episode，按固定时钟选 anchor，先查看当前 RGB 决定目标可见表面，再标记未来 mask。
完整原始 MP4/HDF5 小样本只读复制到本地，不重编码时间、不下载新数据或模型。
只用已观测 H4 构造 observation；25 个原生未来帧用于 tracker/label depth，point ID
在当前阶段固定。原生手和物体伪标签分别报告。

## 固定门与记录

- current query 至少 16 个，固定模型槽 32 个；候选至多 64 个，不按未来幸存重排。
- endpoint 有效/current query ≥80%，报告 32 槽绝对覆盖率及所有 horizon 的有效数。
- 沿用首版 observation/label current depth agreement：median≤30mm、p95≤60mm。
- 刚体诊断 median≤5mm、p95≤15mm，保留未拟合的原始 point labels。
- mask 是本轮 assistant 可视化审阅的保守表面标注，不是官方分割真值；有歧义或
  严重遮挡时拒绝，不扩大到背景/手上凑点。
- 准入只证明该配方可产出样本，不证明真实几何精度、抓取或 Cm 收益。

## Decision Note / 资源

远端已有 producer 与四个 reconstruction GPU 进程保持原状；本轮改用本机一张空闲
GPU，复用本地 Omega/CoTracker。远端只读核对和下载少量原始配对，本地模型推理
最多累计 30 GPU 分钟，单次受控进程最长 20 分钟；最多三个初始窗口。完整小样本
传输≤100MiB，本地新增产物与 scratch 合计≤2GiB，遵守共享 400GB 及本机 GPU
上限。没有新增长期身份或外部服务，也不删除/改写旧数据、checkpoint 或进程。

current target 错、标定不可辨识、OOM、非有限值、陌生 GPU 进程或预算触发时停止
该次执行，保留原因。工程接口错误在原卡记录修复，不把错误实现当作方法负证据。
若一个合理的修复仍在预算内，只补跑受影响窗口；改变处理假设先写短 Decision Note。

## 产物与身份

`outputs/video-data-preparation/egodex-shared-real-processing-20261013-r1/` 保存输入
清单、mask、处理 manifest、每窗状态和可视化；scratch 在项目 `tmp/`。实际固定代码
commit、vendor commit、资产路径与运行设备在执行前写入 manifest。无随机模型初始化
或训练，当前 sampling 为确定性 FPS/固定时钟，因此无需占用训练 seed。

## Limitations / future evidence

少量手工表面 mask 不代表自动 target grounding；局部准入率不代表全量下载集。
depth agreement/刚体残差是自洽检查，不是独立物体 metric ground truth。future mask
审阅不反向改变当前 query，端点 goal 仍属于 hindsight supervised task condition。
自动分割、多个 session 的覆盖、独立几何质量证据与训练收益留待有正向信号后验证。
