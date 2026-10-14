---
schema: ref2dex.probe.v2
probe_id: P-20261014-egodex-shared-batch-ingestion
experiment_id: P-20261014-egodex-shared-batch-ingestion
date: 2026-10-14
task: video-data-preparation
branch: main
git_commit: d61cac75ba7e8ec79981242d03725032c32de013
claim_id: C1
hypothesis_family: HF-egodex-shared-track-admission
probe_index_in_family: 2
seed_pool: probe
seeds: []
decision_changed_if_positive: keep a bounded remote consumer producing auditable Shared-WAM windows
decision_changed_if_negative: fix the identified selector or processing blocker before expanding
status: PLANNED
run_id: egodex-shared-batch-20261014-r1
---

# 远端 Shared-WAM 批量接入

Class: Blocker/Decision. 用户明确授权停止一个远端 GPU 的既有任务并推进新线路批量接入。
保留 H4/K24、原生 2×11 手点、2048 scene、32 固定 object queries、观察/标签独立推理及
scale-only relative IQR≤0.25 等现有数值门。目标是检验自动目标 mask 能否支持可审查的真实
窗口，且队列消费、恢复、资源监督能实际运行；不训练、不形成方法有效性结论。

## Decision Note

- 关键证据：既有三窗 Probe 1 个数值合格、2 个标定拒绝；远端约18万完整配对已 READY，
  GPU0/1/2 为本任务辅助重建，GPU3 为主重建。新线路当前缺自动 mask 和外部监督。
- 行动：核验 GPU0 辅助进程671017 的 UID、命令与 GPU 对应，发送其支持的 SIGUSR1，
  让它完成当前 episode 后退出；保存停止和恢复信息。不动下载、主重建或 GPU1/2。
- 先复用已有模型资产和环境，在独立远端 scope 开发部署自动 mask backend 与受控 consumer。
  当前选择只读当前/H4；未来身份 mask 单独生成；自动候选的训练准入需可视化审查。
- 初次处理最多1 GPU小时、1000尝试窗口、新产物≤2GiB、scratch≤5GiB；新增模型资产≤5GiB。
  先查看20–50个简单刚体 episode 的合格率/拒绝归因，再决定同资源内下一批；外部 owned
  watchdog 限时，模型失败、陌生GPU进程、空间不足或预算触发即停止。
- 用户已授权这次 GPU 切换及批量接入；无新分支、无未知进程操作、无覆盖旧产物。
  未解决的选择歧义进入拒绝/待审队列，不以放宽几何质量门提高通过率。

## Identity / outputs

实现起点为上列提交；实际执行提交将在启动前固定并写入 manifest。
远端独立目录：`/mnt/ugreen_nas/storage/oyx_storage/egodex-shared-batch-20261014-r1/`。
本地证据：`outputs/video-data-preparation/egodex-shared-batch-20261014-r1/`。
原始下载与旧重建只读，aux 停止仅发生在明确授权且核验过的 owned process。

## Limitations / future evidence

自动语言定位和表面分割是伪标签；数值门不能证明目标语义正确或米制真值。
先生成审查产物与明确准入记录，再冻结可训练 train/dev；不承诺全量覆盖或全量完成时间。
独立几何精度、多 session 泛化与训练收益延后。
