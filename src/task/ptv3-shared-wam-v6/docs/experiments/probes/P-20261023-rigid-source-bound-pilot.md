---
schema: ref2dex.probe.v2
probe_id: P-20261023-rigid-source-bound-pilot
experiment_id: P-20261023-rigid-source-bound-pilot
date: 2026-10-23
task: ptv3-shared-wam-v6
branch: main
git_commit: 480ae4e4
classification: Decision
status: RUNNING
run_id: rigid-source-bound-pilot-20261023-r2
claim_id: rigid-data-deployable-coverage
hypothesis_family: source-bound-rigid-collection
probe_index_in_family: 1
seed_pool: probe
seeds: [198, 199, 200, 201, 202, 203]
decision_changed_if_positive: scale collection only when all96 rollouts admit and their reset-declared goals certify; schedule fixed-panel matched training separately
decision_changed_if_negative: repair producer binding or semantics before any TRAIN integration; retain rejected data separately
---

# P-20261023-rigid-source-bound-pilot

Result: collection and independent saved-artifact audit in progress; 36/96 raw rollouts complete at GPU boundary wait.
Decision: owner authorized96 fresh simulation rollouts; retain completed batches while the other executor uses GPU4; no training/model/6k recipe changes.

## 结果解读表

| 观测 | 含义 | 决定 |
|---|---|---|
| 新采96条准入100%，goal可认证 | producer正确记录来源、reset、语义 | 给下一批N与实测成本；扩大采集仍须明确预算 |
| 准入不足100%或goal不可认证 | 绑定/语义仍缺 | 修采集脚本，不混入TRAIN |
| 原108格中的61个零格显著下降 | 机制缺口获得实际覆盖 | 排全量数据重训；本轮不执行 |
| 覆盖补上但固定面板预测不动 | 数据量不是当前瓶颈 | 停止买数据，条件/标签审计 |

区分“新producer可得到认证数据”与“缺口来自无法获得监督/来源”的假设。
最便宜的决定性方案是既有source4/Shadow仿真、既有正式语义adapter和本地已有资产包；
不追补人类旧reset证据、不转换旧Inspire标签、不换模型。初始source4九个零格，
按各格旧窗口数加已配rollout数水位补齐到96条，配额只使用phase与PCA几何类。
每类从官方TRAIN且不在TEST/DEV母资产集合中按固定archive顺序取前两项；
无身份阈值/结果筛选。采集阶段的neutral/open时间表按机制作用于所有同phase样本。
目标phase不等于实测标签；统计仅按原proximity/event合同，未出现的phase如实为零。

每条10s@30Hz=300ticks，另保存初态边界共301状态；维持原60Hz物理/actor，
每两步记录一次，共57,600原生controls+96reset warmup controls，28,800输出ticks。
显式任务指令在reset前落盘并冻结；reset_target_pose每次恢复同一个goal state与root，
保存同clock时间戳、来源digest、execution/reset/slot绑定。Goal复用现有ResetWaypointChannel
与来源分类器，禁止用自身t+24未来作为部署条件；sample_window默认未来goal仍禁止接TRAIN。
实时named-body21与现有ShadowSurface FK逐帧交叉核对，保持778/16part、2048surface/32query。
接触为v5 nearest-vertex proximity，不宣称force或接触意图。

预算：GPU4仅空闲时使用，留GPU5/6空闲；采集2h，含registry/日志/资产≤2GiB；
16–32KiB/tick的规划估算0.44–0.88GiB。写前测量并预留≥30GiB，采集中重复guard，
并发转换进程不触碰。无训练、无checkpoint、无视频、无删除、无生产索引覆盖。
环境/命令/seed逐batch保存；结果以新run的audit-summary、逐条verdict与coverage CSV为准。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/run/collect_rigid_pilot.py prepare \
 --output outputs/ptv3-shared-wam-v6/rigid-source-bound-pilot-20261023-r1
# r1在初始化双reset守卫处失败，0物理采集；资产保留。
# prepare --prepared-from ...r1 --output ...r2复用只读资产；
# collect --gpu 4，批次边界可--resume；audit --wait-for-collection并行CPU验收。
# 两阶段均使用项目内TMPDIR。
```

运行身份：r2首两批采集代码480ae4e4；恢复入口8d97deae只补批次边界恢复，
不改变采集动作/标签/seed/配额。每批simulation-config与executions/resume receipts记录实际代码。
原始header301状态/600controls，前36条所有21点FK与live测量最大差2.552e-6m，
这只是工程preflight，通过正式准入/认证前不计为TRAIN覆盖。首批实际20条、第二批16条；
GPU4被别的执行器占用后等待，不改派GPU5/6以维持两张空闲卡。
