---
schema: ref2dex.probe.v2
probe_id: P-20261023-full-rigid-data
experiment_id: P-20261023-full-rigid-data
date: 2026-10-23
task: ptv3-shared-wam-v6
branch: main
git_commit: recorded-in-run-manifest
classification: Blocker
status: PRE_REGISTERED
run_id: full-rigid-20261023-brief47-r1
claim_id: full-local-rigid-data-coverage
hypothesis_family: full-local-rigid-preparation
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: schedule full-data retraining after independent condition certification gates
decision_changed_if_negative: report structural zero cells and missing native inputs for owner collection or restoration decisions
---

# P-20261023-full-rigid-data

| 观测 | 含义 | 决定 |
|---|---|---|
| 全量准入后覆盖缺口显著缩小 | 数据面可为后续训练提供支撑 | 进入全量数据重训/续训排期；本轮不训练 |
| 准入后仍有大块为0的phase×几何×source组合 | 本地数据结构性缺失，或缺少可恢复的路径/语义证据；分别计数 | 列清缺什么，由机主决定新采集或恢复 |
| 转换后磁盘余量<30GiB | 空间不可持续 | 在写入前预测触线即停；保留产物，报告预算，不删历史outputs |

覆盖范围升级为全部1335条本地legacy母序列、现有1064条active TRAIN母序列所在的
1177条缓存，以及九source完整39526条冻结母序列计划。原pending6000序列批次包含在
全计划中，不能将它误称为6000个窗口。每条母序列都给出准入或拒绝理由，不能重复
相加同一母序列的cm_decoder/coupled/GRAB视图。

要区分的假设：全量现有数据能补阶段/几何/来源覆盖，还是缺口在输入、语义或事前
条件证据。结果改变后续重训排期与是否采集的决定。最便宜做法为已有缓存只读复用、
新序列只转换一次、全tick合同审计，不训练、不换模型、不新采集。

入口：`tools/audit/prepare_full_rigid_data.py`。先inventory，报告固定形状转换NPY与
固定长度JSON的精确字节数；manifest/receipt与异常partial另报上界，不能把上界说成
实际值。执行每64条写批次receipt，每条原子发布独立receipt，可重复execute续跑。
始终预留30GiB，写前按当前批次峰值检查，STOP文件在批次边界停止。CPU最多4worker；
如需MANO GPU重建仅可使用机主指定GPU4/5/6/7中即时可用的设备。禁止训练和删除历史outputs。

准入复用现有刚体合同，不以object名字/source身份放宽，H4/K24、joint21/validity、
surface778、surface2048/normals/query32、contact/part/valid与event及metadata绑定
都要检查。unknown显式保留mask；不生成关节/变形/关系型目标字段。

每个窗口记录source_role与available_before_reset；缺少reset前TaskSpec/角色/时间证明
时UNKNOWN/null，目标label哈希和mtime仅作回溯，不能替代条件可用性证明。近重复去重
另给可复现的几何/时序/接触内容谓词、删除代表与前后覆盖，TRAIN/DEV独立并审计跨集重复。
固定512锚点和旧6000窗口不重写。manifest SHA、环境、seed、命令、receipt路径落盘。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/prepare_full_rigid_data.py \
 --mode inventory --output outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1
# 核对打印预算后启动；再次执行同命令会跳过已有receipt。
/home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/prepare_full_rigid_data.py \
 --mode execute --workers 4 --device cuda:7 \
 --output outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1
```

数据准备产物和完整全量报告在上述run目录；版本与实际命令以run-manifest.json为准。
此工程数据审计不随机抽样；manifest排序seed170、已有手表面采样seed20261019明确记录，
不额外领取训练/评估seed，不将来源认证失败升级成数据非法或模型效果结论。
