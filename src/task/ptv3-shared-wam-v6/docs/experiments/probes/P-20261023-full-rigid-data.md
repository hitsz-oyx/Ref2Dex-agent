---
schema: ref2dex.probe.v2
probe_id: P-20261023-full-rigid-data
experiment_id: P-20261023-full-rigid-data
date: 2026-10-23
task: ptv3-shared-wam-v6
branch: main
git_commit: recorded-in-run-manifest
classification: Blocker
status: HOLD
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
始终预留30GiB，写前按当前批次峰值检查，STOP文件在批次边界停止。CPU最多12worker；
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
 --mode execute --workers 12 --device cuda:7 \
 --output outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1
```

数据准备产物和完整全量报告在上述run目录；版本与实际命令以run-manifest.json为准。
此工程数据审计不随机抽样；manifest排序seed170、已有手表面采样seed20261019明确记录，
不额外领取训练/评估seed，不将来源认证失败升级成数据非法或模型效果结论。

## 2026-10-23 12:0x 全量转换/独立census终态

转换39,527/39,527母序列、TRAIN9,341,050/DEV1,856,282窗口；独立clock全量PASS。
固定转换字节182,331,193,978，receipt/实际文件一致；剩余0、partial0。九源TRAIN齐，
source8 DEV=0；TRAIN零格61→8，旧6k全部存在，新增9,335,050行，旧身份不变。
来源认证仍0/11,197,332；第一版保持A，goal独立消融后置。完整数表与预注册见
[准入报告](../../audit/nine-source-training-admission-20261023.md)及其机器收据。

本卡全量内容去重阶段未完成：full-manifest-r1在legacy官方TEST MANO view无
view_manifest时失败，原输出/日志保留；路径兼容已修并独立重放1,335母序列/633view，
未自动重启。独立readiness-manifest-r1覆盖全部窗口，不能把它称去重后manifest。
全量reader、内容/split、保守磁盘、GPU4/5/6门未全PASS，大训练HOLD，本轮0updates。

## 2026-10-23 13:33 brief47b owner HOLD

机主空间保护令已生效。转换39,527/39,527和所有receipt保留，无新转换批次；中断的是summary r4/r6，r6已退出，r4未退出父进程/资源tracker已SIGSTOP。run级STOP禁止后续自动批次。转换额外需求0 B；summary未完成，不冒称幂等续写或完整发布。未删除历史outputs，未迁移；详细停止点、精确实际占用、预算差额和4090传输代价见[空间HOLD报告](../../audit/brief47b-space-hold-20261023.md)。后续继续等待机主决定，原PREPARED_WITH_GAPS是此前数据状态，当前执行状态HOLD。

## 2026-10-23 13:32 brief47b机主STOP终止派生审计

39,527条九源母序列全部准入，TRAIN9,341,050/DEV1,856,282窗；新转换38,350、复用1,177，
固定182,331,193,978B、剩余0。完整前去重tuple manifest与108格已发布：TRAIN零格8、DEV20，
DexYCB DEV=0。内容去重仅完成20,672条/9,418,140窗，删除253,211（仅前缀），
仍有1,779,192窗未去重；最终去重后索引、旧DEV逐项差异、逐窗认证账本及统一catalog未完成。
13:32:58共享工作区新增owner_brief47b_space_protection_HOLD STOP，automatic_restart=false，
当时60.130GiB，未触30GiB；本执行器遵守HOLD，未移除STOP/自动重启。历史认证0/11,197,332，
新96条26,304个goal证书独立复验PASS、未并入总catalog。0训练/0checkpoint，未删历史outputs。
全量准备目标尚未验收，前文“未自动重启”等为各时刻快照；此终态优先。
[最终暂停报告](../../audit/full-rigid-data-preparation-20261023-brief47-HOLD.md)包含全量前去重九源、
几何和phase分布，两split各108格与全部零格、实际字节、manifest SHA和完整缺失边界。
全量前去重manifest SHA256：`4e40b4ec0bd58d390f8d5cee6047530d1832c3d5acb6080232251236d934c4c3`。
实际派生执行r4代码0d649194；r6代码c86fc6c1（1GiB缓存/内存journal/完成时提交），
14项等价回归PASS；两者都被STOP中止。部分预分配NPY不能当完整manifest使用。
转换代码/环境/命令由run-manifest*.json及逐receipt绑定，原转换未重跑。
