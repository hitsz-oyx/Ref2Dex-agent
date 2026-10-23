---
schema: ref2dex.probe.v2
probe_id: P-20261023-rigid-dataset-admission
experiment_id: P-20261023-rigid-dataset-admission
date: 2026-10-23
task: ptv3-shared-wam-v6
branch: main
git_commit: pending-code-checkpoint
classification: Blocker
status: PRE_REGISTERED
run_id: rigid-dataset-20261023-r1
claim_id: rigid-data-deployable-coverage
hypothesis_family: rigid-dataset-admission
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: append only contract-admitted independently certified rigid coverage before a fixed-panel matched-budget comparison
decision_changed_if_negative: return new collection or explicit native semantic annotation as an owner decision; no source expansion or training
---

# P-20261023-rigid-dataset-admission

| 观测 | 含义 | 决定 |
|---|---|---|
| 新增合法覆盖，固定面板同预算预测误差下降（包括接触窗口/前8帧） | 数据是瓶颈的一部分 | 按覆盖表补数据 |
| 新增覆盖，预测不动 | 瓶颈在条件/优化/标签，不在数据量 | 停止买覆盖，转条件与标签审计 |
| 准入修复后合法增量约0 | 当前合同下本地直接可用增量近上限 | 交回是否需要新采集的机主决策并估算成本 |

本实验区分“旧工具/路径误拒绝”与“真实语义证据缺失”。最便宜的判别是只读实际
6个view/3母序列的1,564个窗口，不做96.5GiB扩转换、不训模型。刚体以固定点池的
SE3输运残差验证；不引入非刚体字段、模型或数据源。

准入：同一物体无关谓词检查21点语义、H4/K24边界与显式validity、手表面、contact/part/event
及registry绑定。已有显式unknown保留mask；缺失字段不能伪装成unknown标签。合法合成
合同必须进入真实sample_window，语义置换/代理替换/坏绑定等反例必须拒绝。

两根manifest原来的三条TRAIN记录不变。cm的失效row.path只从其已声明的绝对数组路径
恢复；不得以物体名称配别的轨迹。coupled同parent、帧长、30Hz时钟及cm wrist变换
共同校验绑定。4096池的固定偶数点变为2048 registry、其中固定32点为query；必须先过
全帧逆位姿刚体一致性和normal检查。数值SE3 mask单独保存，不冒充annotation validity。
若21点语义缺失，不凭新FK/插值定义替旧合同发证，也不把人类母序列标签贴给Inspire。

来源认证复用brief35实证并做冻结6,000行精确有序join与当前pose hash核验；每条写
source_role、available_before_reset、来源hash/文件时间及缺口。未知reset时间保持null，
mtime不证明reset前可得。当前未来标签与未知独立计划的角色分别记录。

去重：PCA几何extent5%、完整H4/K24锚定joint21/query32的每坐标2mm、物体旋转1度、
相同mask/phase、逐帧16part接触比例2%；跨序列也比，不用物体/source身份筛选。每个
删除项指向实际保留代表，不使用传递闭包。输出新索引，不覆盖旧6k或训练配方。
覆盖矩阵为4phase×3数值几何类别×9source，另列unknown，不把proxy当实测force。

运行预算20min CPU/IO、无神经推理/训练/仿真，输出硬上限64MiB、磁盘floor20GiB。
先--plan精确序列化写集，再--execute写新目录；不删历史outputs。任何后续训练仍须
来源/缓存合同及GPU{4,5,6} preflight，固定原512锚点、相同训练预算并报告接触/first8。
本次seed为空，因为没有随机采样或优化。

```bash
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/build_rigid_dataset.py \
 --output outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1 --plan
# 核对磁盘receipt后，同命令把--plan改为--execute。
```

结果以新目录report.json、逐窗verdict/provenance、dedup-pairs、新索引和覆盖CSV为准。
执行前不预填admitted/认证/去重数量；没有训练则预测误差变化为未测。
