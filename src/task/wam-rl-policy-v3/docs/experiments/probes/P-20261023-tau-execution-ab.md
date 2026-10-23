---
schema: ref2dex.probe.v2
probe_id: P-20261023-tau-execution-ab
experiment_id: P-20261023-tau-execution-ab
date: 2026-10-23
task: wam-rl-policy-v3
branch: main
git_commit: 98d115327215a5922028861c25174b579ac4b55b
claim_id: C1
hypothesis_family: HF-tau-execution-ab
probe_index_in_family: 1
seed_pool: probe
seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
classification: Decision
status: COMPLETED
run_id: tau-checkpoint-ab-20261023-r1
decision_changed_if_positive: use step001223 as main tau and queue component decomposition with it if net successes improve without more safety aborts
decision_changed_if_negative: locate the execution stage reached; if successes decrease stop EPE-only tau acceptance and inspect consumed trajectory segment
---

# 结果解读表

| 本轮结果（旧11000→新1223） | 执行环节解释 | 决定 |
|---|---|---|
| 成功2/20→2/20；新增1/丢失1 | 无净成功收益 | 不提升新τ为主线 |
| 安全14→9；joint8→6/table6→3；contact未认证3→8 | 减少安全中止，但接近到闭合仍未打通 | 定位R前8帧腕指联合与闭合准入 |
| stage approach6→4，contact/lift/hold均4/2/2；进入close11→4 | 阶段收益未形成，不能把不abort当成功 | 不凭EPE改善替代执行验收 |

Result: paired20 OLD2/20 NEW2/20; safety14→9, joint8→6, table6→3; contact-not-certified3→8; exact initial20/20 and raw safety40/40.
Decision: no main-tau promotion; inspect approach-to-close progression and the first8-frame R wrist/finger consumption.

[完整逐例、首触限/阶段/误差分布、实际消费段与复现](../../audit/tau-checkpoint-execution-ab-20261023.md)。14,775controls、0updates，17回归通过；独立汇总一致。

事前固定判定规则：

| 配对结果 | 解释 | 决定 |
|---|---|---|
| 新成功数增加且安全中止不增 | 预测改善转为本面板执行收益 | 主线采用新 τ，排新权重成分分解 |
| 成功不变，失败或阶段分布改变 | 改善只影响执行链局部，不能称成功收益 | 明确变化所在环节 |
| 新成功数下降 | 面板 EPE 与本执行需求解耦 | 停止 EPE 作为执行验收，查实际消费段 |
| 成功增加但安全中止增加 | 安全约束未满足 | 不直接提升为主线 |

最便宜的可判别设计：既有 `grasp-v11-coverage40-20261022-r1` 原顺序前20个
reset（20物体各第一条序列），同 native/model seed、reference、原生状态初始化，
OLD/NEW顺序交替。没有单独找到已冻结的20例文件，因此明确使用既有40例的固定前缀；
不是新采样，也不按模型结果筛选。原始n40历史率不混入本轮分母。

唯一实验变量：OLD step011000 与 NEW `tmp/t38-r2/training/step001223.pt`。
两臂同一当前 v6 inference package 支持显式 condition policy，均传 `goal=None` 与
unknown/invalid event，真实 `model.encode` 边界断言goal三字段和event三字段为零/invalid。
v3执行器、相位机、安全门、阶段评分及尺寸目标成功判据全部沿用原v11冻结代码；
冻结旧评分 `staged_progress.py`，不改当前生产默认checkpoint，等待结果裁决。
预测24帧，前8帧输入同R，完整24帧另供腕部plan；只在approach/close消费τ。

产物 `outputs/wam-rl-policy-v3/tau-checkpoint-ab-20261023-r1/`：registration、各worker
原始diff、实际runner/runtime-code、raw trajectory/initial、native audit、result/manifest。
work_version `tau-execution-ab-v1`；GPU4/5/6各worker，7不用，0–3不动。
最多40回合、24,000controls、每worker2h、零训练更新；预计<512MiB、硬预算4GiB，
可用空间<20GiB停止。启动前实测129.33GiB可用，不复制权重或几何cache，不生成视频。
环境：graspenv Python3.8/Torch2.4.1+cu121；native固定venv Python3.8/Torch2.0.1+cu118。

复跑：`tools/run/run_tau_checkpoint_ab.py --registration <run>/registration.json
--worker 0|1|2 --gpu 4|5|6 --output <run>/g4|g5|g6`，使用新输出目录。
结果必须包含配对成功转换、安全中止/死因、首个触限关节、阶段累计与最高阶段、
目标误差分位数与逐reset变化，并验证20对初态逐字段一致。

局限：选定且已暴露的固定小面板，不能证明泛化；接触是现有实测认证口径，非新增接触对
真值；tau新旧训练recipe不同，不能从本A/B分离goal/event因素。成分分解本轮不运行。
