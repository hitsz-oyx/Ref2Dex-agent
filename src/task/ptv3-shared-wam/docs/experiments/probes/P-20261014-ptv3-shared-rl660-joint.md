---
schema: ref2dex.probe.v2
probe_id: P-20261014-ptv3-shared-rl660-joint
experiment_id: P-20261014-ptv3-shared-rl660-joint
date: 2026-10-14
task: ptv3-shared-wam
branch: main
git_commit: 7dfcdec21cd611642a3c9b507136caa8b0ca94c5
claim_id: C1
hypothesis_family: HF-ptv3-shared-rl660-joint
probe_index_in_family: 1
seed_pool: probe
seeds: [124]
decision_changed_if_positive: evaluate the jointly adapted forecaster and Chunk-R on held-out complete robot motions
decision_changed_if_negative: distinguish teacher data quality and retargeter mismatch before changing the shared architecture
status: RUNNING
run_id: rl660-corpus-20261014-r1
---

# 全660条DExplore轨迹：同执行补录与共享预测器/Chunk-R联合微调

Result: RUNNING. 已停止三域预训练并保留可用38k权重；两条真实补录通过命令/PD合同，完整660补录和联合微调准备中。
Decision: 用户明确选择补录完整机器人执行，再联合微调WAM+Chunk-R；不只做airplane。

Class: Decision. 服务于自训练机器人执行底座与可用交互预测。检验固定共享goal架构
在多物体机器人执行上的手/物体/命令预测是否可用；改善后进入独立闭环，不改善则
先区分数据失败轨迹、控制合同与适配缺陷，不把BC loss当成RL增益。

## Decision Note

- 用户授权停止原80k，改用完整660条RL重定向数据，并确认补录实际手点、状态和命令。
- 既有660导出共有166337帧、50类物体，仅qpos373:391和object198:205是实际执行，
  其余人体字段复制自几何参考；没有实际PD命令，不能直接用于完整WAM+R。
- 采用原660名单、对应几何计划及已有official actor作为只读数据生成器；模型只从
  自训练三域38k初始化，Chunk-R从头学习，official actor权重不进入最终模型。
- 采集两张空闲卡4/5，每批16个独立motion，最多3小时、4GB；不影响0–3上的其他工作。
  单批最多900秒，失败、非有限值、命令合同不符或资源超界时停所属进程，保存失败记录。
- 新数据仍按完整序列划分，按物体分层约10%dev，每物体至少一条。全部660为采集名单，
  TRAIN/dev之间不共享同一序列的窗口。准入质量策略在正式训练前固定并报告，不默认为
  所有official rollout都是成功抓取。
- 用户已确认排除明显失跟/掉落窗口。固定门为当前及未来24步的实际物体位置与
  已知参考位置差均<=0.10m；只作为明显失跟准入，不等同于成功抓取判定。
  每序列记录raw/admitted/excluded_windows及固定tick列表；TRAIN-only归一化也仅使用
  准入窗口。没有准入窗口的序列保留原记录并报告，不能声称660条都贡献了训练标签。
- 微调计划：2卡、每卡128/global256、20000 updates，保留既有全forecaster+R与GT-tau
  anchor、K4；不改goal、标签坐标或架构。一次GPU工程检查后启动；max_seconds=86400，
  硬上限4GB训练产物/400GB共享总量，使用已有监督器。无新权限边界。
- 成功后选择dev best做独立native执行；失败保留best/latest和质量分组，先定位标签/控制。

## 采集与数据合同

工具：[collect_rl660_batch.py](../../../tools/run/collect_rl660_batch.py)、
[collect_rl660.py](../../../tools/run/collect_rl660.py)。固定legacy Torch2.0.1/Isaac runtime，
GPU pipeline、30Hz，从真实frame0重置；每个环境对应一个motion，禁止隐式hard-object
oversampling。记录q/dq、11个真实native rigid-body link点、动态物体SE3/velocity、
实际送入pre_physics的规范化命令和实际PD targets。控制t对应state[t+1]，保留terminal。
用独立pd_from_action复算同tick转换，逐步检查捕获命令完全一致、PD差<=1e-6。
参考只作为已知计划和质量检查；监督手/物体均来自本次同一执行，不混用旧660人体坐标。
每物体canonical512点取自实际URDF collision frame，保留32同ID query。

原始成员与字段审计：`outputs/ptv3-shared-wam/rl660-source-audit-20261014-r1/summary.json`。
补录产物：`outputs/ptv3-shared-wam/rl660-corpus-20261014-r1/`；只在完整660回执、唯一成员
和一致controller检查通过后发布COMPLETED/training_allowed=true。

## 工程记录

- `engineering-rl660-collection-20261014-r1`未完成，旧runtime player构造参数不兼容，
  原始日志保留；修复为legacy flat config后重新独立运行，不视作方法负证据。
- `engineering-rl660-collection-20261014-r2`：GPU4，airplane432帧和mug334帧，42.92秒。
  逐步命令/PD检查通过，所有状态与命令有限，实际GPU观察约4213MiB。
  物体reference RMSE分别0.079m/0.865m；mug存在明显失跟，数值合同通过不等于成功示范。
- pilot只用于接线，TRAIN=airplane、dev=mug；不作为多物体方法收益证据。
  全量runtime与模型工程检查结果后续在本卡补充。

## 初始化与checkpoint保存缺陷

预训练停在最后日志39100，完整latest为step38000，objective1.6881；已有best与latest
均指向38k。历史最低22k/objective1.5545没有被独立保留，已不可恢复。初始化必须说明
是best available，不声称历史真正best。原20k、38k和全部curve/日志保留。
已补回归检查重现best被恢复点清理的缺陷；新训练验证创新低时立即保存完整checkpoint，
原子维护best/latest，清理时保护alias所指的登记文件。不能据此追回历史权重。

## Limitations / future evidence

单seed离线Probe，不证明机器人抓取、多物体泛化或最终Cm因果增益。660来自official
actor作为数据生成器，不能称全部是自训练RL演示。重采集与原导出不要求逐位一致，
关键是本次手/物体/动作配对和控制时钟正确。后续必须核对质量分组与真实闭环；
近10%序列dev不是物体不相交测试，不能据此声称unseen-object泛化。
