---
schema: ref2dex.probe.v2
probe_id: P-20261016-shared-action-horizon
experiment_id: P-20261016-shared-action-horizon
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: af861865
claim_id: C1
hypothesis_family: HF-shared-action-horizon
probe_index_in_family: 1
seed_pool: probe
seeds: [198, 192, 148]
decision_changed_if_positive: use current721 plus first-action supervision as the next matched shared route
decision_changed_if_negative: retain full-chunk supervision for current-input shared training and investigate input bottleneck separately
status: UNCLEAR
run_id: shared-action-horizon-20261016-r1
---

# 当前输入与动作监督跨度：执行结果及语义审查更正

Class: Decision。服务C1。预定四臂masked/delta0当前输入×1/8拍动作监督，均从shared7k
权重起步、原WAM/R联合训练及forecast保持。动作输出始终8×15D，forecast另有24拍。
8拍臂原先从shared-current的1k点恢复到7k，1拍臂从同一7k parent fresh optimizer训练7k。
数据strict TRAIN93680/DEV10376、原DEV256、seed198/192/native148；global256，原三LR。
GPU4/5单拍、6/7八拍，GPU1腕部面板，2/3空闲。每臂14000s/7GB、磁盘底线20GiB。
原运行记录和checkpoint保留，不将预算停止当作算法失败。

## 资源恢复与实际代码路径

02:10前后原continuation两臂在step4000之后触及7GB artifact budget停止，随后从完整
step4000恢复模型/Adam/RNG，改为稀疏validation/checkpoint cadence，在
continuation-{masked8,delta08}-sparse完成7000。模型/数据/学习率未因资源恢复改变。

single-masked1在step5000、single-delta01在step6000后也进入sparse恢复，但其实际执行
源码发生了下述监督语义变化。原始日志、失败和恢复的process.command均保留。

## 审查更正：原8拍判无效有误，单拍尾段有真实错配

此前04:55审查把forecast24拍混同为action24拍，错误判定未写action_supervision_steps
的continuation为24拍，并为此重跑fresh8。实际process.command所指旧固定树中，actor
只有8个action_queries、action_scale为8×15、标签也是8×15，robot_loss直接对全部8拍
求均值。新版缺省取action_error.shape[1]，也等于8。因此撤销原continuation的
“24拍INVALID_IMPLEMENTATION”判定，shared-current的1k不存在所谓24拍污染。

fresh8仍是有效8拍运行，但属于额外重复，不是修复24→8所必需。原结果不删除，也不
挑其中更好的一组冒充唯一预设执行。两个fresh8最终manifest均COMPLETED。

真正的实现错误发生在single稀疏恢复：两条single-*-sparse/process.json都指向
`tmp/shared-current-code-20261016-cont7k-sparse/src/task/ptv3-shared-wam-v2/tools/run/train_robot.py`。
该固定树的robot_loss完全不读取action_supervision_steps，仍监督全部8拍。因此：

- single-masked1-sparse：前5000步单拍，后2000步八拍。
- single-delta01-sparse：前6000步单拍，后1000步八拍。

这两个7k端点对“完整7k单拍”标INVALID_IMPLEMENTATION；作为混合协议结果保留，不能
据其归因单拍监督优于8拍。manifest写1和provenance里的主仓库HEAD均不足以认证实际
执行语义。旧git_commit按REF2DEX_RUN_ROOT主仓库取值，不能替代实际导入源码核对。

CPU最小复现抽取实际robot_loss，使用(1,8,15)首拍误差0、后7拍误差1、配置监督1拍：
旧sparse函数loss0.875，新函数0；新版缺省和显式8均0.875，显式24越界报错。native日志
R_output_chunk实际为(controls,2,8,15)，24×33是R_tau。独立只读review一致。
审计入口`tools/audit/audit_horizon_runtime.py`，更正证据为
`outputs/ptv3-shared-wam-v2/horizon-runtime-audit-20261016-r1/correction.json`。
原results_summary.json保留不覆写，其24拍和invalid分类已由本更正取代。

## 最终离线读出：区分有效8拍与混合协议

| 运行 | 实际action监督 | XYZ mean / p95 mm |
| --- | --- | ---: |
| continuation-masked8-sparse | 8拍7000步 | 59.76 / 173.90 |
| continuation-delta08-sparse | 8拍7000步 | 55.91 / 163.36 |
| fresh8-masked（额外重复） | 8拍7000步 | 60.09 / 174.50 |
| fresh8-delta0（额外重复） | 8拍7000步 | 56.13 / 164.95 |
| single-masked1-sparse | 1拍5000 + 8拍2000 | 59.07 / 165.41 |
| single-delta01-sparse | 1拍6000 + 8拍1000 | 55.14 / 161.31 |

这些端点不能再作为完整四组“1拍vs8拍”对照。有效8拍配对和额外重复均显示delta0
在离线首控XYZ上有收益；是否改变监督跨度，当前证据不足。

## 已执行native：只对应各自实际训练协议

seed148、飞机149/球141/杯172 controls、K1。指标partial_hold45_only_no_placement。
完成面板的independent wiring audit通过，只证明所查执行合同，不能认证训练目标。

| 权重 / 对象 | 学生max held | clipping | 状态 |
| --- | ---: | ---: | --- |
| fresh8-masked / 飞机、球、杯 | 0 / 0 / 0 | 41.6% / 8.5% / 19.8% | 完成 |
| fresh8-delta0 / 飞机 | 0 | 13.4% | 完成 |
| fresh8-delta0 / 杯 | 60 | 0% | 完成，有保持 |
| fresh8-delta0 / 球 | — | — | query87越出固定PTv3 workspace |
| single-masked混合 / 飞机、球、杯 | 0 / 0 / 0 | 12.8% / 0% / 4.1% | 完成 |
| single-delta0混合 / 飞机 | — | — | query88越出固定PTv3 workspace |
| single-delta0混合 / 球 | 0 | 22.0% | 完成 |
| single-delta0混合 / 杯 | 117 | 0% | 完成，不能归因单拍 |

当前仍UNCLEAR：有当前信息的8拍策略在杯子出现局部保持信号，但其他对象仍失败/越界；
撤销“完整单拍略优于8拍”的归因，不进入基于该结论的多seed Validation。

## 最便宜的有效后续

四臂均有尚未错误恢复的共同4k checkpoint。已存离线mean/p95：原8拍masked60.17/174.28、
delta0 57.85/170.52；单拍masked56.28/170.82、delta0 53.61/164.65mm。优先补核这四个
前缀的初始化/样本/预算匹配并做固定闭环，成本低于重跑全部训练；本次仅审查，不新开长训。

完整输出仍在`outputs/ptv3-shared-wam-v2/shared-action-horizon-20261016-r1/`。
后续residual集成仍需显式adapter及fresh zero-residual检查；现有离线apply_residual(base,0)
恒等检查不等于完整server集成，本Probe未启动PPO。
