---
schema: ref2dex.research-report.v1
report_id: P-20261023-bc-ppo-usefulness-results
date: 2026-10-23
task: wam-rl-policy-v3
experiment_id: P-20261023-bc-ppo-usefulness
branch: main
run_id: bc-ppo-usefulness-20261023-r1
status: INCONCLUSIVE_REDUCED_BUDGET
---

# 结果解读表

| 实测结果 | 含义 | 决定 |
|---|---|---|
| **预算缩小：14400 controls/arm**；确定性成功BC 0/6 (0.0%) vs PPO 0/6 (0.0%)；joint-limit abort 3→4；改善0/6、回退1/6 | 未满足≥3初态改善无回退 | 本次预算缩小运行未达预注册≥3初态改善无回退；完整1e5分支尚未检验，不能下“PPO无效”结论。 |
| 训练采样BC 0/30、PPO 2/35，PPO成功来自2个初态；最终均值均0/6 | 有真实随机执行成功见证，尚未保留为稳定部署收益 | 保存成功轨迹；不以训练采样替代≥3初态改善无回退门 |
| 原1e5/arm **未测满**，预注册判据没有更改 | 不能报告“1e5无改善”；短预算的无改善只保留预算/覆盖不充分可能性 | 本次不触发1e6排卡；白天优先补足同配方1e5，再按原判据决定1e6；不据此改算法或否决PPO |
| accepted exact joint KL 0.0000234–0.0095190，clip 0.0000–0.1616；0 retries/72 attempts | 精确KL硬门通过，首步异常与回滚频率按下方序列检查 | 数值实现未见旧故障；不把KL小等同任务有效 |
| preflight BC 3.124/PPO 4.060 controls/s，均<8；原配对顺序15.73h | 原规模吞吐账失败 | 按机主授权同比例缩至14.4%，GPU7顺序完成 |

结果时间：2026-10-23 07:00:03 +0800。所有数字只属于固定、已暴露的六初态，
不是独立泛化或完整RELEASE资格。BC冻结；PPO学习/训练时采样与最终均值评估分开报告。

## 预算与实际执行

| 阶段 | 实测controls | 终止回合/预算截断回合（终止含安全早停） | optimizer updates | worker wall秒 |
|---|---:|---:|---:|---:|
| bc | 14400 | 30/1 | 0 | 4015.06 |
| ppo | 14400 | 35/1 | 72 | 3748.42 |
| assessment-bc | 2543 | 6/0 | 0 | 635.20 |
| assessment-ppo | 1292 | 6/0 | 0 | 396.66 |

preflight另计PPO3600+BC600 controls，22 updates；不继承其学习结果。
主预算BC=PPO=14400实际controls。raw字段partial_budget_episode表示剩余预算使计划horizon缩短，可能仍先触发物理早停；上表按实际physical_abort区分安全终止与真正预算截断。末尾不足600不当完整成功回合。
最终两个评估各6初态、相同600-controls上限，物理早停导致实际评估controls可以不同。
评估与preflight的controls在上表/此处明确另列，没有伪装为1e5训练。
所有物理controls合计36835，
主PPO更新72次；controls/s与updates/s没有混用。更新频率以总墙钟为分母，纯优化计算耗时另见每轮`ppo_update.wall_s`。
顺序总预计与08:00截止账见[preflight实测报告](ppo-bc-usefulness-preflight-20261023.md)，
主流程实际从04:28:51至06:57:31，
supervisor状态COMPLETED，截止08:00前完成情况以此时间为准。

主预算按相同六初态及相同循环顺序采样至等量controls；因物理早停不同，重复回合数可以不同。最终终点评估每初态严格一对，不以训练分母不同的总体成功率做主比较。

## 六初态的固定判据

改善=成功增加或physical_joint_limit回合数减少；回退=成功减少或joint abort增加。
成功仍取600 controls末态的实测连续hold、原尺寸相关目标容差和物理安全门。
joint abort保留原安全裕量：平移DOF0–2为0.10m，其余受限转动DOF为0.10rad；名义URDF越界另列。
任一指标冲突也算回退。失败签名变化不被隐藏为任务成功。
全部6对初态逐数组校验通过，最大绝对差0，
native seeds294–299、model seed293。评估manifest的ppo=False表示均值前向且0 updates；PPO评估的residual_checkpoint指向主训练ppo/residual-final.pt。每初态仅一次终点评估，不宣称统计显著性。

| 初态 | 成功BC→PPO | joint abort BC→PPO | 改善 | 回退 | 失败签名BC→PPO |
|---|---:|---:|---|---|---|
| s1_cylindersmall_pass_1 | 0→0 | 0→0 | False | False | stage_contact_not_certified→stage_contact_not_certified |
| s10_alarmclock_pass_1 | 0→0 | 0→0 | False | False | stage_contact_not_certified→physical_table_penetration |
| s10_banana_pass_1 | 0→0 | 0→1 | False | True | physical_table_penetration→physical_joint_limit |
| s10_cubesmall_inspect_1 | 0→0 | 1→1 | False | False | physical_joint_limit→physical_joint_limit |
| s10_cup_drink_1 | 0→0 | 1→1 | False | False | physical_joint_limit→physical_joint_limit |
| s10_cylindermedium_inspect_1 | 0→0 | 1→1 | False | False | physical_joint_limit→physical_joint_limit |

| 终点评估指标 | BC-only | BC+PPO |
|---|---:|---:|
| 关节安全门中止回合数 | 3 | 4 |
| 末态触门关节事件数 | 4 | 4 |
| 控制后名义URDF越界状态数 | 117 | 12 |

名义越界状态数按实际控制后状态计数，不能与安全门中止回合数混算。两arm实际评估暴露为2543/1292 controls，PPO存在更早安全中止；117→12不能单独宣称关节安全改善，主要指标仍为中止回合3→4。

## 训练采样中的成功见证

排除实际预算截断后，BC为0/30、PPO为2/35（5.7%）；两种采样模式不同，主采样分母也不同，不能用这个比率替代确定性配对。

| PPO采样回合 | 初态 | controls | 最长连续hold | 末态目标误差 | 安全中止 |
|---|---|---:|---:|---:|---|
| p0005-A1 | s10_cylindermedium_inspect_1 | 600 | 17.9667s | 3.4977mm | 无 |
| p0034-A1 | s10_cup_drink_1 | 600 | 16.6333s | 1.3859mm | 无 |

两个成功均由原始物理轨迹重放确认，属于当前冻结测量合同的自主随机采样见证，不是teacher动作回放。
最终均值未复现为成功（0/6），且只覆盖2个成功初态，不满足≥3且无回退的原门；它们不能直接证明PPO更新带来稳定收益。
原始路径分别为`ppo/p0005-A1/trajectory.npz`与`ppo/p0034-A1/trajectory.npz`，相对本文末列出的运行输出根目录。

## 失败签名分布

确定性BC：`{"stage_contact_not_certified": 2, "physical_table_penetration": 1, "physical_joint_limit": 3}`。
确定性PPO：`{"stage_contact_not_certified": 1, "physical_table_penetration": 1, "physical_joint_limit": 4}`。
阶段签名只表示当前测量/认证合同，`stage_contact_not_certified`不等价物理完全无接触。
主采样过程（回合数/长度可不同，不能拿其成功率替代上面的配对终点）：

- bc：`{"stage_contact_not_certified": 20, "physical_joint_limit": 5, "physical_table_penetration": 5, "budget_time_truncation": 1}`。
- ppo：`{"physical_table_penetration": 5, "stage_contact_not_certified": 13, "physical_joint_limit": 11, "success": 2, "contact_without_lift": 3, "held_off_goal": 1, "budget_time_truncation": 1}`。

冻结BC主采样的既有重复（排除真正预算截断；各例初态/记录seed/权重相同）：

| 初态 | 回合 | 成功 | joint abort | 终止签名 |
|---|---:|---:|---:|---|
| s1_cylindersmall_pass_1 | 5 | 0 | 0 | {"timeout": 5} |
| s10_alarmclock_pass_1 | 5 | 0 | 2 | {"physical_joint_limit": 2, "timeout": 3} |
| s10_banana_pass_1 | 5 | 0 | 1 | {"physical_joint_limit": 1, "timeout": 3, "physical_table_penetration": 1} |
| s10_cubesmall_inspect_1 | 5 | 0 | 0 | {"physical_table_penetration": 1, "timeout": 4} |
| s10_cup_drink_1 | 5 | 0 | 1 | {"timeout": 3, "physical_joint_limit": 1, "physical_table_penetration": 1} |
| s10_cylindermedium_inspect_1 | 5 | 0 | 1 | {"timeout": 2, "physical_table_penetration": 2, "physical_joint_limit": 1} |

每个初态的5次重复已逐数组核对，初态最大元素差为0，native/model seed均相同。相同已记录条件下，基线存在重复间差异；未定位具体随机/数值来源。原终点配对判据不变，这些重复用于说明不确定性，不把一次终点差异自动升级为稳健因果/泛化结论。

## PPO每轮KL / clip / 熵 / 回报序列

KL是12D Gaussian joint exact KL，若需per-dim则除以12；每格按两个epoch顺序。
全更新KL中位数0.0004155，51/72次低于1e-3。低端更新小于预注册正常量级，
因此“通过.03上限门”不自动等于完整满足“KL在1e-3–5e-2量级”的分支，更不等于任务收益。
clip为PPO probability ratio clip fraction，与最终动作裁剪分开。
熵为12D differential entropy，可以为负；回报为该次on-policy采样的reward sum。
主采样回报范围-1.373072至1.754430，不作为确定性成功率的替代。

| 回合 | 累计controls | exact joint KL | clip fraction | 末epoch熵 | 回报 | retries |
|---|---:|---|---|---:|---:|---:|
| p0000-A1 | 297 | 0.0057718, 0.0095190 | 0.0943, 0.1616 | -12.972698 | -1.234033 | 0 |
| p0001-A1 | 897 | 0.0003086, 0.0018882 | 0.0000, 0.0217 | -12.972700 | -0.671997 | 0 |
| p0002-A1 | 1048 | 0.0002428, 0.0011089 | 0.0000, 0.0066 | -12.972723 | -0.722047 | 0 |
| p0003-A1 | 1202 | 0.0001815, 0.0010004 | 0.0000, 0.0000 | -12.972733 | -1.157715 | 0 |
| p0004-A1 | 1549 | 0.0000979, 0.0004065 | 0.0000, 0.0000 | -12.972757 | -0.975671 | 0 |
| p0005-A1 | 2149 | 0.0001071, 0.0007431 | 0.0000, 0.0000 | -12.972782 | 1.176187 | 0 |
| p0006-A1 | 2293 | 0.0001371, 0.0006118 | 0.0000, 0.0000 | -12.972812 | -1.101467 | 0 |
| p0007-A1 | 2893 | 0.0000474, 0.0002038 | 0.0000, 0.0000 | -12.972838 | -0.659687 | 0 |
| p0008-A1 | 3493 | 0.0001744, 0.0008967 | 0.0000, 0.0000 | -12.972867 | -0.789433 | 0 |
| p0009-A1 | 3574 | 0.0004649, 0.0018987 | 0.0000, 0.0123 | -12.972895 | -1.090403 | 0 |
| p0010-A1 | 4174 | 0.0001068, 0.0004171 | 0.0000, 0.0000 | -12.972917 | -0.167075 | 0 |
| p0011-A1 | 4774 | 0.0000504, 0.0001916 | 0.0000, 0.0000 | -12.972940 | 0.555656 | 0 |
| p0012-A1 | 5374 | 0.0004796, 0.0024693 | 0.0000, 0.0067 | -12.972966 | -0.467752 | 0 |
| p0013-A1 | 5974 | 0.0003218, 0.0013468 | 0.0000, 0.0017 | -12.972986 | -0.686658 | 0 |
| p0014-A1 | 6030 | 0.0006208, 0.0029102 | 0.0000, 0.0179 | -12.972999 | -0.628318 | 0 |
| p0015-A1 | 6630 | 0.0002761, 0.0011377 | 0.0000, 0.0083 | -12.973015 | -0.441434 | 0 |
| p0016-A1 | 6887 | 0.0003736, 0.0013726 | 0.0000, 0.0117 | -12.973043 | -0.830841 | 0 |
| p0017-A1 | 7487 | 0.0000679, 0.0003273 | 0.0000, 0.0000 | -12.973073 | -0.038351 | 0 |
| p0018-A1 | 8087 | 0.0003003, 0.0010195 | 0.0000, 0.0000 | -12.973096 | -0.736023 | 0 |
| p0019-A1 | 8221 | 0.0002080, 0.0006996 | 0.0000, 0.0000 | -12.973127 | -1.201974 | 0 |
| p0020-A1 | 8821 | 0.0005774, 0.0034176 | 0.0000, 0.0200 | -12.973151 | -0.285268 | 0 |
| p0021-A1 | 8918 | 0.0002599, 0.0009940 | 0.0000, 0.0103 | -12.973180 | -1.100353 | 0 |
| p0022-A1 | 9316 | 0.0001238, 0.0004440 | 0.0000, 0.0000 | -12.973214 | -1.373072 | 0 |
| p0023-A1 | 9368 | 0.0002960, 0.0015294 | 0.0000, 0.0000 | -12.973260 | -0.637172 | 0 |
| p0024-A1 | 9968 | 0.0005918, 0.0028283 | 0.0000, 0.0067 | -12.973299 | -0.380456 | 0 |
| p0025-A1 | 10568 | 0.0002882, 0.0010948 | 0.0000, 0.0000 | -12.973334 | -1.021155 | 0 |
| p0026-A1 | 10639 | 0.0003538, 0.0017273 | 0.0000, 0.0000 | -12.973363 | -1.059573 | 0 |
| p0027-A1 | 11239 | 0.0001871, 0.0008037 | 0.0000, 0.0017 | -12.973370 | -0.549702 | 0 |
| p0028-A1 | 11839 | 0.0000814, 0.0001730 | 0.0000, 0.0000 | -12.973368 | -0.703217 | 0 |
| p0029-A1 | 12291 | 0.0000812, 0.0004586 | 0.0000, 0.0000 | -12.973356 | -1.274430 | 0 |
| p0030-A1 | 12891 | 0.0000234, 0.0001933 | 0.0000, 0.0000 | -12.973331 | -0.394080 | 0 |
| p0031-A1 | 13491 | 0.0000683, 0.0004553 | 0.0000, 0.0000 | -12.973316 | -1.067882 | 0 |
| p0032-A1 | 13595 | 0.0001924, 0.0010051 | 0.0000, 0.0096 | -12.973296 | -1.083795 | 0 |
| p0033-A1 | 13776 | 0.0002399, 0.0011301 | 0.0000, 0.0000 | -12.973286 | -1.168898 | 0 |
| p0034-A1 | 14376 | 0.0004140, 0.0022727 | 0.0000, 0.0050 | -12.973275 | 1.754430 | 0 |
| p0035-A1 | 14400 | 0.0003549, 0.0018295 | 0.0000, 0.0000 | -12.973277 | -0.023426 | 0 |

完整逐epoch/loss/lr/reward/return/bootstrap/attempts记录在`summary-final.json`的`update_sequence`，
以及各episode `result.json`、`ppo-rollout.npz`和`reward-parts.json`。
保持Adam1e-5、2epochs、targetKL.03、超门1/4lr重试、gamma.96、GAE.95、clip.2、
原10mm/40mrad/30mrad残差尺度；无v6或BC重训、无新baseline。

## 复核、环境和产物

- raw-bc：PASS，31回合。
- raw-ppo：PASS，36回合。
- raw-assessment-bc：PASS，6回合。
- raw-assessment-ppo：PASS，6回合。
- update-replay：PASS，36回合，重放模型与已保存模型的末态参数最大误差0。

物理重放直接只读运行现有`rescore_large_grasp.py`，用每run冻结评分代码重算原始state、
成功/阶段/PD/奖励，不采用summary标签自证。它保留现有observability局限，非外部接触真值。
GPU7更新重放从相同BC重新应用全部已保存batch，核对GAE/精确KL/裁剪/奖励分项/界和最终参数，
不是新增物理训练预算。相关回归12 passed；本轮Python AST、文档链接、实验卡及seed检查PASS。
全仓`tools/verify.py --changed`仍FAIL：另一张`ptv3-shared-wam-v6/P-20261023-tau-fixed-panel-metric`卡缺claim_id、hypothesis_family、probe_index_in_family，并触发index正整数检查，共4条；未修改该他人文件。
完整日志为`verify-final.log`，本轮针对性检查为`owned-validation.json`。

策略环境`/home2/wyy/miniconda3/envs/graspenv/bin/python`，Torch2.4.1+cu121、CUDA12.1；
原生固定venv/Torch2.0.1+cu118、GPU PhysX、1env、packed IPC、同步setters，详见每episode physics manifest。
main PPO启动HEAD `59fc5f2afcc2d14604aeb2b79ce3d4202296ee98`；BC启动HEAD及实际runner来源独立记录在`code-provenance.json`。BC运行快照已与独立提交88ea57b2逐字节核对，运行中未换源码。冻结依赖与preflight相同，
两arm同一BC起点、R/WAM、686/698维观测、12维latent/动作和phase机。
与现有评测口径的差异/保护文件只读边界已在preflight报告逐项列明，没有借source-ceiling资格判据。

只用GPU7，采样显存峰值6045MiB，采样最小磁盘余量132.23GiB；
当前全输出0.431GB<5GB，无视频、checkpoint每40回合及最终，无覆盖旧产物。
文件根`outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/`：
`preflight-summary.json`、`paired-registration.json`、`assessment-registration.json`、
`paired-supervisor.json`、`resources.jsonl`、`summary-final.json`和上述5个重放报告。
[实验卡](experiments/probes/P-20261023-bc-ppo-usefulness.md)保留原判据与预算修订。

![PPO numeric sequence](../../../../outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/ppo-metrics.png)

[Standalone SVG](../../../../outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/ppo-metrics.svg)
