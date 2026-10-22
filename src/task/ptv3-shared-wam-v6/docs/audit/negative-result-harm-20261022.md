# UNKNOWN / voxel 负结果损害边界（2026-10-22，CPU）

只读已有JSON、冻结代码和P0状态；没有新仿真、训练、GPU操作，也未扩大grid或删除gate。
本报告区分“受影响回合消耗的交互”与“可避免的浪费”：后者的因果数值未被识别。

## 已核计数与集合界

- 实际r3使用WAM v4，冻结`3a0779f6981ee01ffb8e637abb9b7b3fea2437f3`。iteration1570：27792/53174 = **52.266145%**；最终1575：27869/53347 = **52.240988%**。三个rank重复同一全局reduce，不求和。
- 实际501份task：HOLD416 / RELEASE21 / UNKNOWN64。最终初始化draw56449，其中UNKNOWN9314（16.49985%）；draw不是结束回合。名义权重16.95252%也不是实际结束比例。
- 同501份安全prefix合计**56418 commands**，UNKNOWN **2150 / 56418 = 3.810841%**。HOLD49514、RELEASE4754。这是BC准入候选池，不是实际minibatch污染率。
- 实际Scheduled686 cache的layout已逐序列/source/lo-hi核对501/501，train-valid.npy长度56418；它只表示tau有效，不是task-active/include。BC manifest实际task-active train_rows56036；未保存逐行include或sampler访问，UNKNOWN实际BC采样占比仍UNKNOWN，不根据startup_half猜测。
- 冻结host采样+PPO eager reset保证：每个已结束回合对应一次不同初始化。D=56449，E=53347，未匹配draw的容量D−E=3102；它含startup替换/在途/日志尾部，不能指定成员。
- 因而UNKNOWN结束回合U ∈ **[6212, 9314]**，公式max(0,9314−3102) ≤ U ≤9314。这是集合界，**没有把16.5%当作episode比例**。
- UNKNOWN×voxel重叠 ∈ [0, 9314]；二者并集 ∈ **[27869, 37183]**，不能相加计损。
- 同时排除UNKNOWN与voxel后，剩余结束回合N ∈ **[16164, 25478]**。只能写“该变化中的v4训练策略，在定义终态且非voxel退出的结束回合里，仍未观测到完整TaskController成功；确切分母不可恢复”。
- 不能写“修复后二者仍0成功”、0/53347独立评估、v6失败、抓取0，或对相关训练回合套独立二项置信界。去gate会改变未来动作/reset/采样，反事实成功UNKNOWN。
- 计数表明UNKNOWN结束回合不能覆盖所有失败回合；不排除UNKNOWN训练数据通过共享参数影响其余任务。该间接学习损害也不能由集合界识别。

## 交互预算：宽界也是结论

- r3从累计iteration231恢复，新完成1344次更新：**16515072**个control transitions。native实际**16527360**，多12288个已采未更新交互；尾部30次reset的结果未记录，不把最后日志0延伸到它。
- 以下仅界定“已结束受影响回合内部消耗的transition”，不是有用梯度损失、可回收步数或GPU小时。历史未存每回合length/sequence，不能估计精确损耗。

| 范围 | 回合数界 | 该类已结束回合内部transition界 | 占已更新预算界 |
|---|---:|---:|---:|
| unknown | [6212, 9314] | [18636, 16382973] | [0.112842, 99.200131]% |
| voxel | [27869, 27869] | [83607, 16438638] | [0.506247, 99.537186]% |
| unknown_or_voxel | [27869, 37183] | [83607, 16466580] | [0.506247, 99.706377]% |

界的前提：每个已结束失败回合至少3controls（H4首检查在3，reset_probe关闭，其余超时≥8s），最多60s×30Hz=1800；其他已结束回合也至少3controls。
下界m×n_low；上界min(1800×n_high,T−m×(E−n_high))。完整任务UNKNOWN结构性无成功分支，但其局部接近/接触reward和学习信号不必无用。voxel可能拦住真实发散，假阳性与因果损害下界不能由计数推出。

## A3：既有P0状态的协议比较（单列，不能归因历史27792）

- 来源`/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/flow/trajectory.npz`，16环境×ticks3..300 = **4768**个H4状态；V4 float32原gate与已存V6累计标记全部一致（只认证这些状态，不宣称float32/float64边界通用等价）。
- 首个H4准入16/16通过；逐tick有2/16越界，原始不合法状态37/4768。admission-only保留4800controls，逐tick截断保留4744，差56；主要安全prefix仍**2316 vs2316，差0**。
- 相同已存轨迹按两协议截断，完整任务成功都是0/16；这不是去gate后的动态重跑。原物理成本仍4800，不把56当实际省下的训练步。

| sequence / tick | 首点 | 轴/下界余量 | 距物体中心 / 最近目标surface | world xyz |
|---|---|---|---|---|
| s2_cup_pass_1 / 253 (8.4333s) | right/wrist | y / -0.00629m | 2.26087 / 2.213771m | [1.287252, -2.641349, 0.024682] |
| s1_fryingpan_cook_2 / 291 (9.7s) | right/wrist | x / -0.004243m | 2.200588 / 2.098071m | [1.634111, -2.122313, 0.022807] |

两首点都是right wrist（gate点512）；物手已分离约2.2m，不能称这两例是健康抓取的误杀。两例发生在各自prefix104/141之后。原27792条首点/手slot/episode时长不可恢复。
admission-only在后续仍会遭遇模型表示域外输入；安全fallback、mask和行为改变未定义，不能把离线“继续计分”当可执行训练方案。本次不更改grid或gate。

## 缺陷 → 影响范围 → 卡片 → 修订

| 缺陷 | sample / episode / budget范围 | 具体卡 | 撤销 / 降级 / 加注 |
|---|---|---|---|
| UNKNOWN admitted without success branch | 64/501 tasks,9314/56449 initialization draws; episodes=[6212, 9314]; See unknown associated-transition bound; not a direct waste estimate | `src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-inspire-measured-official-ppo.md` | 撤销：All tasks had a valid full-task success opportunity; UNKNOWN explains all0; draw fraction equals ended-episode fraction；降级：0 successes retained only as descriptive training observation with invalid/uncertified opportunity mix；加注：Mode counts, ended UNKNOWN bounds and absent ended-task telemetry |
| PTv3 voxel gate called workspace and immediately terminates | Per-control historical bad points absent; episodes=[27869, 27869]; See voxel associated-transition bound;52.2410% is episode fraction only | `src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-inspire-measured-official-ppo.md` | 撤销：Mechanical workspace failures; all gate exits are false kills; v6 PPO failed；降级：No unique method/generator attribution without qualified executable GT and protocol controls；加注：1570=52.2661%;1575=52.2410%; retain raw counters and missing first-point data |
| Different P0 evaluation gate/censoring scope | Existing separate P0 flow16 fixed rollouts,4768 H4 states;2 environments have voxel exits; episodes=N/A: original P0 did not terminate/reset episodes at these flags; Same saved trace:4800 vs4744 retained full controls;2316 vs2316 primary controls; actual physical ledger remains4800 | `src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261022-wam-v6-executor-qualification.md` | 撤销：Use these two P0 points as the historical27792-point distribution; assume admission-only is safe model inference；降级：Observational protocol comparison only, no gate-removed dynamic result；加注：Both violations beyond reference prefixes; same-trace full task0/16 under both censorings |
| Unfinished P1 and different selected object distribution | A/B partial seed191, C not started; episodes=None; Historical501-task weights and bounds are not transferred to P1 | `src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261022-wam-v6-samples-efficiency-3arm.md` | 撤销：Three-arm no-benefit or C-failure conclusion；降级：HOLD, incomplete method comparison；加注：Keep stopped checkpoints/actual cost; no new runs authorized |

此表是证据范围建议；不覆盖旧数字、不将修复倒灌旧run，不自动授权新实验。P0及P1各自分布/运行身份不能套用r3的UNKNOWN频率。

## 可核路径与命令

- 输入：`/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1/task-reachability.json`、`/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1/voxel-harm-diagnostics.json`及其绑定的实际rank/native manifests。
- 命令：`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/audit_harm_budget_cpu.py`。
- 机器产物：`/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1/harm-budget.json`；逐首点诊断另见`voxel-harm-diagnostics.json`。

冻结证据：
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py:163` — `task.reset(ids)`
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py:233` — `audit['indexed_resets']+=len(selected)`
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py:253` — `audit['motion_visits']`
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/measured_official_env.py:131,163` — `self.rpc('reset'`
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/measured_official_env.py:160` — `self.completed_episodes+=1`
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/measured_official_env.py:98` — `len(self.histories[i])==4`
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/measured_official_env.py:146` — `done[i]=decision.terminated or self.workspace[i]`
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/task_scheduler.py:16` — `episode_timeout_s:float=60.`
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/task_scheduler.py:14` — `acquire_timeout_s:float=15.`
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/goal_scheduler.py:79` — `stall_timeout_s: float = 8.`
- `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/wam-rl-policy-v3/inspire-measured-official-20261020-r3/rank-0/runtime-official/dexgrasp/algorithms/rl/ppo/ppo.py:128,135,151,175` — `current_obs = self.vec_env.reset()`

## 其他已确认缺陷的统一影响表

完整六项表、修复落地与测试见[修复报告](negative-result-fixes-20261022.md)。补充：

| 缺陷 | 只用已有产物的影响量 | 卡片与修订 |
|---|---|---|
| v4/v6身份混淆 | 53347结束回合、16515072更新交互均属v4；不是全部无用预算 | P-20261020-inspire-measured-official-ppo：撤销v6否决，保留v4观察；P1仍HOLD |
| return/phase缺日志 | 4032条三rank日志全部缺字段，53347结束回合缺逐例长度/sequence | 同PPO卡：撤销PPO没reward；精确预算待定，不从episodes比例推算 |
| scale fallback | 当前保留历史cache的50物体scale1请求均有匹配，0/50缺失；其他请求不可恢复 | 同PPO卡/未来load_geometry：潜在防线修复，不能指认为历史零成功原因 |
| SIGINT跳过final-save | ≥380更新/109440样本曝光尾段未保存 | P-20261022-wam-v6-flowmatching：11000为最新保存，11380为最后日志，保留区别 |
