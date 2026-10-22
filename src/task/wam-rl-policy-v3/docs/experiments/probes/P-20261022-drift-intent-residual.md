---
schema: ref2dex.probe.v2
probe_id: P-20261022-drift-intent-residual
experiment_id: P-20261022-drift-intent-residual
date: 2026-10-22
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-drift-intent-residual
probe_index_in_family: 1
seed_pool: probe
seeds: [292, 293, 294, 295]
decision_changed_if_positive: retain recovery-intent BC and bounded residual PPO as candidate; require larger frozen paired efficacy probe before scaling
decision_changed_if_negative: keep frozen R controller and diagnose recovery labels or contact-stage reward before more training
status: COMPLETED
run_id: drift-intent-residual-20261022-r1
---

# 漂移恢复 BC 与有界意图残差 PPO

## 结果解读表

| 结果 | 结论 | 下一步 | 来源 |
|---|---|---|---|
| BC训练误差下降但DEV/闭环不改善 | 恢复标签过拟合，未建立泛化 | 不晋升BC；仅完成已注册PPO技术pilot，不扩预算，DAgger后置 | Hermes 的BC目标重定位 |
| DEV与配对物理均改善 | 有限初态的候选方向；非样本效率/泛化证据 | 提议独立冻结评估，禁止自动三臂 | Hermes 有条件残差PPO |
| reward/PD/残差合同失败 | 工程失败，不能否定方法 | 保存partial，修合同后另run | 外部审计复现要求 |
| 有限PPO零成功或比BC差 | 仅否决当前小预算配方，不能否定v6/RL | 不追加本轮训练 | 尚未与两方独立复跑一致 |

Result: BC400 updates, train MSE.422871→.0910735 but DEV.386376→.460369; no recovery-generalization evidence. Deterministic before/after both0/2; stochastic train0/2,826controls,2optimizer updates; raw6/6 and PPO reward replay2/2 PASS.
Decision: BC overfit and no physical efficacy demonstrated; stop this fixed pilot without expanding training, keep three-arm RL HOLD and DAgger deferred.

分类 Decision。区分“当前R无法把漂移状态带回抓取意图”与“小残差/当前几何teacher也无法恢复”。
最便宜方式：从v10真实已保存状态取漂移样本，不用合成力/接触；监督几何可准入候选减R意图，
每序列≤40样本。DEV序列由词典序每第三条预先划分；标签是解析teacher，不是新增真实专家。
BC 400updates或300s先到，batch≤128，seed292，graspenv GPU4，冻结R/WAM，单checkpoint。
准备数据不读未来真实动作作为恢复标签，不在线DAgger；不把监督MSE下降当作物理收益。

残差仅作用于R给出的物体坐标关键位姿和指预形：每坐标平移≤10mm、旋转向量≤40mrad、
指角≤30mrad，范数上界分别17.32mm/69.28mrad。不是从零学12维机器人动作。
PPO seed293，γ=.96、GAEλ=.95、clip=.2、lr1e-4、最多4epochs/回合、KL>.03停止后续epoch；
原始Gaussian latent计算logprob，tanh有界变换在部署和训练同一入口。原安全gate保持。

启动条件（Hermes）：①v10 raw A2 τ=0/24帧无泄漏/R保留；②阶段化reward就位；
③预算与本表齐备。Reward复用S2c高水位增量/单次事件，HOLD不付release，记录
shaping/event/sparse/time/effort全分项；末态成功取**新几何合同**而非原source调度终止。
物理终止/成功bootstrap0；20s未成功时间截断使用末态value，不能跨reset串GAE。

本轮最多2个训练回合/1200controls/8optimizer updates；训练前后同2初态/同seed各评估1次，
总共≤3600controls、0 v6/R updates、单卡≤1h、产物≤1GiB。效果评估在deterministic均值上，
训练采样的两回合不并入评估分母。小pilot只检查可运行性与奖励/梯度链，不作统计结论。
GPU只在4/5/6空闲后使用；不重训v6、不启动正式三臂、不做完整RELEASE。

实现：`src/wam_rl_policy_v3/intent_residual.py`、`tools/run/train_drift_intent_bc.py`、
`tools/run/run_grasp_geometry_worker.py --residual-checkpoint ... [--ppo]`。
准备数据产物 `outputs/wam-rl-policy-v3/drift-intent-bc-data-20261022-r1/`；运行产物
`outputs/wam-rl-policy-v3/drift-intent-residual-20261022-r1/`。确切命令与实测结果运行后回填。

具名边界：Hermes允许上述有条件启动，外部审计要求区分接口与任务；目前两方尚未独立复跑。
195状态来自9条已用于v10开发的序列，DEV隔离只能约束离线拟合，不能称全新对象泛化。

BC实测：seed292、400updates、1.181861s、GPU4 345–347MiB、Torch峰值20,727,808bytes；
采样GPU利用率1/12/14%，195状态使计算很短。训练/DEV均为归一化意图残差MSE；
训练降78.46%，DEV升19.15%，保留固定400步终态，不按DEV事后挑checkpoint。
BC权重只供本次有界技术pilot，不晋升或扩训。环境与所有展开命令在run内bc/manifest.json及父manifest.json。

## 完整pilot结果、审计与决定

| 阶段（同两初态，native294/295） | 严格成功 | 实际controls | 关节中止 | 更新 |
|---|---:|---:|---:|---:|
| BC确定性均值（before） |0/2|643|1|0|
| PPO随机训练（独立列出） |0/2|826|1|2|
| 更新后确定性均值（after） |0/2|1200|0|0|

总2669/3600 controls，BC400updates另计；R/v6零更新。训练前/训练/训练后三组两初态九字段
一致，raw6/6 PASS；两条PPO的shaping/event/sparse/time/effort逐tick重放2/2一致。两次训练
每次只更新1epoch便触发后续epoch的KL停止，不把名义上限8更新当实际量。

**负结果边界**：三个阶段所有回合均无approach/contact/lift/hold事件阳性。训练event合计0；
主要回报是时间/effort惩罚及一次真实joint abort，不能把0/2当充分的RL方法否决。
BC DEV变差，没有恢复泛化证据；PPO未带来目标成功。原注册budget已结束，不追加训练或调参。

独立policy审计`ppo-policy-audit.json`：旧策略logprob最大重建误差1.90735e-5，GAE均值误差
3.73e-9；首回合物理中止bootstrap0，第二回合600步未成功截断bootstrap=.1389691。
最大意图修正每坐标平移9.95452mm、旋转39.7351mrad、指角9.53515mrad，均在注册上界内。
该界限制R意图，不保证低层状态/接触过程变化同样小。

更新后精确Gaussian KL **.611887 / .165042**，超过target .03；原规则只能阻止后续epoch，
不能控制第一次更新后的KL。软件重放PASS不等于优化器已满足硬trust-region资格。
未来若再跑，先控制/记录首步KL并改善真实漂移覆盖及恢复标签；当前不修改参数、不补跑。

审计先尝试GPU5，被已占用检查拒绝；GPU4/5/6分别由本pilot和并发n40占用，故额外小规模
只读policy重放显式用CPU，未共享GPU、未使用0–3/7。BC和全部在线模型/仿真仍用授权GPU。
同环境为graspenv Python3.8.20/Torch2.4.1+cu121；native另进程dexplore_v120_train。

## Hermes可直接复跑的确切命令

以下读取**已冻结195状态dataset**与两初态registration；从现已补齐的v10重新prepare可能选到
不同trial，因此不能将重新生成的数据默认为本次195状态。raw来源逐条在dataset.json.sources。
输出必须fresh；确认GPU4空闲。PPO每条600上限，两条1200，三阶段总3600上限。

```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
TOOLS=src/task/wam-rl-policy-v3/tools/run
BASE=outputs/wam-rl-policy-v3/drift-intent-residual-20261022-r1
REPRO=outputs/wam-rl-policy-v3/drift-intent-residual-hermes-repeat1
$PY $TOOLS/train_drift_intent_bc.py \
 --dataset outputs/wam-rl-policy-v3/drift-intent-bc-data-20261022-r1/dataset.npz \
 --output "$REPRO/bc" --gpu 4
$PY -u $TOOLS/run_grasp_geometry_worker.py --registration "$BASE/registration.json" \
 --worker 0 --gpu 4 --output "$REPRO/before" --residual-checkpoint "$REPRO/bc/bc.pt"
$PY -u $TOOLS/run_grasp_geometry_worker.py --registration "$BASE/registration.json" \
 --worker 0 --gpu 4 --output "$REPRO/ppo" --residual-checkpoint "$REPRO/bc/bc.pt" --ppo
$PY -u $TOOLS/run_grasp_geometry_worker.py --registration "$BASE/registration.json" \
 --worker 0 --gpu 4 --output "$REPRO/after" --residual-checkpoint "$REPRO/ppo/residual-p001-A1.pt"
$PY src/task/wam-rl-policy-v3/tools/audit/audit_intent_ppo.py \
 --run "$REPRO" --gpu 4 --output "$REPRO/ppo-policy-audit.json"
# 三阶段各自raw复核，期望n2一致；ppo还须reward_match=true。
$PY $TOOLS/rescore_large_grasp.py --worker "$REPRO/ppo" \
 --output "$REPRO/ppo-rescore.json" --verify-mutant
```

复跑期望为合同/分母/预算守住；物理成功和MSE值按真实结果，不预设复现改善。
本次全部展开argv与日志在父manifest.json.commands，权威总结果为summary-final.json。
