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
status: RUNNING
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

Result: BC400 updates, train MSE.422871→.0910735 but DEV.386376→.460369; no recovery-generalization evidence. Physical before/after pilot pending.
Decision: continue BC on bounded recovery intent, then only a registered residual PPO feasibility pilot; three-arm RL HOLD.

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
