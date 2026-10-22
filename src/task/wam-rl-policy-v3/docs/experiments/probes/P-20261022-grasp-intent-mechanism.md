---
schema: ref2dex.probe.v2
probe_id: P-20261022-grasp-intent-mechanism
experiment_id: P-20261022-grasp-intent-mechanism
date: 2026-10-22
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-grasp-intent-mechanism
probe_index_in_family: 1
seed_pool: probe
seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179]
decision_changed_if_positive: retain effective component or prefer geometry if sufficient; test object coverage before RL
decision_changed_if_negative: retain combined measured interface and diagnose approach; no isolated-component or generalization claim
status: COMPLETED
run_id: grasp-v13-mechanism-20261022-r1
---

# 成分消融与同接口 A3/A4

## 结果解读表

| 结果 | 结论 | 决定 |
|---|---|---|
| 腕姿单项优于模板 | 抓哪儿有效 | 扩展抓姿覆盖 |
| 腕姿/手指单项低、联合有效 | 抓哪儿+怎么合手 | 保留联合消费 |
| 时机单项有效 | 新增预测闭合时机可用 | 与实测相位比较，不能倒推v10使用了时机 |
| A3高位或A4未闭合 | 几何足够或上界未成立 | 重估WAM或修GT，不推进RL |

Hermes 指定分解、A3/A4同接口和逐例中止归因；外部审计要求跨物体与可复核几何。仍不一致：原始A1只用实测阶段，模板全替换不是原A2，A4不是最优策略上界。

Decision：复用原A1/A2，seen新增WRIST/FINGER/TIME/T0各40，之后seen/unseen新增A3/A4各40；最多320新物理回合、192000controls、0updates，GPU空闲4/5/6。排队≤4h、每阶段worker≤4h；无技术故障换seed补样、无调参、三臂RL HOLD、不重训v6。先前八物体已经用于诊断，本轮不是独立holdout。

全部事前固定表、测量/模板/A4定义、原始17例审计与命令见[主报告](../../grasp-v13-mechanism-20261022.md)。输出在 `outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/`，实测提交与环境记录到supervisor/worker manifest。

机制物理160回合完成，WRIST/FINGER/TIME/T0各0/40；原始重评分160/160、6臂初态40对、干预重放160/160与R权重复算160/160均PASS。WRIST进入close18、S2c接触15、物理lift/hold0；TIME40回合均无预测接触帧。

GT资格发现接触cap错误截断R8的后续合手：真实源第49帧仍闭合，第50帧才耦合抓取，旧provider最多看到38。回归测试先红后绿；改为原生11点匹配至首次实测抓取，9/9源资格PASS。正式对照采用新run_id `grasp-v13-comparators-20261023-r2`，同一固定初态与seed，160回合；旧9完成+1中断全部保留为工程记录、3668–3669步，不混入正式统计。包括工程的物理上限195289步、0updates，RL HOLD。详见主报告GT资格修订。

最终GT恒等性门另修复逐帧锚定误抹掉物体运动：在源状态应还原源真实未来（旧误差170.924mm）。改为当前匹配源帧的一次锚定，9源恒等性/SE3等变性PASS。最终对照run_id为 `grasp-v13-comparators-20261023-r3`；复用r1/r2首次完成的15个未变A3，失败也保留，其余145回合新执行。旧工程全部留档、不混作完整GT。整轮物理上限192996（含工程），0updates，三臂RL HOLD。

## 最终结算（r3完成，优先于上文运行时状态）

Result: 320 formal episodes; seen A1 35/40 versus each component/A3/A4 0/40; GT never close; 17 aborts fully classified
Decision: retain conditional wrist+finger interface; qualify source-to-R/executor; no generalization or WAM-unique-information claim; three-arm RL HOLD

四个机制臂WRIST/FINGER/TIME/T0各0/40；TIME40/40无预测接触帧。A3/A4在seen与原unseen各0/40、160例均未close；构造资格PASS不等于R/执行器GT上界。原unseen序列均在R TRAIN，不能当训练未见。17例=9投影内跨门槛+7完全桌下重入+1边缘重入，阶段12 approach/5 close，无“其他”归因。

新增正式147811controls，含工程实际153675–153677，0updates；原始评分/独立干预重放/R重算各320/320 PASS、初态40+80组PASS、9源GT恒等/等变PASS。R重算最大1.430512e−6；最终回归316 passed、4 warnings。最终摘要/核对分别为r3的 `summary-final.json` / `delivery-audit.json`；确切一条复跑命令、来源/分歧、全部逐例量见主报告开头。三臂RL HOLD、不重训v6。
