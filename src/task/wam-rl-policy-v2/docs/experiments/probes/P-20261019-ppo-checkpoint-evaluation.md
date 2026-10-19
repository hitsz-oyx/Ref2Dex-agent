---
schema: ref2dex.probe.v2
probe_id: P-20261019-ppo-checkpoint-evaluation
experiment_id: P-20261019-ppo-checkpoint-evaluation
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 5
seed_pool: probe
seeds: [246]
decision_changed_if_positive: continue training if fixed-panel performance improves
decision_changed_if_negative: investigate plateau or regression before increasing training budget
status: COMPLETED
run_id: checkpoint-eval-20261019-r3
---

Decision Probe：判断不同checkpoint在固定初始场景上的确定性策略表现是否改善。
GPU6/7并行独立评估1000/3000/5000/5500轮；每个64环境、同rank0的16 TRAIN物体，
同seed246，首次完整native回合（200步上限），random_time=False，动作act_inference，
不调用PPO更新、不生成checkpoint。graspenv，每个进程900秒硬上限，四次总预算30分钟。
初始状态保存并逐字段对比，首次done后不再纳入该环境统计，避免短回合偏置。
官方成功和抬高10cm、目标5cm内保持60步分别报告。不是DEV或holdout评估；
64场景小样本、单seed，不能据此宣称收敛或泛化。主六卡训练不变。
产物：outputs/wam-rl-policy-v2/checkpoint-eval-20261019-r1/。

## 初始状态对齐审计

r1在构造前设seed，r2又在正式reset前设所有RNG，仍有部分初始状态不一致，
因此两轮数据仅作diagnostic，不用于公平checkpoint比较。r3采用已有回放入口的
两次reset协议：首次预热原生仿真buffer，重设Python/NumPy/Torch/CUDA seed，
再次reset后才记录初始状态并执行策略。首对1000/3000的observation、pose、hand、
goal、完整root state最大差异全为0。该差异与首次sim buffer初始化有关，
尚未单独隔离到具体PhysX内部buffer；不把它误报成已证明的RNG bug。
最终有效run_id=checkpoint-eval-20261019-r3，代码4f2bf728。

## 结果

所有4个checkpoint均完成64个首回合，optimizer_steps=0、workspace_failure=0。
严格对齐组1000/3000/5500初始565观测、手物几何、目标及完整root状态全部逐位相同，
环境物体身份相同。结果分别为：

|累计轮数|成功数/64|保持数/64|平均回报|
|---|---|---|---|
|1000|37 (57.8%)|26 (40.6%)|107.41|
|3000|42 (65.6%)|34 (53.1%)|124.65|
|5500|51 (79.7%)|37 (57.8%)|139.42|

5000轮诊断结果46/64成功、38/64保持、回报135.10；虽然565观测/手物几何完全相同，
完整root状态有差异（actor index%4==3桌面，最大混合状态分量差0.01537），
因此不计入严格matched趋势，不用其与5500的保持率差判断退化。

判断：匹配场景上的成功率仍提高，保持指标提升较小，不能宣称已经收敛。
仅16 TRAIN物体×4环境，单seed，未见物体泛化与统计显著性尚未验证；
继续既定10000轮预算，后续用相同协议评估后期checkpoint。
最终产物见r3/comparison.json、initial-state-audit.json、step-*/evaluation.json。
前两轮不匹配结果不用于训练趋势。GPU6/7测试结束释放，主训练保持运行。

## 后期平台期追加 Probe（2026-10-19）

用户授权GPU6/7评估最近轮数，run_id=late-checkpoint-eval-20261019-r1。
Decision：若8000→最新快照的成功/最终稳定保持仍有>=5个百分点改善，继续既定训练；
若两者均变化≤3个百分点，视为本面板平台期迹象，在10000预算终点复核而不自动加预算。
若退化或不确定，记录逐场景胜负和不确定性，不宣称全局收敛。
最低可判别协议：5500/7000/8000/8483四个固定checkpoint，各16 TRAIN物体×16场景=256，
seed246，确定性动作、首回合200步、0更新。重新评估5500，不将历史64场景率混入本面板。
checkpoint先复制到独立inputs，避免活动训练轮转删除最新快照。固定WAM12000与本地几何，
显式legacy reset/se3_pose/原WAM workspace终止；未把新接口修复混入收敛比较。
复用当前官方PPO评估入口，仅增加终态指标记录；历史未绑定checkpoint显式记录迁移说明。
CPU独立reset RNG、预热、canonical table协议保持一致；逐字段保存并核对初态。
新增ever_held60/final_held60/drop_after_hold/final_task_success/final_stable_success和终止原因。

graspenv/PyTorch2.4.1+cu121/IsaacGym Preview4。每worker≤900s、总≤2400s、输出≤4GiB，
低于20GiB磁盘余量停止；GPU6/7已有monica低占用任务沿用用户授权共享，不改其进程。
按long-running-tasks技能监控；不录像、不生成训练cache、不保存新训练checkpoint。
启动器tools/run/evaluate_ppo_checkpoints.py冻结代码并保存所有命令、PID、资源和输出身份。
四条件初态不匹配则不作为严格配对趋势；不更改主训练，单seed TRAIN Probe不能证明泛化。
