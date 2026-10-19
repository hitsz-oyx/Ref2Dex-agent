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

### 后期评估结果

运行代码fce903a9；四条件各256首回合、0优化更新、workspace failure均0。
5500/7000/8483的完整initial-state.npz五字段及物体身份逐位相同。
8000初态不同（观测最大差5.2262、hand0.0571m、root混合分量46.6941），仅保留诊断，
**不用于8000→8483预注册端点判定，也不据此宣布8000退化**。

|累计轮数|原生曾成功|曾保持60步|最终稳定保持|终态位置成功|失守过|平均回报|
|---|---|---|---|---|---|---|
|5500|202/256 (78.9%)|153 (59.8%)|146 (57.0%)|164 (64.1%)|7|140.82|
|7000|208/256 (81.2%)|171 (66.8%)|161 (62.9%)|184 (71.9%)|10|151.38|
|8483|219/256 (85.5%)|167 (65.2%)|163 (63.7%)|194 (75.8%)|4|155.11|

8000诊断：207曾成功、156曾保持、148最终稳定、回报136.60，初态不匹配。
5500→8483成功+6.64pp（37赢20输），最终稳定+6.64pp（59赢42输）；
7000→8483成功+4.30pp（29赢18输），最终稳定仅+0.78pp（42赢40输）。
以16个物体为cluster的描述性配对bootstrap，后者最终稳定差95%区间约[-5.9,+8.2]pp；
面板只有单seed TRAIN，区间不代表泛化保证。保持收益趋缓，仍有场景交换与成功率改善，
**不能确认收敛**；继续原10000预算，到终点复核，不自动延长训练。
单条件仿真评估150–152s；GPU6/7活跃期约98–100%、约6.8GiB（含原共享任务）。
未改主训练、未新增训练checkpoint。产物：late-checkpoint-eval-20261019-r1下
comparison.json、initial-state-audit.json、checkpoint-trend.png及各step结果。

## 10000终点、全部96物体的平台期复核

用户追加评估六卡混合训练是否到平台。r4已正常完成10000轮。Decision：若末段仍有
≥5pp改善，不把当前预算终点当收敛；若8500→10000与9500→10000的原生曾成功和
最终稳定保持均变化≤3pp，作为本TRAIN面板平台迹象，暂不扩大训练预算。
同时检查以物体为cluster的配对区间是否仍允许≥5pp收益，不宣称全局收敛或WAM净增益。

最低可判别追加：run_id=full-plateau-eval-20261019-r1，7000/8500/9500/10000四checkpoint，
实际六rank全部96 TRAIN物体，每物体8场景=768场景/轮；分为3个32物体×256环境子面板。
固定seed246、确定性首回合200步、0 optimizer更新，保留原WAM12000、SE3/legacy/workspace
任务合同。新建32物体只读资产视图，不修改旧rank资产；checkpoint复制固定后评估。
逐面板核对obs/pose/hand/goal/完整root和env对象身份，未匹配组不得用作严格配对趋势。
该面板与此前16物体×16场景不同，不直接混用此前成功率。

GPU0–3最多4个评估worker，GPU6/7的Inspire长训保持运行，GPU4/5空闲，满足预留两卡；
每worker≤900秒、总≤2400秒、产物≤4GiB、磁盘余量<20GiB停止，环境graspenv/Torch2.4.1+
Isaac Preview4。复用同一原生评估器，launcher只扩展GPU配置与多面板任务命名，不改策略或reward。
产物：outputs/wam-rl-policy-v2/full-plateau-eval-20261019-r1/。

初态预审：panel0仅250/256场景在四轮间完全一致，差异集中6个环境，object actor的root
位置与pose逐行一致，root数组确认每环境hand/object/goal/table四actor的连续布局。
为避免丢弃整个32物体子面板，将匹配粒度细化为单场景：只保留四checkpoint所有5字段及
四actor root、对象身份均完全相同的场景交集，规则不读取策略成败。所有轮次共享同一交集，
并保留768场景原始计数作敏感性描述。报告实际保留数及对象覆盖，低于90%场景或丢失任何
训练物体则不使用完整面板平台判据。没有声称修复底层reset/PhysX初始化差异。
配对bootstrap按object cluster重采样、按保留场景数量加权，避免将同物体环境当独立物体。

### 全96物体终点评估结果

12/12 worker正常完成，总约494秒，每个256首回合，全部0 optimizer更新、0 workspace failure。
四条件共同初态为731/768场景（panel0/1/2分别250/256/225），仍覆盖全部96物体；
排除37场景的依据仅是obs/pose/hand/goal/四actor root与对象身份是否完全一致。
未剔除的四轮共享相同731个场景。以下百分比均以731为分母：

|累计轮数|原生曾成功|最终稳定保持|终态位置成功|平均回报|
|---|---|---|---|---|
|7000|609 (83.31%)|445 (60.88%)|520 (71.14%)|154.65|
|8500|645 (88.24%)|486 (66.48%)|565 (77.29%)|165.45|
|9500|622 (85.09%)|473 (64.71%)|555 (75.92%)|157.56|
|10000|561 (76.74%)|415 (56.77%)|481 (65.80%)|140.97|

8500→10000：原生曾成功−11.49pp（45赢129输），最终稳定−9.71pp（99赢170输）；
物体cluster配对bootstrap95%区间分别[−14.90,−8.15]pp和[−14.50,−4.81]pp。
9500→10000：原生曾成功−8.34pp、最终稳定−7.93pp，区间分别[−11.31,−5.42]pp、
[−12.72,−3.11]pp。退化方向在这一固定TRAIN面板上明确，未满足≤3pp平台判据。

全768场景未筛选的原生曾成功依次642/679/654/591（83.59/88.41/85.16/76.95%），
最终稳定467/510/493/441（60.81/66.41/64.19/57.42%）；趋势一致。
因此不能用少数reset初态不一致解释本次晚期回落，也不能把10000终点称作最佳或已收敛。
8500为本次四个候选中的最佳，应保留作为当前候选；不宣称整个训练过程中全局最优，
不自动延长PPO预算。下一步若继续本路线，先诊断后期优化/策略退化；本次没有定位其原因。
单seed、TRAIN、原WAM12000/legacy协议，不是未见物体泛化，也未隔离WAM的净贡献。

评估launcher代码829726a4/c3a208ed，分析代码f629b6d2；新增2项审计测试通过，
覆盖未观测root差异排除与物体内完全相关样本的cluster区间，未修改活动训练。
[完整比较](../../../../../../outputs/wam-rl-policy-v2/full-plateau-eval-20261019-r1/comparison.json)、
[逐场景初态审计](../../../../../../outputs/wam-rl-policy-v2/full-plateau-eval-20261019-r1/initial-state-audit.json)、
[趋势图](../../../../../../outputs/wam-rl-policy-v2/full-plateau-eval-20261019-r1/checkpoint-trend.png)。

### 后期退化诊断：已发现上游KL度量缺陷，尚未证明完整因果恢复

用户要求六卡Inspire推进同时定位Shadow退化。最低成本先读取现有checkpoint/日志，
在GPU6对同一731条初始输入做无更新推理及实际分布KL核对，0训练更新。
假设依次区分：权重/normalization加载错误、晚期策略更新漂移、自身训练与确定性评估口径偏离。
若仅加载错误则修加载；若真实训练同样退化且优化度量失准，则修优化校准，再另做恢复验证。

shadow-regression-audit-20261019-r1：7000/8500/9500/10000的文件内iteration与文件名一致，
optimizer step分别精确为iteration×20；全部参数有限，四组x_mean/x_scale/y_mean/y_scale逐位相同。
8000→8500随机训练回合曾成功84.91%、曾保持67.40%、平均step reward0.7824，
9500→10000降到80.88%/61.20%/0.7068。因此不是只有确定性评估才出现下降。
相邻checkpoint的固定输入动作确实变化；它们之间的累积KL不是单次PPO更新KL，不能混用。

确定的源码不一致：上游module.py实际scale_tril=diag(exp(log_std)^2)，
真实标准差为exp(2 log_std)；上游ppo.py的adaptive KL公式仍按exp(log_std)标准差计算。
用9500真实腕部标准差0.09247、同方差均值变动0.03的确定探针，Torch真KL=0.052623，
旧proxy=0.004866，desired_KL=.016时旧proxy要求提高LR、真KL要求降低LR。
这证明度量失配，且随某些维度方差变小会低估均值更新KL；是后期更新过松的具体候选机制。
没有据此断言它是唯一原因，也未声称匹配续训后性能已恢复。

r1命令--assert-kl-consistency在真实采样分布对比上exit2，r2同探针用校正公式与Torch KL一致exit0；
新增随机多维分布KL及真实尺度反向LR判定回归通过，完整Task共60项测试通过。
新Inspire采用校准KL并记录优化合同/迁移，保留官方采样和其他算法/超参；旧冻结Shadow产物不热改。
后续证明该改动恢复Shadow表现属于另一个有预算的受控续训，不在本次拿静态审计冒充因果恢复。
产物outputs/wam-rl-policy-v2/shadow-regression-audit-20261019-r{1,2}/report.json。
