---
schema: ref2dex.probe.v2
probe_id: P-20261019-native-ppo-comparison
experiment_id: P-20261019-native-ppo-comparison
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 7
seed_pool: probe
seeds: [246]
decision_changed_if_positive: expand matched-training controls if current route beats historical official policy
decision_changed_if_negative: investigate task coverage and executor before claiming WAM benefit
status: COMPLETED
run_id: native-ppo-comparison-20261019-r2
---

Decision：在当前相同16物体×4环境、seed246、确定性200步首回合协议下，
比较历史official initialized model_10000.pt与当前自训练BC→PPO5500+WAM80000。
GPU6/7各一臂，graspenv，每进程900秒，主训练不改。
native分支严格加载官方300D actor/critic，直接接原生300D观测，不运行WAM推理，
不添加tau、不使用BC actor、不使用WAM workspace失败终止。共用成功/保持指标。
当前启动器为代码复用仍加载BC/WAM对象，但native策略和观测路径均不使用它们。
评估初始状态逐字段核对；训练初始化、面板和交互预算仍不同，
此比较是历史系统表现重测，不是WAM净收益的matched-training消融。
产物：outputs/wam-rl-policy-v2/native-ppo-comparison-20261019-r1/。

## 结果与对齐审计

r1发现重置初态不匹配，不用于配对结果。r2代码bd0817ae：仅评估reset随机数
改为独立CPU RandomState(seed246)生成同均匀分布，再送GPU；避免CUDA/JIT
初始化历史对reset随机流的影响。预热后重设seed、清除goal-reset标志，
正式reset后把桌面root恢复saved初态并清零速度（仅初始条件，后续物理不改）。
两臂采用同协议，初始原生300D、手点、object pose、goal、完整root逐位一致。
当前565D附加WAM轨迹条件，官方300D路径完全不运行WAM推理；新测试覆盖该隔离。
37项测试通过，严格state_dict加载成功。

|模型|成功|抬高10cm且目标5cm内保持60步|平均回报|
|---|---|---|---|
|历史官方初始化PPO10000|45/64 (70.3%)|33/64 (51.6%)|136.07|
|自训练BC→PPO5500+WAM80000|47/64 (73.4%)|38/64 (59.4%)|140.25|

当前路线多2例成功、5例保持，是小幅正向Probe，不能宣称显著优于官方或隔离WAM净贡献。
两臂optimizer_steps=0、workspace_failures=0，逐回合记录在r2对应evaluation.json。
官方微调32物体与本次16物体无重合；当前路线96训练物体包含本次16物体。
这不代表官方发布模型预训练未见这些物体，但说明两臂微调训练覆盖不匹配。
初始化（官方发布权重 vs 自训练BC）、交互预算也不同，因此它是历史系统同面板
评估而不是matched-training消融。保留继续路线的资格，不据此声称WAM有因果增益。

运行耗时原生约7.19秒、当前约49.08秒（64环境首回合），不含启动，
证明接入完整WAM有明显推理开销，不能把成功指标的小幅改善说成计算效率优势。
主六卡训练未动，GPU6/7评估进程已结束。证据r2/comparison.json与initial-state-audit.json。

## 官方发布原始模型补测

用户要求直接重测state_based_model.pt，不再把我们追加10000轮的模型称为官方原始模型。
run_id=published-ppo-evaluation-20261019-r1，GPU6、64环境、相同16物体/seed246/
cpu-reset-rng-and-canonical-table.v1协议，900秒上限，graspenv。
权重来自本地已有outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1/
plus/dexgrasp/example_model/state_based_model.pt；native300路径无WAM推理和PPO更新。
报告iteration=null，因为官方发布权重的训练迭代数未知，不标记为本地10000轮。
与r2历史微调/current WAM的初始状态和指标比较，其他已述训练覆盖限制仍适用。

### 原始发布模型重测结果

r1原始模型30/64成功、22/64保持，但与此前历史比较run的初始状态不匹配，
不作跨run严格配对。r2同步重跑两臂（代码8a93bb34），初始native300、手点、
object pose、goal、完整root状态逐位一致；不以同seed代替实际初态核对。

|模型|官方成功|抬高并保持60步|平均回报|
|---|---|---|---|
|官方发布state_based_model.pt|30/64 (46.875%)|20/64 (31.25%)|89.57|
|自训练BC→PPO5500+最终WAM80000|47/64 (73.4375%)|38/64 (59.375%)|129.08|

增加17个成功、18个保持；当前路线有1个workspace失败，按失败回合计入64分母，
未剔除；两臂都无优化器更新。此结果支持当前完整系统在该TRAIN面板上优于
未经本地微调的官方发布模型，不隔离WAM净增益，也不代表全对象泛化。
官方发布模型内部训练迭代数未知，报告null；明确区别此前本地追加10000轮的模型。
GPU6/7已释放，主训练不变；证据published-ppo-evaluation-20261019-r2/
comparison.json、initial-state-audit.json及两臂evaluation.json。

## 发布权重来源与复现差距审计（2026-10-19）

Decision：官方结果表与本地发布模型成功率存在大差距，先排除权重、actor适配及最近评估重置修改，决定是否必须修正baseline后再比较。
最便宜的动态诊断是使用已有原生official_unidex_ppo.py入口，不使用新增CPU reset RNG/预热重置/桌面规范化；同16物体64环境、seed246、官方发布权重、GPU6、graspenv、600秒/2GiB上限，不训练。
run_id: published-legacy-reset-audit-20261019-r1；git_commit: 218d6362。沿用已有环境资产，不能作为完全上游复现；初态不配对，仅用于排查整组近期重置改动。

审计结果：
- 官方README明确将state_based_model标为trained policy；论文状态策略通过GeoCurriculum和GiGSL（specialist PPO微调、DAgger蒸馏）得到。发布权重本身没有逐轮训练履历，不能把“未经过我们追加微调”说成“官方未训练/未微调”。
- 官方train结果表有3200个object-scale条目；当前16个code+scale全部在官方TRAIN中，其公开成功率算术平均92.5375%。2145为该split去重object code数，不能混同object-scale任务数。历史32物体追加微调集与当前16物体不相交，不意味着官方预训练未见过当前物体。
- 从固定上游commit下载原始checkpoint，17220395 bytes，与本地文件逐字节相同。64个实际初始观测经过原始上游actor和本地适配actor，动作max abs diff=0。
- 旧原生评估入口完成：29/64成功（45.3125%），15/64保持，33/64抬高；仍远低于官方表。它不采用近期CPU reset RNG/预热重置/桌面规范化，说明这组改动不是低分的唯一必要原因。不同初态，不能将29与30的差解释为重置改动的精确效应。
- 该旧入口manifest中的BC_loaded只是bool(model)历史字段，iterations=10000只是未使用默认值；本次mode=evaluate，未训练、未加载我们BC。草率读取这些字段会误报实验身份。

结论修正：保留此前本地配对结果，但撤回“已经复现官方基准并优于官方”的解读。当前存在很大的官方发布模型复现差距。相同初态证明两臂本地比较可比，不证明仿真/资产/协议与官方一致。针对本地分布追加PPO可能补偿差距，尚无因果证据，不能解释为WAM已胜过官方方案。权重文件和actor前向已排除；共享仿真/资产/评估协议尚未隔离。后续优先用单物体原始上游--test与官方资产/物理配置对齐，之后再做同初始化、同训练覆盖/预算的有无tau消融。主训练未更改。

审计产物：outputs/wam-rl-policy-v2/published-provenance-audit-20261019-r1/{panel-audit,checkpoint-identity,actor-equivalence}.json；动态诊断：outputs/wam-rl-policy/published-legacy-reset-audit-20261019-r1/。环境graspenv，Python3.8、Torch2.4.1+cu121，GPU6。

来源：[官方README](https://github.com/PKU-EPIC/UniDexGrasp2)、[论文](https://arxiv.org/html/2304.00464)、[官方逐object-scale结果](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/results/state_based/train_set_results.yaml)。公开结果是不同采样协议的参考值，不与本地64场景视为配对实验。

## 直接官方--test路径（2026-10-19）

Decision：区分低分是否仍存在于原始任务+原始PPO.run(test)中，决定优先修复wrapper还是下沉资产/运行时。最小判别为官方表99.6%的jar-724fb.../0.06单物体、256环境、seed0、200步，GPU6/graspenv/600秒/2GiB。run_id: upstream-state-test-20261019-r1，代码2f101a29。下载固定上游任务/基类，使用上游train.py --test、配置解析、seed、原生reset和测试循环；只去掉未使用视觉/多智能体导入及无效agent_index读取。不采用aggregate扩容或我们任何eval reset修改。资产只软链接现有官方来源文件。该probe仍不证明资产及本地IsaacGym等于论文运行时。

该直接上游测试完成：原始train.py --test → parse_task → process_ppo.test → PPO.run(is_testing)，原始任务/基类/VecTask reset，单jar 256环境成功124/256（48.4375%），官方对应表99.6%。原始循环运行200步，在第199步读取sticky successes。optimizer_steps=0。因最近wrapper、aggregate补丁、CPU RNG及桌面reset改动都不在此路径中，该复现差距不能只归因于这些适配。共享资产、运行时和上游发布权重/结果对应关系仍待隔离；非同初态配对实验。GPU6采样2773MiB/52%，已释放。启动曾因旧IsaacGym中的np.float在当前NumPy不可用而失败，运行副本补充np.float=float兼容别名后成功，不修改系统或外部环境。

官方确有环境与评估说明：Python3.8，论文IsaacGym Preview3/4并测试过4/4；cfg/shadow_hand_grasp.yaml给出200步、60Hz、substeps2、位置迭代8等；脚本加--test进入PPO.run测试分支。它不是独立eval.py，也不是一条自动遍历所有split并复现表格的完备评估流水线。README未锁定完整Torch/NumPy/驱动环境。此前我们复用了算法/任务主体，但用自己的bootstrap/多物体资产组织/评估包装，不能简称完全未改的官方端到端运行。此次直接上游诊断来源和兼容改动均列于run manifest。两臂初态匹配只支持本地比较公平，不能代替官方复现合格检查。

## Issue线索与运行时差分诊断（2026-10-19）

Decision：在单jar原始--test仍48.44%的反馈环中检验显式物理完成同步是否改变低分，以决定优先修复同步还是继续查资产/运行时。baseline/fetch-results两臂各256环境、seed0、GPU6/7、graspenv、原始PPO.run、600秒/每臂2GiB上限，不训练。代码b3f8930e，run_id upstream-runtime-{baseline,fetch-results}-20261019-r1。两臂均增加同样的初始/终止状态保存；只有fetch-results臂在gym.simulate之后调用阻塞fetch_results。先核对初态，比较动作前向和仿真轨迹。保存状态也会同步GPU，若baseline本身改变，需要识别观测效应而不宣称fetch单因素因果。静态资产来源审计同时核对官方手XML/STL、64D特征、全量官方原始prior及本地IsaacGym安装包。

同步差分两臂均124/256，初始观测/root/dof/goal/theta及终止root/观测/success逐位相同，因此显式fetch_results不是此jar低分原因。下一最小Decision：复用本机已有Torch2.2.2+CUDA12.1，仅在graspenv原解释器前置torch/torchgen只读symlink，其他依赖沿用，单jar256/seed0/GPU6/600秒/2GiB，run_id upstream-runtime-torch222-20261019-r1。单独gymtorch扩展目录，不改系统环境或主训练。核对实际torch版本和初态，若成功率/物理仍同级则排除本地2.4.1→2.2.2差异，但不能扩展到1.x或IsaacGym3。

运行时诊断完成：baseline、显式fetch_results、Torch2.2.2均124/256（48.4375%）；两组相对于baseline，初始观测/root/dof/goal/theta和终止root/观测/success逐位一致。Torch2.2.2实际从本地只读shim加载，与当前2.4.1+cu121比较；不能泛化到Torch1.x。无任何训练更新，GPU6/7已释放。

资产来源检查通过：13个实际手部XML/STL逐字节等于固定上游版本；面板16物体的64D特征逐字节等于官方zip，prior全部字段等于官方完整npy转float32后的值；32个实际URDF/引用mesh文件逐字节等于官方下载tar.xz。该检查排除本地提取/拼装损坏，不能证明作者训练时用的版本与目前发布数据完全一致。IsaacGym关键Python/核心库对照本机Preview4原始安装包，无修改证据；并非与Preview3等价。

Issue核查：
- [++ #10](https://github.com/PKU-EPIC/UniDexGrasp2/issues/10)：单物体训练约60%，两位用户附和，无已定位原因或修复；不是同一发布权重评估的直接复现证据。
- [++ #8](https://github.com/PKU-EPIC/UniDexGrasp2/issues/8)：test/model_dir未生效，社区提示续行符问题；我们已明确进入原始test分支，不适用。
- [++ #4](https://github.com/PKU-EPIC/UniDexGrasp2/issues/4)：Graphics nullptr/headless，作者建议去掉headless解决显示；不能据此解释state成功率。
- [++ #7](https://github.com/PKU-EPIC/UniDexGrasp2/issues/7)：作者说明手初态固定，实际仅物体点云即可；不是低分修复。
- [原版 #32](https://github.com/PKU-EPIC/UniDexGrasp/issues/32)：发布ckpt在test_seen_category仅35.6%，无解决回复；属于UniDexGrasp而非++，仅作旁证。
- 检查了相关关闭issue，未发现可以直接解释本次低分的补丁。GitHub API全量列表因匿名rate limit 403不可用，采用公开issue页核查；不声称完整审阅所有未显示讨论。

剩余可判别方向：同一发布权重在论文IsaacGym Preview3上交叉评估；核对checkpoint与逐物体表是否确为同一训练产物/配置。后续[UniGraspTransformer官方仓库](https://github.com/microsoft/UniGraspTransformer)分别组织IsaacGym3/4模型，说明版本兼容需实证，但不能据此断言本次差距来自版本。NVIDIA[公开回复](https://forums.developer.nvidia.com/t/isaac-gym-preview-3-for-linux/345516/5)称Preview3已不再提供官方下载；本机已确认的安装包为Preview4。当前仍未找到根因，不盲改PPO超参、reward或WAM。来源/资产/运行时报告保存在published-provenance-audit-20261019-r1/，调试脚本保存debug-tools/。

所有seed0调用为复现上游默认启动的确定性debug诊断，不是独立Probe统计样本或validation；原系统比较seed246不变。
