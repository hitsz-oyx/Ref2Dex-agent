---
schema: ref2dex.probe.v2
probe_id: P-20261016-official-unidex-ppo
experiment_id: P-20261016-official-unidex-ppo
date: 2026-10-16
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-official-unidex-ppo
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 249]
decision_changed_if_positive: use the qualified official PPO as the common trainer before testing WAM
decision_changed_if_negative: diagnose official engine, asset/bootstrap compatibility, and training budget before attributing failure to WAM
status: UNCLEAR
run_id: official-unidex-{smoke,train32,evaluate}-remote-20261016-r1
---

# 先接通官方UniDexGrasp++状态PPO

Result: 已完成官方源码、fresh 与官方 checkpoint warm-start 两条路径，以及 WAM-v3
直接拼接/adapter Probe；当前整体仍为 `UNCLEAR`，但两种 WAM 接法在 warm-start 上均
`UNPROMISING`。
Decision: 按用户要求先用远端官方PPO训练，底座通过后再接WAM。

Class: Blocker/Decision。用户怀疑自写PPO实现或配方导致失败，要求先跑官方路径。
服务总目标A的训练底座与B的共同对照；本地UniDex任务资格不替代最终DExplore目标。

## Decision Note / 最小判别设计

问题：保留上游状态PPO采集、GAE、分布、优化和奖励，能否从随机初始化学会抓取？
最便宜方法是复用已有32TRAIN物体资产，直接运行固定上游PPO类，不继续修改自写循环。
上游commit `6c9e710b04769bf5d9765e3233820101cb3d718f`。
用户已明确指定远端训练；远端四卡启动前均空闲，先用GPU0，其他卡留待固定评估/后续对照。

第一步工程smoke：256env、32物体、32个8步rollout、seed45、<=600s。
必须验证actor/critic参数改变、有限loss/参数、640次实际Adam step、保存/重新加载。
另用已有官方checkpoint在同TRAIN面板做一次128env确定性正控，只检查环境/加载是否保留
已有抓取能力；不作为fresh训练结果，不用于正式参数初始化。
第二步fresh训练：32TRAIN×32env=1024，seed245、10000个8步rollout（81920000环境步），
训练<=7200s、checkpoint每1000更新，任务总产物<=4GiB，每个run<=2GiB，磁盘保留20GiB。
若smoke显存不足先降工程env数；正式环境数/预算改变须在本卡记录，不静默追结果。

官方`1024,1024,512,512` ELU actor/critic，正交初始化，原300D观测、24D动作，
nsteps8、5epoch×4minibatch、共同Adam3e-4、gamma.96、lambda.95、native reward，
adaptive LR/desiredKL.016，init_noise_std.8。原module实际scale_tril=exp(log_std)^2，
有效初始标准差.64；原KL使用另一尺度约定，首次原样保留，不静默修复。
fresh 分支不加载 BC 或官方 actor 初始化；官方示例脚本默认 `model_dir` 非空，另行记录
的 initialized 分支严格复现该默认 warm-start 配方，用于解释官方 baseline 而不是冒充
从零自训证据。
官方训练random_time=True，只随机第一次episode计时，不是物理中间态RSI。

## 启动适配与证据边界

入口`tools/run/official_unidex_ppo.py`（相对Task）调用原`process_ppo`及
`PPO.run/update`/`RolloutStorage`，逐run复制冻结源码。
状态分支不使用PointNet；仅移除两个无条件vision import，并补未使用的backbone config键。
既有原生环境bootstrap只保留最大异构aggregate容量修复，物理/奖励/动作实现不改；
runtime工作目录与assets软链接位于新run，不能向共享资产目录写入。
progress/log/optimizer保存为被动观测；不改变采样、损失、梯度或参数更新。
依赖只安装到远端本任务独立目录，不能覆盖现有conda环境或他人任务。

固定终点为10000；另记录预定1000中间点作训练进度诊断，不能按DEV挑模型。
确定性评估原始完整199步、random_time=False，报告native sticky success及
held60+lift10cm，保存真实初态。第一轮baseline不构成WAM比较或小数据claim。
至少有8/128个实际保持病例且参数加载/执行通过，才视为足够支持下一项WAM Probe；
否则先区分训练不足与执行缺陷，不关闭PPO/WAM假设。

## 独立审查

AGENTS第14节只读审查未发现自写GAE明确错误：相同输入与gamma下return最大差4.77e-7。
两边均对未裁剪样本计算log-prob、再裁剪执行。关键配方差异包括官方有效探索std.64、
大ELU网、短rollout与纯adaptive LR；自写冷启动std.223、小Tanh网、硬KL回滚，
第25episode三臂只执行55/70/36次actor更新（名义上限260）。这足以动摇此前归因，
但不是已经定位自写实现bug。保持官方原貌建立有信息价值的基线。

## Limitations / future evidence

32物体缩小面板、1024环境，与官方全对象论文结果不是等规模复现。单seed、复用探索面板，
只形成Probe判断；正式WAM收益要在相同官方底座、初始化、交互预算和实际初态下另做对照。
WAM接法尚未启动；不把官方checkpoint评估或微调混成从零学习。

## 已完成结果（2026-10-16）

官方 fresh baseline 已在远端 GPU0 正常完成：1024 env、10000 iterations、81920000
environment steps、200000 Adam steps，`learning-audit=PASS`。训练终点的 native reward/step
约 0.145、critic EV 约 0.905，但 train 固定评估为 `official_success=0/128`、
`held60_with_lift10cm=0`，core-dev 固定评估同样为 `0/128`。因此不能说官方 PPO 在当前
面板上复现了 baseline。

环境和 checkpoint 链路的正对照仍通过：官方发布的 state model 在同 TRAIN 面板
`official_success=65/128`、`held60_with_lift10cm=39/128`、`lifted10cm=71/128`；这表明
零成功不是评估程序完全失效。该正对照不属于 fresh 训练结果。

随后接入冻结 WAM-v2 step80000：观测为 native300 + 三类 WAM summary2304 + relative-goal3，
2607D；仅把官方 ActorCritic 的输入维度从固定 300D 接到 2607D，PPO.run/update、storage、
分布和 KL 约定保持上游。64 env、32 iterations 的 smoke 完成 640 optimizer steps，
`learning-audit=PASS`；固定 128-env 评估为 `official_success=0`、`lifted10cm=0`。
这个结果只证明 WAM 接线可运行，不能评价 WAM 的科学收益。

证据目录：`outputs/wam-rl-policy/official-unidex-remote-evidence-20261016-r1/`；远端原始
产物位于 `/mnt/ugreen_nas/storage/oyx_storage/robot-finetune-portable-20261016-r1/repo/outputs/wam-rl-policy/`。
WAM 运行使用的是当前工作树部署文件，远端 manifest 中继承的 `code_commit=11664741` 只对应
官方 baseline 提交，不能作为 WAM 改动的独立代码身份。

环境记录边界：本卡记录了 simulator environment 数、GPU、资产和运行预算，但没有逐 run
固化 conda 环境、Torch/CUDA 版本、Isaac Gym 与 gym 的 Python 路径、`LD_LIBRARY_PATH`
或 JIT extension cache。因而这些旧 run 的软件环境不能仅凭本卡完全复现；本次 runtime
修正 Probe 才补上了显式软件环境合同，并把环境准备失败与科学结果分开统计。

当前判断：`UNCLEAR`。官方上游 PPO 在当前资产面板上也未从零学会，而已有模型可成功，
因此此前“必然是自写 PPO 数学实现错误”的解释被削弱；仍需最小化核对官方训练所需的
奖励尺度/episode 机制、随机初始化与论文训练配方，再决定是否进入长预算 WAM 对照。
deadline到达时保留已有checkpoint，标记未完成，不把TIME_BUDGET当正常完成。

## 官方初始化配方复核与 WAM-v3（2026-10-17）

复核上游脚本 `dexgrasp/script/run_train_ppo_state.sh` 后确认官方 state PPO 默认传入
`--model_dir=example_model/state_based_model.pt`，即从官方 actor/critic checkpoint
继续 PPO；此前 fresh 0/128 run 不是官方完整 baseline。严格按该入口的 initialized run
完成 1024 env、10000 iterations、81920000 env steps、200000 optimizer steps，audit
通过；固定 TRAIN 为 `103/128 success, 85/128 held60+lift10cm, 105/128 lifted10cm`，
core DEV 为 `98/128, 72/128, 108/128`。这也解释了为什么 fresh run 的负结果不能归因
于 PPO 数学实现。

WAM-v3 使用冻结 `v3-step80000.pt`，仍为 2607D（native300 + WAM summary2304 +
relative-goal3）。官方 checkpoint 的首层 300D 权重保留，新增列置零；这一扩展的固定
正对照保持官方模型 `65/128 success, 39 held60, 71 lifted`，证明输入包装器初始不改行为。
WAM-v3 1000-iteration/64-env probe（20000 optimizer steps）固定 TRAIN 为
`19/128 success, 0 held60, 35 lifted`，同 iteration 的 official initialized
`model_1000.pt` 为 `83/128, 58 held60, 93 lifted`。该差异说明短预算 PPO 更新后 WAM
输入会明显破坏已学策略，但不是最终 WAM 负结论：交互步数仅为官方 full run 的 0.625%。

为匹配 optimizer iteration，WAM-v3 另启动 10000-iteration/64-env run；1024-env
试跑约18秒/iteration、估计52小时，超出7200秒预算，已停止并保留日志。64-env run
预计约4小时，结果完成前不宣称 WAM 收益或失败；该 run 与官方 baseline 不能作完整
environment-step matched comparison。

该 64-env run 在 7200 秒边界完成到 4734/10000，保存了 `model_4000.pt`；固定 TRAIN
评估为 `10/128 success, 6 held60, 17 lifted`，同 iteration 的 official initialized
`model_4000.pt` 为 `103/128, 84 held60, 108 lifted`。它仍不是完整 10000-step 结论，
但与 1000-step 的 `19/128` 对照方向一致：当前直接拼接 WAM summary 并让整套大网络
按官方 PPO 更新，会快速改变原 native policy 而显著损失已学能力。4000-step checkpoint
的 actor 首层更新范数中，新增 WAM/goal 列约 3.68、native 列约 1.34；critic 对应约
7.07 与 5.34，支持“新增输入路径强烈改写策略”的工程解释。下一条路线若继续，应优先
做冻结 native policy 的零初始化 WAM adapter/低学习率对照；当前直接 concatenation 标记
`UNPROMISING`，不把它升级为 WAM 总体无效。

冻结 native 路径的 zero-init WAM adapter 也完成了一个 1000-iteration/64-env probe。
该 adapter 只训练新增的 `wam_actor_adapter`，官方 actor、critic 和 `log_std` 均冻结；
learning-audit 为 `PASS`，且只有 adapter 参数发生变化。原始 published checkpoint 加
同一 WAM-v3 包装器的零初始化正对照仍为 `65/128 success, 39 held60, 71 lifted`，
且直接加载训练目录根部真正的 `model-initial.pt` 得到同样的
`65/128 success, 39 held60, 71 lifted`，所以 WAM 加载和初始行为保持成立。adapter 的
`model_1000.pt` 固定 TRAIN 评估却降为
`1/128 success, 0 held60, 5 lifted`。这比直接拼接路线的 `19/128` 还差，说明在当前
官方 reward、探索噪声和 PPO 更新配方下，新增 WAM actor 通路即使只改 adapter 也会快速
破坏已学 native policy。它仍是单 seed/短 interaction budget 的 Probe 结果，当前将
“official PPO + 直接拼接或该 adapter”标为 `UNPROMISING`，不外推为 WAM 表征本身无效。

下一条判别 probe 固定 published warm-start、WAM-v3 和 zero-init actor adapter，只把并行
环境数从 64 提到 256，保持 1000 iterations 和同一 seed/评估面板；它检验 adapter 退化
是否主要来自小 batch/低 interaction budget。若仍显著低于 published control，停止在这条
官方 PPO adapter 配方上继续加长训练；若恢复，再做 matched environment-step 预算。

256-env 判别 probe 已完成 1000 iterations/20000 optimizer steps（2,048,000 env steps），
audit=PASS；固定 TRAIN 仍只有 `2/128 success, 0 held60, 6 lifted`。因此提高并行环境数
没有恢复已学策略，当前停止继续扩大这条 adapter 配方；该结果仍只支持单 seed/短预算
路线判断，不是对 WAM 表征的正式否定。

为限制 adapter 残差对 native 行为的瞬时破坏，新增显式 `--wam-adapter-gain`，仅在传入
该参数时执行 `gain*tanh(adapter)`；默认 adapter 仍保持原无界行为。`gain=0.05` 的
256-env/1000-iteration run 完成 2,048,000 env steps，audit=PASS，固定 TRAIN 为
`65/128 success, 37 held60, 70 lifted`。同一 256-env/1000-iteration 的 native-only
官方对照为 `73/128, 61 held60, 80 lifted`。因此有界残差把无界 adapter 的 `2/128`
退化修复到接近初始 published policy，但尚未带来相对 native PPO 的改善；当前将它标为
`UNPROMISING` for gain，保留为后续低风险 WAM 接入候选。

为检查官方并行规模，进一步完成 1024-env/300-iteration（2,457,600 env steps）的
gain=0.05 对照。TRAIN 上 WAM/native 分别为 `58/128, 30 held, 63 lifted` 与
`56/128, 33 held, 59 lifted`；但 core-dev 上分别为 `45/128, 23 held, 51 lifted` 与
`69/128, 45 held, 75 lifted`。因此 TRAIN 的小幅 success 差异没有泛化到 DEV，官方规模
下当前有界 WAM adapter 仍标记 `UNPROMISING`，不再追加同配方训练。
同一 WAM/native checkpoint 在 TRAIN 上换用 seed250 重评估后分别为
`50/128, 38 held, 53 lifted` 与 `69/128, 50 held, 76 lifted`，进一步说明 seed249
上的微小 success 优势不稳健。

本轮所有结果的原始 JSON 在
`outputs/wam-rl-policy/official-unidex-remote-evidence-20261016-r1/`；adapter 训练为
`official-unidex-wamv3-adapter-1000-remote-20261017-r1`，固定评估为
`official-unidex-wamv3-adapter-1000-eval-remote-20261017-r3`，published zero-init
control 为 `official-unidex-wamv3-adapter-published-control-eval-remote-20261017-r1`；
256-env 判别训练/评估分别为
`official-unidex-wamv3-adapter-1000-256env-remote-20261017-r1` 和
`official-unidex-wamv3-adapter-1000-256env-eval-remote-20261017-r1`；有界 gain=0.05 的
训练/评估为 `official-unidex-wamv3-bounded-gain005-256env-remote-20261017-r1` 和
`official-unidex-wamv3-bounded-gain005-256env-eval-remote-20261017-r1`，matched native
对照为 `official-unidex-native-1000-256env-remote-20261017-r1` 和
`official-unidex-native-1000-256env-eval-remote-20261017-r1`。
1024-env/300 的 WAM/native TRAIN、WAM/native DEV 产物分别为
`official-unidex-wamv3-bounded-gain005-1024env-300-remote-20261017-r1`、
`official-unidex-native-300-1024env-remote-20261017-r1`、
`official-unidex-wamv3-bounded-gain005-1024env-300-dev-eval-remote-20261017-r1` 和
`official-unidex-native-300-1024env-dev-eval-remote-20261017-r1`；seed250 TRAIN
复评为 `official-unidex-wamv3-bounded-gain005-1024env-300-seed250-eval-remote-20261017-r1`
和 `official-unidex-native-300-1024env-seed250-eval-remote-20261017-r1`。

## 对近期小规模 PPO 卡片的影响审查

这轮官方规模诊断不改写已有卡片的阶段标签，但收紧了它们的适用范围：

| 卡片 | 原判断 | 复核后的含义 |
|---|---|---|
| `P-20261016-independent-motion-ppo` | `PROMISING` | 仍只表示 8 物体、587D 小型策略具备独立动作学习资格；不表示官方抓取收益或 WAM 接入成功。 |
| `P-20261016-shadow-bc-ppo` | `UNCLEAR` | 小面板 frozen-BC residual PPO 的行为读数仍不足以外推官方 PPO；状态不变。 |
| `P-20261016-shadow-scale-continuous` | `UNCLEAR` | scale/critic 拟合和小型闭环读出与官方 baseline 是不同问题；状态不变。 |
| `P-20261016-wam-coordination-bonus` | `UNPROMISING` | 既有协调奖励接法继续停止；官方 WAM 退化不能反向证明该奖励或 WAM 表征无效。 |
| `P-20261016-wam-learned-short-goal` | `UNPROMISING` | endpoint bridge 的预设 gate 仍是独立的表示问题，不被官方 PPO 结果升级或推翻。 |
| `P-20261016-wam-task-readout` | `UNCLEAR` | oracle/readout 结果不能替代线上官方 PPO；状态不变。 |

因此本轮真正被收紧的是“把小规模 PPO 学习资格等同于官方任务收益”的解释，已有卡片的
局部工程观察仍保留。
