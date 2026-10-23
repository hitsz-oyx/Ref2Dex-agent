# τ 从1768末态续跑至判定点

## 结果解读表（启动前预注册）

| 观测结果 | 含义 | 决定 |
|---|---|---|
| 到 step2500 三项仍改善且 CI 排除 0（≥2% 相对） | 配方仍**预算受限** | 完成授权的3000观测点；不擅自开约11k长跑 |
| 有改善但边际 < 前段一半 | **边际递减** | 按斜率外推达标所需预算，不直接长跑 |
| 三项都无 CI 排除 0 的改善 | **配方/监督信号受限**（A 合同无 goal/no-event 的预测上限已到） | 立即STOP，转条件审计 |
| OOM/非有限 loss/loader failure/盘<20GiB/本 run 产物>32GiB | 故障门 | 立即停、保证据、**不自动重启** |

来源：Hermes/机主2026-10-23 14:1x明确授权放宽硬停。无新增外审裁决。Decision为既有同配方预算斜率，不重训新模型/配方，不扩数据、不跑rollout/物理、不碰执行器。

## 最终实测结论

**预算受限：2500三项联合改善且未触发明显衰减；不擅自开长跑。** 固定TRAIN面板证据；执行/heldout收益未测。

新增updates=1232，累计last_step=3000；status=COMPLETED，loss=0.5843357443809509，elapsed_s=5815.27829630021，returncode=0，automatic_restart=false。config完整相等=True。

| step | first8_hand mm | contact_hand mm | first8_object mm | loss | updates/s |
|---:|---:|---:|---:|---:|---:|
| 1223（引用） | 10.2884 | 31.7598 | 6.5043 | — | — |
| 1500（引用） | 10.0343 | 30.0755 | 6.3679 | — | — |
| 2000 | 9.5911 | 27.6275 | 6.0564 | 0.558665 | 0.209699 |
| 2500 | 9.2819 | 26.1951 | 5.8109 | 0.517676 | 0.219541 |
| 3000 | 8.9808 | 24.6594 | 5.6394 | 0.519917 | 0.219631 |

| step / reference | first8_hand Δ [CI95] | contact_hand Δ [CI95] | first8_object Δ [CI95] |
|---|---|---|---|
| 2000 / 1223 | -0.6973 [-0.8158, -0.5788] | -4.1323 [-5.0152, -3.2577] | -0.4479 [-0.6010, -0.3131] |
| 2000 / 1500 | -0.4431 [-0.5414, -0.3453] | -2.4480 [-3.1785, -1.7751] | -0.3115 [-0.4752, -0.1677] |
| 2500 / 1223 | -1.0065 [-1.1430, -0.8691] | -5.5647 [-6.6052, -4.5801] | -0.6934 [-0.8786, -0.5327] |
| 2500 / 1500 | -0.7523 [-0.8721, -0.6354] | -3.8804 [-4.7563, -3.0986] | -0.5570 [-0.7360, -0.3966] |
| 2500 / 2000 | -0.3092 [-0.3965, -0.2271] | -1.4324 [-1.9118, -0.9777] | -0.2455 [-0.3682, -0.1424] |
| 3000 / 1223 | -1.3076 [-1.4639, -1.1496] | -7.1004 [-8.2050, -6.0403] | -0.8649 [-1.0747, -0.6847] |
| 3000 / 1500 | -1.0534 [-1.1973, -0.9132] | -5.4161 [-6.5000, -4.4253] | -0.7285 [-0.9261, -0.5577] |
| 3000 / 2500 | -0.3011 [-0.3758, -0.2310] | -1.5357 [-2.1334, -0.9482] | -0.1715 [-0.2806, -0.0856] |

每1000updates边际斜率（正为改善）：

| 段 | hand | contact | object | 比前段 |
|---|---:|---:|---:|---|
| 1223→1500 | +0.9175 | +6.0806 | +0.4925 | None |
| 1500→2000 | +0.8863 | +4.8960 | +0.6230 | [0.9659426789228799, 0.8051902866209614, 1.265015167391182] |
| 2000→2500 | +0.6184 | +2.8648 | +0.4910 | [0.6976989431935353, 0.5851355153349284, 0.7881671041691609] |
| 2500→3000 | +0.6022 | +3.0714 | +0.3429 | [0.9739184535009197, 1.0721219155078534, 0.6983607856253425] |

约11k预算成本估计（未启动）：

```json
{
  "wall_rate_including_observed_panel_overhead": 0.21185572507919728,
  "wall_hours": 10.489318716270219,
  "wall_GPU_hours": 31.467956148810657,
  "from_step": 3000,
  "to_step": 11000,
  "remaining_updates": 8000,
  "updates_per_s": 0.2196092284139936,
  "GPUs": 3,
  "hours": 10.118983788937266,
  "GPU_hours": 30.3569513668118,
  "authorized_to_start": false,
  "latest_2500_to_3000_significant": [
    true,
    true,
    true
  ],
  "assumption": "same cards/throughput; steady and observed-wall estimates; excludes future queue and changes in thermal throttling; TRAIN only"
}
```

每点metrics/stage-receipt/invocation在tmp/t38-r5/panel-stepNNNNNN-A；父/参考输出只读。完整精度、速度、磁盘与CI见final-analysis.json。

## 终态取证与四行实际落点

原r4已正常保存1768，本轮只做一次明确授权的状态续跑；新增1232更新，累计3000，native status=COMPLETED、returncode0、loader_failure=null、automatic_restart=false。16:30:37硬停尚未到，因先到3000正常结束；stop-receipt的实际退出UTC为`2026-10-23T08:07:53.022703+00:00`。native续跑elapsed=5815.278296s（96.92min），launch→receipt端到端5835.137s；checkpoint.elapsed是旧阶段+本阶段的累计计时，不能与新增阶段耗时混写。

每500步表中的loss是该全局step的world汇总日志值；3000为0.519917425。result.last_loss=0.5843357444是同一步rank0本地loss，不用较小的汇总数冒充本地末次值。loss按批次波动，四行判据只按预注册EPE/CI/斜率，没有事后用loss改单调性或换配方。

2500相对1223三项下降9.7827%/17.5213%/10.6613%，CI均严格负，满足≥2%联合门。3000相对1223下降12.7095%/22.3567%/13.2974%；相对1500下降10.4985%/18.0085%/11.4400%；相对2500下降3.2441%/5.8626%/2.9507%，三套CI三项均排除0。

2500→3000边际斜率为0.602235/3.071442/0.342921mm每1000更新，比上段为0.973918/1.072122/0.698361；此前所有逐项比值也≥0.5。因此按冻结优先规则：预算受限门通过；明显边际递减门未触发；平台门未触发；故障门未触发。**单一结论是本轮固定TRAIN/A面板上仍预算受限，不是对11k收益、heldout或执行收益的保证。**

native result.best_selection仍为旧DEV step128（score1.784256843、trajectory macro27.541454562mm）；这是另一分布/聚合口径，未用于本轮选点或判据。3000是预注册末态，不称全局最佳。未重跑128/init/1223/1500面板、未添加基线/数据/模型；没有rollout、物理或执行器操作，三臂RL仍HOLD。

## Config、优化器、RNG与数据身份终验

terminal-evidence.json中source/config-before/runtime三个config SHA完全相同：8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51；末态checkpoint.config也逐字段相同。原updates2500/max_seconds10000/旧hard epoch保留在该冻结配置内；真正新停止控制仅来自已记录CLI/manifest/checkpoint顶层effective_stop_budget（3000，当前run7200s，epoch1792744237.846317）。这是原样严格resume，并未放松saved.config==config或改模型配置。

末态CPU读取step003000.pt成功：617份AdamW state计数min=max=3000、3参数组、3rank RNG；与初次1769计数证明一起排除optimizer重置。原生DDP三rank参数签名完全相等（sum27571.563436612985、square-sum122079.21814758345），原生检查通过。累计新354816次window draw恰为1232×288，从1769到3000按同seed/step/rank重放source分配，与实际source_draws逐项全等。池仍为6000窗口，这些是重复曝光，不是新增独立数据。

模型/data/goal_policy/events/flow/sampler/停止模块源码指纹均与启动登记相同；recipe SHA36872da2…、curated manifest5bda6a99…、ordered tuple d18ec57e…、seed170、world3/global288/batch96/workers4不变。原frozen panel SHA d10bc847…不变；identity中的旧兼容键panel_gpu47_sha256也等于该原脚本SHA，实际所有前向只在GPU4，训练只在4/5/6，没有使用GPU7。

train-launch.log中FAILED_DATA_LOADER、exceeds sequence cap、AF_UNIX path too long、CUDA out of memory、nonfinite loss、is non-finite匹配均0。终态pgrep返回1/stdout为空、三个自有rank已退出；GPU4/5/6为6/4/4MiB、util0。原生训练峰值reserved23.091797GiB、allocated16.503350GiB。未自动重启、未抢占未知PID。

## 每500步磁盘、显存与receipt

| step | metrics.json | loss / updates/s | panel脚本秒 / 子进程wall秒 | peak allocated bytes | free bytes前→后 | run bytes后 |
|---:|---|---|---|---:|---|---:|
| 2000 | `tmp/t38-r5/panel-step002000-A/metrics.json` | 0.558665 / 0.209699 | 24.381 / 36.078 | 641889792 | 70848454656→70836633600 | 2460485780 |
| 2500 | `tmp/t38-r5/panel-step002500-A/metrics.json` | 0.517676 / 0.219541 | 24.686 / 36.121 | 641889792 | 69568610304→69557088256 | 3688962099 |
| 3000 | `tmp/t38-r5/panel-step003000-A/metrics.json` | 0.519917 / 0.219631 | 25.051 / 38.104 | 641889792 | 68333936640→68317843456 | 4917437642 |

每目录`stage-receipt.json`含相对1223/1500及相邻点的配对Δ/clusterCI95；`invocation.json`保存准确argv和资源预检。母序列124、锚点512、bootstrap4000和anchor SHA均通过同一冻结脚本/NPZ的行、mothers/counts对齐检查。原参考只读，未重算其前向或更换固定contact窗。

独立watchdog观测可用盘最小68317831168bytes（63.626GiB），观测产物最大4917447261bytes；均未接近20GiB/32GiB故障门。末态快照run普通文件3700397717bytes，free68890132480bytes。

保留checkpoint清单：

```json
[
  {
    "file": "step002000.pt",
    "bytes": 1217082657
  },
  {
    "file": "step002500.pt",
    "bytes": 1217082657
  },
  {
    "file": "step003000.pt",
    "bytes": 1217082657
  }
]
```

正常终态receipt后仅清理本r5非评测首步1769，共1文件/1217082657bytes；原r4/1768及全部历史r4/r3/r2/r1/outputs未删除。first-step状态证明JSON仍保留，2000/2500/3000各模型/optimizer/RNG完整checkpoint保留。没有复制父checkpoint。

## 约11k成本估算及边界

3000−2500三项CI都负，满足“最新一段仍改善”；前部给出到11000尚需8000更新的两种算术成本：同3张卡，按steady0.219609约10.119h/30.357GPUh；按含本轮评测暂停的实际循环wall rate0.211856约10.489h/31.468GPUh。约**10.1–10.5小时、3卡、30.4–31.5GPU小时**；不含未来排队和热降频/负载变化，也不保证EPE按斜率持续下降。

本次没有触发“低于前段一半”，所以不强做该分支的1.5倍改善量外推。11000按全局optimizer更新量计，池仍6000；不是独立样本或不同数据分布曝光等价证明。机主/网页GPT对齐后才能决定是否值得给该预算；controller硬限制target≤3000、new-run≤7200，final-analysis.long_budget_estimate.authorized_to_start=false。**未启动长跑、未部署新权重到执行器。**

## 实际运行命令（历史取证，不可覆盖复用原目录）

stage保存/暂停由原launcher环境派生并注入run-local bootstrap；原生完整命令展开如下，GPU固定4/5/6。执行应通过已提交的continue_tau_no_goal_no_event.py，以保留全局recipe互斥和独立看门狗，不应绕过入口直接复用旧输出：

```sh
CUDA_VISIBLE_DEVICES=4,5,6 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$PWD/tmp/t38-r5/bootstrap${PYTHONPATH:+:$PYTHONPATH}" REF2DEX_B43_RUN_ROOT="$PWD/tmp/t38-r5" REF2DEX_RESUME_FIRST_STEP=1769 REF2DEX_RUN_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/b38-1038793 TMPDIR=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/b38-1038793/tmp /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py --config /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/configs/tau_no_goal_no_event_train_v1.json --manifest /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json --stats /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r5/training --resume /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r4/training/step001768.pt --steps 3000 --resume-run-seconds 7200 --resume-hard-stop-epoch 1792744237.846317 --workers 4 --git-commit 188bebd2ff0f8fcc6325ead9dcbfb4456bdd7cfa --preserve-all-checkpoints --checkpoint-budget-gib 32
```

面板2000原argv：

```sh
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel_gpu456.py --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r5/training/step002000.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r5/panel-step002000-A --reference /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/panel-init-A --gpu 4 --batch-size 4
```

面板2500原argv：

```sh
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel_gpu456.py --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r5/training/step002500.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r5/panel-step002500-A --reference /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/panel-init-A --gpu 4 --batch-size 4
```

面板3000原argv：

```sh
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel_gpu456.py --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r5/training/step003000.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r5/panel-step003000-A --reference /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/panel-init-A --gpu 4 --batch-size 4
```

源码MVP提交53fbb180，真实启动树188bebd2ff0f8fcc6325ead9dcbfb4456bdd7cfa；模型/数据等文件SHA另有登记。运行/测试环境为dexplore_repro_py38_torch222_cu121，实际python/torch/cuda版本与loader参数见training/run-manifest.json。独立stage与completion仅记录/评测，不会启动第二个训练。终态git log/HEAD/origin/main/ls-remote收据写tmp/t38-r5/git-final-verification.json；tmp产物不入库。

## 启动前固定协议

上一轮已正常STOPPED到1768，returncode0、loader_failure=null、末态checkpoint为tmp/t38-r4/training/step001768.pt；此次是明确授权的状态续跑，不是故障后的自动重启。新输出tmp/t38-r5，旧r4/r3/r2/r1只读。恢复模型、AdamW所有动量/step及三rank Python/NumPy/Torch/CUDA RNG，第一更新1769，目标全局step3000；最多新增1232更新。seed170/world3/global288/batch96/rank/workers4不变。

冻结config文件与checkpoint内config保持逐字段/逐字节SHA8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51；recipe36872da24a5dc72cc5580d94a048b5fd7a6ed111b21e7053a4b17acbeff9fc5d、manifest5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f、ordered tuple d18ec57ee2de589152131eb2dfab4ac7581ab29275d08756c973dce2c000c3d7、6000窗口均不变。严格resume原函数要求saved.config == config，因此不放松兼容性校验。新增CLI运行预算与config分离：沿用--steps3000指定结束步，新增仅resume可用的run-seconds/hard-stop-epoch覆写停止逻辑，记录于manifest/checkpoint顶层effective_stop_budget，绝不改模型看到的config。budget view仅供原local execution的停止上限准入校验，其他字段保持相等。原expired hard epoch与累计max_seconds不能误杀新授权续跑。默认不传新参数的旧行为保持一致。

硬停止于真正启动后120分钟（提前30秒写STOP保存），或全局3000，先到先停。若资源不足只排队、不抢占；所有GPU计算仅4/5/6。使用原recipe SHA全局flock互斥，启动前pgrep必须为空。故障立即STOP，无自动重启；独立supervisor/阶段watcher都记录磁盘和有限loss。

2000/2500/3000每500步各一次同冻结面板，模型/数据/原panel_gpu456.py d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51逐字不变，同512锚点/124母序列/4000 bootstrap，UNKNOWN goal/event；参考原init-A，另对保存NPZ给new−1223与new−r4/1500的配对母序列聚类CI95。两参考只读，不重跑。每点receipt记录EPE、当前训练loss、实测速率、评测秒数/峰值显存、盘可用/产物bytes。三个rank在checkpoint保存后的barrier暂停并empty_cache，GPU4达8GiB空闲门后面板，结束再release，沿用r4已实测成功的资源调度。

判据操作化沿用前轮：任一评测点相对1223三项都无严格改善（均值下降且CI上界<0）即平台STOP。其他情况完成授权3000；斜率逐项按1000updates尺度，前段1223→r4/1500，后段1500→2000→2500→3000。有显著改善但任一项斜率不足前段一半或非正，边际递减分支优先于累计≥2%行，不能用均值掩盖分歧。故障优先。2500/3000未达到时明确缺证，不强归类。若3000相对1500/1223仍有CI排除0的改善，估算3000→约11k的时间与3卡GPU小时，但不启动长跑；预测不代表执行或heldout收益。

磁盘预算：父1768 checkpoint不复制，采用只读resume；新点2000/2500/3000约3×1217082017=3651246051bytes（3.40GiB），加首个resume DEV可能保存1769及异常末态，按6个full checkpoints+3面板估计<8GiB，预留16GiB峰值。启动要求free≥36GiB，过程硬门free≥20GiB/run≤32GiB；只清理本r5正常终态后的非评测中途checkpoint，故障全留，绝不删旧产物。

代码与预注册先提交，实际启动/续训恢复/面板/终态结果随后填入。

## 代码停止路径diff、验证与实际入口

原train_pretrain.py只增加独立停止参数/预算校验视图/停止判断及预算取证字段，模型/损失/优化器/数据逻辑不改。完整diff：

```diff
diff --git a/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py b/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py
index 2eb0d6db..f5364ebd 100644
--- a/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py
+++ b/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py
@@ -33,6 +33,7 @@ from ptv3_shared_wam_v6.geometry_consistency import validate_geometry_config
 from ptv3_shared_wam_v6.precision import ActivationMonitor
 from ptv3_shared_wam_v6.flow_matching import validate_flow_config
 from ptv3_shared_wam_v6.local_subset import initialize_flow_weights
+from ptv3_shared_wam_v6.stop_budget import resolve_stop_budget, validation_budget_view, stop_budget_exhausted
 from ptv3_shared_wam_v5.training_contract import execution_contract,validate_resume,loader_options,restore_optimizer_state
 from ptv3_shared_wam_v5.selection import trajectory_score,bootstrap_trajectory,TAU_SELECTION
 from ptv3_shared_wam_v4.objectives import normalized_components
@@ -115,6 +116,8 @@ def main():
     p.add_argument('--loader-timeout',type=int,default=120)
     p.add_argument('--recovery-probe-updates',type=int)
     p.add_argument('--continuation-updates',type=int)
+    p.add_argument('--resume-run-seconds',type=float)
+    p.add_argument('--resume-hard-stop-epoch',type=float)
     p.add_argument('--capacity-probe-updates',type=int)
     p.add_argument('--resume',type=Path);p.add_argument('--initialize-from',type=Path)
     p.add_argument('--initialize-geometry-from',type=Path)
@@ -147,6 +150,7 @@ def main():
         # the reader boundary before CUDA/optimizer initialization.
         dataset[0]
     last=config['updates'] if a.steps is None else a.steps
+    stop_budget=resolve_stop_budget(config,last,a.resume,a.resume_run_seconds,a.resume_hard_stop_epoch)
     batch_size=config['batch_size'] if a.batch_size is None else a.batch_size
     if a.recovery_probe_updates is not None and (not a.resume or a.smoke or not 1<=a.recovery_probe_updates<=16):
         raise ValueError('recovery probe requires a checkpoint and1..16 actual updates')
@@ -159,7 +163,7 @@ def main():
         raise ValueError('full training requires a complete, explicitly admitted frozen v6 recipe')
     if last<1 or batch_size<1:raise ValueError('positive bounded steps/batch required')
     world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
-    validate_local_execution(config,recipe,world,last,batch_size)
+    validate_local_execution(validation_budget_view(config,stop_budget),recipe,world,last,batch_size)
     if recipe.get('local_subset_scope') in LOCAL_SCOPES and not (a.resume or a.initialize_from or a.initialize_geometry_from or a.initialize_flow_from or a.initialize_condition_from):
         raise ValueError('local subset requires admitted initialization or its own strict resume')
     if recipe.get('local_subset_scope') == EXPANDED_SCOPE and world==1 and not a.capacity_probe_updates:
@@ -269,7 +273,7 @@ def main():
             loader=dict(context=a.loader_context,workers=a.workers,timeout_s=a.loader_timeout),
             trajectory_selection_protocol=TAU_SELECTION,recovery_probe=bool(a.recovery_probe_updates),continuation_updates=a.continuation_updates,
             event_recipe=recipe,capacity_probe_updates=a.capacity_probe_updates,
-            checkpoint_preservation=dict(all=a.preserve_all_checkpoints,budget_gib=a.checkpoint_budget_gib),
+            checkpoint_preservation=dict(all=a.preserve_all_checkpoints,budget_gib=a.checkpoint_budget_gib),effective_stop_budget=stop_budget,
             source_ids=dataset.meta['source_ids'],max_seconds=config['max_seconds'],min_free_gib=20,config=config))
     start=time.monotonic();times=[];events=[];seen=set();draws=Counter();loss_value=None;step=first-1;completed_step=first-1;checkpoints_written=0;last_saved=None

@@ -290,7 +294,7 @@ def main():
                 execution_contract=contract,resume_binding=resume_binding,migration_binding=migration_binding,topology_transition=topology_transition,initialization_binding=initialization_binding,
                 elapsed_s=previous_elapsed+time.monotonic()-start,
                 selection=dict(score=best_score,step=best_step,checkpoint=best_path,trajectory=tau_best),
-                model=model.state_dict(),optimizer=optimizer.state_dict(),random_state=states),pending)
+                model=model.state_dict(),optimizer=optimizer.state_dict(),random_state=states,effective_stop_budget=stop_budget),pending)
             pending.replace(target);target.chmod(0o664);checkpoints_written+=1
             for name,path in [('best',best_path),('trajectory-best',tau_best.get('checkpoint') if tau_best else None)]:
                 if path:
@@ -318,7 +322,7 @@ def main():
     wrapped.train()
     iterator=iter(loader);loader_failure=None
     for step in range(first,last+1):
-        stop=(a.output/'STOP').exists() or previous_elapsed+time.monotonic()-start>config['max_seconds'] or shutil.disk_usage(a.output).free<20*2**30 or time.time()>=config.get('hard_stop_epoch',float('inf'))
+        stop=(a.output/'STOP').exists() or stop_budget_exhausted(stop_budget,time.monotonic()-start,previous_elapsed,time.time()) or shutil.disk_usage(a.output).free<20*2**30
         flag=torch.tensor(int(stop),device=device)
         if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
         if flag.item():break
@@ -432,7 +436,7 @@ def main():
         result=dict(status='FAILED_DATA_LOADER' if loader_failure else ('COMPLETED' if completed_step==last else 'STOPPED'),
             loader_failure=loader_failure,engineering_smoke=a.smoke,recovery_probe=bool(a.recovery_probe_updates),continuation_updates=a.continuation_updates,optimizer_updates=completed_step-first+1,
             last_step=completed_step,last_loss=loss_value,elapsed_s=time.monotonic()-start,source_ids_seen=sorted(seen),
-            capacity_probe=bool(a.capacity_probe_updates),
+            capacity_probe=bool(a.capacity_probe_updates),effective_stop_budget=stop_budget,
             steady_updates_per_s=(len(times)-2)/(times[-1]-times[1]) if len(times)>2 else None,
             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if device.startswith('cuda') else None,
             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.startswith('cuda') else None,
```

新增stop_budget.py仅管理停止参数；默认旧累计max_seconds/hard epoch行为保持，显式resume预算按当前续跑计时。9项测试覆盖严格config resume、旧过期epoch覆写、previous_elapsed不误扣新预算、无deadline的旧配置JSON兼容、非法/不完整覆写拒绝；均通过。真实冻结config的原local execution准入（预算视图3000/7200、world3/batch96）CPU检查通过，config原字典未变。未重复旧A合同/6000窗口审计或旧面板。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/task/ptv3-shared-wam-v6/src /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q -p no:cacheprovider --basetemp tmp/t38-r5/test-temp-final src/task/ptv3-shared-wam-v6/tests/test_resume_stop_budget.py
/usr/bin/python3 tmp/t38-r5/start_once.py
```

start_once的唯一训练入口为：

```sh
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python src/task/ptv3-shared-wam-v6/tools/run/continue_tau_no_goal_no_event.py --resume tmp/t38-r4/training/step001768.pt --output tmp/t38-r5 --checks src/task/ptv3-shared-wam-v6/docs/audit/brief37-no-goal-no-event-20261023-r1/checks.json --target-step 3000 --run-seconds 7200 --queue
```

该launcher持有recipe SHA全局互斥、确认pgrep/空卡/数据与配置，计算真实启动+7200的hard epoch；调用原生--resume而非weights-only。恢复点与初次1769保存钩子只是状态取证，不是新增基线；评测只在2000/2500/3000。旧输出不覆盖、不拷贝父checkpoint。

身份登记：

```json
{
  "config_before_sha256": "8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51",
  "config_after_sha256": "8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51",
  "stopping_fields": {},
  "all_other_config_fields_identical": true,
  "condition_recipe_sha256": "36872da24a5dc72cc5580d94a048b5fd7a6ed111b21e7053a4b17acbeff9fc5d",
  "data_identity": {
    "manifest_sha256": "5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f",
    "ordered_window_tuple_sha256": "d18ec57ee2de589152131eb2dfab4ac7581ab29275d08756c973dce2c000c3d7",
    "windows": 6000
  },
  "seed": 170,
  "world_size": 3,
  "global_batch": 288,
  "panel_original_sha256": "d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51",
  "panel_gpu47_sha256": "d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51",
  "code_sha256": {
    "src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py": "5734cee8691f3f02dc1d03af77504ef3451d28609dd97147b488ff9b7b79367a",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/model.py": "a9f415b871678d62f852d8baa14dbe9647ba8a97a1fd88a59dbfe57ae1ef37a9",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/data.py": "3726d214e34cd3769a49f8a5e24242f82f919cdb61fa448584fb165d16c60535",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/goal_policy.py": "a1b452f419190549b293a190de6f207ca8e6b211aaa2a497389359b6efb5d9fc",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/events.py": "b5b70b628c0e572b5f2b80db10798ddb2ac34bccfa8962b5b81fcbd17d7aabcc",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/flow_matching.py": "ca558c9d4adec6f3b67a6b0f407f6c0ae1fc830772b6830daacd87d8e9479b53",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/curated_sampling.py": "23df387c799220cf9f04a4ceeb688fd0d989a8c6304670b4d84d21368158c4ec",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/stop_budget.py": "3c9407a8f17f18e2056e183881496fbf3d44098c5ed5bc0927359009221a3921"
  },
  "checkpoint_hook_sha256": "6a2045805ef71af4a88350c267b0745d18b462b61e783cd483ecf9eba1d4e7f4",
  "stage_watch_sha256": "fa84f3ca17837f01e7013d470b2297ff92514b23b4e54aafed008b2aa2ff27d9",
  "resume_launcher_sha256": "016d7b820c8cf423bde9eec655250e6bdbd3113c980ddaeabc965a6796266f14"
}
```

补充统计在启动前固定：2500与3000同时对上一个500步点给配对CI（只处理既有NPZ，无新前向）；用于区分“累计仍好”与“最新一段仍有显著收益”。第6条成本估算在3000相对1500累计三项改善CI排除0时提供，同时必须列出3000−2500的显著性，不用累计改善冒充相邻改善，不据估算自行启动长跑。四行判据不变。

启动前pull --rebase被未提交工作树拒绝（exit128）；不stash、不强推，仅显式提交本轮停止路径/入口/测试/预注册报告，随后普通push。

## 真实续跑启动证据

旧r4已正常收尾1768，本轮原生--resume于UTC06:30:37.886053（本地14:30:37）启动，supervisor1038793、torchrun1038828，GPU4/5/6，源码HEAD188bebd2ff0f8fcc6325ead9dcbfb4456bdd7cfa。新hard epoch1792744237.846317（16:30:37），3000或120分钟先到先停。运行config仍8c157ad4…，saved/current严格相等，只有manifest/checkpoint顶层独立effective_stop_budget扩展。

CPU admission记录父step1768、617个optimizer state、3参数组、3 rank RNG；新首个checkpoint step001769.pt已实写，617个AdamW step计数min=max=1769，证明非optimizer清零后重训（后者会为1）。run-manifest.resume_binding确认bound-effective-batch、parent1768，原生Python/NumPy/Torch/CUDA RNG恢复路径不变。optimizer_restore的moved计数0表示该Torch已将scalar step放到正确设备，不是恢复失败；首版离线取证误将它要求>0，纠正为>=0后完成真实计数核对，未重启训练或改优化器。

resume-live-proof.json采样累计1800、此次新增32步；累计source_draws从1769重放至1800逐项完全一致，包含旧采样器另一处1783/source4的容量越界修复，loader失败日志0。早期实际loss/速度在proof与progress原始JSON中，尚不提前宣布预算结论。父checkpoint读取不复制，resume-admission.json保留原parent数据身份3d0cb55b…（parent manifest+stats hash），curated manifest/ordered tuple另列，不能混为一个SHA。

## 交付与复现收据

收尾再次执行git pull --rebase origin main，被其他未提交工作树拒绝（exit128）；未stash、未强推、未修改/提交他人文件。仅显式提交本报告并普通push；HEAD/origin/main/ls-remote与git log -1核对记录在tmp/t38-r5/git-final-verification.json，交付回答同时报出。tmp/模型/NPZ/运行日志不入库。

实际完整入口为已执行一次的`/usr/bin/python3 tmp/t38-r5/start_once.py`，具体受控launcher、原生CLI与三个原面板argv均在上文；原目录有ONE_SHOT会拒绝再次启动。要做未来实验必须新目录和新的机主授权，本轮没有排队或启动任何3000以后的训练。
