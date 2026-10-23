# brief43：τ同配方分阶段预算延长

## 结果解读表（预注册，逐字保留）

| 观测结果 | 含义 | 会改变哪个决定 |
|---|---|---|
| 到 step2500 三项**均**继续改善且 CI 排除 0（≥2% 相对） | 配方仍是**预算受限**，且斜率未明显衰减 | 允许把"补足到约 11k updates 的长期同配方训练"提上议程（需先与网页 GPT 对齐，不得自行决定） |
| 改善存在但明显衰减（有 CI 排除 0 的改善，但每 1000 updates 的边际改善 < 前一段的一半） | 边际递减 | 先按斜率外推"达到 init→1223 改善量 1.5 倍需要多少 updates"，再决定是否值得占 3 张卡；不直接开长跑 |
| step1500/2000/2500 三项都无 CI 排除 0 的改善（平台或变差） | **配方/监督信号受限**（A 合同无 goal/no-event 的预测上限已到） | 立即停训；τ 的下一步转向监督信号与条件语义，而不是继续加预算 |
| OOM / 非有限 loss / loader failure / `/home2` 可用 <20GiB / 本 run 新产物 >32GiB | 故障门 | 立即停、**不自动重启**、保留证据、不删除历史产物 |

来源：Hermes brief43。只做唯一一次同配方训练，目标2500 optimizer updates；不自行延长到11k。既有init/20/128/1000/1223面板仅引用。新输出tmp/t38-r3，不覆盖历史；不做rollout或物理仿真。

## 最终实测判读

**单一结论：故障门/执行阻塞：停止、保留证据，不自动重启。** 本轮所有结果限同一固定TRAIN面板，未跑rollout，不声称执行或heldout泛化收益。

实际optimizer updates=1570，last_step=1570，状态=FAILED_DATA_LOADER，last_loss=0.6231251955032349，elapsed_s=4878.866605786607，returncode=1；automatic_restart=false。阶段停止原因=TRAINING_FAILED_NO_RESTART。启动PID=555945，supervisor=532926，启动UTC=2026-10-23T00:17:11.319703+00:00，退出UTC=2026-10-23T01:39:14.773531+00:00。

| 面板step | first8_hand mm | fixed_contact_hand mm | first8_object mm | 状态 |
|---:|---:|---:|---:|---|
| init（仅引用） | 16.1694 | 59.2219 | 11.2523 | 既有 |
| 1223（仅引用） | 10.2884 | 31.7598 | 6.5043 | 既有末态 |
| 1500 | 10.0331 | 30.1127 | 6.3766 | 本run唯一一次 |
| 2000 | — | — | — | 未运行：TRAINING_FAILED_NO_RESTART |
| 2500 | — | — | — | 未运行：TRAINING_FAILED_NO_RESTART |

相对step1223的配对Δ和cluster95%CI（new−1223；负值为改善；124母序列/4000次、同锚点与init-A参考）：

| step | first8_hand Δ [CI95] | contact hand Δ [CI95] | first8_object Δ [CI95] | 相对变化% |
|---:|---|---|---|---|
| 1500 | -0.2553 [-0.3423, -0.1715] | -1.6471 [-2.3992, -0.9114] | -0.1277 [-0.2445, -0.0213] | -2.48 / -5.19 / -1.96 |

每1000 updates边际斜率（mm/1000，正值改善；旧r2前段仅引用，新r3后段同配方seed170）：

| 段 | first8_hand | contact hand | first8_object |
|---|---:|---:|---:|
| 1000→1223 | +1.2675 | +11.1300 | +0.8709 |
| 1223→1500 | +0.9216 | +5.9462 | +0.4611 |

逐项斜率比及触发：`[{"segment": [1223, 1500], "versus": [1000, 1223], "ratios": [0.7270656545534826, 0.5342513875737225, 0.5294834123780948], "less_than_half": [false, false, false]}]`。边际递减优先规则为启动前登记，不因结果更改。

故障门优先，本轮不能完成预算受限/边际递减/配方受限三者的科学判定；未达到step2000/2500，不据1500一点外推长跑预算。自动分析文件中的条件线性算术不作为决策依据。1223→1500跨旧r2/新r3实例；两者同seed/配方/初始化，但不是恢复同一优化器后的连续时间段，CI仅覆盖固定面板母序列重采样、不覆盖训练重复方差。

## 启动、身份、资源与保留核对

启动前pgrep原始stdout保存在`tmp/t38-r3/pgrep-before-launch.txt`（空，returncode1）；preflight-attempt.json/explicit-preflight.json记录资源与身份。资源未空闲期间仅原supervisor排队；实际launch.json、train-launch.log、training/progress.json/result.json及stop-receipt.json提供进程/updates/日志/退出码证据。

训练实际非停止配置逐字段与旧配方相同=True；source config SHA=8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51，condition_recipe SHA=36872da24a5dc72cc5580d94a048b5fd7a6ed111b21e7053a4b17acbeff9fc5d，数据身份/6000/seed170/global288/world3见原登记，未改。原脚本d10bc847保持不变；本run仅GPU资源准入副本SHA=aa324ddbc400b6add8ec0be61052ea45fe70191b327cd33dd187ed10afafbf45，计算语句逐字相同。保存钩子SHA=13ca1eb738c19e840c2a10cf1c5bebc591bd675a4b56baf03bdc68d3906851eb，阶段watcher SHA=ef237c07f04e674a8c8a184698c35fc0277c07799629a49ff7f5312c56dee9b5。

| step | 面板目录 | GPU | 秒数 | peak allocated bytes | 可用盘bytes（后） | 总产物bytes（后） |
|---:|---|---:|---:|---:|---:|---:|
| 1500 | `tmp/t38-r3/panel-step001500-A/` | 7 | 24.494 | 641889792 | 271871160320 | 6519304064 |

本run保留checkpoint：

```json
[
  {
    "file": "step000000.pt",
    "bytes": 405716522
  },
  {
    "file": "step000001.pt",
    "bytes": 1217081377
  },
  {
    "file": "step000020.pt",
    "bytes": 1217082017
  },
  {
    "file": "step000128.pt",
    "bytes": 1217082017
  },
  {
    "file": "step001000.pt",
    "bytes": 1217082017
  },
  {
    "file": "step001500.pt",
    "bytes": 1217082017
  },
  {
    "file": "step001570.pt",
    "bytes": 1217082017
  }
]
```

故障后保留全部7个checkpoint作为证据，包括末态step001570.pt；未执行任何清理，pruning.jsonl不存在，删除文件数0。未删除任何历史产物。普通文件占用快照7736423263bytes；详细完整精度分析final-analysis.json。所有面板argv/资源在各点invocation.json与comparison-vs1223.json，原始log位于本run根目录。

终态git log/HEAD/origin/main三项核对收据在tmp/t38-r3/git-final-verification.json，交付回答也逐项报出；tmp产物不入库。

## Loader故障取证与未完成项

2026-10-23本地09:39:04，rank2报告：

```text
RuntimeError: training failed before successful loader delivery: ValueError('batch 96 exceeds sequence cap for source 8: need=25, capacity=24')
```

`train-launch.log`中该原始行及SHA在`tmp/t38-r3/loader-failure-evidence.json`；rank0/1只报告“another rank failed to load its batch”，不能用汇总字串代替rank2真错误。只读源码`src/ptv3_shared_wam_v6/curated_sampling.py:76–85`显示，source有放回抽样后检查`need > len(sequences)*max_per_sequence`并raise；本次具体25>24触发该已有容量检查。没有修改sampler、batch、seed、数据或配方，也没有重放失败batch。`AF_UNIX path too long`、`CUDA out of memory`、`nonfinite loss`日志匹配均0；本次不是brief38短socket问题复发。更深原因及修法留给后续另行授权工作，本轮止于已有日志/代码归因。

原supervisor在UTC01:39:14.773531写stop-receipt，returncode=1、reason=FAILED_DATA_LOADER_NO_RESTART；独立stage_watch随后写STOP并记录TRAINING_FAILED_NO_RESTART。训练自身先完成失败收尾并写末态，不冒称是监控提前救停。实际更新1570、末次训练loss=0.6231251955有限；progress最后常规loss=0.5639325416来自1560日志，不把它混作1570末次loss。耗时4878.867秒，端到端launch→receipt约4923.454秒。训练显存peak reserved=22.9140625GiB、allocated=16.4655942917GiB；稳定速度0.32381168 updates/s。

终态`pgrep -af '[t]rain_pretrain.py'`返回1、stdout为空；GPU4/5/6各4MiB、util0。进程已退出，automatic_restart=false。终态核对在`terminal-evidence.json`：非停止字段仍逐项相同，登记模型/数据/执行入口源码SHA全相同；可用盘270546518016bytes，新run普通文件快照7736431335bytes，低于32GiB。所有故障证据和中途checkpoint均保留。原`tmp/t38-r2/panel_gpu456.py`保持d10bc847原SHA。

只完成step1500面板；step2000、step2500未达、未评测，无checkpoint可供这两点推理。没有另评末态1570，没有新增面板点，没有重复init/1223。不能用故障推出监督信号上限，也不能仅凭1500显著改善批准长期预算。无rollout，执行收益仍未测。

## 新面板实际命令与资源

```sh
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r3/panel_gpu47.py --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r3/training/step001500.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r3/panel-step001500-A --reference /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/panel-init-A --gpu 7 --batch-size 4
```

该命令已执行一次，原输出不可覆盖重跑。`tmp/t38-r3/panel-step001500-A/metrics.json`是三项均值原始读数，`comparison-vs1223.json`是配对CI，`invocation.json`记录原argv/退出码/资源。GPU4当时used23143MiB/free1110MiB，故按指定fallback使用空闲GPU7（used2MiB/free24252MiB）；面板脚本elapsed=24.494211s、整个子进程wall=36.117314s、returncode0、peak allocated=641889792bytes。锚点512、母序列124、4000 bootstrap与anchor SHA均通过原面板/配对核对；面板口径是TRAIN固定锚点的A合同UNKNOWN条件，未混用DEV/best_selection。

运行/取证环境为上述dexplore_repro_py38_torch222_cu121 Python，训练world3/global288/seed170；启动/监控为/usr/bin/python3，源码和SHA见identity登记。预算与预注册提交02f89737/f3b7d27a均早于唯一launch；启动证据提交1ca30345，均已推送。

## 启动前操作化与边界

原表不改。任一评测点若三项都没有“new−1223的CI上界<0且均值下降”，即平台分支，立即STOP；否则继续本次2500上限。边际斜率分指标计算为`(前点EPE−后点EPE)/Δupdates×1000`，前一段从brief41的1000→1223开始，之后1223→1500→2000→2500。若有显著改善但某项正向边际斜率降至前段一半以下（或变非正），不能称三项斜率均未衰减，按保守“边际递减”分支外推；此分支优先于末态绝对降幅≥2%的重叠情况。不以三项平均掩盖分歧。故障优先于科学判读。

预算口径：沿用原launcher的weights-only parent step11000初始化与新optimizer，从本run第1步到2500；不恢复r2优化器，不重跑任何旧面板。旧配置名义updates6000但07:15只到1223；本轮名义上限改2500使唯一运行精确止于目标，而不是放任6000。学习率仅依原warmup50，不依总updates，因此此停止字段不会改变逐步lr。max_seconds4800→10000；原07:15硬停epoch替换为排队在内约6h的有限安全期限（训练自身max_seconds=10000不变），训练正常应先在2500结束。除这三个停止字段，JSON各字段逐项相等。

保存策略按B实施：原launcher和train_pretrain.py完全不改；tmp/t38-r3/bootstrap/sitecustomize.py只在真实train_pretrain进程导入checkpoint模块时，给原checkpoint_due追加128/1500/2000/2500保存点，不碰模型/数据/损失/优化器/RNG或任何前向后向计算。原有其它保存点仍可产生，独立stage_watch仅在原supervisor写出终态收据后删除**本r3**非保留checkpoint，避免与原supervisor文件统计或写盘竞争；过程预计<16GiB，无需冒险边写边删；故障时保留全部证据，不做清理；末态始终保留，.part文件不删。保存钩子工程测试已通过，不进行模型训练。

原冻结面板入口显式拒绝GPU7。为兑现C指定的“4忙则7”，原d10bc847脚本保持逐字不变；r3副本仅改GPU准入一行`(4,5,6)→(4,7)`及对应资源报错文字，所有数据/模型/指标/bootstrap语句逐字不变。保存逐行diff，实际调用副本SHA单独记录，不冒称副本仍为原SHA。原面板CLI reference继续指向既有init-A；从保存的NPZ用同cluster_ci函数追加new−step1223配对CI，不重算init或1223前向。每次记录GPU、秒数、峰值显存与盘。

独立stage_watch负责阶段故障/平台STOP、面板与本run保留清理；原supervisor保持配方SHA全局互斥和短socket路径、有限loss/磁盘/32GiB看门狗。原launcher的estimated_duration_s仍含旧4800秒常量，仅为旧估计字段，不是停止条件；实际按2500/.3235≈7728秒加验证开销估算，结果以进度和退出码为准。

## 预算字段逐字diff与身份

```diff
--- /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/configs/tau_no_goal_no_event_train_v1.json (brief38)
+++ /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/configs/tau_no_goal_no_event_train_v1.json (brief43)
@@ -40,7 +40,7 @@
   "weight_decay": 0.01,
   "clip_grad": 1.0,
   "batch_size": 96,
-  "updates": 6000,
+  "updates": 2500,
   "validation_samples": 32,
   "validation_interval": 500,
   "checkpoint_interval": 1000,
@@ -55,7 +55,7 @@
   "interaction_weight": 1.0,
   "motion_tau_m": 0.002,
   "motion_floor": 0.1,
-  "max_seconds": 4800,
+  "max_seconds": 10000,
   "artifact_budget_bytes": 8589934592,
   "warmup_updates": 50,
   "ptv3": {
@@ -155,6 +155,6 @@
   "curated_manifest": "outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json",
   "condition_recipe": "src/task/ptv3-shared-wam-v6/configs/tau_no_goal_no_event_v1.json",
   "condition_recipe_sha256": "36872da24a5dc72cc5580d94a048b5fd7a6ed111b21e7053a4b17acbeff9fc5d",
-  "hard_stop_epoch": 1792710900.0,
+  "hard_stop_epoch": 1792735439.0,
   "save_initial_weights": true
 }
```

```json
{
  "config_before_sha256": "de74a83d976eca08a7d9027ecda0ae01b5219447b4dc64677fe1f395921018ad",
  "config_after_sha256": "8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51",
  "stopping_fields": {
    "updates": {
      "before": 6000,
      "after": 2500
    },
    "max_seconds": {
      "before": 4800,
      "after": 10000
    },
    "hard_stop_epoch": {
      "before": 1792710900.0,
      "after": 1792735439.0
    }
  },
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
  "panel_gpu47_sha256": "aa324ddbc400b6add8ec0be61052ea45fe70191b327cd33dd187ed10afafbf45",
  "code_sha256": {
    "src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py": "89688641486e6937c7db101b7b0c83ac8afb7c1bf0100d49d8a1e83e366f08f9",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/model.py": "a9f415b871678d62f852d8baa14dbe9647ba8a97a1fd88a59dbfe57ae1ef37a9",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/data.py": "3726d214e34cd3769a49f8a5e24242f82f919cdb61fa448584fb165d16c60535",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/goal_policy.py": "a1b452f419190549b293a190de6f207ca8e6b211aaa2a497389359b6efb5d9fc",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/events.py": "b5b70b628c0e572b5f2b80db10798ddb2ac34bccfa8962b5b81fcbd17d7aabcc",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/flow_matching.py": "ca558c9d4adec6f3b67a6b0f407f6c0ae1fc830772b6830daacd87d8e9479b53"
  },
  "checkpoint_hook_sha256": "13ca1eb738c19e840c2a10cf1c5bebc591bd675a4b56baf03bdc68d3906851eb",
  "stage_watch_sha256": "ef237c07f04e674a8c8a184698c35fc0277c07799629a49ff7f5312c56dee9b5"
}
```

原始文件SHA与run副本/钩子SHA保存在tmp/t38-r3/identity.json。字段核对：condition_recipe_sha256=36872da24a5dc72cc5580d94a048b5fd7a6ed111b21e7053a4b17acbeff9fc5d；manifest_sha256=5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f；ordered_window_tuple_sha256=d18ec57ee2de589152131eb2dfab4ac7581ab29275d08756c973dce2c000c3d7；6000窗口、seed170、batch96/rank×world3=global288、模型/损失/lr逐字不变。已有合同/6000窗口审计只引用checks.json，本轮只做启动身份守门。

## 实际启动命令（已启动一次，禁止原目录重启）

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/tmp/t38-r3/bootstrap${PYTHONPATH:+:$PYTHONPATH}" REF2DEX_B43_RUN_ROOT="$PWD/tmp/t38-r3" /usr/bin/python3 src/task/ptv3-shared-wam-v6/tools/run/launch_tau_no_goal_no_event.py --checks src/task/ptv3-shared-wam-v6/docs/audit/brief37-no-goal-no-event-20261023-r1/checks.json --output tmp/t38-r3 --queue
```

面板资源差异（原文件未改）：

```diff
--- /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel_gpu456.py
+++ /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r3/panel_gpu47.py
@@ -21,7 +21,7 @@

 def main():
     p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--gpu',type=int,default=4);p.add_argument('--batch-size',type=int,default=4);args=p.parse_args()
-    if args.gpu not in (4,5,6):raise ValueError('brief38 panel only authorized on idle GPUs4/5/6')
+    if args.gpu not in (4,7):raise ValueError('brief43 panel only authorized on GPUs4/7')
     if args.output.exists():raise FileExistsError(args.output)
     free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
     if free<8*2**30 or shutil.disk_usage('/home2').free<20*2**30:raise RuntimeError('panel resource preflight failed')
```

## 08:01资源排队实测

显式pgrep返回码1、stdout为空，无train_pretrain训练。身份/磁盘/预算门均通过，但GPU4/5/6已被另一会话的tau-checkpoint-ab作业占用（worker PID508325/508326/508327及其物理host）；本轮只读取进程信息，不干预。原--queue空闲门因此未通过，训练尚未启动，updates=0。

为使排队不挤占已授权2500-update训练窗口，仅将本轮新的绝对安全期限从最初3h扩大为排队在内6h，仍为停止字段；训练max_seconds=10000、updates=2500不变。初版停止配置与调整diff分别保留config-before-queue-adjustment.json、queue-deadline-only.diff，实际最终配置SHA见identity.json。原07:15停止已撤销，不自动扩展至11k。

## 08:17真实启动证据（后续结算以前部实测为准）

资源释放后原queue自动启动一次，UTC=2026-10-23T00:17:11.319703+00:00（本地08:17:11）；supervisor PID532926、torchrun PID555945、native rank PID556007/556008/556009，物理GPU4/5/6。启动提交 `906d7168cdfc5d29ddbc9683621519d5fcb83bd9`，配置SHA `8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51` 与预算登记一致。原始进程/进度/日志/三个真实rank的保存钩子证据在training-start-proof.json；此刻实际step=20、loss=1.473211130，有限，world3/global288/seed170均核对通过。已写出step0与更新后的checkpoint，socket路径错误匹配0。

实际native config去掉三个停止字段后与旧配置逐字段完全相同；不是用离线面板读数冒充训练更新。GPU实测CSV：`4, 22561, 99 / 5, 20299, 97 / 6, 24209, 98`。首段速率/ETA仍受预热和固定DEV验证影响，最终使用真实进度和退出码，不改变配方。平台/故障门仍按预注册表执行；目前没有面板结果，不提前宣告预算或监督上限。

## Git交付记录

终态首次`git pull --rebase origin main`返回128：`cannot pull with rebase: You have unstaged changes`。工作树包含本报告与其他会话修改，未stash、未强推；本轮只显式暂存/提交本报告，随后按brief43普通`git push origin main`。预算配置的三个停止字段已在启动前独立显式提交并推送；不暂存任何他人文件。提交/推送后使用`git log --oneline -1`、`git rev-parse HEAD`、`git rev-parse origin/main`、`git ls-remote origin main`核对，原始结果写`tmp/t38-r3/git-final-verification.json`并在交付回答报出。
