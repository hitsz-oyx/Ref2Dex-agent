# brief43：τ同配方分阶段预算延长

## 结果解读表（预注册，逐字保留）

| 观测结果 | 含义 | 会改变哪个决定 |
|---|---|---|
| 到 step2500 三项**均**继续改善且 CI 排除 0（≥2% 相对） | 配方仍是**预算受限**，且斜率未明显衰减 | 允许把"补足到约 11k updates 的长期同配方训练"提上议程（需先与网页 GPT 对齐，不得自行决定） |
| 改善存在但明显衰减（有 CI 排除 0 的改善，但每 1000 updates 的边际改善 < 前一段的一半） | 边际递减 | 先按斜率外推"达到 init→1223 改善量 1.5 倍需要多少 updates"，再决定是否值得占 3 张卡；不直接开长跑 |
| step1500/2000/2500 三项都无 CI 排除 0 的改善（平台或变差） | **配方/监督信号受限**（A 合同无 goal/no-event 的预测上限已到） | 立即停训；τ 的下一步转向监督信号与条件语义，而不是继续加预算 |
| OOM / 非有限 loss / loader failure / `/home2` 可用 <20GiB / 本 run 新产物 >32GiB | 故障门 | 立即停、**不自动重启**、保留证据、不删除历史产物 |

来源：Hermes brief43。只做唯一一次同配方训练，目标2500 optimizer updates；不自行延长到11k。既有init/20/128/1000/1223面板仅引用。新输出tmp/t38-r3，不覆盖历史；不做rollout或物理仿真。

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

## 实际启动命令（阶段结果待实测）

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
