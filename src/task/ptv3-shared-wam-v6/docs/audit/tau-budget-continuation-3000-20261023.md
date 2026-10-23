# τ 从1768末态续跑至判定点

## 结果解读表（启动前预注册）

| 观测结果 | 含义 | 决定 |
|---|---|---|
| 到 step2500 三项仍改善且 CI 排除 0（≥2% 相对） | 配方仍**预算受限** | 完成授权的3000观测点；不擅自开约11k长跑 |
| 有改善但边际 < 前段一半 | **边际递减** | 按斜率外推达标所需预算，不直接长跑 |
| 三项都无 CI 排除 0 的改善 | **配方/监督信号受限**（A 合同无 goal/no-event 的预测上限已到） | 立即STOP，转条件审计 |
| OOM/非有限 loss/loader failure/盘<20GiB/本 run 产物>32GiB | 故障门 | 立即停、保证据、**不自动重启** |

来源：Hermes/机主2026-10-23 14:1x明确授权放宽硬停。无新增外审裁决。Decision为既有同配方预算斜率，不重训新模型/配方，不扩数据、不跑rollout/物理、不碰执行器。

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
