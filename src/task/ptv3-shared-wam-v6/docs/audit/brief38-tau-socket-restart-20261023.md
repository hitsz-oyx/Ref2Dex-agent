# brief38：τ路径修复与唯一训练最终结算

## 结果解读表（实测）

| 实测结果 | 含义 | 决定 |
|---|---|---|
| 2026-10-23T06:10:43.951435+08:00启动；1223 updates、last_step=1223、last_loss=0.63630688；64条loss日志均有限 | 训练真正运行；2026-10-23T07:14:46.555645+08:00停止，状态STOPPED | 唯一重启已完成；trained@A已测 |
| 新AF_UNIX错误0；returncode=0；automatic_restart=false | 未出现新的训练失败 | 不换配方、不再重启 |
| 06:45前已启动且更新>0 | “整夜未能开训”分支未触发 | 不开新分支/实验 |
| hand前8变化-36.37%，contact变化-46.37%，object前8变化-42.20% | 联合5%/5%/≤5%门：通过 | τ预测改善，执行收益未达门槛（本轮禁止rollout；执行收益未测） |
| 本次结果仅限固定TRAIN面板/本配方/本预算 | 不能由本轮预测改善推出执行或泛化改善 | 执行收益保持未测，禁止rollout |
| 无OOM/非有限loss/磁盘或产物停止门；采样最小可用129.883GiB，产物快照6518865197bytes | 故障门未触发 | 全部历史产物保留，无自动重启 |

**判读：τ预测改善，执行收益未达门槛（本轮禁止rollout；执行收益未测）。** 只在同一512锚点/124母序列、unknown goal/unknown event下比较最终checkpoint与既有init@A；不是独立heldout或执行资格。

## 配对固定面板

| 指标（mm） | init@A（复用） | trained@A | 配对Δ及母序列cluster95%CI（trained−init） | 相对变化 |
|---|---:|---:|---:|---:|
| 前8帧 hand EPE | 16.1694 | 10.2884 | -5.8809 [-6.5041, -5.2801] | -36.37% |
| 固定contact窗 hand EPE | 59.2219 | 31.7598 | -27.4621 [-31.9789, -23.3049] | -46.37% |
| 前8帧 object EPE | 11.2523 | 6.5043 | -4.7480 [-5.7424, -3.8141] | -42.20% |

同锚点SHA `440c00c8609a2e9c6276f328f1aa5136a04708294a09b5dab8321044e6caa706`；4000次按124母序列配对cluster bootstrap，锚点顺序和有效计数逐项一致。预注册绝对门为前8 hand≤15.3609、contact hand≤56.2608、前8 object≤11.8149mm。仅这一面板执行一次，无新基线；原init@A未重算。训练最终step1223，没有挑选best checkpoint。goal/event有效输入数均0，同A条件=True，首次batch重复预测bit-exact=True。

## 身份、硬停与写盘实测

| 项目 | 原值与实际值 |
|---|---|
| condition_recipe_sha256 | `36872da24a5dc72cc5580d94a048b5fd7a6ed111b21e7053a4b17acbeff9fc5d`（未变） |
| manifest_sha256 | `5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f`（未变） |
| ordered_window_tuple_sha256 | `d18ec57ee2de589152131eb2dfab4ac7581ab29275d08756c973dce2c000c3d7`（未变） |
| 窗口/seed | 6000 / 170（未变） |
| config SHA256 | `de74a83d976eca08a7d9027ecda0ae01b5219447b4dc64677fe1f395921018ad`（未变；修改字段为空） |
| hard_stop_epoch | 1792710900（07:15，未变） |
| STOP写入epoch | 1792710876.538071（硬停前23.462s） |
| 最终训练状态 | STOPPED，1223 updates，elapsed_s=3821.347736 |
| 最终checkpoint | `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/training/step001223.pt` |
| 当前训练进程 | 同配方扫描结果[]；没有第二个训练 |

训练启动提交 `a83a3df3dae5bf68ac5edbefd81fd8c27c5c57c3`；原始receipt原因 `HARD_STOP_SAVE_MARGIN_30_SECONDS`。独立supervisor PID308597、torchrun PID308607、rank308630/308631/308632，物理GPU4/5/6；本轮训练未使用0–3/7。Python3.8.20、Torch2.4.1+cu121、CUDA12.1、DDP world3；未改模型/数据/optimizer/lr/batch/workers/loader120s超时。首个更新后checkpoint确实落盘（step1 1217081377bytes），最终checkpoint列表：`[{"file": "step000000.pt", "bytes": 405716522}, {"file": "step000001.pt", "bytes": 1217081377}, {"file": "step000020.pt", "bytes": 1217082017}, {"file": "step000128.pt", "bytes": 1217082017}, {"file": "step001000.pt", "bytes": 1217082017}, {"file": "step001223.pt", "bytes": 1217082017}]`。

写盘口径：新run所有普通文件逻辑占用快照 **6518865197 bytes**（不计symlink重复，不是累计IO字节）；checkpoint合计 **6491125967 bytes**。采样最小可用空间139460603904bytes≥20GiB，最大watchdog产物5290810083bytes<32GiB；没有删除或覆盖旧r1/任何历史outputs。新run为 `tmp/t38-r2`。

实测12个resource-sharer listener均在短runtime路径，绝对85bytes、实际相对绑定47bytes，均<108；逐项证据见training-start-proof.json。最终命令 `rg -n 'AF_UNIX path too long' tmp/t38-r2/train-launch.log` 返回1、stdout空，0匹配；训练64条loss记录均finite。live-lock-proof.json证实同配方在另一output被内核flock拒绝，未创建子进程；实测同配方进程仅一个torchrun+3个DDP rank。每目录ONE_SHOT标记仍保留，全局按配方SHA互斥覆盖训练全程；不自动重启。

原panel脚本完全未改。新run副本仅TASK路径、GPU默认/授权/query/device/provenance字段改变，模型前向、数据、指标和bootstrap逐字相同；原脚本SHA `e1b988740b3a5931ed0b948c38e1bba1fc730c0242a5738a28f6abc5e47e7f9c`，副本SHA `d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51`，实际面板GPU6，推理耗时27.553s。可审阅差异（展示省略空白上下文，原始patch保留在panel-resource-only.diff）：

```diff
--- /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tools/run/evaluate_no_goal_no_event_panel.py
+++ /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel_gpu456.py
@@ -2,7 +2,7 @@
 """Same frozen TRAIN panel under A for init/trained; cluster paired inference only."""
 import argparse,hashlib,json,os,shutil,subprocess,sys,time
 from pathlib import Path
-TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
+TASK=Path('/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6');ROOT=TASK.parents[2]
 sys.path.insert(0,str(TASK/'src'))

 def read(p):return json.loads(Path(p).read_text())
@@ -20,12 +20,12 @@
     return np.percentile(result,[2.5,97.5],axis=0).T.tolist()

 def main():
-    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--gpu',type=int,default=7);p.add_argument('--batch-size',type=int,default=4);args=p.parse_args()
-    if args.gpu!=7:raise ValueError('panel only authorized on GPU7 spare share')
+    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--gpu',type=int,default=4);p.add_argument('--batch-size',type=int,default=4);args=p.parse_args()
+    if args.gpu not in (4,5,6):raise ValueError('brief38 panel only authorized on idle GPUs4/5/6')
     if args.output.exists():raise FileExistsError(args.output)
-    free=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
+    free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
     if free<8*2**30 or shutil.disk_usage('/home2').free<20*2**30:raise RuntimeError('panel resource preflight failed')
-    os.environ['CUDA_VISIBLE_DEVICES']='7';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']='2'
+    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']='2'
     import numpy as np
     import torch
     from ptv3_shared_wam_v6.data import V6Windows,collate
@@ -66,7 +66,7 @@
     report=dict(schema='ref2dex.no-goal-no-event.fixed-panel.v1',scope='TRAIN distribution only; no new rollout or executor benefit',checkpoint=str(args.checkpoint),step=step,rows=len(rows),mothers=len(np.unique(mothers)),
         metrics_order=['first8_hand','fixed_contact_hand','first8_object'],means_mm=means.tolist(),cluster_ci95_mm=ci,bootstrap_repeats=4000,paired_vs_reference=difference,
         same_A_conditions=True,goal_policy=UNKNOWN_GOAL_POLICY,event_policy=UNKNOWN_EVENT_POLICY,event_valid_count=0,goal_valid_count=0,contact_definition=panel['metrics']['contact_window'],contact_threshold=.25,
-        repeated_prediction_bit_exact=repeated,repeat_max_abs_hand_mm=repeat_max,elapsed_s=time.monotonic()-start,physical_gpu=7,free_gpu_bytes_before=free,peak_allocated_bytes=torch.cuda.max_memory_allocated(),anchor_sha256=sha(ROOT/panel['anchor']))
+        repeated_prediction_bit_exact=repeated,repeat_max_abs_hand_mm=repeat_max,elapsed_s=time.monotonic()-start,physical_gpu=args.gpu,free_gpu_bytes_before=free,peak_allocated_bytes=torch.cuda.max_memory_allocated(),anchor_sha256=sha(ROOT/panel['anchor']))
     (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

 if __name__=='__main__':main()
```

## 运行前登记及工程实施记录（保留）

## 结果解读表（06:07预登记；结算时填实测数字）

| 可能结果/当前实测 | 含义 | 决定 |
|---|---|---|
| 06:10:43启动，已3 updates、loss2.35821，07:15硬停待到时核验 | 训练真正运行 | 硬停后同512锚点trained@A面板，附复现命令 |
| 当前无新报错；已在06:45前启动 | 第二原因或时间不足 | 保留证据、停手；不换配方，不另开实验 |
| hand前8/contact均改善≥5%，object前8恶化≤5%（待测） | τ预测改善 | 仅称预测改善，执行收益未达门槛，本轮禁止rollout |
| 两层不改善或只部分改善（待测） | 本次配方+预算未通过或部分改善 | 不否证τ路线；执行收益未测 |
| OOM/非有限值/盘<20GiB/产物>32GiB（待测） | 故障门 | 立即停、不自动重启、不删除历史产物 |

来源：本轮Hermes brief38固定要求。07:15硬停epoch=1792710900；06:45启动截止。已有A合同、bit-exact及6000窗口身份审计仅引用b1fe93d9/b60e61db，不重跑。配方/数据/seed170不得变；只允许路径与按配方SHA全局互斥的最小修复。

## 修复前证据与最小差异

r1 training/result.json：FAILED_DATA_LOADER、0 updates、last_step0、141.75674s，train-launch.log存在AF_UNIX path too long。既有e37ef933通用IPC修复已在HEAD，本轮不改其模块或重复A/身份审计。初检无实际train_pretrain.py进程，GPU4/5/6各2MiB/0%，磁盘约137GiB；新run预计16GiB，保留≥20GiB，不删除失败r1。

回归先红：3 failed/1 passed，原深绝对路径真实Listener确实复现OSError，新增路径/配方锁接口尚缺；修复后5 passed。测试只检验socket/准入，不是训练读数。

逐行diff理由（原始patch见tmp/t38-r2/launch-source.diff；展示省略空白上下文）：runtime_environment把产物目录和IPC目录分开，并在启动前计算listener字节上限；configure_training_ipc仅改socket目录、TMPDIR与multiprocessing tempdir，模型/数据/优化器路径未改。recipe_guard用配方SHA命名的内核flock覆盖排队到终态，并核对同配方旧进程，跨output仍互斥。独立supervisor增加updates/有限loss/落盘字节/IPC目录/硬停倒计时，并补足32GiB停止门；06:45仍未启动明确停止。配置文件未改，改字段列表为空；配方、manifest、6000顺序身份与seed170原样保留，详见新run identity-before.json，已有checks.json仅引用。

```diff
diff --git a/src/task/ptv3-shared-wam-v6/tools/run/launch_tau_no_goal_no_event.py b/src/task/ptv3-shared-wam-v6/tools/run/launch_tau_no_goal_no_event.py
index 9c666f70..b51be01f 100644
--- a/src/task/ptv3-shared-wam-v6/tools/run/launch_tau_no_goal_no_event.py
+++ b/src/task/ptv3-shared-wam-v6/tools/run/launch_tau_no_goal_no_event.py
@@ -1,6 +1,6 @@
 #!/usr/bin/env python3
 """One-shot brief37 supervisor: admitted pool only, watchdog and wall-clock hard stop."""
-import argparse, datetime, hashlib, importlib.util, json, os, shutil, signal, subprocess, sys, time
+import argparse, contextlib, fcntl, math, datetime, hashlib, importlib.util, json, os, shutil, signal, subprocess, sys, time
 from pathlib import Path
 TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
 PYTHON=Path('/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python')
@@ -13,6 +13,47 @@ def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
 def size(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())


+def runtime_environment(out):
+    # Artifact names never enter multiprocessing's AF_UNIX address.
+    runtime=ROOT/'tmp'/('b38-'+str(os.getpid()))
+    socket_example=runtime/'tmp'/('v6i-'+str(2**31-1))/'listener-12345678'
+    if len(os.fsencode(socket_example))>=108:
+        raise ValueError('repository path too deep for AF_UNIX; use a shorter checkout')
+    (runtime/'tmp').mkdir(parents=True,exist_ok=True,mode=0o700)
+    return dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6',PYTHONDONTWRITEBYTECODE='1',
+        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(runtime/'tmp'),REF2DEX_RUN_ROOT=str(runtime))
+
+
+def active_recipe_trainings(recipe_sha):
+    active=[]
+    for entry in Path('/proc').iterdir():
+        if not entry.name.isdigit():continue
+        try:
+            words=(entry/'cmdline').read_bytes().decode().split('\0')
+            if not any(Path(w).name=='train_pretrain.py' for w in words if w):continue
+            if '--config' not in words:raise RuntimeError('unidentified train_pretrain.py process: '+entry.name)
+            config_path=Path(words[words.index('--config')+1])
+            if not config_path.is_absolute():config_path=Path(os.readlink(entry/'cwd'))/config_path
+            if read(config_path).get('condition_recipe_sha256')==recipe_sha:active.append(int(entry.name))
+        except (FileNotFoundError,ProcessLookupError):continue
+    return active
+
+
+@contextlib.contextmanager
+def recipe_guard(recipe_sha,out,lock_root=None):
+    # Kernel-held lock spans queue, child training, and terminal receipt.
+    directory=lock_root or ROOT/'tmp'/'tau-recipe-locks';directory.mkdir(parents=True,exist_ok=True)
+    fd=os.open(str(directory/(recipe_sha+'.lock')),os.O_RDWR|os.O_CREAT,0o600)
+    try:
+        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
+        except BlockingIOError:raise RuntimeError('recipe already locked: '+recipe_sha)
+        active=active_recipe_trainings(recipe_sha)
+        if active:raise RuntimeError('same recipe already running: '+str(active))
+        os.ftruncate(fd,0);os.write(fd,json.dumps(dict(recipe_sha256=recipe_sha,pid=os.getpid(),output=str(out),at=stamp())).encode())
+        yield
+    finally:os.close(fd)
+
+
 def terminal_reason(code, result, watchdog_reason=None):
     if result and result.get('status', '').startswith('FAILED'):
         return result['status']+'_NO_RESTART'
@@ -48,12 +89,20 @@ def preflight(checks,policy,config):
 def main():
     a=argparse.ArgumentParser();a.add_argument('--checks',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--queue',action='store_true');args=a.parse_args()
     out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
+    config_path=TASK/'configs/tau_no_goal_no_event_train_v1.json';config=read(config_path)
+    with recipe_guard(config['condition_recipe_sha256'],out):
+        run_once(args,out,config_path,config)
+
+
+def run_once(args,out,config_path,config):
     lock=os.open(str(out/'ONE_SHOT.lock'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o664);os.write(lock,str(os.getpid()).encode());os.close(lock)
-    config_path=TASK/'configs/tau_no_goal_no_event_train_v1.json';config=read(config_path);policy=read(config['condition_recipe']);checks=read(args.checks)
+    policy=read(config['condition_recipe']);checks=read(args.checks)
     launch=None;process=None;reason=None
     while True:
         r=preflight(checks,policy,config);write(out/'preflight.json',r)
         with (out/'preflight-history.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
+        if time.time()>=config['hard_stop_epoch']-1800:
+            write(out/'queue-status.json',dict(status='START_DEADLINE_0645',training_updates=0,at=stamp()));return
         if r['pass_all']:break
         bad=[k for k,v in r['gates'].items() if not v]
         write(out/'queue-status.json',dict(status='WAITING' if args.queue else 'BLOCKED',blocked=bad,at=stamp(),supervisor_pid=os.getpid(),training_updates=0))
@@ -65,22 +114,29 @@ def main():
          '--config',str(config_path),'--manifest',str(ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json'),
          '--stats',str(ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json'),'--output',str(training),
          '--initialize-condition-from',policy['initialization']['checkpoint'],'--workers','4','--git-commit',head,'--preserve-all-checkpoints','--checkpoint-budget-gib','32']
-    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out/'tmp'),REF2DEX_RUN_ROOT=str(out))
-    (out/'tmp').mkdir(exist_ok=True)
+    env=runtime_environment(out)
     with (out/'train-launch.log').open('x') as log:
         process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
         launch=dict(status='STARTED',at=stamp(),pid=process.pid,pgid=process.pid,supervisor_pid=os.getpid(),physical_gpus=[4,5,6],head=head,
-            policy_sha256=sha(config['condition_recipe']),config_sha256=sha(config_path),data_identity=policy['data_identity'],command=cmd,
+            runtime_root=env['REF2DEX_RUN_ROOT'],tmpdir=env['TMPDIR'],policy_sha256=sha(config['condition_recipe']),config_sha256=sha(config_path),data_identity=policy['data_identity'],command=cmd,
             initialization=policy['initialization'],estimated_duration_s=min(4800,config['hard_stop_epoch']-time.time()),estimated_bytes=r['estimated_output_bytes'],hard_stop_epoch=config['hard_stop_epoch'])
         write(out/'launch.json',launch)
         while process.poll() is None:
             free=shutil.disk_usage('/home2').free;bytes_used=size(out);now=time.time()
             alarm=bytes_used>32*2**30
+            progress=read(training/'progress.json') if (training/'progress.json').exists() else {}
+            checkpoint_bytes=sum(p.stat().st_size for p in training.glob('step*.pt'))
+            loss=progress.get('loss')
+            finite_loss=loss is None or math.isfinite(loss)
             record=dict(at=stamp(),pid=process.pid,free_bytes=free,artifact_bytes=bytes_used,disk_warning=free<30*2**30,artifact_alarm=alarm,
+                optimizer_updates=progress.get('step',0),last_loss=loss,finite_loss=finite_loss,checkpoint_bytes=checkpoint_bytes,
+                socket_directories=[str(p) for p in Path(env['TMPDIR']).glob('v6i-*')],
                 stop_reason=reason,seconds_to_hard_stop=config['hard_stop_epoch']-now)
             with (out/'watchdog.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
             write(out/'watchdog-latest.json',record)
             if free<20*2**30:reason='DISK_STOP_20_GIB'
+            elif alarm:reason='ARTIFACT_STOP_32_GIB'
+            elif not finite_loss:reason='NONFINITE_LOSS'
             elif now>=config['hard_stop_epoch']-30:reason='HARD_STOP_SAVE_MARGIN_30_SECONDS'
             if reason:
                 training.mkdir(exist_ok=True);(training/'STOP').touch(exist_ok=True)
diff --git a/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py b/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py
index 7471e7d7..2eb0d6db 100644
--- a/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py
+++ b/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py
@@ -38,6 +38,19 @@ from ptv3_shared_wam_v5.selection import trajectory_score,bootstrap_trajectory,T
 from ptv3_shared_wam_v4.objectives import normalized_components


+def configure_training_ipc():
+    # The launcher supplies a short runtime root, separate from artifacts.
+    root=Path(os.environ.get('REF2DEX_RUN_ROOT',str(TASK)))
+    ipc=root/'tmp'/('v6i-'+str(os.getpid()))
+    address=os.path.relpath(str(ipc),os.getcwd())
+    if len(os.fsencode(address+'/listener-12345678'))>=108:
+        raise ValueError('AF_UNIX loader path too long; supply a short REF2DEX_RUN_ROOT')
+    ipc.mkdir(parents=True,exist_ok=True,mode=0o700)
+    os.environ['TMPDIR']=address;tempfile.tempdir=address
+    multiprocessing.current_process()._config['tempdir']=address
+    return ipc
+
+
 def write(path,value):
     tmp=path.with_suffix('.json.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
     path.chmod(0o664)
@@ -240,8 +253,7 @@ def main():
                 num_workers=a.workers,pin_memory=device.startswith('cuda'),
                 generator=torch.Generator().manual_seed(seed*1009+rank))
     if a.workers:
-        from ptv3_shared_wam_v6.loader_ipc import configure_loader_ipc
-        configure_loader_ipc()
+        configure_training_ipc()
     kwargs.update(loader_options(a.workers,a.loader_timeout,a.loader_context))
     loader=DataLoader(**kwargs)
     dev_options={k:v for k,v in reader_options.items() if k!='curated_manifest'}
```

## 实际命令与进度

新run根目录：`tmp/t38-r2/`（项目根下，历史r1原样保留）；训练、日志、checkpoint均在此新目录；临时IPC另在项目根`tmp/b38-<supervisor_pid>/tmp/v6i-<rank_pid>`。本机路径示例listener绝对字节85<108，实际字节和目录运行后补齐。

```sh
python3 -m pytest -q src/task/ptv3-shared-wam-v6/tests/test_brief38_launch_paths.py --basetemp=tmp/t38-r2/tests-final
python3 src/task/ptv3-shared-wam-v6/tools/run/launch_tau_no_goal_no_event.py --checks src/task/ptv3-shared-wam-v6/docs/audit/brief37-no-goal-no-event-20261023-r1/checks.json --output tmp/t38-r2 --queue
```

已有panel-init-A基线不重算。原面板脚本硬编码GPU7，与本轮只用4/5/6冲突；停训后仅在新run中保存其副本，修改TASK定位与GPU资源字段为4/5/6，保持模型/数据/512锚点/124母序列/4000次cluster bootstrap全部算法逐字不变；禁止修改原脚本或执行器。执行收益保持未测。

## 06:12训练实测（三重证据，不是离线面板）

启动提交 `a83a3df3dae5bf68ac5edbefd81fd8c27c5c57c3`；06:10:43 supervisor PID308597、torchrun PID308607、rank308630/308631/308632，GPU4/5/6。`training-start-proof.json`保存pgrep进程、progress及日志grep三重证据：step3、loss2.358209827788174（有限），53.245s时steady0.2810updates/s，按当前粗估硬停前约1000次更新，最终以实测为准。三卡显存18.5–20.3GiB，利用率77–85%；不是容量或loader timeout参数变化。

12个实际resource-sharer listener均在 `tmp/b38-308597/tmp/v6i-<rank_pid>/listener-XXXXXXXX`；绝对85字节、实际相对绑定47字节，均<108。`rg -n 'AF_UNIX path too long' tmp/t38-r2/train-launch.log` 返回码1、stdout空（0匹配）；没有调高120s loader timeout。checkpoint step000000=405716522 bytes、step000001=1217081377 bytes，确实落盘；watchdog周期记录updates/last_loss/checkpoint_bytes/磁盘/总产物/倒计时。

回归本机Python 5 passed；实际训练Python3.8环境另5 passed。配置SHA256仍 `de74a83d976eca08a7d9027ecda0ae01b5219447b4dc64677fe1f395921018ad`，配置改动字段为空；配方SHA、manifestSHA、ordered6000 SHA与seed170见identity-before.json，保持brief38原值。既有审计未重做。

停后任务 `tmp/t38-r2/poststop-panel.py` 已独立登记PID311661，只等待stop-receipt；新训练失败则不做面板、不重启。正常停训且updates>0后，使用最终保存checkpoint和旧panel-init-A，在空闲GPU4/5/6做一次trained@A；原GPU7专用脚本只读，副本仅TASK路径与GPU资源/来源字段不同，diff存panel-resource-only.diff。训练硬停不受面板任务影响。

Git：首次 `git pull --rebase origin main` 被其他会话未暂存修改拒绝（`cannot pull with rebase: You have unstaged changes`）；未stash、未改保护文件，显式四路径提交后普通push成功。06:10核对本地与远端均为a83a3df3dae5bf68ac5edbefd81fd8c27c5c57c3；终态再核对。

## 确切复现命令（历史运行记录，不自动重跑）

同配方、同配置文件；以下第一条为本次启动命令，原输出目录已有ONE_SHOT，不能重复覆盖，且06:45启动截止/07:15硬停保持有效。第二条为本次唯一trained@A调用，原输出目录也不能覆盖。

```sh
/usr/bin/python3 src/task/ptv3-shared-wam-v6/tools/run/launch_tau_no_goal_no_event.py --checks src/task/ptv3-shared-wam-v6/docs/audit/brief37-no-goal-no-event-20261023-r1/checks.json --output tmp/t38-r2 --queue
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel_gpu456.py --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/training/step001223.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel-trained-A --reference /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/panel-init-A --gpu 6 --batch-size 4
```

终态核对：07:15前已退出，最终checkpoint1223和唯一trained@A均已完成；当前同配方训练进程为空。末次资源快照已计入最终checkpoint及面板产物，采样时间见delivery-summary.json的snapshot_epoch。原始stop-receipt记录产物6507895808bytes（面板执行前）；最终新run文件快照为6518865197bytes。路径回归5项在系统Python与实际训练Python3.8环境均通过；未重做A合同/6000监督审计，未运行rollout、未修改执行器。
