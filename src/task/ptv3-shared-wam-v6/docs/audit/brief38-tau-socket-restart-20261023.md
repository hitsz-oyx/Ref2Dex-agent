# brief38：τ训练socket路径修复、唯一重启与固定面板

## 结果解读表（06:07预登记；结算时填实测数字）

| 可能结果/当前实测 | 含义 | 决定 |
|---|---|---|
| 启动且updates>0、有限loss，跑到07:15（待测） | 训练真正运行 | 硬停后同512锚点trained@A面板，附复现命令 |
| 新报错或06:45仍未启动（待测） | 第二原因或时间不足 | 保留证据、停手；不换配方，不另开实验 |
| hand前8/contact均改善≥5%，object前8恶化≤5%（待测） | τ预测改善 | 仅称预测改善，执行收益未达门槛，本轮禁止rollout |
| 两层不改善或只部分改善（待测） | 本次配方+预算未通过或部分改善 | 不否证τ路线；执行收益未测 |
| OOM/非有限值/盘<20GiB/产物>32GiB（待测） | 故障门 | 立即停、不自动重启、不删除历史产物 |

来源：本轮Hermes brief38固定要求。07:15硬停epoch=1792710900；06:45启动截止。已有A合同、bit-exact及6000窗口身份审计仅引用b1fe93d9/b60e61db，不重跑。配方/数据/seed170不得变；只允许路径与按配方SHA全局互斥的最小修复。

## 修复前证据与最小差异

r1 training/result.json：FAILED_DATA_LOADER、0 updates、last_step0、141.75674s，train-launch.log存在AF_UNIX path too long。既有e37ef933通用IPC修复已在HEAD，本轮不改其模块或重复A/身份审计。初检无实际train_pretrain.py进程，GPU4/5/6各2MiB/0%，磁盘约137GiB；新run预计16GiB，保留≥20GiB，不删除失败r1。

回归先红：3 failed/1 passed，原深绝对路径真实Listener确实复现OSError，新增路径/配方锁接口尚缺；修复后5 passed。测试只检验socket/准入，不是训练读数。

逐行diff理由：runtime_environment把产物目录和IPC目录分开，并在启动前计算listener字节上限；configure_training_ipc仅改socket目录、TMPDIR与multiprocessing tempdir，模型/数据/优化器路径未改。recipe_guard用配方SHA命名的内核flock覆盖排队到终态，并核对同配方旧进程，跨output仍互斥。独立supervisor增加updates/有限loss/落盘字节/IPC目录/硬停倒计时，并补足32GiB停止门；06:45仍未启动明确停止。配置文件未改，改字段列表为空；配方、manifest、6000顺序身份与seed170原样保留，详见新run identity-before.json，已有checks.json仅引用。

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
