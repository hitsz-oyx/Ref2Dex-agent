"""Resume must preserve the original scientific plan and reject ambiguous state."""
import json,os,sys,threading,time
import pytest
from scripts.resume_continuous_critic_policy import resume_order,rebound_native_command,verify_stop_receipt,assert_source_stable,run_owned_child,EXPECTED_GPU,sha

def test_prefix_is_not_recollected_and_all_original_remaining_updates_are_present():
    order=resume_order()
    assert order[0]==('update',2,548)
    assert [u for k,u,s in order if k=='update']==list(range(2,21))
    assert [s for k,u,s in order if k=='collect']==list(range(549,567))
    assert [(u,s) for k,u,s in order if k=='evaluate']==[(20,568),(20,569)]
    assert order[-1]==('analyze',20,None)

def test_native_rebinding_preserves_template_and_final_only_evaluation(tmp_path):
    heads=tmp_path/'policy_heads.pt';heads.write_bytes(b'unit-test-only packet, no physical model')
    cmd=['python','-u','old/script','--run-dir','old/native','--eval-seed','548','--seed','548','--output','old/result','--output_path','old/player','--continuous-checkpoint','old/u01','--continuous-sha256','oldsha','--expected-update','1','--num_envs','768']
    untouched=list(cmd)
    for seed,update,det in ((549,'2',False),(566,'19',False),(568,'20',True),(569,'20',True)):
        rebound=rebound_native_command(cmd,tmp_path,seed,heads)
        assert rebound[rebound.index('--expected-update')+1]==update
        assert rebound[rebound.index('--eval-seed')+1]==str(seed)
        assert rebound[rebound.index('--continuous-sha256')+1]==sha(heads)
        assert rebound[rebound.index('--num_envs')+1]=='768'
        assert ('--deterministic' in rebound)==det
        assert str(tmp_path/f's{seed}')==rebound[rebound.index('--run-dir')+1]
    assert cmd==untouched
    with pytest.raises(ValueError):rebound_native_command(cmd,tmp_path,548,heads)

def test_stop_receipt_requires_same_host_exact_source_gpu_and_absent_owned_pids(tmp_path):
    source=tmp_path/'source';source.mkdir();m=dict(pid=1810163,phases=[dict(pid=1813639,run_status='RUNNING')])
    (source/'run_manifest.json').write_text(json.dumps(m))
    receipt=tmp_path/'receipt.json';r=dict(source_manifest_sha256=sha(source/'run_manifest.json'),original_parent_pid=1810163,gpu_uuid=EXPECTED_GPU,host_boot_id='test-boot',pid_namespace='pid:[test-only]',observed_on_original_host=True,original_process_stopped=True)
    receipt.write_text(json.dumps(r));gpu=dict(uuid=EXPECTED_GPU)
    assert verify_stop_receipt(receipt,source,m,gpu,'test-boot','pid:[test-only]',lambda p:False)==r
    for key,val in (('observed_on_original_host',False),('original_process_stopped',False),('host_boot_id','other'),('pid_namespace','other'),('source_manifest_sha256','wrong')):
        bad={**r,key:val};receipt.write_text(json.dumps(bad))
        with pytest.raises(ValueError):verify_stop_receipt(receipt,source,m,gpu,'test-boot','pid:[test-only]',lambda p:False)
    receipt.write_text(json.dumps(r))
    with pytest.raises(RuntimeError):verify_stop_receipt(receipt,source,m,gpu,'test-boot','pid:[test-only]',lambda p:p==1813639)
    pinned=sha(source/'run_manifest.json');assert_source_stable(source,pinned)
    (source/'run_manifest.json').write_text(json.dumps({**m,'run_status':'COMPLETED'}))
    with pytest.raises(RuntimeError):assert_source_stable(source,pinned)

def test_source_progress_stops_only_the_new_owned_process_group(tmp_path):
    source=tmp_path/'source';source.mkdir();manifest=source/'run_manifest.json';manifest.write_text('{"run_status":"RUNNING"}')
    pinned=sha(manifest);observed={};started=threading.Event()
    def advance_source():
        assert started.wait(5);time.sleep(.15);manifest.write_text('{"run_status":"COMPLETED"}')
    thread=threading.Thread(target=advance_source,daemon=True);thread.start()
    def spawned(child):observed['child']=child;started.set()
    begin=time.monotonic()
    with pytest.raises(RuntimeError,match='original source progressed'):
        run_owned_child([sys.executable,'-c','import time; time.sleep(60)'],tmp_path,dict(os.environ),tmp_path/'owned.log',lambda:assert_source_stable(source,pinned),8,spawned)
    thread.join(timeout=1)
    assert observed['child'].poll() is not None
    assert time.monotonic()-begin<8
    assert json.loads(manifest.read_text())['run_status']=='COMPLETED'
