"""Training-only success/transient support decision; no neural computation."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();source=args.source.resolve();out=args.output.resolve();assert ROOT in out.parents and not out.exists();begin=time.monotonic();torch.set_num_threads(2)
    protected={}
    for seed in (603,604):
        for name in ('initial.pt','trace.pt','rows.json','results.json','panel_audit.json','physical_metadata.json'):
            p=source/f's{seed}'/name;protected[str(p)]=sha(p)
    for p in (Path(__file__).resolve(),ROOT/'docs/decisions/D-20261002-late-option-support-review.md'):protected[str(p)]=sha(p)
    out.mkdir();m=dict(experiment_id='P-20261002-late-option-support',run_id=out.name,run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source=str(source),input_sha256=protected,execution_device='cpu',device_reason='pure file/label statistics without neural inference',wall_limit_seconds=60,storage_limit_bytes=16<<20)
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    save();parts=[]
    try:
        for seed in (603,604):
            d=source/f's{seed}';assert json.loads((d/'results.json').read_text())['statistical_option_collection'];assert json.loads((d/'panel_audit.json').read_text())['run_status']=='COMPLETED'
            initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);rows=json.loads((d/'rows.json').read_text());assert len(rows)==768 and [row['environment'] for row in rows]==list(range(768))
            motion=initial['motion'].numpy();arm=initial['policy_group'].numpy();stop=initial['phase_stop'].numpy()[motion];progress=trace['progress'].numpy();window=(progress>=stop[None]-74)&(progress<=stop[None]+30);assert np.all(window.sum(0)==105)
            valid=(trace['object_root'].numpy()[...,2]-initial['initial_height'].numpy()[None]>=np.float32(.03))&(trace['clearance'].numpy()>=np.float32(.02));success=~(window&~valid).any(0);transient=(window&valid).any(0)
            assert np.array_equal(success,np.array([r['physical105'] for r in rows],bool));assert np.array_equal(motion,np.array([r['motion'] for r in rows])) and np.array_equal(arm,np.array([r['arm'] for r in rows]))
            domain=np.all(np.abs(initial['option_raw12'].numpy())<=1,axis=1)
            for i in range(768):parts.append(dict(seed=seed,environment=i,motion=int(motion[i]),arm=int(arm[i]),physical105=bool(success[i]),any_joint_lift_in105=bool(transient[i]),within_bounded_actor_raw_domain=bool(domain[i])))
        table={}
        for motion in range(3):
            table[str(motion)]={}
            for name,arms in [('p0',(0,)),('random_options',(1,2,3)),*[(f'arm{k}',(k,)) for k in range(4)]]:
                rows=[r for r in parts if r['motion']==motion and r['arm'] in arms];table[str(motion)][name]=dict(episodes=len(rows),physical105=sum(r['physical105'] for r in rows),any_joint_lift_in105=sum(r['any_joint_lift_in105'] for r in rows),within_actor_domain=sum(r['within_bounded_actor_raw_domain'] for r in rows),within_actor_domain_physical105=sum(r['physical105'] and r['within_bounded_actor_raw_domain'] for r in rows))
        observed=table['0']['random_options'];route='TASK_LEARNING_WITH_POSITIVE_SUPPORT' if observed['physical105']>=8 else ('RETENTION_CONTROL' if observed['any_joint_lift_in105']>=8 else 'EARLIER_CONTACT_ACQUISITION')
        result=dict(run_status='COMPLETED',source_fit_only=True,reused_episodes=len(parts),no_new_physics_or_neural_calls=True,all105_labels_rebuilt_exact=True,counts=table,route_screen_motion0=route,frozen_support_threshold=8,no_universal_infeasibility_or_counterfactual_claim=True,new_optimizer_steps=0)
        (out/'rows.json').write_text(json.dumps(parts)+'\n');(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        for p,h in protected.items():assert sha(p)==h
        m.update(run_status='COMPLETED',inputs_unchanged=True,bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()));assert time.monotonic()-begin<=60 and m['bytes']<=16<<20;print(json.dumps(result),flush=True)
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
