"""Screen the retained uncontaminated4k horizon checkpoints on a fixed native panel."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np
    out=a.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v2' not in out.parents:raise ValueError('Fresh Task output required')
    arms=[('masked8','continuation-masked8','masked',8,1),('delta08','continuation-delta08','delta0',8,4),
          ('masked1','single-masked1','masked',1,5),('delta01','single-delta01','delta0',1,6)]
    for *_,gpu in arms:
        if subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Assigned GPU occupied')
    base=ROOT/'outputs/ptv3-shared-wam-v2/shared-action-horizon-20261016-r1'
    initial=ROOT/'outputs/ptv3-shared-wam-v2/shared-current-continuation-20261015-r2'
    runs=[base/name for _,name,*_ in arms]
    for field in ('rollout_admission.json','encoder_workspace_admission.json'):
        expected=json.loads((runs[0]/field).read_text())
        assert all(json.loads((r/field).read_text())==expected for r in runs)
    expected_dev=np.load(runs[0]/'validation.npy')
    assert all(np.array_equal(np.load(r/'validation.npy'),expected_dev) for r in runs)
    expected_actor=None
    expected_config=None
    contract_fields=('schema','goal_encoding','task_conditioning','shared_width','shared_heads','shared_layers',
        'point_width','ptv3','object_points','horizon','voxel_m','voxel_origin_m','action_representation',
        'geometry_lr','shared_lr','robot_lr','weight_decay','clip_grad','amp','tf32','hand_weight','scene_weight',
        'action_weight','anchor_weight','batch_size','seed','training_rng_seed','validation_seed')
    for key,name,mode,steps,_ in arms:
        r=base/name;m=json.loads((r/'manifest.json').read_text());c=m['config']
        assert c['current_observation_input']==mode and c['seed']==c['training_rng_seed']==198
        assert c['batch_size']==256 and m['provenance']['world_size']==1
        assert c.get('action_supervision_steps',8)==steps
        core={k:c.get(k) for k in contract_fields}
        if expected_config is None:expected_config=core
        assert core==expected_config
        start_manifest=r/'manifest.json' if steps==1 else initial/('train-'+mode+'-r2')/'manifest.json'
        origin=json.loads(start_manifest.read_text())['provenance']
        assert Path(origin['weight_only_robot_parent']).resolve()==ROOT/'outputs/ptv3-shared-wam-v2/shared-native-grasp-20261015-r1/inputs/control-step7000.pt'
        p=r/'initial-actor.npz' if steps==1 else initial/('train-'+mode+'-r2')/'initial-actor.npz'
        with np.load(p) as z:actor={k:z[k].copy() for k in z.files}
        if expected_actor is None:expected_actor=actor
        assert actor.keys()==expected_actor.keys() and all(np.array_equal(v,expected_actor[k]) for k,v in actor.items())
        assert (r/'step4000.pt').is_file()
    out.mkdir(parents=True);started=time.monotonic()
    status=dict(status='RUNNING',pid=os.getpid(),training=False,checkpoint_step=4000,
        matching=dict(initial_actor=True,TRAIN_current_normalizers=True,admission=True,DEV=True,seed_and_batch=True),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip())
    def write():
        temp=out/'process.json.part';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(out/'process.json')
    write()
    # This already-qualified bridge includes own current721 transport; action loss is not run at inference.
    code=ROOT/'tmp/shared-current-code-20261015-r2'
    status['bridge_code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip();write()
    legacy=code/'src/task/ptv3-shared-wam'
    card=code/'src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-shared-current-continuation.md'
    env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES='',TMPDIR=str(ROOT/'tmp'),PYTHONDONTWRITEBYTECODE='1')
    def screen(item):
        key,name,mode,steps,gpu=item;rows=[]
        for label,sequence,controls in [('airplane','s10_airplane_pass_1',149),('sphere','s4_spheresmall_pass_1',141),('mug','s10_mug_pass_1',172)]:
            target=out/(key+'-'+label)
            cmd=[sys.executable,str(legacy/'tools/run/launch_native.py'),'--model-version','v2','--shared-actor',
                '--checkpoint',str(base/name/'step4000.pt'),'--output',str(target),'--gpu',str(gpu),
                '--controls',str(controls),'--execution-steps','1','--seed','148','--sequence',sequence,'--experiment-card',str(card)]
            row=dict(arm=key,sequence=sequence,action_supervision_steps=steps,checkpoint=str(base/name/'step4000.pt'),command=cmd)
            with (out/(key+'-'+label+'.log')).open('x') as log:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=660)
                if result.returncode:
                    errors=[]
                    for peer in ('server','native'):
                        p=target/(peer+'.log')
                        if p.exists():errors += [line for line in p.read_text(errors='replace').splitlines() if 'Error:' in line]
                    row.update(status='WORKSPACE_EXIT' if any('outside fixed PTv3 workspace' in s for s in errors) else 'EXECUTION_FAILED',errors=errors[-4:])
                else:
                    audit=subprocess.run([sys.executable,str(legacy/'tools/audit/audit_native.py'),'--native',str(target/'native')],
                        env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                    row.update(status='COMPLETED' if audit.returncode==0 else 'AUDIT_FAILED',
                        result=json.loads((target/'native/result.json').read_text()))
            rows.append(row)
            (out/(key+'-partial.json')).write_text(json.dumps(rows,indent=2)+'\n')
        return key,rows
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:results=dict(pool.map(screen,arms))
        (out/'comparison.json').write_text(json.dumps(dict(status='COMPLETED',matching=status['matching'],results=results,
            scope='Matched4k prefix native Probe; no formal success-rate claim; workspace exits remain separate'),indent=2)+'\n')
        status['status']='COMPLETED'
    except BaseException as error:status.update(status='FAILED',error=repr(error));raise
    finally:status['elapsed_s']=time.monotonic()-started;write()


if __name__=='__main__':main()
