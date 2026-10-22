"""Bounded handoff executor: finish v10 evidence, frozen coverage, BC, residual pilot.

All children/logs live in the project. Stop on any failed contract; never silently
resample an episode. Launch once with --detached from a committed working tree.
"""
import argparse,json,subprocess,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];TOOLS=ROOT/'src/task/wam-rl-policy-v3/tools/run'
PYTHON=sys.executable


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--detached',action='store_true');a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents:raise ValueError('fresh task output required')
    if a.detached:
        command=[PYTHON,str(Path(__file__).resolve()),'--output',str(out)]
        with out.parent.joinpath(out.name+'-supervisor.log').open('x') as log:child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        print(json.dumps(dict(pid=child.pid,command=command,log=str(log.name))));return
    out.mkdir();receipt=dict(status='RUNNING',stage='WAIT_V10',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),start=time.time(),commands=[],maximum_wall_s=14400)
    def save():
        f=out/'manifest.json';new=f.with_suffix('.new');new.write_text(json.dumps(receipt,indent=2)+'\n');new.replace(f)
    def call(command,log,timeout=7200):
        receipt['commands'].append(dict(command=command,log=str(log)));save()
        with log.open('x') as handle:subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,timeout=timeout,check=True)
    def worker(reg,folder,gpu=4,worker_id=0,extra=None):
        call([PYTHON,'-u',str(TOOLS/'run_grasp_geometry_worker.py'),'--registration',str(reg),'--worker',str(worker_id),'--gpu',str(gpu),'--output',str(folder)]+(extra or []),folder.parent/(folder.name+'.log'),timeout=3700)
    save()
    try:
        v10=ROOT/'outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1'
        while True:
            supervisor=json.loads((v10/'resume2-supervisor.json').read_text())
            if supervisor['status']=='FAILED':raise RuntimeError('v10 supervisor failed; no automatic resampling')
            if supervisor['status']=='COMPLETED':break
            if time.time()-receipt['start']>7200:raise TimeoutError('v10 wait budget')
            time.sleep(30)
        receipt['stage']='VERIFY_V10';save()
        manifests=[v10/('g%d-resume2'%g)/'manifest.json' for g in (4,5,6)]
        flags=[x for m in manifests for x in ('--manifest',str(m))]
        call([PYTHON,str(TOOLS/'summarize_large_grasp.py'),'--registration',str(v10/'registration.json'),*flags,'--output',str(v10/'summary-final.json')],out/'v10-summary.log')
        call([PYTHON,str(TOOLS/'verify_large_initials.py'),'--arms','A1,A2',*flags,'--output',str(v10/'initial-parity-final.json')],out/'v10-initials.log')
        call([PYTHON,str(TOOLS/'verify_grasp_ablation.py'),*flags,'--output',str(v10/'ablation-final.json')],out/'v10-ablation.log')
        for g in (4,5,6):call([PYTHON,str(TOOLS/'rescore_large_grasp.py'),'--worker',str(v10/('g%d-resume2'%g)),'--output',str(v10/('g%d-final-rescore.json'%g)),'--verify-mutant'],out/('v10-g%d-rescore.log'%g))
        summary=json.loads((v10/'summary-final.json').read_text());overcharge=0;native_steps=[]
        for g in (4,5,6):
            old=v10/('g%d-resume'%g);m=json.loads((old/'manifest.json').read_text());native=json.loads((old/m['active_episode']['id']/'physics/manifest.json').read_text());steps=int(native['steps']);native_steps.append(steps);overcharge+=600-steps
        summary['environment_controls_conservative_upper']=summary['environment_controls'];summary['environment_controls']-=overcharge
        summary['interrupted_native_steps']=native_steps;summary['interrupted_controls']=sum(native_steps);summary['budget_reconciliation']='worker carry receipts reserve 600 each; final total uses exact native exit manifests; completed results unchanged'
        (v10/'summary-final.json').write_text(json.dumps(summary,indent=2)+'\n')
        receipt['stage']='COVERAGE';save();coverage=ROOT/'outputs/wam-rl-policy-v3/grasp-v11-coverage-20261022-r1';children=[];logs=[]
        for i,g in enumerate((4,5,6)):
            command=[PYTHON,'-u',str(TOOLS/'run_grasp_geometry_worker.py'),'--registration',str(coverage/'registration.json'),'--worker',str(i),'--gpu',str(g),'--output',str(coverage/('g%d'%g))]
            log=(coverage/('g%d.log'%g)).open('x');logs.append(log);children.append(subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True));receipt['commands'].append(dict(command=command,log=str(log.name)))
        save();started=time.time()
        while any(c.poll() is None for c in children):
            if time.time()-started>7500:
                for c in children:
                    if c.poll() is None:c.terminate()
                raise TimeoutError('coverage wall budget')
            time.sleep(30)
        for log in logs:log.close()
        if any(c.returncode for c in children):raise RuntimeError('coverage worker failed; preserve partials')
        call([PYTHON,str(TOOLS/'summarize_grasp_coverage.py'),'--run',str(coverage),'--output',str(coverage/'summary.json')],out/'coverage-summary.log')
        for g in (4,5,6):call([PYTHON,str(TOOLS/'rescore_large_grasp.py'),'--worker',str(coverage/('g%d'%g)),'--output',str(coverage/('g%d-rescore.json'%g)),'--verify-mutant'],out/('coverage-g%d-rescore.log'%g))
        receipt['stage']='BC';save();bc=out/'bc'
        call([PYTHON,str(TOOLS/'train_drift_intent_bc.py'),'--dataset',str(ROOT/'outputs/wam-rl-policy-v3/drift-intent-bc-data-20261022-r1/dataset.npz'),'--output',str(bc),'--gpu','4'],out/'bc.log',timeout=360)
        # Use two already-developed initial states; these are NOT coverage holdouts.
        original=json.loads((v10/'registration.json').read_text());cases=[next(j for j in original['jobs'] if j['subset']=='seen'),next(j for j in original['jobs'] if j['sequence']=='s10_alarmclock_pass_1')]
        jobs=[dict(j,id='p%03d'%i,worker=0,order=['A1'],native_seed=294+i,model_seed=293,subset='dev') for i,j in enumerate(cases)]
        protocol=dict(protocol='grasp.geometry-se3.intent-residual.dev2.v1',experiment_id='P-20261022-drift-intent-residual',jobs=jobs,arms=['A1'],subsets=['dev'],horizon_controls=600,control_hz=30,worker_wall_s=3600,maximum_controls=1200,ppo_authorized=True,scope='two developed initials, same-seed before/after feasibility only; no efficacy or generalization claim',primary='full20s geometry hold and exact object goal, no safety abort')
        reg=out/'registration.json';reg.write_text(json.dumps(protocol,indent=2)+'\n')
        receipt['stage']='BC_EVAL';save();worker(reg,out/'before',extra=['--residual-checkpoint',str(bc/'bc.pt')])
        receipt['stage']='PPO';save();worker(reg,out/'ppo',extra=['--residual-checkpoint',str(bc/'bc.pt'),'--ppo'])
        last=out/'ppo/residual-p001-A1.pt'
        receipt['stage']='PPO_EVAL';save();worker(reg,out/'after',extra=['--residual-checkpoint',str(last)])
        for name in ('before','ppo','after'):
            call([PYTHON,str(TOOLS/'rescore_large_grasp.py'),'--worker',str(out/name),'--output',str(out/(name+'-rescore.json')),'--verify-mutant'],out/(name+'-rescore.log'))
        receipt.update(status='COMPLETED',stage='DONE',elapsed_s=time.time()-receipt['start']);save()
    except BaseException as error:
        receipt.update(status='FAILED',error=repr(error),elapsed_s=time.time()-receipt['start']);save();raise

if __name__=='__main__':main()
