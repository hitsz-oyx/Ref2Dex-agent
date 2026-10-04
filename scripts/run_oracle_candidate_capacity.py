"""Stable frozen-Q ordering and observed finite-bank capacity, with no refit."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_contact_response_probe import sha, admission, PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child, bytes_in
from scripts.run_oracle_replay_engineering import CONFIG, REFERENCES, POLICY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    out, source = args.output.resolve(), args.source.resolve()
    if out.exists() or ROOT not in out.parents or ROOT not in source.parents:
        raise ValueError('unique owned output and source')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):
        raise ValueError('clean fixed source')
    previous = json.loads((source/'run_manifest.json').read_text())
    audit = json.loads((source/'protocol-audit-final.json').read_text())
    if previous['run_status'] != 'COMPLETED' or audit['run_status'] != 'COMPLETED':
        raise ValueError('qualified terminal source')
    frozen = Path(previous['source_frozen_selectors'])
    protected = [CONFIG,REFERENCES,POLICY,Path(__file__).resolve(),
                 ROOT/'scripts/run_oracle_native_replay.py',ROOT/'src/task/CmResidual/oracle_native.py',
                 ROOT/'src/task/CmResidual/oracle_features.py',ROOT/'scripts/select_single_focal_oracle.py',
                 ROOT/'scripts/analyze_single_focal_oracle.py',ROOT/'scripts/analyze_oracle_candidate_capacity.py',
                 ROOT/'docs/archive/2026-10-04-research-governance/decisions/D-20261003-oracle-candidate-capacity.md',
                 frozen/'fit/selector.pt',frozen/'options.npy',source/'run_manifest.json',
                 source/'results.json',source/'selection/feature_packet.pt',source/'protocol-audit-final.json']
    generation=json.loads(REFERENCES.read_text())
    for reference in generation['references']:
        path=Path(reference['generated'])
        if sha(path)!=reference['generated_sha256']:
            raise ValueError('fixed generated reference identity')
        protected.append(path)
    protected.extend(ROOT/'src/task/CmResidual'/name for name in (
        'native_reset_transaction.py','dexplore_bc_policy.py','observation_hold_policy.py',
        'reference_target_policy.py','static_hold_feasibility.py','continuous_critic_cm.py',
        'selective_finger_response.py','finger_preload.py','tabletop_clearance.py'))
    panels = [source/'baseline']+sorted(p.parent for p in source.glob('queries/*/results.json'))+sorted(p.parent for p in source.glob('deploy/*/results.json'))
    for panel in panels:
        protected.extend(panel/name for name in ('trace.pt','initial.pt','contacts.npy','physics_states.pt',
                                                'physical_metadata.json','contact_frames.json','results.json','native_audit.json'))
    protected.extend(sorted((source/'requests').glob('*.npy')))
    hashes = {str(path):sha(path) for path in protected}
    def fingerprint(path):
        stat=Path(path).stat()
        return (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino)
    fingerprints={path:fingerprint(path) for path in hashes}
    out.mkdir()
    (out/'baseline').symlink_to(source/'baseline',target_is_directory=True)
    (out/'requests').symlink_to(source/'requests',target_is_directory=True)
    (out/'queries').mkdir();(out/'deploy').mkdir();(out/'capacity').mkdir()
    for panel in sorted((source/'queries').iterdir()):
        (out/'queries'/panel.name).symlink_to(panel,target_is_directory=True)
    begin = time.monotonic()
    record = dict(run_status='RUNNING',experiment_id='P-20261003-oracle-candidate-capacity',
                  run_id=out.name,pid=os.getpid(),phases=[],input_sha256=hashes,
                  source_frozen_selectors=str(frozen),inherited_oracle_run=str(source),
                  actual_new_optimizer_steps=0,inherited_optimizer_steps=6000,
                  inherited_short_query_worlds=84,inherited_actual_worlds=27,new_native_worlds=0,
                  wall_limit_seconds=900,storage_limit_bytes=2<<30,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        record['wall_seconds']=time.monotonic()-begin
        temporary=out/'run_manifest.json.tmp'
        temporary.write_text(json.dumps(record,indent=2)+'\n')
        temporary.replace(out/'run_manifest.json')
    def verify():
        if any(fingerprint(path) != stat for path,stat in fingerprints.items()):
            raise RuntimeError('protected input drift')
    def run(name,command,timeout=270):
        verify();gpu=admission(6)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                 OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',MAX_JOBS='2',
                 TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=command,admission=gpu,run_status='RUNNING')
        record['phases'].append(phase);save()
        def spawned(child):
            phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>900 or bytes_in(out)>2<<30:
                raise RuntimeError('capacity budget')
            rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            for row in rows.splitlines():
                fields=[field.strip() for field in row.split(',')]
                if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):
                    raise RuntimeError('GPU contention')
        print(json.dumps(dict(phase=name,status='STARTED')),flush=True)
        try:
            run_owned_child(command,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,timeout,spawned)
            verify();phase['run_status']='COMPLETED'
        except BaseException as error:
            phase.update(run_status='FAILED',error=repr(error));raise
        finally:save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    def native(path,name):
        return [PYTHON,'-u',str(ROOT/'scripts/run_oracle_native_replay.py'),'--output',str(path),
                '--env-config',str(CONFIG),'--references-manifest',str(REFERENCES),
                '--policy-checkpoint',str(POLICY),'--seed','763','--envs','12','--ticks','202',
                '--decision','36','--oracle-horizon','32','--contact-window-only','--retain-pd-target',
                '--source',str(out/'baseline'),'--options',str(out/'requests'/(name+'.npy'))]
    save()
    try:
        run('select',[PYTHON,'-u',str(ROOT/'scripts/select_single_focal_oracle.py'),
                      '--source',str(out),'--models',str(frozen/'fit/selector.pt'),
                      '--feature-packet',str(source/'selection/feature_packet.pt')],90)
        choices=json.loads((out/'selection/choices.json').read_text())
        unique=sorted({(target,int(index)) for arm in choices.values() for target,index in enumerate(arm) if index})
        for target,index in unique:
            name='e%02d_c%02d'%(target,index);prior=source/'deploy'/name
            if prior.exists():
                (out/'deploy'/name).symlink_to(prior,target_is_directory=True)
            else:
                run('deploy_'+name,native(out/'deploy'/name,name));record['new_native_worlds']+=1;save()
        run('analyze',[PYTHON,'-u',str(ROOT/'scripts/analyze_single_focal_oracle.py'),
                       '--source',str(out),'--models',str(frozen/'fit/selector.pt')],120)
        import torch
        initial=torch.load(out/'baseline/initial.pt',map_location='cpu',weights_only=False)
        for target in (initial['motion']==0).nonzero().flatten().tolist():
            for index in range(1,8):
                name='e%02d_c%02d'%(target,index)
                if (out/'deploy'/name).exists():continue
                prior=source/'deploy'/name
                if prior.exists():
                    (out/'capacity'/name).symlink_to(prior,target_is_directory=True)
                else:
                    run('capacity_'+name,native(out/'capacity'/name,name));record['new_native_worlds']+=1;save()
        run('capacity_analyze',[PYTHON,'-u',str(ROOT/'scripts/analyze_oracle_candidate_capacity.py'),
                                '--source',str(out)],120)
        if any(sha(Path(path)) != digest for path,digest in hashes.items()):
            raise RuntimeError('terminal protected SHA audit')
        record.update(run_status='COMPLETED',inputs_unchanged=True,bytes=bytes_in(out))
    except BaseException as error:
        record.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
