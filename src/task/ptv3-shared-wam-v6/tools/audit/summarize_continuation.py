"""Inspect completed real checkpoint counters, DEV and baseline restoration."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('code-root','run','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    sys.path.insert(0,str(a.code_root/'src/task/ptv3-shared-wam-v6/src'))
    from ptv3_shared_wam_v6 import paths
    from ptv3_shared_wam_v6.model import SharedWAMV6
    from ptv3_shared_wam_v6.optimizer_expansion import _group_names
    from ptv3_shared_wam_v5.storage import read_json
    from ptv3_shared_wam_v5.training_contract import validate_resume
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    report=read_json(a.run/'report.json')
    if report['status']!='COMPLETED' or not report['v5_restore'].get('verified'):
        raise ValueError('completed v6 probe plus verified original v5 restoration required')
    final=Path(report['v6_final_checkpoint']);final.chmod(0o664)
    saved=torch.load(final,map_location='cpu',weights_only=False)
    manifest=read_json(a.run/'resume-recovery/run-manifest.json')
    recipe=manifest['event_recipe'];stats=read_json(recipe['stats'])
    model=SharedWAMV6(stats,saved['config'])
    if any(not torch.equal(value,saved['model'][name]) for name,value in model.state_dict().items()
           if name.startswith('normalizer.')):
        raise ValueError('original horizon normalization buffers changed')
    groups=_group_names(model,saved['config']);counters={'original':set(),'event':set()};state_count={'original':0,'event':0}
    for (_,names),group in zip(groups,saved['optimizer']['param_groups']):
        if len(names)!=len(group['params']):raise ValueError('optimizer position/name mismatch')
        for name,identifier in zip(names,group['params']):
            state=saved['optimizer']['state'].get(identifier)
            if state is None:raise ValueError('trained parameter is missing optimizer state: '+name)
            kind='event' if name.startswith('event_goal.') else 'original'
            counters[kind].add(float(state['step']));state_count[kind]+=1
    expected_new=saved['step']-report['baseline_step']
    if counters['original']!={float(saved['step'])} or counters['event']!={float(expected_new)}:
        raise ValueError('old/new optimizer age was reset or mismatched')
    if len(saved['random_state'])!=4 or any(not all(k in r for k in ('python','numpy','torch','cuda')) for r in saved['random_state']):
        raise ValueError('four complete per-rank RNG payloads required')
    binding=validate_resume(saved,saved['config'],manifest['data_identity'],4)
    if saved['migration_binding']['parent_checkpoint']!=report['baseline_checkpoint']:
        raise ValueError('original v5 migration provenance lost')
    restore=Path(report['restore_output']);restored_manifest=read_json(restore/'run-manifest.json')
    restored_progress=read_json(restore/'progress.json')
    if (restored_manifest['schema']!='ref2dex.ptv3-shared-wam.v5'
            or restored_manifest['resume_binding']['checkpoint']!=report['baseline_checkpoint']
            or restored_progress['status']!='RUNNING' or restored_progress['step']<=report['baseline_step']):
        raise ValueError('original v5 baseline did not resume from its own checkpoint')
    devs=[]
    for phase in ('initial-recovery','sustained','resume-recovery'):
        for f in sorted((a.run/phase).glob('dev-step*.json')):
            d=read_json(f);s=d['sources'];u=d['unknown_event_control']['sources']
            def macro(key):return statistics.mean(v['metrics'][key]['mean'] for v in s.values() if v['metrics'][key]['mean'] is not None)
            devs.append(dict(phase=phase,step=int(f.stem.split('step')[-1]),
                objective=d['source_macro_objective'],hand_mm=statistics.mean(v['hand_epe_mm'] for v in s.values()),
                root_relative_mm=macro('hand/root_relative'),short_mm=statistics.mean(macro('hand/h%02d'%h) for h in range(1,5)),
                unknown_hand_mm=statistics.mean(v['hand_epe_mm'] for v in u.values()),
                conditioned_minus_unknown_hand_mm=d['conditioned_minus_unknown_hand_macro_mm'],
                trajectory_score=d['trajectory_selection']['score']))
    result=dict(status='COMPLETED',training_git_commit=report['git_commit'],audit_git_commit=a.git_commit,
        baseline_step=report['baseline_step'],v6_final_step=saved['step'],new_updates=expected_new,
        optimizer_states=state_count,optimizer_steps={k:sorted(v) for k,v in counters.items()},
        four_rng_payloads=True,strict_v6_resume_binding=binding,
        migration_provenance_retained=True,original_horizon_normalization_exact=True,dev=devs,phases=report['phases'],
        restored_v5=dict(path=str(restore),git_commit=restored_manifest['git_commit'],progress=restored_progress),
        python=sys.version,torch=torch.__version__,device='CPU checkpoint/record inspection only',
        limitations=['bounded1000-update stability, not80k guarantee','no matched unconditioned training arm; event-drop DEV is diagnostic, not causal efficacy','GPU step timing and wall throughput include different overheads; not a matched performance benchmark'])
    path=a.output/'summary.json';path.write_text(json.dumps(result,indent=2)+'\n');path.chmod(0o664)
    print(json.dumps({k:v for k,v in result.items() if k not in ('phases',)}),flush=True)


if __name__=='__main__':main()
